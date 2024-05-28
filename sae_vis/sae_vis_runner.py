import math
import random
from collections import defaultdict
from typing import Iterable, cast

import einops
import numpy as np
import torch
from jaxtyping import Int
from rich import print as rprint
from rich.table import Table
from torch import Tensor
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer

from sae_vis.autoencoder import DTYPES, AutoEncoder
from sae_vis.components import (
    ActsHistogramData,
    FeatureTablesData,
    LogitsHistogramData,
)
from sae_vis.data_fetching_fns import FeatureDataGenerator
from sae_vis.data_parsing_fns import (
    get_features_table_data,
    get_logits_table_data,
)
from sae_vis.feature_data import FeatureData
from sae_vis.sae_vis_data import SaeVisConfig, SaeVisData
from sae_vis.sequence_data_generator import (
    SequenceDataGenerator,
)
from sae_vis.transformer_lens_wrapper import TransformerLensWrapper
from sae_vis.utils_fns import (
    FeatureStatistics,
)


class SaeVisRunner:
    def __init__(self, cfg: SaeVisConfig) -> None:
        self.cfg = cfg

        if self.cfg.cache_dir is not None:
            self.cfg.cache_dir.mkdir(parents=True, exist_ok=True)

    @torch.inference_mode()
    def run(
        self,
        encoder: AutoEncoder,
        model: HookedTransformer,
        tokens: Int[Tensor, "batch seq"],
        encoder_B: AutoEncoder | None = None,
    ) -> SaeVisData:
        # Apply random seed
        self.set_seeds()

        # set precision on encoders and model
        encoder = encoder.to(DTYPES[self.cfg.dtype])
        model = cast(HookedTransformer, model.to(DTYPES[self.cfg.dtype]))
        if encoder_B is not None:
            encoder_B.to(DTYPES[self.cfg.dtype])

        # Create objects to store all the data we'll get from `_get_feature_data`
        sae_vis_data = SaeVisData(cfg=self.cfg)
        model.to(self.cfg.device)
        encoder = encoder.to(self.cfg.device)
        time_logs = defaultdict(float)

        features_list = self.handle_features(self.cfg.features, encoder)
        feature_batches = self.get_feature_batches(features_list)
        progress = self.get_progress_bar(tokens, feature_batches, features_list)
        model_wrapper = TransformerLensWrapper(model, self.cfg.hook_point)

        feature_data_generator = FeatureDataGenerator(
            cfg=self.cfg,
            model=model_wrapper,
            encoder=encoder,
            encoder_B=encoder_B,
            tokens=tokens,
        )

        sequence_data_generator = SequenceDataGenerator(
            cfg=self.cfg,
            tokens=tokens,
            W_U=model.W_U,
        )

        # For each batch of features: get new data and update global data storage objects
        # TODO: We should write out json files with the results as this runs rather than storing everything in memory
        for features in feature_batches:
            # model and sae activations calculations.

            (
                all_feat_acts,
                all_resid_post,
                feature_resid_dir,
                feature_out_dir,
                corrcoef_neurons,
                corrcoef_encoder,
                corrcoef_encoder_B,
            ) = feature_data_generator.get_feature_data(features, progress)

            # Get the logits of all features (i.e. the directions this feature writes to the logit output)
            logits = einops.einsum(
                feature_resid_dir,
                model.W_U,
                "feats d_model, d_model d_vocab -> feats d_vocab",
            )

            # ! Get stats (including quantiles, which will be useful for the prompt-centric visualisation)
            feature_stats = FeatureStatistics.create(
                data=einops.rearrange(all_feat_acts, "b s feats -> feats (b s)")
            )

            # ! Data setup code (defining the main objects we'll eventually return)
            feature_data_dict: dict[int, FeatureData] = {
                feat: FeatureData() for feat in features
            }

            # We're using `cfg.feature_centric_layout` to figure out what data we'll need to calculate during this function
            layout = self.cfg.feature_centric_layout

            feature_tables_data = get_features_table_data(
                feature_out_dir=feature_out_dir,
                corrcoef_neurons=corrcoef_neurons,
                corrcoef_encoder=corrcoef_encoder,
                corrcoef_encoder_B=corrcoef_encoder_B,
                n_rows=layout.feature_tables_cfg.n_rows,  # type: ignore
            )

            # Add all this data to the list of FeatureTablesData objects

            for i, (feat, logit_vector) in enumerate(zip(features, logits)):
                feature_data_dict[feat].feature_tables_data = FeatureTablesData(
                    **{k: v[i] for k, v in feature_tables_data.items()}  # type: ignore
                )

                # Get logits histogram data (no title)
                feature_data_dict[
                    feat
                ].logits_histogram_data = LogitsHistogramData.from_data(
                    data=logit_vector.to(
                        torch.float32
                    ),  # need this otherwise fails on MPS
                    n_bins=layout.logits_hist_cfg.n_bins,  # type: ignore
                    tickmode="5 ticks",
                    title=None,
                )

                # Get data for feature activations histogram (including the title!)
                feat_acts = all_feat_acts[..., i]
                nonzero_feat_acts = feat_acts[feat_acts > 0]
                frac_nonzero = nonzero_feat_acts.numel() / feat_acts.numel()
                feature_data_dict[
                    feat
                ].acts_histogram_data = ActsHistogramData.from_data(
                    data=nonzero_feat_acts.to(
                        torch.float32
                    ),  # need this otherwise fails on MPS
                    n_bins=layout.act_hist_cfg.n_bins,  # type: ignore
                    tickmode="5 ticks",
                    title=f"ACTIVATIONS<br>DENSITY = {frac_nonzero:.3%}",
                )

                # Create a MiddlePlotsData object from this, and add it to the dict
                feature_data_dict[feat].logits_table_data = get_logits_table_data(
                    logit_vector=logit_vector,
                    n_rows=layout.logits_table_cfg.n_rows,  # type: ignore
                )

                # ! Calculate all data for the right-hand visualisations, i.e. the sequences

                # Add this feature's sequence data to the list
                feature_data_dict[
                    feat
                ].sequence_data = sequence_data_generator.get_sequences_data(
                    feat_acts=all_feat_acts[..., i],
                    feat_logits=logits[i],
                    resid_post=all_resid_post,
                    feature_resid_dir=feature_resid_dir[i],
                )
                # Update the 2nd progress bar (fwd passes & getting sequence data dominates the runtime of these computations)
                if progress is not None:
                    progress[1].update(1)

            # ! Return the output, as a dict of FeatureData items
            new_feature_data = SaeVisData(
                cfg=self.cfg,
                feature_data_dict=feature_data_dict,
                feature_stats=feature_stats,
            )

            sae_vis_data.update(new_feature_data)

        # Now exited, make sure the progress bar is at 100%
        if progress is not None:
            for pbar in progress:
                pbar.n = pbar.total

        # If verbose, then print the output
        if self.cfg.verbose:
            total_time = sum(time_logs.values())
            table = Table("Task", "Time", "Pct %")
            for task, duration in time_logs.items():
                table.add_row(task, f"{duration:.2f}s", f"{duration/total_time:.1%}")
            rprint(table)

        sae_vis_data.cfg = self.cfg
        sae_vis_data.model = model
        sae_vis_data.encoder = encoder
        sae_vis_data.encoder_B = encoder_B

        return sae_vis_data

    def set_seeds(self) -> None:
        if self.cfg.seed is not None:
            random.seed(self.cfg.seed)
            torch.manual_seed(self.cfg.seed)
            np.random.seed(self.cfg.seed)
        return None

    def handle_features(
        self, features: Iterable[int] | None, encoder_wrapper: AutoEncoder
    ) -> list[int]:
        if features is None:
            return list(range(encoder_wrapper.cfg.d_hidden))
        else:
            return list(features)

    def get_feature_batches(self, features_list: list[int]) -> list[list[int]]:
        # Break up the features into batches
        feature_batches = [
            x.tolist()
            for x in torch.tensor(features_list).split(self.cfg.minibatch_size_features)
        ]
        return feature_batches

    def get_progress_bar(
        self,
        tokens: Int[Tensor, "batch seq"],
        feature_batches: list[list[int]],
        features_list: list[int],
    ):
        # Calculate how many minibatches of tokens there will be (for the progress bar)
        n_token_batches = (
            1
            if (self.cfg.minibatch_size_tokens is None)
            else math.ceil(len(tokens) / self.cfg.minibatch_size_tokens)
        )

        # Get the denominator for each of the 2 progress bars
        totals = (n_token_batches * len(feature_batches), len(features_list))

        # Optionally add two progress bars (one for the forward passes, one for getting the sequence data)
        if self.cfg.verbose:
            progress = [
                tqdm(total=totals[0], desc="Forward passes to cache data for vis"),
                tqdm(total=totals[1], desc="Extracting vis data from cached data"),
            ]
        else:
            progress = None

        return progress
