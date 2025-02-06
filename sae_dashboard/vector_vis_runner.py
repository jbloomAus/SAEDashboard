import math
import random
import re
from collections import defaultdict
from typing import Iterable, List, Union

import einops
import numpy as np
import torch
from jaxtyping import Int
from rich import print as rprint
from rich.table import Table
from sae_lens import SAE
from sae_lens.config import DTYPE_MAP as DTYPES
from torch import Tensor
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer

from sae_dashboard.components import (
    ActsHistogramData,
    DecoderWeightsDistribution,
    FeatureTablesData,
    LogitsHistogramData,
)
from sae_dashboard.data_parsing_fns import (
    get_features_table_data,
    get_logits_table_data,
)
from sae_dashboard.feature_data import FeatureData
from sae_dashboard.neuronpedia.vector_set import VectorSet
from sae_dashboard.sequence_data_generator import SequenceDataGenerator
from sae_dashboard.transformer_lens_wrapper import (
    ActivationConfig,
    TransformerLensWrapper,
)
from sae_dashboard.utils_fns import FeatureStatistics
from sae_dashboard.vector_data_generator import VectorDataGenerator
from sae_dashboard.vector_vis_data import VectorVisConfig, VectorVisData


class VectorDataGeneratorFactory:
    @staticmethod
    def create(
        cfg: VectorVisConfig,
        model: HookedTransformer,
        encoder: VectorSet,
        tokens: Int[Tensor, "batch seq"],
    ) -> VectorDataGenerator:
        """Builds a FeatureDataGenerator using the provided config and model."""
        activation_config = ActivationConfig(
            primary_hook_point=cfg.hook_point,
            auxiliary_hook_points=(
                [
                    re.sub(r"hook_z", "hook_v", cfg.hook_point),
                    re.sub(r"hook_z", "hook_pattern", cfg.hook_point),
                ]
                if cfg.use_dfa
                else []
            ),
        )
        wrapped_model = TransformerLensWrapper(model, activation_config)
        return VectorDataGenerator(
            cfg=cfg, model=wrapped_model, encoder=encoder, tokens=tokens
        )


class VectorVisRunner:
    def __init__(self, cfg: VectorVisConfig) -> None:
        self.cfg = cfg
        self.device = self.cfg.device
        self.dtype = DTYPES[self.cfg.dtype]
        if self.cfg.cache_dir is not None:
            self.cfg.cache_dir.mkdir(parents=True, exist_ok=True)

    @torch.inference_mode()
    def run(
        self,
        encoder: VectorSet,
        model: HookedTransformer,
        tokens: Int[Tensor, "batch seq"],
    ) -> VectorVisData:
        # Apply random seed
        self.set_seeds()

        # Create objects to store all the data we'll get from `_get_feature_data`
        vector_vis_data = VectorVisData(cfg=self.cfg)
        time_logs = defaultdict(float)

        vector_indices_list = self.handle_vector_indices(
            self.cfg.vector_indices, encoder
        )
        vector_batches = self.get_vector_batches(vector_indices_list)
        progress = self.get_progress_bar(tokens, vector_batches, vector_indices_list)

        vector_data_generator = VectorDataGeneratorFactory.create(
            self.cfg, model, encoder, tokens
        )

        sequence_data_generator = SequenceDataGenerator(
            cfg=self.cfg,
            tokens=tokens,
            W_U=model.W_U,
        )

        # all_consolidated_dfa_results = { # TODO: implement DFA for vectors
        #     vector_idx: {} for vector_idx in self.cfg.vector_indices
        # }
        # For each batch of features: get new data and update global data storage objects
        # TODO: We should write out json files with the results as this runs rather than storing everything in memory
        for vector_indices in vector_batches:
            # model and sae activations calculations.

            (
                all_feat_acts,
                _,  # all resid post. no longer used.
                feature_resid_dir,
                feature_out_dir,
                corrcoef_neurons,
                corrcoef_encoder,
                batch_dfa_results,  # type: ignore
            ) = vector_data_generator.get_feature_data(vector_indices, progress)

            # Get the logits of all features (i.e. the directions this feature writes to the logit output)
            logits = einops.einsum(
                feature_resid_dir.to(device=model.W_U.device, dtype=model.W_U.dtype),
                model.W_U,
                "feats d_model, d_model d_vocab -> feats d_vocab",
            ).to(self.device)

            # ! Get stats (including quantiles, which will be useful for the prompt-centric visualisation)
            vector_stats = FeatureStatistics.create(
                data=einops.rearrange(all_feat_acts, "b s feats -> feats (b s)"),
                batch_size=self.cfg.quantile_feature_batch_size,
            )

            # ! Data setup code (defining the main objects we'll eventually return)
            vector_data_dict: dict[int, FeatureData] = {
                vector_idx: FeatureData() for vector_idx in vector_indices
            }

            # We're using `cfg.feature_centric_layout` to figure out what data we'll need to calculate during this function
            layout = self.cfg.feature_centric_layout

            feature_tables_data = get_features_table_data(
                feature_out_dir=feature_out_dir,
                corrcoef_neurons=corrcoef_neurons,
                corrcoef_encoder=corrcoef_encoder,
                n_rows=layout.feature_tables_cfg.n_rows,  # type: ignore
            )

            # Add all this data to the list of FeatureTablesData objects
            # if batch_dfa_results: # TODO: implement DFA for vectors
            #     # Accumulate DFA results across feature batches
            #     for feature_idx, feature_data in batch_dfa_results.items():
            #         all_consolidated_dfa_results[feature_idx].update(feature_data)

            for i, (vector_idx, logit_vector) in enumerate(zip(vector_indices, logits)):
                vector_data_dict[vector_idx].feature_tables_data = FeatureTablesData(
                    **{k: v[i] for k, v in feature_tables_data.items()}  # type: ignore
                )

                # Get logits histogram data (no title)
                vector_data_dict[vector_idx].logits_histogram_data = (
                    LogitsHistogramData.from_data(
                        data=logit_vector.to(
                            torch.float32
                        ),  # need this otherwise fails on MPS
                        n_bins=layout.logits_hist_cfg.n_bins,  # type: ignore
                        tickmode="5 ticks",
                        title=None,
                    )
                )

                # Get data for feature activations histogram (including the title!)
                feat_acts = all_feat_acts[..., i]

                # Create a mask for tokens to ignore based on both ID and position
                ignore_tokens_mask = torch.ones_like(tokens, dtype=torch.bool)
                if self.cfg.ignore_tokens:
                    ignore_tokens_mask &= ~torch.isin(
                        tokens,
                        torch.tensor(
                            list(self.cfg.ignore_tokens),
                            dtype=tokens.dtype,
                            device=tokens.device,
                        ),
                    )
                if self.cfg.ignore_positions:
                    ignore_positions_mask = torch.ones_like(tokens, dtype=torch.bool)
                    ignore_positions_mask[:, self.cfg.ignore_positions] = False
                    ignore_tokens_mask &= ignore_positions_mask

                # Move the mask to the same device as feat_acts
                ignore_tokens_mask = ignore_tokens_mask.to(feat_acts.device)

                # set any masked positions to 0
                masked_feat_acts = feat_acts * ignore_tokens_mask

                # Apply the mask to feat_acts
                nonzero_feat_acts = masked_feat_acts[masked_feat_acts > 0]
                frac_nonzero = nonzero_feat_acts.numel() / masked_feat_acts.numel()

                vector_data_dict[vector_idx].acts_histogram_data = (
                    ActsHistogramData.from_data(
                        data=nonzero_feat_acts.to(
                            torch.float32
                        ),  # need this otherwise fails on MPS
                        n_bins=layout.act_hist_cfg.n_bins,  # type: ignore
                        tickmode="5 ticks",
                        title=f"ACTIVATIONS<br>DENSITY = {frac_nonzero:.3%}",
                    )
                )

                # Create a MiddlePlotsData object from this, and add it to the dict
                vector_data_dict[vector_idx].logits_table_data = get_logits_table_data(
                    logit_vector=logit_vector,
                    n_rows=layout.logits_table_cfg.n_rows,  # type: ignore
                )

                # ! Calculate all data for the right-hand visualisations, i.e. the sequences

                # Add this feature's sequence data to the list
                vector_data_dict[vector_idx].sequence_data = (
                    sequence_data_generator.get_sequences_data(
                        feat_acts=masked_feat_acts,
                        feat_logits=logits[i],
                        resid_post=torch.tensor([]),  # no longer used
                        feature_resid_dir=feature_resid_dir[i],
                    )
                )
                # if self.cfg.use_dfa:
                #     vector_data_dict[vector_idx].dfa_data = all_consolidated_dfa_results.get(
                #         vector_idx, None
                #     )
                #     vector_data_dict[vector_idx].decoder_weights_data = (
                #         get_decoder_weights_distribution(encoder, model, vector_idx)[0]
                #     )

                # Update the 2nd progress bar (fwd passes & getting sequence data dominates the runtime of these computations)
                if progress is not None:
                    progress[1].update(1)

            # ! Return the output, as a dict of FeatureData items
            new_vector_data = VectorVisData(
                cfg=self.cfg,
                vector_data_dict=vector_data_dict,
                vector_stats=vector_stats,
            )

            vector_vis_data.update(new_vector_data)

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

        vector_vis_data.cfg = self.cfg
        vector_vis_data.model = model
        vector_vis_data.encoder = encoder

        return vector_vis_data

    def set_seeds(self) -> None:
        if self.cfg.seed is not None:
            random.seed(self.cfg.seed)
            torch.manual_seed(self.cfg.seed)
            np.random.seed(self.cfg.seed)
        return None

    def handle_vector_indices(
        self, vector_indices: Iterable[int] | None, encoder_wrapper: VectorSet
    ) -> list[int]:
        if vector_indices is None:
            return list(range(encoder_wrapper.cfg.d_vectors))
        else:
            return list(vector_indices)

    def get_vector_batches(self, vector_indices_list: list[int]) -> list[list[int]]:
        # Break up the features into batches
        vector_batches = [
            x.tolist()
            for x in torch.tensor(vector_indices_list).split(
                self.cfg.minibatch_size_features
            )
        ]
        return vector_batches

    def get_progress_bar(
        self,
        tokens: Int[Tensor, "batch seq"],
        vector_batches: list[list[int]],
        vector_indices_list: list[int],
    ):
        # Calculate how many minibatches of tokens there will be (for the progress bar)
        n_token_batches = (
            1
            if (self.cfg.minibatch_size_tokens is None)
            else math.ceil(len(tokens) / self.cfg.minibatch_size_tokens)
        )

        # Get the denominator for each of the 2 progress bars
        totals = (n_token_batches * len(vector_batches), len(vector_indices_list))

        # Optionally add two progress bars (one for the forward passes, one for getting the sequence data)
        if self.cfg.verbose:
            progress = [
                tqdm(total=totals[0], desc="Forward passes to cache data for vis"),
                tqdm(total=totals[1], desc="Extracting vis data from cached data"),
            ]
        else:
            progress = None

        return progress


def get_decoder_weights_distribution(
    encoder: SAE,
    model: HookedTransformer,
    feature_idx: Union[int, List[int]],
) -> List[DecoderWeightsDistribution]:
    if not isinstance(feature_idx, list):
        feature_idx = [feature_idx]

    distribs = []
    for feature in feature_idx:
        att_blocks = einops.rearrange(
            encoder.W_dec[feature, :],
            "(n_head d_head) -> n_head d_head",
            n_head=model.cfg.n_heads,
        ).to("cpu")
        decoder_weights_distribution = (
            att_blocks.norm(dim=1) / att_blocks.norm(dim=1).sum()
        )
        distribs.append(
            DecoderWeightsDistribution(
                model.cfg.n_heads, [float(x) for x in decoder_weights_distribution]
            )
        )

    return distribs
