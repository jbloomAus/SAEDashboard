import math
from collections import defaultdict
from typing import Iterable

import numpy as np
import torch
from jaxtyping import Int
from rich import print as rprint
from rich.table import Table
from torch import Tensor
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer

from sae_vis.autoencoder import AutoEncoder
from sae_vis.data_fetching_fns import get_feature_data
from sae_vis.data_parsing_fns import parse_feature_data
from sae_vis.sae_vis_data import SaeVisConfig, SaeVisData
from sae_vis.transformer_lens_wrapper import TransformerLensWrapper


class SaeVisRunner:
    def __init__(self, cfg: SaeVisConfig) -> None:
        self.cfg = cfg

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

        # Create objects to store all the data we'll get from `_get_feature_data`
        sae_vis_data = SaeVisData(cfg=self.cfg)
        time_logs = defaultdict(float)

        tokens = self.subset_tokens(tokens)
        features_list = self.handle_features(self.cfg.features, encoder)
        feature_batches = self.get_feature_batches(features_list)
        progress = self.get_progress_bar(tokens, feature_batches, features_list)
        model_wrapper = TransformerLensWrapper(model, self.cfg.hook_point)

        # For each batch of features: get new data and update global data storage objects
        # TODO: We should write out json files with the results as this runs rather than storing everything in memory
        # in memory.
        for features in feature_batches:
            (
                all_feat_acts,
                all_resid_post,
                feature_resid_dir,
                feature_out_dir,
                corrcoef_neurons,
                corrcoef_encoder,
                corrcoef_encoder_B,
            ) = get_feature_data(
                encoder=encoder,
                encoder_B=encoder_B,
                model=model_wrapper,
                tokens=tokens,
                feature_indices=features,
                cfg=self.cfg,
                progress=progress,
            )

            # ! Use the data we've collected to make a MultiFeatureData object
            new_feature_data, new_time_logs = parse_feature_data(
                tokens=tokens,
                feature_indices=features,
                all_feat_acts=all_feat_acts,
                feature_resid_dir=feature_resid_dir,
                all_resid_post=all_resid_post,
                W_U=model.W_U,
                cfg=self.cfg,
                feature_out_dir=feature_out_dir,
                corrcoef_neurons=corrcoef_neurons,
                corrcoef_encoder=corrcoef_encoder,
                corrcoef_encoder_B=corrcoef_encoder_B,
                progress=progress,
            )

            assert (
                set(time_logs.keys()) & set(new_time_logs.keys()) == set()
            ), f"Invalid keys: {set(time_logs.keys()) & set(new_time_logs.keys())} should have zero overlap"

            time_logs.update(new_time_logs)

            sae_vis_data.update(new_feature_data)
            for key, value in new_time_logs.items():
                time_logs[key] += value

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
            torch.manual_seed(self.cfg.seed)
            np.random.seed(self.cfg.seed)
        return None

    def subset_tokens(
        self, tokens: Int[Tensor, "batch seq"]
    ) -> Int[Tensor, "batch seq"]:
        """
        We should remove this soon. Not worth it.
        """
        if self.cfg.batch_size is None:
            return tokens
        else:
            return tokens[: self.cfg.batch_size]

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
