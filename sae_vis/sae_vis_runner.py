import math
from collections import defaultdict

import numpy as np
import torch
from jaxtyping import Int
from rich import print as rprint
from rich.table import Table
from torch import Tensor, nn
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer

from sae_vis.data_config_classes import SaeVisConfig
from sae_vis.data_fetching_fns import get_feature_data
from sae_vis.data_storing_fns import SaeVisData
from sae_vis.model_fns import (
    AutoEncoder,
    AutoEncoderConfig,
    TransformerLensWrapper,
)


class SaeVisRunner:
    def __init__(self, cfg: SaeVisConfig) -> None:
        self.cfg = cfg

    @torch.inference_mode()
    def run(
        self,
        encoder: nn.Module,
        model: HookedTransformer,
        tokens: Int[Tensor, "batch seq"],
        encoder_B: AutoEncoder | None = None,
    ) -> SaeVisData:
        # If encoder isn't an AutoEncoder, we wrap it in one
        if not isinstance(encoder, AutoEncoder):
            assert set(
                encoder.state_dict().keys()
            ).issuperset(
                {"W_enc", "W_dec", "b_enc", "b_dec"}
            ), "If encoder isn't an AutoEncoder, it should have weights 'W_enc', 'W_dec', 'b_enc', 'b_dec'"
            d_in, d_hidden = encoder.W_enc.shape
            device = encoder.W_enc.device
            encoder_cfg = AutoEncoderConfig(d_in=d_in, d_hidden=d_hidden)
            encoder_wrapper = AutoEncoder(encoder_cfg).to(device)
            encoder_wrapper.load_state_dict(encoder.state_dict(), strict=False)
        else:
            encoder_wrapper = encoder

        # Apply random seed
        if self.cfg.seed is not None:
            torch.manual_seed(self.cfg.seed)
            np.random.seed(self.cfg.seed)

        # Create objects to store all the data we'll get from `_get_feature_data`
        sae_vis_data = SaeVisData()
        time_logs = defaultdict(float)

        # Slice tokens, if we're only doing a subset of them
        if self.cfg.batch_size is None:
            tokens = tokens
        else:
            tokens = tokens[: self.cfg.batch_size]

        # Get a feature list (need to deal with the case where `self.cfg.features` is an int, or None)
        if self.cfg.features is None:
            assert isinstance(encoder_wrapper.cfg.d_hidden, int)
            features_list = list(range(encoder_wrapper.cfg.d_hidden))
        elif isinstance(self.cfg.features, int):
            features_list = [self.cfg.features]
        else:
            features_list = list(self.cfg.features)

        # Break up the features into batches
        feature_batches = [
            x.tolist()
            for x in torch.tensor(features_list).split(self.cfg.minibatch_size_features)
        ]
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

        # If the model is from TransformerLens, we need to apply a wrapper to it for standardization
        assert isinstance(
            model, HookedTransformer
        ), "Error: non-HookedTransformer models are not yet supported."
        assert isinstance(
            self.cfg.hook_point, str
        ), "Error: self.cfg.hook_point must be a string"
        model_wrapper = TransformerLensWrapper(model, self.cfg.hook_point)

        # For each batch of features: get new data and update global data storage objects
        for features in feature_batches:
            new_feature_data, new_time_logs = get_feature_data(
                encoder=encoder_wrapper,
                encoder_B=encoder_B,
                model=model_wrapper,
                tokens=tokens,
                feature_indices=features,
                cfg=self.cfg,
                progress=progress,
            )
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
        sae_vis_data.encoder = encoder_wrapper
        sae_vis_data.encoder_B = encoder_B

        return sae_vis_data
