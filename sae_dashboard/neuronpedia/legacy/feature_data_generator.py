from __future__ import annotations

import torch
from torch import Tensor

from sae_dashboard.feature_data_generator import FeatureDataGenerator
from sae_dashboard.neuronpedia.legacy import utils_fns as legacy_utils_fns


class LegacyFeatureDataGenerator(FeatureDataGenerator):
    def _transfer_feature_acts_for_output(
        self,
        feature_acts_for_output: Tensor,
    ) -> Tensor:
        return feature_acts_for_output

    def _uses_full_feature_encode_path(self) -> bool:
        return self.encoder.cfg.architecture() in ["topk", "batchtopk", "temporal"]

    def _create_corrcoef_neurons(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        *,
        correlation_device: torch.device,
    ):
        return legacy_utils_fns.RollingCorrCoef(device=torch.device("cpu"))

    def _create_corrcoef_encoder(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        *,
        feature_indices: list[int],
        correlation_device: torch.device,
    ):
        return legacy_utils_fns.RollingCorrCoef(
            indices=feature_indices,
            with_self=True,
            device=torch.device("cpu"),
        )
