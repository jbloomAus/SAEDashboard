from __future__ import annotations

import numpy as np
import torch
from eindex import (
    eindex,  # type: ignore[import-untyped]  # pyright: ignore[reportMissingTypeStubs]
)
from jaxtyping import Float, Int
from torch import Tensor

from sae_dashboard.components import (
    SequenceData,
    SequenceGroupData,
    SequenceMultiGroupData,
)
from sae_dashboard.sequence_data_generator import (
    SequenceCoordinateTable,
    SequenceDataGenerator,
    SequenceSelectionBackend,
)
from sae_dashboard.utils_fns import TopK, k_largest_indices, random_range_indices


class LegacySequenceDataGenerator(SequenceDataGenerator):
    def _selection_feat_acts(
        self,
        feat_acts: Float[Tensor, "batch seq"],
    ) -> Float[Tensor, "batch seq"]:
        target_device = torch.device(self.cfg.device)
        if target_device.type == "cuda" and not torch.cuda.is_available():
            return feat_acts
        return feat_acts.to(device=target_device)

    def _get_indices_dict_legacy_unmasked(
        self,
        buffer: tuple[int, int] | None,
        feat_acts: Float[Tensor, "batch seq"],
    ):
        indices = k_largest_indices(
            feat_acts,
            k=self.seq_cfg.top_acts_group_size,
            buffer=buffer,
        ).cpu()
        indices_dict = {f"TOP ACTIVATIONS<br>MAX = {feat_acts.max():.3f}": indices}

        if self.seq_cfg.n_quantiles > 0:
            quantiles = torch.linspace(
                0,
                feat_acts.max().item(),
                self.seq_cfg.n_quantiles + 1,
                device=feat_acts.device,
            )
            for i in range(self.seq_cfg.n_quantiles - 1, -1, -1):
                lower, upper = quantiles[i : i + 2].tolist()
                pct = float(
                    ((feat_acts >= lower) & (feat_acts <= upper)).float().mean().item()
                )
                indices = random_range_indices(
                    feat_acts,
                    k=self.seq_cfg.quantile_group_size,
                    bounds=(lower, upper),
                    buffer=buffer,
                ).cpu()
                indices_dict[
                    f"INTERVAL {lower:.3f} - {upper:.3f}<br>CONTAINS {pct:.3%}"
                ] = indices

        indices_bold = torch.concat(list(indices_dict.values())).cpu()
        n_bold = indices_bold.shape[0]
        return indices_dict, indices_bold, n_bold

    def get_indices_dict_legacy(
        self,
        buffer: tuple[int, int] | None,
        feat_acts: Float[Tensor, "batch seq"],
        selection_mask: Int[Tensor, "batch seq"] | None = None,
    ):
        if selection_mask is None:
            return self._get_indices_dict_legacy_unmasked(buffer, feat_acts)

        return super().get_indices_dict_legacy(
            buffer,
            feat_acts,
            selection_mask=selection_mask,
        )

    def _package_sequences_data_legacy(
        self,
        token_ids: Int[Tensor, "n_bold buf"],
        feat_acts_coloring: Float[Tensor, "n_bold buf"],
        feat_logits: Float[Tensor, "d_vocab"],
        indices_dict: dict[str, Int[Tensor, "n_bold 2"]],
        indices_bold: Int[Tensor, "n_bold"],
        loss_contribution: Float[Tensor, "n_bold 1"] | None = None,
        top_contribution_to_logits: TopK | None = None,
        bottom_contribution_to_logits: TopK | None = None,
    ) -> SequenceMultiGroupData:
        del loss_contribution, top_contribution_to_logits, bottom_contribution_to_logits

        sequence_groups_data = []
        group_sizes_cumsum = np.cumsum(
            [0] + [len(indices) for indices in indices_dict.values()]
        ).tolist()

        feat_logits = feat_logits.cpu()
        feat_acts_coloring = feat_acts_coloring.cpu()
        token_ids = token_ids.cpu()
        indices_bold = indices_bold.cpu()

        if self.cfg.perform_ablation_experiments:
            raise NotImplementedError(
                "We are not supporting ablation experiments for now."
            )

        for group_idx, group_name in enumerate(indices_dict.keys()):
            seq_data = [
                SequenceData(
                    original_index=int(indices_bold[i, 0].item()),
                    token_ids=token_ids[i].tolist(),
                    feat_acts=[round(f, 4) for f in feat_acts_coloring[i].tolist()],
                    token_logits=feat_logits[token_ids[i]].tolist(),
                    qualifying_token_index=int(indices_bold[i, 1].item()),
                )
                for i in range(
                    group_sizes_cumsum[group_idx], group_sizes_cumsum[group_idx + 1]
                )
            ]
            sequence_groups_data.append(SequenceGroupData(group_name, seq_data))

        return SequenceMultiGroupData(sequence_groups_data)

    def package_sequences_data(
        self,
        token_ids: Int[Tensor, "n_bold buf"],
        feat_acts_coloring: Float[Tensor, "n_bold buf"],
        feat_logits: Float[Tensor, "d_vocab"],
        indices_dict: dict[str, Int[Tensor, "n_bold 2"]],
        indices_bold: Int[Tensor, "n_bold"],
        loss_contribution: Float[Tensor, "n_bold 1"] | None = None,
        top_contribution_to_logits: TopK | None = None,
        bottom_contribution_to_logits: TopK | None = None,
    ):
        return self._package_sequences_data_legacy(
            token_ids=token_ids,
            feat_acts_coloring=feat_acts_coloring,
            feat_logits=feat_logits,
            indices_dict=indices_dict,
            indices_bold=indices_bold,
            loss_contribution=loss_contribution,
            top_contribution_to_logits=top_contribution_to_logits,
            bottom_contribution_to_logits=bottom_contribution_to_logits,
        )

    @torch.inference_mode()
    def get_sequences_data(
        self,
        feat_acts: Float[Tensor, "batch seq"],
        feat_logits: Float[Tensor, "d_vocab"],
        resid_post: Float[Tensor, "batch seq d_model"],
        feature_resid_dir: Float[Tensor, "d_model"],
        selection_mask: Int[Tensor, "batch seq"] | None = None,
        selection_backend: SequenceSelectionBackend = "legacy",
    ) -> SequenceMultiGroupData:
        del feature_resid_dir, selection_mask, selection_backend
        indices_dict, indices_bold, n_bold = self.get_indices_dict(
            self.buffer,
            self._selection_feat_acts(feat_acts),
        )
        indices_buf = self.get_indices_buf(
            indices_bold=indices_bold,
            seq_length=self.seq_length,
            n_bold=n_bold,
            padded_buffer_width=self.padded_buffer_width,
        )
        token_ids = eindex(
            self.tokens,
            indices_buf[:, 1:],
            "[n_bold seq 0] [n_bold seq 1]",
        )
        (
            _,
            feat_acts_coloring,
            _,
        ) = self.index_objects_for_ablation_experiments(
            token_ids=token_ids,
            tokens=self.tokens,
            feat_acts=feat_acts,
            resid_post=resid_post,
            indices_bold=indices_bold,
            indices_buf=indices_buf,
        )
        return self.package_sequences_data(
            token_ids=token_ids,
            feat_acts_coloring=feat_acts_coloring,
            feat_logits=feat_logits,
            indices_dict=indices_dict,
            indices_bold=indices_bold,
        )

    @torch.inference_mode()
    def get_sequence_coordinate_table(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        feat_acts: Float[Tensor, "batch seq"],
        feat_logits: Float[Tensor, "d_vocab"],
        resid_post: Float[Tensor, "batch seq d_model"],
        feature_resid_dir: Float[Tensor, "d_model"],
        selection_mask: Int[Tensor, "batch seq"] | None = None,
        selection_backend: SequenceSelectionBackend = "legacy",
    ) -> SequenceCoordinateTable:
        del selection_backend
        return SequenceDataGenerator.get_sequence_coordinate_table(
            self,
            feat_acts=feat_acts,
            feat_logits=feat_logits,
            resid_post=resid_post,
            feature_resid_dir=feature_resid_dir,
            selection_mask=selection_mask,
            selection_backend="legacy",
        )
