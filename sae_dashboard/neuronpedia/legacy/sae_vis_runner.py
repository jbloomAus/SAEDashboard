from __future__ import annotations

from typing import TYPE_CHECKING, Any

import einops
import torch
from jaxtyping import Int
from sae_lens import SAE
from torch import Tensor

from sae_dashboard.components import ActsHistogramData, FeatureTablesData
from sae_dashboard.data_parsing_fns import (
    get_features_table_data,
    get_logits_table_data,
)
from sae_dashboard.feature_data import FeatureData
from sae_dashboard.feature_data_generator import FeatureDataGenerator
from sae_dashboard.neuronpedia.legacy import utils_fns as legacy_utils_fns
from sae_dashboard.neuronpedia.legacy.sequence_data_generator import (
    LegacySequenceDataGenerator,
)
from sae_dashboard.perf_logging import log_perf_event, timed_stage
from sae_dashboard.sae_vis_data import SaeVisData
from sae_dashboard.utils_fns import FeatureStatistics

if TYPE_CHECKING:
    from sae_lens import HookedSAETransformer

if TYPE_CHECKING:
    from sae_dashboard.sae_vis_runner import SaeVisRunner


def _build_ignore_tokens_mask(
    cfg: Any,
    tokens: Int[Tensor, "batch seq"],
    target_device: torch.device | str,
) -> Tensor:
    ignore_tokens_mask = torch.ones_like(tokens, dtype=torch.bool)
    if cfg.ignore_tokens:
        ignore_tokens_mask &= ~torch.isin(
            tokens,
            torch.tensor(
                list(cfg.ignore_tokens),
                dtype=tokens.dtype,
                device=tokens.device,
            ),
        )
    if cfg.ignore_positions:
        ignore_positions_mask = torch.ones_like(tokens, dtype=torch.bool)
        ignore_positions_mask[:, cfg.ignore_positions] = False
        ignore_tokens_mask &= ignore_positions_mask
    return ignore_tokens_mask.to(target_device)


def run_object_feature_batch(
    runner: SaeVisRunner,
    *,
    feature_batch_index: int,
    features: list[int],
    tokens: Int[Tensor, "batch seq"],
    model: HookedSAETransformer,
    encoder: SAE[Any],
    unembed_matrix: Tensor,
    feature_data_generator: FeatureDataGenerator,
    sequence_data_generator: LegacySequenceDataGenerator,
    progress: Any,
    all_consolidated_dfa_results: dict[int, dict[Any, Any]],
) -> SaeVisData:
    with timed_stage(
        runner.cfg.log_performance,
        "activation_and_encode_total",
        device=runner.device,
        capture_runtime_metrics=True,
        batch=feature_batch_index,
        feature_count=len(features),
    ):
        (
            all_feat_acts,
            _,
            feature_resid_dir,
            feature_out_dir,
            corrcoef_neurons,
            corrcoef_encoder,
            batch_dfa_results,
        ) = feature_data_generator.get_feature_data(features, progress)

    with timed_stage(
        runner.cfg.log_performance,
        "logits_projection",
        device=runner.device,
        batch=feature_batch_index,
        feature_count=len(features),
    ):
        logits = einops.einsum(
            feature_resid_dir.to(
                device=unembed_matrix.device,
                dtype=unembed_matrix.dtype,
            ),
            unembed_matrix,
            "feats d_model, d_model d_vocab -> feats d_vocab",
        ).to(runner.device)

    ignore_tokens_mask = _build_ignore_tokens_mask(
        runner.cfg,
        tokens,
        all_feat_acts.device,
    )
    valid_token_count = int(ignore_tokens_mask.sum().item())

    if runner.cfg.log_performance:
        log_perf_event(
            "packaging_shape_summary",
            batch=feature_batch_index,
            feature_count=len(features),
            token_shape=list(tokens.shape),
            valid_token_count=valid_token_count,
        )

    with timed_stage(
        runner.cfg.log_performance,
        "feature_statistics_packaging",
        device=runner.device,
        batch=feature_batch_index,
        feature_count=len(features),
    ):
        feature_stats_input = einops.rearrange(
            all_feat_acts.to(runner.device),
            "batch seq feats -> feats (batch seq)",
        )
        feature_stats = FeatureStatistics.create(
            data=feature_stats_input,
            batch_size=runner.cfg.quantile_feature_batch_size,
        )

    feature_data_dict: dict[int, FeatureData] = {
        feat: FeatureData() for feat in features
    }
    layout = runner.cfg.feature_centric_layout

    with timed_stage(
        runner.cfg.log_performance,
        "feature_table_packaging",
        device=runner.device,
        batch=feature_batch_index,
        feature_count=len(features),
    ):
        feature_tables_data = get_features_table_data(
            feature_out_dir=feature_out_dir,
            corrcoef_neurons=corrcoef_neurons,
            corrcoef_encoder=corrcoef_encoder,
            n_rows=layout.feature_tables_cfg.n_rows,  # type: ignore
        )
        for row_index, feat in enumerate(features):
            feature_data_dict[feat].feature_tables_data = FeatureTablesData(
                **{name: values[row_index] for name, values in feature_tables_data.items()}  # type: ignore
            )

    if batch_dfa_results:
        for feature_idx, feature_data in batch_dfa_results.items():
            all_consolidated_dfa_results[feature_idx].update(feature_data)

    with timed_stage(
        runner.cfg.log_performance,
        "logits_histogram_packaging",
        device=runner.device,
        batch=feature_batch_index,
        feature_count=len(features),
    ):
        for feat, logit_vector in zip(features, logits):
            feature_data_dict[feat].logits_histogram_data = (
                legacy_utils_fns.logits_histogram_from_data(
                    data=logit_vector.to(torch.float32),
                    n_bins=layout.logits_hist_cfg.n_bins,  # type: ignore
                    tickmode="5 ticks",
                    title=None,
                )
            )

    with timed_stage(
        runner.cfg.log_performance,
        "activation_histogram_packaging",
        device=runner.device,
        batch=feature_batch_index,
        feature_count=len(features),
    ):
        masked_feat_acts_by_feature = []
        for row_index, feat in enumerate(features):
            feat_acts = all_feat_acts[..., row_index]
            masked_feat_acts = feat_acts * ignore_tokens_mask
            masked_feat_acts_by_feature.append(masked_feat_acts)
            significance_floor = getattr(
                runner.cfg, "activation_significance_floor", 0.0
            )
            nonzero_feat_acts = masked_feat_acts[masked_feat_acts > significance_floor]
            frac_nonzero = (
                nonzero_feat_acts.numel() / masked_feat_acts.numel()
                if masked_feat_acts.numel() > 0
                else 0.0
            )
            histogram_title = f"ACTIVATIONS<br>DENSITY = {frac_nonzero:.3%}"
            if nonzero_feat_acts.numel() == 0:
                feature_data_dict[feat].acts_histogram_data = ActsHistogramData(
                    title=histogram_title
                )
            else:
                feature_data_dict[feat].acts_histogram_data = (
                    ActsHistogramData.from_data(
                        data=nonzero_feat_acts.to(torch.float32),
                        n_bins=layout.act_hist_cfg.n_bins,  # type: ignore
                        tickmode="5 ticks",
                        title=histogram_title,
                    )
                )

    with timed_stage(
        runner.cfg.log_performance,
        "logits_table_packaging",
        device=runner.device,
        batch=feature_batch_index,
        feature_count=len(features),
    ):
        for feat, logit_vector in zip(features, logits):
            feature_data_dict[feat].logits_table_data = get_logits_table_data(
                logit_vector=logit_vector,
                n_rows=layout.logits_table_cfg.n_rows,  # type: ignore
            )

    with timed_stage(
        runner.cfg.log_performance,
        "sequence_packaging",
        device=runner.device,
        batch=feature_batch_index,
        feature_count=len(features),
    ):
        for row_index, feat in enumerate(features):
            feature_data_dict[feat].sequence_data = (
                sequence_data_generator.get_sequences_data(
                    feat_acts=masked_feat_acts_by_feature[row_index],
                    feat_logits=logits[row_index],
                    resid_post=torch.tensor([]),
                    feature_resid_dir=feature_resid_dir[row_index],
                )
            )
            if runner.cfg.use_dfa:
                feature_data_dict[feat].dfa_data = all_consolidated_dfa_results.get(
                    feat,
                    None,
                )
                from sae_dashboard.sae_vis_runner import (
                    get_decoder_weights_distribution,
                )

                feature_data_dict[feat].decoder_weights_data = (
                    get_decoder_weights_distribution(encoder, model, feat)[0]
                )
            if progress is not None:
                progress[1].update(1)

    artifact_path = runner._write_sequence_replay_artifact(
        feature_batch_index=feature_batch_index,
        features=features,
        tokens=tokens,
        selection_mask=ignore_tokens_mask,
        valid_token_count=valid_token_count,
        all_feat_acts=all_feat_acts,
        logits=logits,
        feature_resid_dir=feature_resid_dir,
    )
    if artifact_path is not None and runner.cfg.log_performance:
        log_perf_event(
            "sequence_replay_artifact",
            batch=feature_batch_index,
            path=artifact_path,
            feature_count=len(features),
            valid_token_count=valid_token_count,
        )

    return SaeVisData(
        cfg=runner.cfg,
        feature_data_dict=feature_data_dict,
        feature_stats=feature_stats,
    )
