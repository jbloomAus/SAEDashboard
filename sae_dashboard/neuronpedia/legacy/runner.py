from __future__ import annotations

import gc
import os
import time
from pathlib import Path
from typing import Any, Literal, Protocol

import torch
import wandb
from tqdm import tqdm

from sae_dashboard.components_config import (
    ActsHistogramConfig,
    Column,
    FeatureTablesConfig,
    LogitsHistogramConfig,
    LogitsTableConfig,
    SequencesConfig,
)
from sae_dashboard.layout import SaeVisLayoutConfig
from sae_dashboard.neuronpedia.neuronpedia_converter import NeuronpediaConverter
from sae_dashboard.perf_logging import (
    elapsed_timer,
    io_delta,
    log_perf_event,
    process_io_snapshot,
    timed_stage,
)
from sae_dashboard.sae_vis_data import SaeVisConfig


class LegacyPathConfig(Protocol):
    dashboard_output_format: Literal["legacy_json", "columnar"]
    sequence_selection_backend: Literal["legacy", "columnar_gpu"]


def is_preserved_legacy_path(cfg: LegacyPathConfig) -> bool:
    return (
        cfg.dashboard_output_format == "legacy_json"
        and cfg.sequence_selection_backend == "legacy"
    )


def _build_legacy_layout(runner: Any) -> SaeVisLayoutConfig:
    return SaeVisLayoutConfig(
        columns=[
            Column(
                SequencesConfig(
                    stack_mode="stack-all",
                    buffer=None,  # type: ignore[arg-type]
                    compute_buffer=True,
                    n_quantiles=runner.cfg.n_quantiles,
                    top_acts_group_size=runner.cfg.top_acts_group_size,
                    quantile_group_size=runner.cfg.quantile_group_size,
                ),
                ActsHistogramConfig(),
                LogitsHistogramConfig(),
                LogitsTableConfig(),
                FeatureTablesConfig(n_rows=3),
            )
        ]
    )


def _build_legacy_vis_config(
    runner: Any,
    *,
    features_to_process: list[int],
) -> SaeVisConfig:
    cache_dir = runner.cached_activations_dir
    if not runner.cfg.use_cached_activations:
        # Preserve detached baseline's within-run activation reuse without
        # reusing caches across runs.
        cache_dir = Path(runner.cfg.outputs_dir) / "_activation_cache"

    return SaeVisConfig(
        hook_point=runner.hook_name,  # type: ignore[arg-type]
        features=features_to_process,
        minibatch_size_features=runner.cfg.n_features_at_a_time,
        minibatch_size_tokens=runner.cfg.n_prompts_in_forward_pass,
        quantile_feature_batch_size=runner.cfg.quantile_feature_batch_size,
        verbose=True,
        log_performance=runner.cfg.log_performance,
        profile_rolling_substages=runner.cfg.profile_rolling_substages,
        cleanup_each_minibatch=runner.cfg.cleanup_each_minibatch,
        torch_profile=runner.cfg.torch_profile,
        torch_profile_dir=(
            Path(runner.cfg.torch_profile_dir) if runner.cfg.torch_profile_dir else None
        ),
        device=runner.cfg.sae_device or "cpu",
        feature_centric_layout=_build_legacy_layout(runner),
        perform_ablation_experiments=False,
        dtype=runner.cfg.sae_dtype,
        cache_dir=cache_dir,
        ignore_tokens={
            tok_id
            for tok_id in (
                runner.tokenizer.pad_token_id,  # type: ignore[attr-defined]
                runner.tokenizer.bos_token_id,  # type: ignore[attr-defined]
                runner.tokenizer.eos_token_id,  # type: ignore[attr-defined]
            )
            if tok_id is not None
        },
        ignore_positions=runner.cfg.ignore_positions or [],
        ignore_high_activation_norm_multiple=runner.cfg.ignore_high_activation_norm_multiple,
        use_dfa=runner.cfg.use_dfa,
        use_huggingface=runner.cfg.use_huggingface,
        sequence_replay_artifact_dir=(
            Path(runner.cfg.sequence_replay_artifact_dir)
            if runner.cfg.sequence_replay_artifact_dir
            else None
        ),
        correlation_accumulation_device="cpu",
        rolling_coefficient_num_threads=runner.cfg.rolling_coefficient_num_threads,
        activation_significance_floor=runner.cfg.activation_significance_floor,
        sequence_selection_backend=runner.cfg.sequence_selection_backend,
        dashboard_output_format=runner.cfg.dashboard_output_format,
    )


def run_legacy_batch_loop(
    runner: Any,
    *,
    feature_idx: list[list[int]],
    tokens: torch.Tensor,
) -> None:
    with torch.no_grad():
        for feature_batch_count, features_to_process in tqdm(enumerate(feature_idx)):
            if feature_batch_count < runner.cfg.start_batch:
                continue
            if (
                runner.cfg.end_batch is not None
                and feature_batch_count > runner.cfg.end_batch
            ):
                continue

            output_file = f"{runner.cfg.outputs_dir}/batch-{feature_batch_count}.json"
            if os.path.isfile(output_file):
                logline = (
                    f"\n++++++++++ Skipping Batch #{feature_batch_count} output. "
                    f"File exists: {output_file} ++++++++++\n"
                )
                print(logline)
                continue

            print(f"========== Running Batch #{feature_batch_count} ==========")
            runner._log_resource_snapshot(f"pre_batch_{feature_batch_count}")

            feature_vis_config_gpt = _build_legacy_vis_config(
                runner,
                features_to_process=features_to_process,
            )

            batch_start_time = time.perf_counter()
            batch_start_io = process_io_snapshot()
            runner._log_batch_boundary_snapshot(
                "pre_batch", feature_batch_count, batch_start_io
            )
            feature_data = runner._run_feature_batch_with_optional_profile(
                feature_vis_config_gpt,
                tokens,
                feature_batch_count,
            )
            runner._log_resource_snapshot(f"after_feature_run_{feature_batch_count}")

            converter_input_artifact = runner._write_converter_input_artifact(
                feature_data,
                feature_batch_count,
            )

            runner.cfg.model_id = runner.model_id
            runner.cfg.layer = runner.layer
            with timed_stage(
                runner.cfg.log_performance,
                "neuronpedia_conversion_and_json_serialization",
                batch=feature_batch_count,
                feature_count=len(features_to_process),
            ):
                json_object = NeuronpediaConverter.convert_to_np_json(
                    runner.model,
                    feature_data,
                    runner.cfg,
                    runner.vocab_dict,
                )

            write_start_io = process_io_snapshot()
            with elapsed_timer() as write_timing:
                with open(output_file, "w") as f:
                    f.write(json_object)
            write_end_io = process_io_snapshot()
            if runner.cfg.log_performance:
                output_bytes = len(json_object.encode("utf-8"))
                write_wall_s = max(write_timing.get("wall_s", 0.0), 1e-9)
                log_perf_event(
                    "disk_write",
                    batch=feature_batch_count,
                    path=output_file,
                    output_bytes=output_bytes,
                    wall_s=write_wall_s,
                    output_mib_per_s=output_bytes / (1024**2) / write_wall_s,
                    process_io_delta=io_delta(write_start_io, write_end_io),
                )
                if converter_input_artifact is not None:
                    log_perf_event(
                        "converter_input_artifact",
                        batch=feature_batch_count,
                        path=str(converter_input_artifact),
                        size_bytes=converter_input_artifact.stat().st_size,
                    )
            print(f"Output written to {output_file}")

            batch_end_io = process_io_snapshot()
            if runner.cfg.log_performance:
                log_perf_event(
                    "batch_total",
                    batch=feature_batch_count,
                    wall_s=time.perf_counter() - batch_start_time,
                    process_io_delta=io_delta(batch_start_io, batch_end_io),
                )
            runner._log_batch_boundary_snapshot(
                "post_batch", feature_batch_count, batch_end_io
            )
            runner._log_resource_snapshot(f"post_batch_{feature_batch_count}")

            logline = (
                f"\n========== Completed Batch #{feature_batch_count} output: "
                f"{output_file} ==========\n"
            )
            if runner.cfg.use_wandb:
                wandb.log(
                    {"batch": feature_batch_count},
                    step=feature_batch_count,
                )
            del feature_data
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            runner._release_unused_host_memory()
            runner._log_resource_snapshot(f"post_batch_cleanup_{feature_batch_count}")
            print(logline)
