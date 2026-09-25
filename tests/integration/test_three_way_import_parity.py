# pyright: basic, reportPrivateImportUsage=false
"""Three-way generation parity tests (L1, L2).

Validates generation-layer parity across the three dashboard generation paths:

1. **Detached legacy** (golden batch artifacts) — opt-in, requires regenerated golden batches
2. **Current legacy** (JSON path) — always available
3. **Columnar_gpu** (Arrow/Parquet path) — requires CUDA

These tests compare activation row counts and feature statistics
(``frac_nonzero``, ``maxValue``) across all three paths.  Conversion
and DB import are validated separately — see the interpretune test
suite and the neuronpedia repo for those integration layers.

Tolerances follow the parity contract (§6 of phase5_dashboard_parity_tests.md):

=================  ========================  ==========  ==========
Comparison          activation_rows_delta     frac_nonzero  maxValue
=================  ========================  ==========  ==========
det vs cur          0 (exact match)           1e-6          1e-6
cur vs col (RTE)    ≤ 2500                    0.01          0.01
cur vs col (Monology) ≤ 5000                  0.01          0.01
col vs col (self)   0 (byte-identical)        1e-6          1e-6
=================  ========================  ==========  ==========

.. note::
   This is a heavy integration test (loads GPT-2, runs inference).  It takes
   ~22 s per test on GPU (CUDA required).  On CPU it would take 20+ minutes.
   The test is skipped automatically when CUDA is not available.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pytest
import torch

from sae_dashboard.neuronpedia.neuronpedia_runner import (
    NeuronpediaRunner,
    NeuronpediaRunnerConfig,
)
from tests.conftest import _golden_batch_paths, _golden_prompt_cache

# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------

DET_VS_CUR_ROW_DELTA = 0  # exact match
DET_VS_CUR_FRAC_TOL = 1e-6
DET_VS_CUR_MAX_TOL = 1e-6
CUR_VS_COL_ROW_DELTA = 2500  # RTE contract tolerance
CUR_VS_COL_FRAC_TOL = 0.01
CUR_VS_COL_MAX_TOL = 0.01
COL_SELF_ROW_DELTA = 0


# ---------------------------------------------------------------------------
# Helpers — runner
# ---------------------------------------------------------------------------


def _device() -> str:
    """Prefer GPU, fall back to CPU."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def _run_legacy_runner(
    tmpdir: str,
    n_features: int = 4,
    n_prompts: int = 64,
    n_batches: int = 2,
) -> Path:
    """Run the current legacy (JSON) path and return batch output directory."""
    dev = _device()
    cfg = NeuronpediaRunnerConfig(
        sae_set="gpt2-small-res-jb",
        sae_path="blocks.0.hook_resid_pre",
        np_set_name="res-jb",
        from_local_sae=False,
        outputs_dir=os.path.join(tmpdir, "current_legacy"),
        sparsity_threshold=1,
        n_prompts_total=n_prompts,
        n_features_at_a_time=n_features,
        n_prompts_in_forward_pass=16,
        start_batch=0,
        end_batch=n_batches - 1,
        use_wandb=False,
        shuffle_tokens=False,
        sae_device=dev,
        model_device=dev,
    )
    runner = NeuronpediaRunner(cfg)
    runner.run()

    batch_dir = Path(cfg.outputs_dir)
    for entry in os.listdir(str(batch_dir)):
        candidate = batch_dir / entry
        if candidate.is_dir() and (candidate / "batch-0.json").exists():
            return candidate
    return batch_dir


def _run_current_legacy_matching_golden(
    tmpdir: str,
    golden_settings: dict[str, Any],
    pretokenized_dataset_path: str | None = None,
) -> Path:
    """Run the current legacy path with config mirrored from golden batches.

    This replicates the approach used by
    ``test_current_legacy_matches_golden_dense_packed`` — it reads the golden
    batch `run_settings.json` and creates a matching ``NeuronpediaRunnerConfig``
    so the fresh run is apples-to-apples with the committed golden batch.
    For families generated from a committed prompt cache (example_aligned),
    pass ``pretokenized_dataset_path`` so the rerun consumes the same cache.
    """
    dev = _device()
    extra_kwargs: dict[str, Any] = {}
    if pretokenized_dataset_path is not None:
        extra_kwargs["pretokenized_dataset_path"] = pretokenized_dataset_path
        if golden_settings.get("n_tokens_in_prompt"):
            extra_kwargs["n_tokens_in_prompt"] = golden_settings["n_tokens_in_prompt"]
    cfg = NeuronpediaRunnerConfig(
        sae_set=golden_settings.get("sae_set", "gpt2-small-res-jb"),
        sae_path=golden_settings.get("sae_path", "blocks.0.hook_resid_pre"),
        np_set_name=golden_settings.get("np_set_name", "res-jb"),
        from_local_sae=golden_settings.get("from_local_sae", False),
        outputs_dir=str(Path(tmpdir) / "runner_output"),
        sparsity_threshold=golden_settings.get("sparsity_threshold", 1),
        n_prompts_total=golden_settings.get("n_prompts_total", 64),
        n_features_at_a_time=golden_settings.get("n_features_at_a_time", 2),
        n_prompts_in_forward_pass=golden_settings.get("n_prompts_in_forward_pass", 16),
        start_batch=0,
        end_batch=1,
        use_wandb=False,
        shuffle_tokens=False,
        sae_device=dev,
        model_device=dev,
        **extra_kwargs,
    )
    runner = NeuronpediaRunner(cfg)
    runner.run()

    batch_dir = Path(cfg.outputs_dir)
    for entry in os.listdir(str(batch_dir)):
        candidate = batch_dir / entry
        if candidate.is_dir() and (candidate / "batch-0.json").exists():
            return candidate
    return batch_dir


def _run_columnar_runner(
    tmpdir: str,
    n_features: int = 4,
    n_prompts: int = 64,
    n_batches: int = 2,
    pretokenized_dataset_path: str | None = None,
    n_tokens_in_prompt: int | None = None,
) -> Path:
    """Run the columnar_gpu path and return batch output directory."""
    dev = _device()
    extra_kwargs: dict[str, Any] = {}
    if pretokenized_dataset_path is not None:
        extra_kwargs["pretokenized_dataset_path"] = pretokenized_dataset_path
    if n_tokens_in_prompt is not None:
        extra_kwargs["n_tokens_in_prompt"] = n_tokens_in_prompt
    cfg = NeuronpediaRunnerConfig(
        sae_set="gpt2-small-res-jb",
        sae_path="blocks.0.hook_resid_pre",
        np_set_name="res-jb",
        from_local_sae=False,
        outputs_dir=os.path.join(tmpdir, "columnar_gpu"),
        sparsity_threshold=1,
        n_prompts_total=n_prompts,
        n_features_at_a_time=n_features,
        n_prompts_in_forward_pass=16,
        start_batch=0,
        end_batch=n_batches - 1,
        use_wandb=False,
        shuffle_tokens=False,
        dashboard_output_format="columnar",
        sequence_selection_backend="columnar_gpu",
        feature_statistics_backend="arrow",
        logits_histogram_backend="arrow",
        activation_histogram_backend="torch",
        columnar_artifact_format="arrow",
        sae_device=dev,
        model_device=dev,
        **extra_kwargs,
    )
    runner = NeuronpediaRunner(cfg)
    runner.run()

    batch_dir = Path(cfg.outputs_dir)
    for entry in os.listdir(str(batch_dir)):
        candidate = batch_dir / entry
        if candidate.is_dir() and any(
            d.startswith("batch-") for d in os.listdir(str(candidate))
        ):
            return candidate
    return batch_dir


# ---------------------------------------------------------------------------
# Helpers — extraction
# ---------------------------------------------------------------------------


def _legacy_row_count(batch_dir: Path, n_batches: int) -> int:
    total = 0
    for i in range(n_batches):
        bp = batch_dir / f"batch-{i}.json"
        if bp.exists():
            batch = json.loads(bp.read_text())
            total += sum(len(f["activations"]) for f in batch["features"])
    return total


def _legacy_feature_stats(
    batch_dir: Path, n_batches: int
) -> tuple[list[float], list[float]]:
    """Return (frac_nonzero_list, max_value_list) per feature."""
    frac: list[float] = []
    maxv: list[float] = []
    for i in range(n_batches):
        bp = batch_dir / f"batch-{i}.json"
        if not bp.exists():
            continue
        batch = json.loads(bp.read_text())
        for feature in batch["features"]:
            frac.append(feature["frac_nonzero"])
            fm = 0.0
            for act in feature["activations"]:
                vals = act.get("values", [])
                if vals:
                    fm = max(fm, max(vals))
            maxv.append(fm)
    return frac, maxv


def _columnar_row_count(batch_dir: Path, n_batches: int) -> int:
    """Sum activation row counts from columnar manifests."""
    total = 0
    col_batch_dirs = sorted(
        d for d in os.listdir(str(batch_dir)) if d.startswith("batch-")
    )
    for i in range(n_batches):
        if i >= len(col_batch_dirs):
            continue
        col_batch = batch_dir / col_batch_dirs[i]
        # Try direct manifest
        m_path = col_batch / "manifest.json"
        if not m_path.exists():
            # Try one level deep
            for sub in os.listdir(str(col_batch)):
                sub_path = col_batch / sub
                if sub_path.is_dir() and (sub_path / "manifest.json").exists():
                    m_path = sub_path / "manifest.json"
                    break
        if m_path.exists():
            manifest = json.loads(m_path.read_text())
            ar = manifest.get("activation_rows", {})
            if isinstance(ar, dict):
                total += ar.get("total_rows", 0)
            elif isinstance(ar, (int, float)):
                total += int(ar)
    return total


def _columnar_feature_stats(
    batch_dir: Path,
    n_batches: int,
) -> tuple[list[float], list[float]]:
    """Return (frac_nonzero_list, max_value_list) from columnar feature-statistics tables."""
    import pyarrow as pa

    frac: list[float] = []
    maxv: list[float] = []

    col_batch_dirs = sorted(
        d for d in os.listdir(str(batch_dir)) if d.startswith("batch-")
    )
    for i in range(n_batches):
        if i >= len(col_batch_dirs):
            continue
        col_batch = batch_dir / col_batch_dirs[i]
        # Find feature_statistics file
        fs_path: Path | None = None
        for sub in os.listdir(str(col_batch)):
            sub_path = col_batch / sub
            if sub_path.is_dir():
                for f in os.listdir(str(sub_path)):
                    if f.startswith("feature_statistics"):
                        fs_path = sub_path / f
                        break
            elif sub.startswith("feature_statistics"):
                fs_path = sub_path
        if fs_path is None:
            continue
        table = pa.ipc.open_file(str(fs_path)).read_all()
        if "frac_nonzero" in table.column_names:
            frac.extend(table.column("frac_nonzero").to_pylist())
        elif "positive_density" in table.column_names:
            frac.extend(table.column("positive_density").to_pylist())
        if "max" in table.column_names:
            maxv.extend(table.column("max").to_pylist())
    return frac, maxv


def _golden_row_count(dataset_family: str) -> int:
    """Total activation rows from golden batch artifacts."""
    b0, b1, _, _ = _golden_batch_paths(dataset_family)
    if not b0.exists():
        pytest.skip(f"Golden batches not found for {dataset_family}")
    total = 0
    for bp in (b0, b1):
        batch = json.loads(bp.read_text())
        total += sum(len(f["activations"]) for f in batch["features"])
    return total


def _golden_feature_stats(dataset_family: str) -> tuple[list[float], list[float]]:
    """Return (frac_nonzero_list, max_value_list) from golden batches."""
    b0, b1, _, _ = _golden_batch_paths(dataset_family)
    if not b0.exists():
        pytest.skip(f"Golden batches not found for {dataset_family}")
    frac: list[float] = []
    maxv: list[float] = []
    for bp in (b0, b1):
        batch = json.loads(bp.read_text())
        for feature in batch["features"]:
            frac.append(feature["frac_nonzero"])
            fm = 0.0
            for act in feature["activations"]:
                vals = act.get("values", [])
                if vals:
                    fm = max(fm, max(vals))
            maxv.append(fm)
    return frac, maxv


def _golden_run_settings(dataset_family: str) -> dict[str, Any]:
    """Load golden batch run settings."""
    _, _, run_settings_path, _ = _golden_batch_paths(dataset_family)
    if not run_settings_path.exists():
        pytest.skip(f"Golden run_settings not found for {dataset_family}")
    return json.loads(run_settings_path.read_text())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestThreeWayGenerationParity:
    """End-to-end 3-path generation-layer parity.

    Compares activation row counts and per-feature statistics
    (``frac_nonzero``, ``maxValue``) across detached legacy
    (golden batches), current legacy, and columnar_gpu paths.

    All tests require CUDA (columnar_gpu path) and golden batch
    artifacts.  They are skipped automatically when prerequisites
    are not met.
    """

    # ------------------------------------------------------------------
    # L1: dense_packed three-way parity
    # ------------------------------------------------------------------

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available — columnar_gpu path requires GPU",
    )
    def test_three_way_generation_parity_dense_packed(self):
        """L1: Validate row counts and feature statistics across all three
        generation paths using the ``dense_packed`` golden batch family."""
        N_PROMPTS = 64
        N_BATCHES = 2
        DATASET_FAMILY = "dense_packed"

        # 1. Gather golden batch baselines (opt-in)
        golden_rows = _golden_row_count(DATASET_FAMILY)
        golden_frac, golden_max = _golden_feature_stats(DATASET_FAMILY)
        golden_settings = _golden_run_settings(DATASET_FAMILY)

        # 2. Run current legacy matching golden batch config
        with tempfile.TemporaryDirectory() as tmpdir:
            legacy_dir = _run_current_legacy_matching_golden(
                tmpdir,
                golden_settings,
            )
            legacy_rows = _legacy_row_count(legacy_dir, n_batches=N_BATCHES)
            legacy_frac, legacy_max = _legacy_feature_stats(
                legacy_dir, n_batches=N_BATCHES
            )

        # 3. Run columnar_gpu (2 features — columnar path has heavier serialization)
        with tempfile.TemporaryDirectory() as tmpdir:
            columnar_dir = _run_columnar_runner(
                tmpdir,
                n_features=2,
                n_prompts=N_PROMPTS,
                n_batches=N_BATCHES,
            )
            col_rows = _columnar_row_count(columnar_dir, n_batches=N_BATCHES)
            col_frac, col_max = _columnar_feature_stats(
                columnar_dir, n_batches=N_BATCHES
            )

        # 4. Validate det vs cur row count (exact match)
        assert abs(legacy_rows - golden_rows) <= DET_VS_CUR_ROW_DELTA, (
            f"Det vs cur row delta: legacy={legacy_rows}, golden={golden_rows}, "
            f"delta={abs(legacy_rows - golden_rows)}, max={DET_VS_CUR_ROW_DELTA}"
        )

        # 5. Validate cur vs col row count (within RTE contract: ≤ 2500)
        row_delta_cc = abs(legacy_rows - col_rows)
        assert row_delta_cc <= CUR_VS_COL_ROW_DELTA, (
            f"Cur vs col row delta: legacy={legacy_rows}, columnar={col_rows}, "
            f"delta={row_delta_cc}, max={CUR_VS_COL_ROW_DELTA}"
        )

        # 6. Validate det vs cur feature statistics
        min_len = min(len(legacy_frac), len(golden_frac))
        assert min_len > 0, "No features to compare"
        for i in range(min_len):
            assert abs(legacy_frac[i] - golden_frac[i]) <= DET_VS_CUR_FRAC_TOL
            assert abs(legacy_max[i] - golden_max[i]) <= DET_VS_CUR_MAX_TOL

        # 7. Validate cur vs col feature statistics
        min_len_cc = min(len(legacy_frac), len(col_frac))
        assert min_len_cc > 0, "No columnar features to compare"
        for i in range(min_len_cc):
            assert (
                abs(legacy_frac[i] - col_frac[i]) <= CUR_VS_COL_FRAC_TOL
            ), f"Feature {i} frac_nonzero: legacy={legacy_frac[i]}, col={col_frac[i]}"
            assert (
                abs(legacy_max[i] - col_max[i]) <= CUR_VS_COL_MAX_TOL
            ), f"Feature {i} max_value: legacy={legacy_max[i]}, col={col_max[i]}"

    # ------------------------------------------------------------------
    # L2: example_aligned three-way parity
    # ------------------------------------------------------------------

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available — columnar_gpu path requires GPU",
    )
    def test_three_way_generation_parity_example_aligned(self):
        """L2: Example-aligned variant of the three-way generation parity test."""
        N_PROMPTS = 64
        N_BATCHES = 2
        DATASET_FAMILY = "example_aligned"

        golden_rows = _golden_row_count(DATASET_FAMILY)
        golden_frac, golden_max = _golden_feature_stats(DATASET_FAMILY)
        golden_settings = _golden_run_settings(DATASET_FAMILY)
        # The example_aligned family is generated from a committed max-prompt-pad prompt
        # cache (in-tree current-legacy provenance); reruns must consume the same cache.
        prompt_cache = _golden_prompt_cache(DATASET_FAMILY)
        assert prompt_cache is not None, (
            "example_aligned golden family requires its committed prompt_cache; "
            "regenerate with tests/acceptance/generate_golden_batches.py"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            legacy_dir = _run_current_legacy_matching_golden(
                tmpdir,
                golden_settings,
                pretokenized_dataset_path=str(prompt_cache),
            )
            legacy_rows = _legacy_row_count(legacy_dir, n_batches=N_BATCHES)
            legacy_frac, legacy_max = _legacy_feature_stats(
                legacy_dir, n_batches=N_BATCHES
            )

        assert abs(legacy_rows - golden_rows) <= DET_VS_CUR_ROW_DELTA

        min_len = min(len(legacy_frac), len(golden_frac))
        assert min_len > 0
        for i in range(min_len):
            assert abs(legacy_frac[i] - golden_frac[i]) <= DET_VS_CUR_FRAC_TOL
            assert abs(legacy_max[i] - golden_max[i]) <= DET_VS_CUR_MAX_TOL

        # 3. Run columnar_gpu (2 features — columnar path has heavier serialization)
        with tempfile.TemporaryDirectory() as tmpdir:
            columnar_dir = _run_columnar_runner(
                tmpdir,
                n_features=2,
                n_prompts=N_PROMPTS,
                n_batches=N_BATCHES,
                pretokenized_dataset_path=str(prompt_cache),
                n_tokens_in_prompt=golden_settings.get("n_tokens_in_prompt"),
            )
            col_rows = _columnar_row_count(columnar_dir, n_batches=N_BATCHES)
            col_frac, col_max = _columnar_feature_stats(
                columnar_dir, n_batches=N_BATCHES
            )

        row_delta_cc = abs(legacy_rows - col_rows)
        assert row_delta_cc <= CUR_VS_COL_ROW_DELTA

        min_len_cc = min(len(legacy_frac), len(col_frac))
        assert min_len_cc > 0
        for i in range(min_len_cc):
            assert abs(legacy_frac[i] - col_frac[i]) <= CUR_VS_COL_FRAC_TOL
            assert abs(legacy_max[i] - col_max[i]) <= CUR_VS_COL_MAX_TOL
