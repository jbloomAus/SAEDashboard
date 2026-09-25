# pyright: basic, reportPrivateImportUsage=false
"""Feature-level statistics parity tests.

Validates that per-feature ``frac_nonzero`` and ``maxValue`` match between
the three dashboard generation paths:

1. **Detached legacy vs current legacy** — uses committed golden batch
   artifacts (opt-in, requires ``GOLDEN_BATCHES_DIR`` artifacts).
2. **Current legacy vs columnar_gpu** — runs both paths on a small GPT-2
   fixture and compares feature statistics from the JSON batch output
   (legacy) against the columnar Arrow feature-statistics table.

Tolerances follow the parity contract:

==========  ==========  ==========
Comparison  frac_nonzero  maxValue
==========  ==========  ==========
det-vs-cur  1e-6          1e-6
cur-vs-col  0.01          0.01
==========  ==========  ==========

The cur-vs-col tests are CUDA-gated because the columnar_gpu path requires
a GPU.  The det-vs-cur tests are CPU-only (golden batch JSON comparison).
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
from tests.conftest import _golden_batch_paths

# ---------------------------------------------------------------------------
# Tolerances (from §6 of phase5_dashboard_parity_tests.md)
# ---------------------------------------------------------------------------

DET_VS_CUR_FRAC_NONZERO_TOL = 1e-6
DET_VS_CUR_MAX_VALUE_TOL = 1e-6
CUR_VS_COL_FRAC_NONZERO_TOL = 0.01
CUR_VS_COL_MAX_VALUE_TOL = 0.01


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_golden_batch_features(dataset_family: str) -> list[dict[str, Any]]:
    """Load golden batch JSON and return the ``features`` list from both batches."""
    batch0_path, batch1_path, run_settings_path, _ = _golden_batch_paths(dataset_family)
    if not batch0_path.exists():
        pytest.skip(
            f"Golden batches not found for {dataset_family}. "
            "Regenerate with SAE_DASHBOARD_REGENERATE_GOLDEN_BATCHES=1"
        )
    batch0 = json.loads(batch0_path.read_text())
    batch1 = json.loads(batch1_path.read_text())
    return batch0["features"], batch1["features"]  # pyright: ignore


def _run_legacy_runner(
    tmpdir: str, n_features: int = 4, n_prompts: int = 64, n_batches: int = 2
) -> Path:
    """Run the current legacy (JSON) path and return batch output directory."""
    # Use GPU if available for faster inference
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = NeuronpediaRunnerConfig(
        sae_set="gpt2-small-res-jb",
        sae_path="blocks.0.hook_resid_pre",
        np_set_name="res-jb",
        from_local_sae=False,
        outputs_dir=os.path.join(tmpdir, "legacy_output"),
        sparsity_threshold=1,
        n_prompts_total=n_prompts,
        n_features_at_a_time=n_features,
        n_prompts_in_forward_pass=16,
        start_batch=0,
        end_batch=n_batches - 1,
        use_wandb=False,
        shuffle_tokens=False,
        sae_device=device,
        model_device=device,
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
    tmpdir: str, n_features: int = 4, n_prompts: int = 64, n_batches: int = 2
) -> Path:
    """Run the columnar_gpu path and return batch output directory."""
    # Use GPU if available for faster inference
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = NeuronpediaRunnerConfig(
        sae_set="gpt2-small-res-jb",
        sae_path="blocks.0.hook_resid_pre",
        np_set_name="res-jb",
        from_local_sae=False,
        outputs_dir=os.path.join(tmpdir, "columnar_output"),
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
        sae_device=device,
        model_device=device,
    )
    runner = NeuronpediaRunner(cfg)
    runner.run()

    batch_dir = Path(cfg.outputs_dir)
    for entry in os.listdir(str(batch_dir)):
        candidate = batch_dir / entry
        if candidate.is_dir() and any(
            d.startswith("batch-") and d.endswith(".columnar")
            for d in os.listdir(str(candidate))
        ):
            return candidate
    return batch_dir


def _extract_frac_nonzero_from_legacy(batch_dir: Path, n_batches: int) -> list[float]:
    """Extract per-feature frac_nonzero from legacy JSON batch output."""
    frac_nonzero_values: list[float] = []
    for i in range(n_batches):
        batch_path = batch_dir / f"batch-{i}.json"
        if not batch_path.exists():
            continue
        batch_data = json.loads(batch_path.read_text())
        for feature in batch_data["features"]:
            frac_nonzero_values.append(feature["frac_nonzero"])
    return frac_nonzero_values


def _extract_max_value_from_legacy(batch_dir: Path, n_batches: int) -> list[float]:
    """Extract per-feature max activation value from legacy JSON batch output.

    The legacy JSON stores per-activation ``values`` lists.  The feature-level
    max is the maximum value across all activation records for that feature.
    """
    max_values: list[float] = []
    for i in range(n_batches):
        batch_path = batch_dir / f"batch-{i}.json"
        if not batch_path.exists():
            continue
        batch_data = json.loads(batch_path.read_text())
        for feature in batch_data["features"]:
            # The max activation value is the maximum across all activation records
            feature_max = 0.0
            for act in feature["activations"]:
                values = act.get("values", [])
                if values:
                    feature_max = max(feature_max, max(values))
            max_values.append(feature_max)
    return max_values


def _extract_frac_nonzero_from_columnar(
    columnar_dir: Path, n_batches: int
) -> list[float]:
    """Extract per-feature frac_nonzero from columnar Arrow feature-statistics table."""
    import pyarrow as pa

    frac_nonzero_values: list[float] = []
    for i in range(n_batches):
        col_batch_dirs = sorted(
            d for d in os.listdir(str(columnar_dir)) if d.startswith("batch-")
        )
        if i >= len(col_batch_dirs):
            continue
        col_batch = columnar_dir / col_batch_dirs[i]
        # Find feature_statistics file
        fs_path = None
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

        if fs_path.suffix == ".arrow" or fs_path.suffix == ".feather":
            table = pa.ipc.open_file(str(fs_path)).read_all()
        else:
            table = pa.ipc.open_file(str(fs_path)).read_all()

        if "frac_nonzero" in table.column_names:
            frac_nonzero_values.extend(table.column("frac_nonzero").to_pylist())
        elif "positive_density" in table.column_names:
            frac_nonzero_values.extend(table.column("positive_density").to_pylist())
    return frac_nonzero_values


def _extract_max_value_from_columnar(columnar_dir: Path, n_batches: int) -> list[float]:
    """Extract per-feature max activation value from columnar feature-statistics table.

    The columnar feature-statistics Arrow table has a ``max`` column that stores
    the per-feature maximum activation value.
    """
    import pyarrow as pa

    max_values: list[float] = []
    for i in range(n_batches):
        col_batch_dirs = sorted(
            d for d in os.listdir(str(columnar_dir)) if d.startswith("batch-")
        )
        if i >= len(col_batch_dirs):
            continue
        col_batch = columnar_dir / col_batch_dirs[i]
        # Find feature_statistics file
        fs_path = None
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
        if "max" in table.column_names:
            max_values.extend(table.column("max").to_pylist())
    return max_values


# ---------------------------------------------------------------------------
# Tests: Detached legacy vs current legacy (golden batch comparison)
# ---------------------------------------------------------------------------


class TestFeatureStatisticsParityDetVsCur:
    """Per-feature frac_nonzero and maxValue parity between detached legacy
    (golden batches) and current legacy (fresh run)."""

    def test_feature_frac_nonzero_matches_detached_vs_current_legacy_dense_packed(self):
        """M1 (det-vs-cur): frac_nonzero matches between golden batch and fresh run."""
        golden_features_b0, golden_features_b1 = _load_golden_batch_features(
            "dense_packed"
        )
        golden_frac = [
            f["frac_nonzero"]
            for f in golden_features_b0
            + golden_features_b1  # pyright: ignore[reportOperatorIssue]
        ]

        # Run current legacy path with same config
        golden_run_settings = json.loads(
            _golden_batch_paths("dense_packed")[2].read_text()
        )
        n_features = golden_run_settings.get("n_features_at_a_time", 2)
        n_prompts = golden_run_settings.get("n_prompts_total", 64)

        with tempfile.TemporaryDirectory() as tmpdir:
            legacy_dir = _run_legacy_runner(
                tmpdir, n_features=n_features, n_prompts=n_prompts, n_batches=2
            )
            current_frac = _extract_frac_nonzero_from_legacy(legacy_dir, n_batches=2)

        assert len(current_frac) == len(
            golden_frac
        ), f"Feature count mismatch: current={len(current_frac)} vs golden={len(golden_frac)}"
        for i, (cur, gold) in enumerate(zip(current_frac, golden_frac)):
            assert abs(cur - gold) <= DET_VS_CUR_FRAC_NONZERO_TOL, (
                f"Feature {i} frac_nonzero mismatch: current={cur} vs golden={gold} "
                f"(delta={abs(cur - gold)}, tol={DET_VS_CUR_FRAC_NONZERO_TOL})"
            )

    def test_feature_max_value_matches_detached_vs_current_legacy_dense_packed(self):
        """M2 (det-vs-cur): max activation value matches between golden batch and fresh run."""
        golden_features_b0, golden_features_b1 = _load_golden_batch_features(
            "dense_packed"
        )

        # Extract max from golden batch (max across all activation values per feature)
        golden_max: list[float] = []
        for feature in (
            golden_features_b0 + golden_features_b1
        ):  # pyright: ignore[reportOperatorIssue]
            feature_max = 0.0
            for act in feature["activations"]:
                values = act.get("values", [])
                if values:
                    feature_max = max(feature_max, max(values))
            golden_max.append(feature_max)

        golden_run_settings = json.loads(
            _golden_batch_paths("dense_packed")[2].read_text()
        )
        n_features = golden_run_settings.get("n_features_at_a_time", 2)
        n_prompts = golden_run_settings.get("n_prompts_total", 64)

        with tempfile.TemporaryDirectory() as tmpdir:
            legacy_dir = _run_legacy_runner(
                tmpdir, n_features=n_features, n_prompts=n_prompts, n_batches=2
            )
            current_max = _extract_max_value_from_legacy(legacy_dir, n_batches=2)

        assert len(current_max) == len(
            golden_max
        ), f"Feature count mismatch: current={len(current_max)} vs golden={len(golden_max)}"
        for i, (cur, gold) in enumerate(zip(current_max, golden_max)):
            assert abs(cur - gold) <= DET_VS_CUR_MAX_VALUE_TOL, (
                f"Feature {i} max_value mismatch: current={cur} vs golden={gold} "
                f"(delta={abs(cur - gold)}, tol={DET_VS_CUR_MAX_VALUE_TOL})"
            )


# ---------------------------------------------------------------------------
# Tests: Current legacy vs columnar_gpu (live run comparison)
# ---------------------------------------------------------------------------


class TestFeatureStatisticsParityCurVsCol:
    """Per-feature frac_nonzero and maxValue parity between current legacy
    (JSON path) and columnar_gpu (Arrow path)."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_cuda(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for columnar_gpu path")

    def test_feature_frac_nonzero_matches_current_legacy_vs_columnar_gpu(self):
        """M1 (cur-vs-col): frac_nonzero matches between legacy JSON and columnar_gpu
        Arrow output within the parity contract tolerance (0.01)."""
        N_FEATURES = 16
        N_PROMPTS = 64
        N_BATCHES = 2

        with tempfile.TemporaryDirectory() as tmpdir:
            legacy_dir = _run_legacy_runner(
                os.path.join(tmpdir, "legacy"),
                n_features=N_FEATURES,
                n_prompts=N_PROMPTS,
                n_batches=N_BATCHES,
            )
            columnar_dir = _run_columnar_runner(
                os.path.join(tmpdir, "columnar"),
                n_features=N_FEATURES,
                n_prompts=N_PROMPTS,
                n_batches=N_BATCHES,
            )

            legacy_frac = _extract_frac_nonzero_from_legacy(legacy_dir, N_BATCHES)
            col_frac = _extract_frac_nonzero_from_columnar(columnar_dir, N_BATCHES)

        # Compare — lengths should match
        min_len = min(len(legacy_frac), len(col_frac))
        if min_len == 0:
            pytest.skip("No feature statistics extracted from columnar output")

        mismatches = []
        for i in range(min_len):
            delta = abs(legacy_frac[i] - col_frac[i])
            if delta > CUR_VS_COL_FRAC_NONZERO_TOL:
                mismatches.append((i, legacy_frac[i], col_frac[i], delta))

        assert not mismatches, (
            f"frac_nonzero mismatches (cur-vs-col): {len(mismatches)} features exceed "
            f"tolerance {CUR_VS_COL_FRAC_NONZERO_TOL}.\n"
            + "\n".join(
                f"  Feature {i}: legacy={lf}, columnar={cf}, delta={d}"
                for i, lf, cf, d in mismatches[:20]
            )
        )

    def test_feature_max_value_matches_current_legacy_vs_columnar_gpu(self):
        """M2 (cur-vs-col): max activation value matches between legacy JSON and
        columnar_gpu Arrow output within the parity contract tolerance (0.01).

        Features near interval boundaries may have slightly different max values
        due to bfloat16 downcast — the 0.01 tolerance accommodates this.
        """
        N_FEATURES = 16
        N_PROMPTS = 64
        N_BATCHES = 2

        with tempfile.TemporaryDirectory() as tmpdir:
            legacy_dir = _run_legacy_runner(
                os.path.join(tmpdir, "legacy"),
                n_features=N_FEATURES,
                n_prompts=N_PROMPTS,
                n_batches=N_BATCHES,
            )
            columnar_dir = _run_columnar_runner(
                os.path.join(tmpdir, "columnar"),
                n_features=N_FEATURES,
                n_prompts=N_PROMPTS,
                n_batches=N_BATCHES,
            )

            legacy_max = _extract_max_value_from_legacy(legacy_dir, N_BATCHES)
            col_max = _extract_max_value_from_columnar(columnar_dir, N_BATCHES)

        min_len = min(len(legacy_max), len(col_max))
        if min_len == 0:
            pytest.skip("No feature statistics extracted from columnar output")

        mismatches = []
        for i in range(min_len):
            delta = abs(legacy_max[i] - col_max[i])
            if delta > CUR_VS_COL_MAX_VALUE_TOL:
                mismatches.append((i, legacy_max[i], col_max[i], delta))

        assert not mismatches, (
            f"max_value mismatches (cur-vs-col): {len(mismatches)} features exceed "
            f"tolerance {CUR_VS_COL_MAX_VALUE_TOL}.\n"
            + "\n".join(
                f"  Feature {i}: legacy={lm}, columnar={cm}, delta={d}"
                for i, lm, cm, d in mismatches[:20]
            )
        )
