# SAEDashboard Tests

Test suite for the SAEDashboard package, covering unit, integration, acceptance,
and parity tests for the Neuronpedia dashboard generation pipeline.

## Directory Layout

```
tests/
├── acceptance/         # Golden-batch comparison, end-to-end runner output tests
│   ├── golden_batches/ # Committed reference artifacts (dense_packed, example_aligned)
│   │   ├── dense_packed/
│   │   └── example_aligned/
│   └── generate_golden_batches.py
├── benchmark/          # Benchmark harness for SaeVisRunner
├── fixtures/           # Shared test data and configs
├── integration/        # Heavy tests: model loading, CUDA, multiple generation paths
├── unit/               # Fast CPU-only tests: mechanisms, statistics, generators
├── conftest.py         # Shared fixtures (model, autoencoder, tokens, golden batch helpers)
└── helpers.py          # Utility functions for test setup
```

## Optional Dependencies

Some tests require additional infrastructure that is not available in a vanilla
dev environment.  These tests are **opt-in** — they skip gracefully when their
prerequisites are not met.

### Golden Batch Artifacts (detached-legacy comparison)

The golden batch tests (`test_current_legacy_matches_golden_*`,
`test_three_way_generation_to_import_parity_*`) compare the current legacy runner
output against committed reference artifacts stored in
`tests/acceptance/golden_batches/`.

**Prerequisites:**

- Golden batch artifacts must exist under `tests/acceptance/golden_batches/<family>/`:
  `batch-0.json`, `batch-1.json`, `run_settings.json`, `sae_lens.json`
- These are regenerated from the detached baseline worktree at `SAEDashboard-7886eaa`
  (see `tests/acceptance/generate_golden_batches.py`)
- The artifacts are committed to the repository and should be regenerated when
  the golden batch generation path changes

**How to skip:** If the golden batch files are missing, the tests call
`pytest.skip("Golden batches not found for ...")` automatically.

### CUDA (GPU) — columnar_gpu path

Tests that exercise the `columnar_gpu` backend require a CUDA-capable GPU.

**Prerequisites:**

- `torch.cuda.is_available() == True`
- Sufficient VRAM to load GPT-2 Small (~500 MB) and the SAE weights (~100 MB)

**How to skip:** ``@pytest.mark.skipif(not torch.cuda.is_available(), ...)`` on
each test method or class that requires CUDA.  Tests in
``TestThreeWayGenerationParity`` and ``TestFeatureStatisticsParityCurVsCol`` are
marked with this decorator so they are skipped automatically on CPU-only
machines.

## Test Categories

### Unit Tests (`tests/unit/`)

Fast, CPU-only tests that validate individual components.  No model loading required.

| File | Description | Dependencies |
|------|-------------|--------------|
| `test_feature_data_generator.py` | Feature data generation correctness | None |
| `test_legacy_columnar_parity_mechanisms.py` | bfloat16 + interval boundary mechanisms | None |
| `test_feature_statistics_parity.py` | `frac_nonzero`, `maxValue` parity (M1, M2) | **CUDA** (CurVsCol only) |
| `test_neuronpedia_converter.py` | JSON converter output format | None |
| `test_sequence_data_generator.py` | Sequence/activation row generation | None |

### Integration Tests (`tests/integration/`)

Heavier tests that load models and/or run the full generation pipeline.

| File | Description | Dependencies |
|------|-------------|--------------|
| `test_legacy_columnar_parity.py` | Legacy vs columnar_gpu mechanism validation | **CUDA** |
| `test_three_way_import_parity.py` | 3-path generation parity (L1, L2) | **CUDA**, **Golden batches** |
| `test_sae_vis_integration.py` | SaeVisRunner end-to-end | **CUDA** |
| `test_transcoder_integration.py` | Transcoder dashboard generation | **CUDA**, gated model weights |

### Acceptance Tests (`tests/acceptance/`)

Reference-output tests that compare against committed artifacts.

| File | Description | Dependencies |
|------|-------------|--------------|
| `test_neuronpedia_runner.py` | Golden batch comparison (dense_packed, example_aligned) | **Golden batches** |
| `test_neuronpedia_runner_output.py` | Output format validation | None |

## Running Tests

### All fast unit tests (no GPU, no golden batches)

```bash
cd /path/to/SAEDashboard
python -m pytest tests/unit/ \
  -k "not (CurVsCol or columnar_gpu)" \
  -v
```

### Golden batch tests (opt-in)

```bash
# Requires regenerated golden batch artifacts
python -m pytest tests/acceptance/test_neuronpedia_runner.py \
  -k "golden" -v
```

### Full parity test suite (GPU + golden batches required)

```bash
# Three-way generation parity (L1)
python -m pytest tests/integration/test_three_way_import_parity.py \
  -k "dense_packed" -v

# Feature statistics parity (M1, M2 — det-vs-cur, CPU-only part)
python -m pytest tests/unit/test_feature_statistics_parity.py \
  -k "DetVsCur" -v

# Feature statistics parity (M1, M2 — cur-vs-col, CUDA required)
python -m pytest tests/unit/test_feature_statistics_parity.py \
  -k "CurVsCol" -v
```

### With timeout wrapper (recommended for CI)

```bash
python scripts/run_parity_tests.py \
  --test-class DetVsCur \
  --timeout 180
```

## Golden Batch Regeneration

When the dashboard generation logic changes, the golden batches must be
regenerated from the detached baseline worktree:

```bash
SAE_DASHBOARD_REGENERATE_GOLDEN_BATCHES=1 \
python -m pytest tests/acceptance/test_neuronpedia_runner.py \
  -k "golden" -v
```

Or use the direct generation script:

```bash
python tests/acceptance/generate_golden_batches.py
```

## Adding New Tests

- **Unit tests**: Place in `tests/unit/`.  No model loading, no CUDA dependency.
- **Integration tests**: Place in `tests/integration/`.  Use autouse fixtures to
  skip when prerequisites (CUDA, golden batches, DB) are unavailable.
- **Acceptance tests**: Place in `tests/acceptance/`.  Compare against committed
  reference artifacts.

### Prerequisite-skip pattern

```python
import pytest
import torch

class TestMyNewFeature:
    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available — this test requires a GPU",
    )
    def test_with_gpu_requirement(self):
        ...
```

For tests that require golden batch artifacts, use the pattern from
``conftest.py`` (``_golden_batch_paths`` / ``pytest.skip(...)``).

## Upstream Reference

The current upstream/main reference for tracking changes since is commit
`db4839a` on the `jbloomAus/SAEDashboard` repository.  The following test
files were added or significantly extended after that baseline:

| File | Added In | Purpose |
|------|----------|---------|
| `tests/unit/test_legacy_columnar_parity_mechanisms.py` | Phase 5 | bfloat16 / interval boundary mechanism validation |
| `tests/unit/test_feature_statistics_parity.py` | Phase 5 | Per-feature `frac_nonzero` + `maxValue` parity |
| `tests/integration/test_three_way_import_parity.py` | Phase 5 | 3-path generation parity |
| `tests/acceptance/test_neuronpedia_runner.py` (extended) | Phase 5 | Per-feature activation count check in `example_aligned` golden batch test |
