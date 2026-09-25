# pyright: basic, reportPrivateImportUsage=false
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, cast

import numpy as np
import pytest
import torch
from sae_lens import SAE, HookedSAETransformer
from torch import Tensor

import sae_dashboard.perf_logging as perf_logging
import sae_dashboard.sae_vis_runner as sae_vis_runner_module
from sae_dashboard.components import FeatureTablesData, LogitsTableData
from sae_dashboard.neuronpedia.neuronpedia_runner_config import (
    DEFAULT_PARQUET_ROW_GROUP_SIZE,
    NeuronpediaRunnerConfig,
)
from sae_dashboard.sae_vis_data import SaeVisColumnarData, SaeVisConfig
from sae_dashboard.sae_vis_runner import SaeVisRunner
from sae_dashboard.sequence_data_generator import (
    SequenceCoordinateTable,
    SequenceSelectionBackend,
)


class _FakeFeatureDataGenerator:
    def get_feature_data(
        self, features: list[int], progress: list[Any] | None
    ) -> tuple[Tensor, None, Tensor, Tensor, None, None, None]:
        del features, progress
        all_feat_acts = torch.tensor([[[1.0], [0.0]]], dtype=torch.float32)
        feature_resid_dir = torch.tensor([[1.0]], dtype=torch.float32)
        feature_out_dir = torch.tensor([[1.0]], dtype=torch.float32)
        return all_feat_acts, None, feature_resid_dir, feature_out_dir, None, None, None


class _FakeSequenceDataGenerator:
    def __init__(self, cfg: SaeVisConfig, tokens: Tensor, W_U: Tensor) -> None:
        self.cfg = cfg
        self.tokens = tokens
        self.W_U = W_U
        self.buffer: tuple[int, int] | None = None

    def get_indices_dicts_columnar_gpu_batched(
        self,
        buffer: tuple[int, int] | None,
        all_feat_acts: Tensor,
        selection_mask: Tensor | None = None,
        selection_device: Any = None,
        feature_chunk_size: int = 64,
    ) -> list[tuple[dict[str, Tensor], Tensor, int]]:
        del buffer, selection_mask, selection_device, feature_chunk_size
        empty_indices = torch.zeros((0, 2), dtype=torch.long)
        return [
            ({"TOP ACTIVATIONS<br>MAX = 1.000": empty_indices}, empty_indices, 0)
            for _ in range(all_feat_acts.shape[-1])
        ]

    def get_sequences_data(
        self,
        feat_acts: Tensor,
        feat_logits: Tensor | None,
        resid_post: Tensor | None,
        feature_resid_dir: Tensor,
        selection_mask: Tensor | None = None,
        selection_backend: SequenceSelectionBackend = "legacy",
    ) -> list[Any]:
        del (
            feat_acts,
            feat_logits,
            resid_post,
            feature_resid_dir,
            selection_mask,
            selection_backend,
        )
        return []

    def get_sequence_coordinate_table(
        self,
        feat_acts: Tensor,
        feat_logits: Tensor | None,
        resid_post: Tensor | None,
        feature_resid_dir: Tensor,
        selection_mask: Tensor | None = None,
        selection_backend: SequenceSelectionBackend = "legacy",
        precomputed_selection: tuple[dict[str, Tensor], Tensor, int] | None = None,
    ) -> SequenceCoordinateTable:
        del (
            feat_acts,
            feat_logits,
            resid_post,
            feature_resid_dir,
            selection_mask,
            selection_backend,
            precomputed_selection,
        )
        return SequenceCoordinateTable(
            group_names=["TOP ACTIVATIONS<br>MAX = 1.000"],
            group_sizes=[1],
            original_indices=torch.tensor([0], dtype=torch.long),
            qualifying_token_indices=torch.tensor([1], dtype=torch.long),
            source_token_indices=torch.tensor([[0, 1]], dtype=torch.long),
            token_ids=torch.tensor([[101, 102]], dtype=torch.long),
            feat_acts=np.array([[0.123, 0.555]], dtype=np.float64),
            token_logits=torch.tensor([[0.01, 0.02]], dtype=torch.float32),
        )


class _FakeEncoder:
    def __init__(self) -> None:
        self.hook_z_reshaping_mode = False

    def fold_W_dec_norm(self) -> None:
        return None


def _capture_perf_events(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    perf_events: list[dict[str, object]] = []

    def _record(event: str, /, **fields: object) -> None:
        perf_events.append({"event": event, **fields})

    monkeypatch.setattr(perf_logging, "log_perf_event", _record)
    monkeypatch.setattr(sae_vis_runner_module, "log_perf_event", _record)
    monkeypatch.setattr(
        "sae_dashboard.neuronpedia.legacy.sae_vis_runner.log_perf_event",
        _record,
    )
    return perf_events


def _read_columnar_table(table_path: Path) -> Any:
    if table_path.suffix == ".arrow":
        pyarrow = importlib.import_module("pyarrow")
        pyarrow_ipc = importlib.import_module("pyarrow.ipc")
        with pyarrow.memory_map(str(table_path), "r") as source:
            return pyarrow_ipc.RecordBatchFileReader(source).read_all()
    if table_path.suffix == ".parquet":
        pyarrow_parquet = importlib.import_module("pyarrow.parquet")
        return pyarrow_parquet.read_table(table_path)
    raise AssertionError(f"Unsupported columnar table suffix: {table_path.suffix}")


def test_SaeVisRunner_legacy_profiling_surfaces_stage_timings_and_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    perf_events = _capture_perf_events(monkeypatch)
    monkeypatch.setattr(
        "sae_dashboard.sae_vis_runner.FeatureDataGeneratorFactory.create",
        lambda cfg, model, encoder, tokens: _FakeFeatureDataGenerator(),
    )
    monkeypatch.setattr(
        "sae_dashboard.sae_vis_runner.SequenceDataGenerator",
        _FakeSequenceDataGenerator,
    )
    monkeypatch.setattr(
        "sae_dashboard.sae_vis_runner.LegacySequenceDataGenerator",
        _FakeSequenceDataGenerator,
    )
    monkeypatch.setattr(
        "sae_dashboard.sae_vis_runner.get_features_table_data",
        lambda **kwargs: {
            name: [value] for name, value in FeatureTablesData().__dict__.items()
        },
    )
    monkeypatch.setattr(
        "sae_dashboard.neuronpedia.legacy.sae_vis_runner.get_features_table_data",
        lambda **kwargs: {
            name: [value] for name, value in FeatureTablesData().__dict__.items()
        },
    )
    monkeypatch.setattr(
        "sae_dashboard.sae_vis_runner.get_logits_table_data",
        lambda **kwargs: LogitsTableData(),
    )
    monkeypatch.setattr(
        "sae_dashboard.neuronpedia.legacy.sae_vis_runner.get_logits_table_data",
        lambda **kwargs: LogitsTableData(),
    )

    cfg = SaeVisConfig(
        hook_point="blocks.0.hook_resid_pre",
        features=[0],
        minibatch_size_features=1,
        minibatch_size_tokens=1,
        quantile_feature_batch_size=1,
        device="cpu",
        dtype="float32",
        ignore_tokens={0},
        log_performance=True,
        cleanup_each_minibatch=True,
        sequence_replay_artifact_dir=tmp_path / "sequence_replay_artifacts",
    )
    tokens = torch.tensor([[7, 0]], dtype=torch.long)

    sae_vis_data = SaeVisRunner(cfg).run(
        encoder=cast(SAE[Any], _FakeEncoder()),
        model=cast(
            HookedSAETransformer,
            SimpleNamespace(W_U=torch.tensor([[0.002, -0.002]], dtype=torch.float32)),
        ),
        tokens=tokens,
    )
    assert not isinstance(sae_vis_data, SaeVisColumnarData)

    stage_names = {
        str(event["stage"])
        for event in perf_events
        if event.get("event") == "stage_timing"
    }
    assert {
        "activation_and_encode_total",
        "logits_projection",
        "feature_statistics_packaging",
        "logits_histogram_packaging",
        "activation_histogram_packaging",
        "feature_table_packaging",
        "logits_table_packaging",
        "sequence_packaging",
    }.issubset(stage_names)

    packaging_summary = next(
        event
        for event in perf_events
        if event.get("event") == "packaging_shape_summary"
    )
    assert packaging_summary["valid_token_count"] == 1
    assert packaging_summary["token_shape"] == [1, 2]

    assert len(sae_vis_data.feature_data_dict[0].logits_histogram_data.tick_vals) > 1000

    artifact_event = next(
        event
        for event in perf_events
        if event.get("event") == "sequence_replay_artifact"
    )
    artifact_path = Path(str(artifact_event["path"]))
    assert artifact_path.is_file()

    artifact_payload = torch.load(artifact_path, weights_only=False)
    assert artifact_payload["feature_indices"] == [0]
    assert artifact_payload["token_shape"] == [1, 2]
    assert artifact_payload["valid_token_count"] == 1

    assert sae_vis_data.feature_stats.max == [1.0]
    assert sae_vis_data.feature_stats.frac_nonzero == [0.5]
    assert (
        sae_vis_data.feature_data_dict[0].acts_histogram_data.title
        == "ACTIVATIONS<br>DENSITY = 50.000%"
    )


@pytest.mark.parametrize("artifact_format", ["arrow", "parquet"])
def test_SaeVisRunner_columnar_output_writes_importer_compatible_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact_format: Literal["arrow", "parquet"],
) -> None:
    perf_events = _capture_perf_events(monkeypatch)
    monkeypatch.setattr(
        "sae_dashboard.sae_vis_runner.FeatureDataGeneratorFactory.create",
        lambda cfg, model, encoder, tokens: _FakeFeatureDataGenerator(),
    )
    monkeypatch.setattr(
        "sae_dashboard.sae_vis_runner.SequenceDataGenerator",
        _FakeSequenceDataGenerator,
    )
    monkeypatch.setattr(
        "sae_dashboard.sae_vis_runner.get_features_table_data",
        lambda **kwargs: {
            name: [value] for name, value in FeatureTablesData().__dict__.items()
        },
    )
    monkeypatch.setattr(
        "sae_dashboard.sae_vis_runner.get_logits_table_data",
        lambda **kwargs: LogitsTableData(),
    )

    cfg = SaeVisConfig(
        hook_point="blocks.0.hook_resid_pre",
        features=[0],
        minibatch_size_features=1,
        minibatch_size_tokens=1,
        quantile_feature_batch_size=1,
        device="cpu",
        dtype="float32",
        ignore_tokens={0},
        log_performance=True,
        dashboard_output_format="columnar",
        columnar_artifact_dir=tmp_path / "batch-0.columnar",
        columnar_artifact_format=artifact_format,
        columnar_emit_sequence_rows=True,
        columnar_emit_activation_rows=True,
        columnar_emit_activation_copy_rows=True,
        columnar_activation_copy_model_id="model-a",
        columnar_activation_copy_layer="9-source-a",
        columnar_activation_copy_creator_id="creator-a",
        columnar_activation_copy_created_at="2026-01-02T03:04:05",
        columnar_activation_copy_id_prefix="act",
        feature_statistics_backend="arrow",
        logits_histogram_backend="arrow",
        activation_histogram_backend="torch",
    )
    assert cfg.columnar_artifact_dir is not None
    tokens = torch.tensor([[7, 0]], dtype=torch.long)

    columnar_data = SaeVisRunner(cfg).run(
        encoder=cast(SAE[Any], _FakeEncoder()),
        model=cast(
            HookedSAETransformer,
            SimpleNamespace(
                W_U=torch.tensor([[1.0]], dtype=torch.float32),
                tokenizer=SimpleNamespace(
                    convert_ids_to_tokens=lambda token_ids: [
                        f"tok_{token_id}" for token_id in token_ids
                    ],
                    pad_token_id=0,
                ),
            ),
        ),
        tokens=tokens,
    )

    assert isinstance(columnar_data, SaeVisColumnarData)
    assert columnar_data.manifest_path.is_file()

    root_manifest = json.loads(columnar_data.manifest_path.read_text(encoding="utf-8"))
    assert root_manifest["dashboard_output_format"] == "columnar"
    assert root_manifest["columnar_artifact_format"] == artifact_format
    assert root_manifest["batches"] == [
        {
            "artifact_dir": "feature_batch_0",
            "feature_batch_index": 0,
            "feature_indices": [0],
        }
    ]

    batch_manifest_path = (
        cfg.columnar_artifact_dir / "feature_batch_0" / "manifest.json"
    )
    batch_manifest = json.loads(batch_manifest_path.read_text(encoding="utf-8"))
    expected_tables = {
        "feature_statistics",
        "feature_tables",
        "logits_histograms",
        "activation_histograms",
        "logits_tables",
        "sequence_rows",
        "activation_rows",
        "activation_copy_rows",
    }
    assert set(batch_manifest["tables"]) == expected_tables
    assert batch_manifest["row_counts"]["sequence_rows"] == 2
    assert batch_manifest["row_counts"]["activation_rows"] == 1
    assert batch_manifest["row_counts"]["activation_copy_rows"] == 1

    logits_histograms_path = (
        cfg.columnar_artifact_dir
        / "feature_batch_0"
        / batch_manifest["tables"]["logits_histograms"]
    )
    logits_histograms = _read_columnar_table(logits_histograms_path)
    assert "feature_index" in logits_histograms.column_names
    assert "row_index" not in logits_histograms.column_names
    assert logits_histograms.to_pylist()[0]["feature_index"] == 0

    activation_histograms_path = (
        cfg.columnar_artifact_dir
        / "feature_batch_0"
        / batch_manifest["tables"]["activation_histograms"]
    )
    activation_histograms = _read_columnar_table(activation_histograms_path)
    assert "feature_index" in activation_histograms.column_names
    assert "row_index" not in activation_histograms.column_names
    assert activation_histograms.to_pylist()[0]["feature_index"] == 0

    activation_copy_rows_path = (
        cfg.columnar_artifact_dir
        / "feature_batch_0"
        / batch_manifest["tables"]["activation_copy_rows"]
    )
    activation_copy_rows = _read_columnar_table(activation_copy_rows_path).to_pylist()
    assert activation_copy_rows == [
        {
            "id": "act-0-0",
            "tokens": ["tok_101", "tok_102"],
            "dataIndex": None,
            "index": 0,
            "layer": "9-source-a",
            "modelId": "model-a",
            "dataSource": None,
            "maxValue": 0.555,
            "maxValueTokenIndex": 1,
            "minValue": 0.123,
            "values": [0.123, 0.555],
            "dfaValues": [],
            "dfaTargetIndex": None,
            "dfaMaxValue": None,
            "creatorId": "creator-a",
            "createdAt": "2026-01-02T03:04:05",
            "lossValues": [],
            "logitContributions": None,
            "binMin": -1.0,
            "binMax": 1.0,
            "binContains": -1.0,
            "qualifyingTokenIndex": 0,
        }
    ]

    stage_names = {
        str(event["stage"])
        for event in perf_events
        if event.get("event") == "stage_timing"
    }
    assert f"sequence_row_{artifact_format}_stream_write" in stage_names
    assert "activation_copy_row_packaging" in stage_names


def test_SaeVisRunner_deferred_columnar_write_matches_immediate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _capture_perf_events(monkeypatch)
    monkeypatch.setattr(
        "sae_dashboard.sae_vis_runner.FeatureDataGeneratorFactory.create",
        lambda cfg, model, encoder, tokens: _FakeFeatureDataGenerator(),
    )
    monkeypatch.setattr(
        "sae_dashboard.sae_vis_runner.SequenceDataGenerator",
        _FakeSequenceDataGenerator,
    )
    monkeypatch.setattr(
        "sae_dashboard.sae_vis_runner.get_features_table_data",
        lambda **kwargs: {
            name: [value] for name, value in FeatureTablesData().__dict__.items()
        },
    )

    def _make_cfg(artifact_dir: Path, defer: bool) -> SaeVisConfig:
        return SaeVisConfig(
            hook_point="blocks.0.hook_resid_pre",
            features=[0],
            minibatch_size_features=1,
            minibatch_size_tokens=1,
            quantile_feature_batch_size=1,
            device="cpu",
            dtype="float32",
            ignore_tokens={0},
            log_performance=True,
            dashboard_output_format="columnar",
            columnar_defer_batch_write=defer,
            columnar_artifact_dir=artifact_dir,
            columnar_artifact_format="parquet",
            columnar_emit_sequence_rows=True,
            columnar_emit_activation_rows=True,
            columnar_emit_activation_copy_rows=True,
            columnar_activation_copy_model_id="model-a",
            columnar_activation_copy_layer="9-source-a",
            columnar_activation_copy_creator_id="creator-a",
            columnar_activation_copy_created_at="2026-01-02T03:04:05",
            columnar_activation_copy_id_prefix="act",
            feature_statistics_backend="arrow",
            logits_histogram_backend="arrow",
            activation_histogram_backend="torch",
        )

    def _run(cfg: SaeVisConfig) -> SaeVisColumnarData:
        return SaeVisRunner(cfg).run(  # pyright: ignore
            encoder=cast(SAE[Any], _FakeEncoder()),
            model=cast(
                HookedSAETransformer,
                SimpleNamespace(
                    W_U=torch.tensor([[1.0]], dtype=torch.float32),
                    tokenizer=SimpleNamespace(
                        convert_ids_to_tokens=lambda token_ids: [
                            f"tok_{token_id}" for token_id in token_ids
                        ],
                        pad_token_id=0,
                    ),
                ),
            ),
            tokens=torch.tensor([[7, 0]], dtype=torch.long),
        )

    immediate = _run(_make_cfg(tmp_path / "immediate.columnar", defer=False))
    assert immediate.pending_finalize is None
    assert immediate.manifest_path.is_file()

    deferred = _run(_make_cfg(tmp_path / "deferred.columnar", defer=True))
    assert deferred.pending_finalize is not None
    assert deferred.batches == []
    # nothing is on disk until finalize runs — the resume completeness marker
    # (the root manifest) must not exist yet
    assert not deferred.manifest_path.exists()

    finalized = deferred.pending_finalize()
    assert finalized.manifest_path.is_file()
    assert finalized.batches[0].row_counts == immediate.batches[0].row_counts

    immediate_manifest = json.loads(immediate.manifest_path.read_text(encoding="utf-8"))
    finalized_manifest = json.loads(finalized.manifest_path.read_text(encoding="utf-8"))
    assert finalized_manifest == immediate_manifest

    immediate_batch_manifest = json.loads(
        (immediate.artifact_dir / "feature_batch_0" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    finalized_batch_manifest = json.loads(
        (finalized.artifact_dir / "feature_batch_0" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert finalized_batch_manifest == immediate_batch_manifest


@pytest.mark.parametrize("write_page_index", [True, False])
def test_SaeVisRunner_columnar_parquet_honors_write_page_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    write_page_index: bool,
) -> None:
    """The page index must reach the parquet footer, for every table the runner emits.

    Asserted on the real writer path rather than on the config, because a page index cannot be added
    to an existing file: if the flag fails to reach `pq.write_table`, the only remedy is to
    regenerate the corpus. This also covers both writer functions -- `_write_columnar_table` for the
    single-table artifacts and `ParquetWriter` for the streamed `sequence_rows`.
    """
    _capture_perf_events(monkeypatch)
    monkeypatch.setattr(
        "sae_dashboard.sae_vis_runner.FeatureDataGeneratorFactory.create",
        lambda cfg, model, encoder, tokens: _FakeFeatureDataGenerator(),
    )
    monkeypatch.setattr(
        "sae_dashboard.sae_vis_runner.SequenceDataGenerator",
        _FakeSequenceDataGenerator,
    )
    monkeypatch.setattr(
        "sae_dashboard.sae_vis_runner.get_features_table_data",
        lambda **kwargs: {
            name: [value] for name, value in FeatureTablesData().__dict__.items()
        },
    )
    monkeypatch.setattr(
        "sae_dashboard.sae_vis_runner.get_logits_table_data",
        lambda **kwargs: LogitsTableData(),
    )

    cfg = SaeVisConfig(
        hook_point="blocks.0.hook_resid_pre",
        features=[0],
        minibatch_size_features=1,
        minibatch_size_tokens=1,
        quantile_feature_batch_size=1,
        device="cpu",
        dtype="float32",
        ignore_tokens={0},
        dashboard_output_format="columnar",
        columnar_artifact_dir=tmp_path / "batch-0.columnar",
        columnar_artifact_format="parquet",
        columnar_write_page_index=write_page_index,
        columnar_emit_sequence_rows=True,
        columnar_emit_activation_rows=True,
        columnar_emit_activation_copy_rows=True,
        columnar_activation_copy_model_id="model-a",
        columnar_activation_copy_layer="9-source-a",
        columnar_activation_copy_creator_id="creator-a",
        columnar_activation_copy_created_at="2026-01-02T03:04:05",
        columnar_activation_copy_id_prefix="act",
        feature_statistics_backend="arrow",
        logits_histogram_backend="arrow",
        activation_histogram_backend="torch",
    )
    assert cfg.columnar_artifact_dir is not None

    SaeVisRunner(cfg).run(
        encoder=cast(SAE[Any], _FakeEncoder()),
        model=cast(
            HookedSAETransformer,
            SimpleNamespace(
                W_U=torch.tensor([[1.0]], dtype=torch.float32),
                tokenizer=SimpleNamespace(
                    convert_ids_to_tokens=lambda token_ids: [
                        f"tok_{token_id}" for token_id in token_ids
                    ],
                    pad_token_id=0,
                ),
            ),
        ),
        tokens=torch.tensor([[7, 0]], dtype=torch.long),
    )

    pyarrow_parquet = importlib.import_module("pyarrow.parquet")
    written = sorted(cfg.columnar_artifact_dir.rglob("*.parquet"))
    assert written, "expected the columnar run to emit parquet artifacts"

    for table_path in written:
        column = pyarrow_parquet.ParquetFile(table_path).metadata.row_group(0).column(0)
        assert (
            column.has_offset_index is write_page_index
        ), f"{table_path.name} offset index = {column.has_offset_index}, expected {write_page_index}"


def test_SaeVisRunner_write_page_index_defaults_on() -> None:
    """Default-on is the point: a corpus generated without it can only be fixed by regenerating."""
    assert (
        SaeVisConfig(
            hook_point="blocks.0.hook_resid_pre", features=[0]
        ).columnar_write_page_index
        is True
    )
    assert (
        NeuronpediaRunnerConfig(
            sae_set="s", sae_path="p", outputs_dir="o"
        ).columnar_write_page_index
        is True
    )


@pytest.mark.parametrize("row_group_size", [None, 2, 4096])
def test_SaeVisRunner_columnar_parquet_honors_row_group_size(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    row_group_size: int | None,
) -> None:
    """Row group size must reach the parquet footer, on both writer paths.

    Asserted on the written file rather than on the config for the same reason as the page index:
    it is fixed at write time. It matters more, though -- readers prune at ROW GROUP granularity, so
    a single-row-group file costs a reader the whole file to fetch one feature and no page index
    changes that.

    `row_group_size=2` against the fixture's small tables is what makes the difference observable:
    it forces multiple row groups where the pyarrow default would write exactly one.
    """
    _capture_perf_events(monkeypatch)
    monkeypatch.setattr(
        "sae_dashboard.sae_vis_runner.FeatureDataGeneratorFactory.create",
        lambda cfg, model, encoder, tokens: _FakeFeatureDataGenerator(),
    )
    monkeypatch.setattr(
        "sae_dashboard.sae_vis_runner.SequenceDataGenerator",
        _FakeSequenceDataGenerator,
    )
    monkeypatch.setattr(
        "sae_dashboard.sae_vis_runner.get_features_table_data",
        lambda **kwargs: {
            name: [value] for name, value in FeatureTablesData().__dict__.items()
        },
    )
    monkeypatch.setattr(
        "sae_dashboard.sae_vis_runner.get_logits_table_data",
        lambda **kwargs: LogitsTableData(),
    )

    cfg = SaeVisConfig(
        hook_point="blocks.0.hook_resid_pre",
        features=[0],
        minibatch_size_features=1,
        minibatch_size_tokens=1,
        quantile_feature_batch_size=1,
        device="cpu",
        dtype="float32",
        ignore_tokens={0},
        dashboard_output_format="columnar",
        columnar_artifact_dir=tmp_path / "batch-0.columnar",
        columnar_artifact_format="parquet",
        columnar_parquet_row_group_size=row_group_size,
        columnar_emit_sequence_rows=True,
        columnar_emit_activation_rows=True,
        columnar_emit_activation_copy_rows=True,
        columnar_activation_copy_model_id="model-a",
        columnar_activation_copy_layer="9-source-a",
        columnar_activation_copy_creator_id="creator-a",
        columnar_activation_copy_created_at="2026-01-02T03:04:05",
        columnar_activation_copy_id_prefix="act",
        feature_statistics_backend="arrow",
        logits_histogram_backend="arrow",
        activation_histogram_backend="torch",
    )
    assert cfg.columnar_artifact_dir is not None

    SaeVisRunner(cfg).run(
        encoder=cast(SAE[Any], _FakeEncoder()),
        model=cast(
            HookedSAETransformer,
            SimpleNamespace(
                W_U=torch.tensor([[1.0]], dtype=torch.float32),
                tokenizer=SimpleNamespace(
                    convert_ids_to_tokens=lambda token_ids: [
                        f"tok_{token_id}" for token_id in token_ids
                    ],
                    pad_token_id=0,
                ),
            ),
        ),
        tokens=torch.tensor([[7, 0]], dtype=torch.long),
    )

    pyarrow_parquet = importlib.import_module("pyarrow.parquet")
    written = sorted(cfg.columnar_artifact_dir.rglob("*.parquet"))
    assert written, "expected the columnar run to emit parquet artifacts"

    for table_path in written:
        metadata = pyarrow_parquet.ParquetFile(table_path).metadata
        if row_group_size is None:
            assert (
                metadata.num_row_groups == 1
            ), f"{table_path.name} should keep the pyarrow default of one row group"
            continue
        largest = max(
            metadata.row_group(i).num_rows for i in range(metadata.num_row_groups)
        )
        assert largest <= row_group_size, (
            f"{table_path.name} has a {largest}-row group, above the requested {row_group_size}"
        )


def test_SaeVisRunner_row_group_size_defaults_to_a_streamable_value() -> None:
    """Default-on for the same reason as the page index: unfixable without regenerating."""
    assert DEFAULT_PARQUET_ROW_GROUP_SIZE == 4096
    assert (
        SaeVisConfig(
            hook_point="blocks.0.hook_resid_pre", features=[0]
        ).columnar_parquet_row_group_size
        == DEFAULT_PARQUET_ROW_GROUP_SIZE
    )
    assert (
        NeuronpediaRunnerConfig(
            sae_set="s", sae_path="p", outputs_dir="o"
        ).columnar_parquet_row_group_size
        == DEFAULT_PARQUET_ROW_GROUP_SIZE
    )


def test_row_group_size_is_split_out_of_the_ParquetWriter_constructor_kwargs() -> None:
    """`row_group_size` is a per-write argument; passing it to ParquetWriter(...) raises.

    Regression guard: the streamed-batch writer path constructs the writer once and writes many
    batches, so the two kwarg sets genuinely differ and cannot share one dict.
    """
    runner = SaeVisRunner(
        SaeVisConfig(
            hook_point="blocks.0.hook_resid_pre",
            features=[0],
            columnar_artifact_format="parquet",
            columnar_parquet_row_group_size=4096,
        )
    )
    assert runner._parquet_writer_kwargs() == {
        "write_page_index": True,
        "row_group_size": 4096,
    }
    assert runner._parquet_open_kwargs() == {"write_page_index": True}
    assert runner._parquet_write_kwargs() == {"row_group_size": 4096}

    disabled = SaeVisRunner(
        SaeVisConfig(
            hook_point="blocks.0.hook_resid_pre",
            features=[0],
            columnar_artifact_format="parquet",
            columnar_parquet_row_group_size=None,
        )
    )
    assert "row_group_size" not in disabled._parquet_writer_kwargs()
    assert disabled._parquet_write_kwargs() == {}
