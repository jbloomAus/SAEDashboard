# pyright: basic, reportPrivateImportUsage=false
# pyright: reportMissingTypeStubs=false
"""Parity tests for the chunked-device feature-statistics / activation-histogram
builders (Phase 6.2 slice 2) and the memoized activation-row detokenization."""

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import torch

from sae_dashboard.sae_vis_runner import SaeVisRunner
from sae_dashboard.utils_fns import FeatureStatistics, HistogramData
from tests.helpers import build_sae_vis_cfg

if TYPE_CHECKING:
    from sae_dashboard.sequence_data_generator import SequenceCoordinateTable


def _stats_fixture(
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(1234)
    flat_data = (torch.randn(9, 40) * torch.linspace(0.2, 3.0, 9).unsqueeze(-1)).to(
        dtype
    )
    flat_data[3] = 0.0  # all-zero row
    flat_data[4] = -flat_data[4].abs()  # all-negative row
    flat_data[5] = 0.75  # constant-positive row
    valid_mask = torch.ones(40, dtype=torch.bool)
    valid_mask[::7] = False
    return flat_data, valid_mask


def _reference_scalar_table(flat_data: torch.Tensor, valid_mask: torch.Tensor) -> Any:
    feature_stats_input = flat_data[:, valid_mask]
    if feature_stats_input.shape[-1] == 0:
        feature_stats_input = torch.zeros(
            (flat_data.shape[0], 1),
            dtype=flat_data.dtype,
            device=flat_data.device,
        )
    return FeatureStatistics.create_arrow_table(
        data=feature_stats_input,
        include_quantiles=False,
    )


def _reference_histogram_table(
    flat_data: torch.Tensor,
    valid_mask: torch.Tensor,
    titles: list[str],
    n_bins: int = 10,
) -> Any:
    masked = flat_data * valid_mask.to(device=flat_data.device, dtype=flat_data.dtype)
    return HistogramData.from_data_batch_arrow_table(
        data=masked,
        n_bins=n_bins,
        tickmode="5 ticks",
        title=None,
        positive_only=True,
        titles=titles,
        backend="torch",
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_scalar_stats_from_flat_valid_matches_gathered(dtype: torch.dtype) -> None:
    flat_data, valid_mask = _stats_fixture(dtype)
    valid_indices = torch.nonzero(valid_mask, as_tuple=False).flatten()

    chunked = FeatureStatistics.create_scalar_arrow_table_from_flat_valid(
        flat_data,
        valid_indices,
        row_chunk_size=4,
    )
    reference = _reference_scalar_table(flat_data, valid_mask)

    assert chunked.schema.names == reference.schema.names
    assert chunked.to_pydict() == reference.to_pydict()


def test_scalar_stats_from_flat_valid_empty_valid_matches_zero_substitution() -> None:
    flat_data, _ = _stats_fixture()
    empty_indices = torch.zeros(0, dtype=torch.long)

    chunked = FeatureStatistics.create_scalar_arrow_table_from_flat_valid(
        flat_data,
        empty_indices,
        row_chunk_size=4,
    )
    reference = _reference_scalar_table(
        flat_data, torch.zeros(flat_data.shape[-1], dtype=torch.bool)
    )

    assert chunked.to_pydict() == reference.to_pydict()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_scalar_stats_from_flat_valid_cuda_matches_cpu_reference() -> None:
    flat_data, valid_mask = _stats_fixture()
    valid_indices = torch.nonzero(valid_mask, as_tuple=False).flatten()

    chunked = FeatureStatistics.create_scalar_arrow_table_from_flat_valid(
        flat_data,
        valid_indices,
        compute_device="cuda",
        row_chunk_size=4,
    )
    reference = _reference_scalar_table(flat_data, valid_mask)

    assert chunked.to_pydict() == reference.to_pydict()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_histograms_from_flat_valid_match_masked_reference(dtype: torch.dtype) -> None:
    flat_data, valid_mask = _stats_fixture(dtype)
    valid_indices = torch.nonzero(valid_mask, as_tuple=False).flatten()
    titles = [f"row {index}" for index in range(flat_data.shape[0])]

    chunked = HistogramData.from_flat_valid_data_batch_arrow_table(
        flat_data,
        valid_indices,
        n_bins=10,
        tickmode="5 ticks",
        titles=titles,
        row_chunk_size=2,
        row_batch_size=3,
    )
    reference = _reference_histogram_table(flat_data, valid_mask, titles)

    assert chunked.schema.names == reference.schema.names
    assert chunked.to_pydict() == reference.to_pydict()


def test_histograms_from_flat_valid_empty_valid_produces_default_rows() -> None:
    flat_data, _ = _stats_fixture()
    empty_indices = torch.zeros(0, dtype=torch.long)
    titles = [f"row {index}" for index in range(flat_data.shape[0])]

    chunked = HistogramData.from_flat_valid_data_batch_arrow_table(
        flat_data,
        empty_indices,
        n_bins=10,
        tickmode="5 ticks",
        titles=titles,
        row_chunk_size=4,
    )
    reference = _reference_histogram_table(
        flat_data, torch.zeros(flat_data.shape[-1], dtype=torch.bool), titles
    )

    assert chunked.to_pydict() == reference.to_pydict()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_histograms_from_flat_valid_cuda_matches_cpu_reference() -> None:
    flat_data, valid_mask = _stats_fixture()
    valid_indices = torch.nonzero(valid_mask, as_tuple=False).flatten()
    titles = [f"row {index}" for index in range(flat_data.shape[0])]

    chunked = HistogramData.from_flat_valid_data_batch_arrow_table(
        flat_data,
        valid_indices,
        n_bins=10,
        tickmode="5 ticks",
        titles=titles,
        compute_device="cuda",
        row_chunk_size=2,
    ).to_pydict()
    reference = _reference_histogram_table(flat_data, valid_mask, titles).to_pydict()

    # Bin contents, edges semantics, ticks, and titles must match exactly; the
    # display-only bin midpoints (`bar_values`) may differ in the last rounded
    # decimal across devices (float32 mul-add rounding at the 5-decimal boundary).
    for column in ("row_index", "bar_heights", "tick_vals", "title"):
        assert chunked[column] == reference[column], column
    for row_chunked, row_reference in zip(
        chunked["bar_values"], reference["bar_values"]
    ):
        assert len(row_chunked) == len(row_reference)
        for value_chunked, value_reference in zip(row_chunked, row_reference):
            # one 5-decimal display-rounding step (plus float representation slack)
            assert abs(value_chunked - value_reference) <= 1.05e-5


class _CountingTokenizer:
    def __init__(self) -> None:
        self.ids_decoded = 0

    def convert_ids_to_tokens(self, token_ids: list[int]) -> list[str]:
        self.ids_decoded += len(token_ids)
        return [f"tok{token_id}" for token_id in token_ids]


class _FakeModel:
    def __init__(self) -> None:
        self.tokenizer = _CountingTokenizer()


def test_decode_token_ids_cached_matches_direct_and_dedupes() -> None:
    runner = SaeVisRunner(build_sae_vis_cfg())
    model: Any = _FakeModel()

    first = runner._decode_token_ids_cached(model, [5, 3, 5, 9])
    assert first == ["tok5", "tok3", "tok5", "tok9"]
    assert model.tokenizer.ids_decoded == 3  # 5, 3, 9 decoded once

    second = runner._decode_token_ids_cached(model, [9, 9, 5, 7])
    assert second == ["tok9", "tok9", "tok5", "tok7"]
    assert model.tokenizer.ids_decoded == 4  # only 7 was new

    assert runner._decode_token_ids_cached(model, []) == []
    assert model.tokenizer.ids_decoded == 4


def _coordinate_table_fixture(pad_token_id: int = 0) -> "SequenceCoordinateTable":
    from sae_dashboard.sequence_data_generator import SequenceCoordinateTable

    token_ids = torch.tensor(
        [
            [11, 12, 13, 14],
            [21, 22, pad_token_id, pad_token_id],  # trailing pads trimmed
            [pad_token_id, pad_token_id, pad_token_id, pad_token_id],  # all-pad row
            [31, pad_token_id, 33, pad_token_id],  # interior pad retained
        ],
        dtype=torch.long,
    )
    feat_acts = np.array(
        [
            [0.12345, 0.99999, 0.5, 0.25],
            [1.5, 1.5, 0.0, 0.0],  # tie on the rounded max -> first index wins
            [0.0, 0.0, 0.0, 0.0],
            [-0.75, 0.0, 2.0004, 0.1],
        ],
        dtype=np.float64,
    )
    return SequenceCoordinateTable(
        group_names=[
            "TOP ACTIVATIONS<br>MAX = 2.000",
            "INTERVAL 0.500 - 1.000<br>CONTAINS 3.100%",
        ],
        group_sizes=[2, 2],
        original_indices=torch.tensor([0, 1, 2, 3], dtype=torch.long),
        qualifying_token_indices=torch.tensor([1, 2, 1, 3], dtype=torch.long),
        source_token_indices=torch.arange(16, dtype=torch.long).reshape(4, 4),
        token_ids=token_ids,
        feat_acts=feat_acts,
        token_logits=torch.zeros(4, 4, dtype=torch.float32),
    )


@pytest.mark.parametrize("pad_token_id", [0, None])
def test_activation_row_record_batch_vectorized_matches_reference(
    pad_token_id: int | None,
) -> None:
    table = _coordinate_table_fixture()

    def decode(token_ids: list[int]) -> list[str]:
        return [f"tok{token_id}" for token_id in token_ids]

    vectorized = table.to_activation_row_arrow_record_batch(
        7, decode, pad_token_id=pad_token_id
    )
    reference = table.activation_row_arrow_record_batch_from_columns(
        table.to_activation_row_columns(7, decode, pad_token_id=pad_token_id)
    )

    assert vectorized.schema.equals(reference.schema)
    assert vectorized.to_pydict() == reference.to_pydict()


def test_get_logits_table_data_batch_matches_per_feature() -> None:
    from sae_dashboard.data_parsing_fns import (
        get_logits_table_data,
        get_logits_table_data_batch,
    )

    torch.manual_seed(77)
    # tie-free logits so top-k ordering is deterministic across 1D and batched paths
    logits = torch.argsort(torch.rand(5, 64), dim=-1).to(torch.float32)
    logits += torch.rand(5, 64) * 0.4

    batched = get_logits_table_data_batch(logits, n_rows=6)
    for row_index in range(logits.shape[0]):
        reference = get_logits_table_data(logits[row_index], n_rows=6)
        assert batched[row_index].top_logits == pytest.approx(reference.top_logits)
        assert batched[row_index].top_token_ids == list(reference.top_token_ids)
        assert batched[row_index].bottom_logits == pytest.approx(
            reference.bottom_logits
        )
        assert batched[row_index].bottom_token_ids == list(reference.bottom_token_ids)


def test_batched_selection_with_staged_flat_acts_matches_unstaged() -> None:
    import random

    from tests.unit.test_sequence_data_generator import (
        _assert_selections_equal,
        _batched_selection_fixture_generator,
    )

    generator = _batched_selection_fixture_generator()
    torch.manual_seed(2026)
    all_feat_acts = (torch.rand(3, 8, 5, dtype=torch.float32) + 0.01) * torch.linspace(
        0.5, 2.0, 5
    )
    selection_mask = torch.ones(3, 8, dtype=torch.bool)
    selection_mask[:, -1] = False
    masked_acts = all_feat_acts * selection_mask.unsqueeze(-1)
    # staged tensor carries the UNMASKED values (identity at candidate positions)
    staged = all_feat_acts.reshape(-1, 5).T.contiguous()
    if torch.cuda.is_available():
        staged = staged.to("cuda")

    random.seed(555)
    unstaged = generator.get_indices_dicts_columnar_gpu_batched(
        generator.buffer,
        masked_acts,
        selection_mask=selection_mask,
        feature_chunk_size=2,
    )
    random.seed(555)
    staged_result = generator.get_indices_dicts_columnar_gpu_batched(
        generator.buffer,
        masked_acts,
        selection_mask=selection_mask,
        feature_chunk_size=2,
        staged_flat_acts=staged,
    )

    _assert_selections_equal(staged_result, unstaged)


def test_build_sequence_coordinate_tables_batched_matches_per_feature() -> None:
    import random

    from tests.unit.test_sequence_data_generator import (
        _batched_selection_fixture_generator,
    )

    generator = _batched_selection_fixture_generator()
    assert generator.supports_batched_coordinate_tables()
    torch.manual_seed(404)
    n_features = 4
    all_feat_acts = (
        torch.rand(3, 8, n_features, dtype=torch.float32) + 0.01
    ) * torch.linspace(0.5, 2.0, n_features)
    selection_mask = torch.ones(3, 8, dtype=torch.bool)
    selection_mask[:, 0] = False
    masked_acts = all_feat_acts * selection_mask.unsqueeze(-1)
    feat_logits_batch = torch.randn(n_features, 32, dtype=torch.float32)

    random.seed(808)
    selections = generator.get_indices_dicts_columnar_gpu_batched(
        generator.buffer,
        masked_acts,
        selection_mask=selection_mask,
    )
    batched_tables = generator.build_sequence_coordinate_tables_batched(
        masked_acts,
        feat_logits_batch,
        selections,
    )

    assert len(batched_tables) == n_features
    for feature_index in range(n_features):
        reference = generator.get_sequence_coordinate_table(
            feat_acts=masked_acts[..., feature_index],
            feat_logits=feat_logits_batch[feature_index],
            resid_post=torch.empty(0),
            feature_resid_dir=torch.empty(0),
            selection_mask=selection_mask,
            selection_backend="columnar_gpu",
            precomputed_selection=selections[feature_index],
        )
        batched = batched_tables[feature_index]
        assert batched.group_names == reference.group_names
        assert batched.group_sizes == reference.group_sizes
        assert torch.equal(batched.original_indices, reference.original_indices)
        assert torch.equal(
            batched.qualifying_token_indices, reference.qualifying_token_indices
        )
        assert torch.equal(
            batched.source_token_indices.contiguous(),
            reference.source_token_indices,
        )
        assert torch.equal(batched.token_ids, reference.token_ids)
        assert np.array_equal(batched.feat_acts, reference.feat_acts)
        assert torch.equal(batched.token_logits, reference.token_logits)
        assert (
            batched.to_sequence_multi_group_data()
            == reference.to_sequence_multi_group_data()
        )


def test_build_sequence_coordinate_tables_batched_rejects_buffered_config() -> None:
    from tests.unit.test_sequence_data_generator import (
        _batched_selection_fixture_generator,
    )

    generator = _batched_selection_fixture_generator()
    generator.seq_cfg.buffer = (2, 2)  # type: ignore[assignment]
    assert not generator.supports_batched_coordinate_tables()
    with pytest.raises(ValueError, match="buffer=None"):
        generator.build_sequence_coordinate_tables_batched(
            torch.zeros(2, 6, 1), torch.zeros(1, 8), [({}, torch.zeros(0, 2), 0)]
        )


def test_activation_row_record_batches_for_features_match_per_feature() -> None:
    import dataclasses

    from sae_dashboard.sequence_data_generator import SequenceCoordinateTable

    tables: list[tuple[int, Any]] = []
    for feature_index in (3, 9, 27):
        base = _coordinate_table_fixture()
        # vary the values per feature so rows are distinct
        tables.append(
            (
                feature_index,
                dataclasses.replace(
                    base, feat_acts=base.feat_acts * (0.5 + 0.25 * feature_index)
                ),
            )
        )
    # add an empty-row table (feature with no selected sequences)
    empty = dataclasses.replace(
        _coordinate_table_fixture(),
        group_names=["TOP ACTIVATIONS<br>MAX = 0.000"],
        group_sizes=[0],
        original_indices=torch.zeros(0, dtype=torch.long),
        qualifying_token_indices=torch.zeros(0, dtype=torch.long),
        source_token_indices=torch.zeros((0, 4), dtype=torch.long),
        token_ids=torch.zeros((0, 4), dtype=torch.long),
        feat_acts=np.zeros((0, 4), dtype=np.float64),
        token_logits=torch.zeros((0, 4), dtype=torch.float32),
    )
    tables.insert(1, (5, empty))

    def decode(token_ids: list[int]) -> list[str]:
        return [f"tok{token_id}" for token_id in token_ids]

    for pad_token_id in (0, None):
        batched = (
            SequenceCoordinateTable.activation_row_arrow_record_batches_for_features(
                tables, decode, pad_token_id=pad_token_id
            )
        )
        assert len(batched) == len(tables)
        for (feature_index, table), batch_slice in zip(tables, batched):
            reference = table.activation_row_arrow_record_batch_from_columns(
                table.to_activation_row_columns(
                    feature_index, decode, pad_token_id=pad_token_id
                )
            )
            assert batch_slice.schema.equals(reference.schema)
            assert batch_slice.to_pydict() == reference.to_pydict()


def test_build_sequence_coordinate_tables_batched_unmasked_with_mask_matches_masked() -> (
    None
):
    import random

    from tests.unit.test_sequence_data_generator import (
        _batched_selection_fixture_generator,
    )

    generator = _batched_selection_fixture_generator()
    torch.manual_seed(606)
    n_features = 3
    all_feat_acts = torch.randn(3, 8, n_features, dtype=torch.float32)
    selection_mask = torch.ones(3, 8, dtype=torch.bool)
    selection_mask[:, 0] = False
    selection_mask[1, -2:] = False
    masked_acts = all_feat_acts * selection_mask.unsqueeze(-1)
    feat_logits_batch = torch.randn(n_features, 32, dtype=torch.float32)

    random.seed(909)
    selections = generator.get_indices_dicts_columnar_gpu_batched(
        generator.buffer,
        masked_acts,
        selection_mask=selection_mask,
    )
    reference_tables = generator.build_sequence_coordinate_tables_batched(
        masked_acts,
        feat_logits_batch,
        selections,
    )
    # production path post-slice-6: unmasked acts + post-gather masking
    unmasked_tables = generator.build_sequence_coordinate_tables_batched(
        all_feat_acts,
        feat_logits_batch,
        selections,
        ignore_tokens_mask=selection_mask,
    )
    if torch.cuda.is_available():
        device_tables = generator.build_sequence_coordinate_tables_batched(
            all_feat_acts.to("cuda"),
            feat_logits_batch.to("cuda"),
            selections,
            ignore_tokens_mask=selection_mask,
        )
    else:
        device_tables = unmasked_tables

    for reference, unmasked, on_device in zip(
        reference_tables, unmasked_tables, device_tables
    ):
        for candidate in (unmasked, on_device):
            assert torch.equal(candidate.token_ids, reference.token_ids)
            assert np.array_equal(candidate.feat_acts, reference.feat_acts)
            assert torch.equal(
                candidate.token_logits.cpu().float(),
                reference.token_logits.float(),
            )
            assert candidate.group_names == reference.group_names


def test_resolve_feature_acts_output_device_decision() -> None:
    from types import SimpleNamespace

    from sae_dashboard.feature_data_generator import FeatureDataGenerator

    generator = FeatureDataGenerator.__new__(FeatureDataGenerator)
    generator.full_sequence_length = 128

    generator.cfg = SimpleNamespace(  # pyright: ignore
        dashboard_output_format="json", device="cuda"
    )
    assert (
        generator._resolve_feature_acts_output_device(
            total_prompt_count=2490, feature_count=1024
        )
        == "cpu"
    )

    generator.cfg = SimpleNamespace(  # pyright: ignore
        dashboard_output_format="columnar", device="cpu"
    )
    assert (
        generator._resolve_feature_acts_output_device(
            total_prompt_count=2490, feature_count=1024
        )
        == "cpu"
    )

    generator.cfg = SimpleNamespace(  # pyright: ignore
        dashboard_output_format="columnar", device="cuda"
    )
    over_budget = generator._resolve_feature_acts_output_device(
        total_prompt_count=2490, feature_count=1_000_000
    )
    assert over_budget == "cpu"
    if torch.cuda.is_available():
        assert (
            generator._resolve_feature_acts_output_device(
                total_prompt_count=2490, feature_count=1024
            )
            == "cuda"
        )


def test_decode_token_ids_array_cached_matches_list_decoder() -> None:
    runner = SaeVisRunner(build_sae_vis_cfg())
    model: Any = _FakeModel()

    flat_ids = np.array([5, 3, 5, 9, 262143, 3], dtype=np.int64)
    first = runner._decode_token_ids_array_cached(model, flat_ids)
    assert list(first) == ["tok5", "tok3", "tok5", "tok9", "tok262143", "tok3"]
    decoded_after_first = model.tokenizer.ids_decoded

    # steady state: all ids known, no further tokenizer calls
    second = runner._decode_token_ids_array_cached(model, flat_ids[::-1].copy())
    assert list(second) == list(first[::-1])
    assert model.tokenizer.ids_decoded == decoded_after_first

    assert (
        list(runner._decode_token_ids_array_cached(model, np.empty(0, dtype=np.int64)))
        == []
    )


def test_activation_row_batches_with_array_decoder_match_list_decoder() -> None:
    from sae_dashboard.sequence_data_generator import SequenceCoordinateTable

    tables = [(3, _coordinate_table_fixture()), (9, _coordinate_table_fixture())]
    cache: dict[int, str] = {}

    def decode(token_ids: list[int]) -> list[str]:
        for token_id in token_ids:
            cache.setdefault(token_id, f"tok{token_id}")
        return [cache[token_id] for token_id in token_ids]

    def decode_array(flat_ids: np.ndarray) -> np.ndarray:
        return np.array(decode([int(t) for t in flat_ids.tolist()]), dtype=object)

    reference = (
        SequenceCoordinateTable.activation_row_arrow_record_batches_for_features(
            tables, decode, pad_token_id=0
        )
    )
    via_array = (
        SequenceCoordinateTable.activation_row_arrow_record_batches_for_features(
            tables, decode, pad_token_id=0, decode_token_ids_array=decode_array
        )
    )
    for ref_batch, arr_batch in zip(reference, via_array):
        assert arr_batch.to_pydict() == ref_batch.to_pydict()


def test_get_logits_table_data_masked_token_ids_excluded() -> None:
    from sae_dashboard.data_parsing_fns import (
        get_logits_table_data,
        get_logits_table_data_batch,
    )

    # Masked ids hold both extremes so unmasked behavior would select them first.
    logits = torch.linspace(-1.0, 1.0, 32).repeat(3, 1)
    logits[:, 30] = 50.0
    logits[:, 31] = 60.0
    logits[:, 0] = -50.0
    logits[:, 1] = -60.0
    masked_token_ids = torch.tensor([0, 1, 30, 31], dtype=torch.long)

    unmasked = get_logits_table_data_batch(logits, n_rows=4)
    assert unmasked[0].top_token_ids[:2] == [31, 30]
    assert unmasked[0].bottom_token_ids[:2] == [1, 0]

    masked = get_logits_table_data_batch(
        logits, n_rows=4, masked_token_ids=masked_token_ids
    )
    for row in masked:
        assert not set(row.top_token_ids) & {0, 1, 30, 31}
        assert not set(row.bottom_token_ids) & {0, 1, 30, 31}

    reference = get_logits_table_data(
        logits[0], n_rows=4, masked_token_ids=masked_token_ids
    )
    assert masked[0].top_token_ids == list(reference.top_token_ids)
    assert masked[0].bottom_token_ids == list(reference.bottom_token_ids)
    assert masked[0].top_logits == pytest.approx(reference.top_logits)
    assert masked[0].bottom_logits == pytest.approx(reference.bottom_logits)


def test_resolve_feature_acts_output_device_budget_override() -> None:
    from types import SimpleNamespace

    from sae_dashboard.feature_data_generator import FeatureDataGenerator

    generator = FeatureDataGenerator.__new__(FeatureDataGenerator)
    generator.full_sequence_length = 128

    # A zero budget forces host staging even for tiny shapes.
    generator.cfg = SimpleNamespace(  # pyright: ignore
        dashboard_output_format="columnar",
        device="cuda",
        columnar_max_device_staged_acts_bytes=0,
    )
    assert (
        generator._resolve_feature_acts_output_device(
            total_prompt_count=8, feature_count=8
        )
        == "cpu"
    )

    # A raised budget admits shapes the fixed 4 GiB default would reject.
    generator.cfg = SimpleNamespace(  # pyright: ignore
        dashboard_output_format="columnar",
        device="cuda",
        columnar_max_device_staged_acts_bytes=64 * 1024**3,
    )
    if torch.cuda.is_available():
        assert (
            generator._resolve_feature_acts_output_device(
                total_prompt_count=8192, feature_count=4096
            )
            == "cuda"
        )


@pytest.mark.parametrize("row_chunk_size", [1, 3, 64])
def test_scalar_stats_row_chunk_size_override_matches_default(
    row_chunk_size: int,
) -> None:
    torch.manual_seed(11)
    flat_data = torch.randn(9, 40).clamp(min=-0.2)
    valid_indices = torch.arange(40)[torch.rand(40) > 0.25]

    default_table = FeatureStatistics.create_scalar_arrow_table_from_flat_valid(
        flat_data, valid_indices
    )
    chunked_table = FeatureStatistics.create_scalar_arrow_table_from_flat_valid(
        flat_data, valid_indices, row_chunk_size=row_chunk_size
    )
    assert default_table.equals(chunked_table)


@pytest.mark.parametrize("row_chunk_size", [1, 5, 256])
def test_histogram_row_chunk_size_override_matches_default(
    row_chunk_size: int,
) -> None:
    torch.manual_seed(13)
    flat_data = torch.randn(7, 50)
    valid_indices = torch.arange(50)[torch.rand(50) > 0.2]
    titles = [f"feat {i}" for i in range(7)]

    default_table = HistogramData.from_flat_valid_data_batch_arrow_table(
        flat_data, valid_indices, n_bins=10, tickmode="5 ticks", titles=titles
    )
    chunked_table = HistogramData.from_flat_valid_data_batch_arrow_table(
        flat_data,
        valid_indices,
        n_bins=10,
        tickmode="5 ticks",
        titles=titles,
        row_chunk_size=row_chunk_size,
    )
    assert default_table.equals(chunked_table)


@pytest.mark.parametrize("include_constant_row", [False, True])
def test_dense_arrow_histogram_bf16_matches_caller_float32_cast(
    include_constant_row: bool,
) -> None:
    """Raw reduced-precision input must be bit-identical to the historical
    caller-side full-tensor float32 cast (the dense lane now casts per row batch;
    a constant row exercises the non-dense fallback lane)."""
    torch.manual_seed(17)
    logits = torch.randn(6, 64, dtype=torch.bfloat16)
    if include_constant_row:
        logits[2] = 0.5

    raw_table = HistogramData.from_data_batch_arrow_table(
        data=logits,
        n_bins=12,
        tickmode="5 ticks",
        title=None,
        backend="torch",
    )
    cast_table = HistogramData.from_data_batch_arrow_table(
        data=logits.to(torch.float32),
        n_bins=12,
        tickmode="5 ticks",
        title=None,
        backend="torch",
    )
    assert raw_table.equals(cast_table)


@pytest.mark.parametrize("feature_chunk_size", [1, 2, 64])
def test_batched_selection_feature_chunk_size_matches_default(
    feature_chunk_size: int,
) -> None:
    """The columnar_row_chunk_size override reaches the batched selector as
    feature_chunk_size; any chunk size must select identically (chunking only
    batches the math; per-feature RNG consumption order is unchanged)."""
    import random

    from tests.unit.test_sequence_data_generator import (
        _assert_selections_equal,
        _batched_selection_fixture_generator,
    )

    generator = _batched_selection_fixture_generator()
    torch.manual_seed(2027)
    all_feat_acts = (torch.rand(3, 8, 5, dtype=torch.float32) + 0.01) * torch.linspace(
        0.5, 2.0, 5
    )
    selection_mask = torch.ones(3, 8, dtype=torch.bool)
    selection_mask[:, -1] = False
    masked_acts = all_feat_acts * selection_mask.unsqueeze(-1)

    random.seed(556)
    default_result = generator.get_indices_dicts_columnar_gpu_batched(
        generator.buffer,
        masked_acts,
        selection_mask=selection_mask,
    )
    random.seed(556)
    chunked_result = generator.get_indices_dicts_columnar_gpu_batched(
        generator.buffer,
        masked_acts,
        selection_mask=selection_mask,
        feature_chunk_size=feature_chunk_size,
    )

    _assert_selections_equal(chunked_result, default_result)
