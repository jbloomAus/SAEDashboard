# pyright: basic, reportPrivateImportUsage=false
# pyright: reportMissingTypeStubs=false

import random
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
from transformer_lens import (
    HookedTransformer,  # type: ignore[import-untyped]  # pyright: ignore[reportMissingTypeStubs]
)

from sae_dashboard.components_config import SequencesConfig
from sae_dashboard.neuronpedia.legacy.sequence_data_generator import (
    LegacySequenceDataGenerator,
)
from sae_dashboard.sae_vis_data import SaeVisConfig
from sae_dashboard.sequence_data_generator import (
    SequenceCoordinateTable,
    SequenceDataGenerator,
)
from sae_dashboard.utils_fns import k_largest_indices, random_range_indices
from tests.helpers import build_sae_vis_cfg


@pytest.fixture
def sequence_data_generator(
    model: HookedTransformer, tokens: torch.Tensor
) -> SequenceDataGenerator:
    cfg: SaeVisConfig = build_sae_vis_cfg()
    return SequenceDataGenerator(cfg, tokens, model.W_U)


def test_get_sequences_data_expected_duplicates(
    sequence_data_generator: SequenceDataGenerator,
    model: HookedTransformer,
    tokens: torch.Tensor,
) -> None:
    token_scores = torch.arange(tokens.numel(), dtype=torch.float32).reshape(
        tokens.shape
    )
    feat_acts = token_scores / max(1, token_scores.numel())
    feat_logits = torch.linspace(-1.0, 1.0, model.cfg.d_vocab, dtype=torch.float32)
    resid_post = token_scores.unsqueeze(-1).repeat(1, 1, model.cfg.d_model)
    feature_resid_dir = torch.linspace(0.1, 1.0, model.cfg.d_model, dtype=torch.float32)

    sequence_multi_group_data = sequence_data_generator.get_sequences_data(
        feat_acts, feat_logits, resid_post, feature_resid_dir
    )

    all_sequence_data = []
    group_sequence_pairs = []
    for i, group in enumerate(sequence_multi_group_data.seq_group_data):
        all_sequence_data.extend(group.seq_data)
        group_sequence_pairs.extend(
            [(i, sd.original_index, sd.qualifying_token_index) for sd in group.seq_data]
        )

    # Count occurrences of each (original_index, qualifying_token_index) pair
    from collections import Counter

    pair_counts = Counter(
        (sd.original_index, sd.qualifying_token_index) for sd in all_sequence_data
    )

    # Check for duplicates within the same group
    duplicates_in_same_group = False
    for i, group in enumerate(sequence_multi_group_data.seq_group_data):
        group_pairs = [
            (sd.original_index, sd.qualifying_token_index) for sd in group.seq_data
        ]
        group_pair_counts = Counter(group_pairs)
        if any(count > 1 for count in group_pair_counts.values()):
            duplicates_in_same_group = True

    # Assertions
    assert not duplicates_in_same_group, "Duplicates found within the same group"
    assert (
        len(sequence_multi_group_data.seq_group_data[0].seq_data) == 20
    ), "TOP ACTIVATIONS group should have 20 sequences"

    # Check that duplicates only occur between TOP ACTIVATIONS and one other group
    for pair, count in pair_counts.items():
        if count > 1:
            groups_with_pair = [
                i for i, oi, qti in group_sequence_pairs if (oi, qti) == pair
            ]
            assert (
                0 in groups_with_pair
            ), f"Duplicate {pair} not in TOP ACTIVATIONS group"
            assert (
                len(groups_with_pair) == 2
            ), f"Duplicate {pair} found in more than two groups: {groups_with_pair}"


def test_package_sequences_data_no_duplicates(
    sequence_data_generator: SequenceDataGenerator,
) -> None:
    token_ids = torch.randint(0, 1000, (10, 5))
    feat_acts_coloring = torch.randn(10, 5)
    feat_logits = torch.randn(1000)
    indices_dict = {
        "Group1": torch.tensor([[0, 1], [1, 2]]),
        "Group2": torch.tensor([[2, 3], [3, 4]]),
    }
    indices_bold = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4]])

    sequence_multi_group_data = sequence_data_generator.package_sequences_data(
        token_ids, feat_acts_coloring, feat_logits, indices_dict, indices_bold
    )

    all_sequence_data = []
    for (
        group
    ) in (
        sequence_multi_group_data.seq_group_data
    ):  # Changed from sequence_groups to seq_group_data
        all_sequence_data.extend(group.seq_data)  # Changed from sequences to seq_data

    assert len(all_sequence_data) == len(
        set((sd.original_index, sd.qualifying_token_index) for sd in all_sequence_data)
    )


def test_package_sequences_data_rounds_feature_acts_to_4_decimals(
    sequence_data_generator: SequenceDataGenerator,
) -> None:
    token_ids = torch.tensor([[1, 2, 3]])
    feat_acts_coloring = torch.tensor([[0.12344, 0.12345, -1.23456]])
    feat_logits = torch.randn(10)
    indices_dict = {"Group1": torch.tensor([[0, 1]])}
    indices_bold = torch.tensor([[0, 1]])

    sequence_multi_group_data = sequence_data_generator.package_sequences_data(
        token_ids, feat_acts_coloring, feat_logits, indices_dict, indices_bold
    )

    seq_data = sequence_multi_group_data.seq_group_data[0].seq_data[0]

    assert seq_data.feat_acts == [
        round(float(value), 4) for value in feat_acts_coloring[0].tolist()
    ]


def test_package_sequences_data_rounds_bfloat16_feature_acts(
    sequence_data_generator: SequenceDataGenerator,
) -> None:
    token_ids = torch.tensor([[1, 2, 3]])
    feat_acts_coloring = torch.tensor(
        [[0.12344, 0.12345, -1.23456]], dtype=torch.bfloat16
    )
    feat_logits = torch.randn(10)
    indices_dict = {"Group1": torch.tensor([[0, 1]])}
    indices_bold = torch.tensor([[0, 1]])

    sequence_multi_group_data = sequence_data_generator.package_sequences_data(
        token_ids, feat_acts_coloring, feat_logits, indices_dict, indices_bold
    )

    seq_data = sequence_multi_group_data.seq_group_data[0].seq_data[0]

    assert seq_data.feat_acts == [
        round(float(value), 4) for value in feat_acts_coloring[0].float().tolist()
    ]


def test_build_sequence_coordinate_table_round_trips_to_nested_sequence_data(
    sequence_data_generator: SequenceDataGenerator,
) -> None:
    token_ids = torch.tensor([[1, 2, 3], [4, 5, 6], [6, 5, 4]])
    feat_acts_coloring = torch.tensor(
        [
            [0.12344, 0.12345, -1.23456],
            [2.0, 0.0, -0.00004],
            [3.14159, -2.71828, 1.41421],
        ]
    )
    feat_logits = torch.linspace(-1.0, 1.0, 10)
    indices_dict = {
        "Group1": torch.tensor([[10, 1], [11, 2]]),
        "Group2": torch.tensor([[12, 0]]),
    }
    indices_bold = torch.tensor([[10, 1], [11, 2], [12, 0]])

    coordinate_table = sequence_data_generator.build_sequence_coordinate_table(
        token_ids=token_ids,
        feat_acts_coloring=feat_acts_coloring,
        feat_logits=feat_logits,
        indices_dict=indices_dict,
        indices_bold=indices_bold,
    )

    assert coordinate_table.group_names == ["Group1", "Group2"]
    assert coordinate_table.group_sizes == [2, 1]
    assert coordinate_table.group_offsets == [0, 2, 3]
    assert coordinate_table.original_indices.tolist() == [10, 11, 12]
    assert coordinate_table.qualifying_token_indices.tolist() == [1, 2, 0]
    assert coordinate_table.source_token_indices.tolist() == [
        [0, 1, 2],
        [0, 1, 2],
        [0, 1, 2],
    ]
    assert coordinate_table.token_ids.tolist() == token_ids.tolist()
    assert (
        coordinate_table.feat_acts.tolist()
        == np.around(
            feat_acts_coloring.numpy().astype(np.float64, copy=False), 4
        ).tolist()
    )
    assert coordinate_table.token_logits.tolist() == feat_logits[token_ids].tolist()

    sequence_multi_group_data = coordinate_table.to_sequence_multi_group_data()

    assert [group.title for group in sequence_multi_group_data.seq_group_data] == [
        "Group1",
        "Group2",
    ]
    assert [
        len(group.seq_data) for group in sequence_multi_group_data.seq_group_data
    ] == [
        2,
        1,
    ]
    first_sequence = sequence_multi_group_data.seq_group_data[0].seq_data[0]
    assert first_sequence.original_index == 10
    assert first_sequence.qualifying_token_index == 1
    assert first_sequence.token_ids == [1, 2, 3]
    assert first_sequence.feat_acts == [0.1234, 0.1235, -1.2346]
    assert first_sequence.token_logits == feat_logits[token_ids[0]].tolist()


def test_sequence_coordinate_table_writes_flat_arrow_and_parquet_rows(
    sequence_data_generator: SequenceDataGenerator,
    tmp_path: Path,
) -> None:
    pyarrow = pytest.importorskip("pyarrow")
    pyarrow_ipc = pytest.importorskip("pyarrow.ipc")
    pyarrow_parquet = pytest.importorskip("pyarrow.parquet")
    datasets = pytest.importorskip("datasets")

    token_ids = torch.tensor([[1, 2, 3], [4, 5, 6], [6, 5, 4]])
    feat_acts_coloring = torch.tensor(
        [
            [0.12344, 0.12345, -1.23456],
            [2.0, 0.0, -0.00004],
            [3.14159, -2.71828, 1.41421],
        ]
    )
    feat_logits = torch.linspace(-1.0, 1.0, 10)
    indices_dict = {
        "Group1": torch.tensor([[10, 1], [11, 2]]),
        "Group2": torch.tensor([[12, 0]]),
    }
    indices_bold = torch.tensor([[10, 1], [11, 2], [12, 0]])
    source_token_indices = torch.tensor([[7, 8, 9], [10, 11, 12], [20, 21, 22]])

    coordinate_table = sequence_data_generator.build_sequence_coordinate_table(
        token_ids=token_ids,
        feat_acts_coloring=feat_acts_coloring,
        feat_logits=feat_logits,
        indices_dict=indices_dict,
        indices_bold=indices_bold,
        source_token_indices=source_token_indices,
    )

    row_table = coordinate_table.to_sequence_row_arrow_table(feature_index=42)
    records = row_table.to_pylist()

    assert row_table.num_rows == 9
    assert row_table.schema.names == [
        "feature_index",
        "sequence_index",
        "group_index",
        "group_name",
        "group_sequence_index",
        "context_token_index",
        "original_index",
        "qualifying_token_index",
        "source_token_index",
        "token_id",
        "feat_act",
        "token_logit",
    ]
    assert records[0]["feature_index"] == 42
    assert records[0]["sequence_index"] == 0
    assert records[0]["group_index"] == 0
    assert records[0]["group_name"] == "Group1"
    assert records[0]["group_sequence_index"] == 0
    assert records[0]["context_token_index"] == 0
    assert records[0]["original_index"] == 10
    assert records[0]["qualifying_token_index"] == 1
    assert records[0]["source_token_index"] == 7
    assert records[0]["token_id"] == 1
    assert records[0]["feat_act"] == pytest.approx(0.1234)
    assert records[0]["token_logit"] == pytest.approx(float(feat_logits[1]))
    assert records[3]["group_sequence_index"] == 1
    assert records[-1]["group_name"] == "Group2"
    assert records[-1]["source_token_index"] == 22

    arrow_path = tmp_path / "sequence_rows.arrow"
    parquet_path = tmp_path / "sequence_rows.parquet"

    assert coordinate_table.write_sequence_row_arrow_ipc(
        arrow_path, feature_index=42
    ) == len(records)
    assert coordinate_table.write_sequence_row_parquet(
        parquet_path, feature_index=42
    ) == len(records)

    with pyarrow.memory_map(str(arrow_path), "r") as source:
        arrow_records = pyarrow_ipc.open_file(source).read_all().to_pylist()
    parquet_records = pyarrow_parquet.read_table(str(parquet_path)).to_pylist()
    parquet_dataset = datasets.Dataset.from_parquet(str(parquet_path))

    assert arrow_records == records
    assert parquet_records == records
    assert parquet_dataset.num_rows == len(records)
    assert parquet_dataset[0]["feature_index"] == 42


def test_activation_copy_row_record_batch_from_activation_row_record_batch_matches_columns(
    sequence_data_generator: SequenceDataGenerator,
) -> None:
    pytest.importorskip("pyarrow")

    token_ids = torch.tensor([[1, 2, 3], [4, 5, 6], [6, 5, 4]])
    feat_acts_coloring = torch.tensor(
        [
            [0.12344, 0.12345, -1.23456],
            [2.0, 0.0, -0.00004],
            [3.14159, -2.71828, 1.41421],
        ]
    )
    feat_logits = torch.linspace(-1.0, 1.0, 10)
    indices_dict = {
        "Group1": torch.tensor([[10, 1], [11, 2]]),
        "Group2": torch.tensor([[12, 0]]),
    }
    indices_bold = torch.tensor([[10, 1], [11, 2], [12, 0]])

    coordinate_table = sequence_data_generator.build_sequence_coordinate_table(
        token_ids=token_ids,
        feat_acts_coloring=feat_acts_coloring,
        feat_logits=feat_logits,
        indices_dict=indices_dict,
        indices_bold=indices_bold,
    )
    activation_columns = coordinate_table.to_activation_row_columns(
        feature_index=42,
        decode_token_ids=lambda token_ids: [
            f"tok-{token_id}" for token_id in token_ids
        ],
    )
    activation_record_batch = (
        SequenceCoordinateTable.activation_row_arrow_record_batch_from_columns(
            activation_columns
        )
    )

    converted_from_batch = SequenceCoordinateTable.activation_copy_row_arrow_record_batch_from_activation_row_record_batch(
        activation_record_batch,
        model_id="test-model",
        layer="9-test-source",
        creator_id="test-creator",
        created_at="2026-01-02T03:04:05",
        activation_id_prefix="test-activation",
    )
    converted_from_columns = SequenceCoordinateTable.activation_copy_row_arrow_record_batch_from_activation_columns(
        activation_columns,
        model_id="test-model",
        layer="9-test-source",
        creator_id="test-creator",
        created_at="2026-01-02T03:04:05",
        activation_id_prefix="test-activation",
    )

    assert converted_from_batch.schema == converted_from_columns.schema
    assert converted_from_batch.to_pydict() == converted_from_columns.to_pydict()


def test_get_sequence_coordinate_table_matches_get_sequences_data(
    sequence_data_generator: SequenceDataGenerator,
    model: HookedTransformer,
    tokens: torch.Tensor,
) -> None:
    token_scores = torch.arange(tokens.numel(), dtype=torch.float32).reshape(
        tokens.shape
    )
    feat_acts = token_scores / max(1, token_scores.numel())
    feat_logits = torch.linspace(-1.0, 1.0, model.cfg.d_vocab, dtype=torch.float32)
    resid_post = token_scores.unsqueeze(-1).repeat(1, 1, model.cfg.d_model)
    feature_resid_dir = torch.linspace(0.1, 1.0, model.cfg.d_model, dtype=torch.float32)

    coordinate_table = sequence_data_generator.get_sequence_coordinate_table(
        feat_acts=feat_acts,
        feat_logits=feat_logits,
        resid_post=resid_post,
        feature_resid_dir=feature_resid_dir,
    )
    nested_from_table = coordinate_table.to_sequence_multi_group_data()
    nested_direct = sequence_data_generator.get_sequences_data(
        feat_acts=feat_acts,
        feat_logits=feat_logits,
        resid_post=resid_post,
        feature_resid_dir=feature_resid_dir,
    )

    assert nested_from_table == nested_direct


def test_get_sequence_coordinate_table_columnar_gpu_selection_matches_legacy() -> None:
    cfg: SaeVisConfig = build_sae_vis_cfg()
    cfg.feature_centric_layout.seq_cfg.buffer = None  # type: ignore
    cfg.feature_centric_layout.seq_cfg.top_acts_group_size = 4  # type: ignore
    cfg.feature_centric_layout.seq_cfg.n_quantiles = 0  # type: ignore

    tokens = torch.tensor(
        [
            [11, 12, 13, 14, 0],
            [21, 22, 23, 0, 0],
        ],
        dtype=torch.long,
    )
    selection_mask = torch.tensor(
        [
            [True, True, True, True, False],
            [True, True, True, False, False],
        ],
        dtype=torch.bool,
    )
    generator = SequenceDataGenerator(cfg, tokens, torch.zeros(4, 32))
    feat_acts = torch.tensor(
        [
            [0.0, 0.8, 0.1, 1.2, 9.0],
            [0.0, 0.6, 1.0, 8.0, 7.0],
        ],
        dtype=torch.float32,
    )
    feat_logits = torch.linspace(-1.0, 1.0, 32)
    resid_post = torch.empty(0)
    feature_resid_dir = torch.empty(0)

    legacy_table = generator.get_sequence_coordinate_table(
        feat_acts=feat_acts,
        feat_logits=feat_logits,
        resid_post=resid_post,
        feature_resid_dir=feature_resid_dir,
        selection_mask=selection_mask,
        selection_backend="legacy",
    )
    lazy_table = generator.get_sequence_coordinate_table(
        feat_acts=feat_acts,
        feat_logits=feat_logits,
        resid_post=resid_post,
        feature_resid_dir=feature_resid_dir,
        selection_mask=selection_mask,
        selection_backend="columnar_gpu",
    )

    assert (
        lazy_table.to_sequence_multi_group_data()
        == legacy_table.to_sequence_multi_group_data()
    )


def test_get_sequences_data_selection_mask_excludes_ignored_padding_positions() -> None:
    cfg: SaeVisConfig = build_sae_vis_cfg()
    cfg.feature_centric_layout.seq_cfg.buffer = None  # type: ignore
    cfg.feature_centric_layout.seq_cfg.top_acts_group_size = 5  # type: ignore
    cfg.feature_centric_layout.seq_cfg.n_quantiles = 0  # type: ignore

    tokens = torch.tensor(
        [
            [11, 12, 13, 0, 0, 0],
            [21, 22, 0, 0, 0, 0],
        ],
        dtype=torch.long,
    )
    generator = SequenceDataGenerator(cfg, tokens, torch.randn(4, 32))
    feat_acts = torch.tensor(
        [
            [0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.5, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    feat_logits = torch.randn(32)
    resid_post = torch.zeros(tokens.shape[0], tokens.shape[1], 4)
    feature_resid_dir = torch.randn(4)
    selection_mask = torch.tensor(
        [
            [True, True, True, False, False, False],
            [True, True, False, False, False, False],
        ]
    )

    sequence_multi_group_data = generator.get_sequences_data(
        feat_acts=feat_acts,
        feat_logits=feat_logits,
        resid_post=resid_post,
        feature_resid_dir=feature_resid_dir,
        selection_mask=selection_mask,
    )

    top_group = sequence_multi_group_data.seq_group_data[0].seq_data
    assert len(top_group) == 5
    assert all(
        selection_mask[seq.original_index, seq.qualifying_token_index].item()
        for seq in top_group
    )


def test_get_indices_dict_columnar_gpu_matches_legacy_indices_with_selection_mask() -> (
    None
):
    cfg: SaeVisConfig = build_sae_vis_cfg()
    cfg.feature_centric_layout.seq_cfg.buffer = None  # type: ignore
    cfg.feature_centric_layout.seq_cfg.top_acts_group_size = 4  # type: ignore
    cfg.feature_centric_layout.seq_cfg.n_quantiles = 4  # type: ignore
    cfg.feature_centric_layout.seq_cfg.quantile_group_size = 16  # type: ignore

    tokens = torch.arange(18, dtype=torch.long).reshape(3, 6)
    generator = SequenceDataGenerator(cfg, tokens, torch.randn(4, 32))
    feat_acts = torch.tensor(
        [
            [0.0, 0.25, 0.5, 0.75, 1.0, 0.0],
            [0.0, 1.25, 1.5, 1.75, 2.0, 0.0],
            [0.0, 0.4, 0.8, 1.2, 1.6, 0.0],
        ],
        dtype=torch.float32,
    )
    selection_mask = torch.tensor(
        [
            [False, True, True, True, True, False],
            [False, True, True, True, True, False],
            [False, True, True, True, True, False],
        ]
    )

    random.seed(12345)
    legacy_indices_dict, legacy_indices_bold, legacy_n_bold = (
        generator.get_indices_dict_legacy(
            generator.buffer,
            feat_acts,
            selection_mask=selection_mask,
        )
    )
    random.seed(12345)
    lazy_indices_dict, lazy_indices_bold, lazy_n_bold = (
        generator.get_indices_dict_columnar_gpu(
            generator.buffer,
            feat_acts,
            selection_mask=selection_mask,
        )
    )

    assert list(lazy_indices_dict) == list(legacy_indices_dict)
    assert lazy_n_bold == legacy_n_bold
    assert torch.equal(lazy_indices_bold, legacy_indices_bold)


def test_get_indices_dict_legacy_unmasked_matches_baseline_logic() -> None:
    cfg: SaeVisConfig = build_sae_vis_cfg()
    cfg.feature_centric_layout.seq_cfg.buffer = (1, 1)  # type: ignore
    cfg.feature_centric_layout.seq_cfg.top_acts_group_size = 3  # type: ignore
    cfg.feature_centric_layout.seq_cfg.n_quantiles = 3  # type: ignore
    cfg.feature_centric_layout.seq_cfg.quantile_group_size = 4  # type: ignore

    tokens = torch.arange(24, dtype=torch.long).reshape(4, 6)
    generator = LegacySequenceDataGenerator(cfg, tokens, torch.randn(4, 32))
    feat_acts = torch.tensor(
        [
            [0.0, 0.1, 0.5, 1.0, 0.4, 0.0],
            [0.0, 0.2, 0.7, 0.3, 0.9, 0.0],
            [0.0, 0.6, 0.8, 0.4, 0.2, 0.0],
            [0.0, 0.05, 0.15, 0.25, 0.35, 0.0],
        ],
        dtype=torch.float32,
    )

    random.seed(12345)
    indices_dict, indices_bold, n_bold = generator.get_indices_dict_legacy(
        generator.buffer,
        feat_acts,
        selection_mask=None,
    )

    random.seed(12345)
    expected_indices = k_largest_indices(
        feat_acts,
        k=cfg.feature_centric_layout.seq_cfg.top_acts_group_size,  # type: ignore[arg-type]
        buffer=generator.buffer,
    ).cpu()
    expected_indices_dict = {
        f"TOP ACTIVATIONS<br>MAX = {feat_acts.max():.3f}": expected_indices
    }
    quantiles = torch.linspace(
        0,
        feat_acts.max().item(),
        cfg.feature_centric_layout.seq_cfg.n_quantiles + 1,  # type: ignore[operator]
        device=feat_acts.device,
    )
    for i in range(cfg.feature_centric_layout.seq_cfg.n_quantiles - 1, -1, -1):  # type: ignore[operator]
        lower, upper = quantiles[i : i + 2].tolist()
        pct = float(((feat_acts >= lower) & (feat_acts <= upper)).float().mean().item())
        expected_indices_dict[
            f"INTERVAL {lower:.3f} - {upper:.3f}<br>CONTAINS {pct:.3%}"
        ] = random_range_indices(
            feat_acts,
            k=cfg.feature_centric_layout.seq_cfg.quantile_group_size,  # type: ignore[arg-type]
            bounds=(lower, upper),
            buffer=generator.buffer,
        ).cpu()

    expected_indices_bold = torch.concat(list(expected_indices_dict.values())).cpu()

    assert list(indices_dict) == list(expected_indices_dict)
    for group_name, expected in expected_indices_dict.items():
        assert torch.equal(indices_dict[group_name], expected)
    assert torch.equal(indices_bold, expected_indices_bold)
    assert n_bold == int(expected_indices_bold.shape[0])


def test_legacy_packaging_stays_in_legacy_subclass() -> None:
    cfg: SaeVisConfig = build_sae_vis_cfg()

    tokens = torch.arange(6, dtype=torch.long).reshape(2, 3)
    generator = LegacySequenceDataGenerator(cfg, tokens, torch.randn(4, 32))

    token_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    feat_acts_coloring = torch.tensor(
        [[0.12344, 0.12345, -1.23456], [2.34561, -0.44444, 0.0]]
    )
    feat_logits = torch.linspace(-1.0, 1.0, 10)
    indices_dict = {
        "Group1": torch.tensor([[10, 1]]),
        "Group2": torch.tensor([[11, 2]]),
    }
    indices_bold = torch.tensor([[10, 1], [11, 2]])

    sequence_multi_group_data = generator.package_sequences_data(
        token_ids=token_ids,
        feat_acts_coloring=feat_acts_coloring,
        feat_logits=feat_logits,
        indices_dict=indices_dict,
        indices_bold=indices_bold,
    )

    assert [group.title for group in sequence_multi_group_data.seq_group_data] == [
        "Group1",
        "Group2",
    ]
    first_sequence = sequence_multi_group_data.seq_group_data[0].seq_data[0]
    second_sequence = sequence_multi_group_data.seq_group_data[1].seq_data[0]

    assert first_sequence.original_index == 10
    assert first_sequence.qualifying_token_index == 1
    assert first_sequence.token_ids == [1, 2, 3]
    assert first_sequence.feat_acts == [0.1234, 0.1235, -1.2346]
    assert first_sequence.token_logits == feat_logits[token_ids[0]].tolist()
    assert second_sequence.original_index == 11
    assert second_sequence.qualifying_token_index == 2
    assert second_sequence.token_ids == [4, 5, 6]
    assert second_sequence.feat_acts == [2.3456, -0.4444, 0.0]
    assert second_sequence.token_logits == feat_logits[token_ids[1]].tolist()


def test_get_indices_dict_legacy_matches_baseline_selector_without_selection_mask() -> (
    None
):
    cfg: SaeVisConfig = build_sae_vis_cfg()
    seq_cfg = cast(SequencesConfig, cfg.feature_centric_layout.seq_cfg)
    seq_cfg.buffer = None
    seq_cfg.top_acts_group_size = 4
    seq_cfg.n_quantiles = 4
    seq_cfg.quantile_group_size = 16

    tokens = torch.arange(18, dtype=torch.long).reshape(3, 6)
    generator = SequenceDataGenerator(cfg, tokens, torch.randn(4, 32))
    feat_acts = torch.tensor(
        [
            [0.0, 0.25, 0.5, 0.75, 1.0, 0.0],
            [0.0, 1.25, 1.5, 1.75, 2.0, 0.0],
            [0.0, 0.4, 0.8, 1.2, 1.6, 0.0],
        ],
        dtype=torch.float32,
    )

    random.seed(12345)
    legacy_indices_dict, legacy_indices_bold, legacy_n_bold = (
        generator.get_indices_dict_legacy(generator.buffer, feat_acts)
    )

    random.seed(12345)
    expected_indices_dict = {
        f"TOP ACTIVATIONS<br>MAX = {feat_acts.max():.3f}": k_largest_indices(
            feat_acts,
            k=seq_cfg.top_acts_group_size,
            buffer=generator.buffer,
        ).cpu()
    }
    quantiles = torch.linspace(
        0,
        feat_acts.max().item(),
        seq_cfg.n_quantiles + 1,
    )
    for i in range(seq_cfg.n_quantiles - 1, -1, -1):
        lower, upper = quantiles[i : i + 2].tolist()
        pct = ((feat_acts >= lower) & (feat_acts <= upper)).float().mean()
        expected_indices_dict[
            f"INTERVAL {lower:.3f} - {upper:.3f}<br>CONTAINS {pct:.3%}"
        ] = random_range_indices(
            feat_acts,
            k=seq_cfg.quantile_group_size,
            bounds=(lower, upper),
            buffer=generator.buffer,
        ).cpu()

    expected_indices_bold = torch.concat(list(expected_indices_dict.values())).cpu()

    assert list(legacy_indices_dict) == list(expected_indices_dict)
    assert legacy_n_bold == expected_indices_bold.shape[0]
    assert torch.equal(legacy_indices_bold, expected_indices_bold)
    for group_name, expected_indices in expected_indices_dict.items():
        assert torch.equal(legacy_indices_dict[group_name], expected_indices)


def test_get_indices_dict_legacy_matches_baseline_selector_with_buffer_without_selection_mask() -> (
    None
):
    cfg: SaeVisConfig = build_sae_vis_cfg()
    seq_cfg = cast(SequencesConfig, cfg.feature_centric_layout.seq_cfg)
    seq_cfg.buffer = (1, 1)
    seq_cfg.top_acts_group_size = 4
    seq_cfg.n_quantiles = 4
    seq_cfg.quantile_group_size = 16

    tokens = torch.arange(18, dtype=torch.long).reshape(3, 6)
    generator = SequenceDataGenerator(cfg, tokens, torch.randn(4, 32))
    feat_acts = torch.tensor(
        [
            [9.0, 0.25, 0.5, 0.75, 1.0, 8.0],
            [7.0, 1.25, 1.5, 1.75, 2.0, 6.0],
            [5.0, 0.4, 0.8, 1.2, 1.6, 4.0],
        ],
        dtype=torch.float32,
    )

    random.seed(12345)
    legacy_indices_dict, legacy_indices_bold, legacy_n_bold = (
        generator.get_indices_dict_legacy(generator.buffer, feat_acts)
    )

    random.seed(12345)
    expected_indices_dict = {
        f"TOP ACTIVATIONS<br>MAX = {feat_acts.max():.3f}": k_largest_indices(
            feat_acts,
            k=seq_cfg.top_acts_group_size,
            buffer=generator.buffer,
        ).cpu()
    }
    quantiles = torch.linspace(
        0,
        feat_acts.max().item(),
        seq_cfg.n_quantiles + 1,
    )
    for i in range(seq_cfg.n_quantiles - 1, -1, -1):
        lower, upper = quantiles[i : i + 2].tolist()
        pct = ((feat_acts >= lower) & (feat_acts <= upper)).float().mean()
        expected_indices_dict[
            f"INTERVAL {lower:.3f} - {upper:.3f}<br>CONTAINS {pct:.3%}"
        ] = random_range_indices(
            feat_acts,
            k=seq_cfg.quantile_group_size,
            bounds=(lower, upper),
            buffer=generator.buffer,
        ).cpu()

    expected_indices_bold = torch.concat(list(expected_indices_dict.values())).cpu()

    assert list(legacy_indices_dict) == list(expected_indices_dict)
    assert legacy_n_bold == expected_indices_bold.shape[0]
    assert torch.equal(legacy_indices_bold, expected_indices_bold)
    for group_name, expected_indices in expected_indices_dict.items():
        assert torch.equal(legacy_indices_dict[group_name], expected_indices)


def test_get_indices_dict_legacy_skips_candidate_mask_without_selection_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg: SaeVisConfig = build_sae_vis_cfg()
    seq_cfg = cast(SequencesConfig, cfg.feature_centric_layout.seq_cfg)
    seq_cfg.buffer = None
    seq_cfg.top_acts_group_size = 4
    seq_cfg.n_quantiles = 4
    seq_cfg.quantile_group_size = 16

    tokens = torch.arange(18, dtype=torch.long).reshape(3, 6)
    generator = SequenceDataGenerator(cfg, tokens, torch.randn(4, 32))
    feat_acts = torch.tensor(
        [
            [0.0, 0.25, 0.5, 0.75, 1.0, 0.0],
            [0.0, 1.25, 1.5, 1.75, 2.0, 0.0],
            [0.0, 0.4, 0.8, 1.2, 1.6, 0.0],
        ],
        dtype=torch.float32,
    )

    def _fail_candidate_mask(*args, **kwargs):
        del args, kwargs
        raise AssertionError(
            "candidate mask path should not run without a selection mask"
        )

    monkeypatch.setattr(
        generator, "_get_candidate_mask_and_indices", _fail_candidate_mask
    )

    generator.get_indices_dict_legacy(generator.buffer, feat_acts)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Legacy all-zero top-k tie order is only distinguishable on CUDA.",
)
def test_legacy_sequence_generator_preserves_cuda_zero_tie_order() -> None:
    cfg: SaeVisConfig = build_sae_vis_cfg()
    cfg.device = "cuda"
    cfg.feature_centric_layout.seq_cfg.buffer = None  # type: ignore
    cfg.feature_centric_layout.seq_cfg.top_acts_group_size = 20  # type: ignore
    cfg.feature_centric_layout.seq_cfg.n_quantiles = 0  # type: ignore

    tokens = torch.arange(128, dtype=torch.long).repeat(2488, 1)
    generator = LegacySequenceDataGenerator(cfg, tokens, torch.zeros(4, 128))
    sequence_data = generator.get_sequences_data(
        feat_acts=torch.zeros_like(tokens, dtype=torch.float32),
        feat_logits=torch.zeros(128, dtype=torch.float32),
        resid_post=torch.empty(0),
        feature_resid_dir=torch.empty(0),
    )

    top_group = sequence_data.seq_group_data[0].seq_data
    assert [sequence.qualifying_token_index - 1 for sequence in top_group] == [
        18,
        17,
        15,
        16,
        0,
        -1,
        1,
        2,
        10,
        9,
        7,
        8,
        12,
        11,
        13,
        14,
        6,
        5,
        3,
        4,
    ]


def test_get_indices_dict_columnar_gpu_matches_legacy_for_bfloat16_interval_boundaries() -> (
    None
):
    cfg: SaeVisConfig = build_sae_vis_cfg()
    cfg.feature_centric_layout.seq_cfg.buffer = None  # type: ignore
    cfg.feature_centric_layout.seq_cfg.top_acts_group_size = 1  # type: ignore
    cfg.feature_centric_layout.seq_cfg.n_quantiles = 5  # type: ignore
    cfg.feature_centric_layout.seq_cfg.quantile_group_size = 16  # type: ignore

    tokens = torch.arange(8, dtype=torch.long).reshape(1, 8)
    generator = SequenceDataGenerator(cfg, tokens, torch.randn(4, 32))
    feat_acts = torch.tensor(
        [[0.0, 44.25, 90.0, 121.5, 133.0, 177.0, 221.0, 0.0]],
        dtype=torch.bfloat16,
    )
    selection_mask = torch.tensor(
        [[True, True, True, True, True, True, True, False]],
        dtype=torch.bool,
    )

    random.seed(12345)
    legacy_indices_dict, legacy_indices_bold, legacy_n_bold = (
        generator.get_indices_dict_legacy(
            generator.buffer,
            feat_acts,
            selection_mask=selection_mask,
        )
    )
    random.seed(12345)
    lazy_indices_dict, lazy_indices_bold, lazy_n_bold = (
        generator.get_indices_dict_columnar_gpu(
            generator.buffer,
            feat_acts,
            selection_mask=selection_mask,
        )
    )

    assert list(lazy_indices_dict) == list(legacy_indices_dict)
    assert lazy_n_bold == legacy_n_bold
    assert torch.equal(lazy_indices_bold, legacy_indices_bold)
    for group_name, legacy_indices in legacy_indices_dict.items():
        assert torch.equal(lazy_indices_dict[group_name], legacy_indices)


def test_bfloat16_downcast_can_change_interval_membership_vs_float32_baseline() -> None:
    cfg: SaeVisConfig = build_sae_vis_cfg()
    cfg.feature_centric_layout.seq_cfg.buffer = None  # type: ignore
    cfg.feature_centric_layout.seq_cfg.top_acts_group_size = 1  # type: ignore
    cfg.feature_centric_layout.seq_cfg.n_quantiles = 4  # type: ignore
    cfg.feature_centric_layout.seq_cfg.quantile_group_size = 16  # type: ignore

    tokens = torch.arange(4, dtype=torch.long).reshape(1, 4)
    generator = SequenceDataGenerator(cfg, tokens, torch.randn(4, 32))
    feat_acts_float32 = torch.tensor(
        [[0.5001, 0.1, 1.0, 0.0]],
        dtype=torch.float32,
    )
    feat_acts_bfloat16 = feat_acts_float32.to(torch.bfloat16)
    selection_mask = torch.tensor(
        [[True, True, True, False]],
        dtype=torch.bool,
    )

    legacy_indices_dict, _, _ = generator.get_indices_dict_legacy(
        generator.buffer,
        feat_acts_float32,
        selection_mask=selection_mask,
    )
    lazy_indices_dict, _, _ = generator.get_indices_dict_columnar_gpu(
        generator.buffer,
        feat_acts_bfloat16,
        selection_mask=selection_mask,
    )

    target_index = torch.tensor([0, 0], dtype=torch.long)

    def interval_groups_for_target(
        indices_dict: dict[str, torch.Tensor],
    ) -> list[str]:
        return [
            group_name
            for group_name, group_indices in indices_dict.items()
            if group_name.startswith("INTERVAL")
            and any(
                torch.equal(group_index, target_index) for group_index in group_indices
            )
        ]

    legacy_groups = interval_groups_for_target(legacy_indices_dict)
    lazy_groups = interval_groups_for_target(lazy_indices_dict)

    assert float(feat_acts_bfloat16[0, 0].float().item()) == pytest.approx(0.5)
    assert len(legacy_groups) == 1
    assert legacy_groups[0].startswith("INTERVAL 0.500 - 0.750")
    assert len(lazy_groups) == 2
    assert any(group.startswith("INTERVAL 0.250 - 0.500") for group in lazy_groups)
    assert any(group.startswith("INTERVAL 0.500 - 0.750") for group in lazy_groups)


def test_exact_boundary_interval_membership_becomes_disjoint_with_half_open_bins() -> (
    None
):
    cfg: SaeVisConfig = build_sae_vis_cfg()
    cfg.sequence_half_open_interval_bins = True
    cfg.feature_centric_layout.seq_cfg.buffer = None  # type: ignore
    cfg.feature_centric_layout.seq_cfg.top_acts_group_size = 1  # type: ignore
    cfg.feature_centric_layout.seq_cfg.n_quantiles = 4  # type: ignore
    cfg.feature_centric_layout.seq_cfg.quantile_group_size = 16  # type: ignore

    tokens = torch.arange(4, dtype=torch.long).reshape(1, 4)
    generator = SequenceDataGenerator(cfg, tokens, torch.randn(4, 32))
    feat_acts = torch.tensor(
        [[0.5, 0.1, 1.0, 0.0]],
        dtype=torch.float32,
    )
    selection_mask = torch.tensor(
        [[True, True, True, False]],
        dtype=torch.bool,
    )

    random.seed(12345)
    legacy_indices_dict, _, _ = generator.get_indices_dict_legacy(
        generator.buffer,
        feat_acts,
        selection_mask=selection_mask,
    )
    random.seed(12345)
    lazy_indices_dict, _, _ = generator.get_indices_dict_columnar_gpu(
        generator.buffer,
        feat_acts,
        selection_mask=selection_mask,
    )

    target_index = torch.tensor([0, 0], dtype=torch.long)

    def interval_groups_for_target(
        indices_dict: dict[str, torch.Tensor],
    ) -> list[str]:
        return [
            group_name
            for group_name, group_indices in indices_dict.items()
            if group_name.startswith("INTERVAL")
            and any(
                torch.equal(group_index, target_index) for group_index in group_indices
            )
        ]

    legacy_groups = interval_groups_for_target(legacy_indices_dict)
    lazy_groups = interval_groups_for_target(lazy_indices_dict)

    assert len(legacy_groups) == 1
    assert legacy_groups[0].startswith("INTERVAL 0.500 - 0.750")
    assert len(lazy_groups) == 1
    assert lazy_groups[0].startswith("INTERVAL 0.500 - 0.750")


def _batched_selection_fixture_generator(
    n_quantiles: int = 4,
    top_acts_group_size: int = 3,
    quantile_group_size: int = 2,
    batch: int = 3,
    seq: int = 8,
) -> SequenceDataGenerator:
    cfg: SaeVisConfig = build_sae_vis_cfg()
    cfg.feature_centric_layout.seq_cfg.buffer = None  # type: ignore
    cfg.feature_centric_layout.seq_cfg.top_acts_group_size = top_acts_group_size  # type: ignore
    cfg.feature_centric_layout.seq_cfg.n_quantiles = n_quantiles  # type: ignore
    cfg.feature_centric_layout.seq_cfg.quantile_group_size = quantile_group_size  # type: ignore
    tokens = torch.arange(batch * seq, dtype=torch.long).reshape(batch, seq)
    return SequenceDataGenerator(cfg, tokens, torch.randn(4, 32))


def _sequential_selection_reference(
    generator: SequenceDataGenerator,
    all_feat_acts: torch.Tensor,
    selection_mask: torch.Tensor | None,
) -> list[tuple[dict[str, torch.Tensor], torch.Tensor, int]]:
    return [
        generator.get_indices_dict_columnar_gpu(
            generator.buffer,
            all_feat_acts[..., feature_index],
            selection_mask=selection_mask,
        )
        for feature_index in range(all_feat_acts.shape[-1])
    ]


def _assert_selections_equal(
    batched: list[tuple[dict[str, torch.Tensor], torch.Tensor, int]],
    sequential: list[tuple[dict[str, torch.Tensor], torch.Tensor, int]],
) -> None:
    assert len(batched) == len(sequential)
    for feature_index, (batched_sel, sequential_sel) in enumerate(
        zip(batched, sequential)
    ):
        batched_dict, batched_bold, batched_n_bold = batched_sel
        sequential_dict, sequential_bold, sequential_n_bold = sequential_sel
        assert list(batched_dict) == list(sequential_dict), f"feature {feature_index}"
        for group_name in sequential_dict:
            assert torch.equal(
                batched_dict[group_name], sequential_dict[group_name]
            ), f"feature {feature_index} group {group_name!r}"
        assert torch.equal(batched_bold, sequential_bold), f"feature {feature_index}"
        assert batched_n_bold == sequential_n_bold, f"feature {feature_index}"


def test_get_indices_dicts_columnar_gpu_batched_matches_per_feature() -> None:
    generator = _batched_selection_fixture_generator()
    torch.manual_seed(202607)
    # Tie-free positive values with interval overfill so sampling RNG is exercised.
    all_feat_acts = (torch.rand(3, 8, 5, dtype=torch.float32) + 0.01) * torch.linspace(
        0.5, 2.0, 5
    )
    selection_mask = torch.ones(3, 8, dtype=torch.bool)
    selection_mask[:, -1] = False
    masked_acts = all_feat_acts * selection_mask.unsqueeze(-1)

    random.seed(20260703)
    sequential = _sequential_selection_reference(generator, masked_acts, selection_mask)
    random.seed(20260703)
    batched = generator.get_indices_dicts_columnar_gpu_batched(
        generator.buffer,
        masked_acts,
        selection_mask=selection_mask,
        feature_chunk_size=2,
    )

    _assert_selections_equal(batched, sequential)


def test_get_indices_dicts_columnar_gpu_batched_boundary_zero_and_negative() -> None:
    generator = _batched_selection_fixture_generator(quantile_group_size=16)
    all_feat_acts = torch.zeros(3, 8, 3, dtype=torch.float32)
    # Feature 0: value exactly on an interior interval boundary (dual membership).
    all_feat_acts[0, 1, 0] = 1.0
    all_feat_acts[0, 2, 0] = 0.25  # boundary of linspace(0, 1.0, 5)
    all_feat_acts[1, 3, 0] = 0.6
    # Feature 1: all zeros (feat_max == 0 degenerate all-membership case).
    # Feature 2: negative values only (no interval membership).
    all_feat_acts[..., 2] = -torch.rand(3, 8)
    selection_mask = torch.ones(3, 8, dtype=torch.bool)
    masked_acts = all_feat_acts * selection_mask.unsqueeze(-1)

    random.seed(31337)
    sequential = _sequential_selection_reference(generator, masked_acts, selection_mask)
    random.seed(31337)
    batched = generator.get_indices_dicts_columnar_gpu_batched(
        generator.buffer,
        masked_acts,
        selection_mask=selection_mask,
        feature_chunk_size=2,
    )

    _assert_selections_equal(batched, sequential)


def test_get_indices_dicts_columnar_gpu_batched_bfloat16_inputs() -> None:
    generator = _batched_selection_fixture_generator()
    torch.manual_seed(42)
    all_feat_acts = (torch.rand(3, 8, 4, dtype=torch.float32) + 0.01).to(torch.bfloat16)
    selection_mask = torch.ones(3, 8, dtype=torch.bool)
    selection_mask[0, 0] = False
    masked_acts = all_feat_acts * selection_mask.unsqueeze(-1)

    random.seed(7)
    sequential = _sequential_selection_reference(generator, masked_acts, selection_mask)
    random.seed(7)
    batched = generator.get_indices_dicts_columnar_gpu_batched(
        generator.buffer,
        masked_acts,
        selection_mask=selection_mask,
        feature_chunk_size=3,
    )

    _assert_selections_equal(batched, sequential)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_get_indices_dicts_columnar_gpu_batched_cuda_selection_device() -> None:
    generator = _batched_selection_fixture_generator()
    torch.manual_seed(9)
    # Tie-free values so CUDA top-k tie-order differences cannot mask real regressions.
    all_feat_acts = (torch.rand(3, 8, 4, dtype=torch.float32) + 0.01) * torch.linspace(
        0.5, 2.0, 4
    )
    selection_mask = torch.ones(3, 8, dtype=torch.bool)
    masked_acts = (all_feat_acts * selection_mask.unsqueeze(-1)).to(torch.bfloat16)

    random.seed(11)
    sequential = _sequential_selection_reference(generator, masked_acts, selection_mask)
    random.seed(11)
    batched = generator.get_indices_dicts_columnar_gpu_batched(
        generator.buffer,
        masked_acts,
        selection_mask=selection_mask,
        selection_device="cuda",
        feature_chunk_size=3,
    )

    _assert_selections_equal(batched, sequential)


def test_get_sequence_coordinate_table_precomputed_selection_matches_inline() -> None:
    generator = _batched_selection_fixture_generator()
    torch.manual_seed(5)
    all_feat_acts = torch.rand(3, 8, 2, dtype=torch.float32) + 0.01
    selection_mask = torch.ones(3, 8, dtype=torch.bool)
    masked_acts = all_feat_acts * selection_mask.unsqueeze(-1)
    feat_logits = torch.linspace(-1.0, 1.0, 32)

    random.seed(99)
    batched = generator.get_indices_dicts_columnar_gpu_batched(
        generator.buffer,
        masked_acts,
        selection_mask=selection_mask,
    )
    random.seed(99)
    for feature_index in range(masked_acts.shape[-1]):
        inline_table = generator.get_sequence_coordinate_table(
            feat_acts=masked_acts[..., feature_index],
            feat_logits=feat_logits,
            resid_post=torch.empty(0),
            feature_resid_dir=torch.empty(0),
            selection_mask=selection_mask,
            selection_backend="columnar_gpu",
        )
        precomputed_table = generator.get_sequence_coordinate_table(
            feat_acts=masked_acts[..., feature_index],
            feat_logits=feat_logits,
            resid_post=torch.empty(0),
            feature_resid_dir=torch.empty(0),
            selection_mask=selection_mask,
            selection_backend="columnar_gpu",
            precomputed_selection=batched[feature_index],
        )
        assert (
            precomputed_table.to_sequence_multi_group_data()
            == inline_table.to_sequence_multi_group_data()
        )


# ---------------------------------------------------------------------------
# Opt-in selection hygiene (sequence_top_acts_positive_only,
# sequence_dedup_across_groups, sequence_skip_dead_features): columnar backend
# only, all default off — flag-off behavior is pinned by every other test in
# this module.
# ---------------------------------------------------------------------------


def _coordinate_multiset(
    indices_dict: dict[str, torch.Tensor],
) -> list[tuple[int, int]]:
    coordinates: list[tuple[int, int]] = []
    for group_indices in indices_dict.values():
        coordinates.extend((int(row), int(col)) for row, col in group_indices.tolist())
    return coordinates


def test_columnar_top_positive_only_drops_zero_tie_fill() -> None:
    generator = _batched_selection_fixture_generator(
        n_quantiles=0, top_acts_group_size=5
    )
    feat_acts = torch.zeros(3, 8, dtype=torch.float32)
    feat_acts[0, 1] = 2.0
    feat_acts[1, 3] = 1.0
    selection_mask = torch.ones(3, 8, dtype=torch.bool)

    indices_dict_off, _, n_bold_off = generator.get_indices_dict_columnar_gpu(
        generator.buffer, feat_acts, selection_mask=selection_mask
    )
    # Flag off: zero ties fill the remaining TOP slots (inherited behavior).
    assert n_bold_off == 5

    generator.cfg.sequence_top_acts_positive_only = True  # pyright: ignore
    indices_dict_on, _, n_bold_on = generator.get_indices_dict_columnar_gpu(
        generator.buffer, feat_acts, selection_mask=selection_mask
    )
    assert n_bold_on == 2
    top_group = next(iter(indices_dict_on.values()))
    selected = {(int(row), int(col)) for row, col in top_group.tolist()}
    assert selected == {(0, 1), (1, 3)}


def test_columnar_skip_dead_features_emits_empty_selection() -> None:
    generator = _batched_selection_fixture_generator()
    feat_acts = torch.zeros(3, 8, dtype=torch.float32)
    selection_mask = torch.ones(3, 8, dtype=torch.bool)

    indices_dict_off, _, n_bold_off = generator.get_indices_dict_columnar_gpu(
        generator.buffer, feat_acts, selection_mask=selection_mask
    )
    # Flag off: degenerate zero-interval groups still emit rows (inherited behavior).
    assert n_bold_off > 0

    generator.cfg.sequence_skip_dead_features = True  # pyright: ignore
    indices_dict_on, indices_bold_on, n_bold_on = (
        generator.get_indices_dict_columnar_gpu(
            generator.buffer, feat_acts, selection_mask=selection_mask
        )
    )
    assert n_bold_on == 0
    assert indices_bold_on.shape == (0, 2)
    assert list(indices_dict_on) == ["TOP ACTIVATIONS<br>MAX = 0.000"]
    assert indices_dict_on["TOP ACTIVATIONS<br>MAX = 0.000"].shape == (0, 2)


def test_columnar_dedup_across_groups_removes_top_interval_double_selection() -> None:
    # Single dominant max with a sparse highest interval: without dedup the max
    # coordinate is selected by TOP and re-emitted by its containing interval.
    generator = _batched_selection_fixture_generator(
        n_quantiles=4, top_acts_group_size=2, quantile_group_size=8
    )
    feat_acts = torch.zeros(3, 8, dtype=torch.float32)
    feat_acts[0, 1] = 10.0
    feat_acts[1, 3] = 9.5
    feat_acts[2, 5] = 0.5
    selection_mask = torch.ones(3, 8, dtype=torch.bool)

    indices_dict_off, _, _ = generator.get_indices_dict_columnar_gpu(
        generator.buffer, feat_acts, selection_mask=selection_mask
    )
    coordinates_off = _coordinate_multiset(indices_dict_off)
    assert coordinates_off.count((0, 1)) >= 2  # TOP ∩ interval duplicate exists

    generator.cfg.sequence_dedup_across_groups = True  # pyright: ignore
    random.seed(20260707)
    indices_dict_on, _, _ = generator.get_indices_dict_columnar_gpu(
        generator.buffer, feat_acts, selection_mask=selection_mask
    )
    coordinates_on = _coordinate_multiset(indices_dict_on)
    assert len(coordinates_on) == len(set(coordinates_on))
    assert coordinates_on.count((0, 1)) == 1


def test_columnar_batched_matches_sequential_with_hygiene_flags() -> None:
    generator = _batched_selection_fixture_generator(
        n_quantiles=4, top_acts_group_size=3, quantile_group_size=2
    )
    generator.cfg.sequence_top_acts_positive_only = True  # pyright: ignore
    generator.cfg.sequence_dedup_across_groups = True  # pyright: ignore
    generator.cfg.sequence_skip_dead_features = True  # pyright: ignore

    torch.manual_seed(20260707)
    all_feat_acts = (torch.rand(3, 8, 5, dtype=torch.float32) + 0.01) * torch.linspace(
        0.5, 2.0, 5
    )
    all_feat_acts[..., 1] = 0.0  # dead feature exercises the skip path
    all_feat_acts[0, 2, 3] = 0.0  # sparse zeros exercise positive-only truncation
    all_feat_acts[1, 4, 3] = 0.0
    selection_mask = torch.ones(3, 8, dtype=torch.bool)
    selection_mask[:, -1] = False
    masked_acts = all_feat_acts * selection_mask.unsqueeze(-1)

    random.seed(20260707)
    sequential = _sequential_selection_reference(generator, masked_acts, selection_mask)
    random.seed(20260707)
    batched = generator.get_indices_dicts_columnar_gpu_batched(
        generator.buffer,
        masked_acts,
        selection_mask=selection_mask,
        feature_chunk_size=2,
    )

    _assert_selections_equal(batched, sequential)
    for indices_dict, _, _ in batched:
        coordinates = _coordinate_multiset(indices_dict)
        assert len(coordinates) == len(set(coordinates))


def test_columnar_batched_matches_sequential_with_half_open_bins() -> None:
    generator = _batched_selection_fixture_generator(
        n_quantiles=4, top_acts_group_size=2, quantile_group_size=16
    )
    generator.cfg.sequence_half_open_interval_bins = True  # pyright: ignore

    all_feat_acts = torch.zeros(3, 8, 4, dtype=torch.float32)
    # Feature 0: value exactly on an interior boundary of linspace(0, 1.0, 5).
    all_feat_acts[0, 1, 0] = 1.0
    all_feat_acts[0, 2, 0] = 0.25
    all_feat_acts[1, 3, 0] = 0.5
    # Feature 1: generic positive spread; feature 2: dead; feature 3: boundary at 0.75.
    all_feat_acts[..., 1] = torch.rand(3, 8) + 0.01
    all_feat_acts[2, 4, 3] = 2.0
    all_feat_acts[1, 5, 3] = 1.5
    selection_mask = torch.ones(3, 8, dtype=torch.bool)
    masked_acts = all_feat_acts * selection_mask.unsqueeze(-1)

    random.seed(20260708)
    sequential = _sequential_selection_reference(generator, masked_acts, selection_mask)
    random.seed(20260708)
    batched = generator.get_indices_dicts_columnar_gpu_batched(
        generator.buffer,
        masked_acts,
        selection_mask=selection_mask,
        feature_chunk_size=2,
    )

    _assert_selections_equal(batched, sequential)

    # Half-open membership: each boundary coordinate belongs to exactly one interval group.
    boundary_coordinates = [
        torch.tensor([0, 2], dtype=torch.long),  # 0.25 on feature 0
        torch.tensor([1, 3], dtype=torch.long),  # 0.50 on feature 0
    ]
    indices_dict = batched[0][0]
    for coordinate in boundary_coordinates:
        member_groups = [
            group_name
            for group_name, group_indices in indices_dict.items()
            if group_name.startswith("INTERVAL")
            and any(torch.equal(row, coordinate) for row in group_indices)
        ]
        assert len(member_groups) == 1, (coordinate.tolist(), member_groups)
