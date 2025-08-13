import pytest
import torch
from transformer_lens import HookedTransformer

from sae_dashboard.sae_vis_data import SaeVisConfig
from sae_dashboard.sequence_data_generator import SequenceDataGenerator
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
    feat_acts = torch.randn_like(tokens, dtype=torch.float32)
    feat_logits = torch.randn(model.cfg.d_vocab, dtype=torch.float32)
    resid_post = torch.randn(
        tokens.shape[0], tokens.shape[1], model.cfg.d_model, dtype=torch.float32
    )
    feature_resid_dir = torch.randn(model.cfg.d_model, dtype=torch.float32)

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

    num_duplicates = sum(count - 1 for count in pair_counts.values())

    # Assertions
    assert not duplicates_in_same_group, "Duplicates found within the same group"
    assert num_duplicates <= len(sequence_multi_group_data.seq_group_data), (
        f"Too many duplicates: {num_duplicates}"
    )
    assert len(sequence_multi_group_data.seq_group_data[0].seq_data) == 20, (
        "TOP ACTIVATIONS group should have 20 sequences"
    )

    # Check that duplicates only occur between TOP ACTIVATIONS and one other group
    for pair, count in pair_counts.items():
        if count > 1:
            groups_with_pair = [
                i for i, oi, qti in group_sequence_pairs if (oi, qti) == pair
            ]
            assert 0 in groups_with_pair, (
                f"Duplicate {pair} not in TOP ACTIVATIONS group"
            )
            assert len(groups_with_pair) == 2, (
                f"Duplicate {pair} found in more than two groups: {groups_with_pair}"
            )


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
    for group in (
        sequence_multi_group_data.seq_group_data
    ):  # Changed from sequence_groups to seq_group_data
        all_sequence_data.extend(group.seq_data)  # Changed from sequences to seq_data

    assert len(all_sequence_data) == len(
        set((sd.original_index, sd.qualifying_token_index) for sd in all_sequence_data)
    )
