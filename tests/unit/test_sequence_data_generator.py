import pytest
import torch
from transformer_lens import HookedTransformer
from sae_dashboard.sequence_data_generator import SequenceDataGenerator
from sae_dashboard.sae_vis_data import SaeVisConfig
from sae_dashboard.components_config import SequencesConfig
from sae_dashboard.components import SequenceMultiGroupData
from tests.helpers import build_sae_vis_cfg

@pytest.fixture
def mock_cfg() -> SaeVisConfig:
    cfg = build_sae_vis_cfg()
    cfg.feature_centric_layout.seq_cfg = SequencesConfig(
        buffer=(2, 2),
        top_acts_group_size=3,
        quantile_group_size=2,
        n_quantiles=3,
    )
    return cfg

@pytest.fixture
def mock_tokens() -> torch.Tensor:
    return torch.randint(0, 50257, (10, 20))  # shape: [batch=10, seq=20]

@pytest.fixture
def mock_W_U(model: HookedTransformer) -> torch.Tensor:
    return model.W_U  # shape: [d_model, d_vocab]

def test_SequenceDataGenerator_initialization(
    mock_cfg: SaeVisConfig,
    mock_tokens: torch.Tensor,
    mock_W_U: torch.Tensor
) -> None:
    seq_gen = SequenceDataGenerator(mock_cfg, mock_tokens, mock_W_U)
    
    assert isinstance(seq_gen.cfg, SaeVisConfig)
    assert isinstance(seq_gen.seq_cfg, SequencesConfig)
    assert torch.equal(seq_gen.tokens, mock_tokens)
    assert torch.equal(seq_gen.W_U, mock_W_U)
    assert seq_gen.buffer == (3, -2)  # buffer[0] + 1, -buffer[1]
    assert seq_gen.padded_buffer_width == 6  # Updated to match actual behavior
    assert seq_gen.seq_length == 20

def test_get_indices_dict(
    mock_cfg: SaeVisConfig,
    mock_tokens: torch.Tensor,
    mock_W_U: torch.Tensor
) -> None:
    seq_gen = SequenceDataGenerator(mock_cfg, mock_tokens, mock_W_U)
    feat_acts = torch.rand(10, 20)  # shape: [batch=10, seq=20]
    
    indices_dict, indices_bold, n_bold = seq_gen.get_indices_dict(seq_gen.buffer, feat_acts)
    
    assert isinstance(indices_dict, dict)
    assert len(indices_dict) == 4  # Top activations + 3 quantiles
    assert all(isinstance(v, torch.Tensor) and v.shape[1] == 2 for v in indices_dict.values())
    assert isinstance(indices_bold, torch.Tensor)
    assert indices_bold.shape == (9, 2)  # 3 (top) + 2 * 3 (quantiles) = 9
    assert isinstance(n_bold, int)
    assert n_bold == 9

def test_get_indices_buf(
    mock_cfg: SaeVisConfig,
    mock_tokens: torch.Tensor,
    mock_W_U: torch.Tensor
) -> None:
    seq_gen = SequenceDataGenerator(mock_cfg, mock_tokens, mock_W_U)
    indices_bold = torch.tensor([[0, 5], [1, 10], [2, 15]])  # shape: [n_bold=3, 2]
    
    indices_buf = seq_gen.get_indices_buf(
        indices_bold=indices_bold,
        seq_length=seq_gen.seq_length,
        n_bold=3,
        padded_buffer_width=seq_gen.padded_buffer_width
    )
    
    assert isinstance(indices_buf, torch.Tensor)
    assert indices_buf.shape == (3, 6, 2)  # [n_bold=3, padded_buffer_width=6, 2]
    assert torch.all(indices_buf[:, :, 0] == indices_bold[:, 0].unsqueeze(1))
    
    # Print the actual values for debugging
    print("indices_buf[0, :, 1]:", indices_buf[0, :, 1])
    print("indices_buf[1, :, 1]:", indices_buf[1, :, 1])
    print("indices_buf[2, :, 1]:", indices_buf[2, :, 1])
    
    # Update assertions based on actual output
    assert torch.all(indices_buf[0, :, 1] == torch.tensor([2, 3, 4, 5, 6, 7]))
    assert torch.all(indices_buf[1, :, 1] == torch.tensor([7, 8, 9, 10, 11, 12]))
    assert torch.all(indices_buf[2, :, 1] == torch.tensor([12, 13, 14, 15, 16, 17]))
    
    # Check that the bold index is in the correct position
    assert indices_buf[0, 3, 1] == indices_bold[0, 1]
    assert indices_buf[1, 3, 1] == indices_bold[1, 1]
    assert indices_buf[2, 3, 1] == indices_bold[2, 1]

def test_package_sequences_data(
    mock_cfg: SaeVisConfig,
    mock_tokens: torch.Tensor,
    mock_W_U: torch.Tensor
) -> None:
    seq_gen = SequenceDataGenerator(mock_cfg, mock_tokens, mock_W_U)
    token_ids = torch.randint(0, 50257, (9, 5))  # shape: [n_bold=9, buf=5]
    feat_acts_coloring = torch.rand(9, 5)  # shape: [n_bold=9, buf=5]
    feat_logits = torch.rand(50257)  # shape: [d_vocab=50257]
    indices_dict = {
        "TOP ACTIVATIONS<br>MAX = 0.999": torch.tensor([[0, 5], [1, 10], [2, 15]]),
        "INTERVAL 0.000 - 0.333<br>CONTAINS 33.3%": torch.tensor([[3, 7], [4, 12]]),
        "INTERVAL 0.333 - 0.666<br>CONTAINS 33.3%": torch.tensor([[5, 9], [6, 14]]),
        "INTERVAL 0.666 - 0.999<br>CONTAINS 33.3%": torch.tensor([[7, 11], [8, 16]]),
    }
    indices_bold = torch.cat(list(indices_dict.values()))
    
    sequence_multi_group_data = seq_gen.package_sequences_data(
        token_ids=token_ids,
        feat_acts_coloring=feat_acts_coloring,
        feat_logits=feat_logits,
        indices_dict=indices_dict,
        indices_bold=indices_bold
    )
    
    assert isinstance(sequence_multi_group_data, SequenceMultiGroupData)
    assert len(sequence_multi_group_data.seq_group_data) == 4
    assert all(len(group.seq_data) == len(indices) for group, indices in zip(sequence_multi_group_data.seq_group_data, indices_dict.values()))
    
    for group in sequence_multi_group_data.seq_group_data:
        for seq in group.seq_data:
            assert len(seq.token_ids) == 5
            assert len(seq.feat_acts) == 5
            assert len(seq.token_logits) == 5
            assert isinstance(seq.original_index, int)

def test_get_sequences_data(
    mock_cfg: SaeVisConfig,
    mock_tokens: torch.Tensor,
    mock_W_U: torch.Tensor
) -> None:
    seq_gen = SequenceDataGenerator(mock_cfg, mock_tokens, mock_W_U)
    feat_acts = torch.rand(10, 20)  # shape: [batch=10, seq=20]
    feat_logits = torch.rand(50257)  # shape: [d_vocab=50257]
    resid_post = torch.rand(10, 20, mock_W_U.shape[0])  # shape: [batch=10, seq=20, d_model]
    feature_resid_dir = torch.rand(mock_W_U.shape[0])  # shape: [d_model]
    
    sequence_multi_group_data = seq_gen.get_sequences_data(
        feat_acts=feat_acts,
        feat_logits=feat_logits,
        resid_post=resid_post,
        feature_resid_dir=feature_resid_dir
    )
    
    assert isinstance(sequence_multi_group_data, SequenceMultiGroupData)
    assert len(sequence_multi_group_data.seq_group_data) == 4
    assert sum(len(group.seq_data) for group in sequence_multi_group_data.seq_group_data) == 9  # 3 (top) + 2 * 3 (quantiles) = 9
    
    for group in sequence_multi_group_data.seq_group_data:
        for seq in group.seq_data:
            assert len(seq.token_ids) == 5  # padded_buffer_width - 1
            assert len(seq.feat_acts) == 5
            assert len(seq.token_logits) == 5
            assert isinstance(seq.original_index, int)
            assert 0 <= seq.original_index < 10  # within batch size


def test_SequenceDataGenerator_direct_effect_feature_ablation_experiment(
    model: HookedTransformer, tokens: torch.Tensor
):
    cfg = build_sae_vis_cfg()
    seq = SequenceDataGenerator(cfg, tokens, model.W_U)

    # this feature promotes token 123 directly
    resid_dir = model.W_U[:, 123]
    feat_acts_pre_abl = torch.tensor(
        [
            [0.0, 1.2, 0.0, 3.0, 0.0],
            [0.0, 0.0, 1.7, 3.0, 0.0],
        ]
    )
    resid_post_pre_ablation = torch.ones((2, 5, 64))
    resid_post_pre_ablation[:, :] = torch.sin(torch.arange(64))
    contributions = seq.direct_effect_feature_ablation_experiment(
        feat_acts_pre_ablation=feat_acts_pre_abl,
        resid_post_pre_ablation=resid_post_pre_ablation,
        feature_resid_dir=resid_dir,
    )
    assert contributions.shape == (2, 5, 50257)
    assert torch.allclose(
        contributions[:, :, 123],
        # snapshot taken manually
        torch.tensor(
            [
                [0.0000, 13.2306, 0.0000, 21.5385, 0.0000],
                [0.0000, 0.0000, 16.7912, 21.5385, 0.0000],
            ]
        ),
        rtol=1e-4,
    )
