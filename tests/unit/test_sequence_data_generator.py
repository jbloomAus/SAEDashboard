import torch
from transformer_lens import HookedTransformer

from sae_dashboard.sequence_data_generator import SequenceDataGenerator
from tests.helpers import build_sae_vis_cfg


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
