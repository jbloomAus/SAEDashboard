# pyright: basic, reportPrivateImportUsage=false
import torch
from sae_lens import SAE
from torch import nn

from sae_dashboard.feature_data_generator import (
    FeatureDataGenerator,
    FeatureMaskingContext,
    PromptTokenMinibatch,
)


@torch.no_grad()
def test_feature_mask_context(autoencoder: SAE):  # type: ignore
    feature_indices = list(range(10))

    sae_in_mock = torch.randn(10, 64, autoencoder.cfg.d_in)
    features = autoencoder.encode(sae_in_mock)
    original_feature_acts = features[:, :, feature_indices]

    with FeatureMaskingContext(autoencoder, feature_indices):
        new_feature_acts = autoencoder.encode(sae_in_mock)

    assert (original_feature_acts - new_feature_acts).max() <= 1e-5


@torch.no_grad()
def test_feature_mask_context_restores_original_parameters(autoencoder: SAE):  # type: ignore
    feature_indices = list(range(10))
    original_w_dec = autoencoder.W_dec
    original_w_enc = autoencoder.W_enc
    original_b_enc = autoencoder.b_enc
    original_w_dec_value = original_w_dec.detach().clone()
    original_w_enc_value = original_w_enc.detach().clone()
    original_b_enc_value = original_b_enc.detach().clone()  # pyright: ignore

    with FeatureMaskingContext(autoencoder, feature_indices):
        assert autoencoder.W_dec.shape[0] == len(feature_indices)
        assert autoencoder.W_enc.shape[1] == len(feature_indices)
        assert autoencoder.b_enc.shape[0] == len(  # pyright: ignore[reportIndexIssue]
            feature_indices
        )

        autoencoder.W_dec.data.zero_()
        autoencoder.W_enc.data.zero_()
        autoencoder.b_enc.data.zero_()  # pyright: ignore

    assert autoencoder.W_dec is original_w_dec
    assert autoencoder.W_enc is original_w_enc
    assert autoencoder.b_enc is original_b_enc
    assert torch.equal(autoencoder.W_dec, original_w_dec_value)
    assert torch.equal(autoencoder.W_enc, original_w_enc_value)
    assert torch.equal(autoencoder.b_enc, original_b_enc_value)  # pyright: ignore


class _DummyJumpReLUSkipCfg:
    architecture = "jumprelu_skip_transcoder"


class _DummyJumpReLUSkipSAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.cfg = _DummyJumpReLUSkipCfg()
        self.W_dec = nn.Parameter(torch.randn(8, 4))
        self.W_enc = nn.Parameter(torch.randn(4, 8))
        self.b_enc = nn.Parameter(torch.randn(8))
        self.threshold = nn.Parameter(torch.randn(8))


@torch.no_grad()
def test_feature_mask_context_restores_jumprelu_skip_threshold():
    sae = _DummyJumpReLUSkipSAE()
    feature_indices = [1, 3, 5]
    original_threshold = sae.threshold
    original_threshold_value = sae.threshold.detach().clone()

    with FeatureMaskingContext(sae, feature_indices):  # pyright: ignore
        assert sae.threshold.shape[0] == len(feature_indices)
        sae.threshold.data.zero_()

    assert sae.threshold is original_threshold
    assert torch.equal(sae.threshold, original_threshold_value)


def test_get_model_acts_uses_primary_acts_batch_size():
    class _FakeModel:
        def __init__(self):
            self.forward_batch_sizes = []
            self.activation_config = type(
                "ActivationConfig", (), {"primary_hook_point": "hook"}
            )()

        def forward(self, tokens, return_logits=False):
            self.forward_batch_sizes.append(tokens.shape[0])
            return {"hook": torch.ones(tokens.shape[0], tokens.shape[1], 1)}

        @staticmethod
        def _activation_shapes_match_tokens(activation_dict, tokens):
            return tuple(activation_dict["hook"].shape[:2]) == tuple(tokens.shape[:2])

    generator = FeatureDataGenerator.__new__(FeatureDataGenerator)
    generator.cfg = type(  # pyright: ignore
        "Cfg",
        (),
        {
            "primary_acts_batch_size": 2,
            "cache_dir": None,
            "device": "cpu",
            "log_performance": False,
        },
    )()
    generator.model = _FakeModel()  # pyright: ignore

    activation_dict = generator.get_model_acts(
        0,
        PromptTokenMinibatch(
            prompt_indices=tuple(range(5)),
            tokens=torch.zeros(5, 3, dtype=torch.long),
            seq_length=3,
            primary_acts_batch_size=2,
        ),
    )

    assert generator.model.forward_batch_sizes == [2, 2, 1]
    assert activation_dict["hook"].shape == (5, 3, 1)
    assert generator._last_activation_capture_stats.model_forward_passes == 3
    assert generator._last_activation_capture_stats.total_forward_wall_s >= 0.0
