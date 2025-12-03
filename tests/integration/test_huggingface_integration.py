"""
Integration tests comparing TransformerLens and HuggingFace model outputs.

These tests verify that the HuggingFace model wrapper produces similar results
to TransformerLens for SAE dashboard generation.
"""

import pytest
import torch
from sae_lens.saes import StandardSAE, StandardSAEConfig

from sae_dashboard.huggingface_model_wrapper import (
    HFActivationConfig,
    HuggingFaceModelWrapper,
    load_huggingface_model,
)
from sae_dashboard.sae_vis_data import SaeVisConfig
from sae_dashboard.sae_vis_runner import FeatureDataGeneratorFactory


@pytest.fixture
def small_autoencoder():
    """Create a small autoencoder for testing.

    Note: d_in=2 to match tiny-gpt2's n_embd dimension.
    """
    d_in = 2  # tiny-gpt2 has n_embd=2
    d_sae = 8
    cfg = StandardSAEConfig(
        d_in=d_in,
        d_sae=d_sae,
        apply_b_dec_to_input=False,
        dtype="float32",
        normalize_activations="none",
        device="cpu",
    )
    autoencoder = StandardSAE(cfg)
    # Set deterministic weights
    torch.manual_seed(42)
    autoencoder.load_state_dict(
        {
            "W_enc": torch.randn(d_in, d_sae) * 0.1,
            "W_dec": torch.randn(d_sae, d_in) * 0.1,
            "b_enc": torch.zeros(d_sae),
            "b_dec": torch.zeros(d_in),
        }
    )
    return autoencoder


@pytest.mark.slow
class TestHuggingFaceVsTransformerLensActivations:
    """
    Tests comparing activation capture between HuggingFace and TransformerLens.
    """

    @pytest.fixture
    def hf_model_and_tokenizer(self):
        """Load HuggingFace model."""
        return load_huggingface_model(
            "sshleifer/tiny-gpt2",
            device="cpu",
            dtype="float32",
        )

    def test_hf_wrapper_activations_shape(self, hf_model_and_tokenizer):
        """Test that HuggingFace wrapper returns correctly shaped activations."""
        model, tokenizer = hf_model_and_tokenizer

        config = HFActivationConfig(
            primary_hook_point="blocks.0.hook_resid_post",
            auxiliary_hook_points=[],
        )
        wrapper = HuggingFaceModelWrapper(
            model=model,
            tokenizer=tokenizer,
            activation_config=config,
            dtype=torch.float32,
        )

        # Create test tokens
        tokens = tokenizer.encode("Hello world!", return_tensors="pt")
        result = wrapper.forward(tokens, return_logits=False)

        # Check that we got the right hook point
        assert "blocks.0.hook_resid_post" in result
        acts = result["blocks.0.hook_resid_post"]

        # Should be (batch, seq, d_model)
        assert len(acts.shape) == 3
        assert acts.shape[0] == 1
        assert acts.shape[1] == tokens.shape[1]
        # d_model should match the model's hidden size
        assert acts.shape[2] == model.config.n_embd

    def test_hf_wrapper_activations_are_not_zeros(self, hf_model_and_tokenizer):
        """Verify activations are non-trivial (not all zeros)."""
        model, tokenizer = hf_model_and_tokenizer

        config = HFActivationConfig(
            primary_hook_point="blocks.0.hook_resid_post",
            auxiliary_hook_points=[],
        )
        wrapper = HuggingFaceModelWrapper(
            model=model,
            tokenizer=tokenizer,
            activation_config=config,
            dtype=torch.float32,
        )

        tokens = tokenizer.encode("Test input", return_tensors="pt")
        result = wrapper.forward(tokens, return_logits=False)
        acts = result["blocks.0.hook_resid_post"]

        # Activations should not be all zeros
        assert acts.abs().sum() > 0

    def test_hf_wrapper_activations_deterministic(self, hf_model_and_tokenizer):
        """Test that activations are deterministic given same input."""
        model, tokenizer = hf_model_and_tokenizer

        config = HFActivationConfig(
            primary_hook_point="blocks.0.hook_resid_post",
            auxiliary_hook_points=[],
        )
        wrapper = HuggingFaceModelWrapper(
            model=model,
            tokenizer=tokenizer,
            activation_config=config,
            dtype=torch.float32,
        )

        tokens = tokenizer.encode("Same input", return_tensors="pt")

        # Run twice
        result1 = wrapper.forward(tokens, return_logits=False)
        result2 = wrapper.forward(tokens, return_logits=False)

        acts1 = result1["blocks.0.hook_resid_post"]
        acts2 = result2["blocks.0.hook_resid_post"]

        # Should be identical
        assert torch.allclose(acts1, acts2)

    def test_hf_wrapper_w_u_shape(self, hf_model_and_tokenizer):
        """Test that W_U has correct shape."""
        model, tokenizer = hf_model_and_tokenizer

        config = HFActivationConfig(
            primary_hook_point="blocks.0.hook_resid_post",
            auxiliary_hook_points=[],
        )
        wrapper = HuggingFaceModelWrapper(
            model=model,
            tokenizer=tokenizer,
            activation_config=config,
            dtype=torch.float32,
        )

        W_U = wrapper.W_U
        # W_U should be (d_model, vocab_size)
        assert W_U.shape[0] == model.config.n_embd  # d_model
        assert W_U.shape[1] == model.config.vocab_size  # vocab_size


@pytest.mark.slow
class TestSaeVisConfigWithHuggingFace:
    """Test SaeVisConfig with use_huggingface flag."""

    def test_config_with_huggingface_flag(self):
        """Test that config accepts use_huggingface flag."""
        cfg = SaeVisConfig(
            hook_point="blocks.0.hook_resid_post",
            features=[0, 1, 2],
            use_huggingface=True,
            device="cpu",
            dtype="float32",
        )
        assert cfg.use_huggingface is True

    def test_config_without_huggingface_flag(self):
        """Test that config defaults to use_huggingface=False."""
        cfg = SaeVisConfig(
            hook_point="blocks.0.hook_resid_post",
            features=[0, 1, 2],
            device="cpu",
            dtype="float32",
        )
        assert cfg.use_huggingface is False


@pytest.mark.slow
class TestFeatureDataGeneratorFactoryHuggingFace:
    """Test FeatureDataGeneratorFactory with HuggingFace models."""

    @pytest.fixture
    def hf_model_and_tokenizer(self):
        """Load HuggingFace model."""
        return load_huggingface_model(
            "sshleifer/tiny-gpt2",
            device="cpu",
            dtype="float32",
        )

    def test_factory_creates_generator_for_hf(
        self, hf_model_and_tokenizer, small_autoencoder
    ):
        """Test that factory creates generator for HuggingFace model."""
        model, tokenizer = hf_model_and_tokenizer

        cfg = SaeVisConfig(
            hook_point="blocks.0.hook_resid_post",
            features=[0, 1],
            use_huggingface=True,
            device="cpu",
            dtype="float32",
            minibatch_size_tokens=2,
        )

        tokens = tokenizer.encode("Hello world", return_tensors="pt")

        # This should not raise
        generator = FeatureDataGeneratorFactory.create(
            cfg=cfg,
            model=model,
            encoder=small_autoencoder,
            tokens=tokens,
            tokenizer=tokenizer,
        )

        assert generator is not None

    def test_factory_raises_without_tokenizer_for_hf(
        self, hf_model_and_tokenizer, small_autoencoder
    ):
        """Test that factory raises when tokenizer not provided for HF model."""
        model, tokenizer = hf_model_and_tokenizer

        cfg = SaeVisConfig(
            hook_point="blocks.0.hook_resid_post",
            features=[0, 1],
            use_huggingface=True,
            device="cpu",
            dtype="float32",
        )

        tokens = tokenizer.encode("Hello", return_tensors="pt")

        with pytest.raises(ValueError, match="tokenizer must be provided"):
            FeatureDataGeneratorFactory.create(
                cfg=cfg,
                model=model,
                encoder=small_autoencoder,
                tokens=tokens,
                tokenizer=None,  # Missing tokenizer
            )

    def test_factory_raises_for_dfa_with_hf(
        self, hf_model_and_tokenizer, small_autoencoder
    ):
        """Test that factory raises when DFA requested with HuggingFace model."""
        model, tokenizer = hf_model_and_tokenizer

        cfg = SaeVisConfig(
            hook_point="blocks.0.hook_resid_post",
            features=[0, 1],
            use_huggingface=True,
            use_dfa=True,  # DFA not supported
            device="cpu",
            dtype="float32",
        )

        tokens = tokenizer.encode("Hello", return_tensors="pt")

        with pytest.raises(NotImplementedError, match="DFA.*not yet supported"):
            FeatureDataGeneratorFactory.create(
                cfg=cfg,
                model=model,
                encoder=small_autoencoder,
                tokens=tokens,
                tokenizer=tokenizer,
            )
