"""
Unit tests for the HuggingFace model wrapper and hook utilities.
"""

import pytest
import torch

from sae_dashboard.hook_utils import (
    HookInfo,
    get_hf_module_path,
    parse_transformer_lens_hook,
    transformer_lens_to_hf_hook,
)
from sae_dashboard.huggingface_model_wrapper import (
    HFActivationConfig,
    HuggingFaceModelWrapper,
    load_huggingface_model,
)


class TestParseTransformerLensHook:
    """Tests for parse_transformer_lens_hook function."""

    def test_hook_resid_post(self):
        layer_idx, hook_type = parse_transformer_lens_hook("blocks.5.hook_resid_post")
        assert layer_idx == 5
        assert hook_type == "hook_resid_post"

    def test_hook_resid_pre(self):
        layer_idx, hook_type = parse_transformer_lens_hook("blocks.0.hook_resid_pre")
        assert layer_idx == 0
        assert hook_type == "hook_resid_pre"

    def test_hook_mlp_out(self):
        layer_idx, hook_type = parse_transformer_lens_hook("blocks.12.hook_mlp_out")
        assert layer_idx == 12
        assert hook_type == "hook_mlp_out"

    def test_hook_attn_out(self):
        layer_idx, hook_type = parse_transformer_lens_hook("blocks.3.hook_attn_out")
        assert layer_idx == 3
        assert hook_type == "hook_attn_out"

    def test_nested_hook(self):
        layer_idx, hook_type = parse_transformer_lens_hook("blocks.7.attn.hook_pattern")
        assert layer_idx == 7
        assert hook_type == "attn.hook_pattern"

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError, match="Invalid TransformerLens hook name"):
            parse_transformer_lens_hook("invalid_hook_format")

    def test_missing_blocks_prefix_raises(self):
        with pytest.raises(ValueError, match="Invalid TransformerLens hook name"):
            parse_transformer_lens_hook("layers.5.hook_resid_post")


class TestGetHfModulePath:
    """Tests for get_hf_module_path function."""

    def test_hook_resid_post(self):
        path, capture_output, output_index = get_hf_module_path(5, "hook_resid_post")
        assert path == "model.layers.5"
        assert capture_output is True
        assert output_index == 0

    def test_hook_resid_pre(self):
        path, capture_output, output_index = get_hf_module_path(3, "hook_resid_pre")
        assert path == "model.layers.3"
        assert capture_output is False
        assert output_index is None

    def test_hook_mlp_out(self):
        path, capture_output, output_index = get_hf_module_path(7, "hook_mlp_out")
        assert path == "model.layers.7.mlp"
        assert capture_output is True
        assert output_index is None

    def test_hook_attn_out(self):
        path, capture_output, output_index = get_hf_module_path(2, "hook_attn_out")
        assert path == "model.layers.2.self_attn"
        assert capture_output is True
        assert output_index == 0

    def test_unsupported_hook_raises(self):
        with pytest.raises(NotImplementedError, match="not yet supported"):
            get_hf_module_path(0, "hook_z")

    def test_ln_hook_raises(self):
        with pytest.raises(NotImplementedError, match="Layer normalization hooks"):
            get_hf_module_path(0, "ln1.hook_normalized")


class TestTransformerLensToHfHook:
    """Tests for transformer_lens_to_hf_hook function."""

    def test_returns_hook_info(self):
        info = transformer_lens_to_hf_hook("blocks.5.hook_resid_post")
        assert isinstance(info, HookInfo)
        assert info.transformer_lens_name == "blocks.5.hook_resid_post"
        assert info.layer_index == 5
        assert info.hook_type == "hook_resid_post"
        assert info.hf_module_path == "model.layers.5"
        assert info.capture_output is True
        assert info.output_index == 0

    def test_hook_resid_pre(self):
        info = transformer_lens_to_hf_hook("blocks.0.hook_resid_pre")
        assert info.layer_index == 0
        assert info.hook_type == "hook_resid_pre"
        assert info.capture_output is False


class TestHFActivationConfig:
    """Tests for HFActivationConfig dataclass."""

    def test_basic_config(self):
        config = HFActivationConfig(
            primary_hook_point="blocks.5.hook_resid_post",
            auxiliary_hook_points=["blocks.0.hook_resid_pre"],
        )
        assert config.primary_hook_point == "blocks.5.hook_resid_post"
        assert config.auxiliary_hook_points == ["blocks.0.hook_resid_pre"]

    def test_empty_auxiliary(self):
        config = HFActivationConfig(
            primary_hook_point="blocks.2.hook_resid_post",
            auxiliary_hook_points=[],
        )
        assert len(config.auxiliary_hook_points) == 0


# Integration tests that require loading a model (marked as slow)
@pytest.mark.slow
class TestHuggingFaceModelWrapper:
    """Integration tests for HuggingFaceModelWrapper."""

    @pytest.fixture
    def small_model_and_tokenizer(self):
        """Load a small model for testing."""
        model, tokenizer = load_huggingface_model(
            "sshleifer/tiny-gpt2",
            device="cpu",
            dtype="float32",
        )
        return model, tokenizer

    @pytest.fixture
    def wrapper(self, small_model_and_tokenizer):
        """Create a wrapper for testing."""
        model, tokenizer = small_model_and_tokenizer
        config = HFActivationConfig(
            primary_hook_point="blocks.0.hook_resid_post",
            auxiliary_hook_points=[],
        )
        return HuggingFaceModelWrapper(
            model=model,
            tokenizer=tokenizer,
            activation_config=config,
            dtype=torch.float32,
        )

    def test_wrapper_creation(self, wrapper):
        """Test that the wrapper is created correctly."""
        assert wrapper.hook_layer == 0
        assert wrapper.primary_hook_info.hook_type == "hook_resid_post"

    def test_tokenizer_property(self, wrapper):
        """Test that tokenizer is accessible."""
        assert wrapper.tokenizer is not None

    def test_w_u_property(self, wrapper):
        """Test that W_U (unembedding matrix) is accessible."""
        W_U = wrapper.W_U
        assert W_U is not None
        # W_U should be (d_model, vocab_size)
        assert len(W_U.shape) == 2

    def test_forward_returns_activations(self, wrapper):
        """Test that forward pass returns activations."""
        tokens = wrapper.tokenizer.encode("Hello world", return_tensors="pt")
        result = wrapper.forward(tokens, return_logits=False)

        assert isinstance(result, dict)
        assert "blocks.0.hook_resid_post" in result
        activations = result["blocks.0.hook_resid_post"]

        # Check shape: (batch, seq_len, d_model)
        assert len(activations.shape) == 3
        assert activations.shape[0] == 1  # batch size
        assert activations.shape[1] == tokens.shape[1]  # seq length

    def test_forward_with_batch(self, wrapper):
        """Test forward pass with batched input."""
        texts = ["Hello world", "Test sentence"]
        tokens = wrapper.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        )["input_ids"]

        result = wrapper.forward(tokens, return_logits=False)
        activations = result["blocks.0.hook_resid_post"]

        assert activations.shape[0] == 2  # batch size of 2


@pytest.mark.slow
class TestHuggingFaceModelWrapperResidPre:
    """Test hook_resid_pre with HuggingFace wrapper."""

    @pytest.fixture
    def wrapper(self):
        """Create a wrapper for hook_resid_pre testing."""
        model, tokenizer = load_huggingface_model(
            "sshleifer/tiny-gpt2",
            device="cpu",
            dtype="float32",
        )
        config = HFActivationConfig(
            primary_hook_point="blocks.1.hook_resid_pre",
            auxiliary_hook_points=[],
        )
        return HuggingFaceModelWrapper(
            model=model,
            tokenizer=tokenizer,
            activation_config=config,
            dtype=torch.float32,
        )

    def test_resid_pre_returns_activations(self, wrapper):
        """Test that hook_resid_pre captures input activations."""
        tokens = wrapper.tokenizer.encode("Hello", return_tensors="pt")
        result = wrapper.forward(tokens, return_logits=False)

        assert "blocks.1.hook_resid_pre" in result
        activations = result["blocks.1.hook_resid_pre"]
        assert len(activations.shape) == 3
