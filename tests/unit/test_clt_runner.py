"""Tests for CLT (Cross-Layer Transcoder) support in Neuronpedia runner."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from sae_dashboard.neuronpedia.neuronpedia_runner_config import NeuronpediaRunnerConfig


class TestCLTLoading:
    """Test CLT loading and configuration."""

    def test_clt_config_enables_clt(self):
        """Test that use_clt=True enables CLT mode."""
        cfg = NeuronpediaRunnerConfig(
            sae_set="test",
            sae_path="test_path",
            model_id="gpt2",
            huggingface_dataset_path="test",
            outputs_dir="test_outputs",
            use_clt=True,
            clt_layer_idx=5,
        )
        assert cfg.use_clt is True
        assert cfg.clt_layer_idx == 5

    def test_clt_requires_layer_idx(self):
        """Test that CLT requires layer index to be specified."""
        cfg = NeuronpediaRunnerConfig(
            sae_set="test",
            sae_path="test_path",
            model_id="gpt2",
            huggingface_dataset_path="test",
            outputs_dir="test_outputs",
            use_clt=True,
            clt_layer_idx=None,  # This should be allowed but will fail at load time
        )
        assert cfg.clt_layer_idx is None

    def test_clt_weights_filename(self):
        """Test CLT weights filename configuration."""
        cfg = NeuronpediaRunnerConfig(
            sae_set="test",
            sae_path="test_path",
            model_id="gpt2",
            huggingface_dataset_path="test",
            outputs_dir="test_outputs",
            use_clt=True,
            clt_weights_filename="model.safetensors",
        )
        assert cfg.clt_weights_filename == "model.safetensors"


class TestCLTWrapperIntegration:
    """Test CLTLayerWrapper integration."""

    def test_clt_wrapper_initialization(self):
        """Test CLTLayerWrapper can be imported and basic initialization."""
        from sae_dashboard.clt_layer_wrapper import CLTLayerWrapper, CLTWrapperConfig

        # Mock CLT model
        mock_clt = MagicMock()
        mock_clt.config.num_layers = 12
        mock_clt.config.num_features = 32768
        mock_clt.config.d_model = 768
        mock_clt.config.activation_fn = "jumprelu"
        mock_clt.config.normalization_method = "none"
        mock_clt.config.tl_input_template = "blocks.{}.ln2.hook_normalized"
        mock_clt.device = torch.device("cpu")
        mock_clt.dtype = torch.float32
        mock_clt.log_threshold = torch.log(
            torch.ones(12, 32768) * 0.1
        )  # Mock log_threshold

        # Mock encoder and decoder modules
        mock_encoder = MagicMock()
        mock_encoder.weight = torch.randn(32768, 768)
        mock_encoder.bias_param = torch.zeros(32768)
        mock_clt.encoder_module.encoders = {5: mock_encoder}

        mock_decoder = MagicMock()
        mock_decoder.weight = torch.randn(768, 32768)
        mock_decoder.bias_param = torch.zeros(768)
        mock_clt.decoder_module.decoders = {"5->5": mock_decoder}

        # Create wrapper
        wrapper = CLTLayerWrapper(mock_clt, layer_idx=5)

        assert wrapper.layer_idx == 5
        assert wrapper.device == torch.device("cpu")
        assert wrapper.dtype == torch.float32
        assert wrapper.W_enc.shape == (768, 32768)  # Transposed
        assert wrapper.W_dec.shape == (32768, 768)  # Transposed

    def test_clt_wrapper_config(self):
        """Test CLTWrapperConfig creation."""
        from sae_dashboard.clt_layer_wrapper import CLTWrapperConfig

        cfg = CLTWrapperConfig(
            d_sae=32768,
            d_in=768,
            hook_name="blocks.5.ln2.hook_normalized",
            hook_layer=5,
            dtype="float32",
            device="cpu",
        )

        assert cfg.d_sae == 32768
        assert cfg.d_in == 768
        assert cfg.hook_name == "blocks.5.ln2.hook_normalized"
        assert cfg.hook_layer == 5

        # Test to_dict method
        cfg_dict = cfg.to_dict()
        assert isinstance(cfg_dict, dict)
        assert cfg_dict["d_sae"] == 32768

    def test_clt_wrapper_encode(self):
        """Test CLTLayerWrapper encode method."""
        from sae_dashboard.clt_layer_wrapper import CLTLayerWrapper

        # Mock CLT model
        mock_clt = MagicMock()
        mock_clt.config.num_layers = 12
        mock_clt.config.num_features = 32768
        mock_clt.config.d_model = 768
        mock_clt.config.activation_fn = "relu"  # Use relu for simplicity
        mock_clt.config.normalization_method = "none"
        mock_clt.config.tl_input_template = "blocks.{}.ln2.hook_normalized"
        mock_clt.device = torch.device("cpu")
        mock_clt.dtype = torch.float32

        # Mock encoder and decoder
        mock_encoder = MagicMock()
        mock_encoder.weight = torch.eye(10, 768)  # 10x768 (d_sae x d_model)
        mock_encoder.bias_param = torch.zeros(10)
        mock_clt.encoder_module.encoders = {5: mock_encoder}

        mock_decoder = MagicMock()
        mock_decoder.weight = torch.eye(768, 10)  # 768x10 (d_model x d_sae)
        mock_decoder.bias_param = torch.zeros(768)
        mock_clt.decoder_module.decoders = {"5->5": mock_decoder}

        # Create wrapper
        wrapper = CLTLayerWrapper(mock_clt, layer_idx=5)

        # Test encode
        x = torch.randn(2, 5, 768)  # batch x seq x d_model
        encoded = wrapper.encode(x)

        assert encoded.shape == (2, 5, 10)  # batch x seq x d_sae
        assert torch.all(encoded >= 0)  # ReLU should make all values non-negative


class TestCLTConfig:
    """Test CLT configuration handling."""

    def test_clt_config_loading(self):
        """Test loading CLT configuration from cfg.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a mock cfg.json
            cfg_data = {
                "num_features": 32768,
                "num_layers": 12,
                "d_model": 768,
                "model_name": "gpt2",
                "normalization_method": "mean_std",
                "activation_fn": "jumprelu",
                "tl_input_template": "blocks.{}.ln2.hook_normalized",
                "tl_output_template": "blocks.{}.mlp.hook_post",
            }

            cfg_path = Path(tmpdir) / "cfg.json"
            with open(cfg_path, "w") as f:
                json.dump(cfg_data, f)

            # Load and verify
            with open(cfg_path, "r") as f:
                loaded_cfg = json.load(f)

            assert loaded_cfg["num_features"] == 32768
            assert loaded_cfg["tl_input_template"] == "blocks.{}.ln2.hook_normalized"

    def test_clt_normalization_stats(self):
        """Test loading CLT normalization statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock norm_stats.json
            norm_stats = {
                "5": {
                    "inputs": {
                        "mean": [0.1] * 768,
                        "std": [1.0] * 768,
                    }
                }
            }

            stats_path = Path(tmpdir) / "norm_stats.json"
            with open(stats_path, "w") as f:
                json.dump(norm_stats, f)

            # Load and verify
            with open(stats_path, "r") as f:
                loaded_stats = json.load(f)

            assert "5" in loaded_stats
            assert len(loaded_stats["5"]["inputs"]["mean"]) == 768
            assert len(loaded_stats["5"]["inputs"]["std"]) == 768


class TestCLTFeatureMasking:
    """Test feature masking with CLT wrapper."""

    def test_clt_wrapper_fold_w_dec_norm(self):
        """Test fold_W_dec_norm method on CLT wrapper."""
        from sae_dashboard.clt_layer_wrapper import CLTLayerWrapper

        # Mock CLT model
        mock_clt = MagicMock()
        mock_clt.config.num_layers = 12
        mock_clt.config.num_features = 10
        mock_clt.config.d_model = 5
        mock_clt.config.activation_fn = "relu"
        mock_clt.config.normalization_method = "none"
        mock_clt.device = torch.device("cpu")
        mock_clt.dtype = torch.float32

        # Create simple weights
        W_enc = torch.ones(10, 5) * 2.0  # Will be transposed to 5x10
        W_dec = torch.ones(5, 10) * 3.0  # Will be transposed to 10x5

        mock_encoder = MagicMock()
        mock_encoder.weight = W_enc
        mock_encoder.bias_param = torch.ones(10) * 0.5
        mock_clt.encoder_module.encoders = {0: mock_encoder}

        mock_decoder = MagicMock()
        mock_decoder.weight = W_dec
        mock_decoder.bias_param = torch.zeros(5)
        mock_clt.decoder_module.decoders = {"0->0": mock_decoder}

        # Create wrapper
        wrapper = CLTLayerWrapper(mock_clt, layer_idx=0)

        # Original shapes after transpose
        assert wrapper.W_enc.shape == (5, 10)
        assert wrapper.W_dec.shape == (10, 5)

        # Fold W_dec norm
        wrapper.fold_W_dec_norm()

        # Check that folding was applied
        assert wrapper._W_dec_folded is True

        # W_dec norms should be sqrt(3^2 * 5) = sqrt(45) ≈ 6.71
        expected_norm = torch.sqrt(torch.tensor(45.0))

        # W_enc should be scaled by the norms
        expected_w_enc = torch.ones(5, 10) * 2.0 * expected_norm
        assert torch.allclose(wrapper.W_enc, expected_w_enc, rtol=1e-4)

    def test_clt_wrapper_jumprelu_threshold(self):
        """Test JumpReLU threshold handling in CLT wrapper."""
        from sae_dashboard.clt_layer_wrapper import CLTLayerWrapper

        # Mock CLT model with JumpReLU
        mock_clt = MagicMock()
        mock_clt.config.num_layers = 12
        mock_clt.config.num_features = 10
        mock_clt.config.d_model = 5
        mock_clt.config.activation_fn = "jumprelu"
        mock_clt.config.normalization_method = "none"
        mock_clt.device = torch.device("cpu")
        mock_clt.dtype = torch.float32

        # Mock log_threshold
        mock_clt.log_threshold = torch.log(
            torch.ones(12, 10) * 0.1
        )  # 12 layers, 10 features

        # Mock encoder and decoder
        mock_encoder = MagicMock()
        mock_encoder.weight = torch.ones(10, 5)
        mock_encoder.bias_param = torch.zeros(10)
        mock_clt.encoder_module.encoders = {5: mock_encoder}

        mock_decoder = MagicMock()
        mock_decoder.weight = torch.ones(5, 10)
        mock_decoder.bias_param = torch.zeros(5)
        mock_clt.decoder_module.decoders = {"5->5": mock_decoder}

        # Create wrapper
        wrapper = CLTLayerWrapper(mock_clt, layer_idx=5)

        # Check threshold was initialized
        assert wrapper.threshold is not None
        assert wrapper.threshold.shape == (10,)
        assert torch.allclose(wrapper.threshold, torch.ones(10) * 0.1, rtol=1e-4)
