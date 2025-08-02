"""Tests specific to transcoder functionality in NeuronpediaRunner."""

from unittest.mock import Mock, patch

import pytest
import torch

from sae_dashboard.feature_data_generator import FeatureMaskingContext
from sae_dashboard.neuronpedia.neuronpedia_runner import NeuronpediaRunnerConfig


@pytest.fixture
def transcoder_config():
    """Create a config for transcoder testing."""
    return NeuronpediaRunnerConfig(
        sae_set="gemma-scope-2b-pt-transcoders",
        sae_path="layer_15/width_16k/average_l0_8",
        np_set_name="test-transcoders",
        from_local_sae=False,
        huggingface_dataset_path="monology/pile-uncopyrighted",
        sae_dtype="float32",
        model_dtype="float32",
        outputs_dir="test_outputs/",
        sparsity_threshold=1,
        n_prompts_total=128,
        n_tokens_in_prompt=128,
        n_prompts_in_forward_pass=32,
        n_features_at_a_time=2,
        start_batch=0,
        end_batch=1,
        use_wandb=False,
        use_transcoder=True,  # Enable transcoder
    )


@pytest.fixture
def mock_transcoder():
    """Create a mock transcoder object that behaves like a real transcoder."""
    mock = Mock()

    # Mock the config with transcoder-specific attributes
    mock.cfg = Mock()
    mock.cfg.d_in = 2304
    mock.cfg.d_sae = 16384
    mock.cfg.dtype = "torch.float32"
    mock.cfg.device = "cpu"
    mock.cfg.metadata = {
        "model_name": "gemma-2-2b",
        "hook_name": "blocks.15.ln2.hook_normalized",
        "hook_head_index": None,
        "prepend_bos": True,
        "dataset_path": "monology/pile-uncopyrighted",
        "context_size": 1024,
    }
    # Transcoder-specific config attributes
    mock.cfg.hook_name_out = "blocks.15.hook_mlp_out"
    mock.cfg.hook_layer_out = 15
    mock.cfg.architecture = Mock(return_value="jumprelu_transcoder")

    # Mock other required attributes and methods
    mock.W_dec = torch.randn(16384, 2304)
    mock.b_dec = torch.zeros(2304)
    mock.device = "cpu"
    mock.fold_W_dec_norm = Mock()
    mock.encode = Mock(return_value=torch.randn(1, 128, 16384))
    mock.to = Mock(return_value=mock)

    return mock


class TestTranscoderLoading:
    """Test transcoder loading functionality."""

    def test_transcoder_config_enables_transcoder(
        self, transcoder_config: NeuronpediaRunnerConfig
    ):
        """Test that transcoder config properly enables transcoder loading."""
        assert transcoder_config.use_transcoder is True
        assert transcoder_config.use_skip_transcoder is False

    def test_skip_transcoder_config(self, transcoder_config: NeuronpediaRunnerConfig):
        """Test skip transcoder configuration."""
        transcoder_config.use_transcoder = False
        transcoder_config.use_skip_transcoder = True
        assert transcoder_config.use_transcoder is False
        assert transcoder_config.use_skip_transcoder is True

    def test_transcoder_vs_sae_config_differences(
        self, transcoder_config: NeuronpediaRunnerConfig
    ):
        """Test that transcoder config has different settings than SAE config."""
        # Create SAE config
        sae_config = NeuronpediaRunnerConfig(
            sae_set="test-sae",
            sae_path="test-path",
            np_set_name="test",
            huggingface_dataset_path="test",
            outputs_dir="test/",
            use_transcoder=False,
            use_skip_transcoder=False,
        )

        # Verify differences
        assert transcoder_config.use_transcoder is True
        assert sae_config.use_transcoder is False
        assert transcoder_config.sae_set == "gemma-scope-2b-pt-transcoders"
        assert "transcoder" in transcoder_config.sae_set

    def test_hook_layer_extraction_from_hook_name(self):
        """Test extracting layer number from hook_name pattern."""
        # Test the regex pattern used in neuronpedia_runner
        import re

        test_cases = [
            ("blocks.5.hook_resid_pre", 5),
            ("blocks.15.ln2.hook_normalized", 15),
            ("blocks.0.hook_mlp_out", 0),
            ("blocks.23.attn.hook_q", 23),
        ]

        for hook_name, expected_layer in test_cases:
            match = re.search(r"blocks\.(\d+)\.", hook_name)
            assert match is not None
            assert int(match.group(1)) == expected_layer


class TestTranscoderArchitectureHandling:
    """Test handling of different transcoder architectures."""

    def test_feature_masking_with_transcoder_architecture(self):
        """Test that FeatureMaskingContext handles transcoder architectures."""
        # Create a mock SAE with transcoder architecture as a method
        mock_sae = Mock()
        mock_sae.cfg = Mock()
        mock_sae.cfg.architecture = Mock(return_value="jumprelu_transcoder")

        # Set up the required tensors
        mock_sae.W_dec = torch.nn.Parameter(torch.randn(100, 64))
        mock_sae.W_enc = torch.nn.Parameter(torch.randn(64, 100))
        mock_sae.b_enc = torch.nn.Parameter(torch.randn(100))
        mock_sae.threshold = torch.nn.Parameter(torch.randn(100))

        feature_idxs = [0, 1, 2]

        # Test that context manager works with transcoder architecture
        with FeatureMaskingContext(mock_sae, feature_idxs):
            # Check that weights were masked
            assert mock_sae.W_dec.shape == (3, 64)
            assert mock_sae.W_enc.shape == (64, 3)
            assert mock_sae.b_enc.shape == (3,)
            assert mock_sae.threshold.shape == (3,)

        # Check that weights were restored
        assert mock_sae.W_dec.shape == (100, 64)
        assert mock_sae.W_enc.shape == (64, 100)
        assert mock_sae.b_enc.shape == (100,)
        assert mock_sae.threshold.shape == (100,)

    def test_feature_masking_with_architecture_as_attribute(self):
        """Test FeatureMaskingContext when architecture is an attribute not method."""
        # Create a mock SAE with architecture as direct attribute
        mock_sae = Mock()
        mock_sae.cfg = Mock()
        mock_sae.cfg.architecture = "standard_transcoder"  # Direct attribute

        # Set up the required tensors
        mock_sae.W_dec = torch.nn.Parameter(torch.randn(50, 32))
        mock_sae.W_enc = torch.nn.Parameter(torch.randn(32, 50))
        mock_sae.b_enc = torch.nn.Parameter(torch.randn(50))

        feature_idxs = [5, 10]

        # Test that context manager works
        with FeatureMaskingContext(mock_sae, feature_idxs):
            assert mock_sae.W_dec.shape == (2, 32)
            assert mock_sae.W_enc.shape == (32, 2)
            assert mock_sae.b_enc.shape == (2,)

        # Check restoration
        assert mock_sae.W_dec.shape == (50, 32)
        assert mock_sae.W_enc.shape == (32, 50)
        assert mock_sae.b_enc.shape == (50,)

    def test_all_transcoder_architectures(self):
        """Test that all transcoder architecture variants are recognized."""
        architectures_to_test = [
            "transcoder",
            "standard_transcoder",
            "skip_transcoder",
            "jumprelu_transcoder",
            "gated_transcoder",
        ]

        for arch in architectures_to_test:
            mock_sae = Mock()
            mock_sae.cfg = Mock()
            mock_sae.cfg.architecture = arch

            # Set up minimal required attributes
            mock_sae.W_dec = torch.nn.Parameter(torch.randn(10, 5))
            mock_sae.W_enc = torch.nn.Parameter(torch.randn(5, 10))
            mock_sae.b_enc = torch.nn.Parameter(torch.randn(10))

            # For architectures that need extra parameters
            if "jumprelu" in arch:
                mock_sae.threshold = torch.nn.Parameter(torch.randn(10))
            elif "gated" in arch:
                mock_sae.b_gate = torch.nn.Parameter(torch.randn(10))
                mock_sae.r_mag = torch.nn.Parameter(torch.randn(10))
                mock_sae.b_mag = torch.nn.Parameter(torch.randn(10))

            # Should not raise an error
            with FeatureMaskingContext(mock_sae, [0]):
                pass


class TestTranscoderMetadataHandling:
    """Test handling of transcoder metadata and config differences."""

    def test_prepend_bos_in_metadata(self):
        """Test that prepend_bos is correctly accessed from metadata."""
        from sae_dashboard.neuronpedia.neuronpedia_runner import NeuronpediaRunner

        # Create a mock SAE with prepend_bos in metadata
        mock_sae = Mock()
        mock_sae.cfg = Mock()
        # Make sure the metadata dict behaves properly
        metadata = {"prepend_bos": True}
        mock_sae.cfg.metadata = metadata

        # Ensure hasattr returns False for prepend_bos direct attribute
        mock_sae.cfg.prepend_bos = Mock()
        del mock_sae.cfg.prepend_bos

        # Create a runner instance (we'll mock everything else)
        with patch.object(NeuronpediaRunner, "__init__", lambda x, y: None):
            runner = NeuronpediaRunner(None)  # type: ignore
            runner.sae = mock_sae
            runner.cfg = Mock()
            runner.cfg.prefix_tokens = [1, 2, 3]
            runner.cfg.suffix_tokens = None

            # Test the add_prefix_suffix_to_tokens method
            tokens = torch.tensor([[0, 10, 20, 30, 40, 50]])
            result = runner.add_prefix_suffix_to_tokens(tokens)

            # Should have added prefix tokens (3) and kept same total length
            assert result.shape[1] == tokens.shape[1]
            # First token should be BOS (0), then prefix tokens
            assert result[0, 0].item() == 0
            assert result[0, 1].item() == 1
            assert result[0, 2].item() == 2
            assert result[0, 3].item() == 3

    def test_prepend_bos_as_attribute(self):
        """Test backward compatibility when prepend_bos is a direct attribute."""
        from sae_dashboard.neuronpedia.neuronpedia_runner import NeuronpediaRunner

        # Create a mock SAE with prepend_bos as direct attribute
        mock_sae = Mock()
        mock_sae.cfg = Mock()
        mock_sae.cfg.prepend_bos = False  # Direct attribute
        mock_sae.cfg.metadata = {}  # Empty metadata

        # Create a runner instance
        with patch.object(NeuronpediaRunner, "__init__", lambda x, y: None):
            runner = NeuronpediaRunner(None)  # type: ignore
            runner.sae = mock_sae
            runner.cfg = Mock()
            runner.cfg.prefix_tokens = [1, 2, 3]
            runner.cfg.suffix_tokens = None

            # Test the add_prefix_suffix_to_tokens method
            tokens = torch.tensor([[0, 10, 20, 30, 40, 50]])
            result = runner.add_prefix_suffix_to_tokens(tokens)

            # Should handle the direct attribute
            assert result.shape[1] == tokens.shape[1]


class TestTranscoderHookNormalized:
    """Test support for normalized hooks which are common with transcoders."""

    def test_hook_normalized_support(self):
        """Test that hook_normalized is supported in to_resid_direction."""
        from sae_dashboard.transformer_lens_wrapper import to_resid_direction

        # Create a mock model with hook_normalized
        mock_model = Mock()
        mock_model.activation_config = Mock()
        mock_model.activation_config.primary_hook_point = (
            "blocks.15.ln2.hook_normalized"
        )
        mock_model.hook_layer = 15

        # Create a direction tensor
        direction = torch.randn(10, 768)

        # This should not raise an error and should return the direction unchanged
        # (since normalized hooks are already in residual stream basis)
        result = to_resid_direction(direction, mock_model)
        assert torch.equal(result, direction)
