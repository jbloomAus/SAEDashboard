"""Integration tests for transcoder functionality."""

import os
from pathlib import Path

import pytest
import torch

from sae_dashboard.neuronpedia.neuronpedia_runner import (
    NeuronpediaRunner,
    NeuronpediaRunnerConfig,
)


@pytest.mark.skipif(
    not torch.cuda.is_available() and not torch.backends.mps.is_available(),
    reason="Requires GPU for integration test",
)
class TestTranscoderIntegration:
    """Integration tests that actually load and run transcoders."""

    def test_transcoder_dashboard_generation(self, tmp_path):  # type: ignore
        """Test end-to-end transcoder dashboard generation."""
        # Configure for minimal test
        config = NeuronpediaRunnerConfig(
            sae_set="gemma-scope-2b-pt-transcoders",
            sae_path="layer_0/width_16k/average_l0_76",  # Use layer 0 with correct L0 value
            np_set_name="test-integration",
            from_local_sae=False,
            huggingface_dataset_path="monology/pile-uncopyrighted",
            sae_dtype="float32",
            model_dtype="float32",
            outputs_dir=str(tmp_path),
            sparsity_threshold=1,
            n_prompts_total=32,  # Very small for testing
            n_tokens_in_prompt=64,  # Smaller context
            n_prompts_in_forward_pass=16,
            n_features_at_a_time=2,  # Just 2 features
            start_batch=0,
            end_batch=1,  # Single batch
            use_wandb=False,
            use_transcoder=True,
        )

        # Create runner and run
        runner = NeuronpediaRunner(config)

        # Verify transcoder was loaded
        assert hasattr(runner.sae.cfg, "hook_name_out")
        assert "transcoder" in str(runner.sae.cfg.architecture).lower()

        # Run dashboard generation
        runner.run()

        # Check outputs were created
        output_dir = Path(runner.cfg.outputs_dir)
        assert output_dir.exists()
        assert (output_dir / "batch-0.json").exists()
        assert (output_dir / "run_settings.json").exists()

        # Load and verify JSON structure
        import json

        with open(output_dir / "batch-0.json") as f:
            data = json.load(f)
            assert "features" in data
            assert len(data["features"]) == 2  # We requested 2 features
            assert data["model_id"] == "gemma-2-2b"
            assert data["layer"] == 0

    def test_transcoder_vs_sae_differences(self, tmp_path):  # type: ignore
        """Test that transcoders are handled differently from standard SAEs."""
        # Test with transcoder
        transcoder_config = NeuronpediaRunnerConfig(
            sae_set="gemma-scope-2b-pt-transcoders",
            sae_path="layer_0/width_16k/average_l0_76",
            np_set_name="test-transcoder",
            from_local_sae=False,
            huggingface_dataset_path="monology/pile-uncopyrighted",
            outputs_dir=str(tmp_path / "transcoder"),
            n_prompts_total=32,
            use_transcoder=True,
            use_wandb=False,
        )

        # Test with standard SAE
        sae_config = NeuronpediaRunnerConfig(
            sae_set="gemma-scope-2b-pt-res",
            sae_path="layer_0/width_16k/average_l0_105",
            np_set_name="test-sae",
            from_local_sae=False,
            huggingface_dataset_path="monology/pile-uncopyrighted",
            outputs_dir=str(tmp_path / "sae"),
            n_prompts_total=32,
            use_transcoder=False,
            use_wandb=False,
        )

        # Create runners
        transcoder_runner = NeuronpediaRunner(transcoder_config)
        sae_runner = NeuronpediaRunner(sae_config)

        # Verify different handling
        # Transcoders should have hook_name_out
        assert hasattr(transcoder_runner.sae.cfg, "hook_name_out")
        assert not hasattr(sae_runner.sae.cfg, "hook_name_out")

        # Both should have hook_name in metadata
        assert transcoder_runner.hook_name
        assert sae_runner.hook_name

        # Architecture should be different
        transcoder_arch = transcoder_runner.sae.cfg.architecture
        if callable(transcoder_arch):
            transcoder_arch = transcoder_arch()
        assert "transcoder" in transcoder_arch.lower()  # type: ignore

        sae_arch = sae_runner.sae.cfg.architecture
        if callable(sae_arch):
            sae_arch = sae_arch()
        assert "transcoder" not in sae_arch.lower()  # type: ignore


@pytest.mark.skipif(
    "GITHUB_ACTIONS" in os.environ, reason="Skip heavy integration tests in CI"
)
class TestTranscoderEdgeCases:
    """Test edge cases and error handling for transcoders."""

    def test_invalid_transcoder_config(self):
        """Test handling of invalid transcoder configurations."""
        with pytest.raises(Exception):  # Could be ImportError or other
            config = NeuronpediaRunnerConfig(
                sae_set="invalid-set",
                sae_path="invalid-path",
                np_set_name="test",
                huggingface_dataset_path="monology/pile-uncopyrighted",
                outputs_dir="test_outputs/",
                use_transcoder=True,
                use_wandb=False,
            )
            _ = NeuronpediaRunner(config)

    def test_transcoder_with_custom_dtype(self, tmp_path):  # type: ignore
        """Test transcoder loading with custom dtype."""
        config = NeuronpediaRunnerConfig(
            sae_set="gemma-scope-2b-pt-transcoders",
            sae_path="layer_0/width_16k/average_l0_76",
            np_set_name="test-dtype",
            from_local_sae=False,
            huggingface_dataset_path="monology/pile-uncopyrighted",
            sae_dtype="float16",  # Test with float16
            model_dtype="float32",
            outputs_dir=str(tmp_path),
            n_prompts_total=16,
            n_features_at_a_time=1,
            end_batch=1,
            use_transcoder=True,
            use_wandb=False,
        )

        runner = NeuronpediaRunner(config)

        # Verify dtype was applied
        assert runner.sae.W_dec.dtype == torch.float16
