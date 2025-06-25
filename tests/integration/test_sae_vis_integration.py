from typing import Callable, Tuple

import pytest
import torch
from sae_lens import SAE, ActivationsStore
from transformer_lens import HookedTransformer

# from sae_dashboard.feature_data import FeatureData
from sae_dashboard.sae_vis_data import SaeVisConfig, SaeVisData
from sae_dashboard.sae_vis_runner import SaeVisRunner
from sae_dashboard.utils_fns import FeatureStatistics, get_tokens


@pytest.fixture
def setup_test_environment() -> (
    Callable[[], Tuple[HookedTransformer, SAE, torch.Tensor]]
):
    def _setup() -> Tuple[HookedTransformer, SAE, torch.Tensor]:
        # Set up a small-scale test environment
        device = "cpu"  # Use CUDA for testing
        model = HookedTransformer.from_pretrained("gpt2-small", device=device)
        sae, _, _ = SAE.from_pretrained(
            release="gpt2-small-hook-z-kk", sae_id="blocks.5.hook_z", device=device
        )
        sae.fold_W_dec_norm()

        # Create a small token dataset
        activations_store = ActivationsStore.from_sae(
            model=model,
            sae=sae,
            streaming=True,
            store_batch_size_prompts=16,
            n_batches_in_buffer=8,
            device=device,
        )

        token_dataset = get_tokens(activations_store, 256)
        insert = model.to_tokens("Stalinists shriek in the ears of the police that")
        token_dataset[0, :13] = insert[0]

        return model, sae, token_dataset

    return _setup


def test_sae_vis_runner_integration(
    setup_test_environment: Callable[[], Tuple[HookedTransformer, SAE, torch.Tensor]],
):
    model, sae, token_dataset = setup_test_environment()

    # Configure SaeVisConfig for testing
    test_feature_idx = list(range(64))  # Test with 16 features
    feature_vis_config = SaeVisConfig(
        hook_point=sae.cfg.hook_name,
        features=test_feature_idx,
        minibatch_size_features=32,
        minibatch_size_tokens=256,
        verbose=False,
        device="cpu",
        # cache_dir=Path("test_activations_cache"),
        dtype="float32",
        use_dfa=True,
    )

    # Run SaeVisRunner
    data = SaeVisRunner(feature_vis_config).run(
        encoder=sae,
        model=model,
        tokens=token_dataset,
    )
    if data.feature_data_dict[15].dfa_data:
        print(data.feature_data_dict[15].dfa_data[0])
        print(data.feature_data_dict.keys())
    # Verify the structure and content of the resulting SaeVisData object
    assert isinstance(data, SaeVisData)
    assert len(data.feature_data_dict) == len(test_feature_idx)

    for feat_idx, feature_data in data.feature_data_dict.items():
        assert feat_idx in test_feature_idx

        # Check feature_tables_data
        assert hasattr(feature_data, "feature_tables_data")
        assert hasattr(feature_data.feature_tables_data, "neuron_alignment_indices")
        assert hasattr(feature_data.feature_tables_data, "neuron_alignment_values")
        assert isinstance(
            feature_data.feature_tables_data.neuron_alignment_indices, list
        )
        assert isinstance(
            feature_data.feature_tables_data.neuron_alignment_values, list
        )

        # Check logits_histogram_data
        assert hasattr(feature_data, "logits_histogram_data")
        assert hasattr(feature_data.logits_histogram_data, "bar_heights")
        assert hasattr(feature_data.logits_histogram_data, "bar_values")
        assert isinstance(feature_data.logits_histogram_data.bar_heights, list)
        assert isinstance(feature_data.logits_histogram_data.bar_values, list)

        # Check acts_histogram_data
        assert hasattr(feature_data, "acts_histogram_data")
        assert hasattr(feature_data.acts_histogram_data, "bar_heights")
        assert hasattr(feature_data.acts_histogram_data, "bar_values")
        assert isinstance(feature_data.acts_histogram_data.bar_heights, list)
        assert isinstance(feature_data.acts_histogram_data.bar_values, list)

        # Check sequence_data
        # assert hasattr(feature_data, 'sequence_data')
        # assert len(feature_data.sequence_data.seq_group_data) > 0
        # for seq_group in feature_data.sequence_data.seq_group_data:
        #     assert hasattr(seq_group, 'seq_data')
        #     assert len(seq_group.seq_data) > 0

        # Check DFA data
        if feature_vis_config.use_dfa:
            print(f"Checking feature {feat_idx} DFA data")
            assert hasattr(feature_data, "dfa_data")
            assert feature_data.dfa_data is not None
            assert isinstance(feature_data.dfa_data, dict)

            for prompt_idx, prompt_dfa in feature_data.dfa_data.items():
                assert isinstance(prompt_idx, int)
                assert isinstance(prompt_dfa, dict)
                assert "dfaValues" in prompt_dfa
                assert "dfaTargetIndex" in prompt_dfa
                assert "dfaMaxValue" in prompt_dfa

                assert isinstance(prompt_dfa["dfaValues"], list)
                assert isinstance(prompt_dfa["dfaTargetIndex"], int)
                assert isinstance(prompt_dfa["dfaMaxValue"], float)

                assert (
                    len(prompt_dfa["dfaValues"]) == token_dataset.shape[1]
                )  # Should match sequence length
                assert 0 <= prompt_dfa["dfaTargetIndex"] < token_dataset.shape[1]
                assert prompt_dfa["dfaMaxValue"] == max(prompt_dfa["dfaValues"])

    # Additional checks for overall structure
    assert hasattr(data, "cfg")
    assert isinstance(data.cfg, SaeVisConfig)
    assert hasattr(data, "feature_stats")
    assert isinstance(data.feature_stats, FeatureStatistics)

    # Check that the number of prompts in DFA data matches the input
    if feature_vis_config.use_dfa:
        for feature_data in data.feature_data_dict.values():
            assert feature_data.dfa_data
            assert len(feature_data.dfa_data) == token_dataset.shape[0]
