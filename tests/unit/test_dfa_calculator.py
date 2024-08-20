import pytest
import torch
from sae_lens import SAE
from transformer_lens import HookedTransformer

from sae_dashboard.dfa_calculator import DFACalculator


def test_dfa_calculator_initialization(model: HookedTransformer, autoencoder: SAE):
    calculator = DFACalculator(model, autoencoder)
    assert calculator.model == model
    assert calculator.sae == autoencoder


def test_dfa_calculation_shape(
    model: HookedTransformer, autoencoder: SAE, tokens: torch.Tensor
):
    calculator = DFACalculator(model, autoencoder)

    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)

    layer_num = 0
    indices = [0, 1, 2]  # Test with first three features
    max_value_indices = torch.randint(
        0, tokens.shape[1], (tokens.shape[0], len(indices))
    )

    results = calculator.calculate(cache, layer_num, indices, max_value_indices)

    assert len(results) == len(indices)  # One result per feature
    for feature_idx, feature_results in results.items():
        assert feature_idx in indices
        assert len(feature_results) == tokens.shape[0]  # One result per prompt
        for prompt_idx, prompt_result in feature_results.items():
            assert "dfaValues" in prompt_result
            assert "dfaTargetIndex" in prompt_result
            assert "dfaMaxValue" in prompt_result
            assert (
                len(prompt_result["dfaValues"]) == tokens.shape[1]
            )  # Should match sequence length


def test_dfa_calculation_values(
    model: HookedTransformer, autoencoder: SAE, tokens: torch.Tensor
):
    calculator = DFACalculator(model, autoencoder)

    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)

    layer_num = 0
    indices = [0]
    max_value_indices = torch.randint(
        0, tokens.shape[1], (tokens.shape[0], len(indices))
    )

    results = calculator.calculate(cache, layer_num, indices, max_value_indices)

    assert len(results) == 1
    feature_results = results[0]
    assert len(feature_results) == tokens.shape[0]
    for prompt_idx, prompt_result in feature_results.items():
        assert not all(v == 0 for v in prompt_result["dfaValues"])
        assert prompt_result["dfaMaxValue"] == max(prompt_result["dfaValues"])


def test_dfa_calculation_multiple_features(
    model: HookedTransformer, autoencoder: SAE, tokens: torch.Tensor
):
    calculator = DFACalculator(model, autoencoder)

    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)

    layer_num = 0
    indices = [0, 1, 2]
    max_value_indices = torch.randint(
        0, tokens.shape[1], (tokens.shape[0], len(indices))
    )

    results = calculator.calculate(cache, layer_num, indices, max_value_indices)

    assert len(results) == len(indices)
    for feature_idx, feature_results in results.items():
        assert len(feature_results) == tokens.shape[0]
        for prompt_idx, prompt_result in feature_results.items():
            assert (
                prompt_result["dfaTargetIndex"]
                == max_value_indices[prompt_idx, indices.index(feature_idx)].item()
            )


def test_dfa_calculation_different_layers(
    model: HookedTransformer, autoencoder: SAE, tokens: torch.Tensor
):
    calculator = DFACalculator(model, autoencoder)

    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)

    indices = [0]
    max_value_indices = torch.randint(
        0, tokens.shape[1], (tokens.shape[0], len(indices))
    )

    results_layer0 = calculator.calculate(cache, 0, indices, max_value_indices)
    results_layer1 = calculator.calculate(cache, 1, indices, max_value_indices)

    assert results_layer0 != results_layer1


def test_dfa_calculation_edge_cases(
    model: HookedTransformer, autoencoder: SAE, tokens: torch.Tensor
):
    calculator = DFACalculator(model, autoencoder)

    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)

    # Test with empty indices list
    results = calculator.calculate(cache, 0, [], torch.empty(0))
    assert results == {}  # Expect an empty dictionary

    # Test with out of range index
    with pytest.raises(IndexError):
        calculator.calculate(
            cache,
            0,
            [autoencoder.cfg.d_sae],
            torch.randint(0, tokens.shape[1], (tokens.shape[0], 1)),
        )

    # Test with invalid layer number
    with pytest.raises(KeyError):
        calculator.calculate(
            cache,
            model.cfg.n_layers,
            [0],
            torch.randint(0, tokens.shape[1], (tokens.shape[0], 1)),
        )

    # Test with valid inputs
    indices = [0]
    max_value_indices = torch.randint(
        0, tokens.shape[1], (tokens.shape[0], len(indices))
    )
    results = calculator.calculate(cache, 0, indices, max_value_indices)
    assert len(results) == len(indices)
    assert all(
        isinstance(feature_results, dict) for feature_results in results.values()
    )
    assert all(
        len(feature_results) == tokens.shape[0] for feature_results in results.values()
    )


# Add more tests as needed based on your specific implementation and requirements
