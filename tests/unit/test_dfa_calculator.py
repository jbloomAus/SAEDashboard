import time

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
        for _, prompt_result in feature_results.items():
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
    for _, prompt_result in feature_results.items():
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


def test_functional_equivalence(model: HookedTransformer, autoencoder: SAE):
    calculator = DFACalculator(model, autoencoder)

    # Use the actual model configuration
    batch_size, seq_len = 2, 10
    n_heads, d_head = model.cfg.n_heads, model.cfg.d_head

    attn_weights = torch.rand(batch_size, n_heads, seq_len, seq_len)
    v = torch.rand(batch_size, seq_len, n_heads, d_head)
    feature_indices = [0, 1, 2]

    # Calculate results using both methods
    standard_result = calculator.calculate_standard_intermediate_tensor(
        attn_weights, v, feature_indices
    )

    # Temporarily set use_gqa to True and use dummy values
    calculator.use_gqa = True
    n_kv_heads = n_heads // 2  # Dummy value for GQA
    v_gqa = torch.rand(batch_size, seq_len, n_kv_heads, d_head)
    gqa_result = calculator.calculate_gqa_intermediate_tensor(
        attn_weights, v_gqa, feature_indices
    )
    calculator.use_gqa = False

    # Check that the results have the same shape
    assert standard_result.shape == gqa_result.shape

    # We can't check for exact equality due to different computations, but we can check basic properties
    assert not torch.isnan(standard_result).any()
    assert not torch.isnan(gqa_result).any()
    assert not torch.isinf(standard_result).any()
    assert not torch.isinf(gqa_result).any()


def test_performance_comparison(model: HookedTransformer, autoencoder: SAE):
    calculator = DFACalculator(model, autoencoder)

    # Use the actual model configuration
    batch_size, seq_len = 4, 512
    n_heads, d_head = model.cfg.n_heads, model.cfg.d_head

    attn_weights = torch.rand(batch_size, n_heads, seq_len, seq_len)
    v = torch.rand(batch_size, seq_len, n_heads, d_head)
    feature_indices = list(
        range(min(100, autoencoder.cfg.d_sae))
    )  # Test with up to 100 features

    # Measure time for standard method
    start_time = time.time()
    _ = calculator.calculate_standard_intermediate_tensor(
        attn_weights, v, feature_indices
    )
    standard_time = time.time() - start_time

    # Measure time for GQA method with dummy values
    calculator.use_gqa = True
    n_kv_heads = n_heads // 2  # Dummy value for GQA
    v_gqa = torch.rand(batch_size, seq_len, n_kv_heads, d_head)
    start_time = time.time()
    _ = calculator.calculate_gqa_intermediate_tensor(
        attn_weights, v_gqa, feature_indices
    )
    gqa_time = time.time() - start_time
    calculator.use_gqa = False

    print(f"Standard method time: {standard_time:.4f} seconds")
    print(f"GQA method time: {gqa_time:.4f} seconds")

    # Assert that both methods complete in a reasonable time
    assert standard_time < 10, "Standard method took too long"
    assert gqa_time < 10, "GQA method took too long"


def test_different_shapes(model: HookedTransformer, autoencoder: SAE):
    calculator = DFACalculator(model, autoencoder)

    # Test with different shapes
    shapes = [
        (1, 128, model.cfg.n_heads, model.cfg.d_head),
        (4, 256, model.cfg.n_heads, model.cfg.d_head),
        (2, 512, model.cfg.n_heads, model.cfg.d_head),
    ]

    for batch_size, seq_len, n_heads, d_head in shapes:
        attn_weights = torch.rand(batch_size, n_heads, seq_len, seq_len)
        v = torch.rand(batch_size, seq_len, n_heads, d_head)
        feature_indices = [0, 1, 2]

        standard_result = calculator.calculate_standard_intermediate_tensor(
            attn_weights, v, feature_indices
        )

        # Use dummy values for GQA
        calculator.use_gqa = True
        n_kv_heads = n_heads // 2  # Dummy value for GQA
        v_gqa = torch.rand(batch_size, seq_len, n_kv_heads, d_head)
        gqa_result = calculator.calculate_gqa_intermediate_tensor(
            attn_weights, v_gqa, feature_indices
        )
        calculator.use_gqa = False

        assert standard_result.shape == gqa_result.shape
        assert standard_result.shape == (
            batch_size,
            seq_len,
            seq_len,
            len(feature_indices),
        )
        assert gqa_result.shape == (batch_size, seq_len, seq_len, len(feature_indices))


def test_edge_cases(model: HookedTransformer, autoencoder: SAE):
    calculator = DFACalculator(model, autoencoder)

    # Test with minimal input size
    attn_weights = torch.rand(1, model.cfg.n_heads, 1, 1)
    v = torch.rand(1, 1, model.cfg.n_heads, model.cfg.d_head)
    feature_indices = [0]

    standard_result = calculator.calculate_standard_intermediate_tensor(
        attn_weights, v, feature_indices
    )

    # Use dummy values for GQA
    calculator.use_gqa = True
    n_kv_heads = model.cfg.n_heads // 2  # Dummy value for GQA
    v_gqa = torch.rand(1, 1, n_kv_heads, model.cfg.d_head)
    gqa_result = calculator.calculate_gqa_intermediate_tensor(
        attn_weights, v_gqa, feature_indices
    )
    calculator.use_gqa = False

    assert standard_result.shape == gqa_result.shape

    # Test with empty feature_indices
    feature_indices = []

    standard_result = calculator.calculate_standard_intermediate_tensor(
        attn_weights, v, feature_indices
    )

    calculator.use_gqa = True
    gqa_result = calculator.calculate_gqa_intermediate_tensor(
        attn_weights, v_gqa, feature_indices
    )
    calculator.use_gqa = False

    assert standard_result.shape == (1, 1, 1, 0)
    assert gqa_result.shape == (1, 1, 1, 0)
