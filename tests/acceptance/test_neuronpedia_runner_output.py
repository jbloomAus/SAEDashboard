import json
import os
from typing import Any


def load_json_file(file_path: str) -> dict[str, Any]:
    with open(file_path) as f:
        return json.load(f)


def test_neuronpedia_json_output():
    # Assuming the output directory is 'neuronpedia_outputs'
    output_dir = "/Users/curttigges/Projects/SAEDashboard/neuronpedia_outputs/gpt2-small_gpt2-small-hook-z-kk_blocks.5.attn.hook_z"

    # Find all batch files
    batch_files = [
        f
        for f in os.listdir(output_dir)
        if f.startswith("batch-") and f.endswith(".json")
    ]

    assert len(batch_files) > 0, "No batch files found in the output directory"

    for batch_file in batch_files:
        file_path = os.path.join(output_dir, batch_file)
        batch_data = load_json_file(file_path)

        # Check top-level structure
        assert isinstance(batch_data, dict), (
            f"Batch data in {batch_file} is not a dictionary"
        )
        assert "model_id" in batch_data, f"'model_id' missing in {batch_file}"
        assert "layer" in batch_data, f"'layer' missing in {batch_file}"
        assert "sae_set" in batch_data, f"'sae_set' missing in {batch_file}"
        assert "features" in batch_data, f"'features' missing in {batch_file}"

        assert isinstance(batch_data["features"], list), (
            f"'features' in {batch_file} is not a list"
        )

        # Check each feature
        for feature in batch_data["features"]:
            assert isinstance(feature, dict), (
                f"Feature in {batch_file} is not a dictionary"
            )
            assert "feature_index" in feature, (
                f"'feature_index' missing in feature in {batch_file}"
            )
            assert "activations" in feature, (
                f"'activations' missing in feature in {batch_file}"
            )
            assert "neg_str" in feature, f"'neg_str' missing in feature in {batch_file}"
            assert "neg_values" in feature, (
                f"'neg_values' missing in feature in {batch_file}"
            )
            assert "pos_str" in feature, f"'pos_str' missing in feature in {batch_file}"
            assert "pos_values" in feature, (
                f"'pos_values' missing in feature in {batch_file}"
            )
            assert "frac_nonzero" in feature, (
                f"'frac_nonzero' missing in feature in {batch_file}"
            )
            assert "freq_hist_data_bar_heights" in feature, (
                f"'freq_hist_data_bar_heights' missing in feature in {batch_file}"
            )
            assert "freq_hist_data_bar_values" in feature, (
                f"'freq_hist_data_bar_values' missing in feature in {batch_file}"
            )
            assert "logits_hist_data_bar_heights" in feature, (
                f"'logits_hist_data_bar_heights' missing in feature in {batch_file}"
            )
            assert "logits_hist_data_bar_values" in feature, (
                f"'logits_hist_data_bar_values' missing in feature in {batch_file}"
            )
            assert "n_prompts_total" in feature, (
                f"'n_prompts_total' missing in feature in {batch_file}"
            )
            assert "n_tokens_in_prompt" in feature, (
                f"'n_tokens_in_prompt' missing in feature in {batch_file}"
            )
            assert "dataset" in feature, f"'dataset' missing in feature in {batch_file}"
            assert "neuron_alignment_indices" in feature, (
                f"'neuron_alignment_indices' missing in feature in {batch_file}"
            )
            assert "neuron_alignment_values" in feature, (
                f"'neuron_alignment_values' missing in feature in {batch_file}"
            )
            assert "neuron_alignment_l1" in feature, (
                f"'neuron_alignment_l1' missing in feature in {batch_file}"
            )
            assert "correlated_neurons_indices" in feature, (
                f"'correlated_neurons_indices' missing in feature in {batch_file}"
            )
            assert "correlated_neurons_l1" in feature, (
                f"'correlated_neurons_l1' missing in feature in {batch_file}"
            )
            assert "correlated_neurons_pearson" in feature, (
                f"'correlated_neurons_pearson' missing in feature in {batch_file}"
            )
            assert "correlated_features_indices" in feature, (
                f"'correlated_features_indices' missing in feature in {batch_file}"
            )
            assert "correlated_features_l1" in feature, (
                f"'correlated_features_l1' missing in feature in {batch_file}"
            )
            assert "correlated_features_pearson" in feature, (
                f"'correlated_features_pearson' missing in feature in {batch_file}"
            )
            assert "decoder_weights_dist" in feature, (
                f"'decoder_weights_dist' missing in feature in {batch_file}"
            )

            # Check activations
            for activation in feature["activations"]:
                assert isinstance(activation, dict), (
                    f"Activation in feature in {batch_file} is not a dictionary"
                )
                assert "tokens" in activation, (
                    f"'tokens' missing in activation in {batch_file}"
                )
                assert "values" in activation, (
                    f"'values' missing in activation in {batch_file}"
                )
                assert "bin_min" in activation, (
                    f"'bin_min' missing in activation in {batch_file}"
                )
                assert "bin_max" in activation, (
                    f"'bin_max' missing in activation in {batch_file}"
                )
                assert "bin_contains" in activation, (
                    f"'bin_contains' missing in activation in {batch_file}"
                )

                # Check for DFA data (commented out for now)
                assert "dfa_values" in activation, (
                    f"'dfa_values' missing in activation in {batch_file}"
                )
                assert "dfa_maxValue" in activation, (
                    f"'dfa_maxValue' missing in activation in {batch_file}"
                )
                assert "dfa_targetIndex" in activation, (
                    f"'dfa_targetIndex' missing in activation in {batch_file}"
                )

    print("All batch files passed the structure check.")
