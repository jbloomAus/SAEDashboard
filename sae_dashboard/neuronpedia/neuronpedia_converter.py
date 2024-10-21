import json
from typing import Any, Dict, List

import numpy as np
from transformer_lens import HookedTransformer

from sae_dashboard.feature_data import FeatureData
from sae_dashboard.neuronpedia.neuronpedia_dashboard import (
    NeuronpediaDashboardActivation,
    NeuronpediaDashboardBatch,
    NeuronpediaDashboardFeature,
)
from sae_dashboard.neuronpedia.neuronpedia_runner_config import NeuronpediaRunnerConfig
from sae_dashboard.sae_vis_data import SaeVisData


class NpEncoder(json.JSONEncoder):
    def default(self, o: Any):
        if isinstance(o, NeuronpediaDashboardBatch):
            return o.to_dict()
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super(NpEncoder, self).default(o)


class FeatureProcessor:
    """
    Class for processing feature data.
    """

    @staticmethod
    def round_list(to_round: List[float]) -> List[float]:
        """Round a list of floats to 3 decimal places."""
        return list(np.round(to_round, 3))

    @staticmethod
    def ensure_list(input_value: Any) -> List[Any]:
        """Ensure the input is a list."""
        return [input_value] if not isinstance(input_value, list) else input_value

    @staticmethod
    def to_str_tokens_safe(
        model: HookedTransformer, vocab_dict: Dict[int, str], tokens: Any
    ) -> Any:
        """Convert tokens to string tokens safely."""
        OUT_OF_RANGE_TOKEN = "<|outofrange|>"
        vocab_max_index = model.cfg.d_vocab - 1

        if isinstance(tokens, int):
            return (
                OUT_OF_RANGE_TOKEN if tokens > vocab_max_index else vocab_dict[tokens]
            )

        if isinstance(tokens, list):
            tokens = np.array(tokens)

        str_tokens = [
            vocab_dict[t] if t <= vocab_max_index else OUT_OF_RANGE_TOKEN
            for t in tokens.flatten()
        ]

        return np.reshape(str_tokens, tokens.shape).tolist()


class NeuronpediaConverter:
    """
    Class for converting SaeVisData to Neuronpedia format.
    """

    @staticmethod
    def convert_to_np_json(
        model: HookedTransformer,
        sae_data: SaeVisData,
        np_cfg: NeuronpediaRunnerConfig,
        vocab_dict: Dict[int, str],
    ) -> str:
        """
        Convert SaeVisData to Neuronpedia JSON format.

        Args:
            sae_data (SaeVisData): The SAE visualization data.
            np_cfg (NeuronpediaRunnerConfig): Configuration for Neuronpedia runner.
            vocab_dict (Dict[int, str]): Dictionary mapping token IDs to strings.

        Returns:
            str: JSON string representation of the feature data.
        """
        features_outputs = NeuronpediaConverter._process_features(
            model, sae_data, np_cfg, vocab_dict
        )
        batch_data = NeuronpediaConverter._create_batch_data(np_cfg, features_outputs)
        return json.dumps(batch_data, cls=NpEncoder)

    @staticmethod
    def _process_features(
        model: HookedTransformer,
        sae_data: SaeVisData,
        np_cfg: NeuronpediaRunnerConfig,
        vocab_dict: Dict[int, str],
    ) -> List[NeuronpediaDashboardFeature]:
        """Process all features and create NeuronpediaDashboardFeature objects."""
        features_outputs = []
        for feature_index, feature_data in sae_data.feature_data_dict.items():
            feature_output = NeuronpediaDashboardFeature()
            feature_output.feature_index = feature_index

            NeuronpediaConverter._process_feature_tables(feature_output, feature_data)
            NeuronpediaConverter._process_feature_logits(
                feature_output, feature_data, model, vocab_dict
            )
            NeuronpediaConverter._process_feature_histograms(
                feature_output, feature_data
            )
            NeuronpediaConverter._process_feature_activations(
                feature_output, feature_data, model, vocab_dict
            )
            NeuronpediaConverter._process_feature_decoder_weight_dist(
                feature_output, feature_data
            )

            feature_output.n_prompts_total = np_cfg.n_prompts_total
            feature_output.n_tokens_in_prompt = np_cfg.n_tokens_in_prompt
            feature_output.dataset = np_cfg.huggingface_dataset_path

            features_outputs.append(feature_output)
        return features_outputs

    @staticmethod
    def _process_feature_tables(
        feature_output: NeuronpediaDashboardFeature, feature_data: FeatureData
    ) -> None:
        """Process feature tables data and update the feature output."""
        if feature_data.feature_tables_data:
            feature_output.neuron_alignment_indices = (
                feature_data.feature_tables_data.neuron_alignment_indices
            )
            feature_output.neuron_alignment_values = FeatureProcessor.round_list(
                feature_data.feature_tables_data.neuron_alignment_values
            )
            feature_output.neuron_alignment_l1 = FeatureProcessor.round_list(
                feature_data.feature_tables_data.neuron_alignment_l1
            )
            feature_output.correlated_neurons_indices = (
                feature_data.feature_tables_data.correlated_neurons_indices
            )
            feature_output.correlated_neurons_l1 = FeatureProcessor.round_list(
                feature_data.feature_tables_data.correlated_neurons_cossim
            )
            feature_output.correlated_neurons_pearson = FeatureProcessor.round_list(
                feature_data.feature_tables_data.correlated_neurons_pearson
            )
            feature_output.correlated_features_indices = (
                feature_data.feature_tables_data.correlated_features_indices
            )
            feature_output.correlated_features_l1 = FeatureProcessor.round_list(
                feature_data.feature_tables_data.correlated_features_cossim
            )
            feature_output.correlated_features_pearson = FeatureProcessor.round_list(
                feature_data.feature_tables_data.correlated_features_pearson
            )

    @staticmethod
    def _process_feature_logits(
        feature_output: NeuronpediaDashboardFeature,
        feature_data: FeatureData,
        model: HookedTransformer,
        vocab_dict: Dict[int, str],
    ) -> None:
        """Process feature logits data and update the feature output."""
        top_logits = FeatureProcessor.round_list(
            feature_data.logits_table_data.top_logits
        )
        bottom_logits = FeatureProcessor.round_list(
            feature_data.logits_table_data.bottom_logits
        )

        feature_output.neg_str = FeatureProcessor.ensure_list(
            FeatureProcessor.to_str_tokens_safe(
                model, vocab_dict, feature_data.logits_table_data.bottom_token_ids
            )
        )
        feature_output.neg_values = bottom_logits
        feature_output.pos_str = FeatureProcessor.ensure_list(
            FeatureProcessor.to_str_tokens_safe(
                model, vocab_dict, feature_data.logits_table_data.top_token_ids
            )
        )
        feature_output.pos_values = top_logits

    @staticmethod
    def _process_feature_histograms(
        feature_output: NeuronpediaDashboardFeature, feature_data: FeatureData
    ) -> None:
        """Process feature histogram data and update the feature output."""
        if feature_data.acts_histogram_data.title:
            feature_output.frac_nonzero = (
                float(
                    feature_data.acts_histogram_data.title.split(" = ")[1].split("%")[0]
                )
                / 100
            )
        else:
            feature_output.frac_nonzero = 0

        freq_hist_data = feature_data.acts_histogram_data
        feature_output.freq_hist_data_bar_values = FeatureProcessor.round_list(
            freq_hist_data.bar_values
        )
        feature_output.freq_hist_data_bar_heights = FeatureProcessor.round_list(
            freq_hist_data.bar_heights
        )

        logits_hist_data = feature_data.logits_histogram_data
        feature_output.logits_hist_data_bar_heights = FeatureProcessor.round_list(
            logits_hist_data.bar_heights
        )
        feature_output.logits_hist_data_bar_values = FeatureProcessor.round_list(
            logits_hist_data.bar_values
        )

    @staticmethod
    def _process_feature_decoder_weight_dist(
        feature_output: NeuronpediaDashboardFeature,
        feature_data: FeatureData,
    ) -> None:
        """Process feature logits data and update the feature output."""
        if feature_data.decoder_weights_data:
            feature_output.decoder_weights_dist = (
                feature_data.decoder_weights_data.allocation_by_head
            )

    @staticmethod
    def _process_feature_activations(
        feature_output: NeuronpediaDashboardFeature,
        feature_data: FeatureData,
        model: HookedTransformer,
        vocab_dict: Dict[int, str],
    ) -> None:
        """Process feature activations data and update the feature output."""
        activations = []
        for sequence_group in feature_data.sequence_data.seq_group_data:
            bin_min, bin_max, bin_contains = (
                NeuronpediaConverter._parse_sequence_group_title(sequence_group.title)
            )

            for sequence in sequence_group.seq_data:
                if (
                    sequence.top_token_ids is not None
                    and sequence.bottom_token_ids is not None
                    and sequence.top_logits is not None
                    and sequence.bottom_logits is not None
                ):
                    activation = NeuronpediaConverter._create_activation(
                        sequence,
                        bin_min,
                        bin_max,
                        bin_contains,
                        feature_data,
                        model,
                        vocab_dict,
                    )
                    activations.append(activation)

        feature_output.activations = activations

    @staticmethod
    def _parse_sequence_group_title(title: str) -> tuple[float, float, float]:
        """Parse the sequence group title to extract bin information."""
        bin_min, bin_max, bin_contains = 0, 0, 0
        if "TOP ACTIVATIONS" in title:
            bin_min, bin_max, bin_contains = -1, 99, -1
            try:
                bin_max = float(title.split(" = ")[-1])
            except ValueError:
                print(f"Error parsing top activations: {title}")
        elif "INTERVAL" in title:
            try:
                split = title.split("<br>")
                first_split = split[0].split(" ")
                bin_min = float(first_split[1])
                bin_max = float(first_split[-1])
                second_split = split[1].split(" ")
                bin_contains = float(second_split[-1].rstrip("%")) / 100
            except ValueError:
                print(f"Error parsing interval: {title}")
        return bin_min, bin_max, bin_contains

    @staticmethod
    def _create_activation(
        sequence: Any,
        bin_min: float,
        bin_max: float,
        bin_contains: float,
        feature_data: FeatureData,
        model: HookedTransformer,
        vocab_dict: Dict[int, str],
    ) -> NeuronpediaDashboardActivation:
        """Create a NeuronpediaDashboardActivation object from sequence data."""
        activation = NeuronpediaDashboardActivation()
        activation.bin_min = bin_min
        activation.bin_max = bin_max
        activation.bin_contains = bin_contains

        if feature_data.dfa_data is not None:
            if sequence.original_index in feature_data.dfa_data:
                dfa_data = feature_data.dfa_data[sequence.original_index]
                # Round DFA values to three decimal points
                activation.dfa_values = [
                    round(v, 3) for v in dfa_data["dfaValues"][1:]
                ]  # Skip BOS token
                activation.dfa_maxValue = round(
                    max(activation.dfa_values), 3
                )  # Recalculate max to skip BOS token
                activation.dfa_targetIndex = (
                    dfa_data["dfaTargetIndex"] - 1
                )  # Adjust for BOS token
            else:
                # print(
                #     f"Warning: DFA data not found for sequence index {sequence.original_index}"
                # )
                activation.dfa_values = []
                activation.dfa_maxValue = 0
                activation.dfa_targetIndex = -1

        activation.tokens = [
            FeatureProcessor.to_str_tokens_safe(model, vocab_dict, token_id)
            for token_id in sequence.token_ids
        ]
        activation.values = FeatureProcessor.round_list(sequence.feat_acts)

        activation.qualifying_token_index = sequence.qualifying_token_index - 1

        return activation

    @staticmethod
    def _create_batch_data(
        np_cfg: NeuronpediaRunnerConfig,
        features_outputs: List[NeuronpediaDashboardFeature],
    ) -> NeuronpediaDashboardBatch:
        """Create a NeuronpediaDashboardBatch object from processed features."""
        batch_data = NeuronpediaDashboardBatch()

        if np_cfg.model_id is not None and np_cfg.layer is not None:
            batch_data.model_id = np_cfg.model_id
            batch_data.layer = np_cfg.layer
        batch_data.sae_set = (
            np_cfg.sae_set if not np_cfg.np_set_name else np_cfg.np_set_name
        )
        if np_cfg.np_sae_id_suffix is not None:
            batch_data.sae_id_suffix = np_cfg.np_sae_id_suffix
        batch_data.features = features_outputs

        return batch_data
