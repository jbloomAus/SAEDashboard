import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union, cast

import numpy as np
import torch
from transformers import PreTrainedModel

from sae_dashboard.feature_data import FeatureData
from sae_dashboard.neuronpedia.neuronpedia_dashboard import (
    NeuronpediaDashboardActivation,
    NeuronpediaDashboardBatch,
    NeuronpediaDashboardFeature,
)
from sae_dashboard.neuronpedia.neuronpedia_runner_config import (
    NeuronpediaRunnerConfig,
    NeuronpediaVectorRunnerConfig,
)
from sae_dashboard.sae_vis_data import SaeVisData
from sae_dashboard.vector_vis_data import VectorVisData

if TYPE_CHECKING:
    from transformer_lens import HookedTransformer


def _serialize_special_value(value: Any) -> Any:
    if isinstance(value, NeuronpediaDashboardBatch):
        return value.to_dict()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"Unsupported value for serialization: {type(value)!r}")


# Type alias for model types
ModelType = Union["HookedTransformer", PreTrainedModel]


class NpEncoder(json.JSONEncoder):
    def default(self, o: Any):
        try:
            return _serialize_special_value(o)
        except TypeError:
            return super(NpEncoder, self).default(o)


class FeatureProcessor:
    """
    Class for processing feature data.
    """

    @staticmethod
    def round_list(to_round: List[float]) -> List[float]:
        """Round a list of floats to 5 decimal places."""
        return list(np.round(to_round, 5))

    @staticmethod
    def ensure_list(input_value: Any) -> List[Any]:
        """Ensure the input is a list."""
        return [input_value] if not isinstance(input_value, list) else input_value

    @staticmethod
    def to_str_tokens_safe(
        model: ModelType, vocab_dict: Dict[int, str], tokens: Any
    ) -> Any:
        """Convert tokens to string tokens safely."""
        OUT_OF_RANGE_TOKEN = "<|outofrange|>"

        # Get vocab size from the appropriate source
        if hasattr(model, "cfg") and hasattr(model.cfg, "d_vocab"):
            # TransformerLens model
            vocab_max_index = model.cfg.d_vocab - 1
        elif hasattr(model, "config") and hasattr(model.config, "vocab_size"):
            # HuggingFace model
            vocab_max_index = model.config.vocab_size - 1
        else:
            # Fallback to vocab_dict size
            vocab_max_index = len(vocab_dict) - 1

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
        model: ModelType,
        vis_data: Union[SaeVisData, VectorVisData],
        np_cfg: Union[NeuronpediaRunnerConfig, NeuronpediaVectorRunnerConfig],
        vocab_dict: Dict[int, str],
        original_vectors: Optional[torch.Tensor] = None,
        deterministic_json: bool = False,
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
        if isinstance(vis_data, VectorVisData):
            data_dict = vis_data.vector_data_dict
        else:  # SaeVisData
            data_dict = vis_data.feature_data_dict

        features_outputs = NeuronpediaConverter._process_features(
            model,
            data_dict,
            np_cfg,
            vocab_dict,
            original_vectors,
        )
        batch_data = NeuronpediaConverter._create_batch_data(
            np_cfg, features_outputs
        ).to_dict()
        return NeuronpediaConverter.encode_batch_payload(
            batch_data,
            deterministic_json=deterministic_json,
        )

    @staticmethod
    def encode_batch_payload(
        batch_data: dict[str, Any],
        *,
        deterministic_json: bool = False,
    ) -> str:
        """Serialize a ready Neuronpedia batch payload.

        This keeps the preserved baseline stdlib serializer reusable for golden-batch
        parity checks and serializer-only timing measurements without reconstructing
        the full `SaeVisData` input graph.
        """

        return json.dumps(batch_data, cls=NpEncoder)

    @staticmethod
    def convert_preserved_snapshot_to_np_json(
        snapshot: dict[str, Any],
        *,
        deterministic_json: bool = False,
    ) -> str:
        """Replay full converter timing from a preserved runner snapshot.

        The preserved snapshot stores the converter inputs at the runner boundary so
        `convert_to_np_json(...)` can be timed offline against a fixed live batch
        without re-running feature generation.
        """

        model_d_vocab = int(snapshot["model_d_vocab"])
        model_stub = SimpleNamespace(cfg=SimpleNamespace(d_vocab=model_d_vocab))
        feature_data_dict = snapshot.get("feature_data_dict")
        if feature_data_dict is None:
            feature_data = snapshot["feature_data"]
            feature_data_dict = feature_data.feature_data_dict
        vis_data_stub = SimpleNamespace(feature_data_dict=feature_data_dict)
        return NeuronpediaConverter.convert_to_np_json(
            model=cast(Any, model_stub),
            vis_data=cast(Any, vis_data_stub),
            np_cfg=snapshot["runner_cfg"],
            vocab_dict=snapshot["vocab_dict"],
            deterministic_json=deterministic_json,
        )

    @staticmethod
    def _process_features(
        model: ModelType,
        data_dict: Dict[int, FeatureData],  # Update to use data_dict directly
        np_cfg: Union[NeuronpediaRunnerConfig, NeuronpediaVectorRunnerConfig],
        vocab_dict: Dict[int, str],
        original_vectors: Optional[torch.Tensor] = None,
    ) -> List[NeuronpediaDashboardFeature]:
        """Process all features and create NeuronpediaDashboardFeature objects."""
        features_outputs = []
        for feature_index, feature_data in data_dict.items():
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
                feature_output,
                feature_data,
                model,
                vocab_dict,
                trim_trailing_pad_tokens=not (
                    getattr(np_cfg, "dashboard_output_format", None) == "legacy_json"
                    and getattr(np_cfg, "sequence_selection_backend", None) == "legacy"
                ),
            )
            NeuronpediaConverter._process_feature_decoder_weight_dist(
                feature_output, feature_data
            )

            feature_output.n_prompts_total = np_cfg.n_prompts_total
            feature_output.n_tokens_in_prompt = np_cfg.n_tokens_in_prompt
            feature_output.dataset = np_cfg.huggingface_dataset_path

            if original_vectors is not None:
                feature_output.vector = original_vectors[feature_index].tolist()

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
        model: ModelType,
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
        model: ModelType,
        vocab_dict: Dict[int, str],
        *,
        trim_trailing_pad_tokens: bool = True,
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
                        feature_output.feature_index,
                        trim_trailing_pad_tokens=trim_trailing_pad_tokens,
                    )
                    activations.append(activation)

        feature_output.activations = activations

    @staticmethod
    def _parse_sequence_group_title(title: str) -> tuple[float, float, float]:
        """Parse the sequence group title to extract bin information."""
        bin_min, bin_max, bin_contains = 0.0, 0.0, 0.0
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
    def _trim_trailing_pad_tokens(
        token_ids: list[int],
        values: list[float],
        dfa_values: Optional[list[float]],
        pad_token_id: Optional[int],
    ) -> tuple[list[int], list[float], Optional[list[float]]]:
        """Trim trailing pad-token suffixes from exported activation payloads."""
        if pad_token_id is None or not token_ids:
            return token_ids, values, dfa_values

        trimmed_len = len(token_ids)
        while trimmed_len > 0 and token_ids[trimmed_len - 1] == pad_token_id:
            trimmed_len -= 1

        if trimmed_len == len(token_ids):
            return token_ids, values, dfa_values

        trimmed_dfa_values = (
            dfa_values[:trimmed_len] if dfa_values is not None else None
        )
        return token_ids[:trimmed_len], values[:trimmed_len], trimmed_dfa_values

    @staticmethod
    def _create_activation(
        sequence: Any,
        bin_min: float,
        bin_max: float,
        bin_contains: float,
        feature_data: FeatureData,
        model: ModelType,
        vocab_dict: Dict[int, str],
        feature_index: int,
        activation_thresholds: Optional[dict[int, float | int]] = None,
        trim_trailing_pad_tokens: bool = True,
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

        token_ids = list(sequence.token_ids)
        activation_values = FeatureProcessor.round_list(sequence.feat_acts)
        if activation_thresholds is not None:
            threshold = activation_thresholds[feature_index]
            activation_values = [
                v if v >= threshold else 0.0 for v in activation_values
            ]

        if trim_trailing_pad_tokens:
            pad_token_id = getattr(
                getattr(model, "tokenizer", None), "pad_token_id", None
            )
            token_ids, activation_values, activation.dfa_values = (
                NeuronpediaConverter._trim_trailing_pad_tokens(
                    token_ids,
                    activation_values,
                    activation.dfa_values,
                    pad_token_id,
                )
            )

        activation.tokens = [
            FeatureProcessor.to_str_tokens_safe(model, vocab_dict, token_id)
            for token_id in token_ids
        ]
        activation.values = activation_values

        activation.qualifying_token_index = sequence.qualifying_token_index - 1

        return activation

    @staticmethod
    def _create_batch_data(
        np_cfg: Union[NeuronpediaRunnerConfig, NeuronpediaVectorRunnerConfig],
        features_outputs: List[NeuronpediaDashboardFeature],
    ) -> NeuronpediaDashboardBatch:
        """Create a NeuronpediaDashboardBatch object from processed features."""
        batch_data = NeuronpediaDashboardBatch()

        if isinstance(np_cfg, NeuronpediaRunnerConfig):
            # Handle SAE case
            if np_cfg.model_id is not None and np_cfg.layer is not None:
                batch_data.model_id = np_cfg.model_id
                batch_data.layer = np_cfg.layer
            batch_data.sae_set = (
                np_cfg.sae_set if not np_cfg.np_set_name else np_cfg.np_set_name
            )
            if np_cfg.np_sae_id_suffix is not None:
                batch_data.sae_id_suffix = np_cfg.np_sae_id_suffix
        else:
            # Handle Vector case
            if np_cfg.model_id is not None and np_cfg.layer is not None:
                batch_data.model_id = np_cfg.model_id
                batch_data.layer = np_cfg.layer
            # For vectors, we'll use the vector names if provided, otherwise a default name
            vector_set_name = (
                f"vector_set_{','.join(np_cfg.vector_names)}"
                if np_cfg.vector_names
                else "vector_set"
            )
            batch_data.sae_set = vector_set_name
            # No sae_id_suffix needed for vectors

        batch_data.features = features_outputs
        return batch_data
