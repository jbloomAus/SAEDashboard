from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np

EQUAL_VALUE_TOLERANCE = 0.1


# function to check that each value in a list is a float
def check_list_floats(li: list[float]):
    for i in range(len(li)):
        if not isinstance(li[i], float):
            return False
    return True


def equalish(a: Any, b: Any, tol: float = EQUAL_VALUE_TOLERANCE):
    assert type(a) == type(b), f"types do not match: {type(a)} and {type(b)}"

    if (
        isinstance(a, list)
        and isinstance(b, list)
        and check_list_floats(a)
        and check_list_floats(b)
    ):
        close = np.allclose(a, b, tol)
        if not close:
            print(f"Does not match within tolerance: {a} and {b} with tolerance {tol}")
        return close
    elif isinstance(a, float) and isinstance(b, float):
        return abs(a - b) < tol
    else:
        return a == b


@dataclass
class NeuronpediaDashboardActivation:

    def __init__(
        self,
        bin_min: float = 0,
        bin_max: float = 0,
        bin_contains: float = 0,
        tokens: list[str] = [],
        values: list[float] = [],
        qualifying_token_index: int = 0,
        dfa_values: Optional[List[float]] = None,
        dfa_maxValue: Optional[float] = None,
        dfa_targetIndex: Optional[int] = None,
    ):
        self.bin_min = bin_min
        self.bin_max = bin_max
        self.bin_contains = bin_contains
        self.tokens = tokens
        self.values = values
        self.qualifying_token_index = qualifying_token_index
        self.dfa_values = dfa_values
        self.dfa_maxValue = dfa_maxValue
        self.dfa_targetIndex = dfa_targetIndex

    def __eq__(self, other: Any):
        if equalish(self.bin_min, other.bin_min) is False:
            print(f"bin_min does not match: {self.bin_min} and {other.bin_min}")
            return False
        if equalish(self.bin_max, other.bin_max) is False:
            print(f"bin_max does not match: {self.bin_max} and {other.bin_max}")
            return False
        if equalish(self.bin_contains, other.bin_contains, 0.001) is False:
            print(
                f"bin_contains does not match: {self.bin_contains} and {other.bin_contains}"
            )
            return False
        if self.tokens != other.tokens:
            print(f"tokens does not match: {self.tokens} and {other.tokens}")
            return False
        if equalish(self.values, other.values, 0.5) is False:
            print(f"values does not match: {self.values} and {other.values}")
            return False
        return True

    def to_dict(self):
        res = {
            "bin_min": self.bin_min,
            "bin_max": self.bin_max,
            "bin_contains": self.bin_contains,
            "tokens": self.tokens,
            "values": self.values,
            "qualifying_token_index": self.qualifying_token_index,
        }
        if self.dfa_values is not None:
            res["dfa_values"] = self.dfa_values
        if self.dfa_maxValue is not None:
            res["dfa_maxValue"] = self.dfa_maxValue
        if self.dfa_targetIndex is not None:
            res["dfa_targetIndex"] = self.dfa_targetIndex

        return res


@dataclass
class NeuronpediaDashboardFeature:

    def __init__(
        self,
        feature_index: int = 0,
        neuron_alignment_indices: list[int] = [],
        neuron_alignment_values: list[float] = [],
        neuron_alignment_l1: list[float] = [],
        correlated_neurons_indices: list[int] = [],
        correlated_neurons_l1: list[float] = [],
        correlated_neurons_pearson: list[float] = [],
        correlated_features_indices: list[int] = [],
        correlated_features_l1: list[float] = [],
        correlated_features_pearson: list[float] = [],
        neg_str: list[str] = [],
        neg_values: list[float] = [],
        pos_str: list[str] = [],
        pos_values: list[float] = [],
        frac_nonzero: float = 0,
        freq_hist_data_bar_values: list[float] = [],
        freq_hist_data_bar_heights: list[float] = [],
        logits_hist_data_bar_heights: list[float] = [],
        logits_hist_data_bar_values: list[float] = [],
        n_prompts_total: int = 0,
        n_tokens_in_prompt: int = 0,
        dataset: str = "",
        activations: list[dict[str, Any]] = [],
        decoder_weights_dist: list[float] = [],
    ):
        self.feature_index = feature_index
        self.neuron_alignment_indices = neuron_alignment_indices
        self.neuron_alignment_values = neuron_alignment_values
        self.neuron_alignment_l1 = neuron_alignment_l1
        self.correlated_neurons_indices = correlated_neurons_indices
        self.correlated_neurons_l1 = correlated_neurons_l1
        self.correlated_neurons_pearson = correlated_neurons_pearson
        self.correlated_features_indices = correlated_features_indices
        self.correlated_features_l1 = correlated_features_l1
        self.correlated_features_pearson = correlated_features_pearson
        self.neg_str = neg_str
        self.neg_values = neg_values
        self.pos_str = pos_str
        self.pos_values = pos_values
        self.frac_nonzero = frac_nonzero
        self.freq_hist_data_bar_values = freq_hist_data_bar_values
        self.freq_hist_data_bar_heights = freq_hist_data_bar_heights
        self.logits_hist_data_bar_heights = logits_hist_data_bar_heights
        self.logits_hist_data_bar_values = logits_hist_data_bar_values
        self.n_prompts_total = n_prompts_total
        self.n_tokens_in_prompt = n_tokens_in_prompt
        self.dataset = dataset
        self.activations: list[NeuronpediaDashboardActivation] = []
        self.decoder_weights_dist = decoder_weights_dist
        for activation in activations:
            self.activations.append(NeuronpediaDashboardActivation(**activation))

    def __eq__(self, other: Any):
        if self.feature_index != other.feature_index:
            print(
                f"feature_index does not match: {self.feature_index} and {other.feature_index}"
            )
            return False
        if self.neuron_alignment_indices != other.neuron_alignment_indices:
            print(
                f"neuron_alignment_indices does not match: {self.neuron_alignment_indices} and {other.neuron_alignment_indices}"
            )
            return False
        if (
            equalish(self.neuron_alignment_values, other.neuron_alignment_values)
            is False
        ):
            print(equalish(self.neuron_alignment_values, other.neuron_alignment_values))
            print(
                f"neuron_alignment_values does not match: {self.neuron_alignment_values} and {other.neuron_alignment_values}"
            )
            return False
        if equalish(self.neuron_alignment_l1, other.neuron_alignment_l1) is False:
            print(
                f"neuron_alignment_l1 does not match: {self.neuron_alignment_l1} and {other.neuron_alignment_l1}"
            )
            return False

        if self.neg_str != other.neg_str:
            print(f"neg_str does not match: {self.neg_str} and {other.neg_str}")
            return False
        if equalish(self.neg_values, other.neg_values) is False:
            print(
                f"neg_values does not match: {self.neg_values} and {other.neg_values}"
            )
            return False
        if self.pos_str != other.pos_str:
            print(f"pos_str does not match: {self.pos_str} and {other.pos_str}")
            return False
        if equalish(self.pos_values, other.pos_values) is False:
            print(
                f"pos_values does not match: {self.pos_values} and {other.pos_values}"
            )
            return False
        if equalish(self.frac_nonzero, other.frac_nonzero, 0.001) is False:
            print(
                f"frac_nonzero does not match: {self.frac_nonzero} and {other.frac_nonzero}"
            )
            return False
        if (
            equalish(self.freq_hist_data_bar_values, other.freq_hist_data_bar_values)
            is False
        ):
            print(
                f"freq_hist_data_bar_values does not match: {self.freq_hist_data_bar_values} and {other.freq_hist_data_bar_values}"
            )
            return False
        if self.freq_hist_data_bar_heights != other.freq_hist_data_bar_heights:
            print(
                f"freq_hist_data_bar_heights does not match: {self.freq_hist_data_bar_heights} and {other.freq_hist_data_bar_heights}"
            )
            return False
        if self.logits_hist_data_bar_heights != other.logits_hist_data_bar_heights:
            print(
                f"logits_hist_data_bar_heights does not match: {self.logits_hist_data_bar_heights} and {other.logits_hist_data_bar_heights}"
            )
            return False
        if (
            equalish(
                self.logits_hist_data_bar_values, other.logits_hist_data_bar_values
            )
            is False
        ):
            print(
                f"logits_hist_data_bar_values does not match: {self.logits_hist_data_bar_values} and {other.logits_hist_data_bar_values}"
            )
            return False
        if self.n_prompts_total != other.n_prompts_total:
            print(
                f"n_prompts_total does not match: {self.n_prompts_total} and {other.n_prompts_total}"
            )
            return False
        if self.n_tokens_in_prompt != other.n_tokens_in_prompt:
            print(
                f"n_tokens_in_prompt does not match: {self.n_tokens_in_prompt} and {other.n_tokens_in_prompt}"
            )
            return False
        if self.dataset != other.dataset:
            print(f"dataset does not match: {self.dataset} and {other.dataset}")
            return False
        for i, activation in enumerate(self.activations):
            if activation != other.activations[i]:
                print("".join(other.activations[i].tokens))
                print(" ==================================== ")
                print("".join(activation.tokens))
                print(
                    f"activation {i} does not match: {activation} and {other.activations[i]}"
                )
                return False
        if self.decoder_weights_dist != other.decoder_weights_dist:
            print(
                f"decoder_weights_dist does not match: {self.decoder_weights_dist} and {other.decoder_weights_dist}"
            )
            return False
        return True

    def to_dict(self):
        return {
            "feature_index": self.feature_index,
            "neuron_alignment_indices": self.neuron_alignment_indices,
            "neuron_alignment_values": self.neuron_alignment_values,
            "neuron_alignment_l1": self.neuron_alignment_l1,
            "correlated_neurons_indices": self.correlated_neurons_indices,
            "correlated_neurons_l1": self.correlated_neurons_l1,
            "correlated_neurons_pearson": self.correlated_neurons_pearson,
            "correlated_features_indices": self.correlated_features_indices,
            "correlated_features_l1": self.correlated_features_l1,
            "correlated_features_pearson": self.correlated_features_pearson,
            "neg_str": self.neg_str,
            "neg_values": self.neg_values,
            "pos_str": self.pos_str,
            "pos_values": self.pos_values,
            "frac_nonzero": self.frac_nonzero,
            "freq_hist_data_bar_values": self.freq_hist_data_bar_values,
            "freq_hist_data_bar_heights": self.freq_hist_data_bar_heights,
            "logits_hist_data_bar_heights": self.logits_hist_data_bar_heights,
            "logits_hist_data_bar_values": self.logits_hist_data_bar_values,
            "n_prompts_total": self.n_prompts_total,
            "n_tokens_in_prompt": self.n_tokens_in_prompt,
            "dataset": self.dataset,
            "decoder_weights_dist": self.decoder_weights_dist,
            "activations": [activation.to_dict() for activation in self.activations],
        }


# TODO: just add the NPRunnerConfig instead


@dataclass
class NeuronpediaDashboardBatch:

    def __init__(
        self,
        model_id: str = "",
        layer: int = 0,
        sae_set: str = "",
        features: list[dict[str, Any]] = [],
        # settings: NeuronpediaDashboardSettings = NeuronpediaDashboardSettings(),
    ):
        self.model_id = model_id
        self.layer = layer
        self.sae_set = sae_set
        self.features: list[NeuronpediaDashboardFeature] = []
        for feature in features:
            self.features.append(NeuronpediaDashboardFeature(**feature))
        # self.settings = settings

    def __eq__(self, other: Any):
        if self.model_id != other.model_id:
            print(f"model_id does not match: {self.model_id} and {other.model_id}")
            return False
        if self.layer != other.layer:
            print(f"layer does not match: {self.layer} and {other.layer}")
            return False
        if self.sae_set != other.sae_set:
            print(f"sae_set does not match: {self.sae_set} and {other.sae_set}")
            return False
        for i, feature in enumerate(self.features):
            if feature != other.features[i]:
                print(
                    f"feature {feature.feature_index} does not match: {feature} and {other.features[i]}"
                )
                return False
        # if self.settings != other.settings:
        #     print(f"settings does not match: {self.settings} and {other.settings}")
        #     return False
        return True

    def to_dict(self):
        return {
            "model_id": self.model_id,
            "layer": self.layer,
            "sae_set": self.sae_set,
            "features": [feature.to_dict() for feature in self.features],
            # "settings": self.settings.to_dict(),
        }
