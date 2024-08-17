from pathlib import Path
from typing import Dict, List

import einops
import numpy as np
import torch
from jaxtyping import Float, Int
from sae_lens import SAE
from transformer_lens import HookedTransformer


class DFACalculator:
    def __init__(self, model: HookedTransformer, sae: SAE):
        self.model = model
        self.sae = sae

    def calculate(
        self
        # self,
        # activations: Dict[str, Tensor],
        # layer_num: int,
        # index: int,
        # max_value_index: int,
    ) -> Dict:
        pass
        # v = activations[f"blocks.{layer_num}.hook_v"]
        # z = activations[f"blocks.{layer_num}.hook_z"]
        # attn_weights = activations[f"blocks.{layer_num}.hook_pattern"]

        # # Implement the DFA calculation logic here
        # # ...

        # return {
        #     "dfaValues": dfa_values,
        #     "dfaTargetIndex": max_value_index,
        #     "dfaMaxValue": max(dfa_values),
        # }
