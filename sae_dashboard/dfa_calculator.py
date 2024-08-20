
from typing import Dict, List

import einops
import torch
from sae_lens import SAE
from transformer_lens import HookedTransformer


class DFACalculator:
    def __init__(self, model: HookedTransformer, sae: SAE):
        self.model = model
        self.sae = sae

    def calculate(
        self,
        activations: Dict[str, torch.Tensor],
        layer_num: int,
        feature_indices: List[int],
        max_value_indices: torch.Tensor,
    ):
        """Calculate DFA values for a given layer and set of feature indices."""
        if not feature_indices:
            return {}  # Return an empty dictionary if no indices are provided

        v = activations[f"blocks.{layer_num}.attn.hook_v"]
        attn_weights = activations[f"blocks.{layer_num}.attn.hook_pattern"]

        v_cat = einops.rearrange(
            v, "batch src_pos n_heads d_head -> batch src_pos (n_heads d_head)"
        )

        attn_weights_bcast = einops.repeat(
            attn_weights,
            "batch n_heads dest_pos src_pos -> batch dest_pos src_pos (n_heads d_head)",
            d_head=self.model.cfg.d_head,
        )

        decomposed_z_cat = attn_weights_bcast * v_cat.unsqueeze(1)

        W_enc_selected = self.sae.W_enc[:, feature_indices]  # [d_model, num_indices]

        per_src_pos_dfa = einops.einsum(
            decomposed_z_cat,
            W_enc_selected,
            "batch dest_pos src_pos d_model, d_model num_features -> batch dest_pos src_pos num_features",
        )

        n_prompts, seq_len, _, n_features = per_src_pos_dfa.shape

        # Create indices for advanced indexing
        prompt_indices = torch.arange(n_prompts, device=per_src_pos_dfa.device)[
            :, None, None
        ]
        src_pos_indices = torch.arange(seq_len, device=per_src_pos_dfa.device)[
            None, :, None
        ]
        feature_indices_tensor = torch.arange(
            n_features, device=per_src_pos_dfa.device
        )[None, None, :]

        # Expand max_value_indices to match the shape of other indices
        max_value_indices_expanded = max_value_indices[:, None, :]

        # Correctly index per_src_pos_dfa
        per_src_dfa = per_src_pos_dfa[
            prompt_indices,
            max_value_indices_expanded,
            src_pos_indices,
            feature_indices_tensor,
        ]

        # Calculate max values
        max_values, _ = per_src_dfa.max(dim=1)

        results = {feature_idx: {} for feature_idx in feature_indices}
        for i in range(n_prompts):
            for j, feature_idx in enumerate(feature_indices):
                dfa_values = per_src_dfa[i, :, j].tolist()
                results[feature_idx][i] = {
                    "dfaValues": dfa_values,
                    "dfaTargetIndex": max_value_indices[i, j].item(),
                    "dfaMaxValue": max_values[i, j].item(),
                }

        return results
