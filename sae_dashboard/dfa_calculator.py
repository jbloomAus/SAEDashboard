from typing import Dict, List, Union

import einops
import torch
from sae_lens import SAE
from transformer_lens import ActivationCache, HookedTransformer


class DFACalculator:
    """Calculate DFA values for a given layer and set of feature indices."""

    def __init__(self, model: HookedTransformer, sae: SAE):
        self.model = model
        self.sae = sae
        if (
            hasattr(model.cfg, "n_key_value_heads")
            and model.cfg.n_key_value_heads is not None
            and model.cfg.n_key_value_heads < model.cfg.n_heads
        ):
            print("Using GQA")
            self.use_gqa = True
        else:
            self.use_gqa = False

    def calculate(
        self,
        activations: Union[Dict[str, torch.Tensor], ActivationCache],
        layer_num: int,
        feature_indices: List[int],
        max_value_indices: torch.Tensor,
    ) -> Dict[int, Dict[int, Dict[str, List[float]]]]:
        """Calculate DFA values for a given layer and set of feature indices.

        Args:
            activations: Dictionary of activations for the model.
            layer_num: Layer number.
            feature_indices: List of feature indices.
            max_value_indices: Tensor of max value indices.

        Returns:
            Dictionary of DFA values for each feature index (and for each prompt).
        """
        if not feature_indices:
            return {}  # Return an empty dictionary if no indices are provided

        v = activations[f"blocks.{layer_num}.attn.hook_v"]
        attn_weights = activations[f"blocks.{layer_num}.attn.hook_pattern"]

        if self.use_gqa:
            per_src_pos_dfa = self.calculate_gqa_intermediate_tensor(
                attn_weights, v, feature_indices
            )
        else:
            per_src_pos_dfa = self.calculate_standard_intermediate_tensor(
                attn_weights, v, feature_indices
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

        # Index into per_src_pos_dfa
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

    def calculate_standard_intermediate_tensor(
        self,
        attn_weights: torch.Tensor,
        v: torch.Tensor,
        feature_indices: List[int],
    ) -> torch.Tensor:
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

        return per_src_pos_dfa

    def calculate_gqa_intermediate_tensor(
        self, attn_weights: torch.Tensor, v: torch.Tensor, feature_indices: List[int]
    ) -> torch.Tensor:

        n_query_heads = attn_weights.shape[1]
        n_kv_heads = v.shape[2]
        expansion_factor = n_query_heads // n_kv_heads
        v = v.repeat_interleave(expansion_factor, dim=2)

        v_cat = einops.rearrange(
            v, "batch src_pos n_heads d_head -> batch src_pos (n_heads d_head)"
        )

        W_enc_selected = self.sae.W_enc[:, feature_indices]  # [d_model, num_indices]

        # Initialize the result tensor
        n_prompts, seq_len, _ = v_cat.shape
        n_features = len(feature_indices)
        per_src_pos_dfa = torch.zeros(
            (n_prompts, seq_len, seq_len, n_features), device=v_cat.device
        )

        # Process in chunks
        chunk_size = 32  # Adjust this based on your memory constraints
        for i in range(0, seq_len, chunk_size):
            chunk_end = min(i + chunk_size, seq_len)

            # Process a chunk of destination positions
            attn_weights_chunk = attn_weights[:, :, i:chunk_end, :]
            attn_weights_bcast_chunk = einops.repeat(
                attn_weights_chunk,
                "batch n_heads dest_pos src_pos -> batch dest_pos src_pos (n_heads d_head)",
                d_head=self.model.cfg.d_head,
            )
            decomposed_z_cat_chunk = attn_weights_bcast_chunk * v_cat.unsqueeze(1)

            per_src_pos_dfa_chunk = einops.einsum(
                decomposed_z_cat_chunk,
                W_enc_selected,
                "batch dest_pos src_pos d_model, d_model num_features -> batch dest_pos src_pos num_features",
            )

            per_src_pos_dfa[:, i:chunk_end, :, :] = per_src_pos_dfa_chunk

        return per_src_pos_dfa
