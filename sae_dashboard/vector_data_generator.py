from pathlib import Path
from typing import Dict

import einops
import numpy as np
import torch
from jaxtyping import Float, Int
from sae_lens.config import DTYPE_MAP as DTYPES
from torch import Tensor
from tqdm.auto import tqdm

from sae_dashboard.neuronpedia.vector_set import VectorSet
from sae_dashboard.transformer_lens_wrapper import TransformerLensWrapper
from sae_dashboard.utils_fns import RollingCorrCoef

# from sae_dashboard.dfa_calculator import DFACalculator
from sae_dashboard.vector_vis_data import VectorVisConfig

Arr = np.ndarray


class VectorDataGenerator:
    def __init__(
        self,
        cfg: VectorVisConfig,
        tokens: Int[Tensor, "batch seq"],
        model: TransformerLensWrapper,
        encoder: VectorSet,
    ):
        self.cfg = cfg
        self.model = model
        self.encoder = encoder
        self.token_minibatches = self.batch_tokens(tokens)
        self.dfa_calculator = None  # TODO: implement DFA for vectors
        #   (
        #     DFACalculator(model.model, encoder) if cfg.use_dfa else None
        # )

        if cfg.use_dfa:
            if "hook_z" not in encoder.cfg.hook_name:
                raise ValueError(
                    f"DFAs are only supported for hook_z, but got {encoder.cfg.hook_name}"
                )

    @torch.inference_mode()
    def batch_tokens(
        self, tokens: Int[Tensor, "batch seq"]
    ) -> list[Int[Tensor, "batch seq"]]:
        # Get tokens into minibatches, for the fwd pass
        token_minibatches = (
            (tokens,)
            if self.cfg.minibatch_size_tokens is None
            else tokens.split(self.cfg.minibatch_size_tokens)
        )
        token_minibatches = [tok.to(self.cfg.device) for tok in token_minibatches]

        return token_minibatches

    @torch.inference_mode()
    def get_feature_data(  # type: ignore
        self,
        feature_indices: list[int],
        progress: list[tqdm] | None = None,  # type: ignore
    ):  # type: ignore
        # Create lists to store the vector activations
        all_feat_acts = []
        all_dfa_results = {feature_idx: {} for feature_idx in feature_indices}
        # total_prompts = 0

        # Create objects to store the data for computing rolling stats
        corrcoef_neurons = RollingCorrCoef()
        corrcoef_encoder = RollingCorrCoef(indices=feature_indices, with_self=True)

        # Get the selected vectors
        print(f"feature_indices: {feature_indices}")
        feature_vectors = self.encoder.vectors[feature_indices]  # [n_vectors, d_model]

        # if feature_vectors.dim() > 2:
        #     # Remove any extra dimensions to get shape [n_vectors, d_model]
        #     feature_vectors = feature_vectors.squeeze()

        # Process each minibatch
        for i, minibatch in enumerate(self.token_minibatches):
            minibatch = minibatch.to(self.cfg.device)
            model_activation_dict = self.get_model_acts(i, minibatch)
            primary_acts = model_activation_dict[
                self.model.activation_config.primary_hook_point
            ].to(self.encoder.cfg.device)

            # Simple dot product between activations and selected vectors
            feature_acts = torch.einsum("...d,nd->...n", primary_acts, feature_vectors)
            feature_acts = feature_acts.to(DTYPES[self.cfg.dtype])

            self.update_rolling_coefficients(
                model_acts=primary_acts,
                feature_acts=feature_acts,
                corrcoef_neurons=corrcoef_neurons,
                corrcoef_encoder=corrcoef_encoder,
            )

            all_feat_acts.append(feature_acts)

            # Calculate DFA if enabled
            # if self.cfg.use_dfa and self.dfa_calculator:
            #     max_value_indices = torch.argmax(feature_acts, dim=1)
            #     batch_dfa_results = self.dfa_calculator.calculate(
            #         model_activation_dict,
            #         self.model.hook_layer,
            #         feature_indices,
            #         max_value_indices,
            #     )
            #     for feature_idx, feature_data in batch_dfa_results.items():
            #         for prompt_idx in range(feature_data.shape[0]):
            #             global_prompt_idx = total_prompts + prompt_idx
            #             all_dfa_results[feature_idx][global_prompt_idx] = {
            #                 "dfaValues": feature_data[prompt_idx]["dfa_values"].tolist(),
            #                 "dfaTargetIndex": int(feature_data[prompt_idx]["dfa_target_index"]),
            #                 "dfaMaxValue": float(feature_data[prompt_idx]["dfa_max_value"]),
            #             }

            #     total_prompts += len(minibatch)

            # Update progress if provided
            if progress is not None:
                progress[0].update(1)

        all_feat_acts = torch.cat(all_feat_acts, dim=0)

        return (
            all_feat_acts,
            torch.tensor([]),  # No residual post-activation values for vectors
            feature_vectors,  # The vectors themselves serve as the "residual direction"
            feature_vectors,  # The vectors themselves serve as the "output direction"
            corrcoef_neurons,
            corrcoef_encoder,
            all_dfa_results,
        )

    @torch.inference_mode()
    def get_model_acts(
        self,
        minibatch_index: int,
        minibatch_tokens: torch.Tensor,
        use_cache: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        A function that gets the model activations for a given minibatch of tokens.
        Uses np.memmap for efficient caching.
        """
        if self.cfg.cache_dir is not None:
            cache_path = self.cfg.cache_dir / f"model_activations_{minibatch_index}.pt"
            if use_cache and cache_path.exists():
                activation_dict = load_tensor_dict_torch(cache_path, self.cfg.device)
            else:
                activation_dict = self.model.forward(
                    minibatch_tokens.to("cpu"), return_logits=False
                )
                save_tensor_dict_torch(activation_dict, cache_path)
        else:
            activation_dict = self.model.forward(
                minibatch_tokens.to("cpu"), return_logits=False
            )

        return activation_dict

    @torch.inference_mode()
    def update_rolling_coefficients(
        self,
        model_acts: Float[Tensor, "batch seq d_in"],
        feature_acts: Float[Tensor, "batch seq feats"],
        corrcoef_neurons: RollingCorrCoef | None,
        corrcoef_encoder: RollingCorrCoef | None,
    ) -> None:
        """

        Args:
            model_acts: Float[Tensor, "batch seq d_in"]
                The activations of the model, which the SAE was trained on.
            feature_idx: list[int]
                The features we're computing the activations for. This will be used to index the encoder's weights.
            corrcoef_neurons: Optional[RollingCorrCoef]
                The object storing the minimal data necessary to compute corrcoef between feature activations & neurons.
            corrcoef_encoder: Optional[RollingCorrCoef]
                The object storing the minimal data necessary to compute corrcoef between pairwise feature activations.
        """
        # Update the CorrCoef object between feature activation & neurons
        if corrcoef_neurons is not None:
            corrcoef_neurons.update(
                einops.rearrange(feature_acts, "batch seq feats -> feats (batch seq)"),
                einops.rearrange(model_acts, "batch seq d_in -> d_in (batch seq)"),
            )

        # Update the CorrCoef object between pairwise feature activations
        if corrcoef_encoder is not None:
            corrcoef_encoder.update(
                einops.rearrange(feature_acts, "batch seq feats -> feats (batch seq)"),
                einops.rearrange(feature_acts, "batch seq feats -> feats (batch seq)"),
            )


def save_tensor_dict_torch(tensor_dict: Dict[str, torch.Tensor], filename: Path):
    torch.save(tensor_dict, filename)


def load_tensor_dict_torch(filename: Path, device: str) -> Dict[str, torch.Tensor]:
    return torch.load(
        filename, map_location=torch.device(device)
    )  # Directly load to GPU
