from pathlib import Path
from typing import Dict, List

import einops
import numpy as np
import torch
from jaxtyping import Float, Int
from sae_lens import SAE
from sae_lens.config import DTYPE_MAP as DTYPES
from torch import Tensor, nn
from tqdm.auto import tqdm

from sae_dashboard.dfa_calculator import DFACalculator
from sae_dashboard.sae_vis_data import SaeVisConfig
from sae_dashboard.transformer_lens_wrapper import (
    TransformerLensWrapper,
    to_resid_direction,
)
from sae_dashboard.utils_fns import RollingCorrCoef

Arr = np.ndarray


class FeatureDataGenerator:
    def __init__(
        self,
        cfg: SaeVisConfig,
        tokens: Int[Tensor, "batch seq"],
        model: TransformerLensWrapper,
        encoder: SAE,
    ):
        self.cfg = cfg
        self.model = model
        self.encoder = encoder
        self.token_minibatches = self.batch_tokens(tokens)
        self.dfa_calculator = (
            DFACalculator(model.model, encoder) if cfg.use_dfa else None
        )

        if cfg.use_dfa:
            assert (
                "hook_z" in encoder.cfg.hook_name
            ), f"DFAs are only supported for hook_z, but got {encoder.cfg.hook_name}"

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
        # Create lists to store the feature activations & final values of the residual stream
        all_feat_acts = []
        all_dfa_results = {feature_idx: {} for feature_idx in feature_indices}
        total_prompts = 0

        # Create objects to store the data for computing rolling stats
        corrcoef_neurons = RollingCorrCoef()
        corrcoef_encoder = RollingCorrCoef(indices=feature_indices, with_self=True)

        # Get encoder & decoder directions
        feature_out_dir = self.encoder.W_dec[feature_indices]  # [feats d_autoencoder]
        feature_resid_dir = to_resid_direction(
            feature_out_dir, self.model
        )  # [feats d_model]

        # ! Compute & concatenate together all feature activations & post-activation function values
        for i, minibatch in enumerate(self.token_minibatches):
            minibatch.to(self.cfg.device)
            model_activation_dict = self.get_model_acts(i, minibatch)
            primary_acts = model_activation_dict[
                self.model.activation_config.primary_hook_point
            ].to(
                self.encoder.device
            )  # make sure acts are on the correct device

            # Compute feature activations from this
            with FeatureMaskingContext(self.encoder, feature_indices):
                feature_acts = self.encoder.encode(primary_acts).to(
                    DTYPES[self.cfg.dtype]
                )

            self.update_rolling_coefficients(
                model_acts=primary_acts,
                feature_acts=feature_acts,
                corrcoef_neurons=corrcoef_neurons,
                corrcoef_encoder=corrcoef_encoder,
            )

            # Add these to the lists (we'll eventually concat)
            all_feat_acts.append(feature_acts)

            # Calculate DFA
            if self.cfg.use_dfa and self.dfa_calculator:
                max_value_indices = torch.argmax(feature_acts, dim=1)
                batch_dfa_results = self.dfa_calculator.calculate(
                    model_activation_dict,
                    self.model.hook_layer,
                    feature_indices,
                    max_value_indices,
                )
                for feature_idx, feature_data in batch_dfa_results.items():
                    for prompt_idx, prompt_data in feature_data.items():
                        global_prompt_idx = total_prompts + prompt_idx
                        all_dfa_results[feature_idx][global_prompt_idx] = prompt_data

                total_prompts += len(minibatch)

            # Update the 1st progress bar (fwd passes & getting sequence data dominates the runtime of these computations)
            if progress is not None:
                progress[0].update(1)

            
        all_feat_acts = torch.cat(all_feat_acts, dim=0)

        return (
            all_feat_acts,
            torch.tensor([]),  # all_resid_post, no longer used
            feature_resid_dir,
            feature_out_dir,
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


class FeatureMaskingContext:
    def __init__(self, sae: SAE, feature_idxs: List[int]):
        self.sae = sae
        self.feature_idxs = feature_idxs
        self.original_weight = {}

    def __enter__(self):

        ## W_dec
        self.original_weight["W_dec"] = getattr(self.sae, "W_dec").data.clone()
        # mask the weight
        masked_weight = self.sae.W_dec[self.feature_idxs]
        # set the weight
        setattr(self.sae, "W_dec", nn.Parameter(masked_weight))

        ## W_enc
        # clone the weight.
        self.original_weight["W_enc"] = getattr(self.sae, "W_enc").data.clone()
        # mask the weight
        masked_weight = self.sae.W_enc[:, self.feature_idxs]
        # set the weight
        setattr(self.sae, "W_enc", nn.Parameter(masked_weight))

        if self.sae.cfg.architecture == "standard":

            ## b_enc
            self.original_weight["b_enc"] = getattr(self.sae, "b_enc").data.clone()
            # mask the weight
            masked_weight = self.sae.b_enc[self.feature_idxs]
            # set the weight
            setattr(self.sae, "b_enc", nn.Parameter(masked_weight))

        elif self.sae.cfg.architecture == "jumprelu":

            ## b_enc
            self.original_weight["b_enc"] = getattr(self.sae, "b_enc").data.clone()
            # mask the weight
            masked_weight = self.sae.b_enc[self.feature_idxs]
            # set the weight
            setattr(self.sae, "b_enc", nn.Parameter(masked_weight))

            ## threshold
            self.original_weight["threshold"] = getattr(
                self.sae, "threshold"
            ).data.clone()
            # mask the weight
            masked_weight = self.sae.threshold[self.feature_idxs]
            # set the weight
            setattr(self.sae, "threshold", nn.Parameter(masked_weight))

        elif self.sae.cfg.architecture == "gated":

            ## b_gate
            self.original_weight["b_gate"] = getattr(self.sae, "b_gate").data.clone()
            # mask the weight
            masked_weight = self.sae.b_gate[self.feature_idxs]
            # set the weight
            setattr(self.sae, "b_gate", nn.Parameter(masked_weight))

            ## r_mag
            self.original_weight["r_mag"] = getattr(self.sae, "r_mag").data.clone()
            # mask the weight
            masked_weight = self.sae.r_mag[self.feature_idxs]
            # set the weight
            setattr(self.sae, "r_mag", nn.Parameter(masked_weight))

            ## b_mag
            self.original_weight["b_mag"] = getattr(self.sae, "b_mag").data.clone()
            # mask the weight
            masked_weight = self.sae.b_mag[self.feature_idxs]
            # set the weight
            setattr(self.sae, "b_mag", nn.Parameter(masked_weight))
        else:
            raise (ValueError("Invalid architecture"))

        return self

    def __exit__(self, exc_type, exc_value, traceback):  # type: ignore

        # set everything back to normal
        for key, value in self.original_weight.items():
            setattr(self.sae, key, nn.Parameter(value))
