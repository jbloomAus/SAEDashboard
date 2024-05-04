from pathlib import Path

import einops
import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor
from tqdm.auto import tqdm

from sae_vis.autoencoder import DTYPES, AutoEncoder
from sae_vis.sae_vis_data import SaeVisConfig
from sae_vis.transformer_lens_wrapper import (
    TransformerLensWrapper,
    to_resid_dir,
)
from sae_vis.utils_fns import (
    RollingCorrCoef,
)

Arr = np.ndarray


class FeatureDataGenerator:
    def __init__(
        self,
        cfg: SaeVisConfig,
        tokens: Int[Tensor, "batch seq"],
        model: TransformerLensWrapper,
        encoder: AutoEncoder,
        encoder_B: AutoEncoder | None = None,
    ):
        self.cfg = cfg
        self.model = model
        self.encoder = encoder
        self.encoder_B = encoder_B
        self.token_minibatches = self.batch_tokens(tokens)

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
    def get_feature_data(
        self,
        feature_indices: list[int],
        progress: list[tqdm] | None = None,
    ):  # type: ignore
        # Create lists to store the feature activations & final values of the residual stream
        all_resid_post = []
        all_feat_acts = []

        # Create objects to store the data for computing rolling stats
        corrcoef_neurons = RollingCorrCoef()
        corrcoef_encoder = RollingCorrCoef(indices=feature_indices, with_self=True)
        corrcoef_encoder_B = RollingCorrCoef() if self.encoder_B is not None else None

        # Get encoder & decoder directions
        feature_out_dir = self.encoder.W_dec[feature_indices]  # [feats d_autoencoder]
        feature_resid_dir = to_resid_dir(feature_out_dir, self.model)  # [feats d_model]

        # ! Compute & concatenate together all feature activations & post-activation function values

        for i, minibatch in enumerate(self.token_minibatches):
            model_acts, residual = self.get_model_acts(i, minibatch)

            # Compute feature activations from this
            feature_acts = self.encoder.get_feature_acts_subset(
                model_acts, feature_indices
            )

            self.update_rolling_coefficients(
                model_acts=model_acts,
                feature_acts=feature_acts,
                corrcoef_neurons=corrcoef_neurons,
                corrcoef_encoder=corrcoef_encoder,
                corrcoef_encoder_B=corrcoef_encoder_B,
            )

            # Add these to the lists (we'll eventually concat)
            all_feat_acts.append(feature_acts)
            all_resid_post.append(residual)

            # Update the 1st progress bar (fwd passes & getting sequence data dominates the runtime of these computations)
            if progress is not None:
                progress[0].update(1)

        all_feat_acts = torch.cat(all_feat_acts, dim=0)
        all_resid_post = torch.cat(
            all_resid_post, dim=0
        )  # TODO: Check if this actually changes on each iteration and if so how to wasting effort.

        return (
            all_feat_acts,
            all_resid_post,
            feature_resid_dir,
            feature_out_dir,
            corrcoef_neurons,
            corrcoef_encoder,
            corrcoef_encoder_B,
        )

    @torch.inference_mode()
    def get_model_acts(
        self, minibatch_index: int, minibatch_tokens: Int[Tensor, "batch seq"]
    ):
        """
        A function that gets the model activations and residuals for a given minibatch of tokens.
        Handles the existence of a caching directory and whether or not activations are already cached.
        """

        if self.cfg.cache_dir is not None:
            # check if the activations are already cached
            cache_path = (
                self.cfg.cache_dir
                / f"model_activations_and_residuals_{minibatch_index}.safetensors"
            )
            if cache_path.exists():
                model, residual = self.get_cached_activation_results(cache_path)
            else:
                # generate and store the results
                residual, model = self.model.forward(
                    minibatch_tokens, return_logits=False
                )
                tensors = {"activations": model, "residual": residual}
                # could also save tokens to avoid needing to provide them above.
                save_file(tensors, cache_path)
        else:
            residual, model = self.model.forward(minibatch_tokens, return_logits=False)

        return model, residual

    @torch.inference_mode()
    def get_cached_activation_results(self, cache_path: Path):
        with safe_open(cache_path, framework="pt", device=str(self.cfg.device)) as f:  # type: ignore
            model_acts = f.get_tensor("activations")
            residual = f.get_tensor("residual")

        model_acts = model_acts.to(DTYPES[self.cfg.dtype])
        residual = residual.to(DTYPES[self.cfg.dtype])
        return model_acts, residual

    @torch.inference_mode()
    def compute_feat_acts(
        self,
        model_acts: Float[Tensor, "batch seq d_in"],
        feature_idx: list[int],
        encoder: AutoEncoder,
    ) -> Float[Tensor, "batch seq feats"]:
        """
        This function computes the feature activations, given a bunch of model data. It also updates the rolling correlation
        coefficient objects, if they're given.

        Args:
            model_acts: Float[Tensor, "batch seq d_in"]
                The activations of the model, which the SAE was trained on.
            feature_idx: list[int]
                The features we're computing the activations for. This will be used to index the encoder's weights.
            encoder: AutoEncoder
                The encoder object, which we use to calculate the feature activations.
            encoder_B: Optional[AutoEncoder]
                The encoder-B object, which we use to calculate the feature activations.
        """
        # Get the feature act direction by indexing encoder.W_enc, and the bias by indexing encoder.b_enc
        feature_act_dir = encoder.W_enc[:, feature_idx]  # (d_in, feats)
        feature_bias = encoder.b_enc[feature_idx]  # (feats,)

        # Calculate & store feature activations (we need to store them so we can get the sequence & histogram vis later)
        x_cent = model_acts - encoder.b_dec
        feat_acts_pre = einops.einsum(
            x_cent, feature_act_dir, "batch seq d_in, d_in feats -> batch seq feats"
        )
        feat_acts = F.relu(feat_acts_pre + feature_bias)

        return feat_acts

    @torch.inference_mode()
    def update_rolling_coefficients(
        self,
        model_acts: Float[Tensor, "batch seq d_in"],
        feature_acts: Float[Tensor, "batch seq feats"],
        corrcoef_neurons: RollingCorrCoef | None,
        corrcoef_encoder: RollingCorrCoef | None,
        corrcoef_encoder_B: RollingCorrCoef | None,
        encoder_B: AutoEncoder | None = None,
    ) -> None:
        """

        Args:
            model_acts: Float[Tensor, "batch seq d_in"]
                The activations of the model, which the SAE was trained on.
            feature_idx: list[int]
                The features we're computing the activations for. This will be used to index the encoder's weights.
            encoder_B: Optional[AutoEncoder]
                The encoder-B object, which we use to calculate the feature activations.
            corrcoef_neurons: Optional[RollingCorrCoef]
                The object storing the minimal data necessary to compute corrcoef between feature activations & neurons.
            corrcoef_encoder: Optional[RollingCorrCoef]
                The object storing the minimal data necessary to compute corrcoef between pairwise feature activations.
            corrcoef_encoder_B: Optional[RollingCorrCoef]
                The object storing minimal data to compute corrcoef between feature activations & encoder-B features.
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

        # Calculate encoder-B feature acts (we don't need to store encoder-B acts; it's just for left-hand feature tables)
        if corrcoef_encoder_B is not None:
            feat_acts_B = encoder_B.get_feature_acts(model_acts)  # type: ignore (we know encoder_B is not None)

            # Update the CorrCoef object between feature activation & encoder-B features
            corrcoef_encoder_B.update(
                einops.rearrange(feature_acts, "batch seq feats -> feats (batch seq)"),
                einops.rearrange(
                    feat_acts_B, "batch seq d_hidden -> d_hidden (batch seq)"
                ),
            )
