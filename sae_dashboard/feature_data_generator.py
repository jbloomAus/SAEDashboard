from pathlib import Path
from typing import List

import einops
import numpy as np
import torch
from jaxtyping import Float, Int
from sae_lens import SAE
from sae_lens.config import DTYPE_MAP as DTYPES
from torch import Tensor, nn
from tqdm.auto import tqdm

from sae_dashboard.sae_vis_data import SaeVisConfig
from sae_dashboard.transformer_lens_wrapper import TransformerLensWrapper, to_resid_dir
from sae_dashboard.utils_fns import RollingCorrCoef

Arr = np.ndarray


class FeatureDataGenerator:
    def __init__(
        self,
        cfg: SaeVisConfig,
        tokens: Int[Tensor, "batch seq"],
        model: TransformerLensWrapper,
        encoder: SAE,
        encoder_B: SAE | None = None,
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
    def get_feature_data(  # type: ignore
        self,
        feature_indices: list[int],
        progress: list[tqdm] | None = None,  # type: ignore
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
            minibatch.to(self.cfg.device)
            model_acts = self.get_model_acts(i, minibatch)

            # Compute feature activations from this
            with FeatureMaskingContext(self.encoder, feature_indices):
                feature_acts = self.encoder.encode(model_acts).to(
                    DTYPES[self.cfg.dtype]
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
            # all_resid_post.append(residual)

            # Update the 1st progress bar (fwd passes & getting sequence data dominates the runtime of these computations)
            if progress is not None:
                progress[0].update(1)

        all_feat_acts = torch.cat(all_feat_acts, dim=0)
        # all_resid_post = torch.cat(
        #     all_resid_post, dim=0
        # )  # TODO: Check if this actually changes on each iteration and if so how to wasting effort.

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
        self, minibatch_index: int, minibatch_tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        A function that gets the model activations for a given minibatch of tokens.
        Uses np.memmap for efficient caching.
        """
        if self.cfg.cache_dir is not None:
            cache_path = self.cfg.cache_dir / f"model_activations_{minibatch_index}.dat"
            if cache_path.exists():
                model_acts = load_tensor_memmap(cache_path)
            else:
                model_acts = self.model.forward(minibatch_tokens, return_logits=False)
                save_tensor_memmap(model_acts, cache_path)
        else:
            model_acts = self.model.forward(minibatch_tokens, return_logits=False)

        if "cuda" in self.cfg.device:
            # async copy to device
            model_acts = model_acts.to(self.cfg.device, non_blocking=True)
        else:
            model_acts = model_acts.to(self.cfg.device)

        return model_acts

    @torch.inference_mode()
    def update_rolling_coefficients(
        self,
        model_acts: Float[Tensor, "batch seq d_in"],
        feature_acts: Float[Tensor, "batch seq feats"],
        corrcoef_neurons: RollingCorrCoef | None,
        corrcoef_encoder: RollingCorrCoef | None,
        corrcoef_encoder_B: RollingCorrCoef | None,
        encoder_B: SAE | None = None,
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


def save_tensor_memmap(tensor: torch.Tensor, filename: Path):
    array = tensor.float().cpu().numpy()
    shape = array.shape
    dtype = array.dtype

    # Save shape and dtype info
    np.savez(filename.with_suffix(".npz"), shape=shape, dtype=str(dtype))

    # Save the actual data as memmap
    mm = np.memmap(filename, dtype=dtype, mode="w+", shape=shape)
    mm[:] = array[:]
    mm.flush()


def load_tensor_memmap(filename: Path) -> torch.Tensor:
    # Load shape and dtype info
    info = np.load(filename.with_suffix(".npz"))
    shape = tuple(info["shape"])
    dtype = np.dtype(info["dtype"].item())

    # Load the memmap
    mm = np.memmap(filename, dtype=dtype, mode="r", shape=shape)

    # Convert to torch tensor
    return torch.from_numpy(mm)


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
