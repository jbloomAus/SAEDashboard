import time

import einops
import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor
from tqdm.auto import tqdm

from sae_vis.autoencoder import AutoEncoder
from sae_vis.sae_vis_data import SaeVisConfig
from sae_vis.transformer_lens_wrapper import (
    TransformerLensWrapper,
    to_resid_dir,
)
from sae_vis.utils_fns import (
    RollingCorrCoef,
)

Arr = np.ndarray


@torch.inference_mode()
def get_feature_data(
    encoder: AutoEncoder,
    encoder_B: AutoEncoder | None,
    model: TransformerLensWrapper,
    token_minibatches: list[Int[Tensor, "batch seq"]],
    feature_indices: list[int],
    cfg: SaeVisConfig,
    progress: list[tqdm] | None = None,
):  # type: ignore
    """
    Gets data that will be used to create the sequences in the feature-centric HTML visualisation.

    Note - this function isn't called directly by the user, it actually gets called by the `get_feature_data` function
    which does exactly the same thing except it also batches this computation by features (in accordance with the
    arguments `features` and `minibatch_size_features` from the SaeVisConfig object).

    Args:
        encoder: AutoEncoder
            The encoder whose features we'll be analyzing.

        encoder_B: AutoEncoder
            The encoder we'll be using as a reference (i.e. finding the B-features with the highest correlation). This
            is only necessary if we're generating the left-hand tables (i.e. cfg.include_left_tables=True).

        model: TransformerLensWrapper
            The model we'll be using to get the feature activations. It's a wrapping of the base TransformerLens model.

        token_minibatches: list[Int[Tensor, "batch seq"]]
            The tokens we're analyzing. These are split into minibatches, which are then used to get the model's
            activations.

        feature_indices: Union[int, list[int]]
            The features we're actually computing. These might just be a subset of the model's full features.

        cfg: SaeVisConfig
            Feature visualization parameters, containing a bunch of other stuff. See the SaeVisConfig docstring for
            more information.

        progress: Optional[list[tqdm]]
            An optional list containing progress bars for the forward passes and the sequence data. This is used to
            update the progress bars as the computation runs.

    Returns:
        sae_vis_data: SaeVisData
            Containing data for creating each feature visualization, as well as data for rank-ordering the feature
            visualizations when it comes time to make the prompt-centric view (the `feature_act_quantiles` attribute).

        time_log: dict[str, float]
            A dictionary containing the time taken for each step of the computation. This is optionally printed at the
            end of the `get_feature_data` function, if `cfg.verbose` is set to True.
    """
    # ! Boring setup code
    time_logs = {
        "(1) Initialization": 0.0,
        "(2) Forward passes to gather model activations": 0.0,
        "(3) Computing feature acts from model acts": 0.0,
    }

    t0 = time.time()

    # ! Data setup code (defining the main objects we'll eventually return, for each of 5 possible vis components)

    # Create lists to store the feature activations & final values of the residual stream
    all_resid_post = []
    all_feat_acts = []

    # Create objects to store the data for computing rolling stats
    corrcoef_neurons = RollingCorrCoef()
    corrcoef_encoder = RollingCorrCoef(indices=feature_indices, with_self=True)
    corrcoef_encoder_B = RollingCorrCoef() if encoder_B is not None else None

    # Get encoder & decoder directions
    feature_out_dir = encoder.W_dec[feature_indices]  # [feats d_autoencoder]
    feature_resid_dir = to_resid_dir(feature_out_dir, model)  # [feats d_model]

    time_logs["(1) Initialization"] = time.time() - t0

    # ! Compute & concatenate together all feature activations & post-activation function values

    for i, minibatch in enumerate(token_minibatches):
        # Fwd pass, get model activations
        t0 = time.time()

        if cfg.cache_dir is not None:
            # check if the activations are already cached
            cache_path = (
                cfg.cache_dir / f"model_activations_and_residuals_{i}.safetensors"
            )

            if cache_path.exists():
                with safe_open(cache_path, framework="pt", device=cfg.device) as f:  # type: ignore
                    model_acts = f.get_tensor("activations")
                    residual = f.get_tensor("residual")
            else:
                residual, model_acts = model.forward(minibatch, return_logits=False)
                tensors = {"activations": model_acts, "residual": residual}
                # could also save tokens to avoid needing to provide them above.
                save_file(tensors, cache_path)

        # residual, model_acts = model.forward(minibatch, return_logits=False)
        time_logs["(2) Forward passes to gather model activations"] += time.time() - t0

        # Compute feature activations from this
        t0 = time.time()
        feature_acts = compute_feat_acts(
            model_acts=model_acts,
            feature_idx=feature_indices,
            encoder=encoder,
        )

        update_rolling_coefficients(
            model_acts=model_acts,
            feature_acts=feature_acts,
            corrcoef_neurons=corrcoef_neurons,
            corrcoef_encoder=corrcoef_encoder,
            corrcoef_encoder_B=corrcoef_encoder_B,
        )

        time_logs["(3) Computing feature acts from model acts"] += time.time() - t0

        # Add these to the lists (we'll eventually concat)
        all_feat_acts.append(feature_acts)
        all_resid_post.append(residual)

        # Update the 1st progress bar (fwd passes & getting sequence data dominates the runtime of these computations)
        if progress is not None:
            progress[0].update(1)

    all_feat_acts = torch.cat(all_feat_acts, dim=0)
    all_resid_post = torch.cat(all_resid_post, dim=0)

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
def compute_feat_acts(
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
            einops.rearrange(feat_acts_B, "batch seq d_hidden -> d_hidden (batch seq)"),
        )
