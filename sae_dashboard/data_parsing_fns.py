import re

import einops
import numpy as np
import torch
from eindex import eindex
from jaxtyping import Float, Int
from sae_lens import SAE
from torch import Tensor
from transformer_lens import HookedTransformer, utils

from sae_dashboard.components import LogitsTableData, SequenceData
from sae_dashboard.sae_vis_data import SaeVisData
from sae_dashboard.transformer_lens_wrapper import (
    ActivationConfig,
    TransformerLensWrapper,
    to_resid_direction,
)
from sae_dashboard.utils_fns import RollingCorrCoef, TopK

Arr = np.ndarray


def get_features_table_data(
    feature_out_dir: Float[Tensor, "feats d_out"],
    n_rows: int,
    corrcoef_neurons: RollingCorrCoef | None = None,
    corrcoef_encoder: RollingCorrCoef | None = None,
) -> dict[str, list[list[int]] | list[list[float]]]:
    # ! Calculate all data for the left-hand column visualisations, i.e. the 3 tables
    # Store kwargs (makes it easier to turn the tables on and off individually)
    feature_tables_data: dict[str, list[list[int]] | list[list[float]]] = {}

    # Table 1: neuron alignment, based on decoder weights
    # if layout.feature_tables_cfg.neuron_alignment_table:
    # Let's just always do this.
    add_neuron_alignment_data(
        feature_out_dir=feature_out_dir,
        feature_tables_data=feature_tables_data,
        n_rows=n_rows,
    )

    # Table 2: neurons correlated with this feature, based on their activations
    if corrcoef_neurons is not None:
        add_feature_neuron_correlations(
            corrcoef_neurons=corrcoef_neurons,
            feature_tables_data=feature_tables_data,
            n_rows=n_rows,
        )

    # Table 3: primary encoder features correlated with this feature, based on their activations
    if corrcoef_encoder is not None:
        add_intra_encoder_correlations(
            corrcoef_encoder=corrcoef_encoder,
            feature_tables_data=feature_tables_data,
            n_rows=n_rows,
        )

    return feature_tables_data


def add_intra_encoder_correlations(
    corrcoef_encoder: RollingCorrCoef,
    feature_tables_data: dict[str, list[list[int]] | list[list[float]]],
    n_rows: int,
):
    enc_indices, enc_pearson, enc_cossim = corrcoef_encoder.topk_pearson(
        k=n_rows,
    )
    feature_tables_data["correlated_features_indices"] = enc_indices
    feature_tables_data["correlated_features_pearson"] = enc_pearson
    feature_tables_data["correlated_features_cossim"] = enc_cossim


def add_neuron_alignment_data(
    feature_out_dir: Float[Tensor, "feats d_out"],
    feature_tables_data: dict[str, list[list[int]] | list[list[float]]],
    n_rows: int,
):
    top3_neurons_aligned = TopK(tensor=feature_out_dir.float(), k=n_rows, largest=True)
    feature_out_l1_norm = feature_out_dir.abs().sum(dim=-1, keepdim=True)
    pct_of_l1: Arr = np.absolute(top3_neurons_aligned.values) / utils.to_numpy(
        feature_out_l1_norm.float()
    )
    feature_tables_data["neuron_alignment_indices"] = (
        top3_neurons_aligned.indices.tolist()
    )
    feature_tables_data["neuron_alignment_values"] = (
        top3_neurons_aligned.values.tolist()
    )
    feature_tables_data["neuron_alignment_l1"] = pct_of_l1.tolist()


def add_feature_neuron_correlations(
    corrcoef_neurons: RollingCorrCoef,
    feature_tables_data: dict[str, list[list[int]] | list[list[float]]],
    n_rows: int,
):
    neuron_indices, neuron_pearson, neuron_cossim = corrcoef_neurons.topk_pearson(
        k=n_rows,
    )

    feature_tables_data["correlated_neurons_indices"] = neuron_indices
    feature_tables_data["correlated_neurons_pearson"] = neuron_pearson
    feature_tables_data["correlated_neurons_cossim"] = neuron_cossim


def get_logits_table_data(logit_vector: Float[Tensor, "d_vocab"], n_rows: int):
    # Get logits table data
    top_logits = TopK(logit_vector.float(), k=n_rows, largest=True)
    bottom_logits = TopK(logit_vector.float(), k=n_rows, largest=False)

    top_logit_values = top_logits.values.tolist()
    top_token_ids = top_logits.indices.tolist()

    bottom_logit_values = bottom_logits.values.tolist()
    bottom_token_ids = bottom_logits.indices.tolist()

    logits_table_data = LogitsTableData(
        bottom_logits=bottom_logit_values,
        bottom_token_ids=bottom_token_ids,
        top_logits=top_logit_values,
        top_token_ids=top_token_ids,
    )

    return logits_table_data


# @torch.inference_mode()
# def get_feature_data(
#     encoder: AutoEncoder,
#     model: HookedTransformer,
#     tokens: Int[Tensor, "batch seq"],
#     cfg: SaeVisConfig,
# ) -> SaeVisData:
#     """
#     This is the main function which users will run to generate the feature visualization data. It batches this
#     computation over features, in accordance with the arguments in the SaeVisConfig object (we don't want to compute all
#     the features at once, since might give OOMs).

#     See the `_get_feature_data` function for an explanation of the arguments, as well as a more detailed explanation
#     of what this function is doing.

#     The return object is the merged SaeVisData objects returned by the `_get_feature_data` function.
#     """
#     pass

#     # return sae_vis_data


@torch.inference_mode()
def parse_prompt_data(
    tokens: Int[Tensor, "batch seq"],
    str_toks: list[str],
    sae_vis_data: SaeVisData,
    feat_acts: Float[Tensor, "seq feats"],
    feature_resid_dir: Float[Tensor, "feats d_model"],
    resid_post: Float[Tensor, "seq d_model"],
    W_U: Float[Tensor, "d_model d_vocab"],
    feature_idx: list[int] | None = None,
    num_top_features: int = 10,
) -> dict[str, tuple[list[int], list[str]]]:
    """
    Gets data needed to create the sequences in the prompt-centric vis (displaying dashboards for the most relevant
    features on a prompt).

    This function exists so that prompt dashboards can be generated without using our AutoEncoder or
    TransformerLens(Wrapper) classes.

    Args:
        tokens: Int[Tensor, "batch seq"]
            The tokens we'll be using to get the feature activations. Note that we might not be using all of them; the
            number used is determined by `fvp.total_batch_size`.

        str_toks:  list[str]
            The tokens as a list of strings, so that they can be visualized in HTML.

        sae_vis_data: SaeVisData
             The object storing all data for each feature. We'll set each `feature_data.prompt_data` to the
             data we get from `prompt`.

        feat_acts: Float[Tensor, "seq feats"]
            The activations values of the features across the sequence.

        feature_resid_dir: Float[Tensor, "feats d_model"]
            The directions that each feature writes to the residual stream.

        resid_post: Float[Tensor, "seq d_model"]
            The activations of the final layer of the model before the unembed.

        W_U: Float[Tensor, "d_model d_vocab"]
            The model's unembed weights for the logit lens.

        feature_idx: list[int] or None
            The features we're actually computing. These might just be a subset of the model's full features.

        num_top_features: int
            The number of top features to display in this view, for any given metric.

    Returns:
        scores_dict: dict[str, tuple[list[int], list[str]]]
            A dictionary mapping keys like "act_quantile|'django' (0)" to a tuple of lists, where the first list is the
            feature indices, and the second list is the string-formatted values of the scores.

    As well as returning this dictionary, this function will also set `FeatureData.prompt_data` for each feature in
    `sae_vis_data` (this is necessary for getting the prompts in the prompt-centric vis). Note this design choice could
    have been done differently (i.e. have this function return a list of the prompt data for each feature). I chose this
    way because it means the FeatureData._get_html_data_prompt_centric can work fundamentally the same way as
    FeatureData._get_html_data_feature_centric, rather than treating the prompt data object as a different kind of
    component in the vis.
    """

    device = sae_vis_data.cfg.device

    if feature_idx is None:
        feature_idx = list(sae_vis_data.feature_data_dict.keys())
    n_feats = len(feature_idx)
    assert (
        feature_resid_dir.shape[0] == n_feats
    ), f"The number of features in feature_resid_dir ({feature_resid_dir.shape[0]}) does not match the number of feature indices ({n_feats})"

    assert (
        feat_acts.shape[1] == n_feats
    ), f"The number of features in feat_acts ({feat_acts.shape[1]}) does not match the number of feature indices ({n_feats})"

    feats_loss_contribution = torch.empty(
        size=(n_feats, tokens.shape[1] - 1), device=device
    )
    # Some logit computations which we only need to do once
    # correct_token_unembeddings = model_wrapped.W_U[:, tokens[0, 1:]] # [d_model seq]
    orig_logits = (
        resid_post / resid_post.std(dim=-1, keepdim=True)
    ) @ W_U  # [seq d_vocab]
    raw_logits = feature_resid_dir @ W_U  # [feats d_vocab]

    for i, feat in enumerate(feature_idx):
        # ! Calculate the sequence data for each feature, and store it as FeatureData.prompt_data

        # Get this feature's output vector, using an outer product over the feature activations for all tokens
        resid_post_feature_effect = einops.einsum(
            feat_acts[:, i], feature_resid_dir[i], "seq, d_model -> seq d_model"
        )

        # Ablate the output vector from the residual stream, and get logits post-ablation
        new_resid_post = resid_post - resid_post_feature_effect
        new_logits = (new_resid_post / new_resid_post.std(dim=-1, keepdim=True)) @ W_U

        # Get the top5 & bottom5 changes in logits (don't bother with `efficient_topk` cause it's small)
        contribution_to_logprobs = orig_logits.log_softmax(
            dim=-1
        ) - new_logits.log_softmax(dim=-1)
        # Convert to float32 for compatibility with TopK
        top_contribution_to_logits = TopK(contribution_to_logprobs.float()[:-1], k=5)
        bottom_contribution_to_logits = TopK(
            contribution_to_logprobs.float()[:-1], k=5, largest=False
        )

        # Get the change in loss (which is negative of change of logprobs for correct token)
        loss_contribution = eindex(
            -contribution_to_logprobs[:-1], tokens[0, 1:], "seq [seq]"
        )
        feats_loss_contribution[i, :] = loss_contribution

        # Store the sequence data
        sae_vis_data.feature_data_dict[feat].prompt_data = SequenceData(
            token_ids=tokens.squeeze(0).tolist(),
            feat_acts=[round(f, 4) for f in feat_acts[:, i].tolist()],
            loss_contribution=[0.0] + loss_contribution.tolist(),
            token_logits=raw_logits[i, tokens.squeeze(0)].tolist(),
            top_token_ids=top_contribution_to_logits.indices.tolist(),
            top_logits=top_contribution_to_logits.values.tolist(),
            bottom_token_ids=bottom_contribution_to_logits.indices.tolist(),
            bottom_logits=bottom_contribution_to_logits.values.tolist(),
        )

    # ! Lastly, return a dictionary mapping each key like 'act_quantile|"django" (0)' to a list of feature indices & scores

    # Get a dict with keys like f"act_quantile|'My' (1)" and values (feature indices list, feature score values list)
    scores_dict: dict[str, tuple[list[int], list[str]]] = {}

    for seq_pos, seq_key in enumerate([f"{t!r} ({i})" for i, t in enumerate(str_toks)]):
        # Filter the feature activations, since we only need the ones that are non-zero
        feat_acts_nonzero_filter = utils.to_numpy(feat_acts[seq_pos] > 0)
        feat_acts_nonzero_locations = np.nonzero(feat_acts_nonzero_filter)[0].tolist()
        _feat_acts = feat_acts[seq_pos, feat_acts_nonzero_filter]  # [feats_filtered,]
        _feature_idx = np.array(feature_idx)[feat_acts_nonzero_filter]

        if feat_acts_nonzero_filter.sum() > 0:
            k = min(num_top_features, _feat_acts.numel())

            # Get the top features by activation size. This is just applying a TopK function to the feat acts (which
            # were stored by the code before this). The feat acts are formatted to 3dp.
            act_size_topk = TopK(_feat_acts, k=k, largest=True)
            top_features = _feature_idx[act_size_topk.indices].tolist()
            formatted_scores = [f"{v:.3f}" for v in act_size_topk.values]
            scores_dict[f"act_size|{seq_key}"] = (top_features, formatted_scores)

            # Get the top features by activation quantile. We do this using the `feature_act_quantiles` object, which
            # was stored `sae_vis_data`. This quantiles object has a method to return quantiles for a given set of
            # data, as well as the precision (we make the precision higher for quantiles closer to 100%, because these
            # are usually the quantiles we're interested in, and it lets us to save space in `feature_act_quantiles`).
            act_quantile, act_precision = sae_vis_data.feature_stats.get_quantile(
                _feat_acts, feat_acts_nonzero_locations
            )
            act_quantile_topk = TopK(act_quantile, k=k, largest=True)
            act_formatting = [
                f".{act_precision[i]-2}%" for i in act_quantile_topk.indices
            ]
            top_features = _feature_idx[act_quantile_topk.indices].tolist()
            formatted_scores = [
                f"{v:{f}}" for v, f in zip(act_quantile_topk.values, act_formatting)
            ]
            scores_dict[f"act_quantile|{seq_key}"] = (top_features, formatted_scores)

        # We don't measure loss effect on the first token
        if seq_pos == 0:
            continue

        # Filter the loss effects, since we only need the ones which have non-zero feature acts on the tokens before them
        prev_feat_acts_nonzero_filter = utils.to_numpy(feat_acts[seq_pos - 1] > 0)
        _loss_contribution = feats_loss_contribution[
            prev_feat_acts_nonzero_filter, seq_pos - 1
        ]  # [feats_filtered,]
        _feature_idx_prev = np.array(feature_idx)[prev_feat_acts_nonzero_filter]

        if prev_feat_acts_nonzero_filter.sum() > 0:
            k = min(num_top_features, _loss_contribution.numel())

            # Get the top features by loss effect. This is just applying a TopK function to the loss effects (which were
            # stored by the code before this). The loss effects are formatted to 3dp. We look for the most negative
            # values, i.e. the most loss-reducing features.
            loss_contribution_topk = TopK(_loss_contribution, k=k, largest=False)
            top_features = _feature_idx_prev[loss_contribution_topk.indices].tolist()
            formatted_scores = [f"{v:+.3f}" for v in loss_contribution_topk.values]
            scores_dict[f"loss_effect|{seq_key}"] = (top_features, formatted_scores)
    return scores_dict


@torch.inference_mode()
def get_prompt_data(
    sae_vis_data: SaeVisData,
    prompt: str,
    num_top_features: int,
) -> dict[str, tuple[list[int], list[str]]]:
    """
    Gets data that will be used to create the sequences in the prompt-centric HTML visualisation, i.e. an object of
    type SequenceData for each of our features.

    Args:
        sae_vis_data:     The object storing all data for each feature. We'll set each `feature_data.prompt_data` to the
                          data we get from `prompt`.
        prompt:           The prompt we'll be using to get the feature activations.#
        num_top_features: The number of top features we'll be getting data for.

    Returns:
        scores_dict:      A dictionary mapping keys like "act_quantile|0" to a tuple of lists, where the first list is
                          the feature indices, and the second list is the string-formatted values of the scores.

    As well as returning this dictionary, this function will also set `FeatureData.prompt_data` for each feature in
    `sae_vis_data`. This is because the prompt-centric vis will call `FeatureData._get_html_data_prompt_centric` on each
    feature data object, so it's useful to have all the data in once place! Even if this will get overwritten next
    time we call `get_prompt_data` for this same `sae_vis_data` object.
    """

    # ! Boring setup code
    feature_idx = list(sae_vis_data.feature_data_dict.keys())
    encoder = sae_vis_data.encoder
    assert isinstance(encoder, SAE)
    model = sae_vis_data.model
    assert isinstance(model, HookedTransformer)
    cfg = sae_vis_data.cfg
    assert isinstance(cfg.hook_point, str), f"{cfg.hook_point=}, expected a string"

    str_toks: list[str] = model.tokenizer.tokenize(prompt)  # type: ignore
    tokens = model.tokenizer.encode(prompt, return_tensors="pt").to(  # type: ignore
        sae_vis_data.cfg.device
    )
    assert isinstance(tokens, torch.Tensor)

    # Determine auxiliary hook points
    auxiliary_hook_points = []
    if "hook_resid" not in cfg.hook_point:
        if match := re.match(r"blocks\.(\d+)\.", cfg.hook_point):
            layer_num = int(match.group(1))
            auxiliary_hook_points = [f"blocks.{layer_num}.hook_resid_post"]

    model_wrapped = TransformerLensWrapper(
        model, ActivationConfig(cfg.hook_point, auxiliary_hook_points)
    )

    # Get consistent dtype for all operations
    dtype = model.W_U.dtype

    activations = model_wrapped(tokens, return_logits=False)
    act_post = activations[cfg.hook_point]

    hook_point = auxiliary_hook_points[0] if auxiliary_hook_points else cfg.hook_point
    resid_post = activations[hook_point].squeeze(0).to(dtype)

    feature_act_dir = encoder.W_enc[:, feature_idx]  # [d_in feats]
    feature_out_dir = encoder.W_dec[feature_idx]  # [feats d_in]
    feature_resid_dir = to_resid_direction(feature_out_dir, model_wrapped).to(dtype)

    assert (
        feature_act_dir.T.shape
        == feature_out_dir.shape
        == (len(feature_idx), encoder.cfg.d_in)
    )

    feat_acts = (
        encoder.encode(act_post.to(encoder.device, encoder.dtype))[..., feature_idx]
        .squeeze(0)
        .to(act_post.device, dtype)
    )

    # ! Use the data we've collected to make the scores_dict and update the sae_vis_data
    scores_dict = parse_prompt_data(
        tokens=tokens,
        str_toks=str_toks,
        sae_vis_data=sae_vis_data,
        feat_acts=feat_acts,
        feature_resid_dir=feature_resid_dir,
        resid_post=resid_post,
        W_U=model.W_U,
        feature_idx=feature_idx,
        num_top_features=num_top_features,
    )

    return scores_dict
