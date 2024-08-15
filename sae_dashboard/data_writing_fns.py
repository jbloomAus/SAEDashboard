import itertools
from copy import deepcopy
from pathlib import Path

from tqdm.auto import tqdm

from sae_dashboard.data_parsing_fns import get_prompt_data
from sae_dashboard.html_fns import HTML
from sae_dashboard.sae_vis_data import SaeVisData
from sae_dashboard.utils_fns import get_decode_html_safe_fn

METRIC_TITLES = {
    "act_size": "Activation Size",
    "act_quantile": "Activation Quantile",
    "loss_effect": "Loss Effect",
}


def save_feature_centric_vis(
    sae_vis_data: SaeVisData,
    filename: str | Path,
    feature_idx: int | None = None,
    include_only: list[int] | None = None,
    separate_files: bool = False,
) -> None:
    """
    Returns the HTML string for the view which lets you navigate between different features.

    Args:
        sae_vis_data:   Object containing visualization data.
        filename:       The HTML filepath we'll save the visualization to. If separate_files is True, this is used as a base name.
        feature_idx:    This is the default feature index we'll start on. If None, we use the first feature.
        include_only:   Optional list of specific features to include.
        separate_files: If True, saves each feature to a separate HTML file.
    """
    # Set the default argument for the dropdown (i.e. when the page first loads)
    first_feature = (
        next(iter(sae_vis_data.feature_data_dict))
        if (feature_idx is None)
        else feature_idx
    )

    # Get tokenize function (we only need to define it once)
    assert sae_vis_data.model is not None
    assert sae_vis_data.model.tokenizer is not None
    decode_fn = get_decode_html_safe_fn(sae_vis_data.model.tokenizer)

    # Create iterator
    if include_only is not None:
        iterator = [(i, sae_vis_data.feature_data_dict[i]) for i in include_only]
    else:
        iterator = list(sae_vis_data.feature_data_dict.items())
    if sae_vis_data.cfg.verbose:
        iterator = tqdm(iterator, desc="Saving feature-centric vis")

    HTML_OBJ = HTML()  # Initialize HTML object for combined file

    # For each FeatureData object, we get the html_obj for its feature-centric vis
    for feature, feature_data in iterator:
        html_obj = feature_data._get_html_data_feature_centric(
            sae_vis_data.cfg.feature_centric_layout, decode_fn
        )

        if separate_files:
            feature_HTML_OBJ = HTML()  # Initialize a new HTML object for each feature
            feature_HTML_OBJ.js_data[str(feature)] = deepcopy(html_obj.js_data)
            feature_HTML_OBJ.html_data = deepcopy(html_obj.html_data)

            # Add the aggdata
            feature_HTML_OBJ.js_data = {
                "AGGDATA": sae_vis_data.feature_stats.aggdata,
                "DASHBOARD_DATA": feature_HTML_OBJ.js_data,
            }

            # Generate filename for this feature
            feature_filename = Path(filename).with_stem(
                f"{Path(filename).stem}_feature_{feature}"
            )

            # Save the HTML for this feature
            feature_HTML_OBJ.get_html(
                layout_columns=sae_vis_data.cfg.feature_centric_layout.columns,
                layout_height=sae_vis_data.cfg.feature_centric_layout.height,
                filename=feature_filename,
                first_key=str(feature),
            )
        else:
            # Original behavior: accumulate all features in one HTML object
            HTML_OBJ.js_data[str(feature)] = deepcopy(html_obj.js_data)
            if feature == first_feature:
                HTML_OBJ.html_data = deepcopy(html_obj.html_data)

    if not separate_files:
        # Add the aggdata
        HTML_OBJ.js_data = {
            "AGGDATA": sae_vis_data.feature_stats.aggdata,
            "DASHBOARD_DATA": HTML_OBJ.js_data,
        }

        # Save our full HTML
        HTML_OBJ.get_html(
            layout_columns=sae_vis_data.cfg.feature_centric_layout.columns,
            layout_height=sae_vis_data.cfg.feature_centric_layout.height,
            filename=filename,
            first_key=str(first_feature),
        )


def save_prompt_centric_vis(
    sae_vis_data: SaeVisData,
    prompt: str,
    filename: str | Path,
    metric: str | None = None,
    seq_pos: int | None = None,
    num_top_features: int = 10,
):
    """
    Returns the HTML string for the view which lets you navigate between different features.

    Args:
        prompt:     The user-input prompt.
        model:      Used to get the tokenizer (for converting token IDs to string tokens).
        filename:   The HTML filepath we'll save the visualization to.
        metric:     This is the default scoring metric we'll start on. If None, we use 'act_quantile'.
        seq_pos:    This is the default seq pos we'll start on. If None, we use 0.
    """
    # Initialize the object we'll eventually get_html from
    HTML_OBJ = HTML()

    # Run forward passes on our prompt, and store the data within each FeatureData object as `self.prompt_data` as
    # well as returning the scores_dict (which maps from score hash to a list of feature indices & formatted scores)

    scores_dict = get_prompt_data(
        sae_vis_data=sae_vis_data,
        prompt=prompt,
        num_top_features=num_top_features,
    )

    # Get all possible values for dropdowns
    str_toks = sae_vis_data.model.tokenizer.tokenize(prompt)  # type: ignore
    str_toks = [
        t.replace("|", "│") for t in str_toks
    ]  # vertical line -> pipe (hacky, so key splitting on | works)
    str_toks_list = [f"{t!r} ({i})" for i, t in enumerate(str_toks)]
    metric_list = ["act_quantile", "act_size", "loss_effect"]

    # Get default values for dropdowns
    first_metric = "act_quantile" or metric
    first_seq_pos = str_toks_list[0 if seq_pos is None else seq_pos]
    first_key = f"{first_metric}|{first_seq_pos}"

    # Get tokenize function (we only need to define it once)
    assert sae_vis_data.model is not None
    assert sae_vis_data.model.tokenizer is not None
    decode_fn = get_decode_html_safe_fn(sae_vis_data.model.tokenizer)

    # For each (metric, seqpos) object, we merge the prompt-centric views of each of the top features, then we merge
    # these all together into our HTML_OBJ
    for _metric, _seq_pos in itertools.product(metric_list, range(len(str_toks))):
        # Create the key for this given combination of metric & seqpos, and get our top features & scores
        key = f"{_metric}|{str_toks_list[_seq_pos]}"
        if key not in scores_dict:
            continue
        feature_idx_list, scores_formatted = scores_dict[key]

        # Create HTML object, to store each feature column for all the top features for this particular key
        html_obj = HTML()

        for i, (feature_idx, score_formatted) in enumerate(
            zip(feature_idx_list, scores_formatted)
        ):
            # Get HTML object at this column (which includes JavaScript to dynamically set the title)
            html_obj += sae_vis_data.feature_data_dict[
                feature_idx
            ]._get_html_data_prompt_centric(
                layout=sae_vis_data.cfg.prompt_centric_layout,
                decode_fn=decode_fn,
                column_idx=i,
                bold_idx=_seq_pos,
                title=f"<h3>#{feature_idx}<br>{METRIC_TITLES[_metric]} = {score_formatted}</h3><hr>",
            )

        # Add the JavaScript (which includes the titles for each column)
        HTML_OBJ.js_data[key] = deepcopy(html_obj.js_data)

        # Set the HTML data to be the one with the most columns (since different options might have fewer cols)
        if len(HTML_OBJ.html_data) < len(html_obj.html_data):
            HTML_OBJ.html_data = deepcopy(html_obj.html_data)

    # Check our first key is in the scores_dict (if not, we should pick a different key)
    assert first_key in scores_dict, "\n".join(
        [
            f"Key {first_key} not found in {scores_dict.keys()=}.",
            "This means that there are no features with a nontrivial score for this choice of key & metric.",
        ]
    )

    # Add the aggdata
    HTML_OBJ.js_data = {
        "AGGDATA": sae_vis_data.feature_stats.aggdata,
        "DASHBOARD_DATA": HTML_OBJ.js_data,
    }

    # Save our full HTML
    HTML_OBJ.get_html(
        layout_columns=sae_vis_data.cfg.prompt_centric_layout.columns,
        layout_height=sae_vis_data.cfg.prompt_centric_layout.height,
        filename=filename,
        first_key=first_key,
    )
