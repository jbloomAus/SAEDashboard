from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
from dataclasses_json import dataclass_json

from sae_vis.components_config import (
    ActsHistogramConfig,
    FeatureTablesConfig,
    LogitsHistogramConfig,
    LogitsTableConfig,
    PromptConfig,
    SequencesConfig,
)
from sae_vis.html_fns import (
    HTML,
    bgColorMap,
    uColorMap,
)
from sae_vis.utils_fns import (
    HistogramData,
    max_or_1,
    to_str_tokens,
    unprocess_str_tok,
)

PRECISION = 4


@dataclass_json
@dataclass
class FeatureTablesData:
    """
    This contains all the data necessary to make the left-hand tables in prompt-centric visualization. See diagram
    in readme:

        https://github.com/callummcdougall/sae_vis#data_storing_fnspy

    Inputs:
        neuron_alignment...
            The data for the neuron alignment table (each of its 3 columns). In other words, the data containing which
            neurons in the transformer the encoder feature is most aligned with.

        correlated_neurons...
            The data for the correlated neurons table (each of its 3 columns). In other words, the data containing which
            neurons in the transformer are most correlated with the encoder feature.

        correlated_features...
            The data for the correlated features table (each of its 3 columns). In other words, the data containing
            which features in this encoder are most correlated with each other.

        correlated_b_features...
            The data for the correlated features table (each of its 3 columns). In other words, the data containing
            which features in encoder-B are most correlated with those in the original encoder. Note, this one might be
            absent if we're not using a B-encoder.
    """

    neuron_alignment_indices: list[int] = field(default_factory=list)
    neuron_alignment_values: list[float] = field(default_factory=list)
    neuron_alignment_l1: list[float] = field(default_factory=list)
    correlated_neurons_indices: list[int] = field(default_factory=list)
    correlated_neurons_pearson: list[float] = field(default_factory=list)
    correlated_neurons_cossim: list[float] = field(default_factory=list)
    correlated_features_indices: list[int] = field(default_factory=list)
    correlated_features_pearson: list[float] = field(default_factory=list)
    correlated_features_cossim: list[float] = field(default_factory=list)
    correlated_b_features_indices: list[int] = field(default_factory=list)
    correlated_b_features_pearson: list[float] = field(default_factory=list)
    correlated_b_features_cossim: list[float] = field(default_factory=list)

    def _get_html_data(
        self,
        cfg: FeatureTablesConfig,
        decode_fn: Callable[[int | list[int]], str | list[str]],
        id_suffix: str,
        column: int | tuple[int, int],
        component_specific_kwargs: dict[str, Any] = {},
    ) -> HTML:
        """
        Returns the HTML for the left-hand tables, wrapped in a 'grid-column' div.

        Note, we only ever use this obj in the context of the left-hand column of the feature-centric vis, and it's
        always the same width & height, which is why there's no customization available for this function.
        """
        # Read HTML from file, and replace placeholders with real ID values
        html_str = (
            Path(__file__).parent / "html" / "feature_tables_template.html"
        ).read_text()
        html_str = html_str.replace("FEATURE_TABLES_ID", f"feature-tables-{id_suffix}")

        # Create dictionary storing the data
        data: dict[str, list[dict[str, str | float]]] = {}

        # Store the neuron alignment data, if it exists
        if len(self.neuron_alignment_indices) > 0:
            assert len(self.neuron_alignment_indices) >= cfg.n_rows, "Not enough rows!"
            data["neuronAlignment"] = [
                {"index": I, "value": f"{V:+.3f}", "percentageL1": f"{L:.1%}"}
                for I, V, L in zip(
                    self.neuron_alignment_indices,
                    self.neuron_alignment_values,
                    self.neuron_alignment_l1,
                )
            ]

        # Store the other 3, if they exist (they're all in the same format, so we can do it in a for loop)
        for name, js_name in zip(
            ["correlated_neurons", "correlated_features", "correlated_b_features"],
            ["correlatedNeurons", "correlatedFeatures", "correlatedFeaturesB"],
        ):
            if len(getattr(self, f"{name}_indices")) > 0:
                # assert len(getattr(self, f"{name}_indices")) >= cfg.n_rows, "Not enough rows!"
                data[js_name] = [
                    {"index": I, "value": f"{P:+.3f}", "percentageL1": f"{C:+.3f}"}
                    for I, P, C in zip(
                        getattr(self, f"{name}_indices")[: cfg.n_rows],
                        getattr(self, f"{name}_pearson")[: cfg.n_rows],
                        getattr(self, f"{name}_cossim")[: cfg.n_rows],
                    )
                ]

        return HTML(
            html_data={column: html_str},
            js_data={"featureTablesData": {id_suffix: data}},
        )


@dataclass_json
@dataclass
class ActsHistogramData(HistogramData):
    def _get_html_data(
        self,
        cfg: ActsHistogramConfig,
        decode_fn: Callable[[int | list[int]], str | list[str]],
        id_suffix: str,
        column: int | tuple[int, int],
        component_specific_kwargs: dict[str, Any] = {},
    ) -> HTML:
        """
        Converts data -> HTML object, for the feature activations histogram (i.e. the histogram over all sampled tokens,
        showing the distribution of activations for this feature).
        """
        # We can't post-hoc change the number of bins, so check this wasn't changed in the config
        # assert cfg.n_bins == len(self.bar_heights),\
        #     "Can't post-hoc change `n_bins` in histogram config - you need to regenerate data."

        # Read HTML from file, and replace placeholders with real ID values
        html_str = (
            Path(__file__).parent / "html" / "acts_histogram_template.html"
        ).read_text()
        html_str = html_str.replace("HISTOGRAM_ACTS_ID", f"histogram-acts-{id_suffix}")

        # Process colors for frequency histogram; it's darker at higher values
        bar_values_normed = [
            (0.4 * max(self.bar_values) + 0.6 * v) / max(self.bar_values)
            for v in self.bar_values
        ]
        bar_colors = [bgColorMap(v) for v in bar_values_normed]

        # Next we create the data dict
        data: dict[str, Any] = {
            "y": self.bar_heights,
            "x": self.bar_values,
            "ticks": self.tick_vals,
            "colors": bar_colors,
        }
        if self.title is not None:
            data["title"] = self.title

        return HTML(
            html_data={column: html_str},
            js_data={"actsHistogramData": {id_suffix: data}},
        )


@dataclass_json
@dataclass
class LogitsHistogramData(HistogramData):
    def _get_html_data(
        self,
        cfg: LogitsHistogramConfig,
        decode_fn: Callable[[int | list[int]], str | list[str]],
        id_suffix: str,
        column: int | tuple[int, int],
        component_specific_kwargs: dict[str, Any] = {},
    ) -> HTML:
        """
        Converts data -> HTML object, for the logits histogram (i.e. the histogram over all tokens in the vocab, showing
        the distribution of direct logit effect on that token).
        """
        # We can't post-hoc change the number of bins, so check this wasn't changed in the config
        # assert cfg.n_bins == len(self.bar_heights),\
        #     "Can't post-hoc change `n_bins` in histogram config - you need to regenerate data."

        # Read HTML from file, and replace placeholders with real ID values
        html_str = (
            Path(__file__).parent / "html" / "logits_histogram_template.html"
        ).read_text()
        html_str = html_str.replace(
            "HISTOGRAM_LOGITS_ID", f"histogram-logits-{id_suffix}"
        )

        data: dict[str, Any] = {
            "y": self.bar_heights,
            "x": self.bar_values,
            "ticks": self.tick_vals,
        }
        if self.title is not None:
            data["title"] = self.title

        return HTML(
            html_data={column: html_str},
            js_data={"logitsHistogramData": {id_suffix: data}},
        )


@dataclass_json
@dataclass
class LogitsTableData:
    bottom_token_ids: list[int] = field(default_factory=list)
    bottom_logits: list[float] = field(default_factory=list)
    top_token_ids: list[int] = field(default_factory=list)
    top_logits: list[float] = field(default_factory=list)

    def _get_html_data(
        self,
        cfg: LogitsTableConfig,
        decode_fn: Callable[[int | list[int]], str | list[str]],
        id_suffix: str,
        column: int | tuple[int, int],
        component_specific_kwargs: dict[str, Any] = {},
    ) -> HTML:
        """
        Converts data -> HTML object, for the logits table (i.e. the top and bottom affected tokens by this feature).
        """
        # Crop the lists to `cfg.n_rows` (first checking the config doesn't ask for more rows than we have)
        assert cfg.n_rows <= len(self.bottom_logits)
        bottom_token_ids = self.bottom_token_ids[: cfg.n_rows]
        bottom_logits = self.bottom_logits[: cfg.n_rows]
        top_token_ids = self.top_token_ids[: cfg.n_rows]
        top_logits = self.top_logits[: cfg.n_rows]

        # Get the negative and positive background values (darkest when equals max abs)
        max_value = max(
            max(top_logits[: cfg.n_rows]), -min(bottom_logits[: cfg.n_rows])
        )
        neg_bg_values = np.absolute(bottom_logits[: cfg.n_rows]) / max_value
        pos_bg_values = np.absolute(top_logits[: cfg.n_rows]) / max_value

        # Get the string tokens, using the decode function
        neg_str = to_str_tokens(decode_fn, bottom_token_ids[: cfg.n_rows])
        pos_str = to_str_tokens(decode_fn, top_token_ids[: cfg.n_rows])

        # Read HTML from file, and replace placeholders with real ID values
        html_str = (
            Path(__file__).parent / "html" / "logits_table_template.html"
        ).read_text()
        html_str = html_str.replace("LOGITS_TABLE_ID", f"logits-table-{id_suffix}")

        # Create object for storing JS data
        data: dict[str, list[dict[str, str | float]]] = {
            "negLogits": [],
            "posLogits": [],
        }

        # Get data for the tables of pos/neg logits
        for i in range(len(neg_str)):
            data["negLogits"].append(
                {
                    "symbol": unprocess_str_tok(neg_str[i]),
                    "value": round(bottom_logits[i], 2),
                    "color": f"rgba(255,{int(255*(1-neg_bg_values[i]))},{int(255*(1-neg_bg_values[i]))},0.5)",
                }
            )
            data["posLogits"].append(
                {
                    "symbol": unprocess_str_tok(pos_str[i]),
                    "value": round(top_logits[i], 2),
                    "color": f"rgba({int(255*(1-pos_bg_values[i]))},{int(255*(1-pos_bg_values[i]))},255,0.5)",
                }
            )

        return HTML(
            html_data={column: html_str},
            js_data={"logitsTableData": {id_suffix: data}},
        )


@dataclass_json
@dataclass
class SequenceData:
    """
    This contains all the data necessary to make a sequence of tokens in the vis. See diagram in readme:

        https://github.com/callummcdougall/sae_vis#data_storing_fnspy

    Always-visible data:
        token_ids:              List of token IDs in the sequence
        feat_acts:              Sizes of activations on this sequence
        loss_contribution:   Effect on loss of this feature, for this particular token (neg = helpful)

    Data which is visible on hover:
        token_logits:       The logits of the particular token in that sequence (used for line on logits histogram)
        top_token_ids:     List of the top 5 logit-boosted tokens by this feature
        top_logits:        List of the corresponding 5 changes in logits for those tokens
        bottom_token_ids:  List of the bottom 5 logit-boosted tokens by this feature
        bottom_logits:     List of the corresponding 5 changes in logits for those tokens
    """

    token_ids: list[int] = field(default_factory=list)
    feat_acts: list[float] = field(default_factory=list)
    loss_contribution: list[float] = field(default_factory=list)

    token_logits: list[float] = field(default_factory=list)
    top_token_ids: list[list[int]] = field(default_factory=list)
    top_logits: list[list[float]] = field(default_factory=list)
    bottom_token_ids: list[list[int]] = field(default_factory=list)
    bottom_logits: list[list[float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """
        Filters the logits & token IDs by removing any elements which are zero (this saves space in the eventual
        JavaScript).
        """
        self.seq_len = len(self.token_ids)
        self.top_logits, self.top_token_ids = self._filter(
            self.top_logits, self.top_token_ids
        )
        self.bottom_logits, self.bottom_token_ids = self._filter(
            self.bottom_logits, self.bottom_token_ids
        )

    def _filter(
        self, float_list: list[list[float]], int_list: list[list[int]]
    ) -> tuple[list[list[float]], list[list[int]]]:
        """
        Filters the list of floats and ints, by removing any elements which are zero. Note - the absolute values of the
        floats are monotonic non-increasing, so we can assume that all the elements we keep will be the first elements
        of their respective lists. Also reduces precisions of feature activations & logits.
        """
        # Next, filter out zero-elements and reduce precision
        float_list = [
            [round(f, PRECISION) for f in floats if abs(f) > 1e-6]
            for floats in float_list
        ]
        int_list = [ints[: len(floats)] for ints, floats in zip(int_list, float_list)]
        return float_list, int_list

    def _get_html_data(
        self,
        cfg: PromptConfig | SequencesConfig,
        decode_fn: Callable[[int | list[int]], str | list[str]],
        id_suffix: str,
        column: int | tuple[int, int],
        component_specific_kwargs: dict[str, Any] = {},
    ) -> HTML:
        """
        Args:

        Returns:
            js_data: list[dict[str, Any]]
                The data for this sequence, in the form of a list of dicts for each token (where the dict stores things
                like token, feature activations, etc).
        """
        assert isinstance(
            cfg, (PromptConfig, SequencesConfig)
        ), f"Invalid config type: {type(cfg)}"
        seq_group_id = component_specific_kwargs.get("seq_group_id", None)
        max_feat_act = component_specific_kwargs.get("max_feat_act", None)
        max_loss_contribution = component_specific_kwargs.get(
            "max_loss_contribution", None
        )
        bold_idx = component_specific_kwargs.get("bold_idx", None)
        permanent_line = component_specific_kwargs.get("permanent_line", False)
        first_in_group = component_specific_kwargs.get("first_in_group", True)
        title = component_specific_kwargs.get("title", None)
        hover_above = component_specific_kwargs.get("hover_above", False)

        # If we didn't supply a sequence group ID, then we assume this sequence is on its own, and give it a unique ID
        if seq_group_id is None:
            seq_group_id = f"prompt-{column:03d}"

        # If we didn't specify bold_idx, then set it to be the midpoint
        if bold_idx is None:
            bold_idx = self.seq_len // 2

        # If we only have data for the bold token, we pad out everything with zeros or empty lists
        only_bold = isinstance(cfg, SequencesConfig) and not (cfg.compute_buffer)
        if only_bold:
            assert bold_idx != "max", "Don't know how to deal with this case yet."
            feat_acts = [
                self.feat_acts[0] if (i == bold_idx) else 0.0
                for i in range(self.seq_len)
            ]
            loss_contribution = [
                self.loss_contribution[0] if (i == bold_idx) + 1 else 0.0
                for i in range(self.seq_len)
            ]
            pos_ids = [
                self.top_token_ids[0] if (i == bold_idx) + 1 else []
                for i in range(self.seq_len)
            ]
            neg_ids = [
                self.bottom_token_ids[0] if (i == bold_idx) + 1 else []
                for i in range(self.seq_len)
            ]
            pos_val = [
                self.top_logits[0] if (i == bold_idx) + 1 else []
                for i in range(self.seq_len)
            ]
            neg_val = [
                self.bottom_logits[0] if (i == bold_idx) + 1 else []
                for i in range(self.seq_len)
            ]
        else:
            feat_acts = deepcopy(self.feat_acts)
            loss_contribution = deepcopy(self.loss_contribution)
            pos_ids = deepcopy(self.top_token_ids)
            neg_ids = deepcopy(self.bottom_token_ids)
            pos_val = deepcopy(self.top_logits)
            neg_val = deepcopy(self.bottom_logits)

        # EXPERIMENT: let's just hardcode everything except feature acts to be 0's for now.
        loss_contribution = [0.0 for _ in range(self.seq_len)]
        pos_ids = [[] for _ in range(self.seq_len)]
        neg_ids = [[] for _ in range(self.seq_len)]
        pos_val = [[] for _ in range(self.seq_len)]
        neg_val = [[] for _ in range(self.seq_len)]
        ### END EXPERIMENT

        # Get values for converting into colors later
        bg_denom = max_feat_act or max_or_1(self.feat_acts)
        u_denom = max_loss_contribution or max_or_1(self.loss_contribution, abs=True)
        bg_values = (np.maximum(feat_acts, 0.0) / max(1e-4, bg_denom)).tolist()
        u_values = (np.array(loss_contribution) / max(1e-4, u_denom)).tolist()

        # If we sent in a prompt rather than this being sliced from a longer sequence, then the pos_ids etc will be shorter
        # than the token list by 1, so we need to pad it at the first token
        if isinstance(cfg, PromptConfig):
            assert (
                len(pos_ids)
                == len(neg_ids)
                == len(pos_val)
                == len(neg_val)
                == len(self.token_ids) - 1
            ), "If this is a single prompt, these lists must be the same length as token_ids or 1 less"
            pos_ids = [[]] + pos_ids
            neg_ids = [[]] + neg_ids
            pos_val = [[]] + pos_val
            neg_val = [[]] + neg_val

        assert (
            len(pos_ids)
            == len(neg_ids)
            == len(pos_val)
            == len(neg_val)
            == len(self.token_ids)
        ), "If this is part of a sequence group etc are given, they must be the same length as token_ids"

        # Process the tokens to get str toks
        toks = to_str_tokens(decode_fn, self.token_ids)
        pos_toks = [to_str_tokens(decode_fn, pos) for pos in pos_ids]
        neg_toks = [to_str_tokens(decode_fn, neg) for neg in neg_ids]

        # Define the JavaScript object which will be used to populate the HTML string
        js_data_list = []

        for i in range(len(self.token_ids)):
            # We might store a bunch of different case-specific data in the JavaScript object for each token. This is
            # done in the form of a disjoint union over different dictionaries (which can each be empty or not), this
            # minimizes the size of the overall JavaScript object. See function in `tokens_script.js` for more.
            kwargs_bold: dict[str, bool] = {}
            kwargs_hide: dict[str, bool] = {}
            kwargs_this_token_active: dict[str, Any] = {}
            kwargs_prev_token_active: dict[str, Any] = {}
            kwargs_hover_above: dict[str, bool] = {}

            # Get args if this is the bolded token (we make it bold, and maybe add permanent line to histograms)
            if bold_idx is not None:
                kwargs_bold["isBold"] = (bold_idx == i) or (
                    bold_idx == "max" and i == np.argmax(feat_acts).item()
                )
                if kwargs_bold["isBold"] and permanent_line:
                    kwargs_bold["permanentLine"] = True

            # If we only have data for the bold token, we hide all other tokens' hoverdata (and skip other kwargs)
            if (
                only_bold
                and isinstance(bold_idx, int)
                and (i not in {bold_idx, bold_idx + 1})
            ):
                kwargs_hide["hide"] = True

            else:
                # Get args if we're making the tooltip hover above token (default is below)
                if hover_above:
                    kwargs_hover_above["hoverAbove"] = True

                # If feature active on this token, get background color and feature act (for hist line)
                if abs(feat_acts[i]) > 1e-8:
                    kwargs_this_token_active = dict(
                        featAct=round(feat_acts[i], PRECISION),
                        bgColor=bgColorMap(bg_values[i]),
                    )

                # If prev token active, get the top/bottom logits table, underline color, and loss effect (for hist line)
                pos_toks_i, neg_toks_i, pos_val_i, neg_val_i = (
                    pos_toks[i],
                    neg_toks[i],
                    pos_val[i],
                    neg_val[i],
                )
                if len(pos_toks_i) + len(neg_toks_i) > 0:
                    # Create dictionary
                    kwargs_prev_token_active = dict(
                        posToks=pos_toks_i,
                        negToks=neg_toks_i,
                        posVal=pos_val_i,
                        negVal=neg_val_i,
                        lossEffect=round(loss_contribution[i], PRECISION),
                        uColor=uColorMap(u_values[i]),
                    )

            js_data_list.append(
                dict(
                    tok=unprocess_str_tok(toks[i]),
                    tokID=self.token_ids[i],
                    tokenLogit=round(self.token_logits[i], PRECISION),
                    **kwargs_bold,
                    **kwargs_this_token_active,
                    **kwargs_prev_token_active,
                    **kwargs_hover_above,
                )
            )

        # Create HTML string (empty by default since sequences are added by JavaScript) and JS data
        html_str = ""
        js_seq_group_data: dict[str, Any] = {"data": [js_data_list]}

        # Add group-specific stuff if this is the first sequence in the group
        if first_in_group:
            # Read HTML from file, replace placeholders with real ID values
            html_str = (
                Path(__file__).parent / "html" / "sequences_group_template.html"
            ).read_text()
            html_str = html_str.replace("SEQUENCE_GROUP_ID", seq_group_id)

            # Get title of sequence group, and the idSuffix to match up with a histogram
            js_seq_group_data["idSuffix"] = id_suffix
            if title is not None:
                js_seq_group_data["title"] = title

        return HTML(
            html_data={column: html_str},
            js_data={"tokenData": {seq_group_id: js_seq_group_data}},
        )


@dataclass_json
@dataclass
class SequenceGroupData:
    """
    This contains all the data necessary to make a single group of sequences (e.g. a quantile in prompt-centric
    visualization). See diagram in readme:

        https://github.com/callummcdougall/sae_vis#data_storing_fnspy

    Inputs:
        title:      The title that this sequence group will have, if any. This is used in `_get_html_data`. The titles
                    will actually be in the HTML strings, not in the JavaScript data.
        seq_data:   The data for the sequences in this group.
    """

    title: str = ""
    seq_data: list[SequenceData] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.seq_data)

    @property
    def max_feat_act(self) -> float:
        """Returns maximum value of feature activation over all sequences in this group."""
        return max_or_1([act for seq in self.seq_data for act in seq.feat_acts])

    @property
    def max_loss_contribution(self) -> float:
        """Returns maximum value of loss contribution over all sequences in this group."""
        return max_or_1(
            [loss for seq in self.seq_data for loss in seq.loss_contribution], abs=True
        )

    def _get_html_data(
        self,
        cfg: SequencesConfig,
        decode_fn: Callable[[int | list[int]], str | list[str]],
        id_suffix: str,
        column: int | tuple[int, int],
        component_specific_kwargs: dict[str, Any] = {},
        # These default values should be correct when we only have one sequence group, because when we call this from
        # a SequenceMultiGroupData we'll override them)
    ) -> HTML:
        """
        This creates a single group of sequences, i.e. title plus some number of vertically stacked sequences.

        Note, `column` is treated specially here, because the col might overflow (hence colulmn could be a tuple).

        Args (from component-specific kwargs):
            seq_group_id:   The id of the sequence group div. This will usually be passed as e.g. "seq-group-001".
            group_size:     Max size of sequences in the group (i.e. we truncate after this many, if argument supplied).
            max_feat_act:   If supplied, then we use this as the most extreme value (for coloring by feature act).

        Returns:
            html_obj:       Object containing the HTML and JavaScript data for this seq group.
        """
        seq_group_id = component_specific_kwargs.get("seq_group_id", None)
        group_size = component_specific_kwargs.get("group_size", None)
        max_feat_act = component_specific_kwargs.get("max_feat_act", self.max_feat_act)
        max_loss_contribution = component_specific_kwargs.get(
            "max_loss_contribution", self.max_loss_contribution
        )

        # Get the data that will go into the div (list of list of dicts, i.e. containing all data for seqs in group). We
        # start with the title.
        html_obj = HTML()

        # If seq_group_id is not supplied, then we assume this is the only sequence in the column, and we name the group
        # after the column
        if seq_group_id is None:
            seq_group_id = f"seq-group-{column:03d}"

        # Accumulate the HTML data for each sequence in this group
        for i, seq in enumerate(self.seq_data[:group_size]):
            html_obj += seq._get_html_data(
                cfg=cfg,
                # pass in a PromptConfig object
                decode_fn=decode_fn,
                id_suffix=id_suffix,
                column=column,
                component_specific_kwargs=dict(
                    bold_idx="max" if cfg.buffer is None else cfg.buffer[0],
                    permanent_line=False,  # in a group, we're never showing a permanent line (only for single seqs)
                    max_feat_act=max_feat_act,
                    max_loss_contribution=max_loss_contribution,
                    seq_group_id=seq_group_id,
                    first_in_group=(i == 0),
                    title=self.title,
                ),
            )

        return html_obj


@dataclass_json
@dataclass
class SequenceMultiGroupData:
    """
    This contains all the data necessary to make multiple groups of sequences (e.g. the different quantiles in the
    prompt-centric visualization). See diagram in readme:

        https://github.com/callummcdougall/sae_vis#data_storing_fnspy
    """

    seq_group_data: list[SequenceGroupData] = field(default_factory=list)

    def __getitem__(self, idx: int) -> SequenceGroupData:
        return self.seq_group_data[idx]

    @property
    def max_feat_act(self) -> float:
        """Returns maximum value of feature activation over all sequences in this group."""
        return max_or_1([seq_group.max_feat_act for seq_group in self.seq_group_data])

    @property
    def max_loss_contribution(self) -> float:
        """Returns maximum value of loss contribution over all sequences in this group."""
        return max_or_1(
            [seq_group.max_loss_contribution for seq_group in self.seq_group_data]
        )

    def _get_html_data(
        self,
        cfg: SequencesConfig,
        decode_fn: Callable[[int | list[int]], str | list[str]],
        id_suffix: str,
        column: int | tuple[int, int],
        component_specific_kwargs: dict[str, Any] = {},
    ) -> HTML:
        """
        Args:
            decode_fn:                  Mapping from token IDs to string tokens.
            id_suffix:                  The suffix for the ID of the div containing the sequences.
            column:                     The index of this column. Note that this will be an int, but we might end up
                                        turning it into a tuple if we overflow into a new column.
            component_specific_kwargs:  Contains any specific kwargs that could be used to customize this component.

        Returns:
            html_obj:  Object containing the HTML and JavaScript data for these multiple seq groups.
        """
        assert isinstance(column, int)

        # Get max activation value & max loss contributions, over all sequences in all groups
        max_feat_act = component_specific_kwargs.get("max_feat_act", self.max_feat_act)
        max_loss_contribution = component_specific_kwargs.get(
            "max_loss_contribution", self.max_loss_contribution
        )

        # Get the correct column indices for the sequence groups, depending on how group_wrap is configured. Note, we
        # deal with overflowing columns by extending the dictionary, i.e. our column argument isn't just `column`, but
        # is a tuple of `(column, x)` where `x` is the number of times we've overflowed. For instance, if we have mode
        # 'stack-none' then our columns are `(column, 0), (column, 1), (column, 1), (column, 1), (column, 2), ...`
        n_groups = len(self.seq_group_data)
        n_quantile_groups = n_groups - 1
        match cfg.stack_mode:
            case "stack-all":
                # Here, we stack all groups into 1st column
                cols = [column for _ in range(n_groups)]
            case "stack-quantiles":
                # Here, we give 1st group its own column, and stack all groups into second column
                cols = [(column, 0)] + [(column, 1) for _ in range(n_quantile_groups)]
            case "stack-none":
                # Here, we stack groups into columns as [1, 3, 3, ...]
                cols = [
                    (column, 0),
                    *[(column, 1 + int(i / 3)) for i in range(n_quantile_groups)],
                ]
            case _:
                raise ValueError(
                    f"Invalid stack_mode: {cfg.stack_mode}. Expected in 'stack-{{all,quantiles,none}}'."
                )

        # Create the HTML object, and add all the sequence groups to it, possibly across different columns
        html_obj = HTML()
        for i, (col, group_size, sequences_group) in enumerate(
            zip(cols, cfg.group_sizes, self.seq_group_data)
        ):
            html_obj += sequences_group._get_html_data(
                cfg=cfg,
                decode_fn=decode_fn,
                id_suffix=id_suffix,
                column=col,
                component_specific_kwargs=dict(
                    group_size=group_size,
                    max_feat_act=max_feat_act,
                    max_loss_contribution=max_loss_contribution,
                    seq_group_id=f"seq-group-{column}-{i}",  # we label our sequence groups with (index, column)
                ),
            )

        return html_obj


GenericData = (
    FeatureTablesData
    | ActsHistogramData
    | LogitsTableData
    | LogitsHistogramData
    | SequenceMultiGroupData
    | SequenceData
)
