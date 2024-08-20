from dataclasses import dataclass, field
from typing import Any, Callable, List, Literal, Optional

from sae_dashboard.components import (
    ActsHistogramData,
    FeatureTablesData,
    GenericData,
    LogitsHistogramData,
    LogitsTableData,
    SequenceData,
    SequenceMultiGroupData,
)
from sae_dashboard.components_config import (
    ActsHistogramConfig,
    FeatureTablesConfig,
    GenericComponentConfig,
    LogitsHistogramConfig,
    LogitsTableConfig,
    PromptConfig,
    SequencesConfig,
)
from sae_dashboard.html_fns import HTML
from sae_dashboard.layout import SaeVisLayoutConfig


@dataclass
class DFAData:
    dfaValues: List[List[float]] = field(default_factory=list)
    dfaTargetIndex: List[int] = field(default_factory=list)
    dfaMaxValue: float = 0.0


@dataclass
class FeatureData:
    """
    This contains all the data necessary to make the feature-centric visualization, for a single feature. See
    diagram in readme:

        https://github.com/callummcdougall/sae_vis#data_storing_fnspy

    Args:
        feature_idx:    Index of the feature in question (not used within this class's methods, but used elsewhere).
        cfg:            Contains layout parameters which are important in the `get_html` function.

        The other args are the 6 possible components we might have in the feature-centric vis, i.e. this is where we
        store the actual data. Note that one of these arguments is `prompt_data` which is only applicable in the prompt-
        centric view.

    This is used in both the feature-centric and prompt-centric views. In the feature-centric view, a single one
    of these objects creates the HTML for a single feature (i.e. a full screen). In the prompt-centric view, a single
    one of these objects will create one column of the full screen vis.
    """

    feature_tables_data: FeatureTablesData = field(
        default_factory=lambda: FeatureTablesData()
    )
    acts_histogram_data: ActsHistogramData = field(
        default_factory=lambda: ActsHistogramData()
    )
    logits_table_data: LogitsTableData = field(
        default_factory=lambda: LogitsTableData()
    )
    logits_histogram_data: LogitsHistogramData = field(
        default_factory=lambda: LogitsHistogramData()
    )
    sequence_data: SequenceMultiGroupData = field(
        default_factory=lambda: SequenceMultiGroupData()
    )
    prompt_data: SequenceData = field(default_factory=lambda: SequenceData())
    dfa_data: Optional[dict[int, dict[str, Any]]] = None

    def __post_init__(self):
        if self.dfa_data is None:
            self.dfa_data = DFAData()

    def get_component_from_config(self, config: GenericComponentConfig) -> GenericData:
        """
        Given a config object, returns the corresponding data object stored by this instance. For instance, if the input
        is an `FeatureTablesConfig` instance, then this function returns `self.feature_tables_data`.
        """
        CONFIG_CLASS_MAP = {
            FeatureTablesConfig.__name__: self.feature_tables_data,
            ActsHistogramConfig.__name__: self.acts_histogram_data,
            LogitsTableConfig.__name__: self.logits_table_data,
            LogitsHistogramConfig.__name__: self.logits_histogram_data,
            SequencesConfig.__name__: self.sequence_data,
            PromptConfig.__name__: self.prompt_data,
            # Add DFA config here if we create a specific config for it
        }
        config_class_name = config.__class__.__name__
        assert (
            config_class_name in CONFIG_CLASS_MAP
        ), f"Invalid component config: {config_class_name}"
        return CONFIG_CLASS_MAP[config_class_name]

    def _get_html_data_feature_centric(
        self,
        layout: SaeVisLayoutConfig,
        decode_fn: Callable[[int | list[int]], str | list[str]],
    ) -> HTML:
        """
        Returns the HTML object for a single feature-centric view. These are assembled together into the full feature-
        centric view.

        Args:
            decode_fn:  We use this function to decode the token IDs into string tokens.

        Returns:
            html_obj.html_data:
                Contains a dictionary with keys equal to columns, and values equal to the HTML strings. These will be
                turned into grid-column elements, and concatenated.
            html_obj.js_data:
                Contains a dictionary with keys = component names, and values = JavaScript data that will be used by the
                scripts we'll eventually dump in.
        """
        # Create object to store all HTML
        html_obj = HTML()

        # For every column in this feature-centric layout, we add all the components in that column
        for column_idx, column_components in layout.columns.items():
            for component_config in column_components:
                component = self.get_component_from_config(component_config)

                html_obj += component._get_html_data(
                    cfg=component_config,
                    decode_fn=decode_fn,
                    column=column_idx,
                    id_suffix="0",  # we only use this if we have >1 set of histograms, i.e. prompt-centric vis
                )

        return html_obj

    def _get_html_data_prompt_centric(
        self,
        layout: SaeVisLayoutConfig,
        decode_fn: Callable[[int | list[int]], str | list[str]],
        column_idx: int,
        bold_idx: int | Literal["max"],
        title: str,
    ) -> HTML:
        """
        Returns the HTML object for a single column of the prompt-centric view. These are assembled together into a full
        screen of a prompt-centric view, and then they're further assembled together into the full prompt-centric view.

        Args:
            decode_fn:  We use this function to decode the token IDs into string tokens.
            column_idx: This method only gives us a single column (of the prompt-centric vis), so we need to know which
                        column this is (for the JavaScript data).
            bold_idx:   Which index should be bolded in the sequence data. If "max", we default to bolding the max-act
                        token in each sequence.
            title:      The title for this column, which will be used in the JavaScript data.

        Returns:
            html_obj.html_data:
                Contains a dictionary with the single key `str(column_idx)`, representing the single column. This will
                become a single grid-column element, and will get concatenated with others of these.
            html_obj.js_data:
                Contains a dictionary with keys = component names, and values = JavaScript data that will be used by the
                scripts we'll eventually dump in.
        """
        # Create object to store all HTML
        html_obj = HTML()

        # Verify that we only have a single column
        assert layout.columns.keys() == {
            0
        }, f"prompt_centric_layout should only have 1 column, instead found cols {layout.columns.keys()}"
        assert (
            layout.prompt_cfg is not None
        ), "prompt_centric_cfg should include a PromptConfig, but found None"
        if layout.seq_cfg is not None:
            assert (layout.seq_cfg.n_quantiles == 0) or (
                layout.seq_cfg.stack_mode == "stack-all"
            ), "prompt_centric_layout should have stack_mode='stack-all' if n_quantiles > 0, so that it fits in 1 col"

        # Get the maximum color over both the prompt and the sequences
        max_feat_act = max(
            max(self.prompt_data.feat_acts), self.sequence_data.max_feat_act
        )
        max_loss_contribution = max(
            max(self.prompt_data.loss_contribution),
            self.sequence_data.max_loss_contribution,
        )

        # For every component in the single column of this prompt-centric layout, add all the components in that column
        for component_config in layout.columns[0]:
            component = self.get_component_from_config(component_config)

            html_obj += component._get_html_data(
                cfg=component_config,
                decode_fn=decode_fn,
                column=column_idx,
                id_suffix=str(column_idx),
                component_specific_kwargs=dict(  # only used by SequenceData (the prompt)
                    bold_idx=bold_idx,
                    permanent_line=True,
                    hover_above=True,
                    max_feat_act=max_feat_act,
                    max_loss_contribution=max_loss_contribution,
                ),
            )

        # Add the title in JavaScript, and the empty title element in HTML
        html_obj.html_data[column_idx] = (
            f"<div id='column-{column_idx}-title'></div>\n{html_obj.html_data[column_idx]}"
        )
        html_obj.js_data["gridColumnTitlesData"] = {str(column_idx): title}

        return html_obj
