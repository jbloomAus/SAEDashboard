from dataclasses import dataclass
from typing import Any, Iterator, Literal

SEQUENCES_CONFIG_HELP = dict(
    buffer="How many tokens to add as context to each sequence, on each side. The tokens chosen for the top acts / \
quantile groups can't be outside the buffer range. If None, we use the entire sequence as context.",
    compute_buffer="If False, then we don't compute the loss effect, activations, or any other data for tokens \
other than the bold tokens in our sequences (saving time).",
    n_quantiles="Number of quantile groups for the sequences. If zero, we only show top activations, no quantile \
groups.",
    top_acts_group_size="Number of sequences in the 'top activating sequences' group.",
    quantile_group_size="Number of sequences in each of the sequence quantile groups.",
    top_logits_hoverdata="Number of top/bottom logits to show in the hoverdata for each token.",
    stack_mode="How to stack the sequence groups.\n  'stack-all' = all groups are stacked in a single column \
(scrolls vertically if it overflows)\n  'stack-quantiles' = first col contains top acts, second col contains all \
quantile groups\n  'stack-none' = we stack in a way which ensures no vertical scrolling.",
    hover_below="Whether the hover information about a token appears below or above the token.",
)

ACTIVATIONS_HISTOGRAM_CONFIG_HELP = dict(
    n_bins="Number of bins for the histogram.",
)

LOGITS_HISTOGRAM_CONFIG_HELP = dict(
    n_bins="Number of bins for the histogram.",
)

LOGITS_TABLE_CONFIG_HELP = dict(
    n_rows="Number of top/bottom logits to show in the table.",
)

FEATURE_TABLES_CONFIG_HELP = dict(
    n_rows="Number of rows to show for each feature table.",
    neuron_alignment_table="Whether to show the neuron alignment table.",
    correlated_neurons_table="Whether to show the correlated neurons table.",
    correlated_features_table="Whether to show the (pairwise) correlated features table.",
    correlated_b_features_table="Whether to show the correlated encoder-B features table.",
)


@dataclass
class BaseComponentConfig:
    def data_is_contained_in(self, other: "BaseComponentConfig") -> bool:
        """
        This returns False only when the data that was computed based on `other` wouldn't be enough to show the data
        that was computed based on `self`. For instance, if `self` was a config object with 10 rows, and `other` had
        just 5 rows, then this would return False. A less obvious example: if `self` was a histogram config with 50 bins
        then `other` would need to have exactly 50 bins (because we can't change the bins after generating them).
        """
        return True

    @property
    def help_dict(self) -> dict[str, str]:
        """
        This is a dictionary which maps the name of each argument to a description of what it does. This is used when
        printing out the help for a config object, to show what each argument does.
        """
        return {}


@dataclass
class PromptConfig(BaseComponentConfig):
    pass


@dataclass
class SequencesConfig(BaseComponentConfig):
    buffer: tuple[int, int] | None = (5, 5)
    compute_buffer: bool = True
    n_quantiles: int = 10
    top_acts_group_size: int = 20
    quantile_group_size: int = 5
    top_logits_hoverdata: int = 5
    stack_mode: Literal["stack-all", "stack-quantiles", "stack-none"] = "stack-all"
    hover_below: bool = True

    def data_is_contained_in(self, other: BaseComponentConfig) -> bool:
        assert isinstance(other, self.__class__)
        return all(
            [
                self.buffer is None
                or (
                    other.buffer is not None and self.buffer[0] <= other.buffer[0]
                ),  # the buffer needs to be <=
                self.buffer is None
                or (other.buffer is not None and self.buffer[1] <= other.buffer[1]),
                int(self.compute_buffer)
                <= int(
                    other.compute_buffer
                ),  # we can't compute the buffer if we didn't in `other`
                self.n_quantiles
                in {
                    0,
                    other.n_quantiles,
                },  # we actually need the quantiles identical (or one to be zero)
                self.top_acts_group_size
                <= other.top_acts_group_size,  # group size needs to be <=
                self.quantile_group_size
                <= other.quantile_group_size,  # each quantile group needs to be <=
                self.top_logits_hoverdata
                <= other.top_logits_hoverdata,  # hoverdata rows need to be <=
            ]
        )

    def __post_init__(self):
        # Get list of group lengths, based on the config params
        self.group_sizes = [self.top_acts_group_size] + [
            self.quantile_group_size
        ] * self.n_quantiles

    @property
    def help_dict(self) -> dict[str, str]:
        return SEQUENCES_CONFIG_HELP


@dataclass
class ActsHistogramConfig(BaseComponentConfig):
    n_bins: int = 50

    def data_is_contained_in(self, other: BaseComponentConfig) -> bool:
        assert isinstance(other, self.__class__)
        return self.n_bins == other.n_bins

    @property
    def help_dict(self) -> dict[str, str]:
        return ACTIVATIONS_HISTOGRAM_CONFIG_HELP


@dataclass
class LogitsHistogramConfig(BaseComponentConfig):
    n_bins: int = 50

    def data_is_contained_in(self, other: BaseComponentConfig) -> bool:
        assert isinstance(other, self.__class__)
        return self.n_bins == other.n_bins

    @property
    def help_dict(self) -> dict[str, str]:
        return LOGITS_HISTOGRAM_CONFIG_HELP


@dataclass
class LogitsTableConfig(BaseComponentConfig):
    n_rows: int = 10

    def data_is_contained_in(self, other: BaseComponentConfig) -> bool:
        assert isinstance(other, self.__class__)
        return self.n_rows <= other.n_rows

    @property
    def help_dict(self) -> dict[str, str]:
        return LOGITS_TABLE_CONFIG_HELP


@dataclass
class FeatureTablesConfig(BaseComponentConfig):
    n_rows: int = 3
    neuron_alignment_table: bool = True
    correlated_neurons_table: bool = True
    correlated_features_table: bool = True
    correlated_b_features_table: bool = False

    def data_is_contained_in(self, other: BaseComponentConfig) -> bool:
        assert isinstance(other, self.__class__)
        return all(
            [
                self.n_rows <= other.n_rows,
                self.neuron_alignment_table <= other.neuron_alignment_table,
                self.correlated_neurons_table <= other.correlated_neurons_table,
                self.correlated_features_table <= other.correlated_features_table,
                self.correlated_b_features_table <= other.correlated_b_features_table,
            ]
        )

    @property
    def help_dict(self) -> dict[str, str]:
        return FEATURE_TABLES_CONFIG_HELP


GenericComponentConfig = (
    PromptConfig
    | SequencesConfig
    | ActsHistogramConfig
    | LogitsHistogramConfig
    | LogitsTableConfig
    | FeatureTablesConfig
)


class Column:
    def __init__(
        self,
        *args: GenericComponentConfig,
        width: int | None = None,
    ):
        self.components = list(args)
        self.width = width

    def __iter__(self) -> Iterator[Any]:
        return iter(self.components)

    def __getitem__(self, idx: int) -> Any:
        return self.components[idx]

    def __len__(self) -> int:
        return len(self.components)
