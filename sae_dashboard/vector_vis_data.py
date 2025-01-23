import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from dataclasses_json import dataclass_json
from rich import print as rprint
from rich.table import Table
from transformer_lens import HookedTransformer

from sae_dashboard.neuronpedia.vector_set import VectorSet
from sae_dashboard.feature_data import FeatureData

from sae_dashboard.layout import SaeVisLayoutConfig
from sae_dashboard.utils_fns import FeatureStatistics

VECTOR_CONFIG_DICT = dict(
    hook_point="The hook point to use for the VectorSet",
    features="The set of features which we'll be gathering data for. If an integer, we only get data for 1 feature",
    batch_size="The number of sequences we'll gather data for. If supplied then it can't be larger than `tokens[0]`, \
if not then we use all of `tokens`",
    minibatch_size_tokens="The minibatch size we'll use to split up the full batch during forward passes, to avoid \
OOMs.",
    minibatch_size_features="The feature minibatch size we'll use to split up our features, to avoid OOM errors",
    seed="Random seed, for reproducibility (e.g. sampling quantiles)",
    verbose="Whether to print out progress messages and other info during the data gathering process",
)

OUT_OF_RANGE_TOKEN = "<|outofrange|>"


@dataclass_json
@dataclass
class VectorVisConfig:
    # Data
    hook_point: str
    vector_indices: Iterable[int]
    minibatch_size_features: int = 256
    minibatch_size_tokens: int = 64
    quantile_feature_batch_size: int = 64
    perform_ablation_experiments: bool = False
    device: str = "cpu"
    dtype: str = "float32"
    ignore_tokens: set[int] = field(default_factory=set)
    ignore_positions: list[int] = field(default_factory=list)

    # Vis
    feature_centric_layout: SaeVisLayoutConfig = field(
        default_factory=SaeVisLayoutConfig.default_feature_centric_layout
    )
    prompt_centric_layout: SaeVisLayoutConfig = field(
        default_factory=SaeVisLayoutConfig.default_prompt_centric_layout
    )

    # Additional computations
    use_dfa: bool = False

    # Misc
    seed: int | None = 0
    verbose: bool = False
    cache_dir: Path | None = None  # Path to cache the data

    def to_dict(self) -> dict[str, Any]:
        """Used for type hinting (the actual method comes from the `dataclass_json` decorator)."""
        ...

    def help(self, title: str = "SaeVisConfig"):
        """
        Performs the `help` method for both of the layout objects, as well as for the non-layout-based configs.
        """
        # Create table for all the non-layout-based params
        table = Table(
            "Param", "Value (default)", "Description", title=title, show_lines=True
        )

        # Populate table (middle row is formatted based on whether value has changed from default)
        for param, desc in VECTOR_CONFIG_DICT.items():
            value = getattr(self, param)
            value_default = getattr(self.__class__, param, "no default")
            if value != value_default:
                value_default_repr = (
                    "no default"
                    if value_default == "no default"
                    else repr(value_default)
                )
                value_str = f"[b dark_orange]{value!r}[/]\n({value_default_repr})"
            else:
                value_str = f"[b #00aa00]{value!r}[/]"
            table.add_row(param, value_str, f"[i]{desc}[/]")

        # Print table, and print the help trees for the layout objects
        rprint(table)
        self.feature_centric_layout.help(
            title="SaeVisLayoutConfig: feature-centric vis", key=False
        )
        self.prompt_centric_layout.help(
            title="SaeVisLayoutConfig: prompt-centric vis", key=False
        )


@dataclass_json
@dataclass
class _VectorVisData:
    """
    Dataclass which is used to store the data for the VectorVisData class. It excludes everything which isn't easily
    serializable, only saving the raw data.
    """

    vector_data_dict: dict[int, FeatureData] = field(default_factory=dict)
    vector_stats: FeatureStatistics = field(default_factory=FeatureStatistics)

    @classmethod
    def from_dict(
        cls, data: dict[str, Any]
    ) -> (
        "_VectorVisData"
    ): ...  # just for type hinting; the method comes from 'dataclass_json'

    def to_dict(
        self,
    ) -> dict[
        str, Any
    ]: ...  # just for type hinting; the method comes from 'dataclass_json'


@dataclass
class VectorVisData:
    """
    This contains all the data necessary for constructing the feature-centric visualization, over multiple
    features (i.e. being able to navigate through them). See diagram in readme:

        https://github.com/callummcdougall/sae_vis#data_storing_fnspy

    Args:
        feature_data_dict:  Contains the data for each individual feature-centric vis.
        feature_stats:      Contains the stats over all features (including the quantiles of activation values for each
                            feature (used for rank-ordering features in the prompt-centric vis).
        cfg:                The vis config, used for the both the data gathering and the vis layout.
        model:              The model which our encoder was trained on.
        encoder:            The encoder used to get the feature activations.
    """

    cfg: VectorVisConfig  # = field(default_factory=VectorVisConfig)
    vector_data_dict: dict[int, FeatureData] = field(default_factory=dict)
    vector_stats: FeatureStatistics = field(default_factory=FeatureStatistics)

    model: HookedTransformer | None = None
    encoder: VectorSet | None = None

    def update(self, other: "VectorVisData") -> None:
        """
        Updates a VectorVisData object with the data from another VectorVisData object. This is useful during the
        `get_feature_data` function, since this function is broken up into different groups of features then merged
        together.
        """
        if other is None:
            return
        self.vector_data_dict.update(other.vector_data_dict)
        self.vector_stats.update(other.vector_stats)

    def save_json(self: "VectorVisData", filename: str | Path) -> None:
        """
        Saves an VectorVisData instance to a JSON file. The config, model & encoder arguments must be user-supplied.
        """
        if isinstance(filename, str):
            filename = Path(filename)
        assert filename.suffix == ".json", "Filename must have a .json extension"

        _self = _VectorVisData(
            vector_data_dict=self.vector_data_dict,
            vector_stats=self.vector_stats,
        )

        with open(filename, "w") as f:
            json.dump(_self.to_dict(), f)

    @classmethod
    def load_json(
        cls,
        filename: str | Path,
        cfg: VectorVisConfig,
        model: HookedTransformer,
        encoder: VectorSet,
    ) -> "VectorVisData":
        """
        Loads an VectorVisData instance from JSON file. The config, model & encoder arguments must be user-supplied.
        """
        if isinstance(filename, str):
            filename = Path(filename)
        assert filename.suffix == ".json", "Filename must have a .json extension"

        with open(filename) as f:
            data = json.load(f)

        _self = _VectorVisData.from_dict(data)

        self = VectorVisData(
            cfg=cfg,
            vector_data_dict=_self.vector_data_dict,
            vector_stats=_self.vector_stats,
            model=model,
            encoder=encoder,
        )

        return self
    

