from dataclasses import asdict, dataclass, field

from dataclasses_json import dataclass_json
from rich import print as rprint
from rich.tree import Tree

from sae_dashboard.components_config import (
    ActsHistogramConfig,
    BaseComponentConfig,
    Column,
    FeatureTablesConfig,
    LogitsHistogramConfig,
    LogitsTableConfig,
    PromptConfig,
    SequencesConfig,
)

KEY_LAYOUT_VIS = """Key: 
  the tree shows which components will be displayed in each column (from left to right)
  arguments are [b #00aa00]green[/]
  arguments changed from their default are [b dark_orange]orange[/], with default in brackets
  argument descriptions are in [i]italics[/i]
"""


@dataclass_json
@dataclass
class SaeVisLayoutConfig:
    """
    This object allows you to set all the ways the feature vis will be laid out.

    Args (specified by the user):
        columns:
            A list of `Column` objects, where each `Column` contains a list of component configs.
        height:
            The height of the vis (in pixels).

    Args (defined during __init__):
        seq_cfg:
            The `SequencesConfig` object, which contains all the parameters for the top activating sequences (and the
            quantile groups).
        act_hist_cfg:
            The `ActsHistogramConfig` object, which contains all the parameters for the activations histogram.
        logits_hist_cfg:
            The `LogitsHistogramConfig` object, which contains all the parameters for the logits histogram.
        logits_table_cfg:
            The `LogitsTableConfig` object, which contains all the parameters for the logits table.
        feature_tables_cfg:
            The `FeatureTablesConfig` object, which contains all the parameters for the feature tables.
        prompt_cfg:
            The `PromptConfig` object, which contains all the parameters for the prompt-centric vis.
    """

    columns: dict[int | tuple[int, int], Column] = field(default_factory=dict)
    height: int = 750

    seq_cfg: SequencesConfig | None = None
    act_hist_cfg: ActsHistogramConfig | None = None
    logits_hist_cfg: LogitsHistogramConfig | None = None
    logits_table_cfg: LogitsTableConfig | None = None
    feature_tables_cfg: FeatureTablesConfig | None = None
    prompt_cfg: PromptConfig | None = None

    def __init__(self, columns: list[Column], height: int = 750):
        """
        The __init__ method will allow you to extract things like `self.seq_cfg` from the object (even though they're
        initially stored in the `columns` attribute). It also verifies that there are no duplicate components (which is
        redundant, and could mess up the HTML).
        """
        # Define the columns (as dict) and the height
        self.columns = {idx: col for idx, col in enumerate(columns)}
        self.height = height

        # Get a list of all our components, and verify there's no duplicates
        all_components = [
            component for column in self.columns.values() for component in column
        ]
        all_component_names = [
            comp.__class__.__name__.rstrip("Config") for comp in all_components
        ]
        assert len(all_component_names) == len(
            set(all_component_names)
        ), "Duplicate components in layout config"
        self.components: dict[str, BaseComponentConfig] = {
            name: comp for name, comp in zip(all_component_names, all_components)
        }

        # Once we've verified this, store each config component as an attribute
        for comp, comp_name in zip(all_components, all_component_names):
            match comp_name:
                case "Prompt":
                    self.prompt_cfg = comp
                case "Sequences":
                    self.seq_cfg = comp
                case "ActsHistogram":
                    self.act_hist_cfg = comp
                case "LogitsHistogram":
                    self.logits_hist_cfg = comp
                case "LogitsTable":
                    self.logits_table_cfg = comp
                case "FeatureTables":
                    self.feature_tables_cfg = comp
                case _:
                    raise ValueError(f"Unknown component name {comp_name}")

    def data_is_contained_in(self, other: "SaeVisLayoutConfig") -> bool:
        """
        Returns True if `self` uses only data that would already exist in `other`. This is useful because our prompt-
        centric vis needs to only use data that was already computed as part of our initial data gathering. For example,
        if our SaeVisData object only contains the first 10 rows of the logits table, then we can't show the top 15 rows
        in the prompt centric view!
        """
        for comp_name, comp in self.components.items():
            # If the component in `self` is not present in `other`, return False
            if comp_name not in other.components:
                return False
            # If the component in `self` is present in `other`, but the `self` component is larger, then return False
            comp_other = other.components[comp_name]
            if not comp.data_is_contained_in(comp_other):
                return False

        return True

    def help(
        self,
        title: str = "SaeVisLayoutConfig",
        key: bool = True,
    ) -> Tree | None:
        """
        This prints out a tree showing the layout of the vis, by column (as well as the values of the arguments for each
        config object, plus their default values if they changed, and the descriptions of each arg).
        """

        # Create tree (with title and optionally the key explaining arguments)
        if key:
            title += "\n\n" + KEY_LAYOUT_VIS
        tree = Tree(title)

        n_columns = len(self.columns)

        # For each column, add a tree node
        for column_idx, vis_components in self.columns.items():
            n_components = len(vis_components)
            tree_column = tree.add(f"Column {column_idx}")

            # For each component in that column, add a tree node
            for component_idx, vis_component in enumerate(vis_components):
                n_params = len(asdict(vis_component))
                tree_component = tree_column.add(
                    f"{vis_component.__class__.__name__}".rstrip("Config")
                )

                # For each config parameter of that component
                for param_idx, (param, value) in enumerate(
                    asdict(vis_component).items()
                ):
                    # Get line break if we're at the final parameter of this component (unless it's the final component
                    # in the final column)
                    suffix = "\n" if (param_idx == n_params - 1) else ""
                    if (component_idx == n_components - 1) and (
                        column_idx == n_columns - 1
                    ):
                        suffix = ""

                    # Get argument description, and its default value
                    desc = vis_component.help_dict.get(param, "")
                    value_default = getattr(
                        vis_component.__class__, param, "no default"
                    )

                    # Add tree node (appearance is different if value is changed from default)
                    if value != value_default:
                        info = f"[b dark_orange]{param}: {value!r}[/] ({value_default!r}) \n[i]{desc}[/i]{suffix}"
                    else:
                        info = (
                            f"[b #00aa00]{param}: {value!r}[/] \n[i]{desc}[/i]{suffix}"
                        )
                    tree_component.add(info)

        rprint(tree)

    @classmethod
    def default_feature_centric_layout(cls) -> "SaeVisLayoutConfig":
        return cls(
            columns=[
                Column(FeatureTablesConfig()),
                Column(
                    ActsHistogramConfig(), LogitsTableConfig(), LogitsHistogramConfig()
                ),
                Column(SequencesConfig(stack_mode="stack-none")),
            ],
            height=750,
        )

    @classmethod
    def default_prompt_centric_layout(cls) -> "SaeVisLayoutConfig":
        return cls(
            columns=[
                Column(
                    PromptConfig(),
                    ActsHistogramConfig(),
                    LogitsTableConfig(n_rows=5),
                    SequencesConfig(top_acts_group_size=10, n_quantiles=0),
                    width=450,
                ),
            ],
            height=1000,
        )
