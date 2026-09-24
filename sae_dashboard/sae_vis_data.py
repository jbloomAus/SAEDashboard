import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Literal

from dataclasses_json import dataclass_json
from rich import print as rprint
from rich.table import Table
from sae_lens import SAE

from sae_dashboard.feature_data import FeatureData
from sae_dashboard.layout import SaeVisLayoutConfig
from sae_dashboard.neuronpedia.neuronpedia_runner_config import (
    DEFAULT_PARQUET_ROW_GROUP_SIZE,
)
from sae_dashboard.utils_fns import FeatureStatistics

if TYPE_CHECKING:
    from transformer_lens import HookedTransformer

SAE_CONFIG_DICT = dict(
    hook_point="The hook point to use for the SAE",
    features="The set of features which we'll be gathering data for. If an integer, we only get data for 1 feature",
    batch_size="The number of sequences we'll gather data for. If supplied then it can't be larger than `tokens[0]`, \
if not then we use all of `tokens`",
    minibatch_size_tokens="The minibatch size we'll use to split up the full batch during forward passes, to avoid \
OOMs.",
    prompt_minibatch_schedule="Optional runner-resolved prompt schedule used to trim shorter prompt buckets during \
activation capture while preserving the original full-width output layout.",
    primary_acts_batch_size="Optional internal activation-capture chunk size used inside each token minibatch, to \
reduce peak model-forward memory without changing the dashboard minibatch shape.",
    minibatch_size_features="The feature minibatch size we'll use to split up our features, to avoid OOM errors",
    seed="Random seed, for reproducibility (e.g. sampling quantiles)",
    verbose="Whether to print out progress messages and other info during the data gathering process",
    log_performance="Whether to emit per-stage performance timings for dashboard generation",
    profile_rolling_substages="Whether to emit nested rolling-correlation substage timings and runtime metrics. Disabled by default so normal timing runs keep only the aggregate rolling stage.",
    cleanup_each_minibatch="Whether to run gc.collect() and torch.cuda.empty_cache() after each activation minibatch. This can reduce peak memory in constrained runs but is disabled by default because it slows benchmark generation.",
    sequence_replay_artifact_dir="Optional directory where per-feature-batch sequence replay bundles are written for offline get_indices_dict(...) replay.",
    torch_profile="Whether the Neuronpedia runner should wrap this run in torch.profiler",
    torch_profile_dir="Optional directory for torch profiler traces",
    correlation_accumulation_device="Policy for where correlation accumulators should live during packaging.",
    rolling_coefficient_num_threads="Optional torch intra-op thread count override applied only during rolling correlation updates.",
    feature_statistics_backend="Backend used to build columnar feature statistics when dashboard_output_format is columnar.",
    logits_histogram_backend="Backend used to build columnar logits histograms when dashboard_output_format is columnar.",
    activation_histogram_backend="Backend used to build positive-only activation histograms when dashboard_output_format is columnar.",
    defer_component_construction="Whether columnar/dashboard callers should avoid rebuilding the legacy nested component graph when not needed.",
    columnar_defer_batch_write="When set, SaeVisRunner.run returns SaeVisColumnarData with a pending_finalize callable that performs the CPU-side activation-row builds and all artifact/manifest writes when invoked, enabling callers to overlap them with the next batch's forward/encode.",
    sequence_selection_backend="Candidate-selection backend for sequence packaging.",
    dashboard_output_format="Output mode for dashboard generation: legacy JSON or importer-compatible columnar bundles.",
    columnar_artifact_dir="Root directory for columnar bundle output when dashboard_output_format is columnar.",
    columnar_artifact_format="On-disk format for columnar tables: Arrow IPC or Parquet.",
    columnar_write_page_index="Write a Parquet page index. Enables page-granular range reads for HTTP-streamed artifacts; cannot be added later without regenerating.",
    columnar_parquet_row_group_size="Rows per Parquet row group. Readers prune at ROW GROUP granularity, so one row group per file makes a single-feature read cost the whole file; like the page index this is fixed at write time. None keeps the pyarrow default (one row group per written table).",
    columnar_emit_activation_rows="Whether to emit semantic activation_rows tables alongside sequence_rows in columnar mode.",
    columnar_emit_activation_copy_rows="Whether to emit Neuronpedia Activation COPY-shaped activation_copy_rows in columnar mode.",
    columnar_activation_copy_model_id="Optional modelId override for activation_copy_rows payloads.",
    columnar_activation_copy_layer="Optional Neuronpedia source/layer id embedded into activation_copy_rows payloads.",
    columnar_activation_copy_creator_id="Optional creator id embedded into activation_copy_rows payloads.",
    columnar_activation_copy_created_at="Optional createdAt timestamp embedded into activation_copy_rows payloads.",
    columnar_activation_copy_id_prefix="Prefix used when synthesizing activation_copy_rows ids.",
)

OUT_OF_RANGE_TOKEN = "<|outofrange|>"


@dataclass_json
@dataclass
class SaeVisConfig:
    # Data
    hook_point: str
    features: Iterable[int]
    minibatch_size_features: int = 256
    minibatch_size_tokens: int = 64
    prompt_minibatch_schedule: list[dict[str, Any]] | None = None
    primary_acts_batch_size: int | None = None
    quantile_feature_batch_size: int = 64
    perform_ablation_experiments: bool = False
    device: str = "cpu"
    dtype: str = "float32"
    ignore_tokens: set[int] = field(default_factory=set)
    ignore_positions: list[int] = field(default_factory=list)
    # If set, filter out activations at token positions whose hidden-state norm
    # exceeds `median_norm * ignore_high_activation_norm_multiple` (per
    # forward-pass minibatch). Useful for models like Qwen which exhibit random
    # high-norm activation sinks deep in the sequence. None disables filtering.
    ignore_high_activation_norm_multiple: float | None = None

    # Model loading
    use_huggingface: bool = (
        False  # If True, use HuggingFace Transformers directly instead of TransformerLens
    )

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
    log_performance: bool = False
    profile_rolling_substages: bool = False
    cleanup_each_minibatch: bool = False
    sequence_replay_artifact_dir: Path | None = None
    torch_profile: bool = False
    torch_profile_dir: Path | None = None
    correlation_accumulation_device: Literal["auto", "cpu", "cuda"] = "auto"
    rolling_coefficient_num_threads: int | None = None
    activation_significance_floor: float = 0.0
    feature_statistics_backend: Literal["object", "arrow"] = "object"
    logits_histogram_backend: Literal["object", "arrow"] = "object"
    activation_histogram_backend: Literal["torch"] = "torch"
    defer_component_construction: bool = False
    columnar_defer_batch_write: bool = False
    sequence_selection_backend: Literal["legacy", "columnar_gpu"] = "legacy"
    # Opt-in selection hygiene for the columnar backend only (the deprecated legacy lane keeps its
    # historical selection semantics bit-for-bit). All three default off to preserve parity with the
    # preserved-baseline selection contract.
    sequence_top_acts_positive_only: bool = False
    sequence_dedup_across_groups: bool = False
    sequence_skip_dead_features: bool = False
    # Numpy-histogram-style interval membership: [lower, upper) for all quantile intervals except the
    # highest, which stays closed. Applies to the in-tree legacy AND columnar selectors (the preserved
    # pre-PR lane under neuronpedia/legacy is untouched). Off = historical double-inclusive bounds.
    sequence_half_open_interval_bins: bool = False
    # Optional regex over token strings (e.g. r"^<(0x[0-9A-Fa-f]{2}|unused\d+)>$"); matching vocab
    # rows are excluded from logits tables on the columnar path. None preserves unmasked logits.
    logits_table_mask_token_pattern: str | None = None
    # Opt-in columnar peak-GPU-memory controls (§ prompt-dimension scaling). None preserves the
    # historical fixed 4 GiB device budgets / packaging chunk shapes; outputs are bit-identical at
    # any setting (row/token chunking and host staging are exact), only peak memory and speed move.
    # columnar_max_device_staged_acts_bytes caps BOTH the generation-side device retention of the
    # (prompts, seq, feats) activation tensor AND the packaging-side full-matrix staging (0 forces
    # host staging); columnar_row_chunk_size overrides the feature-row chunk the arrow packaging
    # loops (defaults 256 stats / 128 histograms) AND the batched sequence selector (default 64)
    # iterate with, bounding the density/token-count-scaled per-chunk transients on dense layers.
    columnar_max_device_staged_acts_bytes: int | None = None
    columnar_row_chunk_size: int | None = None
    dashboard_output_format: Literal["legacy_json", "columnar"] = "legacy_json"
    columnar_artifact_dir: Path | None = None
    columnar_artifact_format: Literal["arrow", "parquet"] = "arrow"
    columnar_write_page_index: bool = True
    columnar_parquet_row_group_size: int | None = DEFAULT_PARQUET_ROW_GROUP_SIZE
    columnar_emit_sequence_rows: bool = False
    columnar_emit_activation_rows: bool = False
    columnar_emit_activation_copy_rows: bool = False
    columnar_activation_copy_model_id: str | None = None
    columnar_activation_copy_layer: str | None = None
    columnar_activation_copy_creator_id: str | None = None
    columnar_activation_copy_created_at: str | None = None
    columnar_activation_copy_id_prefix: str = "columnar-activation"
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
        for param, desc in SAE_CONFIG_DICT.items():
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


@dataclass
class SaeVisColumnarBatch:
    feature_batch_index: int
    feature_indices: list[int]
    artifact_dir: Path
    manifest_path: Path
    row_counts: dict[str, int]


@dataclass
class SaeVisColumnarData:
    cfg: SaeVisConfig
    artifact_dir: Path
    manifest_path: Path
    batches: list[SaeVisColumnarBatch]
    # When columnar_defer_batch_write is enabled, no artifacts have been written yet and
    # `batches` is empty; invoking pending_finalize performs the deferred CPU packaging
    # and all writes (root manifest last) and returns the completed SaeVisColumnarData.
    pending_finalize: "Callable[[], SaeVisColumnarData] | None" = None


@dataclass_json
@dataclass
class _SaeVisData:
    """
    Dataclass which is used to store the data for the SaeVisData class. It excludes everything which isn't easily
    serializable, only saving the raw data.
    """

    feature_data_dict: dict[int, FeatureData] = field(default_factory=dict)
    feature_stats: FeatureStatistics = field(default_factory=FeatureStatistics)

    @classmethod
    def from_dict(
        cls, data: dict[str, Any]
    ) -> (
        "_SaeVisData"
    ): ...  # just for type hinting; the method comes from 'dataclass_json'

    def to_dict(
        self,
    ) -> dict[
        str, Any
    ]: ...  # just for type hinting; the method comes from 'dataclass_json'


@dataclass
class SaeVisData:
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

    cfg: SaeVisConfig  # = field(default_factory=SaeVisConfig)
    feature_data_dict: dict[int, FeatureData] = field(default_factory=dict)
    feature_stats: FeatureStatistics = field(default_factory=FeatureStatistics)

    model: "HookedTransformer | None" = None
    encoder: SAE | None = None  # type: ignore

    def update(self, other: "SaeVisData") -> None:
        """
        Updates a SaeVisData object with the data from another SaeVisData object. This is useful during the
        `get_feature_data` function, since this function is broken up into different groups of features then merged
        together.
        """
        if other is None:
            return
        self.feature_data_dict.update(other.feature_data_dict)
        self.feature_stats.update(other.feature_stats)

    def save_json(self: "SaeVisData", filename: str | Path) -> None:
        """
        Saves an SaeVisData instance to a JSON file. The config, model & encoder arguments must be user-supplied.
        """
        if isinstance(filename, str):
            filename = Path(filename)
        assert filename.suffix == ".json", "Filename must have a .json extension"

        _self = _SaeVisData(
            feature_data_dict=self.feature_data_dict,
            feature_stats=self.feature_stats,
        )

        with open(filename, "w") as f:
            json.dump(_self.to_dict(), f)

    @classmethod
    def load_json(
        cls,
        filename: str | Path,
        cfg: SaeVisConfig,
        model: "HookedTransformer",
        encoder: SAE,  # type: ignore
    ) -> "SaeVisData":
        """
        Loads an SaeVisData instance from JSON file. The config, model & encoder arguments must be user-supplied.
        """
        if isinstance(filename, str):
            filename = Path(filename)
        assert filename.suffix == ".json", "Filename must have a .json extension"

        with open(filename) as f:
            data = json.load(f)

        _self = _SaeVisData.from_dict(data)

        self = SaeVisData(
            cfg=cfg,
            feature_data_dict=_self.feature_data_dict,
            feature_stats=_self.feature_stats,
            model=model,
            encoder=encoder,
        )

        return self
