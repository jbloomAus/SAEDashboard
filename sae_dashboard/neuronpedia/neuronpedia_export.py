"""Convert SAEDashboard ``batch-*.json`` outputs to the Neuronpedia upload
format.

This module is the in-process replacement for the standalone
``convert-saedashboard-to-neuronpedia-export.py`` script. It reads the per-batch
JSON files emitted by :class:`NeuronpediaRunner` and produces the directory
layout expected by Neuronpedia's bulk-import tooling:

    {exports_dir}/{model_id}/{source_id}/
        release.jsonl
        model.jsonl
        sourceset.jsonl
        source.jsonl
        features/{batch_name}.jsonl.gz
        activations/{batch_name}.jsonl.gz

The dataclasses here mirror those in the ``neuronpedia-utils`` package
(``neuronpedia_utils.db_models.*``). They are inlined intentionally because
``neuronpedia-utils`` itself depends on ``sae-dashboard``; importing it as a
dependency here would create a circular dependency.
"""

from __future__ import annotations

import gzip
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, List, Optional

import orjson

# Hardcoded fallback creator ID used by Neuronpedia bulk-imports when no
# ``DEFAULT_CREATOR_ID`` env var or explicit override is provided. Matches the
# value baked into the standalone convert script.
FALLBACK_CREATOR_ID = "clkht01d40000jv08hvalcvly"

# Tokens that are treated as BOS for the optional ``zero_out_bos_token``
# behaviour. Models like Gemma 2 weren't trained with a BOS token, so its
# activation should be zeroed out before display.
BOS_TOKEN_LITERALS = ("<bos>", "<|endoftext|>")


# ---------------------------------------------------------------------------
# Inlined neuronpedia-utils dataclasses
# ---------------------------------------------------------------------------


@dataclass
class Activation:
    id: str
    tokens: List[str]
    index: str
    layer: str
    modelId: str
    maxValue: float
    maxValueTokenIndex: int
    minValue: float
    values: List[float]
    creatorId: str
    dataSource: Optional[str] = None
    dataIndex: Optional[str] = None
    dfaValues: List[float] = field(default_factory=list)
    dfaTargetIndex: Optional[int] = None
    dfaMaxValue: Optional[float] = None
    createdAt: datetime = field(default_factory=datetime.now)
    lossValues: List[float] = field(default_factory=list)
    logitContributions: Optional[str] = None
    binMin: Optional[float] = None
    binMax: Optional[float] = None
    binContains: Optional[float] = None
    qualifyingTokenIndex: Optional[int] = None
    zIndices: Optional[List[List[int]]] = None
    zValues: Optional[List[float]] = None


@dataclass
class Feature:
    modelId: str
    layer: str
    index: str
    creatorId: Optional[str] = None
    createdAt: datetime = field(default_factory=datetime.now)
    maxActApprox: Optional[float] = 0
    hasVector: bool = False
    vector: List[float] = field(default_factory=list)
    vectorLabel: Optional[str] = None
    vectorDefaultSteerStrength: Optional[float] = 10
    hookName: Optional[str] = None
    topkCosSimIndices: List[int] = field(default_factory=list)
    topkCosSimValues: List[float] = field(default_factory=list)
    neuron_alignment_indices: List[int] = field(default_factory=list)
    neuron_alignment_values: List[float] = field(default_factory=list)
    neuron_alignment_l1: List[float] = field(default_factory=list)
    correlated_neurons_indices: List[int] = field(default_factory=list)
    correlated_neurons_pearson: List[float] = field(default_factory=list)
    correlated_neurons_l1: List[float] = field(default_factory=list)
    correlated_features_indices: List[int] = field(default_factory=list)
    correlated_features_pearson: List[float] = field(default_factory=list)
    correlated_features_l1: List[float] = field(default_factory=list)
    neg_str: List[str] = field(default_factory=list)
    neg_values: List[float] = field(default_factory=list)
    pos_str: List[str] = field(default_factory=list)
    pos_values: List[float] = field(default_factory=list)
    frac_nonzero: float = 0
    freq_hist_data_bar_heights: List[float] = field(default_factory=list)
    freq_hist_data_bar_values: List[float] = field(default_factory=list)
    logits_hist_data_bar_heights: List[float] = field(default_factory=list)
    logits_hist_data_bar_values: List[float] = field(default_factory=list)
    decoder_weights_dist: List[float] = field(default_factory=list)
    sourceSetName: Optional[str] = None


@dataclass
class Model:
    id: str
    instruct: bool
    creatorId: str
    displayNameShort: str = ""
    displayName: str = ""
    tlensId: Optional[str] = None
    dimension: Optional[int] = None
    visibility: str = "PUBLIC"
    defaultSourceSetName: Optional[str] = None
    defaultSourceId: Optional[str] = None
    inferenceEnabled: bool = False
    layers: int = 0
    neuronsPerLayer: int = 0
    createdAt: datetime = field(default_factory=datetime.now)
    owner: str = ""
    updatedAt: datetime = field(default_factory=datetime.now)
    website: Optional[str] = None


@dataclass
class Source:
    id: str
    modelId: str
    setName: str
    creatorId: str
    hasDashboards: bool = True
    inferenceEnabled: bool = False
    inferenceHosts: List[str] = field(default_factory=list)
    saelensConfig: Optional[str] = None
    saelensRelease: Optional[str] = None
    saelensSaeId: Optional[str] = None
    hfRepoId: Optional[str] = None
    hfFolderId: Optional[str] = None
    visibility: str = "PUBLIC"
    defaultOfModelId: Optional[str] = None
    hasUmap: bool = False
    hasUmapLogSparsity: bool = False
    hasUmapClusters: bool = False
    num_prompts: Optional[int] = None
    num_tokens_in_prompt: Optional[int] = None
    dataset: Optional[str] = None
    notes: Optional[str] = None
    createdAt: datetime = field(default_factory=datetime.now)


@dataclass
class SourceRelease:
    name: str
    description: str
    creatorName: str
    creatorId: str
    visibility: str = "PUBLIC"
    isNewUi: bool = False
    featured: bool = False
    descriptionShort: Optional[str] = None
    urls: List[str] = field(default_factory=list)
    creatorEmail: Optional[str] = None
    creatorNameShort: Optional[str] = None
    defaultSourceSetName: Optional[str] = None
    defaultSourceId: Optional[str] = None
    defaultUmapSourceIds: List[str] = field(default_factory=list)
    createdAt: datetime = field(default_factory=datetime.now)


@dataclass
class SourceSet:
    modelId: str
    name: str
    creatorId: str
    hasDashboards: bool = True
    visibility: str = "PRIVATE"
    description: str = ""
    type: str = ""
    creatorName: str = ""
    urls: List[str] = field(default_factory=list)
    creatorEmail: Optional[str] = None
    releaseName: Optional[str] = None
    defaultOfModelId: Optional[str] = None
    defaultRange: int = 1
    defaultShowBreaks: bool = True
    showDfa: bool = False
    showCorrelated: bool = False
    showHeadAttribution: bool = False
    showUmap: bool = False
    createdAt: datetime = field(default_factory=datetime.now)


# ---------------------------------------------------------------------------
# ID generator
# ---------------------------------------------------------------------------


class FastPseudoCuid:
    """Counter-based pseudo-CUID generator (~100x faster than ``cuid2``).

    Produces strings of the form ``c<pid_hex><start_time_hex><counter_hex>``
    truncated to ``length``. The values are not collision-resistant across
    machines, but are sufficient for assigning per-row activation IDs in a
    single export run.
    """

    def __init__(self, length: int = 25):
        self.length = length
        self._counter = 0
        self._prefix = f"c{os.getpid():04x}{int(time.time()):08x}"

    def generate(self) -> str:
        self._counter += 1
        return f"{self._prefix}{self._counter:08x}"[: self.length]


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

# orjson serializes ``datetime`` natively. Dataclasses are serialized via
# ``row.__dict__`` to mirror the original
# ``convert-saedashboard-to-neuronpedia-export.py`` script.


def _dump_jsonl(rows: List[Any], path: str) -> None:
    with open(path, "wb") as f:
        for row in rows:
            f.write(orjson.dumps(row.__dict__))
            f.write(b"\n")


def _gzip_in_place(path: str, compresslevel: int = 5) -> None:
    """Gzip ``path`` to ``path + '.gz'`` and delete the original."""
    with open(path, "rb") as f_in:
        data = f_in.read()
    with open(path + ".gz", "wb") as f_out:
        f_out.write(gzip.compress(data, compresslevel=compresslevel))
    os.remove(path)


# ---------------------------------------------------------------------------
# Export config
# ---------------------------------------------------------------------------


@dataclass
class NeuronpediaExportConfig:
    """Configuration for the post-runner Neuronpedia export step."""

    # Where the runner emitted its ``batch-*.json`` files.
    saedashboard_output_dir: str

    # Where to write the converted Neuronpedia layout. The function will
    # create ``{exports_dir}/{model_name}/{source_id}/...`` underneath this.
    exports_dir: str

    # Author / release metadata.
    creator_name: str
    creator_id: str
    release_id: str
    release_title: str
    url: str

    # Model / source identifiers.
    model_name: str
    neuronpedia_source_set_id: str
    neuronpedia_source_set_description: str
    hf_weights_repo_id: str
    hf_weights_path: str
    hook_point: str
    layer_num: int

    # Dashboard generation parameters (recorded on the Source row).
    prompts_huggingface_dataset_path: str
    n_prompts_total: int
    n_tokens_in_prompt: int

    # If True, zero out activations on tokens matching ``BOS_TOKEN_LITERALS``.
    zero_out_bos_token: bool = False


# ---------------------------------------------------------------------------
# Export entrypoint
# ---------------------------------------------------------------------------


def _build_source_id(layer_num: int, source_set_id: str, suffix: Optional[str]) -> str:
    base = f"{layer_num}-{source_set_id}"
    return base + (f"__{suffix}" if suffix else "")


def _hook_name_for_feature(layer_num: int, hook_point: str) -> str:
    return f"blocks.{layer_num}.{hook_point}"


def _strip_blocks_prefix(hook_name: str) -> str:
    """Strip ``blocks.<layer>.`` from a TransformerLens hook name.

    Examples:
        ``blocks.17.hook_resid_post`` -> ``hook_resid_post``
        ``blocks.20.attn.hook_z`` -> ``attn.hook_z``
        ``hook_resid_post`` -> ``hook_resid_post`` (already stripped)
    """
    return re.sub(r"^blocks\.\d+\.", "", hook_name)


def export_neuronpedia_dashboards(cfg: NeuronpediaExportConfig) -> str:
    """Convert ``batch-*.json`` files in ``cfg.saedashboard_output_dir`` to the
    Neuronpedia bulk-import layout under ``cfg.exports_dir``.

    Returns the directory containing the most recently written source (i.e.
    ``{exports_dir}/{model_name}/{source_id}``). The release / model /
    source-set jsonl files are written once (idempotent if they already
    exist); ``source.jsonl`` is rewritten on every batch since it's small and
    cheap.
    """
    cuid_generator = FastPseudoCuid(length=25)
    created_at = datetime.now()

    output_path_base = os.path.join(cfg.exports_dir, cfg.model_name)
    os.makedirs(output_path_base, exist_ok=True)

    hf_folder_id = "/".join(cfg.hf_weights_path.split("/")[:-1])

    batch_files = sorted(
        f
        for f in os.listdir(cfg.saedashboard_output_dir)
        if f.startswith("batch-") and f.endswith(".json")
    )
    if not batch_files:
        raise FileNotFoundError(
            f"No batch-*.json files found in {cfg.saedashboard_output_dir!r}; "
            "run the dashboard generator first."
        )

    final_output_dir = ""
    for batch_file in batch_files:
        batch_path = os.path.join(cfg.saedashboard_output_dir, batch_file)
        print(f"reading activations from batch file {batch_file}")
        with open(batch_path, "rb") as f:
            batch_data = orjson.loads(f.read())

        source_suffix = batch_data.get("sae_id_suffix")
        source_id = _build_source_id(
            cfg.layer_num, cfg.neuronpedia_source_set_id, source_suffix
        )
        final_output_dir = os.path.join(output_path_base, source_id)
        os.makedirs(final_output_dir, exist_ok=True)

        _maybe_write_release(cfg, final_output_dir, created_at)
        _maybe_write_model(cfg, final_output_dir, created_at)
        _maybe_write_sourceset(cfg, final_output_dir, created_at)
        _write_source(cfg, final_output_dir, source_id, hf_folder_id)

        _process_batch(
            cfg=cfg,
            batch_data=batch_data,
            batch_file_name=batch_file.replace(".json", ""),
            source_id=source_id,
            source_dir=final_output_dir,
            cuid_generator=cuid_generator,
            created_at=created_at,
        )

    print(
        "\n==================== Neuronpedia export complete. ====================\n"
        f"Exports written to: {final_output_dir}"
    )
    return final_output_dir


def _maybe_write_release(
    cfg: NeuronpediaExportConfig, source_dir: str, created_at: datetime
) -> None:
    path = os.path.join(source_dir, "release.jsonl")
    if os.path.exists(path):
        return
    release = SourceRelease(
        name=cfg.release_id,
        description=cfg.release_title,
        descriptionShort=cfg.release_title,
        urls=[cfg.url] if cfg.url else [],
        creatorNameShort=cfg.creator_name,
        creatorName=cfg.creator_name,
        creatorId=cfg.creator_id,
        createdAt=created_at,
    )
    _dump_jsonl([release], path)


def _maybe_write_model(
    cfg: NeuronpediaExportConfig, source_dir: str, created_at: datetime
) -> None:
    path = os.path.join(source_dir, "model.jsonl")
    if os.path.exists(path):
        return
    model = Model(
        id=cfg.model_name,
        instruct=cfg.model_name.endswith("-it"),
        displayNameShort=cfg.model_name,
        displayName=cfg.model_name,
        creatorId=cfg.creator_id,
        createdAt=created_at,
        updatedAt=created_at,
    )
    _dump_jsonl([model], path)


def _maybe_write_sourceset(
    cfg: NeuronpediaExportConfig, source_dir: str, created_at: datetime
) -> None:
    path = os.path.join(source_dir, "sourceset.jsonl")
    if os.path.exists(path):
        return
    sourceset = SourceSet(
        modelId=cfg.model_name,
        name=cfg.neuronpedia_source_set_id,
        creatorId=cfg.creator_id,
        createdAt=created_at,
        creatorName=cfg.creator_name,
        releaseName=cfg.release_id,
        description=cfg.neuronpedia_source_set_description,
        visibility="PUBLIC",
    )
    _dump_jsonl([sourceset], path)


def _write_source(
    cfg: NeuronpediaExportConfig,
    source_dir: str,
    source_id: str,
    hf_folder_id: str,
) -> None:
    path = os.path.join(source_dir, "source.jsonl")
    source = Source(
        modelId=cfg.model_name,
        setName=cfg.neuronpedia_source_set_id,
        visibility="PUBLIC",
        dataset=cfg.prompts_huggingface_dataset_path,
        id=source_id,
        num_prompts=cfg.n_prompts_total,
        num_tokens_in_prompt=cfg.n_tokens_in_prompt,
        hfRepoId=cfg.hf_weights_repo_id,
        hfFolderId=hf_folder_id,
        creatorId=cfg.creator_id,
    )
    _dump_jsonl([source], path)


def _process_batch(
    cfg: NeuronpediaExportConfig,
    batch_data: dict[str, Any],
    batch_file_name: str,
    source_id: str,
    source_dir: str,
    cuid_generator: FastPseudoCuid,
    created_at: datetime,
) -> None:
    activations: List[Activation] = []
    features: List[Feature] = []

    for feature_data in batch_data["features"]:
        has_vector = bool(feature_data.get("vector"))
        new_feature = Feature(
            modelId=cfg.model_name,
            layer=source_id,
            index=feature_data["feature_index"],
            creatorId=cfg.creator_id,
            createdAt=created_at,
            hasVector=has_vector,
            vector=feature_data.get("vector", []),
            vectorLabel=None,
            hookName=(
                _hook_name_for_feature(cfg.layer_num, cfg.hook_point)
                if "vector" in feature_data
                else None
            ),
            topkCosSimIndices=[],
            topkCosSimValues=[],
            neuron_alignment_indices=feature_data["neuron_alignment_indices"],
            neuron_alignment_values=feature_data["neuron_alignment_values"],
            neuron_alignment_l1=feature_data["neuron_alignment_l1"],
            correlated_neurons_indices=feature_data["correlated_neurons_indices"],
            correlated_neurons_pearson=feature_data["correlated_neurons_pearson"],
            correlated_neurons_l1=feature_data["correlated_neurons_l1"],
            correlated_features_indices=feature_data["correlated_features_indices"],
            correlated_features_pearson=feature_data["correlated_features_pearson"],
            correlated_features_l1=feature_data["correlated_features_l1"],
            neg_str=feature_data["neg_str"],
            neg_values=feature_data["neg_values"],
            pos_str=feature_data["pos_str"],
            pos_values=feature_data["pos_values"],
            frac_nonzero=feature_data["frac_nonzero"],
            freq_hist_data_bar_heights=feature_data["freq_hist_data_bar_heights"],
            freq_hist_data_bar_values=feature_data["freq_hist_data_bar_values"],
            logits_hist_data_bar_heights=feature_data["logits_hist_data_bar_heights"],
            logits_hist_data_bar_values=feature_data["logits_hist_data_bar_values"],
            decoder_weights_dist=feature_data["decoder_weights_dist"],
        )

        max_act_approx = 0.0
        for activation_data in feature_data["activations"]:
            if cfg.zero_out_bos_token:
                _zero_out_bos_inplace(
                    activation_data,
                    source_id=source_id,
                    feature_index=feature_data["feature_index"],
                    batch_file_name=batch_file_name,
                )

            values = activation_data["values"]
            max_value = max(values)
            max_value_token_index = values.index(max_value)
            if max_value > max_act_approx:
                max_act_approx = max_value

            activations.append(
                Activation(
                    id=cuid_generator.generate(),
                    tokens=activation_data["tokens"],
                    modelId=cfg.model_name,
                    layer=source_id,
                    index=feature_data["feature_index"],
                    maxValue=max_value,
                    maxValueTokenIndex=max_value_token_index,
                    minValue=min(values),
                    values=values,
                    dfaValues=activation_data.get("dfa_values", []),
                    dfaTargetIndex=activation_data.get("dfa_targetIndex"),
                    dfaMaxValue=activation_data.get("dfa_maxValue"),
                    creatorId=cfg.creator_id,
                    createdAt=created_at,
                    lossValues=activation_data.get("loss_values", []),
                    logitContributions=activation_data.get("logit_contributions"),
                    binContains=activation_data["bin_contains"],
                    binMax=activation_data["bin_max"],
                    binMin=activation_data["bin_min"],
                    qualifyingTokenIndex=activation_data["qualifying_token_index"],
                    dataIndex=None,
                    dataSource=None,
                )
            )

        new_feature.maxActApprox = max_act_approx
        if has_vector:
            new_feature.vectorDefaultSteerStrength = max_act_approx
        features.append(new_feature)

    features_dir = os.path.join(source_dir, "features")
    os.makedirs(features_dir, exist_ok=True)
    features_file_path = os.path.join(features_dir, f"{batch_file_name}.jsonl")
    _dump_jsonl(features, features_file_path)
    _gzip_in_place(features_file_path)

    activations_dir = os.path.join(source_dir, "activations")
    os.makedirs(activations_dir, exist_ok=True)
    activations_file_path = os.path.join(activations_dir, f"{batch_file_name}.jsonl")
    _dump_jsonl(activations, activations_file_path)
    _gzip_in_place(activations_file_path)


def _zero_out_bos_inplace(
    activation_data: dict[str, Any],
    *,
    source_id: str,
    feature_index: Any,
    batch_file_name: str,
) -> None:
    tokens = activation_data["tokens"]
    values = activation_data["values"]
    for i, token in enumerate(tokens):
        if token in BOS_TOKEN_LITERALS and values[i] != 0:
            print(
                f"Zeroing out BOS token {token} at index {i}, "
                f"source_id: {source_id}, feature_index: {feature_index}, "
                f"file: {batch_file_name}"
            )
            values[i] = 0


def resolve_creator_id(explicit: Optional[str]) -> str:
    """Resolve the Neuronpedia creator ID.

    Resolution order:
        1. Explicit ``--neuronpedia-creator-id`` (or config) value.
        2. ``DEFAULT_CREATOR_ID`` environment variable.
        3. ``FALLBACK_CREATOR_ID`` (with a printed warning).
    """
    if explicit:
        return explicit
    env_value = os.getenv("DEFAULT_CREATOR_ID")
    if env_value:
        return env_value
    print(
        "WARNING: no --neuronpedia-creator-id or DEFAULT_CREATOR_ID env var set; "
        f"falling back to {FALLBACK_CREATOR_ID!r}."
    )
    return FALLBACK_CREATOR_ID


def derive_hook_point_from_hook_name(hook_name: str) -> str:
    """Convert a TransformerLens hook name to the bare hook-point suffix.

    See :func:`_strip_blocks_prefix` for examples.
    """
    return _strip_blocks_prefix(hook_name)
