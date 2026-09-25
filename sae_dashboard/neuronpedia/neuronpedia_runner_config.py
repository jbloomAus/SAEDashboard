import warnings
from dataclasses import dataclass, field
from typing import Any, List, Literal, Optional

#: Rows per Parquet row group for columnar artifacts.
#:
#: Parquet readers prune at ROW GROUP granularity, not page granularity, so a file written as a
#: single row group costs a reader the whole file to fetch one feature -- a page index does not
#: change that. Measured on a published 4,096-feature dashboard artifact (35.6 MiB, ~34 rows per
#: feature), fetching every column one feature needs:
#:
#:   row_group_size   file size vs 1 row group   bytes read for one feature
#:   1 (previous)     --                         35.6 MiB  (100%)
#:   1024             +20.3%                      0.82 MiB   (1.9%)
#:   4096             +6.0%                       1.30 MiB   (3.4%)
#:   8192             -9.8%                       2.30 MiB   (7.2%)
#:
#: 4096 sits in the flat part of that curve for both 4,096- and 16,384-feature artifacts (the
#: optimum moves only as sqrt(features per file), so it is insensitive to batch size). Larger row
#: groups actually SHRINK the file, because a single 139k-row group defeats dictionary encoding.
#:
#: Like ``write_page_index``, this is fixed at write time: it cannot be changed without rewriting
#: the file.
DEFAULT_PARQUET_ROW_GROUP_SIZE = 4096

DEFAULT_SPARSITY_THRESHOLD = -6
DEFAULT_PROMPT_BUCKET_SCALE_LIMIT = 4.0
DEFAULT_PROMPT_PRIMARY_ACTS_SCALE_LIMIT = 4.0
DEFAULT_PROMPT_BATCH_SIZE_ROUND_TO = 8
LEGACY_DASHBOARD_PATH_DEPRECATION_MESSAGE = (
    "The legacy JSON dashboard path (dashboard_output_format='legacy_json' with "
    "sequence_selection_backend='legacy') is deprecated and retained only for compatibility/baseline checks. "
    "Prefer dashboard_output_format='columnar' with sequence_selection_backend='columnar_gpu' for new runs."
)


def is_legacy_dashboard_path(cfg: "NeuronpediaRunnerConfig") -> bool:
    return (
        cfg.dashboard_output_format == "legacy_json"
        and cfg.sequence_selection_backend == "legacy"
    )


def warn_if_deprecated_legacy_dashboard_path(cfg: "NeuronpediaRunnerConfig") -> None:
    if is_legacy_dashboard_path(cfg):
        warnings.warn(
            LEGACY_DASHBOARD_PATH_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=2
        )


@dataclass
class NeuronpediaRunnerConfig:
    sae_set: str
    sae_path: str
    outputs_dir: str
    np_sae_id_suffix: Optional[str] = (
        None  # this is the __[np_sae_id_suffix] after the SAE Set
    )
    np_set_name: Optional[str] = None
    from_local_sae: bool = False
    sparsity_threshold: int = DEFAULT_SPARSITY_THRESHOLD
    huggingface_dataset_path: str = ""
    huggingface_dataset_config_name: Optional[str] = None
    huggingface_dataset_split: Optional[str] = None
    huggingface_dataset_text_field: Optional[str] = None
    # Prompt dataset contract:
    # - load_dataset for Hub datasets and local/file-backed builders
    # - load_from_disk for local Dataset.save_to_disk() prompt caches
    # - legacy_jsonl only for deprecated JSONL dashboard exports
    # pretokenized_dataset_path still implies load_from_disk when reused as prompt_dataset_path.
    prompt_dataset_mode: str = "load_dataset"
    prompt_dataset_path: Optional[str] = None
    prompt_dataset_name: Optional[str] = None
    prompt_dataset_split: Optional[str] = None
    prompt_dataset_text_field: Optional[str] = None
    prompt_dataset_data_files: tuple[str, ...] = field(default_factory=tuple)
    prompt_dataset_data_dir: Optional[str] = None
    prompt_dataset_metadata_path: Optional[str] = None
    prompt_dataset_trust_remote_code: Optional[bool] = None
    pretokenized_dataset_path: Optional[str] = None
    # shared_tokens_file points at the staged tokens_*.pt tensor used by all layer runs. If omitted and
    # pretokenized_dataset_path is set, the runner generates tokens_*.pt, tokens_*.effective_lengths.pt, and
    # tokens_*.metadata.json beside that pretokenized dataset.
    shared_tokens_file: Optional[str] = None
    deduplicate_shared_prompt_tokens: bool = True
    strict_shared_prompt_count: bool = False
    # prompt_bucket_schedule_file is an explicit schedule artifact. auto_prompt_bucket_schedule derives the same
    # scheduling structure directly from the staged effective-length sidecar when no schedule file is supplied.
    prompt_bucket_schedule_file: Optional[str] = None
    auto_prompt_bucket_schedule: bool = False
    # Optional explicit inclusive ceilings for auto prompt bucketing. When left empty, the runner derives ceilings
    # from prompt-length quantiles in the staged effective-length sidecar.
    prompt_bucket_ceilings: tuple[int, ...] = field(default_factory=tuple)
    prompt_bucket_scale_limit: float = DEFAULT_PROMPT_BUCKET_SCALE_LIMIT
    prompt_primary_acts_scale_limit: float = DEFAULT_PROMPT_PRIMARY_ACTS_SCALE_LIMIT
    prompt_batch_size_round_to: int = DEFAULT_PROMPT_BATCH_SIZE_ROUND_TO
    dataset_streaming: bool = True

    # ACTIVATION STORE PARAMETERS
    # token pars
    n_prompts_total: int = 24576
    n_tokens_in_prompt: int = 128
    n_prompts_in_forward_pass: int = 32
    primary_acts_batch_size: Optional[int] = None

    # batching
    n_features_at_a_time: int = 128
    quantile_feature_batch_size: int = 64
    start_batch: int = 0
    end_batch: Optional[int] = None

    # quantiles
    n_quantiles: int = 5
    top_acts_group_size: int = 20
    quantile_group_size: int = 5

    # additional calculations
    use_dfa: bool = False

    model_dtype: str = ""
    sae_dtype: str = ""
    model_id: Optional[str] = None
    layer: Optional[int] = None

    sae_device: str | None = None
    activation_store_device: str | None = None
    model_device: str | None = None
    model_n_devices: int | None = None
    use_wandb: bool = False

    shuffle_tokens: bool = True
    prefix_str: Optional[str] = None
    suffix_str: Optional[str] = None
    ignore_positions: Optional[List[int]] = None
    prepend_bos: Optional[bool] = None  # Override SAE default if specified

    # If set, filter out activations at token positions whose hidden-state norm
    # exceeds `median_norm * ignore_high_activation_norm_multiple` (computed
    # per forward-pass minibatch). Useful for models like Qwen which exhibit
    # random, unpredictable high-norm activation "sinks" hundreds of tokens
    # into the sequence that otherwise dominate max activating examples and
    # correlation statistics. A typical value is 10. None disables filtering.
    ignore_high_activation_norm_multiple: Optional[float] = None

    # If True, replace transformer blocks above the SAE's hook layer with
    # ``nn.Identity()`` after loading the model. This frees VRAM since those
    # blocks are never executed (the forward pass already uses
    # ``stop_at_layer=hook_layer + 1`` in the TransformerLens wrapper). The
    # embedding, ``ln_final``, and ``unembed`` (``W_U``) layers are preserved
    # for logit-direction calculations. Defaults to False to preserve the
    # previous behaviour; enable to reduce memory usage when generating
    # dashboards for SAEs on early/middle layers of large models.
    free_unused_model_layers: bool = False

    hf_model_path: Optional[str] = None
    # Name the capture location explicitly, overriding the hook name the SAE declares. Needed where
    # an SAE's declared TransformerLens name denotes a different tensor from the one it was trained
    # on; see `resolve_capture_hook_name`. Unset means "trust the SAE's metadata", which is correct
    # for every release whose declared name is accurate.
    capture_hook_name: Optional[str] = None
    model_wrapper: str = "hooked"
    bridge_enable_compatibility_mode: bool = True
    bridge_compatibility_mode_kwargs: dict[str, Any] = field(
        default_factory=lambda: {"no_processing": True}
    )
    log_resource_snapshots: bool = False
    log_hook_aliases: bool = False
    log_performance: bool = False
    profile_rolling_substages: bool = False
    cleanup_each_minibatch: bool = False
    correlation_accumulation_device: Literal["auto", "cpu", "cuda"] = "auto"
    rolling_coefficient_num_threads: Optional[int] = None
    activation_significance_floor: float = 0.0
    converter_input_artifact_dir: Optional[str] = None
    sequence_replay_artifact_dir: Optional[str] = None
    feature_statistics_backend: Literal["object", "arrow"] = "arrow"
    logits_histogram_backend: Literal["object", "arrow"] = "arrow"
    activation_histogram_backend: Literal["torch"] = "torch"
    defer_component_construction: bool = False
    sequence_selection_backend: Literal["legacy", "columnar_gpu"] = "legacy"
    # Opt-in selection hygiene (columnar backend only; the legacy lane keeps its
    # historical selection semantics). All default off to preserve the
    # preserved-baseline selection/parity contract.
    sequence_top_acts_positive_only: bool = False
    sequence_dedup_across_groups: bool = False
    sequence_skip_dead_features: bool = False
    # Opt-in columnar peak-GPU-memory controls (bit-identical outputs at any setting;
    # None keeps the historical fixed 4 GiB device budgets / chunk shapes). The byte
    # budget caps device retention/staging of the activation matrix (0 forces host
    # staging); the row chunk bounds the per-chunk packaging AND batched-selection
    # transients on dense layers.
    columnar_max_device_staged_acts_bytes: int | None = None
    columnar_row_chunk_size: int | None = None
    # Numpy-histogram-style interval membership ([lower, upper), highest interval closed) for the
    # in-tree legacy and columnar selectors; the preserved pre-PR lane is untouched.
    sequence_half_open_interval_bins: bool = False
    # Optional regex over token strings; matching vocab rows are excluded from logits
    # tables on the columnar path (e.g. Gemma byte-fallback/unused rows).
    logits_table_mask_token_pattern: Optional[str] = None
    dashboard_output_format: Literal["legacy_json", "columnar"] = "legacy_json"
    columnar_artifact_format: Literal["arrow", "parquet"] = "arrow"
    columnar_write_page_index: bool = True
    columnar_parquet_row_group_size: Optional[int] = DEFAULT_PARQUET_ROW_GROUP_SIZE
    columnar_emit_sequence_rows: bool = False
    columnar_emit_activation_rows: bool = True
    columnar_emit_activation_copy_rows: bool = False
    # Overlap each batch's CPU packaging tail and artifact writes with the next batch's
    # forward/encode via a single background writer (columnar mode only). Batch
    # completion markers (per-batch root manifests) are still written in order, so
    # batch-level resume — including across GPUs — is unchanged.
    overlap_batch_packaging: bool = False
    columnar_activation_copy_model_id: Optional[str] = None
    torch_profile: bool = False
    torch_profile_dir: Optional[str] = None
    use_cached_activations: bool = True

    # If true, we load a Transcoder (inherits from SAE) instead of a standard SAE.
    use_transcoder: bool = False

    # If true, we load a SkipTranscoder (inherits from Transcoder) instead.
    use_skip_transcoder: bool = False

    # CLT (Cross-Layer Transcoder) specific parameters
    use_clt: bool = False
    clt_layer_idx: Optional[int] = None
    clt_dtype: str = ""
    # Optional filename for CLT weights (supports .safetensors or .pt). If empty, default search order will be used.
    clt_weights_filename: str = ""

    # Optional sae_lens loader/converter name used when loading SAEs from
    # HuggingFace (passed to SAE.from_pretrained's `converter` arg). Accepts a
    # short registry name from sae_lens' NAMED_PRETRAINED_SAE_LOADERS
    # (e.g. "dictionary_learning_1", "gemma_2", "sparsify",
    # "connor_rob_hook_z") or the full function name exported by
    # sae_lens.loading.pretrained_sae_loaders (e.g.
    # "dictionary_learning_sae_huggingface_loader_1"). If None, sae_lens infers
    # the loader from the release.
    sae_converter_name: Optional[str] = None

    # HuggingFace mode - use HuggingFace Transformers directly instead of TransformerLens
    use_huggingface: bool = False

    # ------------------------------------------------------------------
    # Neuronpedia bulk-import export options
    # ------------------------------------------------------------------
    # When True, after generating ``batch-*.json`` files, the runner will
    # additionally convert them to the Neuronpedia bulk-import directory
    # layout (release.jsonl / model.jsonl / sourceset.jsonl / source.jsonl
    # plus gzipped per-batch features and activations). See
    # ``sae_dashboard.neuronpedia.neuronpedia_export`` for details.
    output_neuronpedia_exports: bool = False

    # Where to write the converted exports. Defaults to
    # ``{outputs_dir}/../neuronpedia_exports`` when None.
    neuronpedia_exports_dir: Optional[str] = None

    # Author / release metadata. Required when ``output_neuronpedia_exports``
    # is True.
    neuronpedia_creator_name: Optional[str] = None
    neuronpedia_creator_id: Optional[str] = None  # falls back to env var
    neuronpedia_release_id: Optional[str] = None
    neuronpedia_release_title: Optional[str] = None
    neuronpedia_release_url: Optional[str] = None
    neuronpedia_source_set_description: Optional[str] = None

    # Model name written to the export. Used as ``Model.id``, the export
    # ``{model_name}/`` subdirectory, and ``modelId`` on every Feature /
    # Activation / Source / SourceSet row. Required when
    # ``output_neuronpedia_exports`` is True.
    neuronpedia_model_name: Optional[str] = None

    # Override values for the Source row. Defaults to ``sae_set`` /
    # ``sae_path`` respectively when None.
    neuronpedia_hf_weights_repo_id: Optional[str] = None
    neuronpedia_hf_weights_path: Optional[str] = None

    # If True, zero out activations on tokens matching ``<bos>`` /
    # ``<|endoftext|>`` in the exported activations. Useful for models like
    # Gemma 2 that weren't trained with a BOS token.
    neuronpedia_zero_out_bos_token: bool = False


@dataclass
class NeuronpediaVectorRunnerConfig:
    # Vector loading parameters
    outputs_dir: str  # Where to save outputs
    vector_names: Optional[List[str]] = None  # Names for each vector (optional)

    # Token generation parameters
    n_prompts_total: int = 24576
    n_tokens_in_prompt: int = 128
    n_prompts_in_forward_pass: int = 32
    prepend_bos: bool = True  # TODO: eventually include this in vector set export
    prepend_chat_template_text: Optional[str] = None

    # Batching parameters
    n_vectors_at_a_time: int = 128  # Similar to n_features_at_a_time
    quantile_vector_batch_size: int = 64
    start_batch: int = 0
    end_batch: Optional[int] = None

    # additional calculations
    use_dfa: bool = False
    include_original_vectors_in_output: bool = False
    activation_thresholds: Optional[dict[int, float | int]] = None
    # Quantile parameters for activation analysis
    n_quantiles: int = 5
    top_acts_group_size: int = 30
    quantile_group_size: int = 5

    # Device and dtype settings
    model_dtype: str = ""
    vector_dtype: str = ""
    model_id: Optional[str] = None
    layer: Optional[int] = None
    activation_store_device: str | None = None
    model_device: Optional[str] = None
    vector_device: Optional[str] = None
    model_n_devices: Optional[int] = None

    # Dataset parameters
    huggingface_dataset_path: str = ""

    # Additional settings
    use_wandb: bool = False
    shuffle_tokens: bool = True
    prefix_str: Optional[str] = None
    suffix_str: Optional[str] = None
    ignore_positions: Optional[List[int]] = None
