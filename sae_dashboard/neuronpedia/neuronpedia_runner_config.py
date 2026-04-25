from dataclasses import dataclass
from typing import List, Optional

DEFAULT_SPARSITY_THRESHOLD = -6


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

    # ACTIVATION STORE PARAMETERS
    # token pars
    n_prompts_total: int = 24576
    n_tokens_in_prompt: int = 128
    n_prompts_in_forward_pass: int = 32

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

    rounding_precision: int = 3  # Number of decimal places for rounding

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
    rounding_precision: int = 3  # Number of decimal places for rounding
    shuffle_tokens: bool = True
    prefix_str: Optional[str] = None
    suffix_str: Optional[str] = None
    ignore_positions: Optional[List[int]] = None
