from dataclasses import dataclass

DEFAULT_SPARSITY_THRESHOLD = -6


@dataclass
class NeuronpediaRunnerConfig:
    sae_set: str
    sae_path: str
    outputs_dir: str
    np_sae_id_suffix: str | None = (
        None  # this is the __[np_sae_id_suffix] after the SAE Set
    )
    np_set_name: str | None = None
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
    end_batch: int | None = None

    # quantiles
    n_quantiles: int = 5
    top_acts_group_size: int = 30
    quantile_group_size: int = 5

    # additional calculations
    use_dfa: bool = False

    model_dtype: str = ""
    sae_dtype: str = ""
    model_id: str | None = None
    layer: int | None = None

    sae_device: str | None = None
    activation_store_device: str | None = None
    model_device: str | None = None
    model_n_devices: int | None = None
    use_wandb: bool = False

    shuffle_tokens: bool = True
    prefix_tokens: list[int] | None = None
    suffix_tokens: list[int] | None = None
    ignore_positions: list[int] | None = None

    hf_model_path: str | None = None


@dataclass
class NeuronpediaVectorRunnerConfig:
    # Vector loading parameters
    outputs_dir: str  # Where to save outputs
    vector_names: list[str] | None = None  # Names for each vector (optional)

    # Token generation parameters
    n_prompts_total: int = 24576
    n_tokens_in_prompt: int = 128
    n_prompts_in_forward_pass: int = 32
    prepend_bos: bool = True  # TODO: eventually include this in vector set export
    prepend_chat_template_text: str | None = None

    # Batching parameters
    n_vectors_at_a_time: int = 128  # Similar to n_features_at_a_time
    quantile_vector_batch_size: int = 64
    start_batch: int = 0
    end_batch: int | None = None

    # additional calculations
    use_dfa: bool = False
    include_original_vectors_in_output: bool = False
    activation_thresholds: dict[int, float | int] | None = None
    # Quantile parameters for activation analysis
    n_quantiles: int = 5
    top_acts_group_size: int = 30
    quantile_group_size: int = 5

    # Device and dtype settings
    model_dtype: str = ""
    vector_dtype: str = ""
    model_id: str | None = None
    layer: int | None = None
    activation_store_device: str | None = None
    model_device: str | None = None
    vector_device: str | None = None
    model_n_devices: int | None = None

    # Dataset parameters
    huggingface_dataset_path: str = ""

    # Additional settings
    use_wandb: bool = False
    shuffle_tokens: bool = True
    prefix_tokens: list[int] | None = None
    suffix_tokens: list[int] | None = None
    ignore_positions: list[int] | None = None
