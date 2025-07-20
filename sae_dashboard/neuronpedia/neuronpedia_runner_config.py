from dataclasses import dataclass
from typing import List, Literal, Optional

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
    disable_concat_sequences: bool = False
    sequence_separator_token: int | Literal["bos", "eos", "sep"] | None = "bos"

    # batching
    n_features_at_a_time: int = 128
    quantile_feature_batch_size: int = 64
    start_batch: int = 0
    end_batch: Optional[int] = None

    # quantiles
    n_quantiles: int = 5
    top_acts_group_size: int = 30
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
    prefix_tokens: Optional[List[int]] = None
    suffix_tokens: Optional[List[int]] = None
    ignore_positions: Optional[List[int]] = None

    hf_model_path: Optional[str] = None


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
    prefix_tokens: Optional[List[int]] = None
    suffix_tokens: Optional[List[int]] = None
    ignore_positions: Optional[List[int]] = None
