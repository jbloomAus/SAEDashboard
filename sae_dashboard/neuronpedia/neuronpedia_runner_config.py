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
