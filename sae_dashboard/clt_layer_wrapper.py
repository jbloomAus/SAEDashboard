import json

# Added dataclass, field, asdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

# import torch.nn as nn # Unused
# from torch.distributed import ProcessGroup # Unused
# from types import SimpleNamespace # Unused import
from typing import (  # Added Optional, Union and List
    TYPE_CHECKING,
    List,
    Optional,
    Union,
)

import torch
import torch.nn.functional as F
from clt.models.activations import BatchTopK  # type: ignore

if TYPE_CHECKING:
    import torch.distributed  # Import for ProcessGroup type hint
    from clt.models.clt import CrossLayerTranscoder  # type: ignore


# Placeholder for dist if torch.distributed is not available or initialized
class MockDist:
    def is_initialized(self) -> bool:
        return False

    def get_world_size(
        self, group: "Optional[torch.distributed.ProcessGroup]" = None
    ) -> int:
        return 1

    def all_gather_into_tensor(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        group: "Optional[torch.distributed.ProcessGroup]" = None,
    ) -> None:
        # In non-distributed setting, just copy input to output (assuming output is sized correctly)
        if output_tensor.shape[0] == 1 * input_tensor.shape[0]:
            output_tensor.copy_(input_tensor)
        else:
            # This case shouldn't happen if called correctly, but handle defensively
            raise ValueError(
                "Output tensor size doesn't match input tensor size in mock all_gather"
            )

    def all_gather(
        self,
        tensor_list: List[torch.Tensor],
        input_tensor: torch.Tensor,
        group: "Optional[torch.distributed.ProcessGroup]" = None,
    ) -> None:
        """Mock all_gather for a list of tensors."""
        if self.get_world_size(group) == 1:
            if len(tensor_list) == 1:
                tensor_list[0].copy_(input_tensor)
            else:
                raise ValueError(
                    "tensor_list size must be 1 in mock all_gather when world_size is 1"
                )
        else:
            # This mock doesn't support actual gathering for world_size > 1.
            # It's primarily for the dist.all_gather call in _gather_weight,
            # which should ideally not proceed if world_size > 1 and dist is MockDist.
            # However, _gather_weight checks dist.is_initialized() and dist.get_world_size() first.
            raise NotImplementedError(
                "MockDist.all_gather not implemented for world_size > 1"
            )


try:
    import torch.distributed as dist

    if not dist.is_available():
        dist = MockDist()  # type: ignore
except ImportError:
    dist = MockDist()  # type: ignore


@dataclass
class CLTMetadata:
    """Simple metadata class for CLT wrapper compatibility."""

    hook_name: str
    hook_layer: int
    model_name: Optional[str] = None
    context_size: Optional[int] = None
    prepend_bos: bool = True
    hook_head_index: Optional[int] = None
    seqpos_slice: Optional[slice] = None


@dataclass
class CLTWrapperConfig:
    """Configuration dataclass for the CLTLayerWrapper."""

    # Fields without defaults first
    d_sae: int
    d_in: int
    hook_name: str
    hook_layer: int
    dtype: str
    device: str
    # Fields with defaults last
    architecture: str = "jumprelu"
    hook_head_index: Optional[int] = None
    model_name: Optional[str] = None
    dataset_path: Optional[str] = None
    context_size: Optional[int] = None
    prepend_bos: bool = True
    normalize_activations: bool = False
    dataset_trust_remote_code: bool = False
    seqpos_slice: Optional[slice] = None
    model_from_pretrained_kwargs: dict = field(default_factory=dict)
    metadata: Optional[CLTMetadata] = None

    def to_dict(self) -> dict:
        """Convert config to dictionary for compatibility with SAE interface."""
        return asdict(self)


class CLTLayerWrapper:
    """Wraps a single layer of a CrossLayerTranscoder to mimic the SAE interface.

    This allows reusing existing dashboard components that expect an SAE object.
    It specifically provides access to the encoder and the *same-layer* decoder weights
    for the specified layer index.
    """

    cfg: CLTWrapperConfig  # Add type hint for the config attribute
    threshold: Optional[torch.Tensor] = (
        None  # For JumpReLU, set by FeatureMaskingContext
    )

    def __init__(
        self,
        clt: "CrossLayerTranscoder",
        layer_idx: int,
        clt_model_dir_path: Optional[str] = None,
    ):
        self.clt = clt
        self.layer_idx = layer_idx
        self.device = clt.device
        self.dtype = clt.dtype
        self.hook_z_reshaping_mode = False  # Added to satisfy SAE interface

        # Validate layer index
        if not (0 <= layer_idx < clt.config.num_layers):
            raise ValueError(
                f"Invalid layer_idx {layer_idx} for CLT with {clt.config.num_layers} layers."
            )

        # --- Create the Wrapper Config ---
        # Try to get model_name from the underlying clt config if it exists
        clt_model_name = getattr(clt.config, "model_name", None)
        clt_dataset_path = getattr(clt.config, "dataset_path", None)
        clt_context_size = getattr(
            clt.config, "context_size", 128
        )  # Default to 128 if not set
        clt_prepend_bos = getattr(clt.config, "prepend_bos", True)
        # Use the activation_fn from CLT config for the wrapper's architecture and encode method
        self.activation_fn = getattr(clt.config, "activation_fn", "jumprelu")
        clt_model_from_pretrained_kwargs = getattr(
            clt.config, "model_from_pretrained_kwargs", {}
        )

        # --- Load CLT-specific normalization stats if applicable ---
        self.clt_norm_mean: Optional[torch.Tensor] = None
        self.clt_norm_std: Optional[torch.Tensor] = None
        wrapper_will_normalize_specifically = False
        clt_norm_method = getattr(clt.config, "normalization_method", "none")

        if clt_norm_method in ["auto", "estimated_mean_std", "mean_std"]:
            if clt_model_dir_path:
                norm_stats_file = Path(clt_model_dir_path) / "norm_stats.json"
                if norm_stats_file.exists():
                    try:
                        with open(norm_stats_file, "r") as f:
                            stats_data = json.load(f)

                        layer_stats = stats_data.get(str(self.layer_idx), {}).get(
                            "inputs", {}
                        )
                        mean_vals = layer_stats.get("mean")
                        std_vals = layer_stats.get("std")

                        if mean_vals is not None and std_vals is not None:
                            self.clt_norm_mean = torch.tensor(
                                mean_vals, device=self.device, dtype=torch.float32
                            ).unsqueeze(0)
                            self.clt_norm_std = (
                                torch.tensor(
                                    std_vals, device=self.device, dtype=torch.float32
                                )
                                + 1e-6
                            ).unsqueeze(0)
                            if torch.any(self.clt_norm_std <= 0):
                                print(
                                    f"Warning: Loaded std for layer {self.layer_idx} contains non-positive values after adding epsilon. Disabling specific normalization."
                                )
                                self.clt_norm_mean = None
                                self.clt_norm_std = None
                            else:
                                wrapper_will_normalize_specifically = True
                                print(
                                    f"CLTLayerWrapper: Loaded norm_stats.json for layer {self.layer_idx}. Wrapper will apply specific normalization."
                                )
                        else:
                            print(
                                f"Warning: norm_stats.json found, but missing 'mean' or 'std' for layer {self.layer_idx} inputs. Wrapper will not normalize specifically."
                            )
                    except Exception as e:
                        print(
                            f"Warning: Error loading or parsing norm_stats.json from {norm_stats_file}: {e}. Wrapper will not normalize specifically."
                        )
                else:
                    print(
                        f"Warning: normalization_method is '{clt_norm_method}' but norm_stats.json not found at {norm_stats_file}. Wrapper will not normalize specifically."
                    )
            else:
                print(
                    f"Warning: normalization_method is '{clt_norm_method}' but clt_model_dir_path not provided. Wrapper cannot load norm_stats.json and will not normalize specifically."
                )

        # Determine normalize_activations flag for ActivationsStore based on CLT config and wrapper's capability
        # This flag in self.cfg controls ActivationsStore. ActivationsStore should only normalize if the wrapper *isn't* doing specific normalization AND the CLT expected some form of normalization.
        clt_config_indicated_normalization = clt_norm_method != "none"
        normalize_activations_for_store = clt_config_indicated_normalization and (
            not wrapper_will_normalize_specifically
        )
        if normalize_activations_for_store:
            print(
                f"CLTLayerWrapper: Setting normalize_activations=True for ActivationsStore (CLT method: {clt_norm_method}, wrapper specific norm: False)."
            )
        elif clt_config_indicated_normalization and wrapper_will_normalize_specifically:
            print(
                f"CLTLayerWrapper: Setting normalize_activations=False for ActivationsStore (CLT method: {clt_norm_method}, wrapper specific norm: True)."
            )
        else:  # not clt_config_indicated_normalization
            print(
                f"CLTLayerWrapper: Setting normalize_activations=False for ActivationsStore (CLT method: {clt_norm_method})."
            )

        # Initialize self.threshold if activation is jumprelu
        # This must happen AFTER self.activation_fn, self.device, self.dtype, self.layer_idx, and self.clt are set.
        if self.activation_fn == "jumprelu":
            if (
                hasattr(self.clt, "log_threshold")
                and self.clt.log_threshold is not None
            ):
                if 0 <= self.layer_idx < self.clt.log_threshold.shape[0]:
                    # The log_threshold from CLT is [num_layers, num_features]
                    # We need the threshold for the current layer_idx
                    layer_thresholds = torch.exp(
                        self.clt.log_threshold[self.layer_idx].clone().detach()
                    )
                    self.threshold = layer_thresholds.to(
                        device=self.device, dtype=self.dtype
                    )
                    print(
                        f"CLTLayerWrapper: Initialized self.threshold for layer {self.layer_idx} from clt.log_threshold."
                    )
                else:
                    print(
                        f"Warning: CLTLayerWrapper layer_idx {self.layer_idx} is out of bounds for clt.log_threshold "
                        f"(shape {self.clt.log_threshold.shape}). self.threshold will be None."
                    )
                    self.threshold = None
            else:
                print(
                    f"Warning: Underlying CLT model for layer {self.layer_idx} does not have 'log_threshold' or it's None, "
                    f"but activation_fn is 'jumprelu'. self.threshold will be None."
                )
                self.threshold = None
        # else: self.threshold remains its default None, which is fine for other activation functions.

        # Get the hook name using prioritized templates
        hook_name_template = getattr(clt.config, "tl_input_template", None)
        if hook_name_template:
            hook_name = hook_name_template.format(layer_idx)
            print(f"Using TL hook name template: {hook_name_template} -> {hook_name}")
        else:
            hook_name_template = getattr(clt.config, "mlp_input_template", None)
            if hook_name_template:
                hook_name = hook_name_template.format(layer_idx)
                print(
                    f"Warning: tl_input_template not found. Using mlp_input_template: {hook_name_template} -> {hook_name}"
                )
            else:
                # Fallback for older configs without any template
                hook_name = f"blocks.{layer_idx}.hook_mlp_in"
                print(
                    f"Warning: Neither tl_input_template nor mlp_input_template found. Falling back to hardcoded: {hook_name}"
                )


        self.cfg = CLTWrapperConfig(
            d_sae=clt.config.num_features,  # This is the d_sae of the *entire* CLT layer, not a sub-batch
            d_in=clt.config.d_model,
            hook_name=hook_name,
            hook_layer=layer_idx,
            hook_head_index=None,
            dtype=str(self.dtype).replace("torch.", ""),
            device=str(self.device),
            architecture=self.activation_fn,  # Use the determined activation_fn
            model_name=clt_model_name,
            dataset_path=clt_dataset_path,
            context_size=clt_context_size,
            prepend_bos=clt_prepend_bos,
            normalize_activations=normalize_activations_for_store,
            dataset_trust_remote_code=False,
            seqpos_slice=None,
            model_from_pretrained_kwargs=clt_model_from_pretrained_kwargs,
            metadata=CLTMetadata(
                hook_name=hook_name,
                hook_layer=layer_idx,
                model_name=clt_model_name,
                context_size=clt_context_size,
                prepend_bos=clt_prepend_bos,
                hook_head_index=None,
                seqpos_slice=None,
            ),
        )
        # --- End Config Creation ---

        # Extract and potentially gather weights
        # Ensure weights are detached and cloned to avoid modifying the original CLT
        # Original W_enc from CLT encoder module is [d_sae_layer, d_model]
        # We transpose to match sae-lens W_enc convention: [d_model, d_sae_layer]
        self.W_enc = (
            self._gather_encoder_weight(clt.encoder_module.encoders[layer_idx].weight)
            .t()
            .contiguous()
        )
        # For W_dec, use the decoder from the same layer to itself
        decoder_key = f"{layer_idx}->{layer_idx}"
        if decoder_key not in clt.decoder_module.decoders:
            raise KeyError(f"Decoder key {decoder_key} not found in CLT decoders.")
        # Original W_dec from CLT decoder module is [d_model, d_sae_layer]
        # We transpose to match sae-lens W_dec convention: [d_sae_layer, d_model]
        self.W_dec = (
            self._gather_decoder_weight(clt.decoder_module.decoders[decoder_key].weight)
            .t()
            .contiguous()
        )

        self.b_enc = self._gather_encoder_bias(
            clt.encoder_module.encoders[layer_idx].bias_param
        )
        # For b_dec, use the bias from the same-layer decoder
        self.b_dec = self._gather_decoder_bias(
            clt.decoder_module.decoders[decoder_key].bias_param
        )

        # Cache for folded weights if needed
        self._W_dec_folded = False
        # Thresholds for JumpReLU will be handled by FeatureMaskingContext if architecture is 'jumprelu'
        # by setting self.threshold directly on the wrapper instance.

    # --- Façade methods mimicking SAE --- #
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes input using the CLTLayerWrapper's own W_enc and b_enc,
        respecting masks applied by FeatureMaskingContext.
        Applies the activation function specified in self.activation_fn.
        """
        # x is [..., d_model]
        # self.W_enc after masking (by FeatureMaskingContext) should be [d_model, N_FEATURES_IN_BATCH]
        # self.b_enc after masking (by FeatureMaskingContext) should be [N_FEATURES_IN_BATCH]

        original_shape = x.shape
        if x.ndim > 2:  # Ensure x is [N, d_model] for F.linear
            # self.cfg.d_in should be d_model
            x_reshaped = x.reshape(-1, self.cfg.d_in)
        else:
            x_reshaped = x

        x_to_process = x_reshaped
        # Apply CLT-specific normalization if stats were loaded
        if self.clt_norm_mean is not None and self.clt_norm_std is not None:
            # Ensure calculation is done in float32 for precision, then cast back
            x_float32 = x_to_process.to(torch.float32)
            normalized_x = (x_float32 - self.clt_norm_mean) / self.clt_norm_std
            x_to_process = normalized_x.to(x.dtype)

        # F.linear(input, weight, bias) expects weight to be [out_features, in_features]
        # self.W_enc is [d_model, N_FEATURES_IN_BATCH], so its transpose is [N_FEATURES_IN_BATCH, d_model]
        hidden_pre = F.linear(
            x_to_process, self.W_enc.T, self.b_enc
        )  # Output: [N, N_FEATURES_IN_BATCH]

        # Apply activation function
        if self.activation_fn == "relu":
            encoded_acts = torch.relu(hidden_pre)
        elif self.activation_fn == "jumprelu":
            if not hasattr(self, "threshold") or self.threshold is None:
                raise AttributeError(
                    "JumpReLU activation selected, but 'self.threshold' is not available on CLTLayerWrapper. "
                    "FeatureMaskingContext should set this if architecture is 'jumprelu'."
                )
            encoded_acts = torch.where(
                hidden_pre > self.threshold, hidden_pre, torch.zeros_like(hidden_pre)
            )
        elif self.activation_fn == "batchtopk":
            k_val: float
            batchtopk_k_abs = getattr(self.clt.config, "batchtopk_k", None)
            batchtopk_k_frac = getattr(self.clt.config, "batchtopk_frac", None)

            if batchtopk_k_abs is not None:
                # This k is global. For the current batch of features, we use a per-layer approximation.
                k_val = float(batchtopk_k_abs) / self.clt.config.num_layers
                k_val = max(
                    1.0, k_val
                )  # Ensure at least 1 feature is kept if k/num_layers is small
            elif batchtopk_k_frac is not None:
                k_val = float(
                    batchtopk_k_frac
                )  # Fraction applies directly to current N_FEATURES_IN_BATCH
            else:
                # Fallback: if neither k nor frac is specified, keep all features currently being processed.
                # This matches the fallback in CrossLayerTranscoder.encode for its per-layer batchtopk.
                print(
                    f"Warning: CLTLayerWrapper using batchtopk, but neither 'batchtopk_k' nor 'batchtopk_frac' defined in CLTConfig. Defaulting to keeping all {hidden_pre.size(-1)} features in the current batch."
                )
                k_val = float(hidden_pre.size(-1))

            straight_through_flag = getattr(
                self.clt.config, "batchtopk_straight_through", False
            )
            encoded_acts = BatchTopK.apply(hidden_pre, k_val, straight_through_flag)
        else:
            raise ValueError(
                f"Unsupported activation function in CLTLayerWrapper: {self.activation_fn}"
            )

        if x.ndim > 2:
            # Reshape back to original batch/sequence dimensions, with the last dim being N_FEATURES_IN_BATCH
            encoded_acts = encoded_acts.reshape(*original_shape[:-1], -1)

        return encoded_acts

    def turn_off_forward_pass_hook_z_reshaping(self):
        """Stub method to satisfy SAE interface. CLTWrapper does not use this."""
        # This mode is not applicable to CLTLayerWrapper, so this method is a no-op.
        pass

    # Note: CLTLayerWrapper does not have a separate `decode` method façade
    # because the dashboard primarily uses W_dec directly for analysis (e.g., logits).
    # The CLT's actual decode logic (summing across layers) isn't needed here.

    def fold_W_dec_norm(self):
        """Folds the L2 norm of W_dec into W_enc and b_enc.

        Mirrors the logic in sae_lens.SAE.fold_W_dec_norm.
        Important for ensuring that W_enc activations directly correspond
        to the output norm when using the wrapped W_dec.
        """
        if self._W_dec_folded:
            print("Warning: W_dec norm already folded.")
            return

        if self.W_dec is None or self.W_enc is None:
            print("Warning: Cannot fold W_dec norm, weights not available.")
            return

        # Detach W_dec before calculating norm to avoid gradient issues
        # W_dec is [N_FEATURES_IN_BATCH, d_model] (after masking context and init)
        # Norm should be taken over d_model dim (dim=1)

        # Use W_dec with its original dtype for norm calculation
        w_dec_for_norm = self.W_dec.detach()
        w_dec_norms = torch.norm(
            w_dec_for_norm, dim=1, keepdim=True
        )  # [N_FEATURES_IN_BATCH, 1]

        w_dec_norms = torch.where(
            w_dec_norms == 0, torch.ones_like(w_dec_norms), w_dec_norms
        )

        # self.W_enc is [d_model, N_FEATURES_IN_BATCH]
        # We want to scale each column of W_enc (each feature's encoder vector)
        # by the corresponding feature's w_dec_norm.
        # Ensure dtypes match for multiplication, then cast W_enc back if necessary
        original_w_enc_dtype = self.W_enc.dtype
        self.W_enc.data = (self.W_enc.data.to(w_dec_norms.dtype) * w_dec_norms.t()).to(
            original_w_enc_dtype
        )

        if self.b_enc is not None:
            # self.b_enc is [N_FEATURES_IN_BATCH]
            # w_dec_norms.squeeze() is [N_FEATURES_IN_BATCH]
            original_b_enc_dtype = self.b_enc.dtype
            self.b_enc.data = (
                self.b_enc.data.to(w_dec_norms.dtype) * w_dec_norms.squeeze()
            ).to(original_b_enc_dtype)

        # Store the norms for potential unfolding or reference
        self._w_dec_norms_backup = w_dec_norms
        self._W_dec_folded = True
        print("Folded W_dec norm into W_enc and b_enc.")

    def unfold_W_dec_norm(self):
        """Unfolds the L2 norm of W_dec from W_enc and b_enc."""
        if not self._W_dec_folded or not hasattr(self, "_w_dec_norms_backup"):
            print("Warning: W_dec norm not folded or backup norms not found.")
            return

        if self.W_enc is None:
            print("Warning: Cannot unfold W_dec norm, W_enc not available.")
            return

        # Retrieve the norms used for folding
        w_dec_norms = self._w_dec_norms_backup
        # Avoid division by zero (should have been handled in fold, but double check)
        w_dec_norms = torch.where(
            w_dec_norms == 0, torch.ones_like(w_dec_norms), w_dec_norms
        )

        original_w_enc_dtype = self.W_enc.dtype
        self.W_enc.data = (self.W_enc.data.to(w_dec_norms.dtype) / w_dec_norms.t()).to(
            original_w_enc_dtype
        )

        if self.b_enc is not None:
            original_b_enc_dtype = self.b_enc.dtype
            self.b_enc.data = (
                self.b_enc.data.to(w_dec_norms.dtype) / w_dec_norms.squeeze()
            ).to(original_b_enc_dtype)

        del self._w_dec_norms_backup
        self._W_dec_folded = False
        print("Unfolded W_dec norm from W_enc and b_enc.")

    def to(self, device: Union[str, torch.device]):
        """Moves the wrapper and underlying components to the specified device."""
        target_device = torch.device(device)

        # Move the underlying CLT model
        try:
            self.clt.to(target_device)
        except Exception as e:
            print(
                f"Warning: Failed to move underlying CLT model to {target_device}: {e}"
            )
            # Continue trying to move wrapper components

        # Move the wrapper's stored tensors
        if self.W_enc is not None:
            self.W_enc = self.W_enc.to(target_device)
        if self.W_dec is not None:
            self.W_dec = self.W_dec.to(target_device)
        if self.b_enc is not None:
            self.b_enc = self.b_enc.to(target_device)
        if self.b_dec is not None:
            self.b_dec = self.b_dec.to(target_device)
        if (
            hasattr(self, "_w_dec_norms_backup")
            and self._w_dec_norms_backup is not None
        ):
            self._w_dec_norms_backup = self._w_dec_norms_backup.to(target_device)

        # Update device attributes
        self.device = target_device
        self.cfg.device = str(target_device)

        # Update activation_fn related thresholds if they exist (e.g. for JumpReLU)
        if hasattr(self, "threshold") and self.threshold is not None:
            self.threshold = self.threshold.to(target_device)

        if self.clt_norm_mean is not None:  # Added to move norm stats
            self.clt_norm_mean = self.clt_norm_mean.to(target_device)
        if self.clt_norm_std is not None:  # Added to move norm stats
            self.clt_norm_std = self.clt_norm_std.to(target_device)

        print(f"Moved CLTLayerWrapper to {target_device}")
        return self

    # --- Helper methods for Tensor Parallelism --- #

    def _gather_weight(
        self,
        weight_shard: torch.Tensor,
        gather_dim: int = 0,
        target_full_dim_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Gather a weight tensor shard across TP ranks."""
        if not dist.is_initialized() or dist.get_world_size() == 1:
            return weight_shard.clone().detach()

        world_size = dist.get_world_size()
        # Create a list to hold all gathered tensors
        tensor_list = [torch.empty_like(weight_shard) for _ in range(world_size)]
        dist.all_gather(tensor_list, weight_shard)

        # Concatenate along the specified dimension
        full_weight = torch.cat(tensor_list, dim=gather_dim)

        # Trim padding if necessary
        if target_full_dim_size is not None:
            if gather_dim == 0:
                if full_weight.shape[0] > target_full_dim_size:
                    full_weight = full_weight[:target_full_dim_size, :]
            elif gather_dim == 1:
                if full_weight.shape[1] > target_full_dim_size:
                    full_weight = full_weight[:, :target_full_dim_size]
            # Add other gather_dim cases if needed

        return full_weight.detach()

    def _gather_encoder_weight(self, weight_shard: torch.Tensor) -> torch.Tensor:
        """Gather ColumnParallelLinear weight (sharded along output/feature dim)."""
        # ColumnParallel weight is [d_sae_local, d_model]
        # We need to gather along dim 0 to get [d_sae_full_for_layer, d_model]
        return self._gather_weight(
            weight_shard,
            gather_dim=0,
            target_full_dim_size=self.clt.config.num_features,
        )

    def _gather_decoder_weight(self, weight_shard: torch.Tensor) -> torch.Tensor:
        """Gather RowParallelLinear weight (sharded along input/feature dim)."""
        # RowParallel weight is [d_model, d_sae_local]
        # We need to gather along dim 1 to get [d_model, d_sae_full_for_layer]
        return self._gather_weight(
            weight_shard,
            gather_dim=1,
            target_full_dim_size=self.clt.config.num_features,
        )

    def _gather_bias(
        self,
        bias_shard: Optional[torch.Tensor],
        gather_dim: int = 0,
        target_full_dim_size: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        """Gather a bias tensor shard across TP ranks."""
        if bias_shard is None:
            return None
        # Biases are typically sharded along the same dimension as the weight's corresponding output dim
        return self._gather_weight(
            bias_shard, gather_dim=gather_dim, target_full_dim_size=target_full_dim_size
        )

    def _gather_encoder_bias(
        self, bias_shard_candidate: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """Gather ColumnParallelLinear bias (sharded along output/feature dim).

        Defensively checks if the provided candidate is actually a Tensor.
        """
        # Check if the provided object is a Tensor
        if isinstance(bias_shard_candidate, torch.Tensor):
            # Encoder bias shape [d_sae_local], gather along dim 0
            return self._gather_bias(
                bias_shard_candidate,
                gather_dim=0,
                target_full_dim_size=self.clt.config.num_features,
            )
        else:
            # If it's None, bool, or anything else, treat as no bias
            return None

    def _gather_decoder_bias(
        self, bias_shard_candidate: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """Gather RowParallelLinear bias (NOT sharded, but might need broadcast/check).

        Defensively checks if the provided candidate is actually a Tensor.
        """
        # Check if the provided object is a Tensor
        if isinstance(bias_shard_candidate, torch.Tensor):
            # RowParallelLinear bias is typically not sharded (added after all-reduce)
            # However, let's check world size and return a clone if TP=1, or verify replication if TP>1
            if not dist.is_initialized() or dist.get_world_size() == 1:
                return bias_shard_candidate.clone().detach()

            # In TP > 1, the bias should be identical across ranks. Verify this.
            world_size = dist.get_world_size()
            tensor_list = [
                torch.empty_like(bias_shard_candidate) for _ in range(world_size)
            ]
            dist.all_gather(tensor_list, bias_shard_candidate)
            # Check if all gathered biases are the same
            for i in range(1, world_size):
                if not torch.equal(tensor_list[0], tensor_list[i]):
                    raise RuntimeError(
                        "RowParallelLinear bias shards are not identical across TP ranks, which is unexpected."
                    )
            # Return the bias from rank 0 (or any rank, as they are identical)
            return tensor_list[0].clone().detach()
        else:
            # If it's None, bool, or anything else, treat as no bias
            return None
