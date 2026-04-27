"""
HuggingFace model wrapper for SAE Dashboard.

This module provides a wrapper for HuggingFace Transformers models that provides
a standardized interface compatible with SAE Dashboard's feature data generation,
without requiring TransformerLens.
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

from sae_dashboard.hook_utils import (
    HookInfo,
    detect_model_architecture,
    get_layer_module_path,
    get_submodule_by_path,
    transformer_lens_to_hf_hook,
)

DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


@dataclass
class HFActivationConfig:
    """Configuration for which activations to capture during forward pass."""

    primary_hook_point: (
        str  # TransformerLens-style hook name, e.g., "blocks.5.hook_resid_post"
    )
    auxiliary_hook_points: List[str]  # Additional hook points to capture


class StopForward(Exception):
    """Exception used to stop the forward pass early."""

    pass


class HuggingFaceModelWrapper(nn.Module):
    """
    Wrapper for HuggingFace Transformers models that provides a standardized interface
    compatible with SAE Dashboard's feature data generation.

    This wrapper mimics the interface of TransformerLensWrapper but uses native
    HuggingFace models and PyTorch hooks instead of TransformerLens.

    Args:
        model: A HuggingFace AutoModelForCausalLM instance
        tokenizer: The tokenizer for the model
        activation_config: Configuration specifying which activations to capture
        dtype: The data type to use for activations
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        activation_config: HFActivationConfig,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.model = model
        self._tokenizer = tokenizer
        self.activation_config = activation_config
        self.dtype = dtype

        # Parse hook points
        self.primary_hook_info = transformer_lens_to_hf_hook(
            activation_config.primary_hook_point
        )
        self.auxiliary_hook_infos = [
            transformer_lens_to_hf_hook(hp)
            for hp in activation_config.auxiliary_hook_points
        ]

        # Determine layer for early stopping
        self.hook_layer = self.primary_hook_info.layer_index

        # Detect model architecture and get layer path pattern
        self.model_architecture = detect_model_architecture(model)
        self._layer_path_template = self._detect_layer_path_template()

        # Get the unembedding matrix for logits computation
        self._W_U = self._get_unembed_matrix()

    def _detect_layer_path_template(self) -> str:
        """Detect the path template to transformer layers."""
        try:
            path = get_layer_module_path(self.model, 0)
            # Replace the layer index with a placeholder
            return re.sub(r"\.\d+$", ".{}", path)
        except ValueError:
            # Default to common pattern
            return "model.layers.{}"

    def _get_layer_module(self, layer_index: int):
        """Get the transformer layer module at the given index."""
        path = self._layer_path_template.format(layer_index)
        return get_submodule_by_path(self.model, path)

    def _get_unembed_matrix(self) -> Tensor:
        """Get the unembedding (lm_head) weight matrix."""
        # Try direct lm_head access
        if hasattr(self.model, "lm_head"):
            return self.model.lm_head.weight.data.T  # (d_model, vocab_size)
        # Try language_model.lm_head for multimodal models (e.g., Gemma 3)
        elif hasattr(self.model, "language_model") and hasattr(
            self.model.language_model, "lm_head"
        ):
            return self.model.language_model.lm_head.weight.data.T
        # Try get_output_embeddings
        elif hasattr(self.model, "get_output_embeddings"):
            output_embeddings = self.model.get_output_embeddings()
            if output_embeddings is not None:
                return output_embeddings.weight.data.T
        raise ValueError("Could not find unembedding matrix in model")

    @property
    def W_U(self) -> Tensor:
        """The unembedding matrix (d_model, vocab_size)."""
        return self._W_U

    @property
    def W_out(self):
        """MLP output weights - used for computing residual directions."""
        # This is architecture-dependent and may not be directly accessible
        # Return None for now; most SAE hooks are on residual stream where this isn't needed
        return None

    @property
    def W_O(self):
        """Attention output weights - used for computing residual directions."""
        # This is architecture-dependent and complex to extract
        # Return None for now; most SAE hooks are on residual stream where this isn't needed
        return None

    @property
    def tokenizer(self):
        """The tokenizer for this model."""
        return self._tokenizer

    @torch.inference_mode()
    def forward(
        self,
        tokens: Int[Tensor, "batch seq"],
        return_logits: bool = True,
    ) -> Dict[str, Tensor]:
        """
        Execute a forward pass and capture activations at specified hook points.

        Args:
            tokens: Input token IDs of shape (batch, seq_length)
            return_logits: Whether to include output logits in the result

        Returns:
            Dictionary mapping hook point names to activation tensors
        """
        activation_dict: Dict[str, Tensor] = {}
        handles: List[torch.utils.hooks.RemovableHandle] = []

        # Move tokens to model device
        device = next(self.model.parameters()).device
        tokens = tokens.to(device)

        try:
            # Register hooks for all hook points we need to capture
            all_hook_infos = [self.primary_hook_info] + self.auxiliary_hook_infos

            for hook_info in all_hook_infos:
                handle = self._register_activation_hook(hook_info, activation_dict)
                handles.append(handle)

            # Register hook for early stopping after the target layer
            stop_handle = self._register_stop_hook()
            handles.append(stop_handle)

            # Run forward pass
            try:
                outputs = self.model(input_ids=tokens, output_hidden_states=False)
            except StopForward:
                # Expected - we stopped early
                pass

        finally:
            # Always remove hooks
            for handle in handles:
                handle.remove()

        # Add logits to output if requested
        # Note: We won't have logits if we stopped early, which is expected
        if return_logits and "output" not in activation_dict:
            # If we need logits, we'd need to run a full forward pass
            # For now, we skip this as it's not always needed
            pass

        return activation_dict

    def _register_activation_hook(
        self, hook_info: HookInfo, activation_dict: Dict[str, Tensor]
    ) -> torch.utils.hooks.RemovableHandle:
        """Register a hook to capture activations."""

        # Get the module to hook
        module = self._get_module_for_hook(hook_info)

        def hook_fn(module, input, output):
            if hook_info.capture_output:
                # Capture output
                if isinstance(output, tuple) and hook_info.output_index is not None:
                    act = output[hook_info.output_index]
                else:
                    act = output
            else:
                # Capture input
                if isinstance(input, tuple):
                    act = input[0]
                else:
                    act = input

            # Ensure tensor and correct dtype
            if isinstance(act, Tensor):
                activation_dict[hook_info.transformer_lens_name] = act.to(self.dtype)

        return module.register_forward_hook(hook_fn)

    def _register_stop_hook(self) -> torch.utils.hooks.RemovableHandle:
        """Register a hook to stop the forward pass after the target layer."""
        # Get the module after which we want to stop
        layer_module = self._get_layer_module(self.hook_layer)

        def stop_hook(module, input, output):
            raise StopForward()

        return layer_module.register_forward_hook(stop_hook)

    def _get_module_for_hook(self, hook_info: HookInfo):
        """Get the module to register a hook on based on hook info."""
        hook_type = hook_info.hook_type

        if hook_type in ["hook_resid_post", "hook_resid_pre"]:
            # For residual stream hooks, we hook the entire layer
            return self._get_layer_module(hook_info.layer_index)

        elif hook_type == "hook_mlp_out":
            # Hook the MLP sublayer
            layer = self._get_layer_module(hook_info.layer_index)
            if hasattr(layer, "mlp"):
                return layer.mlp
            elif hasattr(layer, "feed_forward"):
                return layer.feed_forward
            else:
                raise ValueError(
                    f"Cannot find MLP module in layer {hook_info.layer_index}"
                )

        elif hook_type == "hook_attn_out":
            # Hook the attention sublayer
            layer = self._get_layer_module(hook_info.layer_index)
            if hasattr(layer, "self_attn"):
                return layer.self_attn
            elif hasattr(layer, "attention"):
                return layer.attention
            else:
                raise ValueError(
                    f"Cannot find attention module in layer {hook_info.layer_index}"
                )

        else:
            raise NotImplementedError(
                f"Hook type '{hook_type}' is not yet supported for HuggingFace models"
            )


def to_resid_direction_hf(
    direction: Float[Tensor, "feats d_in"],
    model: HuggingFaceModelWrapper,
) -> Float[Tensor, "feats d_model"]:
    """
    Convert a direction to residual stream space for HuggingFace models.

    For most residual stream hooks (hook_resid_post, hook_resid_pre), the direction
    is already in residual stream space. For other hooks, additional transformations
    may be needed.

    Args:
        direction: The direction tensor of shape (feats, d_in)
        model: The HuggingFaceModelWrapper instance

    Returns:
        The direction in residual stream space
    """
    hook_point = model.activation_config.primary_hook_point

    # For residual stream hooks, no transformation needed
    if "resid" in hook_point or "_out" in hook_point or "hook_mlp_in" in hook_point:
        return direction

    # For other hook types, we would need to apply transformations
    # This is currently not fully implemented for HuggingFace models
    # as it requires extracting specific weight matrices
    raise NotImplementedError(
        f"Direction transformation for hook point '{hook_point}' is not yet "
        f"implemented for HuggingFace models. Use TransformerLens for non-residual hooks."
    )


def load_huggingface_model(
    model_name: str,
    device: str = "cuda",
    dtype: str = "float32",
    **kwargs,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a HuggingFace model and tokenizer.

    Args:
        model_name: The model name or path (HuggingFace model ID)
        device: The device to load the model on
        dtype: The data type for the model weights
        **kwargs: Additional arguments passed to from_pretrained

    Returns:
        Tuple of (model, tokenizer)
    """
    torch_dtype = DTYPES.get(dtype, torch.float32)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set up padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device if device != "cpu" else None,
        **kwargs,
    )

    if device == "cpu":
        model = model.to(device)

    model.eval()

    return model, tokenizer


def create_huggingface_wrapper(
    model_name_or_model,
    hook_point: str,
    auxiliary_hook_points: Optional[List[str]] = None,
    device: str = "cuda",
    dtype: str = "float32",
    tokenizer: Optional[AutoTokenizer] = None,
    **kwargs,
) -> HuggingFaceModelWrapper:
    """
    Create a HuggingFaceModelWrapper from a model name or existing model.

    Args:
        model_name_or_model: Either a model name string or an AutoModelForCausalLM instance
        hook_point: The primary TransformerLens-style hook point
        auxiliary_hook_points: Additional hook points to capture
        device: Device to use (if loading from name)
        dtype: Data type to use
        tokenizer: Optional tokenizer (required if model is passed directly)
        **kwargs: Additional arguments for model loading

    Returns:
        HuggingFaceModelWrapper instance
    """
    if isinstance(model_name_or_model, str):
        model, tokenizer = load_huggingface_model(
            model_name_or_model, device=device, dtype=dtype, **kwargs
        )
    else:
        model = model_name_or_model
        if tokenizer is None:
            raise ValueError("tokenizer must be provided when passing a model directly")

    activation_config = HFActivationConfig(
        primary_hook_point=hook_point,
        auxiliary_hook_points=auxiliary_hook_points or [],
    )

    torch_dtype = DTYPES.get(dtype, torch.float32)

    return HuggingFaceModelWrapper(
        model=model,
        tokenizer=tokenizer,
        activation_config=activation_config,
        dtype=torch_dtype,
    )
