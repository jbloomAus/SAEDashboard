"""
Hook utilities for converting between TransformerLens and HuggingFace/PyTorch hook names.

This module provides functions to convert TransformerLens hook names to their corresponding
HuggingFace/PyTorch module paths, enabling support for models loaded directly via HuggingFace
Transformers without requiring TransformerLens.
"""

import re
from dataclasses import dataclass
from typing import Optional, Tuple

# Mapping from TransformerLens model names to HuggingFace model names
# Add new mappings here as needed
TRANSFORMERLENS_TO_HUGGINGFACE_MODEL_MAP = {
    # GPT-2 variants
    "gpt2-small": "openai-community/gpt2",
    "gpt2": "openai-community/gpt2",
    "gpt2-medium": "openai-community/gpt2-medium",
    "gpt2-large": "openai-community/gpt2-large",
    "gpt2-xl": "openai-community/gpt2-xl",
    # GPT-Neo
    "gpt-neo-125M": "EleutherAI/gpt-neo-125M",
    "gpt-neo-1.3B": "EleutherAI/gpt-neo-1.3B",
    "gpt-neo-2.7B": "EleutherAI/gpt-neo-2.7B",
    # Pythia
    "pythia-70m": "EleutherAI/pythia-70m",
    "pythia-160m": "EleutherAI/pythia-160m",
    "pythia-410m": "EleutherAI/pythia-410m",
    "pythia-1b": "EleutherAI/pythia-1b",
    "pythia-1.4b": "EleutherAI/pythia-1.4b",
    "pythia-2.8b": "EleutherAI/pythia-2.8b",
    "pythia-6.9b": "EleutherAI/pythia-6.9b",
    "pythia-12b": "EleutherAI/pythia-12b",
    # Gemma
    "gemma-2b": "google/gemma-2b",
    "gemma-2b-it": "google/gemma-2b-it",
    "gemma-7b": "google/gemma-7b",
    "gemma-7b-it": "google/gemma-7b-it",
    "gemma-2-2b": "google/gemma-2-2b",
    "gemma-2-2b-it": "google/gemma-2-2b-it",
    "gemma-2-9b": "google/gemma-2-9b",
    "gemma-2-9b-it": "google/gemma-2-9b-it",
    # Llama
    "llama-7b": "meta-llama/Llama-2-7b-hf",
    "llama-13b": "meta-llama/Llama-2-13b-hf",
    "llama-7b-chat": "meta-llama/Llama-2-7b-chat-hf",
    "llama-13b-chat": "meta-llama/Llama-2-13b-chat-hf",
    # Mistral
    "mistral-7b": "mistralai/Mistral-7B-v0.1",
    "mistral-7b-instruct": "mistralai/Mistral-7B-Instruct-v0.1",
}


def convert_model_name_tl_to_hf(model_name: str) -> str:
    """
    Convert a TransformerLens model name to a HuggingFace model name.

    This function handles the mapping between TransformerLens naming conventions
    and HuggingFace model repository names. If no mapping is found, the original
    name is returned (it may already be a valid HuggingFace name).

    Args:
        model_name: The model name (TransformerLens or HuggingFace format)

    Returns:
        The HuggingFace model name/path

    Example:
        >>> convert_model_name_tl_to_hf("gpt2-small")
        'openai-community/gpt2'
        >>> convert_model_name_tl_to_hf("google/gemma-2b")
        'google/gemma-2b'  # Already a HuggingFace name, returned as-is
    """
    # Check if it's in our mapping
    if model_name in TRANSFORMERLENS_TO_HUGGINGFACE_MODEL_MAP:
        return TRANSFORMERLENS_TO_HUGGINGFACE_MODEL_MAP[model_name]

    # If it contains a slash, it's likely already a HuggingFace path
    if "/" in model_name:
        return model_name

    # Return as-is and hope it works
    return model_name


@dataclass
class HookInfo:
    """Information about a hook point in both TransformerLens and HuggingFace formats."""

    transformer_lens_name: str  # e.g., "blocks.5.hook_resid_post"
    layer_index: int  # e.g., 5
    hook_type: str  # e.g., "hook_resid_post"
    hf_module_path: str  # e.g., "model.layers.5"
    capture_output: bool = True  # Whether to capture the output (True) or input (False)
    output_index: Optional[int] = None  # If output is a tuple, which index to use


def parse_transformer_lens_hook(hook_name: str) -> Tuple[int, str]:
    """
    Parse a TransformerLens hook name into its components.

    Args:
        hook_name: TransformerLens hook name like "blocks.5.hook_resid_post"

    Returns:
        Tuple of (layer_index, hook_type)

    Raises:
        ValueError: If the hook name doesn't match the expected format
    """
    # Pattern for standard blocks.{layer}.{hook_type} format
    match = re.match(r"blocks\.(\d+)\.(.+)", hook_name)
    if not match:
        raise ValueError(
            f"Invalid TransformerLens hook name format: {hook_name}. "
            f"Expected format: blocks.{{layer}}.{{hook_type}}"
        )

    layer_index = int(match.group(1))
    hook_type = match.group(2)

    return layer_index, hook_type


def get_hf_module_path(
    layer_index: int,
    hook_type: str,
    model_type: str = "auto",
) -> Tuple[str, bool, Optional[int]]:
    """
    Get the HuggingFace module path for a given hook type.

    This function maps TransformerLens hook types to their corresponding HuggingFace
    module paths. Different model architectures may have different module structures.

    Args:
        layer_index: The layer index (0-indexed)
        hook_type: The TransformerLens hook type (e.g., "hook_resid_post")
        model_type: The model type for architecture-specific mappings.
                   "auto" will use common patterns that work for most models.

    Returns:
        Tuple of (hf_module_path, capture_output, output_index)
        - hf_module_path: Path to the HuggingFace module
        - capture_output: Whether to capture output (True) or input (False)
        - output_index: If output is a tuple, which index to use (None for direct output)

    Raises:
        NotImplementedError: If the hook type is not yet supported
    """
    # Most HuggingFace models follow the pattern: model.layers.{layer_index}
    # Some models use model.transformer.h.{layer_index} (GPT-2 style)

    if hook_type == "hook_resid_post":
        # Residual stream after the layer (output of the layer)
        # In HuggingFace, this is the output of the transformer layer
        # The output is typically a tuple where the first element is the hidden states
        return f"model.layers.{layer_index}", True, 0

    elif hook_type == "hook_resid_pre":
        # Residual stream before the layer (input to the layer)
        # We capture the input to the layer
        return f"model.layers.{layer_index}", False, None

    elif hook_type == "hook_mlp_out":
        # Output of the MLP sublayer
        # In most HuggingFace models, this is model.layers.{layer}.mlp
        return f"model.layers.{layer_index}.mlp", True, None

    elif hook_type == "hook_attn_out":
        # Output of the attention sublayer
        # In most HuggingFace models, this is model.layers.{layer}.self_attn
        return f"model.layers.{layer_index}.self_attn", True, 0

    elif hook_type.startswith("ln1.") or hook_type.startswith("ln2."):
        # Layer normalization hooks - these are more complex and vary by model
        raise NotImplementedError(
            f"Layer normalization hooks ({hook_type}) are not yet supported for HuggingFace models. "
            f"Please use TransformerLens for these hook types."
        )

    elif "hook_z" in hook_type:
        # Attention output before projection - complex, architecture-dependent
        raise NotImplementedError(
            f"Attention hook_z ({hook_type}) is not yet supported for HuggingFace models. "
            f"Please use TransformerLens for this hook type."
        )

    else:
        raise NotImplementedError(
            f"Hook type '{hook_type}' is not yet supported for HuggingFace models. "
            f"Supported types: hook_resid_post, hook_resid_pre, hook_mlp_out, hook_attn_out"
        )


def transformer_lens_to_hf_hook(hook_name: str, model_type: str = "auto") -> HookInfo:
    """
    Convert a TransformerLens hook name to HuggingFace hook information.

    Args:
        hook_name: TransformerLens hook name like "blocks.5.hook_resid_post"
        model_type: The model type for architecture-specific mappings

    Returns:
        HookInfo with all necessary information to hook into a HuggingFace model

    Example:
        >>> info = transformer_lens_to_hf_hook("blocks.5.hook_resid_post")
        >>> print(info.hf_module_path)  # "model.layers.5"
        >>> print(info.layer_index)      # 5
    """
    layer_index, hook_type = parse_transformer_lens_hook(hook_name)
    hf_module_path, capture_output, output_index = get_hf_module_path(
        layer_index, hook_type, model_type
    )

    return HookInfo(
        transformer_lens_name=hook_name,
        layer_index=layer_index,
        hook_type=hook_type,
        hf_module_path=hf_module_path,
        capture_output=capture_output,
        output_index=output_index,
    )


def get_submodule_by_path(model, path: str):
    """
    Get a submodule from a model by its path string.

    Args:
        model: The HuggingFace model
        path: Dot-separated path to the submodule (e.g., "model.layers.5")

    Returns:
        The submodule at the specified path

    Raises:
        AttributeError: If the path doesn't exist in the model
    """
    parts = path.split(".")
    module = model
    for part in parts:
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module


def detect_model_architecture(model) -> str:
    """
    Detect the model architecture from a HuggingFace model.

    Args:
        model: The HuggingFace model

    Returns:
        A string identifying the model architecture type
    """
    model_type = getattr(model.config, "model_type", "unknown")

    # Map model types to architecture categories
    architecture_map = {
        "llama": "llama",
        "mistral": "llama",  # Mistral uses Llama-style architecture
        "gemma": "gemma",
        "gemma2": "gemma",
        "gpt2": "gpt2",
        "gpt_neo": "gpt2",
        "gpt_neox": "gpt_neox",
        "pythia": "gpt_neox",
        "qwen2": "llama",
        "phi": "phi",
    }

    return architecture_map.get(model_type, "auto")


def get_layer_module_path(model, layer_index: int) -> str:
    """
    Get the path to a transformer layer in a HuggingFace model.

    Different model architectures use different paths to their transformer layers.
    This function attempts to find the correct path.

    Args:
        model: The HuggingFace model
        layer_index: The layer index

    Returns:
        The path to the transformer layer

    Raises:
        ValueError: If the layer path cannot be determined
    """
    # Try common patterns
    patterns = [
        f"model.layers.{layer_index}",  # Llama, Mistral, Gemma, Qwen2
        f"transformer.h.{layer_index}",  # GPT-2
        f"gpt_neox.layers.{layer_index}",  # GPT-NeoX, Pythia
        f"model.decoder.layers.{layer_index}",  # Some encoder-decoder models
    ]

    for pattern in patterns:
        try:
            get_submodule_by_path(model, pattern)
            return pattern
        except (AttributeError, IndexError, KeyError):
            continue

    raise ValueError(
        f"Could not find transformer layer {layer_index} in model. "
        f"Tried patterns: {patterns}. "
        f"Model config type: {getattr(model.config, 'model_type', 'unknown')}"
    )
