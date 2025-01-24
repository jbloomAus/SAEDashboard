from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List

import torch


@dataclass
class VectorSetConfig:
    # Core vector properties
    d_in: int
    d_vectors: int  # Number of vectors (replaces d_sae)
    vector_names: List[str]  # Names for each vector
    
    # Model/hook details
    model_name: str
    hook_name: str  # e.g., 'hook_resid_pre'
    hook_layer: int
    hook_head_index: int | None
    prepend_bos: bool
    
    # Dataset details (needed for activation store)
    context_size: int
    dataset_path: str
    
    # Device/dtype settings
    dtype: str
    device: str
    
    # Optional metadata
    model_from_pretrained_kwargs: Dict[str, Any] = field(default_factory=dict)


class VectorSet:
    """Class to handle probe vectors, similar to how SAE handles features"""
    def __init__(
        self,
        vectors: torch.Tensor,  # [n_vectors, d_model] 
        names: List[str],
        hook_point: str,
        hook_layer: int,
        hook_head_index: int | None,
        prepend_bos: bool,
        device: str = "cpu",
        dtype: str = "float32",
        dataset_path: str = "",
        context_size: int = 128,
        model_name: str = "",
        model_from_pretrained_kwargs: Dict[str, Any] = None,
    ):
        self.vectors = vectors.to(device=device)
        self.names = names
        self.hook_point = hook_point
        self.hook_layer = hook_layer
        self.hook_head_index = hook_head_index
        self.prepend_bos = prepend_bos
        
        self.cfg = VectorSetConfig(
            d_in=vectors.shape[1],
            d_vectors=len(vectors),
            vector_names=names,
            model_name=model_name,
            hook_name=hook_point,
            hook_layer=hook_layer,
            hook_head_index=hook_head_index,
            prepend_bos=prepend_bos,
            context_size=context_size,
            dataset_path=dataset_path,
            dtype=dtype,
            device=device,
            model_from_pretrained_kwargs=model_from_pretrained_kwargs or {}
        )

    @classmethod
    def from_json(
        cls, 
        json_path: str | Path, 
        d_model: int,
        hook_point: str,
        hook_layer: int,
        model_name: str,
        names: List[str] | None = None,
        **kwargs
    ) -> "VectorSet":
        """Load vectors from a JSON file where they're stored as 1D arrays
        
        Args:
            json_path: Path to JSON file containing vectors
            d_model: Model dimension to reshape vectors into
            hook_point: Name of hook point (e.g. 'resid_pre')
            hook_layer: Layer number
            model_name: Name of model
            names: Optional list of names for vectors. If None, will generate numbered names
            **kwargs: Additional arguments passed to VectorSet constructor
        """
        with open(json_path) as f:
            data = json.load(f)
        
        # Convert 1D arrays to 2D tensor
        vectors_list = [torch.tensor(vec) for vec in data['vectors']]
        vectors = torch.stack(vectors_list)
        
        # Reshape if needed
        if vectors.dim() == 2 and vectors.shape[1] != d_model:
            n_vectors = vectors.shape[0]
            vectors = vectors.reshape(n_vectors, d_model)
            
        # Generate names if not provided
        if names is None:
            names = [f"vector_{i}" for i in range(len(vectors))]
            
        return cls(
            vectors=vectors,
            names=names,
            hook_point=hook_point,
            hook_layer=hook_layer,
            hook_head_index=None,  # Assuming these are not head-specific vectors
            prepend_bos=True,      # Default to True for most use cases
            model_name=model_name,
            **kwargs
        )

    def encode(self, acts: torch.Tensor) -> torch.Tensor:
        """Compute dot product between activations and probe vectors"""
        return torch.einsum('...d,nd->...n', acts, self.vectors)