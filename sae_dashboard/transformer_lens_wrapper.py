from __future__ import annotations

import re
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Sequence, Tuple, cast

import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor
from transformer_lens.hook_points import HookPoint

if TYPE_CHECKING:
    from sae_lens import HookedSAETransformer

DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


@dataclass
class ActivationConfig:
    primary_hook_point: str
    auxiliary_hook_points: List[str]


def activation_shapes_match_tokens(
    activation_dict: Dict[str, Tensor],
    tokens: Int[Tensor, "batch seq"],
) -> bool:
    """Whether every captured activation is consistent with the input token shape.

    Most hooks are ``[batch, seq, ...]``. Attention pattern/score hooks are
    ``[batch, heads, seq_q, seq_k]`` — the head axis sits where seq usually is, so they
    are validated on the batch and the two seq axes instead of being misflagged as
    prompt-length mismatches.
    """
    expected_shape = tuple(tokens.shape[:2])
    for hook_name, activation in activation_dict.items():
        if activation.ndim < 2:
            continue
        if tuple(activation.shape[:2]) == expected_shape:
            continue
        if (
            hook_name.endswith(("hook_pattern", "hook_attn_scores"))
            and activation.ndim >= 4
            and activation.shape[0] == expected_shape[0]
            and tuple(activation.shape[2:4]) == (expected_shape[1], expected_shape[1])
        ):
            continue
        return False
    return True


class TransformerLensWrapper(nn.Module):
    """
    This class wraps around & extends the TransformerLens model, so that we can make sure things like the forward
    function have a standardized signature.
    """

    def __init__(
        self,
        model: HookedSAETransformer,
        activation_config: ActivationConfig,
        disable_kv_cache: bool = False,
    ):
        super().__init__()
        self.model = model
        self.activation_config = activation_config
        # Bridge forwards run the underlying HF model with use_cache=True by default,
        # paying DynamicCache initialization churn on every full-sequence forward even
        # though the KV cache is never reused here. Opt-in per lane so the preserved
        # legacy path keeps its exact previous behavior; only effective for bridge
        # models (HookedTransformer forwards do not accept the kwarg).
        self.disable_kv_cache = bool(disable_kv_cache) and hasattr(
            model, "original_model"
        )
        self.validate_hook_points()
        self.hook_layer = self.get_layer(self.activation_config.primary_hook_point)

    def validate_hook_points(self):
        """Checks that the hook points are valid and that the model has them"""
        assert (
            self.activation_config.primary_hook_point in self.model.hook_dict
        ), f"Invalid hook point: {self.activation_config.primary_hook_point}"

        for hook_point in self.activation_config.auxiliary_hook_points:
            assert (
                hook_point in self.model.hook_dict
            ), f"Invalid hook point: {hook_point}"

    def get_layer(self, hook_point: str):
        """Get the layer (so we can do the early stopping in our forward pass)"""
        layer_match = re.match(r"blocks\.(\d+)\.", hook_point)
        assert (
            layer_match
        ), f"Error: expecting hook_point to be 'blocks.{{layer}}.{{...}}', but got {hook_point!r}"
        return int(layer_match.group(1))

    @staticmethod
    def _activation_shapes_match_tokens(
        activation_dict: Dict[str, Tensor],
        tokens: Int[Tensor, "batch seq"],
    ) -> bool:
        return activation_shapes_match_tokens(activation_dict, tokens)

    @staticmethod
    def _concat_activation_dicts(
        activation_dicts: list[Dict[str, Tensor]],
    ) -> Dict[str, Tensor]:
        if not activation_dicts:
            return {}
        return {
            key: torch.cat(
                [activation_dict[key] for activation_dict in activation_dicts], dim=0
            )
            for key in activation_dicts[0]
        }

    def _should_bypass_final_layer_logits(self, return_logits: bool) -> bool:
        if return_logits:
            return False
        model_cfg = getattr(self.model, "cfg", None)
        n_layers = getattr(model_cfg, "n_layers", None)
        return isinstance(n_layers, int) and self.hook_layer == n_layers - 1

    def _resolve_body_model(self) -> nn.Module | None:
        original_model = getattr(self.model, "original_model", None)
        if original_model is None:
            return None
        for attr_name in ("model", "transformer", "language_model", "base_model"):
            body_model = getattr(original_model, attr_name, None)
            if isinstance(body_model, nn.Module):
                return body_model
        return None

    def _run_final_layer_without_logits(
        self,
        tokens: Int[Tensor, "batch seq"],
        hooks: list[tuple[str, Callable[[Tensor, HookPoint], None]]],
    ) -> None:
        body_model = self._resolve_body_model()
        if body_model is None:
            raise RuntimeError(
                "Final-layer activation-only forward requires a Bridge body model to bypass lm_head."
            )

        hooks_context = getattr(self.model, "hooks", None)
        context_manager = cast(
            AbstractContextManager[Any],
            (
                hooks_context(fwd_hooks=hooks)
                if callable(hooks_context)
                else nullcontext()
            ),
        )
        with context_manager:
            if self.disable_kv_cache:
                body_model(input_ids=tokens, use_cache=False)
            else:
                body_model(input_ids=tokens)

    def _run_with_hooks_once(
        self,
        tokens: Int[Tensor, "batch seq"],
        return_logits: bool,
        hooks: list[tuple[str, Callable[[Tensor, HookPoint], None]]],
    ) -> Dict[str, Tensor]:
        activation_dict: Dict[str, Tensor] = {}
        return_type = "logits" if return_logits else None

        def build_act_dict(
            hooks_to_collect: Sequence[Tuple[str, Callable[[Tensor, HookPoint], None]]],
        ) -> None:
            for hook_point, _ in hooks_to_collect:
                activation: Tensor = self.model.hook_dict[hook_point].ctx.pop(
                    "activation"
                )
                if "hook_z" in hook_point:
                    activation = activation.flatten(-2, -1)
                activation_dict[hook_point] = activation

        output = None
        forward_kwargs: dict[str, Any] = (
            {"use_cache": False} if self.disable_kv_cache else {}
        )
        if self._should_bypass_final_layer_logits(return_logits):
            self._run_final_layer_without_logits(tokens, hooks)
        else:
            output = self.model.run_with_hooks(
                tokens,
                return_type=return_type,
                stop_at_layer=self.hook_layer + 1,
                fwd_hooks=hooks,  # type: ignore[arg-type]
                **forward_kwargs,
            )

        build_act_dict(hooks)

        if return_logits:
            activation_dict["output"] = cast(Tensor, output)
        else:
            del output

        return activation_dict

    def forward(  # type: ignore
        self,
        tokens: Int[Tensor, "batch seq"],
        return_logits: bool = True,
    ) -> Dict[str, Tensor]:
        """Executes a forward pass, collecting specific hook point activations and optionally logit outputs"""
        hooks: List[Tuple[str, Callable[[Tensor, HookPoint], None]]] = [
            (self.activation_config.primary_hook_point, self.hook_fn_store_act)
        ] + [
            (point, self.hook_fn_store_act)
            for point in self.activation_config.auxiliary_hook_points
        ]

        try:
            model_device = next(self.model.parameters()).device
        except StopIteration:
            model_device = tokens.device
        if tokens.device != model_device:
            tokens = tokens.to(model_device)

        activation_dict = self._run_with_hooks_once(tokens, return_logits, hooks)
        if self._activation_shapes_match_tokens(activation_dict, tokens):
            return activation_dict

        if tokens.shape[0] <= 1:
            observed_shapes = {
                name: tuple(activation.shape)
                for name, activation in activation_dict.items()
            }
            raise RuntimeError(
                "TransformerBridge returned activation shapes that do not match the input tokens and "
                "the minibatch cannot be split further. "
                f"tokens_shape={tuple(tokens.shape)} observed_shapes={observed_shapes}"
            )

        split_size = max(1, tokens.shape[0] // 2)
        repaired_chunks = [
            self.forward(token_chunk, return_logits=return_logits)
            for token_chunk in tokens.split(split_size)
        ]
        return self._concat_activation_dicts(repaired_chunks)

    def hook_fn_store_act(self, activation: torch.Tensor, hook: HookPoint):
        hook.ctx["activation"] = activation

    @property
    def tokenizer(self):  # type: ignore
        return self.model.tokenizer

    @property
    def W_U(self):
        if hasattr(self.model, "W_U"):
            return self.model.W_U
        if hasattr(self.model, "unembed") and hasattr(self.model.unembed, "W_U"):
            return self.model.unembed.W_U
        raise AttributeError(f"{type(self.model).__name__} does not expose W_U")

    @property
    def W_out(self):
        return self.model.W_out

    @property
    def W_O(self):
        return self.model.W_O


def to_resid_direction(
    direction: Float[Tensor, "feats d_in"], model: TransformerLensWrapper
):
    """
    Takes a direction (eg. in the post-ReLU MLP activations) and returns the corresponding direction in the residual stream.

    Args:
        direction:
            The direction in the activations, i.e. shape (feats, d_in) where d_in could be d_model, d_mlp, etc.
        model:
            The model, which should be a HookedTransformerWrapper or similar.
    """
    # If this SAE was trained on the residual stream or attn/mlp out, then we don't need to do anything
    if (
        "resid" in model.activation_config.primary_hook_point
        or "_out" in model.activation_config.primary_hook_point
        or "hook_mlp_in" in model.activation_config.primary_hook_point
        or "mlp.hook_in" in model.activation_config.primary_hook_point
    ):
        return direction

    # If it was trained on the MLP layer, then we apply the W_out map
    elif ("pre" in model.activation_config.primary_hook_point) or (
        "post" in model.activation_config.primary_hook_point
    ):
        return direction @ model.W_out[model.hook_layer].to(
            device=direction.device, dtype=direction.dtype
        )

    elif "hook_z" in model.activation_config.primary_hook_point:
        # device= as well as dtype=: with the SAE and model on different CUDA devices
        # (multi-GPU hosts) the weight must follow the direction's device.
        return direction @ model.W_O[model.hook_layer].flatten(0, 1).to(
            device=direction.device, dtype=direction.dtype
        )

    # For hook_mlp_out (output of MLP)
    elif "hook_mlp_out" in model.activation_config.primary_hook_point:
        return direction

    # For hook_attn_out (output of attention layer)
    elif "hook_attn_out" in model.activation_config.primary_hook_point:
        return direction

    # For hook_resid_post (always a residual stream direction)
    elif "hook_resid_post" in model.activation_config.primary_hook_point:
        return direction

    # For hook_normalized (e.g. ln1.hook_normalized, ln2.hook_normalized)
    # These are normalized residual stream, already in residual stream basis
    elif "hook_normalized" in model.activation_config.primary_hook_point:
        return direction

    # For hook_resid_pre (residual stream before MLP)
    elif "hook_resid_pre" in model.activation_config.primary_hook_point:
        return direction

    # Others not yet supported
    else:
        raise NotImplementedError(
            "The hook your SAE was trained on isn't yet supported"
        )
