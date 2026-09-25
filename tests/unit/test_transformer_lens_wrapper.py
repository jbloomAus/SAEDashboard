# pyright: basic, reportPrivateImportUsage=false
import pytest
import torch
from torch import nn
from transformer_lens import HookedTransformer

from sae_dashboard.transformer_lens_wrapper import (
    ActivationConfig,
    TransformerLensWrapper,
)


class _MockHookPoint:
    def __init__(self, name: str):
        self.name = name
        self.ctx: dict[str, torch.Tensor] = {}


class _HookContextManager:
    def __init__(self, model, fwd_hooks):
        self.model = model
        self.fwd_hooks = fwd_hooks

    def __enter__(self):
        self.model.active_hooks = list(self.fwd_hooks)
        return self

    def __exit__(self, exc_type, exc, tb):
        self.model.active_hooks = []
        return False


class _FinalLayerBridgeBody(nn.Module):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent

    def forward(self, input_ids):
        activations = input_ids.float().unsqueeze(-1)
        for hook_name, hook_fn in self.parent.active_hooks:
            hook_fn(activations, self.parent.hook_dict[hook_name])
        return activations


class _FinalLayerBridgeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.cfg = type("Cfg", (), {"n_layers": 1})()
        self.hook_dict = {
            "blocks.0.hook_resid_pre": _MockHookPoint("blocks.0.hook_resid_pre"),
        }
        self.active_hooks: list[tuple[str, object]] = []
        self.original_model = type(
            "OriginalModel", (), {"model": _FinalLayerBridgeBody(self)}
        )()
        self.run_with_hooks_called = False

    def hooks(self, *, fwd_hooks):
        return _HookContextManager(self, fwd_hooks)

    def run_with_hooks(self, *args, **kwargs):
        self.run_with_hooks_called = True
        raise AssertionError(
            "run_with_hooks should not be called for final-layer activation-only passes"
        )


class _TruncatingBridgeModel(nn.Module):
    def __init__(self, safe_batch_size: int = 32):
        super().__init__()
        self.safe_batch_size = safe_batch_size
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.return_types_seen: list[str | None] = []
        self.hook_dict = {
            "blocks.0.hook_resid_pre": _MockHookPoint("blocks.0.hook_resid_pre"),
            "blocks.0.attn.hook_z": _MockHookPoint("blocks.0.attn.hook_z"),
        }

    def run_with_hooks(self, tokens, stop_at_layer, fwd_hooks, return_type="logits"):
        del stop_at_layer
        self.return_types_seen.append(return_type)
        effective_batch = min(tokens.shape[0], self.safe_batch_size)
        truncated_tokens = tokens[:effective_batch].float()
        for hook_name, hook_fn in fwd_hooks:
            hook = self.hook_dict[hook_name]
            if "hook_z" in hook_name:
                activation = (
                    truncated_tokens.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 2, 1)
                )
            else:
                activation = truncated_tokens.unsqueeze(-1)
            hook_fn(activation, hook)
        if return_type is None:
            return None
        return truncated_tokens.unsqueeze(-1) + 1000.0


@pytest.fixture(scope="module")
def real_model() -> HookedTransformer:
    return HookedTransformer.from_pretrained("gpt2-small")


@pytest.fixture
def valid_activation_config(real_model: HookedTransformer) -> ActivationConfig:
    return ActivationConfig(
        primary_hook_point="blocks.5.hook_resid_post",
        auxiliary_hook_points=[
            "blocks.0.hook_resid_pre",
            "blocks.4.hook_mlp_out",
            "blocks.3.attn.hook_z",
        ],
    )


def test_initialization(
    real_model: HookedTransformer, valid_activation_config: ActivationConfig
) -> None:
    wrapper = TransformerLensWrapper(real_model, valid_activation_config)  # type: ignore
    assert wrapper.model == real_model
    assert wrapper.activation_config == valid_activation_config
    assert wrapper.hook_layer == 5


def test_validate_hook_points(
    real_model: HookedTransformer, valid_activation_config: ActivationConfig
) -> None:
    wrapper = TransformerLensWrapper(real_model, valid_activation_config)  # type: ignore
    wrapper.validate_hook_points()  # This should not raise an exception


def test_validate_hook_points_invalid(real_model: HookedTransformer) -> None:
    invalid_config = ActivationConfig(
        primary_hook_point="blocks.15.invalid_hook",
        auxiliary_hook_points=["blocks.0.hook_resid_pre"],
    )
    with pytest.raises(AssertionError):
        TransformerLensWrapper(real_model, invalid_config)  # type: ignore


def test_get_layer(
    real_model: HookedTransformer, valid_activation_config: ActivationConfig
) -> None:
    wrapper = TransformerLensWrapper(real_model, valid_activation_config)  # type: ignore
    assert wrapper.get_layer("blocks.2.hook_mlp_out") == 2
    with pytest.raises(AssertionError):
        wrapper.get_layer("invalid_hook_point")


@pytest.mark.parametrize("return_logits", [True, False])
def test_forward(
    real_model: HookedTransformer,
    valid_activation_config: ActivationConfig,
    return_logits: bool,
) -> None:
    wrapper = TransformerLensWrapper(real_model, valid_activation_config)  # type: ignore
    tokens = torch.randint(0, real_model.cfg.d_vocab, (2, 10))

    activation_dict = wrapper.forward(tokens, return_logits=return_logits)

    assert isinstance(activation_dict, dict)

    expected_keys = set(
        valid_activation_config.auxiliary_hook_points
        + [valid_activation_config.primary_hook_point]
    )
    if return_logits:
        expected_keys.add("output")

    assert set(activation_dict.keys()) == expected_keys

    # Check shapes of activations
    assert activation_dict["blocks.0.hook_resid_pre"].shape == (
        2,
        10,
        real_model.cfg.d_model,
    )
    assert activation_dict["blocks.4.hook_mlp_out"].shape == (
        2,
        10,
        real_model.cfg.d_model,
    )
    assert activation_dict["blocks.3.attn.hook_z"].shape == (
        2,
        10,
        real_model.cfg.n_heads * real_model.cfg.d_head,
    )
    assert activation_dict["blocks.5.hook_resid_post"].shape == (
        2,
        10,
        real_model.cfg.d_model,
    )

    if return_logits:
        assert activation_dict["output"].shape == (2, 10, real_model.cfg.d_model)
    else:
        assert "output" not in activation_dict

    # Additional checks
    for key, value in activation_dict.items():
        assert isinstance(value, torch.Tensor), f"Value for {key} is not a torch.Tensor"
        assert (
            value.shape[0] == 2 and value.shape[1] == 10
        ), f"Incorrect batch or sequence dimension for {key}"

    # Check that 'hook_z' tensors are flattened
    for key in activation_dict:
        if "hook_z" in key:
            assert (
                len(activation_dict[key].shape) == 3
            ), f"{key} should be flattened to 3 dimensions"


def test_hook_fn_store_act(
    real_model: HookedTransformer, valid_activation_config: ActivationConfig
) -> None:
    wrapper = TransformerLensWrapper(real_model, valid_activation_config)  # type: ignore
    activation = torch.randn(2, 10, real_model.cfg.d_model)
    hook = type("MockHook", (), {"ctx": {}})()

    wrapper.hook_fn_store_act(activation, hook)  # type: ignore
    assert torch.all(hook.ctx["activation"] == activation)  # type: ignore


def test_property_access(
    real_model: HookedTransformer, valid_activation_config: ActivationConfig
) -> None:
    wrapper = TransformerLensWrapper(real_model, valid_activation_config)  # type: ignore

    assert torch.all(wrapper.W_U == real_model.W_U)
    assert torch.all(wrapper.W_out == real_model.W_out)
    assert torch.all(wrapper.W_O == real_model.W_O)
    assert wrapper.tokenizer == real_model.tokenizer


@pytest.fixture
def truncating_bridge_wrapper() -> TransformerLensWrapper:
    config = ActivationConfig(
        primary_hook_point="blocks.0.hook_resid_pre",
        auxiliary_hook_points=["blocks.0.attn.hook_z"],
    )
    return TransformerLensWrapper(_TruncatingBridgeModel(), config)  # type: ignore[arg-type]


def test_forward_repairs_truncated_bridge_batches(
    truncating_bridge_wrapper: TransformerLensWrapper,
) -> None:
    tokens = torch.arange(64 * 5).reshape(64, 5)

    activation_dict = truncating_bridge_wrapper.forward(tokens, return_logits=True)

    assert activation_dict["blocks.0.hook_resid_pre"].shape == (64, 5, 1)
    assert activation_dict["blocks.0.attn.hook_z"].shape == (64, 5, 2)
    assert activation_dict["output"].shape == (64, 5, 1)
    assert torch.equal(
        activation_dict["blocks.0.hook_resid_pre"].squeeze(-1),
        tokens.float(),
    )
    assert torch.equal(
        activation_dict["output"].squeeze(-1),
        tokens.float() + 1000.0,
    )


def test_activation_shape_check_detects_truncation(
    truncating_bridge_wrapper: TransformerLensWrapper,
) -> None:
    tokens = torch.arange(48 * 3).reshape(48, 3)
    activation_dict = {
        "blocks.0.hook_resid_pre": tokens[:32].float().unsqueeze(-1),
    }

    assert not truncating_bridge_wrapper._activation_shapes_match_tokens(
        activation_dict, tokens
    )


def test_forward_suppresses_logits_when_not_requested(
    truncating_bridge_wrapper: TransformerLensWrapper,
) -> None:
    tokens = torch.arange(8 * 3).reshape(8, 3)

    activation_dict = truncating_bridge_wrapper.forward(tokens, return_logits=False)

    assert "output" not in activation_dict
    assert truncating_bridge_wrapper.model.return_types_seen == [None]


def test_forward_bypasses_lm_head_for_final_layer_activation_only_pass() -> None:
    config = ActivationConfig(
        primary_hook_point="blocks.0.hook_resid_pre",
        auxiliary_hook_points=[],
    )
    wrapper = TransformerLensWrapper(_FinalLayerBridgeModel(), config)  # type: ignore[arg-type]
    tokens = torch.arange(12).reshape(4, 3)

    activation_dict = wrapper.forward(tokens, return_logits=False)

    assert "output" not in activation_dict
    assert activation_dict["blocks.0.hook_resid_pre"].shape == (4, 3, 1)
    assert torch.equal(
        activation_dict["blocks.0.hook_resid_pre"].squeeze(-1), tokens.float()
    )
    assert wrapper.model.run_with_hooks_called is False
