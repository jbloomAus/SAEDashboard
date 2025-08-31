import pytest
import torch
from transformer_lens import HookedTransformer

from sae_dashboard.transformer_lens_wrapper import (
    ActivationConfig,
    TransformerLensWrapper,
)


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
