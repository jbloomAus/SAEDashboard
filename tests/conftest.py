import pytest
import torch
from jaxtyping import Int
from sae_lens import SAE, SAEConfig
from torch import Tensor
from transformer_lens import HookedTransformer


@pytest.fixture
def model() -> HookedTransformer:
    model = HookedTransformer.from_pretrained("tiny-stories-1M", device="cpu")
    model.eval()
    return model


@pytest.fixture()
def tokens(model: HookedTransformer) -> Int[Tensor, "batch seq"]:
    return model.to_tokens(
        [
            "But what about second breakfast?" * 3,
            "Nothing is cheesier than cheese." * 3,
        ]
    )


@pytest.fixture
def autoencoder() -> SAE:
    cfg = SAEConfig(
        d_in=64,
        d_sae=128,
        apply_b_dec_to_input=False,
        context_size=128,
        model_name="TEST",
        hook_name="test",
        hook_layer=0,
        prepend_bos=True,
        dataset_path="test/test",
        dtype="float32",
        activation_fn_str="relu",
        finetuning_scaling_factor=False,
        hook_head_index=None,
        normalize_activations="none",
        device="cpu",
        sae_lens_training_version=None,
        dataset_trust_remote_code=True,
    )

    autoencoder = SAE(cfg)
    # set weights and biases to hardcoded values so tests are consistent
    seed1 = torch.tensor([0.1, -0.2, 0.3, -0.4] * 16)  # 64
    seed2 = torch.tensor([0.2, -0.1, 0.4, -0.2] * 32)  # 64 x 2
    seed3 = torch.tensor([0.3, -0.3, 0.6, -0.6] * 16)  # 64
    seed4 = torch.tensor([-0.4, 0.4, 0.8, -0.8] * 32)  # 64 x 2
    W_enc_base = torch.cat([torch.eye(64), torch.eye(64)], dim=-1)
    W_dec_base = torch.cat([torch.eye(64), torch.eye(64)], dim=0)
    autoencoder.load_state_dict(
        {
            "W_enc": W_enc_base + torch.outer(seed1, seed2),
            "W_dec": W_dec_base + torch.outer(seed4, seed3),
            "b_enc": torch.zeros_like(autoencoder.b_enc) + 0.5,
            "b_dec": torch.zeros_like(autoencoder.b_dec) + 0.3,
        }
    )

    return autoencoder
