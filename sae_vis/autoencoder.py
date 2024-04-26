from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer_lens import utils

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


@dataclass
class AutoEncoderConfig:
    """Class for storing configuration parameters for the autoencoder"""

    d_in: int
    d_hidden: int  # | None = None
    dict_mult: int  # | None = None

    l1_coeff: float = 3e-4

    # def __post_init__(self):
    #     assert (
    #         int(self.d_hidden is None) + int(self.dict_mult is None) == 1
    #     ), "Exactly one of d_hidden or dict_mult must be provided"
    #     if (self.d_hidden is None) and isinstance(self.dict_mult, int):
    #         self.d_hidden = self.d_in * self.dict_mult
    #     elif (self.dict_mult is None) and isinstance(self.d_hidden, int):
    #         assert self.d_hidden % self.d_in == 0, "d_hidden must be a multiple of d_in"
    #         self.dict_mult = self.d_hidden // self.d_in


class AutoEncoder(nn.Module):
    def __init__(self, cfg: AutoEncoderConfig):
        super().__init__()
        self.cfg = cfg

        assert isinstance(cfg.d_hidden, int)

        # W_enc has shape (d_in, d_encoder), where d_encoder is a multiple of d_in (cause dictionary learning; overcomplete basis)
        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(torch.empty(cfg.d_in, cfg.d_hidden))
        )
        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(torch.empty(cfg.d_hidden, cfg.d_in))
        )
        self.b_enc = nn.Parameter(torch.zeros(cfg.d_hidden))
        self.b_dec = nn.Parameter(torch.zeros(cfg.d_in))
        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)

    def forward(self, x: torch.Tensor):
        feature_acts = self.get_feature_acts(x)

        x_reconstruct = feature_acts @ self.W_dec + self.b_dec
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).sum(-1).mean(0)
        l1_loss = self.cfg.l1_coeff * (feature_acts.float().abs().sum())
        loss = l2_loss + l1_loss
        return loss, x_reconstruct, feature_acts, l2_loss, l1_loss

    def get_feature_acts(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x - self.b_dec @ self.W_enc + self.b_enc)

    @torch.no_grad()
    def remove_parallel_component_of_grads(self):
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(
            -1, keepdim=True
        ) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj

    @classmethod
    def load_from_hf(cls, version: str):
        """
        Loads the saved autoencoder from HuggingFace.

        Note, this is a classmethod, because we'll be using it as `auto_encoder = AutoEncoder.load_from_hf("run1")`

        Version is expected to be an int, or "run1" or "run2"

        version 25 is the final checkpoint of the first autoencoder run,
        version 47 is the final checkpoint of the second autoencoder run.
        """
        assert version in ["run1", "run2"]
        version_num = 25 if version == "run1" else 47

        # Load in state dict
        state_dict = utils.download_file_from_hf(
            "NeelNanda/sparse_autoencoder", f"{version_num}.pt", force_is_torch=True
        )
        assert isinstance(state_dict, dict)
        assert set(state_dict.keys()) == {"W_enc", "W_dec", "b_enc", "b_dec"}
        d_in, d_hidden = state_dict["W_enc"].shape

        # Create autoencoder
        cfg = AutoEncoderConfig(
            d_in=d_in, d_hidden=d_hidden, dict_mult=d_hidden // d_in
        )
        encoder = cls(cfg)
        encoder.load_state_dict(state_dict)
        return encoder

    def __repr__(self) -> str:
        return f"AutoEncoder(d_in={self.cfg.d_in}, dict_mult={self.cfg.dict_mult})"
