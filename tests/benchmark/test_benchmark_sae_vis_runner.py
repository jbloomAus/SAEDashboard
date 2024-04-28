import os
from pathlib import Path

import pytest
import torch
from huggingface_hub import hf_hub_download
from memray import Tracker
from sae_lens.training.activations_store import ActivationsStore
from sae_lens.training.session_loader import LMSparseAutoencoderSessionloader
from sae_lens.training.sparse_autoencoder import SparseAutoencoder
from tqdm import tqdm
from transformer_lens import HookedTransformer

from sae_vis.autoencoder import AutoEncoder, AutoEncoderConfig
from sae_vis.components_config import (
    ActsHistogramConfig,
    Column,
    FeatureTablesConfig,
    LogitsHistogramConfig,
    LogitsTableConfig,
    SequencesConfig,
)
from sae_vis.data_writing_fns import save_feature_centric_vis
from sae_vis.layout import SaeVisLayoutConfig
from sae_vis.sae_vis_data import SaeVisConfig
from sae_vis.sae_vis_runner import SaeVisRunner

ROOT_DIR = Path(__file__).parent.parent

DEVICE = "mps"

TEST_FEATURES = list(range(128))


@pytest.fixture
def model() -> HookedTransformer:
    model = HookedTransformer.from_pretrained("gpt2-small")
    model.to(DEVICE)
    return model


@pytest.fixture()
def sae():
    # Get autoencoder
    hook_point = "blocks.0.hook_resid_pre"
    sae_path = hf_hub_download(
        repo_id="jbloom/GPT2-Small-SAEs-Reformatted",
        filename=f"{hook_point}/sae_weights.safetensors",
    )
    hf_hub_download(
        repo_id="jbloom/GPT2-Small-SAEs-Reformatted",
        filename=f"{hook_point}/cfg.json",
    )
    hf_hub_download(
        repo_id="jbloom/GPT2-Small-SAEs-Reformatted",
        filename=f"{hook_point}/sparsity.safetensors",
    )
    gpt2_sae = SparseAutoencoder.load_from_pretrained(os.path.dirname(sae_path))
    gpt2_sae.to(DEVICE)

    return gpt2_sae


@pytest.fixture
def tokens(sae: SparseAutoencoder):
    def get_tokens(
        activation_store: ActivationsStore,
        n_batches_to_sample_from: int = 2**10,
        n_prompts_to_select: int = 4096 * 6,
    ):
        all_tokens_list = []
        pbar = tqdm(range(n_batches_to_sample_from))
        for _ in pbar:
            batch_tokens = activation_store.get_batch_tokens()
            batch_tokens = batch_tokens[torch.randperm(batch_tokens.shape[0])][
                : batch_tokens.shape[0]
            ]
            all_tokens_list.append(batch_tokens)

        all_tokens = torch.cat(all_tokens_list, dim=0)
        all_tokens = all_tokens[torch.randperm(all_tokens.shape[0])]
        return all_tokens[:n_prompts_to_select]

    # Get tokens, set model
    loader = LMSparseAutoencoderSessionloader(sae.cfg)
    _, _, activation_store = loader.load_sae_training_group_session()
    all_tokens_gpt = get_tokens(activation_store)
    return all_tokens_gpt


@pytest.fixture
def cfg(
    sae: SparseAutoencoder,
) -> SaeVisConfig:
    layout = SaeVisLayoutConfig(
        columns=[
            Column(
                SequencesConfig(
                    stack_mode="stack-all",
                    buffer=None,  # type: ignore
                    compute_buffer=True,
                    n_quantiles=5,
                    top_acts_group_size=20,
                    quantile_group_size=5,
                ),
                ActsHistogramConfig(),
                LogitsHistogramConfig(),
                LogitsTableConfig(),
                FeatureTablesConfig(n_rows=3),
            )
        ]
    )
    feature_vis_config_gpt = SaeVisConfig(
        hook_point=sae.cfg.hook_point,
        minibatch_size_features=128,
        minibatch_size_tokens=64,
        features=TEST_FEATURES,
        verbose=True,
        feature_centric_layout=layout,
    )

    return feature_vis_config_gpt


def test_benchmark_sae_vis_runner(
    cfg: SaeVisConfig,
    sae: SparseAutoencoder,
    model: HookedTransformer,
    tokens: torch.Tensor,
):
    # we've deleted the casting code so I'll have to re-implement it here

    os.remove("memory_profile.bin")
    os.remove("flamegraph.html")

    assert set(
        sae.state_dict().keys()
    ).issuperset(
        {"W_enc", "W_dec", "b_enc", "b_dec"}
    ), "If encoder isn't an AutoEncoder, it should have weights 'W_enc', 'W_dec', 'b_enc', 'b_dec'"
    d_in, d_hidden = sae.W_enc.shape
    device = sae.W_enc.device
    encoder_cfg = AutoEncoderConfig(
        d_in=d_in, d_hidden=d_hidden, dict_mult=d_hidden // d_in
    )
    autoencoder = AutoEncoder(encoder_cfg).to(device)
    autoencoder.load_state_dict(sae.state_dict(), strict=False)

    with Tracker("memory_profile.bin"):
        sae_vis_data = SaeVisRunner(cfg).run(
            encoder=autoencoder, model=model, tokens=tokens
        )

    # to view the flamegraph, run the following:
    # ! memray flamegraph memory_profile.bin --output flamegraph.html
    # ! open flamegraph.html

    save_path = "./gpt2_feature_centric_vis_test.html"
    save_feature_centric_vis(sae_vis_data, save_path)  # type: ignore
