from pathlib import Path

import pytest
import torch
from transformer_lens import HookedTransformer

from sae_dashboard.neuronpedia import neuronpedia_runner as runner_module
from sae_dashboard.neuronpedia.neuronpedia_runner import NeuronpediaRunner
from sae_dashboard.neuronpedia.neuronpedia_runner_config import NeuronpediaRunnerConfig


@pytest.fixture
def runner_config() -> NeuronpediaRunnerConfig:
    return NeuronpediaRunnerConfig(
        sae_set="gpt2-small-res-jb",
        sae_path="blocks.5.hook_resid_pre",
        outputs_dir="test_outputs",
        n_prompts_total=256,
        n_tokens_in_prompt=128,
        huggingface_dataset_path="monology/pile-uncopyrighted",
    )


@pytest.fixture
def neuronpedia_runner(
    runner_config: NeuronpediaRunnerConfig, model: HookedTransformer
) -> NeuronpediaRunner:
    runner = NeuronpediaRunner(runner_config)
    runner.model = model  # type: ignore
    return runner


def test_generate_tokens_no_duplicates(neuronpedia_runner: NeuronpediaRunner) -> None:
    tokens = neuronpedia_runner.generate_tokens(
        neuronpedia_runner.activations_store, n_prompts=256
    )
    assert tokens.shape == (256, neuronpedia_runner.cfg.n_tokens_in_prompt)
    # Move to CPU before checking uniqueness
    tokens_cpu = tokens.cpu()
    assert len(torch.unique(tokens_cpu, dim=0)) == 256


def test_get_tokens_no_duplicates(
    neuronpedia_runner: NeuronpediaRunner, tmp_path: Path
) -> None:
    neuronpedia_runner.cfg.outputs_dir = str(tmp_path)
    tokens = neuronpedia_runner.get_tokens()
    assert tokens.shape == (
        neuronpedia_runner.cfg.n_prompts_total,
        neuronpedia_runner.cfg.n_tokens_in_prompt,
    )
    # Move to CPU before checking uniqueness
    tokens_cpu = tokens.cpu()
    assert (
        len(torch.unique(tokens_cpu, dim=0)) == neuronpedia_runner.cfg.n_prompts_total
    )


def test_get_tokens_uses_tokens_path(tmp_path: Path) -> None:
    tokens = torch.arange(4 * 8).reshape(4, 8)
    tokens_path = tmp_path / "chat_tokens.pt"
    torch.save(tokens, tokens_path)
    runner = NeuronpediaRunner.__new__(NeuronpediaRunner)
    runner.cfg = NeuronpediaRunnerConfig(
        sae_set="sae/repo",
        sae_path="blocks.5.hook_resid_pre",
        outputs_dir=str(tmp_path),
        n_prompts_total=4,
        n_tokens_in_prompt=8,
        tokens_path=str(tokens_path),
    )
    assert torch.equal(runner.get_tokens(), tokens)

    runner.cfg.n_tokens_in_prompt = 16
    with pytest.raises(ValueError, match="expected"):
        runner.get_tokens()


def test_run_neuronpedia_export_uses_neuronpedia_model_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = NeuronpediaRunnerConfig(
        sae_set="sae/repo",
        sae_path="blocks.5.hook_resid_pre",
        np_set_name="resid-pre",
        outputs_dir=str(tmp_path / "dashboards"),
        output_neuronpedia_exports=True,
        neuronpedia_exports_dir=str(tmp_path / "exports"),
        neuronpedia_creator_name="Test Creator",
        neuronpedia_release_id="test-release",
        neuronpedia_release_title="Test Release",
        neuronpedia_release_url="https://example.com/release",
        neuronpedia_source_set_description="Residual stream",
        neuronpedia_model_name="neuronpedia-model",
    )
    runner = NeuronpediaRunner.__new__(NeuronpediaRunner)
    runner.cfg = cfg
    runner.layer = 5
    runner.model_id = "hf-org/hf-model-path"
    runner.hook_name = "blocks.5.hook_resid_pre"

    captured: dict[str, runner_module.NeuronpediaExportConfig] = {}

    def fake_export_neuronpedia_dashboards(
        export_cfg: runner_module.NeuronpediaExportConfig,
    ) -> str:
        captured["export_cfg"] = export_cfg
        return str(tmp_path / "exports" / "neuronpedia-model" / "5-resid-pre")

    monkeypatch.setattr(
        runner_module,
        "export_neuronpedia_dashboards",
        fake_export_neuronpedia_dashboards,
    )

    runner._run_neuronpedia_export()

    assert captured["export_cfg"].exports_dir == str(tmp_path / "exports")
    assert captured["export_cfg"].model_name == "neuronpedia-model"


# def test_add_prefix_suffix_to_tokens(neuronpedia_runner: NeuronpediaRunner) -> None:
#     # modify the config to add a prefix / suffix
#     neuronpedia_runner.cfg.prefix_tokens = [101, 102, 103]  # Example prefix tokens
#     neuronpedia_runner.cfg.suffix_tokens = [104, 105, 106]  # Example suffix tokens

#     # get the tokens
#     tokens = neuronpedia_runner.get_tokens()
#     tokens = neuronpedia_runner.add_prefix_suffix_to_tokens(tokens)

#     # check that the tokens have the prefix and suffix
#     assert torch.allclose(tokens[:, 1:4].cpu(), torch.tensor([101, 102, 103]))
#     assert torch.allclose(tokens[:, -3:].cpu(), torch.tensor([104, 105, 106]))

#     # assert the first token position is still the bos
#     assert torch.allclose(
#         tokens[:, 0].cpu(),
#         torch.tensor(
#             [neuronpedia_runner.model.to_single_token("<|endoftext|>")],
#             dtype=torch.int64,
#         ),
#     )


# def test_add_prefix_suffix_to_tokens_prepend_bos_false(
#     neuronpedia_runner: NeuronpediaRunner,
# ) -> None:
#     # modify the config to add a prefix / suffix
#     neuronpedia_runner.cfg.prefix_tokens = [101, 102, 103]  # Example prefix tokens
#     neuronpedia_runner.cfg.suffix_tokens = [104, 105, 106]  # Example suffix tokens

#     # get the tokens
#     neuronpedia_runner.sae.cfg.prepend_bos = False
#     tokens = neuronpedia_runner.get_tokens()
#     tokens = neuronpedia_runner.add_prefix_suffix_to_tokens(tokens)

#     # check that the tokens have the prefix and suffix
#     assert torch.allclose(tokens[:, 0:3].cpu(), torch.tensor([101, 102, 103]))
#     assert torch.allclose(tokens[:, -3:].cpu(), torch.tensor([104, 105, 106]))

#     # assert the first token position is still the bos
#     assert not torch.allclose(
#         tokens[:, 0].cpu(),
#         torch.tensor(
#             [neuronpedia_runner.model.to_single_token("<|endoftext|>")],
#             dtype=torch.int64,
#         ),
#     )
