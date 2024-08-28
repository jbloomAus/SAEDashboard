from pathlib import Path

import pytest
import torch
from transformer_lens import HookedTransformer

from sae_dashboard.neuronpedia.neuronpedia_runner import NeuronpediaRunner
from sae_dashboard.neuronpedia.neuronpedia_runner_config import NeuronpediaRunnerConfig


@pytest.fixture
def runner_config() -> NeuronpediaRunnerConfig:
    return NeuronpediaRunnerConfig(
        sae_set="gpt2-small-res-jb",
        sae_path="blocks.5.hook_resid_pre",
        outputs_dir="test_outputs",
        n_prompts_total=1000,
        n_tokens_in_prompt=128,
        huggingface_dataset_path="Skylion007/openwebtext",
    )


@pytest.fixture
def neuronpedia_runner(
    runner_config: NeuronpediaRunnerConfig, model: HookedTransformer
) -> NeuronpediaRunner:
    runner = NeuronpediaRunner(runner_config)
    runner.model = model
    return runner


def test_generate_tokens_no_duplicates(neuronpedia_runner: NeuronpediaRunner) -> None:
    tokens = neuronpedia_runner.generate_tokens(
        neuronpedia_runner.activations_store, n_prompts=1000
    )
    assert tokens.shape == (1000, neuronpedia_runner.cfg.n_tokens_in_prompt)
    # Move to CPU before checking uniqueness
    tokens_cpu = tokens.cpu()
    assert len(torch.unique(tokens_cpu, dim=0)) == 1000


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
