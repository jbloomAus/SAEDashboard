import json
import os
from typing import Type, TypeVar
from unittest.mock import MagicMock, patch

from sae_dashboard.neuronpedia.neuronpedia_dashboard import NeuronpediaDashboardBatch
from sae_dashboard.neuronpedia.neuronpedia_runner import (
    NeuronpediaRunner,
    NeuronpediaRunnerConfig,
)

# depending on if device type, the results may be slightly different
CORRECT_VALUE_TOLERANCE = 0.1

T = TypeVar("T")


def json_to_class(json_file: str, cls: Type[T]) -> T:
    with open(json_file, "r") as file:
        data = json.load(file)
    return cls(**data)


@patch("sae_dashboard.neuronpedia.neuronpedia_runner.wandb.log")
@patch("sae_dashboard.neuronpedia.neuronpedia_runner.wandb.init")
def test_simple_neuronpedia_runner(
    mock_wandb_init: MagicMock, mock_wandb_log: MagicMock
):
    np_output_folder = "neuronpedia_outputs/test_simple"
    act_cache_folder = "cached_activations"
    correct_outputs_folder = "tests/acceptance/test_simple"
    sae_set = "gpt2-small-res-jb"
    sae_path = "blocks.0.hook_resid_pre"
    num_features_per_batch = 2
    num_batches = 2

    # delete output files if present
    os.system(f"rm -rf {np_output_folder}")
    os.system(f"rm -rf {act_cache_folder}")

    # # we make two batches of 2 features each
    cfg = NeuronpediaRunnerConfig(
        sae_set=sae_set,
        sae_path=sae_path,
        np_set_name="res-jb",
        from_local_sae=False,
        outputs_dir=np_output_folder,
        sparsity_threshold=1,
        n_prompts_total=5000,
        n_features_at_a_time=num_features_per_batch,
        n_prompts_in_forward_pass=32,
        start_batch=0,
        end_batch=num_batches - 1,
        use_wandb=True,
        shuffle_tokens=False,
        huggingface_dataset_path="monology/pile-uncopyrighted",
    )

    runner = NeuronpediaRunner(cfg)
    runner.run()

    # assert the actual features/batches
    for i in range(0, num_batches - 1):
        correct_path = os.path.join(correct_outputs_folder, f"batch-{i}.json")

        correct_data = json_to_class(correct_path, NeuronpediaDashboardBatch)

        test_path = os.path.join(runner.cfg.outputs_dir, f"batch-{i}.json")
        assert os.path.exists(test_path), f"file {test_path} does not exist"
        test_data = json_to_class(test_path, NeuronpediaDashboardBatch)

        assert test_data == correct_data

    assert "run_settings.json" in os.listdir(runner.cfg.outputs_dir)


@patch("sae_dashboard.neuronpedia.neuronpedia_runner.wandb.log")
@patch("sae_dashboard.neuronpedia.neuronpedia_runner.wandb.init")
def test_benchmark_neuronpedia_runner(
    mock_wandb_init: MagicMock, mock_wandb_log: MagicMock
):
    np_output_folder = "neuronpedia_outputs/benchmark"
    sae_set = "gpt2-small-res-jb"
    sae_path = "blocks.0.hook_resid_pre"
    print(sae_path)

    # delete output files if present
    os.system(f"rm -rf {np_output_folder}")
    cfg = NeuronpediaRunnerConfig(
        sae_set=sae_set,
        sae_path=sae_path,
        from_local_sae=False,
        outputs_dir=np_output_folder,
        sparsity_threshold=1,
        n_prompts_total=1024,
        n_features_at_a_time=32,
        start_batch=0,
        end_batch=8,
        use_wandb=True,
        sae_device="cpu",
        model_device="cpu",
        model_n_devices=1,
        activation_store_device="cpu",
        huggingface_dataset_path="monology/pile-uncopyrighted",
    )

    runner = NeuronpediaRunner(cfg)
    runner.run()


@patch("sae_dashboard.neuronpedia.neuronpedia_runner.wandb.log")
@patch("sae_dashboard.neuronpedia.neuronpedia_runner.wandb.init")
def test_simple_neuronpedia_runner_hook_z_sae(
    mock_wandb_init: MagicMock, mock_wandb_log: MagicMock
):
    np_output_folder = "neuronpedia_outputs/test_attn"
    act_cache_folder = "cached_activations"
    sae_set = "gpt2-small-hook-z-kk"
    sae_path = "blocks.0.hook_z"
    num_features_per_batch = 2
    num_batches = 2

    # delete output files if present
    os.system(f"rm -rf {np_output_folder}")
    os.system(f"rm -rf {act_cache_folder}")

    # # we make two batches of 2 features each
    cfg = NeuronpediaRunnerConfig(
        sae_set=sae_set,
        sae_path=sae_path,
        np_set_name="att-kk",
        from_local_sae=False,
        outputs_dir=np_output_folder,
        sparsity_threshold=1,
        n_prompts_total=5000,
        n_features_at_a_time=num_features_per_batch,
        n_prompts_in_forward_pass=32,
        start_batch=0,
        end_batch=num_batches - 1,
        use_wandb=True,
        shuffle_tokens=False,
        huggingface_dataset_path="monology/pile-uncopyrighted",
    )

    runner = NeuronpediaRunner(cfg)
    runner.run()

    assert "run_settings.json" in os.listdir(runner.cfg.outputs_dir)


@patch("sae_dashboard.neuronpedia.neuronpedia_runner.wandb.log")
@patch("sae_dashboard.neuronpedia.neuronpedia_runner.wandb.init")
def test_neuronpedia_runner_prefix_suffix_it_model(
    mock_wandb_init: MagicMock, mock_wandb_log: MagicMock
):
    np_output_folder = "neuronpedia_outputs/test_masking"
    act_cache_folder = "cached_activations"
    sae_set = "gpt2-small-res-jb"
    sae_path = "blocks.0.hook_resid_pre"
    num_features_per_batch = 2
    num_batches = 2

    # delete output files if present
    os.system(f"rm -rf {np_output_folder}")
    os.system(f"rm -rf {act_cache_folder}")

    # # we make two batches of 2 features each
    cfg = NeuronpediaRunnerConfig(
        sae_set=sae_set,
        sae_path=sae_path,
        np_set_name="res-jb",
        from_local_sae=False,
        outputs_dir=np_output_folder,
        sparsity_threshold=1,
        n_prompts_total=5000,
        n_features_at_a_time=num_features_per_batch,
        n_prompts_in_forward_pass=32,
        start_batch=0,
        end_batch=num_batches - 1,
        use_wandb=True,
        shuffle_tokens=False,
        prefix_tokens=[106, 1645, 108],
        suffix_tokens=[107, 108],
        ignore_positions=[0, 1, 2],
        huggingface_dataset_path="monology/pile-uncopyrighted",
    )

    runner = NeuronpediaRunner(cfg)
    runner.run()

    assert "run_settings.json" in os.listdir(runner.cfg.outputs_dir)
