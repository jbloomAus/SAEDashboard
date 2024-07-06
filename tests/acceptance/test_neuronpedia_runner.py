import json
import os
from typing import Type, TypeVar

from sae_lens.toolkit.pretrained_saes import download_sae_from_hf

from sae_dashboard.neuronpedia.neuronpedia_feature import NeuronpediaDashboardBatch
from sae_dashboard.neuronpedia.neuronpedia_runner import (
    NeuronpediaRunner,
    NeuronpediaRunnerConfig,
)

CORRECT_VALUE_TOLERANCE = 0.1

T = TypeVar("T")


def json_to_class(json_file: str, cls: Type[T]) -> T:
    with open(json_file, "r") as file:
        data = json.load(file)
    return cls(**data)


# pytest -s tests/acceptance/test_neuronpedia_runner.py::test_simple_neuronpedia_runner
def test_simple_neuronpedia_runner():

    (_, SAE_WEIGHTS_PATH, _) = download_sae_from_hf(
        "jbloom/GPT2-Small-SAEs-Reformatted", "blocks.0.hook_resid_pre"
    )

    NP_OUTPUT_FOLDER = "neuronpedia_outputs/test_simple"
    ACT_CACHE_FOLDER = "cached_activations"
    CORRECT_OUTPUTS_FOLDER = "tests/acceptance/test_simple"
    SAE_SET = "res-jb"
    SAE_PATH = os.path.dirname(SAE_WEIGHTS_PATH)
    NUM_FEATURES_PER_BATCH = 2
    NUM_BATCHES = 2

    # # delete output files if present
    os.system(f"rm -rf {NP_OUTPUT_FOLDER}")
    os.system(f"rm -rf {ACT_CACHE_FOLDER}")

    # # we make two batches of 2 features each
    cfg = NeuronpediaRunnerConfig(
        sae_set=SAE_SET,
        sae_path=SAE_PATH,
        outputs_dir=NP_OUTPUT_FOLDER,
        sparsity_threshold=-5,
        n_batches_to_sample_from=1000,
        n_prompts_to_select=5000,
        n_features_at_a_time=NUM_FEATURES_PER_BATCH,
        start_batch=0,
        end_batch=NUM_BATCHES - 1,
        use_wandb=True,
        shuffle_tokens=False,
    )

    runner = NeuronpediaRunner(cfg)
    runner.run()

    # assert sparsity/skipped
    # load skipped_indexes.json file
    skipped_path = os.path.join(NP_OUTPUT_FOLDER, "skipped_indexes.json")
    assert os.path.exists(skipped_path), f"file {skipped_path} does not exist"
    with open(skipped_path, "r") as file:
        skipped_test_data = json.load(file)
        # load skipped_indexes.json file from CORRECT_OUTPUTS_FOLDER
        skipped_correct_path = os.path.join(
            CORRECT_OUTPUTS_FOLDER, "skipped_indexes.json"
        )
        with open(skipped_correct_path, "r") as file:
            skipped_correct_data = json.load(file)
            assert skipped_test_data == skipped_correct_data

    # assert the actual features/batches
    for i in range(0, NUM_BATCHES - 1):
        correct_path = os.path.join(CORRECT_OUTPUTS_FOLDER, f"batch-{i}.json")

        correct_data = json_to_class(correct_path, NeuronpediaDashboardBatch)

        test_path = os.path.join(NP_OUTPUT_FOLDER, f"batch-{i}.json")
        assert os.path.exists(test_path), f"file {test_path} does not exist"
        test_data = json_to_class(test_path, NeuronpediaDashboardBatch)

        assert test_data == correct_data


# pytest -s tests/benchmark/test_neuronpedia_runner.py::test_benchmark_neuronpedia_runner
def test_benchmark_neuronpedia_runner():

    # MODEL_ID = "gpt2-small"

    (_, SAE_WEIGHTS_PATH, _) = download_sae_from_hf(
        "jbloom/GPT2-Small-SAEs-Reformatted", "blocks.0.hook_resid_pre"
    )

    NP_OUTPUT_FOLDER = "neuronpedia_outputs/benchmark"
    SAE_SET = "res-jb"
    SAE_PATH = os.path.dirname(SAE_WEIGHTS_PATH)
    print(SAE_PATH)

    # delete output files if present
    os.system(f"rm -rf {NP_OUTPUT_FOLDER}")
    cfg = NeuronpediaRunnerConfig(
        sae_set=SAE_SET,
        sae_path=SAE_PATH,
        outputs_dir=NP_OUTPUT_FOLDER,
        sparsity_threshold=-5,
        n_batches_to_sample_from=1000,
        n_prompts_to_select=1000,
        n_features_at_a_time=32,
        start_batch=0,
        end_batch=8,
        use_wandb=True,
        sae_device="cpu",
        model_device="cpu",
        model_n_devices=1,
        activation_store_device="cpu",
    )

    runner = NeuronpediaRunner(cfg)
    runner.run()


# assume we have 4 devices, will put model on first 3, SAE on the last.
def test_benchmark_neuronpedia_runner_distributed():

    # MODEL_ID = "gpt2-small"

    (_, SAE_WEIGHTS_PATH, _) = download_sae_from_hf(
        "jbloom/GPT2-Small-SAEs-Reformatted", "blocks.0.hook_resid_pre"
    )

    NP_OUTPUT_FOLDER = "neuronpedia_outputs/benchmark"
    SAE_SET = "res-jb"
    SAE_PATH = os.path.dirname(SAE_WEIGHTS_PATH)
    print(SAE_PATH)

    # delete output files if present
    os.system(f"rm -rf {NP_OUTPUT_FOLDER}")
    cfg = NeuronpediaRunnerConfig(
        sae_set=SAE_SET,
        sae_path=SAE_PATH,
        outputs_dir=NP_OUTPUT_FOLDER,
        sparsity_threshold=-5,
        n_batches_to_sample_from=1000,
        n_prompts_to_select=1000,
        n_features_at_a_time=32,
        start_batch=0,
        end_batch=8,
        use_wandb=True,
        sae_device="cuda:3",
        model_device="cuda",
        model_n_devices=3,
        activation_store_device="cpu",
    )

    runner = NeuronpediaRunner(cfg)
    runner.run()
