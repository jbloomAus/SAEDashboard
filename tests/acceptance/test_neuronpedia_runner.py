import os

from sae_lens.toolkit.pretrained_saes import download_sae_from_hf

from sae_dashboard.neuronpedia.neuronpedia_runner import (
    NeuronpediaRunner,
    NeuronpediaRunnerConfig,
)


# pytest -s tests/benchmark/test_neuronpedia_runner.py::test_benchmark_neuronpedia_runner
def test_benchmark_neuronpedia_runner():

    # MODEL_ID = "gpt2-small"

    (_, SAE_WEIGHTS_PATH, _) = download_sae_from_hf(
        "jbloom/GPT2-Small-SAEs-Reformatted", "blocks.0.hook_resid_pre"
    )

    NP_OUTPUT_FOLDER = "neuronpedia_outputs/benchmark"
    SAE_ID = "res-jb"
    SAE_PATH = os.path.dirname(SAE_WEIGHTS_PATH)
    print(SAE_PATH)

    # delete output files if present
    os.system(f"rm -rf {NP_OUTPUT_FOLDER}")
    cfg = NeuronpediaRunnerConfig(
        sae_id=SAE_ID,
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
    SAE_ID = "res-jb"
    SAE_PATH = os.path.dirname(SAE_WEIGHTS_PATH)
    print(SAE_PATH)

    # delete output files if present
    os.system(f"rm -rf {NP_OUTPUT_FOLDER}")
    cfg = NeuronpediaRunnerConfig(
        sae_id=SAE_ID,
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
