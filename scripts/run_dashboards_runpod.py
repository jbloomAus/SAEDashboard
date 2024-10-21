import os
import shutil

from sae_dashboard.neuronpedia.neuronpedia_runner import (
    NeuronpediaRunner,
    NeuronpediaRunnerConfig,
)

# python neuronpedia.py generate --sae-set=res-jb --sae-path=/opt/Gemma-2b-Residual-Stream-SAEs/gemma_2b_blocks.10.hook_resid_post_16384 --dataset-path=Skylion007/openwebtext --log-sparsity=-6 --dtype= --feat-per-batch=128 --n-prompts=24576 --n-context-tokens=128 --n-prompts-in-forward-pass=128 --resume-from-batch=0 --end-at-batch=-1

# tuples of sae_path, np_sae_id_suffix
sae_paths_and_np_sae_id_suffixes: list[tuple[str, str | None]] = [
    ("layer_1/width_16k/average_l0_56", None),
    ("layer_34/width_16k/average_l0_47", None),
]


def copy_output_folder(source: str, destination: str):
    """
    Copy the contents of the source folder to the destination folder.
    If the destination folder doesn't exist, it will be created.
    """
    if not os.path.exists(destination):
        os.makedirs(destination)

    # Find the most recently created subfolder in the source directory
    subfolders = [
        f for f in os.listdir(source) if os.path.isdir(os.path.join(source, f))
    ]
    if not subfolders:
        print(f"No subfolders found in {source}")
        return

    latest_subfolder = max(
        subfolders, key=lambda f: os.path.getctime(os.path.join(source, f))
    )
    source_path = os.path.join(source, latest_subfolder)
    dest_path = os.path.join(destination, latest_subfolder)

    # Copy the subfolder
    shutil.copytree(source_path, dest_path, dirs_exist_ok=True)
    print(f"Copied {source_path} to {dest_path}")


for sae_path, np_sae_id_suffix in sae_paths_and_np_sae_id_suffixes:
    # LOCAL PATHS
    NP_OUTPUT_FOLDER = "neuronpedia_outputs/"
    ACT_CACHE_FOLDER = "cached_activations"

    # NP SET NAME
    NP_SET_NAME = "gemmascope-mlp-16k-l0_32plus"
    SAE_SET = "gemma-scope-9b-pt-mlp"
    SAE_PATH = sae_path
    NP_SAE_ID_SUFFIX = np_sae_id_suffix

    # DATAEST
    HF_DATASET_PATH = "monology/pile-uncopyrighted"

    SPARSITY_THRESHOLD = 1

    # IMPORTANT
    SAE_DTYPE = "float32"
    MODEL_DTYPE = "bfloat16"

    # PERFORMANCE SETTING
    N_PROMPTS = 24576
    N_TOKENS_IN_PROMPT = 128
    N_PROMPTS_IN_FORWARD_PASS = 256
    NUM_FEATURES_PER_BATCH = 256

    if __name__ == "__main__":

        # delete output files if present
        # os.system(f"rm -rf {NP_OUTPUT_FOLDER}")
        # os.system(f"rm -rf {ACT_CACHE_FOLDER}")

        # # we make two batches of 2 features each
        cfg = NeuronpediaRunnerConfig(
            sae_set=SAE_SET,
            sae_path=sae_path,
            np_sae_id_suffix=NP_SAE_ID_SUFFIX,
            np_set_name=NP_SET_NAME,
            from_local_sae=False,
            huggingface_dataset_path=HF_DATASET_PATH,
            sae_dtype=SAE_DTYPE,
            model_dtype=MODEL_DTYPE,
            outputs_dir=NP_OUTPUT_FOLDER,
            sparsity_threshold=SPARSITY_THRESHOLD,
            n_prompts_total=N_PROMPTS,
            n_tokens_in_prompt=N_TOKENS_IN_PROMPT,
            n_prompts_in_forward_pass=N_PROMPTS_IN_FORWARD_PASS,
            n_features_at_a_time=NUM_FEATURES_PER_BATCH,
            start_batch=0,
            use_wandb=True,
            use_dfa=False,
            # TESTING ONLY
            # end_batch=6,
        )

        runner = NeuronpediaRunner(cfg)
        runner.run()

        # Copy the output folder to /workspace/NP_OUTPUT_FOLDER
        copy_output_folder(NP_OUTPUT_FOLDER, f"/workspace/{NP_OUTPUT_FOLDER}")
