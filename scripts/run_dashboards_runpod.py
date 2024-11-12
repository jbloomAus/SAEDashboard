import os
import shutil

from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory

from sae_dashboard.neuronpedia.neuronpedia_runner import (
    NeuronpediaRunner,
    NeuronpediaRunnerConfig,
)

N_PROMPTS = 32768

HF_DATASET_PATH = "monology/pile-uncopyrighted"

SPARSITY_THRESHOLD = 1
SAE_DTYPE = "float32"
MODEL_DTYPE = "bfloat16"

N_TOKENS_IN_PROMPT = 128
N_PROMPTS_IN_FORWARD_PASS = 256
NUM_FEATURES_PER_BATCH = 512

list_of_saes: list[tuple[str, str]] = [
    (
        "blocks.19.hook_resid_post__trainer_0_step_0",
        "gemma-2-2b/19-sae_bench-standard-res-4k__trainer_0_step_0",
    ),
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


for saelens_sae_id, saelens_np_id in list_of_saes:

    directory = get_pretrained_saes_directory()

    def find_saelens_release_from_neuronpedia_id(neuronpedia_id: str) -> str:
        for release, item in directory.items():
            for _, np_id in item.neuronpedia_id.items():  # type: ignore
                if np_id == neuronpedia_id:
                    return release
        raise ValueError(f"Neuronpedia ID {neuronpedia_id} not found")

    SAELENS_RELEASE = find_saelens_release_from_neuronpedia_id(saelens_np_id)
    print(f"SAELENS_RELEASE: {SAELENS_RELEASE}")

    NP_SET_NAME = saelens_np_id.split("/")[1].split("-", 1)[
        1
    ]  # sae_bench-topk-res-16k__trainer_23_step_final
    if "__" in NP_SET_NAME:
        NP_SET_NAME, NP_SAE_ID_SUFFIX = NP_SET_NAME.rsplit("__", 1)
    else:
        NP_SAE_ID_SUFFIX = None

    print(f"NP_SET_NAME: {NP_SET_NAME}")
    print(f"NP_SAE_ID_SUFFIX: {NP_SAE_ID_SUFFIX}")

    # LOCAL PATHS
    NP_OUTPUT_FOLDER = "neuronpedia_outputs/"
    ACT_CACHE_FOLDER = "cached_activations"

    if __name__ == "__main__":

        # delete output files if present
        # os.system(f"rm -rf {NP_OUTPUT_FOLDER}")
        # os.system(f"rm -rf {ACT_CACHE_FOLDER}")

        # # we make two batches of 2 features each
        cfg = NeuronpediaRunnerConfig(
            sae_set=SAELENS_RELEASE,
            sae_path=saelens_sae_id,
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
        # copy_output_folder(NP_OUTPUT_FOLDER, f"/workspace/{NP_OUTPUT_FOLDER}")
