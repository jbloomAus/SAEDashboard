# %%
import os
import shutil

from sae_dashboard.neuronpedia.neuronpedia_runner import (
    NeuronpediaRunner,
    NeuronpediaRunnerConfig,
)

# ================================================
# Configuration for CLT dashboards (GPT-2 Example)
# ================================================

# --- Crucial Parameters ---
# This MUST be the path to the LOCAL directory containing the saved CLT model
# and its 'cfg.json'. Loading from HF release is not yet supported for CLTs.
SAE_PATH = "/Users/curttigges/Projects/SAEDashboard/clt_test_pythia_160m"
# The main script will now loop over layers 0-11.
# CLT_LAYER_IDX = 1  # Example: Choose the layer index you want to visualize features for
# Optional: Specify the exact weights filename. If empty, the script will search
# for the first .safetensors file, then model.safetensors, then model.pt.
CLT_WEIGHTS_FILENAME = "pythia_160m_clt_jumprelu.safetensors"  # or "model.pt"

# --- Output & Naming ---
# These will be set dynamically in the main loop for each layer.
# NP_OUTPUT_FOLDER = f"neuronpedia_outputs_clt_pythia_160m_L{CLT_LAYER_IDX}/"
# ACT_CACHE_FOLDER = f"cached_activations_clt_pythia_160m_L{CLT_LAYER_IDX}"
# This name is primarily for organizing Neuronpedia outputs
# NP_SET_NAME = f"decode-clt-32k"
# This is less relevant for local loading but still required by the config
SAE_SET = "pythia-160m-clt-32k"

# --- Data & Model ---
# The base model ID should match the one the CLT was trained on (in cfg.json)
# GPT-2 is used as an example here.
BASE_MODEL_ID = "EleutherAI/pythia-160m"  # Ensure this matches your CLT's base model
HF_DATASET_PATH = "monology/pile-uncopyrighted"  # Dataset for generating activations

# --- Performance & Batching ---
# Adjust these based on your hardware and desired speed/granularity
NUM_FEATURES_PER_BATCH = 128  # How many features to process in one dashboard batch
START_BATCH = 0
NUM_BATCHES_TO_RUN = None  # How many batches to generate (set to None for all)
N_PROMPTS = 65536  # Total number of prompts for activation generation
N_TOKENS_IN_PROMPT = 128  # Context size for activation generation
N_PROMPTS_IN_FORWARD_PASS = 64  # Batch size for model forward passes

# --- Dtypes ---
# Specify dtypes for the base model and the CLT.
# Leave clt_dtype empty ("") to use the dtype saved in the CLT's config.
MODEL_DTYPE = "float32"
# SAE_DTYPE = "float32" # This config field is less relevant when clt_dtype is used
CLT_DTYPE = ""  # e.g., "float16" or "" to use CLT's config

# --- Misc ---
SPARSITY_THRESHOLD = 1  # Set to 1 to disable sparsity filtering for testing
USE_WANDB = False  # Enable WandB logging if desired

# %%
from transformer_lens import HookedTransformer  # type: ignore

model = HookedTransformer.from_pretrained("EleutherAI/pythia-160m")

print(model)


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


# %%
if __name__ == "__main__":

    # Basic check if the placeholder path was changed
    if SAE_PATH == "/path/to/your/local/gpt2_clt_model_directory":
        print(
            "ERROR: Please update the SAE_PATH variable in the script to your actual local CLT model directory."
        )
        exit(1)

    for clt_layer_idx in range(12):
        print(f"\n\n--- Starting Dashboard Generation for Layer {clt_layer_idx} ---")

        # --- Dynamic Configuration for each layer ---
        NP_OUTPUT_FOLDER = f"neuronpedia_outputs_clt_pythia_160m_L{clt_layer_idx}/"
        ACT_CACHE_FOLDER = f"cached_activations_clt_pythia_160m_L{clt_layer_idx}"
        NP_SET_NAME = f"decode-clt-32k"

        print("--- Running CLT Dashboard Generation ---")
        print(f"CLT Model Path: {SAE_PATH}")
        print(f"Target Layer:   {clt_layer_idx}")
        print(f"Output Folder:  {NP_OUTPUT_FOLDER}")
        print(f"Dataset:        {HF_DATASET_PATH}")
        print("--------------------------------------")

        # delete output files if present
        # Consider making this optional or adding a prompt
        if os.path.exists(ACT_CACHE_FOLDER):
            print(f"Deleting cache folder: {ACT_CACHE_FOLDER}")
            shutil.rmtree(ACT_CACHE_FOLDER)

        # Configure for CLT
        cfg = NeuronpediaRunnerConfig(
            # Model and Path Settings
            sae_set=SAE_SET,
            sae_path=SAE_PATH,
            from_local_sae=True,  # Must be True for CLT currently
            model_id=BASE_MODEL_ID,  # Pass base model ID here if needed downstream
            # CLT Specific Settings
            use_clt=True,
            clt_layer_idx=clt_layer_idx,
            clt_dtype=CLT_DTYPE,
            clt_weights_filename=CLT_WEIGHTS_FILENAME,
            use_transcoder=False,  # Ensure other loaders are off
            use_skip_transcoder=False,  # Ensure other loaders are off
            # Output Settings
            np_set_name=NP_SET_NAME,
            outputs_dir=NP_OUTPUT_FOLDER,
            # Dataset and Activation Settings
            huggingface_dataset_path=HF_DATASET_PATH,
            n_prompts_total=N_PROMPTS,
            n_tokens_in_prompt=N_TOKENS_IN_PROMPT,
            n_prompts_in_forward_pass=N_PROMPTS_IN_FORWARD_PASS,
            cache_dir=ACT_CACHE_FOLDER,
            # Batching and Processing Settings
            n_features_at_a_time=NUM_FEATURES_PER_BATCH,
            # start_batch=START_BATCH,
            # end_batch=START_BATCH + NUM_BATCHES_TO_RUN,
            # Dtype Settings
            # sae_dtype=SAE_DTYPE, # Can be omitted if clt_dtype is primary
            model_dtype=MODEL_DTYPE,
            # Misc Settings
            sparsity_threshold=SPARSITY_THRESHOLD,
            use_wandb=USE_WANDB,
        )

        runner = NeuronpediaRunner(cfg)
        runner.run()

        # Copy the output folder to /workspace/NP_OUTPUT_FOLDER
        copy_output_folder(NP_OUTPUT_FOLDER, f"/workspace/{NP_OUTPUT_FOLDER}")
        print(f"--- CLT Dashboard Generation Complete for Layer {clt_layer_idx} ---")
        print(f"Output saved to: {NP_OUTPUT_FOLDER} and /workspace/{NP_OUTPUT_FOLDER}")


# Example CLI usage (conceptual - requires local path):
# python -m sae_dashboard.neuronpedia.neuronpedia_runner \\
#   --sae-set="gpt2-clt-local" \\
#   --sae-path="/path/to/your/local/gpt2_clt_model_directory" \\
#   --from-local-sae \\
#   --np-set-name="clt-gpt2-L6" \\
#   --dataset-path="NeelNanda/pile-10k" \\
#   --output-dir="neuronpedia_outputs_clt_gpt2_L6/" \\
#   --model_dtype="float32" \\
#   --clt-dtype="" \\
#   --sparsity-threshold=1 \\
#   --n-prompts=4096 \\
#   --n-tokens-in-prompt=128 \\
#   --n-prompts-in-forward-pass=64 \\
#   --n-features-per-batch=64 \\
#   --start-batch=0 \\
#   --end-batch=5 \\
#   --use-clt \\
#   --clt-layer-idx=6

# %%
