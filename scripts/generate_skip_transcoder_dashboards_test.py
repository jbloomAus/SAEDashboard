# %%
import os

from sae_dashboard.neuronpedia.neuronpedia_runner import (
    NeuronpediaRunner,
    NeuronpediaRunnerConfig,
)

# Configuration for SkipTranscoder dashboards

NP_OUTPUT_FOLDER = "neuronpedia_outputs_skip_tc/"
ACT_CACHE_FOLDER = "cached_activations_skip_tc"
NP_SET_NAME = "skip-transcoder-Llama-3.2-1B-131k-nobos-relu"
SAE_SET = "llama-3.2-1b-relu-skip-transcoders"  # Adjust to your actual HF release name
SAE_PATH = "blocks.9.hook_resid_mid"  # Adjust to your actual SkipTranscoder ID
NUM_FEATURES_PER_BATCH = 10
NUM_BATCHES = 10
HF_DATASET_PATH = "monology/pile-uncopyrighted"

SPARSITY_THRESHOLD = 1

# IMPORTANT
SAE_DTYPE = "float32"
MODEL_DTYPE = "bfloat16"

# PERFORMANCE SETTING
# N_PROMPTS = 24576
N_PROMPTS = 4096
N_TOKENS_IN_PROMPT = 128
N_PROMPTS_IN_FORWARD_PASS = 32


# %%
if __name__ == "__main__":

    # delete output files if present
    os.system(f"rm -rf {NP_OUTPUT_FOLDER}")
    os.system(f"rm -rf {ACT_CACHE_FOLDER}")

    # Configure for SkipTranscoder
    cfg = NeuronpediaRunnerConfig(
        sae_set=SAE_SET,
        sae_path=SAE_PATH,
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
        # Enable SkipTranscoder loading
        use_skip_transcoder=True,
        # TESTING ONLY
        end_batch=6,
    )

    runner = NeuronpediaRunner(cfg)
    runner.run()


# Example CLI usage:
# neuronpedia-runner \
#   --sae-set="gemma-scope-2b-pt-skiptranscoders" \
#   --sae-path="layer_15/width_16k/skip_l0_8" \
#   --np-set-name="gemmascope-skip-tc-16k" \
#   --dataset-path="monology/pile-uncopyrighted" \
#   --output-dir="neuronpedia_outputs_skip_tc/" \
#   --dtype="float32" \
#   --sparsity-threshold=1 \
#   --n-prompts=4096 \
#   --n-tokens-in-prompt=128 \
#   --n-prompts-in-forward-pass=128 \
#   --n-features-per-batch=10 \
#   --start-batch=0 \
#   --end-batch=6 \
#   --use-wandb \
#   --use-skip-transcoder

# %%
