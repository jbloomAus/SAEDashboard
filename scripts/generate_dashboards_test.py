import os

from sae_dashboard.neuronpedia.neuronpedia_runner import (
    NeuronpediaRunner,
    NeuronpediaRunnerConfig,
)

# python neuronpedia.py generate --sae-set=res-jb --sae-path=/opt/Gemma-2b-Residual-Stream-SAEs/gemma_2b_blocks.10.hook_resid_post_16384 --dataset-path=Skylion007/openwebtext --log-sparsity=-6 --dtype= --feat-per-batch=128 --n-prompts=24576 --n-context-tokens=128 --n-prompts-in-forward-pass=128 --resume-from-batch=0 --end-at-batch=-1


NP_OUTPUT_FOLDER = "neuronpedia_outputs/"
ACT_CACHE_FOLDER = "cached_activations"
NP_SET_NAME = "gemmascope-res-65k"
SAE_SET = "gemma-scope-2b-pt-res-canonical"
SAE_PATH = "layer_0/width_65k/canonical"
NUM_FEATURES_PER_BATCH = 2
NUM_BATCHES = 2
HF_DATASET_PATH = "monology/pile-uncopyrighted"


SPARSITY_THRESHOLD = 1

# IMPORTANT
SAE_DTYPE = "float32"
MODEL_DTYPE = "bfloat16"

# PERFORMANCE SETTING
# N_PROMPTS = 24576
N_PROMPTS = 128
N_TOKENS_IN_PROMPT = 128
N_PROMPTS_IN_FORWARD_PASS = 128


if __name__ == "__main__":
    # delete output files if present
    os.system(f"rm -rf {NP_OUTPUT_FOLDER}")
    os.system(f"rm -rf {ACT_CACHE_FOLDER}")

    # # we make two batches of 2 features each
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
        # TESTING ONLY
        end_batch=6,
    )

    runner = NeuronpediaRunner(cfg)
    runner.run()


# neuronpedia-runner \
#   --sae-set="gemma-scope-2b-pt-res-canonical" \
#   --sae-path="layer_0/width_65k/canonical" \
#   --np-set-name="gemmascope-res-65k" \
#   --dataset-path="monology/pile-uncopyrighted" \
#   --output-dir="neuronpedia_outputs/" \
#   --dtype="float32" \
#   --sparsity-threshold=1 \
#   --n-prompts=128 \
#   --n-tokens-in-prompt=128 \
#   --n-prompts-in-forward-pass=128 \
#   --n-features-per-batch=2 \
#   --start-batch=0 \
#   --end-batch=6 \
#   --use-wandb
