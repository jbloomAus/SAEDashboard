#\!/usr/bin/env python3
"""Minimal test script to verify transcoder loading works"""

import os
from sae_dashboard.neuronpedia.neuronpedia_runner import (
    NeuronpediaRunner,
    NeuronpediaRunnerConfig,
)

# Configuration
NP_OUTPUT_FOLDER = "test_outputs/"
SAE_SET = "gemma-scope-2b-pt-transcoders"
SAE_PATH = "layer_15/width_16k/average_l0_8"
HF_DATASET_PATH = "monology/pile-uncopyrighted"

# Clean up previous outputs
os.system(f"rm -rf {NP_OUTPUT_FOLDER}")

# Create config
cfg = NeuronpediaRunnerConfig(
    sae_set=SAE_SET,
    sae_path=SAE_PATH,
    np_set_name="test-transcoders",
    from_local_sae=False,
    huggingface_dataset_path=HF_DATASET_PATH,
    sae_dtype="float32",
    model_dtype="bfloat16",
    outputs_dir=NP_OUTPUT_FOLDER,
    sparsity_threshold=1,
    n_prompts_total=128,  # Small number for testing
    n_tokens_in_prompt=128,
    n_prompts_in_forward_pass=32,
    n_features_at_a_time=2,  # Just 2 features for quick test
    start_batch=0,
    end_batch=1,  # Just one batch
    use_wandb=False,  # Disable wandb for testing
    use_transcoder=True,  # Enable transcoder loading
)

# Run
runner = NeuronpediaRunner(cfg)
runner.run()

print("\nTest completed! Check the outputs in:", NP_OUTPUT_FOLDER)