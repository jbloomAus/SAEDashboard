# I'm running this in an A100 with 90GB of GPU Ram.
# I'm using TransformerLens 2.2 which I manually installed from source.
# I'm a few edits to fix bfloat16 errors (but I've since made PR's so latest SAE Lens / SAE dashboard should be fine here).
import os

from sae_dashboard.neuronpedia.neuronpedia_runner import (
    NeuronpediaRunner,
    NeuronpediaRunnerConfig,
)

# GET WEIGHTS FROM WANDB
# import wandb
# run = wandb.init()
# artifact = run.use_artifact('jbloom/gemma-2-9b_test/sae_gemma-2-9b_blocks.24.hook_resid_post_114688:v7', type='model')
# artifact_dir = artifact.download()


# Get Sparsity from Wandb (and manually move it accross)
# import wandb
# run = wandb.init()
# artifact = run.use_artifact('jbloom/gemma-2-9b_test/sae_gemma-2-9b_blocks.24.hook_resid_post_114688_log_feature_sparsity:v7', type='log_feature_sparsity')
# artifact_dir = artifact.download()

NP_OUTPUT_FOLDER = "neuronpedia_outputs/gemma-2-9b-test"
SAE_SET = "res-jb-test"
SAE_PATH = "artifacts/sae_gemma-2-9b_blocks.24.hook_resid_post_114688:v7"
print(SAE_PATH)

# delete output files if present
os.system(f"rm -rf {NP_OUTPUT_FOLDER}")
cfg = NeuronpediaRunnerConfig(
    sae_set=SAE_SET,
    sae_path=SAE_PATH,
    outputs_dir=NP_OUTPUT_FOLDER,
    sparsity_threshold=-5,
    n_batches_to_sample_from=1024,
    n_prompts_to_select=1024,
    n_features_at_a_time=128,
    start_batch=0,
    end_batch=8,
    use_wandb=True,
    sae_device="cuda",
    model_device="cuda",
    model_n_devices=1,
    activation_store_device="cpu",
    dtype="bfloat16",
)

runner = NeuronpediaRunner(cfg)
runner.run()
