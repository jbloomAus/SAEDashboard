import json
import os
from typing import Any, List, Type, TypeVar

from sae_dashboard.neuronpedia.neuronpedia_dashboard import NeuronpediaDashboardBatch
from sae_dashboard.neuronpedia.neuronpedia_runner import (
    NeuronpediaRunner,
    NeuronpediaRunnerConfig,
)

# from sae_lens.toolkit.pretrained_saes import download_sae_from_hf


# depending on if device type, the results may be slightly different
CORRECT_VALUE_TOLERANCE = 0.1

T = TypeVar("T")


def json_to_class(json_file: str, cls: Type[T]) -> T:
    with open(json_file, "r") as file:
        data = json.load(file)
    return cls(**data)


# Fields to skip during comparison (these can vary between runs)
SKIP_COMPARISON_FIELDS = {
    "freq_hist_data_bar_heights",
    "logits_hist_data_bar_heights",
    "correlated_neurons_indices",
}


def compare_values_with_tolerance(
    val1: Any, val2: Any, tolerance: float = CORRECT_VALUE_TOLERANCE, path: str = ""
) -> List[str]:
    """
    Recursively compare two values with tolerance for floats.
    Returns a list of differences found.
    """
    differences = []

    # Skip certain fields that can vary between runs
    path_parts = path.split(".")
    if any(field in path_parts for field in SKIP_COMPARISON_FIELDS):
        return differences

    if isinstance(val1, dict) and isinstance(val2, dict):
        all_keys = set(val1.keys()) | set(val2.keys())
        for key in all_keys:
            # Skip fields we don't want to compare
            if key in SKIP_COMPARISON_FIELDS:
                continue
            if key not in val1:
                differences.append(f"{path}.{key}: missing in first value")
            elif key not in val2:
                differences.append(f"{path}.{key}: missing in second value")
            else:
                differences.extend(
                    compare_values_with_tolerance(
                        val1[key], val2[key], tolerance, f"{path}.{key}"
                    )
                )
    elif isinstance(val1, list) and isinstance(val2, list):
        if len(val1) != len(val2):
            differences.append(
                f"{path}: list length mismatch ({len(val1)} vs {len(val2)})"
            )
        else:
            for i, (v1, v2) in enumerate(zip(val1, val2)):
                differences.extend(
                    compare_values_with_tolerance(v1, v2, tolerance, f"{path}[{i}]")
                )
    elif isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
        if abs(val1 - val2) > tolerance:
            differences.append(f"{path}: {val1} vs {val2} (diff: {abs(val1 - val2)})")
    elif val1 != val2:
        differences.append(f"{path}: {val1!r} vs {val2!r}")

    return differences


def compare_batches_with_tolerance(
    batch1: NeuronpediaDashboardBatch,
    batch2: NeuronpediaDashboardBatch,
    tolerance: float = CORRECT_VALUE_TOLERANCE,
) -> List[str]:
    """
    Compare two NeuronpediaDashboardBatch objects with tolerance for numerical values.
    Returns a list of differences found.
    """
    dict1 = batch1.to_dict() if hasattr(batch1, "to_dict") else batch1.__dict__
    dict2 = batch2.to_dict() if hasattr(batch2, "to_dict") else batch2.__dict__
    return compare_values_with_tolerance(dict1, dict2, tolerance, "batch")


# pytest -s tests/acceptance/test_neuronpedia_runner.py::test_simple_neuronpedia_runner
def test_simple_neuronpedia_runner():
    # (_, SAE_WEIGHTS_PATH, _) = download_sae_from_hf(
    #     "jbloom/GPT2-Small-SAEs-Reformatted", "blocks.0.hook_resid_pre"
    # )

    NP_OUTPUT_FOLDER = "neuronpedia_outputs/test_simple"
    ACT_CACHE_FOLDER = "cached_activations"
    CORRECT_OUTPUTS_FOLDER = "tests/acceptance/test_simple"
    SAE_SET = "gpt2-small-res-jb"
    SAE_PATH = "blocks.0.hook_resid_pre"
    NUM_FEATURES_PER_BATCH = 2
    NUM_BATCHES = 2

    # delete output files if present
    os.system(f"rm -rf {NP_OUTPUT_FOLDER}")
    os.system(f"rm -rf {ACT_CACHE_FOLDER}")

    # # we make two batches of 2 features each
    cfg = NeuronpediaRunnerConfig(
        sae_set=SAE_SET,
        sae_path=SAE_PATH,
        np_set_name="res-jb",
        from_local_sae=False,
        outputs_dir=NP_OUTPUT_FOLDER,
        sparsity_threshold=1,
        n_prompts_total=5000,
        n_features_at_a_time=NUM_FEATURES_PER_BATCH,
        n_prompts_in_forward_pass=32,
        start_batch=0,
        end_batch=NUM_BATCHES - 1,
        use_wandb=False,
        shuffle_tokens=False,
    )

    runner = NeuronpediaRunner(cfg)
    runner.run()

    # assert the actual features/batches
    for i in range(NUM_BATCHES):
        correct_path = os.path.join(CORRECT_OUTPUTS_FOLDER, f"batch-{i}.json")

        correct_data = json_to_class(correct_path, NeuronpediaDashboardBatch)

        test_path = os.path.join(runner.cfg.outputs_dir, f"batch-{i}.json")
        assert os.path.exists(test_path), f"file {test_path} does not exist"
        test_data = json_to_class(test_path, NeuronpediaDashboardBatch)

        # Use detailed comparison
        differences = compare_batches_with_tolerance(
            test_data, correct_data, tolerance=CORRECT_VALUE_TOLERANCE
        )
        if differences:
            diff_msg = f"\nDifferences in batch-{i}.json:\n"
            for diff in differences[:50]:  # Limit to first 50
                diff_msg += f"  {diff}\n"
            if len(differences) > 50:
                diff_msg += f"  ... and {len(differences) - 50} more differences\n"
            assert False, diff_msg

    assert "run_settings.json" in os.listdir(runner.cfg.outputs_dir)


def test_simple_neuronpedia_runner_different_dtypes_sae_model():

    NP_OUTPUT_FOLDER = "neuronpedia_outputs/test_simple"
    ACT_CACHE_FOLDER = "cached_activations"
    CORRECT_OUTPUTS_FOLDER = "tests/acceptance/test_simple"
    SAE_SET = "gpt2-small-res-jb"
    SAE_PATH = "blocks.0.hook_resid_pre"
    NUM_FEATURES_PER_BATCH = 2
    NUM_BATCHES = 2

    # delete output files if present
    os.system(f"rm -rf {NP_OUTPUT_FOLDER}")
    os.system(f"rm -rf {ACT_CACHE_FOLDER}")

    # # we make two batches of 2 features each
    cfg = NeuronpediaRunnerConfig(
        sae_set=SAE_SET,
        sae_path=SAE_PATH,
        np_set_name="res-jb",
        from_local_sae=False,
        outputs_dir=NP_OUTPUT_FOLDER,
        sparsity_threshold=1,
        n_prompts_total=5000,
        n_features_at_a_time=NUM_FEATURES_PER_BATCH,
        n_prompts_in_forward_pass=32,
        start_batch=0,
        end_batch=NUM_BATCHES - 1,
        use_wandb=False,
        shuffle_tokens=False,
        model_dtype="bfloat16",
        sae_dtype="float32",
    )

    runner = NeuronpediaRunner(cfg)
    runner.run()

    # assert sparsity/skipped
    # load skipped_indexes.json file
    # skipped_path = os.path.join(NP_OUTPUT_FOLDER, "skipped_indexes.json")
    # assert os.path.exists(skipped_path), f"file {skipped_path} does not exist"
    # with open(skipped_path, "r") as file:
    #     skipped_test_data = json.load(file)
    #     # load skipped_indexes.json file from CORRECT_OUTPUTS_FOLDER
    #     skipped_correct_path = os.path.join(
    #         CORRECT_OUTPUTS_FOLDER, "skipped_indexes.json"
    #     )
    #     with open(skipped_correct_path, "r") as file:
    #         skipped_correct_data = json.load(file)
    #         assert skipped_test_data == skipped_correct_data

    # assert the actual features/batches
    for i in range(NUM_BATCHES):
        correct_path = os.path.join(CORRECT_OUTPUTS_FOLDER, f"batch-{i}.json")

        correct_data = json_to_class(correct_path, NeuronpediaDashboardBatch)

        test_path = os.path.join(runner.cfg.outputs_dir, f"batch-{i}.json")
        assert os.path.exists(test_path), f"file {test_path} does not exist"
        test_data = json_to_class(test_path, NeuronpediaDashboardBatch)

        # Use detailed comparison
        differences = compare_batches_with_tolerance(
            test_data, correct_data, tolerance=CORRECT_VALUE_TOLERANCE
        )
        if differences:
            diff_msg = f"\nDifferences in batch-{i}.json:\n"
            for diff in differences[:50]:  # Limit to first 50
                diff_msg += f"  {diff}\n"
            if len(differences) > 50:
                diff_msg += f"  ... and {len(differences) - 50} more differences\n"
            assert False, diff_msg


# pytest -s tests/benchmark/test_neuronpedia_runner.py::test_benchmark_neuronpedia_runner
def test_benchmark_neuronpedia_runner():
    NP_OUTPUT_FOLDER = "neuronpedia_outputs/benchmark"
    SAE_SET = "gpt2-small-res-jb"
    SAE_PATH = "blocks.0.hook_resid_pre"

    # delete output files if present
    os.system(f"rm -rf {NP_OUTPUT_FOLDER}")
    cfg = NeuronpediaRunnerConfig(
        sae_set=SAE_SET,
        sae_path=SAE_PATH,
        from_local_sae=False,
        outputs_dir=NP_OUTPUT_FOLDER,
        sparsity_threshold=1,
        n_prompts_total=1024,
        n_features_at_a_time=32,
        start_batch=0,
        end_batch=8,
        use_wandb=False,
        sae_device="cpu",
        model_device="cpu",
        model_n_devices=1,
        activation_store_device="cpu",
    )

    runner = NeuronpediaRunner(cfg)
    runner.run()


def test_simple_neuronpedia_runner_hook_z_sae():
    NP_OUTPUT_FOLDER = "neuronpedia_outputs/test_attn"
    ACT_CACHE_FOLDER = "cached_activations"
    SAE_SET = "gpt2-small-hook-z-kk"
    SAE_PATH = "blocks.0.hook_z"
    NUM_FEATURES_PER_BATCH = 2
    NUM_BATCHES = 2

    # delete output files if present
    os.system(f"rm -rf {NP_OUTPUT_FOLDER}")
    os.system(f"rm -rf {ACT_CACHE_FOLDER}")

    # # we make two batches of 2 features each
    cfg = NeuronpediaRunnerConfig(
        sae_set=SAE_SET,
        sae_path=SAE_PATH,
        np_set_name="att-kk",
        from_local_sae=False,
        outputs_dir=NP_OUTPUT_FOLDER,
        sparsity_threshold=1,
        n_prompts_total=5000,
        n_features_at_a_time=NUM_FEATURES_PER_BATCH,
        n_prompts_in_forward_pass=32,
        start_batch=0,
        end_batch=NUM_BATCHES - 1,
        use_wandb=False,
        shuffle_tokens=False,
    )

    runner = NeuronpediaRunner(cfg)
    runner.run()

    assert "run_settings.json" in os.listdir(runner.cfg.outputs_dir)


def test_neuronpedia_runner_prefix_suffix_it_model():
    NP_OUTPUT_FOLDER = "neuronpedia_outputs/test_masking"
    ACT_CACHE_FOLDER = "cached_activations"
    SAE_SET = "gpt2-small-res-jb"
    SAE_PATH = "blocks.0.hook_resid_pre"
    NUM_FEATURES_PER_BATCH = 2
    NUM_BATCHES = 2

    # delete output files if present
    os.system(f"rm -rf {NP_OUTPUT_FOLDER}")
    os.system(f"rm -rf {ACT_CACHE_FOLDER}")

    # # we make two batches of 2 features each
    cfg = NeuronpediaRunnerConfig(
        sae_set=SAE_SET,
        sae_path=SAE_PATH,
        np_set_name="res-jb",
        from_local_sae=False,
        outputs_dir=NP_OUTPUT_FOLDER,
        sparsity_threshold=1,
        n_prompts_total=5000,
        n_features_at_a_time=NUM_FEATURES_PER_BATCH,
        n_prompts_in_forward_pass=32,
        start_batch=0,
        end_batch=NUM_BATCHES - 1,
        use_wandb=False,
        shuffle_tokens=False,
        # prefix_tokens=[106, 1645, 108],
        # suffix_tokens=[107, 108],
        ignore_positions=[0, 1, 2],
    )

    runner = NeuronpediaRunner(cfg)
    runner.run()

    assert "run_settings.json" in os.listdir(runner.cfg.outputs_dir)


# pytest -s tests/acceptance/test_neuronpedia_runner.py::test_huggingface_neuronpedia_runner
def test_huggingface_neuronpedia_runner():
    """
    Test that HuggingFace and TransformerLens backends produce consistent outputs.

    This test runs NeuronpediaRunner twice - once with HuggingFace Transformers backend
    and once with TransformerLens backend - then compares the outputs to ensure they
    produce similar results with tolerance for numerical differences.
    """
    HF_OUTPUT_FOLDER = "neuronpedia_outputs/test_huggingface"
    TL_OUTPUT_FOLDER = "neuronpedia_outputs/test_transformerlens"
    ACT_CACHE_FOLDER = "cached_activations"
    SAE_SET = "gpt2-small-res-jb"
    SAE_PATH = "blocks.7.hook_resid_pre"
    NUM_FEATURES_PER_BATCH = 2
    NUM_BATCHES = 2

    # delete output files if present
    os.system(f"rm -rf {HF_OUTPUT_FOLDER}")
    os.system(f"rm -rf {TL_OUTPUT_FOLDER}")
    os.system(f"rm -rf {ACT_CACHE_FOLDER}")

    # Common config settings
    common_config = dict(
        sae_set=SAE_SET,
        sae_path=SAE_PATH,
        np_set_name="res-jb",
        from_local_sae=False,
        sparsity_threshold=1,
        n_prompts_total=5000,
        n_features_at_a_time=NUM_FEATURES_PER_BATCH,
        n_prompts_in_forward_pass=32,
        start_batch=0,
        end_batch=NUM_BATCHES - 1,
        use_wandb=False,
        shuffle_tokens=False,
    )

    # Run with HuggingFace backend
    print("\n=== Running with HuggingFace backend ===")
    hf_cfg = NeuronpediaRunnerConfig(
        **common_config,
        outputs_dir=HF_OUTPUT_FOLDER,
        use_huggingface=True,
    )
    hf_runner = NeuronpediaRunner(hf_cfg)
    hf_runner.run()

    # Clear cache between runs to ensure fresh activations
    os.system(f"rm -rf {ACT_CACHE_FOLDER}")

    # Run with TransformerLens backend
    print("\n=== Running with TransformerLens backend ===")
    tl_cfg = NeuronpediaRunnerConfig(
        **common_config,
        outputs_dir=TL_OUTPUT_FOLDER,
        use_huggingface=False,
    )
    tl_runner = NeuronpediaRunner(tl_cfg)
    tl_runner.run()

    # Compare outputs from both backends
    print("\n=== Comparing HuggingFace vs TransformerLens outputs ===")
    for i in range(NUM_BATCHES):
        # Get first subdirectory in each output folder
        hf_subdir = next(
            (
                d
                for d in os.listdir(HF_OUTPUT_FOLDER)
                if os.path.isdir(os.path.join(HF_OUTPUT_FOLDER, d))
            ),
            None,
        )
        tl_subdir = next(
            (
                d
                for d in os.listdir(TL_OUTPUT_FOLDER)
                if os.path.isdir(os.path.join(TL_OUTPUT_FOLDER, d))
            ),
            None,
        )

        assert hf_subdir is not None, f"No subdirectory found in {HF_OUTPUT_FOLDER}"
        assert tl_subdir is not None, f"No subdirectory found in {TL_OUTPUT_FOLDER}"

        hf_path = os.path.join(HF_OUTPUT_FOLDER, hf_subdir, f"batch-{i}.json")
        tl_path = os.path.join(TL_OUTPUT_FOLDER, tl_subdir, f"batch-{i}.json")

        assert os.path.exists(hf_path), f"HuggingFace output {hf_path} does not exist"
        assert os.path.exists(
            tl_path
        ), f"TransformerLens output {tl_path} does not exist"

        hf_data = json_to_class(hf_path, NeuronpediaDashboardBatch)
        tl_data = json_to_class(tl_path, NeuronpediaDashboardBatch)

        # Compare with tolerance for numerical differences
        differences = compare_batches_with_tolerance(
            hf_data, tl_data, tolerance=CORRECT_VALUE_TOLERANCE
        )

        # Print differences for debugging
        if differences:
            print(
                f"\nDifferences found in batch-{i}.json (HuggingFace vs TransformerLens):"
            )
            for diff in differences:  # Limit output
                print(f"  {diff}")

        # Check for structural differences (missing keys or list length mismatches)
        structural_differences = [
            d for d in differences if "missing" in d or "list length mismatch" in d
        ]

        assert (
            len(structural_differences) == 0
        ), f"Structural differences found between HuggingFace and TransformerLens: {structural_differences}"

        # Also assert on numerical differences - they should be within tolerance
        assert (
            len(differences) == 0
        ), f"Numerical differences found between HuggingFace and TransformerLens outputs in batch-{i}: {len(differences)} differences"

    assert "run_settings.json" in os.listdir(hf_runner.cfg.outputs_dir)
    assert "run_settings.json" in os.listdir(tl_runner.cfg.outputs_dir)

    print("\n=== HuggingFace and TransformerLens outputs match! ===")
