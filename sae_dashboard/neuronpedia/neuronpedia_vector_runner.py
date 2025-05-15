import argparse
import gc
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Set, Tuple

import numpy as np
import torch
import wandb
import wandb.sdk
from matplotlib import colors
from sae_lens.training.activations_store import ActivationsStore
from tqdm import tqdm
from transformer_lens import HookedTransformer

from sae_dashboard.components_config import (
    ActsHistogramConfig,
    Column,
    FeatureTablesConfig,
    LogitsHistogramConfig,
    LogitsTableConfig,
    SequencesConfig,
)

# from sae_dashboard.data_writing_fns import save_feature_centric_vis
from sae_dashboard.layout import SaeVisLayoutConfig
from sae_dashboard.neuronpedia.neuronpedia_converter import NeuronpediaConverter
from sae_dashboard.neuronpedia.neuronpedia_runner_config import (
    NeuronpediaVectorRunnerConfig,
)
from sae_dashboard.neuronpedia.vector_set import VectorSet
from sae_dashboard.utils_fns import has_duplicate_rows
from sae_dashboard.vector_vis_data import VectorVisConfig
from sae_dashboard.vector_vis_runner import VectorVisRunner

# set TOKENIZERS_PARALLELISM to false to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
RUN_SETTINGS_FILE = "run_settings.json"
OUT_OF_RANGE_TOKEN = "<|outofrange|>"

BG_COLOR_MAP = colors.LinearSegmentedColormap.from_list(
    "bg_color_map", ["white", "darkorange"]
)


DEFAULT_FALLBACK_DEVICE = "cpu"

# TODO: add more anomalies here
HTML_ANOMALIES = {
    "âĢĶ": "—",
    "âĢĵ": "–",
    "âĢľ": "“",
    "âĢĿ": "”",
    "âĢĺ": "‘",
    "âĢĻ": "’",
    "âĢĭ": " ",  # TODO: this is actually zero width space
    "Ġ": " ",
    "Ċ": "\n",
    "ĉ": "\t",
}


class NeuronpediaVectorRunner:
    def __init__(
        self,
        vector_set: VectorSet,
        cfg: NeuronpediaVectorRunnerConfig,
    ):
        self.cfg = cfg

        self._setup_devices()

        # Load vectors and create VectorSet object
        self.vector_set = vector_set

        # Convert vectors to specified dtype if provided
        if self.cfg.vector_dtype != "":
            if self.cfg.vector_dtype == "float16":
                self.vector_set.vectors = self.vector_set.vectors.to(
                    dtype=torch.float16
                )
            elif self.cfg.vector_dtype == "float32":
                self.vector_set.vectors = self.vector_set.vectors.to(
                    dtype=torch.float32
                )
            elif self.cfg.vector_dtype == "bfloat16":
                self.vector_set.vectors = self.vector_set.vectors.to(
                    dtype=torch.bfloat16
                )
            else:
                raise ValueError(
                    f"Unsupported dtype: {self.cfg.vector_dtype}, we support float16, float32, bfloat16"
                )

        # If we didn't override dtype, then use float32 as default for vectors
        if self.cfg.vector_dtype == "":
            print("Using default vector dtype: float32")
            self.cfg.vector_dtype = "float32"
        else:
            print(f"Using specified vector dtype: {self.cfg.vector_dtype}")

        if self.cfg.model_dtype == "":
            self.cfg.model_dtype = "float32"

        print(f"SAE Device: {self.cfg.vector_device}")
        print(f"Model Device: {self.cfg.model_device}")
        print(f"Model Num Devices: {self.cfg.model_n_devices}")
        print(f"Activation Store Device: {self.cfg.activation_store_device}")
        print(f"Dataset Path: {self.cfg.huggingface_dataset_path}")
        print(f"Forward Pass size: {self.cfg.n_tokens_in_prompt}")

        # number of tokens
        n_tokens_total = self.cfg.n_prompts_total * self.cfg.n_tokens_in_prompt
        print(f"Total number of tokens: {n_tokens_total}")
        print(f"Total number of contexts (prompts): {self.cfg.n_prompts_total}")

        # TODO: figure out the best way to handle this for VectorSets
        # get the sae's cfg and check if it has from pretrained kwargs
        # with open(f"{self.cfg.sae_path}/cfg.json", "r") as f:
        # sae_cfg_json = self.vector_set.cfg.to_dict()
        # sae_from_pretrained_kwargs = sae_cfg_json.get("from_pretrained_kwargs", {})
        # print("SAE Config on disk:")
        # print(json.dumps(sae_cfg_json, indent=2))
        # if sae_from_pretrained_kwargs != {}:
        #     print("SAE has from_pretrained_kwargs", sae_from_pretrained_kwargs)
        # else:
        #     print(
        #         "SAE does not have from_pretrained_kwargs. Standard TransformerLens Loading"
        #     )

        # self.sae.cfg.dataset_path = self.cfg.huggingface_dataset_path
        # self.sae.cfg.context_size = self.cfg.n_tokens_in_prompt

        print(f"Vector DType: {self.cfg.vector_dtype}")
        print(f"Model DType: {self.cfg.model_dtype}")

        # Initialize Model
        self.model_id = self.vector_set.cfg.model_name
        self.cfg.model_id = self.model_id
        self.layer = self.vector_set.cfg.hook_layer
        self.cfg.layer = self.layer
        self.model = HookedTransformer.from_pretrained(
            model_name=self.model_id,
            device=self.cfg.model_device,
            n_devices=self.cfg.model_n_devices or 1,
            **self.vector_set.cfg.model_from_pretrained_kwargs,
            dtype=self.cfg.model_dtype,
        )

        # Initialize Activations Store
        # defaults here are copied from the SAE code path
        self.activations_store = ActivationsStore(
            model=self.model,
            dataset=self.cfg.huggingface_dataset_path,
            streaming=True,
            hook_name=self.vector_set.hook_point,
            hook_layer=self.vector_set.hook_layer,
            hook_head_index=self.vector_set.hook_head_index,
            context_size=self.cfg.n_tokens_in_prompt,
            d_in=self.vector_set.cfg.d_in,
            n_batches_in_buffer=16,  # apparently this doesn't matter
            total_training_tokens=10**9,
            store_batch_size_prompts=8,  # apparently this doesn't matter either
            train_batch_size_tokens=4096,
            prepend_bos=True,
            normalize_activations="none",
            device=torch.device(self.cfg.activation_store_device or "cpu"),
            dtype=self.cfg.vector_dtype,
        )
        self.cached_activations_dir = Path(
            f"./cached_activations/{self.model_id}_{self.vector_set.cfg.hook_name}_{self.cfg.n_prompts_total}prompts"
        )

        # override the number of context tokens if we specified one
        # this is useful because sometimes the default context tokens is too large for us to quickly generate
        if self.cfg.n_tokens_in_prompt is not None:
            self.activations_store.context_size = self.cfg.n_tokens_in_prompt

        if not os.path.exists(cfg.outputs_dir):
            os.makedirs(cfg.outputs_dir)
        self.cfg.outputs_dir = self.create_output_directory()

        self.vocab_dict = self.get_vocab_dict()

    def _setup_devices(self):
        """Device setup for vector runner"""

        # Get device defaults. But if we have overrides, then use those.
        device_count = 1

        # Set correct device, use multi-GPU if we have it
        if torch.backends.mps.is_available():
            self.cfg.vector_device = self.cfg.vector_device or "mps"
            self.cfg.model_device = self.cfg.model_device or "mps"
            self.cfg.model_n_devices = self.cfg.model_n_devices or 1
            self.cfg.activation_store_device = self.cfg.activation_store_device or "mps"
        elif torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            if device_count > 1:
                self.cfg.vector_device = (
                    self.cfg.vector_device or f"cuda:{device_count - 1}"
                )
                self.cfg.model_n_devices = self.cfg.model_n_devices or (
                    device_count - 1
                )
            else:
                self.cfg.vector_device = self.cfg.vector_device or "cuda"
            self.cfg.model_device = self.cfg.model_device or "cuda"
            self.cfg.vector_device = self.cfg.vector_device or "cuda"
            self.cfg.activation_store_device = (
                self.cfg.activation_store_device or "cuda"
            )
        else:
            self.cfg.vector_device = self.cfg.vector_device or "cpu"
            self.cfg.model_device = self.cfg.model_device or "cpu"
            self.cfg.model_n_devices = self.cfg.model_n_devices or 1
            self.cfg.activation_store_device = self.cfg.activation_store_device or "cpu"

        print(f"Device Count: {device_count}")

    def create_output_directory(self) -> str:
        """
        Creates the output directory for storing generated features.

        Returns:
            Path: The path to the created output directory.
        """
        outputs_subdir = f"{self.model_id}_{self.vector_set.cfg.hook_name}"
        outputs_dir = Path(self.cfg.outputs_dir).joinpath(outputs_subdir)
        if outputs_dir.exists() and outputs_dir.is_file():
            raise ValueError(
                f"Error: Output directory {outputs_dir.as_posix()} exists and is a file."
            )
        outputs_dir.mkdir(parents=True, exist_ok=True)
        return str(outputs_dir)

    def hash_tensor(self, tensor: torch.Tensor) -> Tuple[int, ...]:
        return tuple(tensor.cpu().numpy().flatten().tolist())

    def generate_tokens(
        self,
        activations_store: ActivationsStore,
        n_prompts: int = 4096 * 6,
    ) -> torch.Tensor:
        all_tokens_list = []
        unique_sequences: Set[Tuple[int, ...]] = set()
        pbar = tqdm(range(n_prompts // activations_store.store_batch_size_prompts))

        prepend_tokens = None
        if (
            self.cfg.prepend_chat_template_text is not None
            and self.model.tokenizer is not None
        ):
            print(
                f"Prepending chat template text: {self.cfg.prepend_chat_template_text}"
            )
            prepend_tokens = torch.tensor(
                self.model.tokenizer(
                    self.cfg.prepend_chat_template_text, add_special_tokens=False
                ).input_ids
            ).to(activations_store.device)
            self.prepend_tokens_length = prepend_tokens.shape[0]

        for _ in pbar:
            batch_tokens = activations_store.get_batch_tokens()
            if self.cfg.shuffle_tokens:
                batch_tokens = batch_tokens[torch.randperm(batch_tokens.shape[0])]

            # Check for duplicates and only add unique sequences
            for seq in batch_tokens:
                seq_hash = self.hash_tensor(seq)
                if seq_hash not in unique_sequences:
                    unique_sequences.add(seq_hash)
                    # if we prepend chat template text, then we need to remove the BOS token from the sequence
                    if prepend_tokens is not None:
                        all_tokens_list.append(
                            torch.cat(
                                [prepend_tokens.unsqueeze(0), seq[1:].unsqueeze(0)],
                                dim=1,
                            )[
                                :, : activations_store.context_size
                            ]  # trim it back to the context size
                        )
                    else:
                        all_tokens_list.append(seq.unsqueeze(0))

            # Early exit if we've collected enough unique sequences
            if len(all_tokens_list) >= n_prompts:
                break

        all_tokens = torch.cat(all_tokens_list, dim=0)[:n_prompts]
        if self.cfg.shuffle_tokens:
            all_tokens = all_tokens[torch.randperm(all_tokens.shape[0])]

        return all_tokens

    def add_prefix_suffix_to_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        original_length = tokens.shape[1]
        bos_tokens = tokens[:, 0]  # might not be if sae.cfg.prepend_bos is False
        prefix_length = len(self.cfg.prefix_tokens) if self.cfg.prefix_tokens else 0
        suffix_length = len(self.cfg.suffix_tokens) if self.cfg.suffix_tokens else 0

        # return tokens if no prefix or suffix
        if self.cfg.prefix_tokens is None and self.cfg.suffix_tokens is None:
            return tokens

        # Calculate how many tokens to keep from the original
        keep_length = original_length - prefix_length - suffix_length

        if keep_length <= 0:
            raise ValueError("Prefix and suffix are too long for the given tokens.")

        # Trim original tokens
        tokens = tokens[:, : keep_length - self.vector_set.cfg.prepend_bos]

        if self.cfg.prefix_tokens:
            prefix = torch.tensor(self.cfg.prefix_tokens).to(tokens.device)
            prefix_repeated = prefix.unsqueeze(0).repeat(tokens.shape[0], 1)
            # if sae.cfg.prepend_bos, then add that before the suffix
            if self.vector_set.cfg.prepend_bos:
                bos = bos_tokens.unsqueeze(1)
                prefix_repeated = torch.cat([bos, prefix_repeated], dim=1)
            tokens = torch.cat([prefix_repeated, tokens], dim=1)

        if self.cfg.suffix_tokens:
            suffix = torch.tensor(self.cfg.suffix_tokens).to(tokens.device)
            suffix_repeated = suffix.unsqueeze(0).repeat(tokens.shape[0], 1)
            tokens = torch.cat([tokens, suffix_repeated], dim=1)

        # assert length hasn't changed
        assert tokens.shape[1] == original_length
        return tokens

    def get_feature_batches(self):
        # Simplified batching for vectors
        n_vectors = self.vector_set.cfg.d_vectors
        vector_indices = list(range(n_vectors))

        # Divide into batches
        n_subarrays = np.ceil(n_vectors / self.cfg.n_vectors_at_a_time).astype(int)
        batches = np.array_split(vector_indices, n_subarrays)
        return [x.tolist() for x in batches]

    def get_tokens(self):
        tokens_file = f"{self.cfg.outputs_dir}/tokens_{self.cfg.n_prompts_total}.pt"
        if os.path.isfile(tokens_file):
            print("Tokens exist, loading them.")
            tokens = torch.load(tokens_file)
        else:
            print("Tokens don't exist, making them.")
            tokens = self.generate_tokens(
                self.activations_store,
                self.cfg.n_prompts_total,
            )
            torch.save(
                tokens,
                tokens_file,
            )

        assert not has_duplicate_rows(tokens), "Duplicate rows in tokens"

        return tokens

    def get_vocab_dict(self) -> Dict[int, str]:
        # get vocab
        vocab_dict = self.model.tokenizer.vocab  # type: ignore
        new_vocab_dict = {}
        # Replace substrings in the keys of vocab_dict using HTML_ANOMALIES
        for k, v in vocab_dict.items():  # type: ignore
            modified_key = k
            for anomaly in HTML_ANOMALIES:
                modified_key = modified_key.replace(anomaly, HTML_ANOMALIES[anomaly])
            new_vocab_dict[v] = modified_key
        vocab_dict = new_vocab_dict
        # pad with blank tokens to the actual vocab size
        for i in range(len(vocab_dict), self.model.cfg.d_vocab):
            vocab_dict[i] = OUT_OF_RANGE_TOKEN
        return vocab_dict

    # TODO: make this function simpler
    def run(self):
        run_settings_path = self.cfg.outputs_dir + "/" + RUN_SETTINGS_FILE
        run_settings = self.cfg.__dict__
        with open(run_settings_path, "w") as f:
            json.dump(run_settings, f, indent=4)

        wandb_cfg = self.cfg.__dict__
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        if self.cfg.use_wandb:
            wandb.init(
                project="vector-dashboard-generation",
                name=f"{self.model_id}_{self.vector_set.cfg.hook_name}_{current_time}",
                save_code=True,
                mode="online",
                config=wandb_cfg,
            )

        # Use number of vectors instead of features
        self.n_features = self.vector_set.cfg.d_vectors
        assert self.n_features is not None

        feature_idx = self.get_feature_batches()
        if self.cfg.start_batch >= len(feature_idx):
            print(
                f"Start batch {self.cfg.start_batch} is greater than number of batches {len(feature_idx)}, exiting"
            )
            exit()

        tokens = self.get_tokens()
        tokens = self.add_prefix_suffix_to_tokens(tokens)

        if self.activations_store is not None:
            del self.activations_store

        with torch.no_grad():
            for feature_batch_count, features_to_process in tqdm(
                enumerate(feature_idx)
            ):
                if feature_batch_count < self.cfg.start_batch:
                    feature_batch_count = feature_batch_count + 1
                    continue
                if (
                    self.cfg.end_batch is not None
                    and feature_batch_count > self.cfg.end_batch
                ):
                    feature_batch_count = feature_batch_count + 1
                    continue

                output_file = f"{self.cfg.outputs_dir}/batch-{feature_batch_count}.json"
                # if output_file exists, skip
                if os.path.isfile(output_file):
                    logline = f"\n++++++++++ Skipping Batch #{feature_batch_count} output. File exists: {output_file} ++++++++++\n"
                    print(logline)
                    continue

                print(f"========== Running Batch #{feature_batch_count} ==========")

                layout = SaeVisLayoutConfig(
                    columns=[
                        Column(
                            SequencesConfig(
                                stack_mode="stack-all",
                                buffer=None,  # type: ignore
                                compute_buffer=True,
                                n_quantiles=self.cfg.n_quantiles,
                                top_acts_group_size=self.cfg.top_acts_group_size,
                                quantile_group_size=self.cfg.quantile_group_size,
                            ),
                            ActsHistogramConfig(),
                            LogitsHistogramConfig(),
                            LogitsTableConfig(),
                            FeatureTablesConfig(n_rows=3),
                        )
                    ]
                )

                if self.prepend_tokens_length is not None:
                    if self.cfg.ignore_positions is not None:
                        self.cfg.ignore_positions.extend(
                            range(self.prepend_tokens_length)
                        )
                    else:
                        self.cfg.ignore_positions = list(
                            range(self.prepend_tokens_length)
                        )

                vector_vis_config_gpt = VectorVisConfig(
                    hook_point=self.vector_set.cfg.hook_name,
                    vector_indices=features_to_process,
                    minibatch_size_features=self.cfg.n_vectors_at_a_time,
                    minibatch_size_tokens=self.cfg.n_prompts_in_forward_pass,
                    quantile_feature_batch_size=self.cfg.quantile_vector_batch_size,
                    verbose=True,
                    device=self.cfg.vector_device or DEFAULT_FALLBACK_DEVICE,
                    feature_centric_layout=layout,
                    perform_ablation_experiments=False,
                    dtype=self.cfg.vector_dtype,
                    cache_dir=self.cached_activations_dir,
                    ignore_tokens={
                        self.model.tokenizer.pad_token_id,  # type: ignore
                        self.model.tokenizer.bos_token_id,  # type: ignore
                        self.model.tokenizer.eos_token_id,  # type: ignore
                    },  # type: ignore
                    ignore_positions=self.cfg.ignore_positions or [],
                    ignore_thresholds=self.cfg.activation_thresholds,
                    use_dfa=self.cfg.use_dfa,
                )

                vector_data = VectorVisRunner(vector_vis_config_gpt).run(
                    encoder=self.vector_set,
                    model=self.model,
                    tokens=tokens,
                )

                self.cfg.model_id = self.model_id
                self.cfg.layer = self.layer
                # add the original vectors if include_original_vectors_in_output is True
                json_object = NeuronpediaConverter.convert_to_np_json(
                    self.model,
                    vector_data,
                    self.cfg,
                    self.vocab_dict,
                    (
                        self.vector_set.vectors
                        if self.cfg.include_original_vectors_in_output
                        else None
                    ),
                )
                with open(
                    output_file,
                    "w",
                ) as f:
                    f.write(json_object)
                print(f"Output written to {output_file}")

                logline = f"\n========== Completed Batch #{feature_batch_count} output: {output_file} ==========\n"
                if self.cfg.use_wandb:
                    wandb.log(
                        {"batch": feature_batch_count},
                        step=feature_batch_count,
                    )
                # Clean up after each batch
                del vector_data
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        if self.cfg.use_wandb:
            wandb.sdk.finish()


def main():
    parser = argparse.ArgumentParser(
        description="Run Neuronpedia vector feature generation"
    )
    # Required vector loading parameters
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument(
        "--vector-names", nargs="+", help="Optional names for each vector"
    )

    # Dataset parameters
    parser.add_argument(
        "--dataset-path", required=True, help="HuggingFace dataset path"
    )

    # Vector set parameters
    parser.add_argument(
        "--vector-set-path", required=True, help="Path to the vector set JSON file"
    )

    # Token generation parameters
    parser.add_argument(
        "--n-prompts", type=int, default=24576, help="Total number of prompts"
    )
    parser.add_argument(
        "--n-tokens-in-prompt",
        type=int,
        default=128,
        help="Number of tokens in each prompt",
    )
    parser.add_argument(
        "--n-prompts-in-forward-pass",
        type=int,
        default=32,
        help="Number of prompts to process in each forward pass",
    )
    parser.add_argument(
        "--no-prepend-bos",
        action="store_false",
        dest="prepend_bos",
        help="Don't prepend BOS token to sequences",
    )

    # Batching parameters
    parser.add_argument(
        "--n-vectors-at-a-time",
        type=int,
        default=128,
        help="Number of vectors to process at once",
    )
    parser.add_argument(
        "--quantile-vector-batch-size",
        type=int,
        default=64,
        help="Batch size for quantile calculations",
    )
    parser.add_argument(
        "--start-batch", type=int, default=0, help="Starting batch number"
    )
    parser.add_argument(
        "--end-batch", type=int, default=None, help="Ending batch number"
    )

    # Device and dtype settings
    parser.add_argument(
        "--model-dtype", default="", help="Data type for model computations"
    )
    parser.add_argument(
        "--vector-dtype", default="", help="Data type for vector computations"
    )
    parser.add_argument("--activation-store-device", help="Device for activation store")
    parser.add_argument("--model-device", help="Device for model")
    parser.add_argument("--vector-device", help="Device for vector operations")
    parser.add_argument(
        "--model-n-devices", type=int, help="Number of devices for model"
    )

    # Quantile parameters
    parser.add_argument(
        "--n-quantiles", type=int, default=5, help="Number of quantiles"
    )
    parser.add_argument(
        "--top-acts-group-size",
        type=int,
        default=30,
        help="Group size for top activations",
    )
    parser.add_argument(
        "--quantile-group-size", type=int, default=5, help="Group size for quantiles"
    )

    # Additional settings
    parser.add_argument("--use-dfa", action="store_true", help="Use DFA calculations")
    parser.add_argument(
        "--use-wandb", action="store_true", help="Use Weights & Biases for logging"
    )
    parser.add_argument(
        "--no-shuffle-tokens",
        action="store_false",
        dest="shuffle_tokens",
        help="Don't shuffle tokens",
    )
    parser.add_argument(
        "--prefix-tokens", type=int, nargs="+", help="Tokens to prefix to each sequence"
    )
    parser.add_argument(
        "--suffix-tokens", type=int, nargs="+", help="Tokens to append to each sequence"
    )
    parser.add_argument(
        "--ignore-positions",
        type=int,
        nargs="+",
        help="Positions to ignore in sequences",
    )

    args = parser.parse_args()

    cfg = NeuronpediaVectorRunnerConfig(
        outputs_dir=args.output_dir,
        vector_names=args.vector_names,
        n_prompts_total=args.n_prompts,
        n_tokens_in_prompt=args.n_tokens_in_prompt,
        n_prompts_in_forward_pass=args.n_prompts_in_forward_pass,
        prepend_bos=args.prepend_bos,
        n_vectors_at_a_time=args.n_vectors_at_a_time,
        quantile_vector_batch_size=args.quantile_vector_batch_size,
        start_batch=args.start_batch,
        end_batch=args.end_batch,
        use_dfa=args.use_dfa,
        n_quantiles=args.n_quantiles,
        top_acts_group_size=args.top_acts_group_size,
        quantile_group_size=args.quantile_group_size,
        model_dtype=args.model_dtype,
        vector_dtype=args.vector_dtype,
        activation_store_device=args.activation_store_device,
        model_device=args.model_device,
        vector_device=args.vector_device,
        model_n_devices=args.model_n_devices,
        huggingface_dataset_path=args.dataset_path,
        use_wandb=args.use_wandb,
        shuffle_tokens=args.shuffle_tokens,
        prefix_tokens=args.prefix_tokens,
        suffix_tokens=args.suffix_tokens,
        ignore_positions=args.ignore_positions,
    )

    # Note: You'll need to pass a VectorSet instance here
    vector_set = VectorSet.load(args.vector_set_path)
    runner = NeuronpediaVectorRunner(vector_set=vector_set, cfg=cfg)
    runner.run()


if __name__ == "__main__":
    main()
    main()
