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
from sae_lens.sae import SAE
from sae_lens.toolkit.pretrained_saes import load_sparsity
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
from sae_dashboard.neuronpedia.neuronpedia_runner_config import NeuronpediaRunnerConfig
from sae_dashboard.sae_vis_data import SaeVisConfig
from sae_dashboard.sae_vis_runner import SaeVisRunner
from sae_dashboard.utils_fns import has_duplicate_rows

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


class NeuronpediaRunner:
    def __init__(
        self,
        cfg: NeuronpediaRunnerConfig,
    ):
        self.cfg = cfg

        # Get device defaults. But if we have overrides, then use those.
        device_count = 1
        # Set correct device, use multi-GPU if we have it
        if torch.backends.mps.is_available():
            self.cfg.sae_device = self.cfg.sae_device or "mps"
            self.cfg.model_device = self.cfg.model_device or "mps"
            self.cfg.model_n_devices = self.cfg.model_n_devices or 1
            self.cfg.activation_store_device = self.cfg.activation_store_device or "mps"
        elif torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            if device_count > 1:
                self.cfg.sae_device = self.cfg.sae_device or f"cuda:{device_count - 1}"
                self.cfg.model_n_devices = self.cfg.model_n_devices or (
                    device_count - 1
                )
            else:
                self.cfg.sae_device = self.cfg.sae_device or "cuda"
            self.cfg.model_device = self.cfg.model_device or "cuda"
            self.cfg.sae_device = self.cfg.sae_device or "cuda"
            self.cfg.activation_store_device = (
                self.cfg.activation_store_device or "cuda"
            )
        else:
            self.cfg.sae_device = self.cfg.sae_device or "cpu"
            self.cfg.model_device = self.cfg.model_device or "cpu"
            self.cfg.model_n_devices = self.cfg.model_n_devices or 1
            self.cfg.activation_store_device = self.cfg.activation_store_device or "cpu"

        # Initialize SAE, defaulting to SAE dtype unless we override
        if self.cfg.from_local_sae:
            self.sae = SAE.load_from_pretrained(
                path=self.cfg.sae_path,
                device=self.cfg.sae_device or DEFAULT_FALLBACK_DEVICE,
                dtype=self.cfg.sae_dtype if self.cfg.sae_dtype != "" else None,
            )
        else:
            self.sae, _, _ = SAE.from_pretrained(
                release=self.cfg.sae_set,
                sae_id=self.cfg.sae_path,
                device=self.cfg.sae_device or DEFAULT_FALLBACK_DEVICE,
            )
            if self.cfg.sae_dtype != "":
                if self.cfg.sae_dtype == "float16":
                    self.sae.to(dtype=torch.float16)
                elif self.cfg.sae_dtype == "float32":
                    self.sae.to(dtype=torch.float32)
                elif self.cfg.sae_dtype == "bfloat16":
                    self.sae.to(dtype=torch.bfloat16)
                else:
                    raise ValueError(
                        f"Unsupported dtype: {self.cfg.sae_dtype}, we support float16, float32, bfloat16"
                    )

        # If we didn't override dtype, then use the SAE's dtype
        if self.cfg.sae_dtype == "":
            print(f"Using SAE configured dtype: {self.sae.cfg.dtype}")
            self.cfg.sae_dtype = self.sae.cfg.dtype
        else:
            print(f"Overriding sae dtype to {self.cfg.sae_dtype}")

        if self.cfg.model_dtype == "":
            self.cfg.model_dtype = "float32"

        # double sure this works
        self.sae.to(self.cfg.sae_device or DEFAULT_FALLBACK_DEVICE)
        self.sae.cfg.device = self.cfg.sae_device or DEFAULT_FALLBACK_DEVICE

        if self.cfg.huggingface_dataset_path == "":
            self.cfg.huggingface_dataset_path = self.sae.cfg.dataset_path

        print(f"Device Count: {device_count}")
        print(f"SAE Device: {self.cfg.sae_device}")
        print(f"Model Device: {self.cfg.model_device}")
        print(f"Model Num Devices: {self.cfg.model_n_devices}")
        print(f"Activation Store Device: {self.cfg.activation_store_device}")
        print(f"Dataset Path: {self.cfg.huggingface_dataset_path}")
        print(f"Forward Pass size: {self.cfg.n_tokens_in_prompt}")

        # number of tokens
        n_tokens_total = self.cfg.n_prompts_total * self.cfg.n_tokens_in_prompt
        print(f"Total number of tokens: {n_tokens_total}")
        print(f"Total number of contexts (prompts): {self.cfg.n_prompts_total}")

        # get the sae's cfg and check if it has from pretrained kwargs
        # with open(f"{self.cfg.sae_path}/cfg.json", "r") as f:
        sae_cfg_json = self.sae.cfg.to_dict()
        sae_from_pretrained_kwargs = sae_cfg_json.get("from_pretrained_kwargs", {})
        print("SAE Config on disk:")
        print(json.dumps(sae_cfg_json, indent=2))
        if sae_from_pretrained_kwargs != {}:
            print("SAE has from_pretrained_kwargs", sae_from_pretrained_kwargs)
        else:
            print(
                "SAE does not have from_pretrained_kwargs. Standard TransformerLens Loading"
            )

        self.sae.cfg.dataset_path = self.cfg.huggingface_dataset_path
        self.sae.cfg.context_size = self.cfg.n_tokens_in_prompt

        self.sae.fold_W_dec_norm()

        print(f"SAE DType: {self.cfg.sae_dtype}")
        print(f"Model DType: {self.cfg.model_dtype}")

        # Initialize Model
        self.model_id = self.sae.cfg.model_name
        self.layer = self.sae.cfg.hook_layer
        self.model = HookedTransformer.from_pretrained(
            model_name=self.model_id,
            device=self.cfg.model_device,
            n_devices=self.cfg.model_n_devices or 1,
            **sae_from_pretrained_kwargs,
            dtype=self.cfg.model_dtype,
        )

        # Initialize Activations Store
        self.activations_store = ActivationsStore.from_sae(
            model=self.model,
            sae=self.sae,
            streaming=True,
            store_batch_size_prompts=8,  # these don't matter
            n_batches_in_buffer=16,  # these don't matter
            device=self.cfg.activation_store_device or "cpu",
        )
        self.cached_activations_dir = Path(
            f"./cached_activations/{self.model_id}_{self.cfg.sae_set}_{self.sae.cfg.hook_name}_{self.sae.cfg.d_sae}width_{self.cfg.n_prompts_total}prompts"
        )

        # override the number of context tokens if we specified one
        # this is useful because sometimes the default context tokens is too large for us to quickly generate
        if self.cfg.n_tokens_in_prompt is not None:
            self.activations_store.context_size = self.cfg.n_tokens_in_prompt

        # if we have additional info, add it to the outputs subdir
        self.np_sae_id_suffix = self.cfg.np_sae_id_suffix

        if not os.path.exists(cfg.outputs_dir):
            os.makedirs(cfg.outputs_dir)
        self.cfg.outputs_dir = self.create_output_directory()

        self.vocab_dict = self.get_vocab_dict()

    def create_output_directory(self) -> str:
        """
        Creates the output directory for storing generated features.

        Returns:
            Path: The path to the created output directory.
        """
        outputs_subdir = f"{self.model_id}_{self.cfg.sae_set}_{self.sae.cfg.hook_name}_{self.sae.cfg.d_sae}"
        if self.np_sae_id_suffix is not None:
            outputs_subdir += f"_{self.np_sae_id_suffix}"
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

        for _ in pbar:
            batch_tokens = activations_store.get_batch_tokens()
            if self.cfg.shuffle_tokens:
                batch_tokens = batch_tokens[torch.randperm(batch_tokens.shape[0])]

            # Check for duplicates and only add unique sequences
            for seq in batch_tokens:
                seq_hash = self.hash_tensor(seq)
                if seq_hash not in unique_sequences:
                    unique_sequences.add(seq_hash)
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
        tokens = tokens[:, : keep_length - self.sae.cfg.prepend_bos]

        if self.cfg.prefix_tokens:
            prefix = torch.tensor(self.cfg.prefix_tokens).to(tokens.device)
            prefix_repeated = prefix.unsqueeze(0).repeat(tokens.shape[0], 1)
            # if sae.cfg.prepend_bos, then add that before the suffix
            if self.sae.cfg.prepend_bos:
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

    def get_alive_features(self) -> list[int]:
        # skip sparsity
        if self.cfg.sparsity_threshold == 1:
            print("Skipping sparsity because sparsity_threshold was set to 1")
            target_feature_indexes = list(range(self.sae.cfg.d_sae))
        else:
            # if we have feature sparsity, then use it to only generate outputs for non-dead features
            self.target_feature_indexes: list[int] = []
            sparsity = load_sparsity(self.cfg.sae_path)
            # convert sparsity to logged sparsity if it's not
            # TODO: standardize the sparsity file format
            if len(sparsity) > 0 and sparsity[0] >= 0:
                sparsity = torch.log10(sparsity + 1e-10)
            target_feature_indexes = (
                (sparsity > self.cfg.sparsity_threshold)
                .nonzero(as_tuple=True)[0]
                .tolist()
            )
        return target_feature_indexes

    def get_feature_batches(self):

        # divide into batches
        feature_idx = torch.tensor(self.target_feature_indexes)
        n_subarrays = np.ceil(len(feature_idx) / self.cfg.n_features_at_a_time).astype(
            int
        )
        feature_idx = np.array_split(feature_idx, n_subarrays)
        feature_idx = [x.tolist() for x in feature_idx]

        return feature_idx

    def record_skipped_features(self):
        # write dead into file so we can create them as dead in Neuronpedia
        skipped_indexes = set(range(self.n_features)) - set(self.target_feature_indexes)
        skipped_indexes_json = json.dumps(
            {
                "model_id": self.model_id,
                "layer": str(self.layer),
                "sae_set": self.cfg.sae_set,
                "log_sparsity": self.cfg.sparsity_threshold,
                "skipped_indexes": list(skipped_indexes),
            }
        )
        with open(f"{self.cfg.outputs_dir}/skipped_indexes.json", "w") as f:
            f.write(skipped_indexes_json)

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
        for k, v in vocab_dict.items():
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
        wandb_cfg["sae_cfg"] = self.sae.cfg.to_dict()

        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        set_name = (
            self.cfg.sae_set if self.cfg.np_set_name is None else self.cfg.np_set_name
        )
        if self.cfg.use_wandb:
            wandb.init(
                project="sae-dashboard-generation",
                name=f"{self.model_id}_{set_name}_{self.sae.cfg.hook_name}_{current_time}",
                save_code=True,
                mode="online",
                config=wandb_cfg,
            )

        self.n_features = self.sae.cfg.d_sae
        assert self.n_features is not None

        self.target_feature_indexes = self.get_alive_features()

        feature_idx = self.get_feature_batches()
        if self.cfg.start_batch >= len(feature_idx):
            print(
                f"Start batch {self.cfg.start_batch} is greater than number of batches {len(feature_idx)}, exiting"
            )
            exit()

        self.record_skipped_features()
        tokens = self.get_tokens()
        tokens = self.add_prefix_suffix_to_tokens(tokens)

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

                feature_vis_config_gpt = SaeVisConfig(
                    hook_point=self.sae.cfg.hook_name,
                    features=features_to_process,
                    minibatch_size_features=self.cfg.n_features_at_a_time,
                    minibatch_size_tokens=self.cfg.n_prompts_in_forward_pass,
                    quantile_feature_batch_size=self.cfg.quantile_feature_batch_size,
                    verbose=True,
                    device=self.cfg.sae_device or DEFAULT_FALLBACK_DEVICE,
                    feature_centric_layout=layout,
                    perform_ablation_experiments=False,
                    dtype=self.cfg.sae_dtype,
                    cache_dir=self.cached_activations_dir,
                    ignore_tokens={self.model.tokenizer.pad_token_id, self.model.tokenizer.bos_token_id, self.model.tokenizer.eos_token_id},  # type: ignore
                    ignore_positions=self.cfg.ignore_positions or [],
                    use_dfa=self.cfg.use_dfa,
                )

                feature_data = SaeVisRunner(feature_vis_config_gpt).run(
                    encoder=self.sae,
                    model=self.model,
                    tokens=tokens,
                )

                # if feature_batch_count == 0:
                #     html_save_path = (
                #         f"{self.cfg.outputs_dir}/batch-{feature_batch_count}.html"
                #     )
                #     save_feature_centric_vis(
                #         sae_vis_data=feature_data,
                #         filename=html_save_path,
                #         # use only the first 10 features for the dashboard
                #         include_only=features_to_process[
                #             : max(10, len(features_to_process))
                #         ],
                #     )

                #     if self.cfg.use_wandb:
                #         wandb.log(
                #             data={
                #                 "batch": feature_batch_count,
                #                 "dashboard": wandb.Html(open(html_save_path)),
                #             },
                #             step=feature_batch_count,
                #         )
                self.cfg.model_id = self.model_id
                self.cfg.layer = self.layer
                json_object = NeuronpediaConverter.convert_to_np_json(
                    self.model, feature_data, self.cfg, self.vocab_dict
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
                del feature_data
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        if self.cfg.use_wandb:
            wandb.sdk.finish()


def main():
    parser = argparse.ArgumentParser(description="Run Neuronpedia feature generation")
    parser.add_argument("--sae-set", required=True, help="SAE set name")
    parser.add_argument("--sae-path", required=True, help="Path to SAE")
    parser.add_argument("--np-set-name", required=True, help="Neuronpedia set name")
    parser.add_argument(
        "--np-sae-id-suffix",
        required=False,
        help="Additional suffix on Neuronpedia for the SAE ID. Goes after the SAE Set like so: __[np-sae-id-suffix]. Used for additional l0s, training steps, etc.",
    )
    parser.add_argument(
        "--dataset-path", required=True, help="HuggingFace dataset path"
    )
    parser.add_argument(
        "--sae_dtype", default="float32", help="Data type for sae computations"
    )
    parser.add_argument(
        "--model_dtype", default="float32", help="Data type for model computations"
    )
    parser.add_argument(
        "--output-dir", default="neuronpedia_outputs/", help="Output directory"
    )
    parser.add_argument(
        "--sparsity-threshold", type=int, default=1, help="Sparsity threshold"
    )
    parser.add_argument("--n-prompts", type=int, default=128, help="Number of prompts")
    parser.add_argument(
        "--n-tokens-in-prompt", type=int, default=128, help="Number of tokens in prompt"
    )
    parser.add_argument(
        "--n-prompts-in-forward-pass",
        type=int,
        default=128,
        help="Number of prompts in forward pass",
    )
    parser.add_argument(
        "--n-features-per-batch",
        type=int,
        default=2,
        help="Number of features per batch",
    )
    parser.add_argument(
        "--start-batch", type=int, default=0, help="Starting batch number"
    )
    parser.add_argument(
        "--end-batch", type=int, default=None, help="Ending batch number"
    )
    parser.add_argument(
        "--use-wandb", action="store_true", help="Use Weights & Biases for logging"
    )
    parser.add_argument(
        "--from-local-sae", action="store_true", help="Load SAE from local path"
    )

    args = parser.parse_args()

    cfg = NeuronpediaRunnerConfig(
        sae_set=args.sae_set,
        sae_path=args.sae_path,
        np_set_name=args.np_set_name,
        np_sae_id_suffix=args.np_sae_id_suffix,
        from_local_sae=args.from_local_sae,
        huggingface_dataset_path=args.dataset_path,
        sae_dtype=args.sae_dtype,
        model_dtype=args.model_dtype,
        outputs_dir=args.output_dir,
        sparsity_threshold=args.sparsity_threshold,
        n_prompts_total=args.n_prompts,
        n_tokens_in_prompt=args.n_tokens_in_prompt,
        n_prompts_in_forward_pass=args.n_prompts_in_forward_pass,
        n_features_at_a_time=args.n_features_per_batch,
        start_batch=args.start_batch,
        end_batch=args.end_batch,
        use_wandb=args.use_wandb,
    )

    runner = NeuronpediaRunner(cfg)
    runner.run()


if __name__ == "__main__":
    main()
    main()
