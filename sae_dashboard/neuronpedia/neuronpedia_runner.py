import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import wandb
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
from sae_dashboard.layout import SaeVisLayoutConfig
from sae_dashboard.neuronpedia.neuronpedia_dashboard import (
    NeuronpediaDashboardActivation,
    NeuronpediaDashboardBatch,
    NeuronpediaDashboardFeature,
)
from sae_dashboard.sae_vis_data import SaeVisConfig, SaeVisData
from sae_dashboard.sae_vis_runner import SaeVisRunner

# set TOKENIZERS_PARALLELISM to false to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

OUT_OF_RANGE_TOKEN = "<|outofrange|>"

BG_COLOR_MAP = colors.LinearSegmentedColormap.from_list(
    "bg_color_map", ["white", "darkorange"]
)

DEFAULT_SPARSITY_THRESHOLD = -6
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


class NpEncoder(json.JSONEncoder):
    def default(self, o: Any):
        if isinstance(o, NeuronpediaDashboardBatch):
            return o.to_dict()
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super(NpEncoder, self).default(o)


@dataclass
class NeuronpediaRunnerConfig:

    sae_set: str
    sae_path: str
    outputs_dir: str
    from_local_sae: bool = False
    sparsity_threshold: int = DEFAULT_SPARSITY_THRESHOLD
    huggingface_dataset_path: str = ""

    # ACTIVATION STORE PARAMETERS
    # token pars
    n_prompts_total: int = 24576
    n_tokens_in_prompt: int = 128
    n_prompts_in_forward_pass: int = 32

    # batching
    n_features_at_a_time: int = 128
    start_batch: int = 0
    end_batch: Optional[int] = None

    # quantiles
    n_quantiles: int = 5
    top_acts_group_size: int = 20
    quantile_group_size: int = 5

    dtype: str = ""

    sae_device: str | None = None
    activation_store_device: str | None = None
    model_device: str | None = None
    model_n_devices: int | None = None
    use_wandb: bool = False

    shuffle_tokens: bool = True


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
                dtype=self.cfg.dtype if self.cfg.dtype != "" else None,
            )
        else:
            self.sae, _, _ = SAE.from_pretrained(
                release=self.cfg.sae_set,
                sae_id=self.cfg.sae_path,
                device=self.cfg.sae_device or DEFAULT_FALLBACK_DEVICE,
            )
            if self.cfg.dtype != "":
                if self.cfg.dtype == "float16":
                    self.sae.to(dtype=torch.float16)
                elif self.cfg.dtype == "float32":
                    self.sae.to(dtype=torch.float32)
                elif self.cfg.dtype == "bfloat16":
                    self.sae.to(dtype=torch.bfloat16)
                else:
                    raise ValueError(
                        f"Unsupported dtype: {self.cfg.dtype}, we support float16, float32, bfloat16"
                    )

        # If we didn't override dtype, then use the SAE's dtype
        if self.cfg.dtype == "":
            print(f"Using SAE configured dtype: {self.sae.cfg.dtype}")
            self.cfg.dtype = self.sae.cfg.dtype
        else:
            print(f"Overriding dtype to {self.cfg.dtype}")
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

        print(f"DType: {self.cfg.dtype}")

        # Initialize Model
        self.model_id = self.sae.cfg.model_name
        self.layer = self.sae.cfg.hook_layer
        self.model = HookedTransformer.from_pretrained(
            model_name=self.model_id,
            device=self.cfg.model_device,
            n_devices=self.cfg.model_n_devices or 1,
            **sae_from_pretrained_kwargs,
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
            f"./cached_activations/{self.model_id}_{self.cfg.sae_set}_{self.sae.cfg.hook_name}"
        )

        # override the number of context tokens if we specified one
        # this is useful because sometimes the default context tokens is too large for us to quickly generate
        if self.cfg.n_tokens_in_prompt is not None:
            self.activations_store.context_size = self.cfg.n_tokens_in_prompt

        if not os.path.exists(cfg.outputs_dir):
            os.makedirs(cfg.outputs_dir)
        self.outputs_dir = cfg.outputs_dir

        self.vocab_dict = self.get_vocab_dict()

    def generate_tokens(
        self,
        activations_store: ActivationsStore,
        n_prompts: int = 4096 * 6,
    ):
        all_tokens_list = []
        pbar = tqdm(range(n_prompts // activations_store.store_batch_size_prompts))
        for _ in pbar:
            batch_tokens = activations_store.get_batch_tokens()
            if self.cfg.shuffle_tokens:
                batch_tokens = batch_tokens[torch.randperm(batch_tokens.shape[0])][
                    : batch_tokens.shape[0]
                ]
            all_tokens_list.append(batch_tokens)

        all_tokens = torch.cat(all_tokens_list, dim=0)
        if self.cfg.shuffle_tokens:
            all_tokens = all_tokens[torch.randperm(all_tokens.shape[0])]
        return all_tokens

    def round_list(self, to_round: list[float]):
        return list(np.round(to_round, 3))

    def ensure_list(self, input_value: list[str] | str):
        if not isinstance(input_value, list):
            return [input_value]
        return input_value

    def to_str_tokens_safe(
        self,
        vocab_dict: Dict[int, str],
        tokens: Union[int, List[int], torch.Tensor],
    ) -> list[str] | str:
        """
        does to_str_tokens, except handles out of range
        """
        assert self.model is not None
        vocab_max_index = self.model.cfg.d_vocab - 1
        # Deal with the int case separately
        if isinstance(tokens, int):
            if tokens > vocab_max_index:
                return OUT_OF_RANGE_TOKEN
            return vocab_dict[tokens]

        # If the tokens are a (possibly nested) list, turn them into a tensor
        if isinstance(tokens, list):
            tokens = torch.tensor(tokens)

        # Get flattened list of tokens
        str_tokens = [
            (vocab_dict[t] if t <= vocab_max_index else OUT_OF_RANGE_TOKEN)
            for t in tokens.flatten().tolist()
        ]

        # Reshape
        return np.reshape(str_tokens, tokens.shape).tolist()

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
        with open(f"{self.outputs_dir}/skipped_indexes.json", "w") as f:
            f.write(skipped_indexes_json)

    def get_tokens(self):

        tokens_file = f"{self.outputs_dir}/tokens_{self.cfg.n_prompts_total}.pt"
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

        wandb_cfg = self.cfg.__dict__
        wandb_cfg["sae_cfg"] = self.sae.cfg.to_dict()
        if self.cfg.use_wandb:
            wandb.init(
                project="sae-dashboard-generation",
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

        del self.activations_store

        with torch.no_grad():
            feature_batch_count = 0
            for features_to_process in tqdm(feature_idx):

                if feature_batch_count < self.cfg.start_batch:
                    feature_batch_count = feature_batch_count + 1
                    continue
                if (
                    self.cfg.end_batch is not None
                    and feature_batch_count > self.cfg.end_batch
                ):
                    feature_batch_count = feature_batch_count + 1
                    continue

                output_file = f"{self.outputs_dir}/batch-{feature_batch_count}.json"
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
                    verbose=True,
                    device=self.cfg.sae_device or DEFAULT_FALLBACK_DEVICE,
                    feature_centric_layout=layout,
                    perform_ablation_experiments=False,
                    dtype=self.cfg.dtype,
                    cache_dir=self.cached_activations_dir,
                    ignore_tokens={self.model.tokenizer.pad_token_id, self.model.tokenizer.bos_token_id, self.model.tokenizer.eos_token_id},  # type: ignore
                )

                feature_data = SaeVisRunner(feature_vis_config_gpt).run(
                    encoder=self.sae,
                    model=self.model,
                    tokens=tokens,
                )
                json_object = self.convert_feature_data_to_np_json(feature_data)
                with open(
                    output_file,
                    "w",
                ) as f:
                    f.write(json_object)
                print(f"Output written to {output_file}")

                logline = f"\n========== Completed Batch #{feature_batch_count} output: {output_file} ==========\n"
                if self.cfg.use_wandb:
                    wandb.log({"batch": feature_batch_count})

                feature_batch_count = feature_batch_count + 1

        if self.cfg.use_wandb:
            wandb.finish()

    def convert_feature_data_to_np_json(self, feature_data: SaeVisData):

        features_outputs: list[NeuronpediaDashboardFeature] = []
        for _, feat_index in enumerate(feature_data.feature_data_dict.keys()):
            feature = feature_data.feature_data_dict[feat_index]

            feature_output: NeuronpediaDashboardFeature = NeuronpediaDashboardFeature()
            feature_output.feature_index = feat_index

            top10_logits = self.round_list(feature.logits_table_data.top_logits)
            bottom10_logits = self.round_list(feature.logits_table_data.bottom_logits)

            if feature.feature_tables_data:
                feature_output.neuron_alignment_indices = (
                    feature.feature_tables_data.neuron_alignment_indices
                )
                feature_output.neuron_alignment_values = self.round_list(
                    feature.feature_tables_data.neuron_alignment_values
                )
                feature_output.neuron_alignment_l1 = self.round_list(
                    feature.feature_tables_data.neuron_alignment_l1
                )
                feature_output.correlated_neurons_indices = (
                    feature.feature_tables_data.correlated_neurons_indices
                )
                feature_output.correlated_neurons_l1 = self.round_list(
                    feature.feature_tables_data.correlated_neurons_cossim
                )
                feature_output.correlated_neurons_pearson = self.round_list(
                    feature.feature_tables_data.correlated_neurons_pearson
                )
                feature_output.correlated_features_indices = (
                    feature.feature_tables_data.correlated_features_indices
                )
                feature_output.correlated_features_l1 = self.round_list(
                    feature.feature_tables_data.correlated_features_cossim
                )
                feature_output.correlated_features_pearson = self.round_list(
                    feature.feature_tables_data.correlated_features_pearson
                )

            feature_output.neg_str = self.ensure_list(
                self.to_str_tokens_safe(
                    self.vocab_dict, feature.logits_table_data.bottom_token_ids
                )
            )
            feature_output.neg_values = bottom10_logits
            feature_output.pos_str = self.ensure_list(
                self.to_str_tokens_safe(
                    self.vocab_dict, feature.logits_table_data.top_token_ids
                )
            )
            feature_output.pos_values = top10_logits

            feature_output.frac_nonzero = (
                float(feature.acts_histogram_data.title.split(" = ")[1].split("%")[0])
                / 100
                if feature.acts_histogram_data.title is not None
                else 0
            )

            freq_hist_data = feature.acts_histogram_data
            freq_bar_values = self.round_list(freq_hist_data.bar_values)
            feature_output.freq_hist_data_bar_values = freq_bar_values
            feature_output.freq_hist_data_bar_heights = self.round_list(
                freq_hist_data.bar_heights
            )

            logits_hist_data = feature.logits_histogram_data
            feature_output.logits_hist_data_bar_heights = self.round_list(
                logits_hist_data.bar_heights
            )
            feature_output.logits_hist_data_bar_values = self.round_list(
                logits_hist_data.bar_values
            )

            # save settings so we know what we used to generate this dashboard
            feature_output.n_prompts_total = self.cfg.n_prompts_total
            feature_output.n_tokens_in_prompt = self.cfg.n_tokens_in_prompt
            feature_output.dataset = self.cfg.huggingface_dataset_path

            activations = []
            sdbs = feature.sequence_data
            for sgd in sdbs.seq_group_data:
                binMin = 0
                binMax = 0
                binContains = 0
                if "TOP ACTIVATIONS" in sgd.title:
                    binMin = -1
                    try:
                        binMax = float(sgd.title.split(" = ")[-1])
                    except ValueError:
                        print(f"Error parsing top acts: {sgd.title}")
                        binMax = 99
                    binContains = -1
                elif "INTERVAL" in sgd.title:
                    try:
                        split = sgd.title.split("<br>")
                        firstSplit = split[0].split(" ")
                        binMin = float(firstSplit[1])
                        binMax = float(firstSplit[-1])
                        secondSplit = split[1].split(" ")
                        binContains = float(secondSplit[-1].rstrip("%")) / 100
                    except ValueError:
                        print(f"Error parsing interval: {sgd.title}")
                for sd in sgd.seq_data:
                    if (
                        sd.top_token_ids is not None
                        and sd.bottom_token_ids is not None
                        and sd.top_logits is not None
                        and sd.bottom_logits is not None
                    ):
                        activation: NeuronpediaDashboardActivation = (
                            NeuronpediaDashboardActivation()
                        )
                        activation.bin_min = binMin
                        activation.bin_max = binMax
                        activation.bin_contains = binContains
                        strs = []
                        for i in range(len(sd.token_ids)):
                            strs.append(
                                self.to_str_tokens_safe(
                                    self.vocab_dict, sd.token_ids[i]
                                )
                            )
                        activation.tokens = strs
                        activation.values = self.round_list(sd.feat_acts)

                        activations.append(activation)
            feature_output.activations = activations

            features_outputs.append(feature_output)

        batch_data = NeuronpediaDashboardBatch()
        batch_data.model_id = self.model_id
        batch_data.layer = self.layer
        batch_data.sae_set = self.cfg.sae_set
        batch_data.features = features_outputs

        # no additional settings currently needed
        # settings = NeuronpediaDashboardSettings()
        # settings.n_batches_to_sample_from = self.cfg.n_prompts_total
        # settings.n_prompt_to_select = self.cfg.n_prompts_total
        # batch_data.settings = settings

        json_object = json.dumps(batch_data, cls=NpEncoder)

        return json_object
