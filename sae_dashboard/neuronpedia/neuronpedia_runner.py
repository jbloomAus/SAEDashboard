import argparse
import ctypes
import gc
import importlib
import inspect
import json
import os
import shutil
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Set, Tuple, cast

import numpy as np
import torch
import wandb
import wandb.sdk
from datasets import Dataset, IterableDataset
from matplotlib import colors
from sae_lens import SAE, ActivationsStore

from sae_dashboard.tl_compat import hooked_sae_transformer_cls

# `move_to_model_device` is an opt-in seam added in SAELens (see the coordinated Scalable Dashboards
# PR set). It lets this runner stage prompt batches once, on the activations-store device, instead of
# paying a transfer to the model device that `generate_tokens` immediately undoes. It is a pure
# optimization: `generate_tokens` returns `.cpu()` tensors either way.
#
# Released SAELens does not have the kwarg, so passing it unconditionally raises TypeError on every
# batch and makes this runner unusable against a released SAELens. Detect it once here rather than
# requiring an unreleased SAELens just to run the suite.
_ACTIVATIONS_STORE_SUPPORTS_DEVICE_SEAM = (
    "move_to_model_device"
    in inspect.signature(ActivationsStore.get_batch_tokens).parameters
)
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from sae_dashboard.components_config import (
    ActsHistogramConfig,
    Column,
    FeatureTablesConfig,
    LogitsHistogramConfig,
    LogitsTableConfig,
    SequencesConfig,
)

# from sae_dashboard.data_writing_fns import save_feature_centric_vis
from sae_dashboard.hook_utils import convert_model_name_tl_to_hf
from sae_dashboard.layout import SaeVisLayoutConfig
from sae_dashboard.neuronpedia.legacy import runner as legacy_runner
from sae_dashboard.neuronpedia.neuronpedia_converter import NeuronpediaConverter
from sae_dashboard.neuronpedia.neuronpedia_export import (
    NeuronpediaExportConfig,
    derive_hook_point_from_hook_name,
    export_neuronpedia_dashboards,
    resolve_creator_id,
)
from sae_dashboard.neuronpedia.neuronpedia_runner_config import (
    DEFAULT_PARQUET_ROW_GROUP_SIZE,
    DEFAULT_PROMPT_BATCH_SIZE_ROUND_TO,
    DEFAULT_PROMPT_BUCKET_SCALE_LIMIT,
    DEFAULT_PROMPT_PRIMARY_ACTS_SCALE_LIMIT,
    NeuronpediaRunnerConfig,
    is_legacy_dashboard_path,
    warn_if_deprecated_legacy_dashboard_path,
)
from sae_dashboard.neuronpedia.prompt_bucketing import derive_prompt_bucket_ceilings
from sae_dashboard.neuronpedia.prompt_datasets import (
    PromptDatasetConfig,
    PromptDatasetMaterialization,
    load_prompt_dataset,
    resolve_prompt_dataset,
)
from sae_dashboard.perf_logging import (
    cpu_snapshot,
    elapsed_timer,
    io_delta,
    log_perf_event,
    process_io_snapshot,
    timed_stage,
)
from sae_dashboard.sae_vis_data import SaeVisColumnarData, SaeVisConfig
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
DEFAULT_BRIDGE_COMPATIBILITY_KWARGS = {"no_processing": True}

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

_LIBC: ctypes.CDLL | None = None
try:
    _LIBC = ctypes.CDLL("libc.so.6")
except OSError:
    pass


def get_sae_loader(loader_name: str):
    """Resolve a sae_lens HuggingFace SAE loader by name.

    Accepts either:
      - A short registry name from sae_lens' ``NAMED_PRETRAINED_SAE_LOADERS``,
        e.g. ``"dictionary_learning_1"``, ``"gemma_2"``, ``"sparsify"``.
      - A full loader function name exported by
        ``sae_lens.loading.pretrained_sae_loaders``, e.g.
        ``"dictionary_learning_sae_huggingface_loader_1"``.
    """
    module = importlib.import_module("sae_lens.loading.pretrained_sae_loaders")

    # Prefer the short registry name if available.
    named_loaders = getattr(module, "NAMED_PRETRAINED_SAE_LOADERS", None)
    if named_loaders is not None and loader_name in named_loaders:
        return named_loaders[loader_name]

    # Fall back to looking up a function by its full module-level name for
    # backwards compatibility with earlier usage.
    if hasattr(module, loader_name):
        return getattr(module, loader_name)

    available_short = sorted(named_loaders.keys()) if named_loaders else []
    raise ValueError(
        f"Unknown SAE loader '{loader_name}'. "
        f"Expected either a short registry name (e.g. one of {available_short}) "
        f"or the full name of a loader function exported by "
        f"sae_lens.loading.pretrained_sae_loaders."
    )


def resolve_capture_hook_name(
    metadata: Any, use_huggingface: bool, capture_hook_name: str | None = None
) -> str:
    """The hook the runner should capture at.

    ``capture_hook_name`` wins over everything when set: it is how an operator names the capture
    location explicitly, which is the only way to reach a tensor whose TransformerLens name the SAE
    does not declare. Gemma Scope 2 transcoders are the motivating case: they are trained on the
    block norm's output, which a TransformerBridge names ``blocks.{i}.ln2.hook_out`` (equivalently
    ``blocks.{i}.mlp.hook_in``), while their metadata declares ``blocks.{i}.hook_mlp_in``, the
    residual *before* that norm.

    Deliberately an explicit setting rather than a translation from ``hf_hook_name``: deriving a
    TransformerLens name from a module path would mean carrying yet another architecture-keyed
    mapping table, and divergence between such tables is the defect this setting exists to work
    around.

    ``hook_name`` is a TransformerLens *convention*, and for some releases it names a different tensor
    from the one the SAE was trained on: a Gemma Scope 2 transcoder declares
    ``pre_feedforward_layernorm.output``, the norm's output, while TransformerLens fires
    ``hook_mlp_in`` on the residual *before* that norm. ``hf_hook_name`` is carried through from the
    artifact's own config, so where it exists it is authoritative and needs no name translation.

    Additive by construction: only consulted on the HuggingFace path, and only when the loader
    published the field, so releases that publish nothing keep the TransformerLens name.
    """

    def _get(key: str) -> Any:
        if hasattr(metadata, key):
            return getattr(metadata, key)
        if isinstance(metadata, dict):
            return metadata.get(key)
        return None

    hook_name = _get("hook_name")
    if capture_hook_name:
        if capture_hook_name != hook_name:
            print(
                f"Capturing at {capture_hook_name!r} (explicitly configured) rather than the SAE's "
                f"declared hook name {hook_name!r}. The Neuronpedia source record still labels this "
                f"source with its configured `hook_point`, whose vocabulary cannot express every "
                f"capture location, so the label and the capture location may read differently."
            )
        return capture_hook_name
    if not use_huggingface:
        return hook_name
    hf_hook_name = _get("hf_hook_name")
    if not hf_hook_name:
        return hook_name
    if hf_hook_name != hook_name:
        print(
            f"Capturing at the SAE's declared module path {hf_hook_name!r} rather than its "
            f"TransformerLens hook name {hook_name!r} (--huggingface)."
        )
    return hf_hook_name


class NeuronpediaRunner:
    def __init__(
        self,
        cfg: NeuronpediaRunnerConfig,
    ):
        self.cfg = cfg
        self._columnar_write_executor: ThreadPoolExecutor | None = None
        self._pending_columnar_writes: deque[tuple[int, int, Future[Any]]] = deque()
        self._last_sae_vis_runner: Any | None = None
        warn_if_deprecated_legacy_dashboard_path(cfg)

        # Fail fast if Neuronpedia export was requested but required metadata
        # is missing — better to surface this before spending hours generating
        # batches.
        if cfg.output_neuronpedia_exports:
            self._validate_neuronpedia_export_cfg()

        # Initialize core components
        self.device_count = self._setup_devices()
        self._load_sae_or_transcoder()
        if cfg.prepend_bos is not None:
            # if metadata doesnt exist, create it
            if not hasattr(self.sae.cfg, "metadata"):
                self.sae.cfg["metadata"] = {}  # type: ignore
            self.sae.cfg.metadata["prepend_bos"] = cfg.prepend_bos  # type: ignore
        self._configure_dtypes()
        self._extract_model_info()
        self._initialize_model()
        self._setup_activation_store()
        self._setup_output_directory()
        self.vocab_dict = self.get_vocab_dict()

    def _setup_devices(self) -> int:
        """Set up device configuration based on available hardware."""
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

        return device_count

    def _release_unused_host_memory(self) -> None:
        try:
            pyarrow = importlib.import_module("pyarrow")
        except ImportError:
            pyarrow = None

        if pyarrow is not None:
            release_unused = getattr(
                pyarrow.default_memory_pool(), "release_unused", None
            )
            if callable(release_unused):
                release_unused()

        malloc_trim = getattr(_LIBC, "malloc_trim", None) if _LIBC is not None else None
        if callable(malloc_trim):
            malloc_trim(0)

    def _load_sae_or_transcoder(self):
        """Load SAE, Transcoder, SkipTranscoder, or CLT based on configuration."""
        # Validate that only one loader type is specified
        flags = [
            self.cfg.use_transcoder,
            self.cfg.use_skip_transcoder,
            self.cfg.use_clt,
        ]
        if sum(flags) > 1:
            raise ValueError(
                "Only one of --use-transcoder, --use-skip-transcoder, or --use-clt can be set."
            )
        if self.cfg.use_clt and self.cfg.clt_layer_idx is None:
            raise ValueError("--clt-layer-idx must be specified when using --use-clt.")

        if self.cfg.use_skip_transcoder:
            # Dynamically import to avoid dependency issues when Transcoder isn't used
            try:
                from sae_lens import SkipTranscoder  # type: ignore
            except ImportError as e:
                raise ImportError(
                    "SkipTranscoder class not found in sae_lens. Install a version of sae_lens that provides it "
                    "or disable --use-skip-transcoder."
                ) from e
            LoaderClass = SkipTranscoder
            loader_kwargs = {}
            # TODO: Check if SkipTranscoder supports local loading via path= kwarg
            # if self.cfg.from_local_sae:
            #     loader_kwargs["path"] = self.cfg.sae_path
            # else:
            loader_kwargs["release"] = self.cfg.sae_set
            loader_kwargs["sae_id"] = self.cfg.sae_path

            self.sae = LoaderClass.from_pretrained(  # type: ignore
                device=self.cfg.sae_device or DEFAULT_FALLBACK_DEVICE, **loader_kwargs
            )
            # SkipTranscoder doesn't directly support dtype override in from_pretrained, apply after
            if self.cfg.sae_dtype:
                self._apply_sae_dtype_override()

        elif self.cfg.use_transcoder:
            # Dynamically import to avoid dependency issues when Transcoder isn't used
            try:
                from sae_lens import Transcoder  # type: ignore
            except ImportError as e:
                raise ImportError(
                    "Transcoder class not found in sae_lens. Install a version of sae_lens that provides "
                    "Transcoder or disable --use-transcoder."
                ) from e
            LoaderClass = Transcoder

            if self.cfg.from_local_sae:
                # Transcoder might not have load_from_pretrained, use from_pretrained
                self.sae = LoaderClass.from_pretrained(  # type: ignore
                    path=self.cfg.sae_path,  # type: ignore
                    device=self.cfg.sae_device or DEFAULT_FALLBACK_DEVICE,
                    # dtype=self.cfg.sae_dtype if self.cfg.sae_dtype != "" else None, # Dtype applied after
                )
            else:
                self.sae = LoaderClass.from_pretrained(  # type: ignore
                    release=self.cfg.sae_set,
                    sae_id=self.cfg.sae_path,
                    device=self.cfg.sae_device or DEFAULT_FALLBACK_DEVICE,
                    converter=(
                        get_sae_loader(self.cfg.sae_converter_name)
                        if self.cfg.sae_converter_name
                        else None
                    ),
                )
            # Apply dtype override after loading for Transcoder as well
            if self.cfg.sae_dtype:
                self._apply_sae_dtype_override()
        elif self.cfg.use_clt:
            # Dynamically import CLT components only when needed
            if self.cfg.from_local_sae:
                clt_dir = Path(self.cfg.sae_path)
                clt_config_path = clt_dir / "cfg.json"
                if clt_config_path.is_file():
                    try:
                        from clt.config.clt_config import CLTConfig  # type: ignore
                        from clt.models.clt import CrossLayerTranscoder  # type: ignore
                    except ImportError as e:
                        raise ImportError(
                            "CLT components (CrossLayerTranscoder, CLTConfig) not found. "
                            "Ensure the 'clt' package is installed and available."
                        ) from e

                    try:
                        clt_cfg = CLTConfig.from_json(clt_config_path)  # type: ignore
                    except Exception as e:
                        raise ValueError(
                            f"Failed to load CLT config from {clt_config_path}: {e}"  # type: ignore
                        ) from e

                    self.clt = CrossLayerTranscoder(
                        config=clt_cfg,
                        process_group=None,
                        device=self.cfg.sae_device or DEFAULT_FALLBACK_DEVICE,  # type: ignore
                    )
                    self._load_clt_weights()
                else:
                    raise ValueError(
                        "CLT local loading currently requires cfg.json in the CLT directory. "
                        "The temporary GemmaScope2 params-layer loader was intentionally backed out from the "
                        "streamlined dashboard Phase 1 PR stack."
                    )

                # Apply dtype override if specified
                if self.cfg.clt_dtype:
                    try:
                        dtype_torch = getattr(torch, self.cfg.clt_dtype)
                        self.clt.to(dtype=dtype_torch)
                        print(f"Overriding CLT dtype to {self.cfg.clt_dtype}")
                    except AttributeError:
                        raise ValueError(f"Invalid clt_dtype: {self.cfg.clt_dtype}")
                elif hasattr(self.clt.config, "dtype") and self.clt.config.dtype:  # type: ignore
                    self.cfg.clt_dtype = str(self.clt.config.dtype).replace(  # type: ignore
                        "torch.", ""
                    )
                    print(f"Using CLT configured dtype: {self.cfg.clt_dtype}")
                else:
                    self.cfg.clt_dtype = "float32"
                    print(
                        f"CLT dtype not specified, defaulting to {self.cfg.clt_dtype}"
                    )

                # Create wrapper for the specific layer
                from sae_dashboard.clt_layer_wrapper import CLTLayerWrapper

                assert self.cfg.clt_layer_idx is not None  # Already validated above
                self.sae = CLTLayerWrapper(
                    self.clt,
                    self.cfg.clt_layer_idx,
                    clt_model_dir_path=self.cfg.sae_path,
                )
                print(f"Created CLTLayerWrapper for layer {self.cfg.clt_layer_idx}")
            else:
                raise NotImplementedError(
                    "Loading CLT from non-local path (e.g., HF release) is not yet implemented."
                )
        else:
            LoaderClass = SAE
            if self.cfg.from_local_sae:
                self.sae = SAE.load_from_pretrained(  # type: ignore
                    path=self.cfg.sae_path,
                    device=self.cfg.sae_device or DEFAULT_FALLBACK_DEVICE,
                    dtype=self.cfg.sae_dtype if self.cfg.sae_dtype != "" else None,
                )
            else:
                self.sae = LoaderClass.from_pretrained(  # type: ignore
                    release=self.cfg.sae_set,
                    sae_id=self.cfg.sae_path,
                    device=self.cfg.sae_device or DEFAULT_FALLBACK_DEVICE,
                    converter=(
                        get_sae_loader(self.cfg.sae_converter_name)
                        if self.cfg.sae_converter_name
                        else None
                    ),
                )
                if self.cfg.sae_dtype != "":
                    self._apply_sae_dtype_override()

    def _load_clt_weights(self):
        """Load CLT weights from file."""
        weights_path = self._resolve_clt_weights_path()

        print(f"Loading CLT state dict from: {weights_path}")

        # Choose loader based on file extension
        if weights_path.suffix == ".safetensors":
            try:
                from safetensors.torch import load_file as safe_load_file
            except ImportError as e:
                raise ImportError(
                    "safetensors library is required to load .safetensors files. Install via `pip install safetensors`."
                ) from e
            state_dict = safe_load_file(weights_path)
        else:
            state_dict = torch.load(weights_path, map_location=self.cfg.sae_device)

        # Load the state dict
        self.clt.load_state_dict(state_dict)
        print("CLT state dict loaded successfully.")

    def _resolve_clt_weights_path(self, prefer_filename: str | None = None) -> Path:
        candidate_paths: list[Path] = []
        if prefer_filename:
            candidate_paths.append(Path(self.cfg.sae_path) / prefer_filename)
        if self.cfg.clt_weights_filename:
            candidate_paths.append(
                Path(self.cfg.sae_path) / self.cfg.clt_weights_filename
            )

        # If no explicit filename or the file doesn't exist, search common patterns
        candidate_paths.extend(sorted(Path(self.cfg.sae_path).glob("*.safetensors")))
        candidate_paths.append(Path(self.cfg.sae_path) / "model.safetensors")
        candidate_paths.append(Path(self.cfg.sae_path) / "model.pt")
        candidate_paths.append(Path(self.cfg.sae_path) / "model.bin")

        deduped_candidates: list[Path] = []
        seen: set[Path] = set()
        for candidate in candidate_paths:
            if candidate in seen:
                continue
            deduped_candidates.append(candidate)
            seen.add(candidate)

        for candidate in deduped_candidates:
            if candidate.is_file():
                return candidate

        raise FileNotFoundError(
            f"No CLT weights file found in {self.cfg.sae_path}. "
            f"Expected one of: {', '.join(str(path) for path in deduped_candidates)}"
        )

    def _apply_sae_dtype_override(self):
        """Apply dtype override to SAE."""
        if self.cfg.sae_dtype == "float16":
            self.sae.to(dtype=torch.float16)  # type: ignore
        elif self.cfg.sae_dtype == "float32":
            self.sae.to(dtype=torch.float32)  # type: ignore
        elif self.cfg.sae_dtype == "bfloat16":
            self.sae.to(dtype=torch.bfloat16)  # type: ignore
        else:
            raise ValueError(
                f"Unsupported dtype: {self.cfg.sae_dtype}, we support float16, float32, bfloat16"
            )

    def _configure_dtypes(self):
        """Configure data types for SAE and model."""
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
            self.cfg.huggingface_dataset_path = self.sae.cfg.metadata.dataset_path  # type: ignore
        if self.cfg.prompt_dataset_path is None:
            if self.cfg.pretokenized_dataset_path:
                self.cfg.prompt_dataset_path = self.cfg.pretokenized_dataset_path
            else:
                self.cfg.prompt_dataset_path = self.cfg.huggingface_dataset_path
        if self.cfg.prompt_dataset_name is None:
            self.cfg.prompt_dataset_name = self.cfg.huggingface_dataset_config_name
        if self.cfg.prompt_dataset_split is None:
            self.cfg.prompt_dataset_split = self.cfg.huggingface_dataset_split
        if self.cfg.prompt_dataset_text_field is None:
            self.cfg.prompt_dataset_text_field = self.cfg.huggingface_dataset_text_field
        if (
            self.cfg.pretokenized_dataset_path
            and self.cfg.prompt_dataset_path == self.cfg.pretokenized_dataset_path
        ):
            self.cfg.prompt_dataset_mode = "load_from_disk"
        if (
            self.cfg.prompt_dataset_mode in {"load_dataset", "legacy_jsonl"}
            and self.cfg.prompt_dataset_path
        ):
            self.cfg.huggingface_dataset_path = self.cfg.prompt_dataset_path

        self._print_configuration()

        self.sae.cfg.dataset_path = self.cfg.huggingface_dataset_path  # type: ignore
        self.sae.cfg.context_size = self.cfg.n_tokens_in_prompt  # type: ignore

        # Handle architecture as either attribute or method
        architecture = self.sae.cfg.architecture
        if callable(architecture):
            architecture = architecture()

        # Skip fold_W_dec_norm for CLT wrappers and TemporalSAE as they don't support this method
        if "CLTLayerWrapper" in str(type(self.sae)) or architecture in [
            "temporal",
            "topk",
        ]:
            print("NeuronpediaRunner: Skipping fold_W_dec_norm().")
        else:
            self.sae.fold_W_dec_norm()

        print(f"SAE DType: {self.cfg.sae_dtype}")
        print(f"Model DType: {self.cfg.model_dtype}")

    def _print_configuration(self):
        """Print configuration details."""
        print(f"Device Count: {self.device_count}")
        print(f"SAE Device: {self.cfg.sae_device}")
        print(f"Model Device: {self.cfg.model_device}")
        print(f"Model Num Devices: {self.cfg.model_n_devices}")
        print(f"Activation Store Device: {self.cfg.activation_store_device}")
        print(
            "Prompt Dataset: "
            f"mode={self.cfg.prompt_dataset_mode} path={self.cfg.prompt_dataset_path or self.cfg.huggingface_dataset_path}"
        )
        print(f"Forward Pass size: {self.cfg.n_tokens_in_prompt}")

        # number of tokens
        n_tokens_total = self.cfg.n_prompts_total * self.cfg.n_tokens_in_prompt
        print(f"Total number of tokens: {n_tokens_total}")
        print(f"Total number of contexts (prompts): {self.cfg.n_prompts_total}")

        # get the sae's cfg and check if it has from pretrained kwargs
        sae_cfg_json = self.sae.cfg.to_dict()
        self.sae_from_pretrained_kwargs = sae_cfg_json.get(
            "model_from_pretrained_kwargs", {}
        )
        print("SAE Config on disk:")
        print(json.dumps(sae_cfg_json, indent=2))
        if self.sae_from_pretrained_kwargs != {}:
            print("SAE has from_pretrained_kwargs", self.sae_from_pretrained_kwargs)
        else:
            print(
                "SAE does not have from_pretrained_kwargs. Standard TransformerLens Loading"
            )

    def _extract_model_info(self):
        """Extract model ID and layer information from SAE configuration."""
        # For transcoders, model_name might be in metadata
        if hasattr(self.sae.cfg, "model_name"):
            self.model_id = self.sae.cfg.model_name  # type: ignore
        elif (
            hasattr(self.sae.cfg, "metadata") and "model_name" in self.sae.cfg.metadata  # type: ignore
        ):
            self.model_id = self.sae.cfg.metadata["model_name"]  # type: ignore
        else:
            raise ValueError("Could not find model_name in SAE config")

        self.cfg.model_id = self.model_id

        # If the user explicitly provided --layer-num, that wins over any
        # auto-detection. This is useful for HuggingFace-style hook_names
        # (e.g. "model.language_model.layers.17") that don't match the
        # TransformerLens "blocks.<N>.<hook>" pattern.
        if self.cfg.layer is not None:
            self.layer = self.cfg.layer
        elif hasattr(self.sae.cfg, "hook_layer"):
            self.layer = self.sae.cfg.hook_layer  # type: ignore
        elif hasattr(self.sae.cfg, "hook_layer_out"):
            self.layer = self.sae.cfg.hook_layer_out  # type: ignore
        else:
            hook_name = self.sae.cfg.metadata.get("hook_name", "")  # type: ignore
            import re

            match = re.search(r"blocks\.(\d+)\.", hook_name)
            if match:
                self.layer = int(match.group(1))
            else:
                raise ValueError(
                    "Could not find hook_layer in SAE config or extract from "
                    f"hook_name {hook_name!r}. Pass --layer-num explicitly to "
                    "set the layer index."
                )

        self.cfg.layer = self.layer

    @staticmethod
    def _resolve_torch_dtype(dtype_name: str) -> torch.dtype:
        try:
            return getattr(torch, dtype_name)
        except AttributeError as exc:
            raise ValueError(f"Unsupported torch dtype: {dtype_name}") from exc

    @staticmethod
    def _current_rss_bytes() -> int | None:
        status_path = Path("/proc/self/status")
        if not status_path.exists():
            return None
        for line in status_path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            if line.startswith("VmRSS:"):
                parts = line.split()
                if len(parts) >= 2:
                    return int(parts[1]) * 1024
        return None

    def _log_resource_snapshot(self, stage: str) -> None:
        if not self.cfg.log_resource_snapshots:
            return
        rss_bytes = self._current_rss_bytes()
        rss_gib = f"{rss_bytes / (1024**3):.2f}" if rss_bytes is not None else "unknown"
        cuda_allocated_gib = "n/a"
        cuda_reserved_gib = "n/a"
        cuda_max_allocated_gib = "n/a"
        if torch.cuda.is_available():
            device = torch.device(self.cfg.model_device or "cuda")
            cuda_allocated_gib = (
                f"{torch.cuda.memory_allocated(device) / (1024**3):.2f}"
            )
            cuda_reserved_gib = f"{torch.cuda.memory_reserved(device) / (1024**3):.2f}"
            cuda_max_allocated_gib = (
                f"{torch.cuda.max_memory_allocated(device) / (1024**3):.2f}"
            )
        print(
            "[runner_resource] "
            f"stage={stage} "
            f"wrapper={self.cfg.model_wrapper} "
            f"rss_gib={rss_gib} "
            f"cuda_allocated_gib={cuda_allocated_gib} "
            f"cuda_reserved_gib={cuda_reserved_gib} "
            f"cuda_max_allocated_gib={cuda_max_allocated_gib}"
        )

    def _log_batch_boundary_snapshot(
        self, stage: str, feature_batch_count: int, io_snapshot: dict[str, int]
    ) -> None:
        if not self.cfg.log_performance:
            return
        log_perf_event(
            "batch_boundary",
            stage=stage,
            batch=feature_batch_count,
            cpu=cpu_snapshot(),
            io=io_snapshot,
        )

    def _run_feature_batch_with_optional_profile(
        self,
        feature_vis_config_gpt: SaeVisConfig,
        tokens: torch.Tensor,
        feature_batch_count: int,
    ):
        run_kwargs: dict[str, Any] = {
            "encoder": self.sae,  # type: ignore
            "model": self.model,
            "tokens": tokens,
        }
        if self.cfg.use_huggingface:
            run_kwargs["tokenizer"] = self.tokenizer

        sae_vis_runner = SaeVisRunner(feature_vis_config_gpt)
        sae_vis_runner.adopt_token_string_caches(
            getattr(self, "_last_sae_vis_runner", None)
        )
        self._last_sae_vis_runner = sae_vis_runner

        if not self.cfg.torch_profile:
            return sae_vis_runner.run(**run_kwargs)

        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        profile_dir = Path(
            self.cfg.torch_profile_dir or Path(self.cfg.outputs_dir) / "torch_profiles"
        )
        profile_dir.mkdir(parents=True, exist_ok=True)
        trace_path = profile_dir / f"batch-{feature_batch_count}.trace.json"
        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        ) as profiler:
            feature_data = sae_vis_runner.run(**run_kwargs)
        profiler.export_chrome_trace(str(trace_path))
        log_perf_event(
            "torch_profile_trace", batch=feature_batch_count, path=trace_path
        )
        return feature_data

    def _log_hook_alias_summary(self) -> None:
        if not self.cfg.log_hook_aliases:
            return
        hook_dict_keys = list(getattr(self.model, "hook_dict", {}).keys())
        hook_aliases = getattr(self.model, "hook_aliases", {})
        print(
            "[runner_hook_summary] "
            f"wrapper={self.cfg.model_wrapper} "
            f"hook_count={len(hook_dict_keys)} "
            f"alias_count={len(hook_aliases)}"
        )
        for hook_name in [
            self.hook_name,
            getattr(self.sae.cfg.metadata, "hook_name_out", None),
        ]:
            if hook_name is None:
                continue
            print(
                f"[runner_hook_summary] requested_hook={hook_name} present_in_hook_dict={hook_name in hook_dict_keys}"
            )
        print(f"[runner_hook_summary] sample_hooks={hook_dict_keys[:12]}")

    def _prompt_dataset_config(self) -> PromptDatasetConfig:
        dataset_path = self.cfg.prompt_dataset_path
        if dataset_path is None and self.cfg.pretokenized_dataset_path:
            dataset_path = self.cfg.pretokenized_dataset_path
        if not dataset_path:
            dataset_path = self.cfg.huggingface_dataset_path
        prompt_dataset_mode = self.cfg.prompt_dataset_mode
        if (
            self.cfg.pretokenized_dataset_path
            and dataset_path == self.cfg.pretokenized_dataset_path
        ):
            prompt_dataset_mode = "load_from_disk"
        return PromptDatasetConfig(
            dataset_path=dataset_path,
            mode=cast(Any, prompt_dataset_mode),
            dataset_name=self.cfg.prompt_dataset_name
            or self.cfg.huggingface_dataset_config_name,
            split=self.cfg.prompt_dataset_split or self.cfg.huggingface_dataset_split,
            text_field=self.cfg.prompt_dataset_text_field
            or self.cfg.huggingface_dataset_text_field,
            data_files=self.cfg.prompt_dataset_data_files or None,
            data_dir=self.cfg.prompt_dataset_data_dir,
            streaming=self.cfg.dataset_streaming,
            trust_remote_code=self.cfg.prompt_dataset_trust_remote_code,
            metadata_path=self.cfg.prompt_dataset_metadata_path,
        )

    def _materialize_prompt_dataset(self) -> Dataset | IterableDataset:
        self._prompt_dataset_materialization = load_prompt_dataset(
            resolve_prompt_dataset(self._prompt_dataset_config()),
            max_rows=self.cfg.n_prompts_total,
        )
        materialization = self._prompt_dataset_materialization
        dataset_source = materialization.dataset_source
        if isinstance(dataset_source, Dataset):
            print(
                "NeuronpediaRunner: Materialized prompt dataset "
                f"mode={materialization.resolution.mode} rows={len(dataset_source)} "
                f"path={materialization.resolution.dataset_path} "
                f"split={materialization.resolution.split}"
            )
        else:
            print(
                "NeuronpediaRunner: Materialized streaming prompt dataset "
                f"mode={materialization.resolution.mode} path={materialization.resolution.dataset_path} "
                f"split={materialization.resolution.split}"
            )
        return dataset_source

    def _initialize_model(self):
        """Initialize the transformer model."""
        self._log_resource_snapshot("pre_model_init")
        self.hook_name = resolve_capture_hook_name(
            self.sae.cfg.metadata,  # type: ignore[arg-type]
            use_huggingface=self.cfg.use_huggingface,
            capture_hook_name=self.cfg.capture_hook_name,
        )

        if self.cfg.model_wrapper == "bridge":
            from sae_lens.analysis.compat import has_transformer_bridge
            from sae_lens.analysis.sae_transformer_bridge import SAETransformerBridge

            if not has_transformer_bridge():
                raise ImportError(
                    "SAETransformerBridge requires transformer-lens v3+ support in sae_lens."
                )

            if self.cfg.model_n_devices not in (None, 1):
                print(
                    "NeuronpediaRunner: model_n_devices is not currently supported by "
                    "SAETransformerBridge.boot_transformers(); using a single-device bridge load."
                )

            bridge_model_name = self.cfg.hf_model_path or self.model_id
            self.model = SAETransformerBridge.boot_transformers(
                bridge_model_name,  # type: ignore[arg-type]
                device=self.cfg.model_device,
                dtype=self._resolve_torch_dtype(self.cfg.model_dtype),
            )
            if self.cfg.bridge_enable_compatibility_mode:
                compatibility_kwargs = dict(DEFAULT_BRIDGE_COMPATIBILITY_KWARGS)
                compatibility_kwargs.update(self.cfg.bridge_compatibility_mode_kwargs)
                print(
                    "NeuronpediaRunner: Enabling TransformerBridge compatibility mode "
                    f"with kwargs={compatibility_kwargs}"
                )
                self.model.enable_compatibility_mode(**compatibility_kwargs)
            self.tokenizer = getattr(self.model, "tokenizer", None)
        elif self.cfg.use_huggingface:
            print(f"Loading HuggingFace model: {self.model_id}")

            if self.cfg.hf_model_path:
                model_path = self.cfg.hf_model_path
            else:
                model_path = convert_model_name_tl_to_hf(self.model_id)
                if model_path != self.model_id:
                    print(f"Converted model name: {self.model_id} -> {model_path}")

            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            dtype_map = {
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }
            torch_dtype = dtype_map.get(self.cfg.model_dtype, torch.float32)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                device_map=(
                    self.cfg.model_device if self.cfg.model_device != "cpu" else None
                ),
                **self.sae_from_pretrained_kwargs,
            )

            if self.cfg.model_device == "cpu":
                self.model = self.model.to("cpu")

            self.model.eval()
            self._add_to_tokens_method()
            print(f"HuggingFace model loaded on device: {self.cfg.model_device}")
        else:
            hf_model = None
            if self.cfg.hf_model_path:
                print(f"Loading custom HF model from: {self.cfg.hf_model_path}")
                hf_model = AutoModelForCausalLM.from_pretrained(
                    self.cfg.hf_model_path,
                )

            dtype_map = {
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }
            torch_dtype = dtype_map.get(self.cfg.model_dtype, torch.float32)
            self.model = hooked_sae_transformer_cls().from_pretrained_no_processing(
                model_name=self.model_id,  # type: ignore
                device=self.cfg.model_device,
                n_devices=self.cfg.model_n_devices or 1,
                hf_model=hf_model,
                **self.sae_from_pretrained_kwargs,
                dtype=torch_dtype,
            )
            self.tokenizer = self.model.tokenizer  # type: ignore
        if (
            self.cfg.use_transcoder
            or self.cfg.use_skip_transcoder
            or self.cfg.use_clt
            or "hook_mlp_in" in self.hook_name  # type: ignore
        ) and hasattr(self.model, "set_use_hook_mlp_in"):
            cast(Any, self.model).set_use_hook_mlp_in(True)

        if self.cfg.free_unused_model_layers and self.cfg.model_wrapper != "bridge":
            self._free_unused_model_layers()

        self._log_hook_alias_summary()
        self._log_resource_snapshot("post_model_init")

    def _add_to_tokens_method(self):
        """Add a ``to_tokens`` method to HuggingFace models for sae_lens compatibility."""
        tokenizer = self.tokenizer
        model = self.model

        def to_tokens(
            text,
            prepend_bos=True,
            padding_side="right",
            move_to_device=True,
            truncate=True,
        ):
            if isinstance(text, str):
                text = [text]

            original_padding_side = tokenizer.padding_side
            tokenizer.padding_side = padding_side
            encoded = tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=truncate,
                add_special_tokens=False,
            )
            tokenizer.padding_side = original_padding_side

            tokens = encoded["input_ids"]
            if prepend_bos:
                bos_token_id = tokenizer.bos_token_id
                if bos_token_id is None:
                    bos_token_id = tokenizer.eos_token_id
                if bos_token_id is not None:
                    bos_column = torch.full(
                        (tokens.shape[0], 1),
                        bos_token_id,
                        dtype=tokens.dtype,
                        device=tokens.device,
                    )
                    tokens = torch.cat([bos_column, tokens], dim=1)

            if move_to_device:
                device = next(model.parameters()).device
                tokens = tokens.to(device)

            return tokens

        import types

        self.model.to_tokens = types.MethodType(
            lambda self, *args, **kwargs: to_tokens(*args, **kwargs), self.model
        )
        self.model.tokenizer = tokenizer

    def _get_layer_container(self):
        """Return the ``nn.ModuleList`` holding transformer decoder blocks."""
        import torch.nn as nn

        if not self.cfg.use_huggingface:
            blocks = getattr(self.model, "blocks", None)
            return blocks if isinstance(blocks, nn.ModuleList) else None

        from sae_dashboard.hook_utils import (
            get_layer_module_path,
            get_submodule_by_path,
        )

        try:
            layer0_path = get_layer_module_path(self.model, 0)
        except ValueError:
            return None

        parent_path, _, _ = layer0_path.rpartition(".")
        if not parent_path:
            return None

        try:
            container = get_submodule_by_path(self.model, parent_path)
        except (AttributeError, IndexError, KeyError):
            return None

        return container if isinstance(container, nn.ModuleList) else None

    def _free_unused_model_layers(self):
        """Replace unused transformer blocks above the SAE hook layer with ``nn.Identity()``."""
        import torch.nn as nn

        if self.layer is None:
            print("free_unused_model_layers: hook layer is unknown; skipping.")
            return

        blocks = self._get_layer_container()
        if blocks is None:
            print(
                "free_unused_model_layers: could not locate the transformer block container on this model; skipping."
            )
            return

        hook_name = getattr(self, "hook_name", "") or ""
        unsafe_markers = ("hook_pre", "hook_post", "hook_z")
        matched_marker = next(
            (marker for marker in unsafe_markers if marker in hook_name), None
        )
        if matched_marker is not None:
            print(
                f"free_unused_model_layers: skipping trim — hook point '{hook_name}' contains '{matched_marker}', "
                "which would cause `to_resid_direction` to read stacked weights from replaced layers."
            )
            return

        n_total = len(blocks)
        first_unused = self.layer + 1
        if first_unused >= n_total:
            print(
                f"free_unused_model_layers: SAE hook layer is {self.layer} of {n_total}; nothing above the hook layer to free."
            )
            return

        freed = 0
        for i in range(first_unused, n_total):
            if not isinstance(blocks[i], nn.Identity):
                for p in list(blocks[i].parameters()):
                    p.data = torch.empty(0, device=p.device, dtype=p.dtype)
                for b in list(blocks[i].buffers()):
                    b.data = torch.empty(0, device=b.device, dtype=b.dtype)
                blocks[i] = nn.Identity()
                freed += 1

        if not self.cfg.use_huggingface and hasattr(self.model, "setup"):
            self.model.setup()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(
            f"free_unused_model_layers: freed {freed} transformer block(s) above hook layer {self.layer} "
            f"(model had {n_total} total)."
        )

    def _setup_activation_store(self):
        """Set up the activation store for data generation."""
        # set the context size to the number of tokens in the prompt
        self.sae.cfg.metadata.context_size = self.cfg.n_tokens_in_prompt  # type: ignore
        dataset_source = self._materialize_prompt_dataset()
        dataset_streaming = self.cfg.dataset_streaming
        if not isinstance(dataset_source, str):
            dataset_streaming = isinstance(dataset_source, IterableDataset)
        self.activations_store = ActivationsStore.from_sae(
            model=self.model,
            sae=self.sae,  # type: ignore
            dataset=dataset_source,
            streaming=dataset_streaming,
            context_size=self.cfg.n_tokens_in_prompt,
            store_batch_size_prompts=8,  # these don't matter
            n_batches_in_buffer=16,  # these don't matter
            disable_concat_sequences=True,
            device=self.cfg.activation_store_device or "cpu",
        )
        self.cached_activations_dir = Path(
            f"./cached_activations/{self.model_id}_{self.cfg.sae_set}_{self.hook_name}_{self.sae.cfg.d_sae}width_{self.cfg.n_prompts_total}prompts"
        )
        self._log_resource_snapshot("post_activation_store_setup")

    def _setup_output_directory(self):
        """Set up the output directory for results."""
        # if we have additional info, add it to the outputs subdir
        self.np_sae_id_suffix = self.cfg.np_sae_id_suffix

        if not os.path.exists(self.cfg.outputs_dir):
            os.makedirs(self.cfg.outputs_dir)
        self.cfg.outputs_dir = self.create_output_directory()
        if not is_legacy_dashboard_path(self.cfg):
            self._stage_shared_tokens_file(
                require_effective_lengths=self._schedule_requires_effective_lengths()
            )

    def create_output_directory(self) -> str:
        """
        Creates the output directory for storing generated features.

        Returns:
            Path: The path to the created output directory.
        """
        if not self.model_id:
            raise ValueError(
                "model_id must be resolved before creating the output directory."
            )
        outputs_subdir = f"{self._sanitize_path_component(self.model_id)}_{self.cfg.sae_set}_{self.hook_name}_{self.sae.cfg.d_sae}"
        if self.np_sae_id_suffix is not None:
            outputs_subdir += f"_{self.np_sae_id_suffix}"
        outputs_dir = Path(self.cfg.outputs_dir).joinpath(outputs_subdir)
        if outputs_dir.exists() and outputs_dir.is_file():
            raise ValueError(
                f"Error: Output directory {outputs_dir.as_posix()} exists and is a file."
            )
        outputs_dir.mkdir(parents=True, exist_ok=True)
        return str(outputs_dir)

    def _resolved_neuronpedia_set_name(self) -> str:
        set_name = (
            self.cfg.sae_set if self.cfg.np_set_name is None else self.cfg.np_set_name
        )
        if self.np_sae_id_suffix is None:
            return set_name
        return f"{set_name}__{self.np_sae_id_suffix}"

    @staticmethod
    def _sanitize_path_component(value: str) -> str:
        return value.replace("/", "_").replace("\\", "_")

    def _tokens_file_path(self) -> Path:
        return Path(self.cfg.outputs_dir) / f"tokens_{self.cfg.n_prompts_total}.pt"

    @staticmethod
    def _effective_lengths_file_path(tokens_file: Path) -> Path:
        return tokens_file.with_suffix(".effective_lengths.pt")

    @staticmethod
    def _shared_prompt_metadata_file(tokens_file: Path) -> Path:
        return tokens_file.with_suffix(".metadata.json")

    def _shared_tokens_source_path(self) -> Path | None:
        if self.cfg.shared_tokens_file:
            return Path(self.cfg.shared_tokens_file)
        materialization = getattr(self, "_prompt_dataset_materialization", None)
        if not isinstance(materialization, PromptDatasetMaterialization):
            return None
        if materialization.resolution.mode == "load_from_disk":
            default_path = (
                Path(materialization.resolution.dataset_path)
                / f"tokens_{self.cfg.n_prompts_total}.pt"
            )
            self.cfg.shared_tokens_file = str(default_path)
            return default_path
        if materialization.is_tokenized:
            default_path = self._tokens_file_path()
            self.cfg.shared_tokens_file = str(default_path)
            return default_path
        return None

    def _prepare_shared_tokens_from_prompt_dataset(
        self, target_tokens_file: Path
    ) -> Path:
        materialization = getattr(self, "_prompt_dataset_materialization", None)
        if not isinstance(materialization, PromptDatasetMaterialization):
            raise ValueError(
                "A materialized prompt dataset is required to generate shared token sidecars."
            )
        if not materialization.is_tokenized or materialization.token_column is None:
            raise ValueError(
                "Shared prompt token sidecars require a tokenized prompt dataset with an input_ids or tokens column."
            )

        dataset = materialization.dataset_source
        tokens_column = materialization.token_column
        pad_token_id = materialization.pad_token_id
        if pad_token_id is None:
            model_tokenizer = getattr(getattr(self, "model", None), "tokenizer", None)
            model_pad_token_id = getattr(model_tokenizer, "pad_token_id", None)
            if model_pad_token_id is not None:
                pad_token_id = int(model_pad_token_id)

        dataset_rows = len(dataset) if isinstance(dataset, Dataset) else None
        unique_sequences: set[tuple[int, ...]] = set()
        token_rows: list[torch.Tensor] = []
        effective_lengths: list[int] = []
        for row in dataset:
            row_mapping = cast(Mapping[str, Any], row)
            row_tokens = torch.as_tensor(row_mapping[tokens_column], dtype=torch.long)
            if row_tokens.numel() < self.cfg.n_tokens_in_prompt:
                raise ValueError(
                    "Pretokenized row is shorter than n_tokens_in_prompt="
                    f"{self.cfg.n_tokens_in_prompt}: {row_tokens.numel()}"
                )
            row_tokens = row_tokens[: self.cfg.n_tokens_in_prompt].cpu()
            attention_mask_value = row_mapping.get(
                materialization.attention_mask_column or "attention_mask"
            )
            if attention_mask_value is not None:
                attention_mask = torch.as_tensor(attention_mask_value, dtype=torch.long)
                effective_length = int(
                    attention_mask[: self.cfg.n_tokens_in_prompt].sum().item()
                )
            elif pad_token_id is not None:
                nonpad_indices = torch.nonzero(
                    row_tokens != pad_token_id, as_tuple=False
                )
                effective_length = (
                    int(nonpad_indices[-1].item()) + 1
                    if nonpad_indices.numel() > 0
                    else 0
                )
            else:
                effective_length = int(row_tokens.numel())

            row_key = tuple(row_tokens.tolist())
            if (
                row_key in unique_sequences
                and self.cfg.deduplicate_shared_prompt_tokens
            ):
                continue
            unique_sequences.add(row_key)
            token_rows.append(row_tokens)
            effective_lengths.append(effective_length)
            if len(token_rows) >= self.cfg.n_prompts_total:
                break

        if not token_rows:
            raise ValueError(
                f"Prompt dataset {materialization.resolution.dataset_path} did not yield any prompt tokens."
            )
        if (
            self.cfg.strict_shared_prompt_count
            and len(token_rows) < self.cfg.n_prompts_total
        ):
            raise ValueError(
                "Pretokenized dataset did not satisfy the requested prompt count: "
                f"rows={len(token_rows)} requested_prompts={self.cfg.n_prompts_total} "
                f"dataset_rows={dataset_rows if dataset_rows is not None else 'unknown'} unique_rows={len(unique_sequences)} "
                f"deduplicate={self.cfg.deduplicate_shared_prompt_tokens} "
                f"path={materialization.resolution.dataset_path}"
            )

        target_tokens_file.parent.mkdir(parents=True, exist_ok=True)
        token_tensor = torch.stack(token_rows, dim=0)
        effective_length_tensor = torch.tensor(effective_lengths, dtype=torch.int32)
        torch.save(token_tensor, target_tokens_file)
        torch.save(
            effective_length_tensor,
            self._effective_lengths_file_path(target_tokens_file),
        )
        self._shared_prompt_metadata_file(target_tokens_file).write_text(
            json.dumps(
                {
                    "requested_prompts": self.cfg.n_prompts_total,
                    "tensor_shape": list(token_tensor.shape),
                    "dataset_rows": dataset_rows,
                    "unique_rows": len(unique_sequences),
                    "tokens_per_prompt": self.cfg.n_tokens_in_prompt,
                    "deduplicate": self.cfg.deduplicate_shared_prompt_tokens,
                    "source_dataset_mode": materialization.resolution.mode,
                    "source_dataset_path": materialization.resolution.dataset_path,
                    "source_metadata_path": materialization.metadata_path,
                    "effective_lengths_file": str(
                        self._effective_lengths_file_path(target_tokens_file)
                    ),
                    "effective_length_min": min(effective_lengths),
                    "effective_length_max": max(effective_lengths),
                    "effective_length_mean": sum(effective_lengths)
                    / len(effective_lengths),
                    "pad_token_id": pad_token_id,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        print(
            "NeuronpediaRunner: Prepared shared prompt tokens "
            f"rows={len(token_rows)} dataset_rows={dataset_rows if dataset_rows is not None else 'unknown'} unique_rows={len(unique_sequences)} "
            f"requested_prompts={self.cfg.n_prompts_total} tokens_per_prompt={self.cfg.n_tokens_in_prompt} "
            f"deduplicate={self.cfg.deduplicate_shared_prompt_tokens} mode={materialization.resolution.mode} path={target_tokens_file}"
        )
        return target_tokens_file

    def _schedule_requires_effective_lengths(self) -> bool:
        return self.cfg.auto_prompt_bucket_schedule or bool(
            self.cfg.prompt_bucket_schedule_file
        )

    def _ensure_shared_tokens_source_file(
        self, *, require_effective_lengths: bool = False
    ) -> Path | None:
        if getattr(self, "_prompt_dataset_materialization", None) is None and (
            self.cfg.pretokenized_dataset_path
            or self.cfg.prompt_dataset_mode in {"load_from_disk", "legacy_jsonl"}
        ):
            self._materialize_prompt_dataset()

        source = self._shared_tokens_source_path()
        if source is None:
            return None

        effective_lengths_file = self._effective_lengths_file_path(source)
        if source.is_file() and (
            effective_lengths_file.is_file() or not require_effective_lengths
        ):
            return source

        materialization = getattr(self, "_prompt_dataset_materialization", None)
        if (
            isinstance(materialization, PromptDatasetMaterialization)
            and materialization.is_tokenized
        ):
            return self._prepare_shared_tokens_from_prompt_dataset(source)

        candidate_paths = (
            (source, effective_lengths_file) if require_effective_lengths else (source,)
        )
        missing_paths = [str(path) for path in candidate_paths if not path.is_file()]
        raise FileNotFoundError(
            "Shared prompt token sidecars do not exist: " + ", ".join(missing_paths)
        )

    @staticmethod
    def _round_down_to_multiple(value: int, *, multiple: int) -> int:
        if multiple <= 1 or value <= multiple:
            return value
        rounded_value = (value // multiple) * multiple
        return rounded_value if rounded_value > 0 else value

    @staticmethod
    def _scale_batch_size(
        base_value: int,
        *,
        scale: float,
        scale_limit: float,
        max_value: int,
        round_to: int,
    ) -> int:
        scaled_value = max(base_value, int(base_value * min(scale, scale_limit)))
        scaled_value = min(scaled_value, max_value)
        return NeuronpediaRunner._round_down_to_multiple(
            scaled_value, multiple=round_to
        )

    @staticmethod
    def _prefer_exact_primary_acts_partition(
        *,
        n_prompts_in_forward_pass: int,
        primary_acts_batch_size: int | None,
        max_prompt_count: int,
        max_prompt_reduction_fraction: float = 0.10,
        max_primary_acts_reduction_fraction: float = 0.10,
    ) -> tuple[int, int | None]:
        capped_prompts = min(n_prompts_in_forward_pass, max_prompt_count)
        if primary_acts_batch_size is None:
            return capped_prompts, None

        capped_acts = min(primary_acts_batch_size, capped_prompts)
        if capped_acts <= 0 or capped_prompts % capped_acts == 0:
            return capped_prompts, capped_acts

        min_prompts = max(
            1, int(np.ceil(capped_prompts * (1.0 - max_prompt_reduction_fraction)))
        )
        min_acts = max(
            1, int(np.ceil(capped_acts * (1.0 - max_primary_acts_reduction_fraction)))
        )

        best_pair: tuple[int, int] | None = None
        for acts in range(capped_acts, min_acts - 1, -1):
            prompts = (capped_prompts // acts) * acts
            if prompts < min_prompts or prompts <= 0:
                continue

            candidate = (prompts, acts)
            if (
                best_pair is None
                or candidate[0] > best_pair[0]
                or (candidate[0] == best_pair[0] and candidate[1] > best_pair[1])
            ):
                best_pair = candidate

        if best_pair is not None:
            return best_pair
        return capped_prompts, capped_acts

    def _normalized_prompt_bucket_ceilings(
        self, effective_lengths: list[int]
    ) -> tuple[int, ...]:
        return derive_prompt_bucket_ceilings(
            effective_lengths,
            max_context_size=self.cfg.n_tokens_in_prompt,
            explicit_bucket_ceilings=self.cfg.prompt_bucket_ceilings,
        )

    def _load_prompt_effective_lengths(self, tokens: torch.Tensor) -> list[int]:
        effective_lengths_path = self._effective_lengths_file_path(
            self._tokens_file_path()
        )
        if not effective_lengths_path.is_file():
            self._stage_shared_tokens_file(require_effective_lengths=True)

        if not effective_lengths_path.is_file():
            raise FileNotFoundError(
                f"Prompt bucket schedule requires staged effective lengths sidecar: {effective_lengths_path}"
            )

        effective_lengths_tensor = torch.load(
            effective_lengths_path, map_location="cpu"
        )
        if not isinstance(effective_lengths_tensor, torch.Tensor):
            raise TypeError(
                f"Prompt bucket effective lengths sidecar must contain a tensor: {effective_lengths_path}"
            )

        effective_lengths = [
            int(length) for length in effective_lengths_tensor.tolist()
        ]
        if len(effective_lengths) != tokens.shape[0]:
            raise ValueError(
                "Prompt bucket schedule effective-length count does not match tokens rows: "
                f"lengths={len(effective_lengths)} tokens={tokens.shape[0]}"
            )
        return effective_lengths

    def _auto_prompt_bucket_entries(
        self, effective_lengths: list[int]
    ) -> list[dict[str, Any]]:
        if self.cfg.n_prompts_in_forward_pass <= 0:
            raise ValueError(
                "n_prompts_in_forward_pass must be positive when auto prompt bucketing is enabled."
            )
        if self.cfg.prompt_bucket_scale_limit <= 0:
            raise ValueError(
                "prompt_bucket_scale_limit must be positive when auto prompt bucketing is enabled."
            )
        if self.cfg.prompt_primary_acts_scale_limit <= 0:
            raise ValueError(
                "prompt_primary_acts_scale_limit must be positive when auto prompt bucketing is enabled."
            )
        if self.cfg.prompt_batch_size_round_to <= 0:
            raise ValueError(
                "prompt_batch_size_round_to must be positive when auto prompt bucketing is enabled."
            )

        bucket_entries: list[dict[str, Any]] = []
        lower_exclusive = 0
        for bucket_ceiling in self._normalized_prompt_bucket_ceilings(
            effective_lengths
        ):
            bucket_prompt_count = sum(
                1
                for effective_length in effective_lengths
                if lower_exclusive < effective_length <= bucket_ceiling
            )
            if bucket_prompt_count <= 0:
                lower_exclusive = bucket_ceiling
                continue

            scale = self.cfg.n_tokens_in_prompt / bucket_ceiling
            prompts_in_forward_pass = self._scale_batch_size(
                self.cfg.n_prompts_in_forward_pass,
                scale=scale,
                scale_limit=self.cfg.prompt_bucket_scale_limit,
                max_value=bucket_prompt_count,
                round_to=self.cfg.prompt_batch_size_round_to,
            )

            primary_acts_batch_size = self.cfg.primary_acts_batch_size
            if primary_acts_batch_size is not None:
                primary_acts_batch_size = self._scale_batch_size(
                    primary_acts_batch_size,
                    scale=scale,
                    scale_limit=self.cfg.prompt_primary_acts_scale_limit,
                    max_value=prompts_in_forward_pass,
                    round_to=self.cfg.prompt_batch_size_round_to,
                )

            prompts_in_forward_pass, primary_acts_batch_size = (
                self._prefer_exact_primary_acts_partition(
                    n_prompts_in_forward_pass=prompts_in_forward_pass,
                    primary_acts_batch_size=primary_acts_batch_size,
                    max_prompt_count=bucket_prompt_count,
                )
            )

            bucket_entries.append(
                {
                    "bucket_ceiling": bucket_ceiling,
                    "prompt_count": bucket_prompt_count,
                    "selected_config": {
                        "n_prompts_in_forward_pass": prompts_in_forward_pass,
                        "primary_acts_batch_size": primary_acts_batch_size,
                    },
                }
            )
            lower_exclusive = bucket_ceiling

        return bucket_entries

    def _expand_prompt_bucket_schedule(
        self,
        *,
        bucket_entries: list[dict[str, Any]],
        effective_lengths: list[int],
        tokens: torch.Tensor,
        schedule_path: Path | None = None,
    ) -> list[dict[str, Any]]:
        schedule: list[dict[str, Any]] = []
        scheduled_prompt_indices: set[int] = set()
        lower_exclusive = 0
        for bucket_entry in sorted(
            bucket_entries,
            key=lambda entry: int(
                entry.get("bucket_ceiling", entry.get("upper_inclusive", 0))
            ),
        ):
            if not isinstance(bucket_entry, dict):
                raise ValueError(
                    f"Prompt bucket entry must be a mapping, got: {bucket_entry!r}"
                )
            bucket_ceiling = int(
                bucket_entry.get(
                    "bucket_ceiling", bucket_entry.get("upper_inclusive", 0)
                )
            )
            if bucket_ceiling <= lower_exclusive:
                raise ValueError(
                    f"Prompt bucket ceilings must increase strictly; got {bucket_ceiling} after {lower_exclusive}."
                )
            selected_config = bucket_entry.get("selected_config")
            if not isinstance(selected_config, dict):
                if schedule_path is None:
                    raise ValueError(
                        f"Auto prompt bucket entry {bucket_ceiling} is missing selected_config data."
                    )
                raise ValueError(
                    f"Prompt bucket entry {bucket_ceiling} is missing selected_config data in {schedule_path}"
                )
            bucket_prompt_count = int(bucket_entry.get("prompt_count", 0))
            bucket_indices = [
                index
                for index, effective_length in enumerate(effective_lengths)
                if lower_exclusive < effective_length <= bucket_ceiling
            ]
            if bucket_prompt_count != len(bucket_indices):
                raise ValueError(
                    "Prompt bucket schedule count does not match staged effective lengths for bucket "
                    f"{bucket_ceiling}: expected={bucket_prompt_count} actual={len(bucket_indices)}"
                )
            prompts_in_forward_pass = int(
                selected_config.get(
                    "n_prompts_in_forward_pass", self.cfg.n_prompts_in_forward_pass
                )
            )
            if prompts_in_forward_pass <= 0:
                raise ValueError(
                    f"Prompt bucket schedule n_prompts_in_forward_pass must be positive for bucket {bucket_ceiling}."
                )
            primary_acts_batch_size = selected_config.get(
                "primary_acts_batch_size", self.cfg.primary_acts_batch_size
            )
            if primary_acts_batch_size is not None:
                primary_acts_batch_size = int(primary_acts_batch_size)

            for start_index in range(0, len(bucket_indices), prompts_in_forward_pass):
                prompt_indices = bucket_indices[
                    start_index : start_index + prompts_in_forward_pass
                ]
                if not prompt_indices:
                    continue
                schedule.append(
                    {
                        "prompt_indices": prompt_indices,
                        "seq_length": bucket_ceiling,
                        "primary_acts_batch_size": primary_acts_batch_size,
                    }
                )
                scheduled_prompt_indices.update(prompt_indices)
            lower_exclusive = bucket_ceiling

        expected_prompt_indices = set(range(tokens.shape[0]))
        if scheduled_prompt_indices != expected_prompt_indices:
            missing = sorted(expected_prompt_indices - scheduled_prompt_indices)
            extras = sorted(scheduled_prompt_indices - expected_prompt_indices)
            raise ValueError(
                "Prompt bucket schedule does not partition the staged token rows exactly: "
                f"missing={missing[:10]} extras={extras[:10]}"
            )

        return schedule

    @staticmethod
    def _stage_shared_file(source: Path, target: Path) -> None:
        if target.exists():
            return
        try:
            target.symlink_to(source)
        except OSError:
            shutil.copy2(source, target)

    def _stage_shared_tokens_file(
        self, *, require_effective_lengths: bool = False
    ) -> None:
        source = self._ensure_shared_tokens_source_file(
            require_effective_lengths=require_effective_lengths
        )
        if source is None:
            return
        if not source.is_file():
            raise FileNotFoundError(f"Shared tokens file does not exist: {source}")

        target = self._tokens_file_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        if source == target:
            return
        self._stage_shared_file(source, target)

        source_base_name = (
            source.name[: -len(source.suffix)] if source.suffix else source.name
        )
        target_base_name = (
            target.name[: -len(target.suffix)] if target.suffix else target.name
        )
        for sidecar in sorted(source.parent.glob(f"{source_base_name}.*")):
            if not sidecar.is_file() or sidecar == source:
                continue
            relative_suffix = sidecar.name[len(source_base_name) :]
            sidecar_target = target.parent / f"{target_base_name}{relative_suffix}"
            self._stage_shared_file(sidecar, sidecar_target)

    def _write_converter_input_artifact(
        self, feature_data: Any, batch_num: int
    ) -> Path | None:
        if not self.cfg.converter_input_artifact_dir:
            return None

        artifact_dir = Path(self.cfg.converter_input_artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifact_dir / f"converter_input_batch_{batch_num}.pt"
        torch.save(
            {
                "feature_data_dict": feature_data.feature_data_dict,
                "runner_cfg": self.cfg,
                "vocab_dict": self.vocab_dict,
                "model_d_vocab": int(cast(Any, self.model).cfg.d_vocab),
                "model_id": self.model_id,
                "layer": self.layer,
                "hook_name": self.hook_name,
                "batch_num": batch_num,
            },
            artifact_path,
        )
        return artifact_path

    def _resolved_columnar_activation_copy_layer(self) -> str:
        if self.layer is None:
            raise ValueError(
                "layer must be resolved before columnar activation metadata is built."
            )
        return f"{self.layer}-{self._resolved_neuronpedia_set_name()}"

    def _resolved_columnar_activation_copy_id_prefix(self) -> str:
        return f"{self._resolved_columnar_activation_copy_layer()}-activation"

    def _load_prompt_bucket_schedule(
        self, tokens: torch.Tensor
    ) -> list[dict[str, Any]] | None:
        if (
            not self.cfg.prompt_bucket_schedule_file
            and not self.cfg.auto_prompt_bucket_schedule
        ):
            return None
        effective_lengths = self._load_prompt_effective_lengths(tokens)

        if self.cfg.prompt_bucket_schedule_file:
            schedule_path = Path(self.cfg.prompt_bucket_schedule_file)
            if not schedule_path.is_file():
                raise FileNotFoundError(
                    f"Prompt bucket schedule file does not exist: {schedule_path}"
                )

            payload = json.loads(schedule_path.read_text(encoding="utf-8"))
            bucket_entries = payload.get("buckets")
            if not isinstance(bucket_entries, list):
                raise ValueError(
                    f"Prompt bucket schedule file is missing a 'buckets' list: {schedule_path}"
                )

            return self._expand_prompt_bucket_schedule(
                bucket_entries=bucket_entries,
                effective_lengths=effective_lengths,
                tokens=tokens,
                schedule_path=schedule_path,
            )

        bucket_entries = self._auto_prompt_bucket_entries(effective_lengths)
        return self._expand_prompt_bucket_schedule(
            bucket_entries=bucket_entries,
            effective_lengths=effective_lengths,
            tokens=tokens,
        )

    def _log_token_snapshot(
        self,
        stage: str,
        tokens: torch.Tensor | None = None,
    ) -> None:
        if not self.cfg.log_resource_snapshots:
            return

        token_shape = tuple(tokens.shape) if tokens is not None else None
        token_device = str(tokens.device) if tokens is not None else "n/a"
        token_dtype = str(tokens.dtype) if tokens is not None else "n/a"
        token_bytes_mib = (
            f"{tokens.numel() * tokens.element_size() / (1024**2):.2f}"
            if tokens is not None
            else "0.00"
        )
        print(
            "[runner_token_snapshot] "
            f"stage={stage} "
            f"token_shape={token_shape} "
            f"token_device={token_device} "
            f"token_dtype={token_dtype} "
            f"token_bytes_mib={token_bytes_mib}"
        )
        self._log_resource_snapshot(stage)

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

        self._log_token_snapshot("before_generate_tokens")

        for batch_idx in pbar:
            batch_tokens = activations_store.get_batch_tokens(
                **(
                    {"move_to_model_device": False}
                    if _ACTIVATIONS_STORE_SUPPORTS_DEVICE_SEAM
                    else {}
                )
            )
            if batch_idx == 0:
                self._log_token_snapshot("after_get_batch_tokens_0", batch_tokens)
            if self.cfg.shuffle_tokens:
                batch_tokens = batch_tokens[torch.randperm(batch_tokens.shape[0])]
                if batch_idx == 0:
                    self._log_token_snapshot(
                        "after_shuffle_batch_tokens_0", batch_tokens
                    )

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

        all_tokens = all_tokens.cpu()
        self._log_token_snapshot("after_generate_tokens", all_tokens)

        return all_tokens

    def add_prefix_suffix_to_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        self._log_token_snapshot("before_add_prefix_suffix", tokens)
        original_length = tokens.shape[1]
        bos_tokens = tokens[:, 0]  # might not be if sae.cfg.prepend_bos is False

        # return tokens if no prefix or suffix
        if self.cfg.prefix_str is None and self.cfg.suffix_str is None:
            return tokens

        # generate tokens for the prefix and suffix
        prefix_tokens = (
            self.model.tokenizer.encode(self.cfg.prefix_str)  # type: ignore
            if self.cfg.prefix_str is not None
            else []
        )
        suffix_tokens = (
            self.model.tokenizer.encode(self.cfg.suffix_str)  # type: ignore
            if self.cfg.suffix_str is not None
            else []
        )

        # Calculate how many tokens to keep from the original
        keep_length = original_length - len(prefix_tokens) - len(suffix_tokens)

        if keep_length <= 0:
            raise ValueError("Prefix and suffix are too long for the given tokens.")

        # Trim original tokens
        if (
            hasattr(self.sae.cfg.metadata, "prepend_bos")
            and self.sae.cfg.metadata.prepend_bos  # type: ignore
        ):
            tokens = tokens[:, : keep_length - self.sae.cfg.metadata.prepend_bos]  # type: ignore
        else:
            tokens = tokens[:, :keep_length]

        if self.cfg.prefix_str:
            prefix = torch.tensor(
                prefix_tokens, dtype=tokens.dtype, device=tokens.device
            )
            prefix_repeated = prefix.unsqueeze(0).repeat(tokens.shape[0], 1)
            # if sae.cfg.prepend_bos, then add that before the suffix
            if (
                hasattr(self.sae.cfg.metadata, "prepend_bos")
                and self.sae.cfg.metadata.prepend_bos  # type: ignore
            ):
                bos = bos_tokens.unsqueeze(1)
                prefix_repeated = torch.cat([bos, prefix_repeated], dim=1)
            tokens = torch.cat([prefix_repeated, tokens], dim=1)

        if self.cfg.suffix_str:
            suffix = torch.tensor(
                suffix_tokens, dtype=tokens.dtype, device=tokens.device
            )
            suffix_repeated = suffix.unsqueeze(0).repeat(tokens.shape[0], 1)
            tokens = torch.cat([tokens, suffix_repeated], dim=1)

        # assert length hasn't changed
        assert tokens.shape[1] == original_length
        self._log_token_snapshot("after_add_prefix_suffix", tokens)
        return tokens

    def _log_columnar_output_summary(
        self,
        feature_batch_count: int,
        feature_count: int,
        feature_data: "SaeVisColumnarData",
    ) -> None:
        if self.cfg.log_performance:
            log_perf_event(
                "columnar_output_summary",
                batch=feature_batch_count,
                feature_count=feature_count,
                artifact_dir=str(feature_data.artifact_dir),
                manifest_path=str(feature_data.manifest_path),
                row_counts={
                    batch.feature_batch_index: batch.row_counts
                    for batch in feature_data.batches
                },
            )
        print(f"Columnar output written to {feature_data.manifest_path}")

    def _enqueue_columnar_finalize(
        self,
        feature_batch_count: int,
        feature_count: int,
        pending_finalize: Any,
    ) -> None:
        """Submit a deferred columnar batch write to the single ordered writer.

        In-flight writes are bounded to one, so per-batch completion markers (each
        batch's root manifest, written last inside finalize) land in batch order and a
        crash loses at most the in-flight batch — preserving the batch-level resume
        granularity the pipeline relies on, including across GPUs."""
        if self._columnar_write_executor is None:
            self._columnar_write_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="columnar-batch-write"
            )
        while len(self._pending_columnar_writes) >= 1:
            self._drain_oldest_columnar_write()
        future = self._columnar_write_executor.submit(pending_finalize)
        self._pending_columnar_writes.append(
            (feature_batch_count, feature_count, future)
        )

    def _drain_oldest_columnar_write(self) -> None:
        feature_batch_count, feature_count, future = (
            self._pending_columnar_writes.popleft()
        )
        try:
            feature_data = future.result()
        except Exception:
            # Do not let later batches write completion markers past a failed batch;
            # cancel anything queued and surface the failure.
            for _, _, pending_future in self._pending_columnar_writes:
                pending_future.cancel()
            self._pending_columnar_writes.clear()
            raise
        self._log_columnar_output_summary(
            feature_batch_count, feature_count, feature_data
        )

    def _drain_columnar_writes(self) -> None:
        try:
            while self._pending_columnar_writes:
                self._drain_oldest_columnar_write()
        finally:
            if self._columnar_write_executor is not None:
                self._columnar_write_executor.shutdown(wait=True)
                self._columnar_write_executor = None

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
        tokens_file = self._tokens_file_path()
        self._log_token_snapshot("before_get_tokens")
        should_stage_shared_tokens = (
            not is_legacy_dashboard_path(self.cfg)
            or self.cfg.shared_tokens_file is not None
            or self._schedule_requires_effective_lengths()
        )
        if not tokens_file.is_file() and should_stage_shared_tokens:
            self._stage_shared_tokens_file(
                require_effective_lengths=self._schedule_requires_effective_lengths()
            )
        if tokens_file.is_file():
            print("Tokens exist, loading them.")
            tokens = torch.load(tokens_file, map_location="cpu").cpu()
            self._log_token_snapshot("loaded_tokens_from_cache", tokens)
        else:
            print("Tokens don't exist, making them.")
            tokens = self.generate_tokens(
                self.activations_store,
                self.cfg.n_prompts_total,
            )
            torch.save(
                tokens.cpu(),
                tokens_file,
            )
            self._log_token_snapshot("saved_generated_tokens", tokens)

        if has_duplicate_rows(tokens):
            print(
                "NeuronpediaRunner: Loaded tokens contain duplicate rows; continuing because "
                "pretokenized prompt datasets may preserve distinct examples with identical token contexts."
            )

        return tokens

    def get_vocab_dict(self) -> Dict[int, str]:
        # get vocab - use tokenizer which is set for both HuggingFace and TransformerLens
        vocab_dict: dict = self.tokenizer.vocab  # type: ignore
        new_vocab_dict = {}
        # Replace substrings in the keys of vocab_dict using HTML_ANOMALIES
        for k, v in vocab_dict.items():  # type: ignore
            modified_key = k
            for anomaly in HTML_ANOMALIES:
                modified_key = modified_key.replace(anomaly, HTML_ANOMALIES[anomaly])
            new_vocab_dict[v] = modified_key
        vocab_dict = new_vocab_dict

        # Get vocab size from the appropriate source
        if self.cfg.use_huggingface:
            vocab_size = getattr(self.model.config, "vocab_size", len(vocab_dict))
        else:
            vocab_size = self.model.cfg.d_vocab

        # pad with blank tokens to the actual vocab size
        for i in range(len(vocab_dict), vocab_size):
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
                name=f"{self.model_id}_{set_name}_{self.hook_name}_{current_time}",
                save_code=True,
                mode="online",
                config=wandb_cfg,
            )

        self.n_features = self.sae.cfg.d_sae
        assert self.n_features is not None

        self.target_feature_indexes = list(range(self.n_features))

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

        if legacy_runner.is_preserved_legacy_path(self.cfg):
            legacy_runner.run_legacy_batch_loop(
                self,
                feature_idx=feature_idx,
                tokens=tokens,
            )
        else:
            prompt_minibatch_schedule = self._load_prompt_bucket_schedule(tokens)
            self._log_token_snapshot("tokens_ready_for_batches", tokens)

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

                    if self.cfg.dashboard_output_format == "columnar":
                        output_root = (
                            Path(self.cfg.outputs_dir)
                            / f"batch-{feature_batch_count}.columnar"
                        )
                        output_file = str(output_root / "manifest.json")
                        if output_root.exists() and not Path(output_file).is_file():
                            shutil.rmtree(output_root)
                    else:
                        output_root = None
                        output_file = (
                            f"{self.cfg.outputs_dir}/batch-{feature_batch_count}.json"
                        )

                    if Path(output_file).is_file():
                        logline = (
                            f"\n++++++++++ Skipping Batch #{feature_batch_count} output. "
                            f"File exists: {output_file} ++++++++++\n"
                        )
                        print(logline)
                        continue

                    print(f"========== Running Batch #{feature_batch_count} ==========")
                    self._log_resource_snapshot(f"pre_batch_{feature_batch_count}")

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
                        hook_point=self.hook_name,  # type: ignore
                        features=features_to_process,
                        minibatch_size_features=self.cfg.n_features_at_a_time,
                        minibatch_size_tokens=self.cfg.n_prompts_in_forward_pass,
                        primary_acts_batch_size=self.cfg.primary_acts_batch_size,
                        quantile_feature_batch_size=self.cfg.quantile_feature_batch_size,
                        verbose=True,
                        log_performance=self.cfg.log_performance,
                        profile_rolling_substages=self.cfg.profile_rolling_substages,
                        cleanup_each_minibatch=self.cfg.cleanup_each_minibatch,
                        torch_profile=self.cfg.torch_profile,
                        torch_profile_dir=(
                            Path(self.cfg.torch_profile_dir)
                            if self.cfg.torch_profile_dir
                            else None
                        ),
                        device=self.cfg.sae_device or DEFAULT_FALLBACK_DEVICE,
                        feature_centric_layout=layout,
                        perform_ablation_experiments=False,
                        dtype=self.cfg.sae_dtype,
                        prompt_minibatch_schedule=prompt_minibatch_schedule,
                        cache_dir=(
                            self.cached_activations_dir
                            if self.cfg.use_cached_activations
                            else None
                        ),
                        ignore_tokens={
                            tok_id
                            for tok_id in (
                                self.tokenizer.pad_token_id,  # type: ignore
                                self.tokenizer.bos_token_id,  # type: ignore
                                self.tokenizer.eos_token_id,  # type: ignore
                            )
                            if tok_id is not None
                        },
                        ignore_positions=self.cfg.ignore_positions or [],
                        ignore_high_activation_norm_multiple=self.cfg.ignore_high_activation_norm_multiple,
                        use_dfa=self.cfg.use_dfa,
                        use_huggingface=self.cfg.use_huggingface,
                        sequence_replay_artifact_dir=(
                            Path(self.cfg.sequence_replay_artifact_dir)
                            if self.cfg.sequence_replay_artifact_dir
                            else None
                        ),
                        correlation_accumulation_device=self.cfg.correlation_accumulation_device,
                        rolling_coefficient_num_threads=self.cfg.rolling_coefficient_num_threads,
                        activation_significance_floor=self.cfg.activation_significance_floor,
                        feature_statistics_backend=self.cfg.feature_statistics_backend,
                        logits_histogram_backend=self.cfg.logits_histogram_backend,
                        activation_histogram_backend=self.cfg.activation_histogram_backend,
                        defer_component_construction=self.cfg.defer_component_construction,
                        sequence_selection_backend=self.cfg.sequence_selection_backend,
                        sequence_top_acts_positive_only=self.cfg.sequence_top_acts_positive_only,
                        sequence_dedup_across_groups=self.cfg.sequence_dedup_across_groups,
                        sequence_skip_dead_features=self.cfg.sequence_skip_dead_features,
                        sequence_half_open_interval_bins=self.cfg.sequence_half_open_interval_bins,
                        columnar_max_device_staged_acts_bytes=(
                            self.cfg.columnar_max_device_staged_acts_bytes
                        ),
                        columnar_row_chunk_size=(self.cfg.columnar_row_chunk_size),
                        logits_table_mask_token_pattern=self.cfg.logits_table_mask_token_pattern,
                        dashboard_output_format=self.cfg.dashboard_output_format,
                        columnar_defer_batch_write=(
                            self.cfg.dashboard_output_format == "columnar"
                            and self.cfg.overlap_batch_packaging
                        ),
                        columnar_artifact_dir=output_root,
                        columnar_artifact_format=self.cfg.columnar_artifact_format,
                        columnar_write_page_index=self.cfg.columnar_write_page_index,
                        columnar_parquet_row_group_size=self.cfg.columnar_parquet_row_group_size,
                        columnar_emit_sequence_rows=self.cfg.columnar_emit_sequence_rows,
                        columnar_emit_activation_rows=self.cfg.columnar_emit_activation_rows,
                        columnar_emit_activation_copy_rows=self.cfg.columnar_emit_activation_copy_rows,
                        columnar_activation_copy_model_id=(
                            self.cfg.columnar_activation_copy_model_id or self.model_id
                        ),
                        columnar_activation_copy_layer=self._resolved_columnar_activation_copy_layer(),
                        columnar_activation_copy_creator_id=(
                            os.getenv("DEFAULT_CREATOR_ID") or ""
                        ),
                        columnar_activation_copy_created_at=datetime.now(timezone.utc)
                        .replace(tzinfo=None)
                        .isoformat(),
                        columnar_activation_copy_id_prefix=self._resolved_columnar_activation_copy_id_prefix(),
                    )

                    self._log_token_snapshot(
                        f"before_feature_run_{feature_batch_count}", tokens
                    )
                    batch_start_time = time.perf_counter()
                    batch_start_io = process_io_snapshot()
                    self._log_batch_boundary_snapshot(
                        "pre_batch", feature_batch_count, batch_start_io
                    )
                    feature_data = self._run_feature_batch_with_optional_profile(
                        feature_vis_config_gpt,
                        tokens,
                        feature_batch_count,
                    )
                    self._log_resource_snapshot(
                        f"after_feature_run_{feature_batch_count}"
                    )

                    converter_input_artifact = None
                    if self.cfg.dashboard_output_format != "columnar":
                        converter_input_artifact = self._write_converter_input_artifact(
                            feature_data,
                            feature_batch_count,
                        )

                    self.cfg.model_id = self.model_id
                    self.cfg.layer = self.layer
                    if self.cfg.dashboard_output_format == "columnar":
                        if not isinstance(feature_data, SaeVisColumnarData):
                            raise TypeError(
                                "Columnar dashboard output requires SaeVisRunner to return SaeVisColumnarData."
                            )
                        if feature_data.pending_finalize is not None:
                            self._enqueue_columnar_finalize(
                                feature_batch_count,
                                len(features_to_process),
                                feature_data.pending_finalize,
                            )
                        else:
                            self._log_columnar_output_summary(
                                feature_batch_count,
                                len(features_to_process),
                                feature_data,
                            )
                    else:
                        with timed_stage(
                            self.cfg.log_performance,
                            "neuronpedia_conversion_and_json_serialization",
                            batch=feature_batch_count,
                            feature_count=len(features_to_process),
                        ):
                            json_object = NeuronpediaConverter.convert_to_np_json(
                                cast(Any, self.model),
                                cast(Any, feature_data),
                                self.cfg,
                                self.vocab_dict,
                            )
                        write_start_io = process_io_snapshot()
                        with elapsed_timer() as write_timing:
                            with open(
                                output_file,
                                "w",
                            ) as f:
                                f.write(json_object)
                        write_end_io = process_io_snapshot()
                        if self.cfg.log_performance:
                            output_bytes = len(json_object.encode("utf-8"))
                            write_wall_s = max(write_timing.get("wall_s", 0.0), 1e-9)
                            log_perf_event(
                                "disk_write",
                                batch=feature_batch_count,
                                path=output_file,
                                output_bytes=output_bytes,
                                wall_s=write_wall_s,
                                output_mib_per_s=output_bytes
                                / (1024**2)
                                / write_wall_s,
                                process_io_delta=io_delta(write_start_io, write_end_io),
                            )
                            if converter_input_artifact is not None:
                                log_perf_event(
                                    "converter_input_artifact",
                                    batch=feature_batch_count,
                                    path=str(converter_input_artifact),
                                    size_bytes=converter_input_artifact.stat().st_size,
                                )
                        print(f"Output written to {output_file}")
                    batch_end_io = process_io_snapshot()
                    if self.cfg.log_performance:
                        log_perf_event(
                            "batch_total",
                            batch=feature_batch_count,
                            wall_s=time.perf_counter() - batch_start_time,
                            process_io_delta=io_delta(batch_start_io, batch_end_io),
                        )
                    self._log_batch_boundary_snapshot(
                        "post_batch", feature_batch_count, batch_end_io
                    )
                    self._log_resource_snapshot(f"post_batch_{feature_batch_count}")

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
                    self._release_unused_host_memory()
                    self._log_resource_snapshot(
                        f"post_batch_cleanup_{feature_batch_count}"
                    )
                self._drain_columnar_writes()
        if self.cfg.use_wandb:
            wandb.sdk.finish()

        if self.cfg.output_neuronpedia_exports:
            self._run_neuronpedia_export()

    def _validate_neuronpedia_export_cfg(self) -> None:
        """Ensure all required metadata for the Neuronpedia export is set."""
        required_fields = {
            "neuronpedia_creator_name": self.cfg.neuronpedia_creator_name,
            "neuronpedia_release_id": self.cfg.neuronpedia_release_id,
            "neuronpedia_release_title": self.cfg.neuronpedia_release_title,
            "neuronpedia_release_url": self.cfg.neuronpedia_release_url,
            "neuronpedia_source_set_description": (
                self.cfg.neuronpedia_source_set_description
            ),
            "neuronpedia_model_name": self.cfg.neuronpedia_model_name,
        }
        missing = [name for name, value in required_fields.items() if not value]
        if missing:
            raise ValueError(
                "--output-neuronpedia-exports requires all of: "
                + ", ".join(f"--{name.replace('_', '-')}" for name in missing)
            )
        if not self.cfg.np_set_name:
            raise ValueError(
                "--output-neuronpedia-exports requires --np-set-name "
                "(used as the Neuronpedia source-set ID)."
            )

    def _run_neuronpedia_export(self) -> None:
        """Convert ``batch-*.json`` outputs to the Neuronpedia bulk-import layout."""
        assert self.cfg.np_set_name is not None  # Already validated.
        assert self.cfg.neuronpedia_creator_name is not None
        assert self.cfg.neuronpedia_release_id is not None
        assert self.cfg.neuronpedia_release_title is not None
        assert self.cfg.neuronpedia_release_url is not None
        assert self.cfg.neuronpedia_source_set_description is not None
        assert self.cfg.neuronpedia_model_name is not None
        assert self.layer is not None
        assert self.model_id is not None

        exports_dir = self.cfg.neuronpedia_exports_dir or os.path.join(
            os.path.dirname(self.cfg.outputs_dir.rstrip(os.sep)) or ".",
            "neuronpedia_exports",
        )

        export_cfg = NeuronpediaExportConfig(
            saedashboard_output_dir=self.cfg.outputs_dir,
            exports_dir=exports_dir,
            creator_name=self.cfg.neuronpedia_creator_name,
            creator_id=resolve_creator_id(self.cfg.neuronpedia_creator_id),
            release_id=self.cfg.neuronpedia_release_id,
            release_title=self.cfg.neuronpedia_release_title,
            url=self.cfg.neuronpedia_release_url,
            model_name=self.cfg.neuronpedia_model_name,
            neuronpedia_source_set_id=self.cfg.np_set_name,
            neuronpedia_source_set_description=(
                self.cfg.neuronpedia_source_set_description
            ),
            hf_weights_repo_id=(
                self.cfg.neuronpedia_hf_weights_repo_id or self.cfg.sae_set
            ),
            hf_weights_path=(self.cfg.neuronpedia_hf_weights_path or self.cfg.sae_path),
            hook_point=derive_hook_point_from_hook_name(str(self.hook_name)),
            layer_num=self.layer,
            prompts_huggingface_dataset_path=self.cfg.huggingface_dataset_path,
            n_prompts_total=self.cfg.n_prompts_total,
            n_tokens_in_prompt=self.cfg.n_tokens_in_prompt,
            zero_out_bos_token=self.cfg.neuronpedia_zero_out_bos_token,
        )

        print(
            "\n========== Converting batch outputs to Neuronpedia "
            "bulk-import format =========="
        )
        export_neuronpedia_dashboards(export_cfg)


def main():
    def parse_json_dict_arg(raw_value: str | None, flag_name: str) -> dict[str, Any]:
        if not raw_value:
            return {}
        parsed = json.loads(raw_value)
        if not isinstance(parsed, dict):
            raise ValueError(f"{flag_name} must decode to a JSON object.")
        return parsed

    parser = argparse.ArgumentParser(description="Run Neuronpedia feature generation")
    parser.add_argument("--sae-set", required=True, help="SAE set name")
    parser.add_argument("--sae-path", required=True, help="Path to SAE")
    parser.add_argument("--np-set-name", required=True, help="Neuronpedia set name")
    parser.add_argument(
        "--np-sae-id-suffix",
        required=False,
        help=(
            "Additional suffix on Neuronpedia for the SAE ID. Goes after the SAE Set like so: "
            "__[np-sae-id-suffix]. Used for additional l0s, training steps, etc."
        ),
    )
    parser.add_argument(
        "--dataset-path",
        "--prompt-dataset-path",
        dest="dataset_path",
        default=None,
        help="Prompt dataset path or Hugging Face dataset identifier.",
    )
    parser.add_argument(
        "--prompt-dataset-mode",
        choices=("load_dataset", "load_from_disk", "legacy_jsonl"),
        default="load_dataset",
        help=(
            "Prompt dataset loading mode. Use load_dataset for builder-backed datasets, load_from_disk for local "
            "Dataset.save_to_disk() prompt caches, and legacy_jsonl only for deprecated legacy JSONL dashboard "
            "compatibility."
        ),
    )
    parser.add_argument(
        "--dataset-config-name",
        "--prompt-dataset-name",
        dest="dataset_config_name",
        default=None,
    )
    parser.add_argument(
        "--dataset-split", "--prompt-dataset-split", dest="dataset_split", default=None
    )
    parser.add_argument(
        "--dataset-text-field",
        "--prompt-dataset-text-field",
        dest="dataset_text_field",
        default=None,
    )
    parser.add_argument("--prompt-dataset-data-files", nargs="+", default=None)
    parser.add_argument("--prompt-dataset-data-dir", default=None)
    parser.add_argument("--prompt-dataset-metadata-path", default=None)
    parser.add_argument(
        "--prompt-dataset-trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--pretokenized-dataset-path",
        default=None,
        help="Path to a local HuggingFace dataset saved with an input_ids column.",
    )
    parser.add_argument(
        "--shared-tokens-file",
        default=None,
        help=(
            "Optional path to a shared precomputed tokens_*.pt file staged into each layer output directory. "
            "If omitted and --pretokenized-dataset-path is provided, the runner will generate the tokens and "
            "effective-length sidecars in the pretokenized dataset directory."
        ),
    )
    parser.add_argument(
        "--deduplicate-shared-prompt-tokens",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Whether runner-generated shared prompt sidecars deduplicate identical token rows before truncating to "
            "n_prompts_total."
        ),
    )
    parser.add_argument(
        "--strict-shared-prompt-count",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Require runner-generated shared prompt sidecars to materialize exactly n_prompts_total rows, raising if "
            "deduplication or dataset shortfall leaves fewer rows."
        ),
    )
    parser.add_argument(
        "--prompt-bucket-schedule-file",
        default=None,
        help=(
            "Optional path to a selected_bucket_configs.json artifact. When paired with --shared-tokens-file, "
            "the runner trims model forwards per prompt-length bucket while preserving one packaging pass."
        ),
    )
    parser.add_argument(
        "--auto-prompt-bucket-schedule",
        action="store_true",
        help=(
            "Derive the prompt-length schedule from the staged effective-length sidecar instead of requiring an "
            "external selected_bucket_configs.json file. When paired with --pretokenized-dataset-path, the sidecar "
            "can be generated automatically if it does not already exist."
        ),
    )
    parser.add_argument(
        "--prompt-bucket-ceilings",
        default=None,
        help=(
            "Optional comma-separated inclusive prompt-length ceilings for auto prompt bucketing. When omitted, the "
            "runner derives ceilings from staged effective-length quantiles plus the maximum observed effective "
            "length."
        ),
    )
    parser.add_argument(
        "--prompt-bucket-scale-limit",
        type=float,
        default=DEFAULT_PROMPT_BUCKET_SCALE_LIMIT,
        help="Maximum multiplicative prompt-batch scale used when auto prompt bucketing shorter prompt ranges.",
    )
    parser.add_argument(
        "--prompt-primary-acts-scale-limit",
        type=float,
        default=DEFAULT_PROMPT_PRIMARY_ACTS_SCALE_LIMIT,
        help=(
            "Maximum multiplicative primary-acts scale used when auto prompt bucketing shorter prompt ranges."
        ),
    )
    parser.add_argument(
        "--prompt-batch-size-round-to",
        type=int,
        default=DEFAULT_PROMPT_BATCH_SIZE_ROUND_TO,
        help="Round auto prompt-bucket batch sizes down to this multiple.",
    )
    parser.add_argument(
        "--dataset-streaming",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to stream the prompt dataset when loading it through datasets.load_dataset().",
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
        "--primary-acts-batch-size",
        type=int,
        default=None,
        help=(
            "Optional internal activation-capture chunk size. Keeps --n-prompts-in-forward-pass as the logical "
            "dashboard batch while splitting model forwards into smaller chunks for low-memory GPUs."
        ),
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
        "--no-shuffle-tokens",
        action="store_false",
        dest="shuffle_tokens",
        help="Don't shuffle tokens",
    )
    parser.add_argument(
        "--from-local-sae", action="store_true", help="Load SAE from local path"
    )
    parser.add_argument(
        "--hf-model-path",
        type=str,
        default=None,
        help="Optional: Path to custom HuggingFace model to use instead of default weights",
    )
    parser.add_argument(
        "--model-wrapper",
        choices=("hooked", "bridge"),
        default="hooked",
        help="Model wrapper to use for dashboard generation.",
    )
    parser.add_argument(
        "--bridge-enable-compatibility-mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable TransformerBridge compatibility mode so legacy hook aliases remain available.",
    )
    parser.add_argument(
        "--bridge-compatibility-mode-kwargs-json",
        type=str,
        default=None,
        help="JSON object of kwargs passed to TransformerBridge.enable_compatibility_mode().",
    )
    parser.add_argument(
        "--log-resource-snapshots",
        action="store_true",
        help="Emit simple RSS and CUDA memory snapshots at key runner stages.",
    )
    parser.add_argument(
        "--log-hook-aliases",
        action="store_true",
        help="Emit hook alias summary information to debug HookedTransformer versus TransformerBridge migration.",
    )
    parser.add_argument(
        "--log-performance",
        action="store_true",
        help="Emit per-batch and per-stage wall-clock, CPU, CUDA, and process I/O timing diagnostics.",
    )
    parser.add_argument(
        "--profile-rolling-substages",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Emit nested rolling-correlation substage timings and runtime metrics. "
            "Disabled by default so normal timing runs keep only the aggregate rolling stage."
        ),
    )
    parser.add_argument(
        "--converter-input-artifact-dir",
        default=None,
        help=(
            "Optional directory where per-batch converter-input snapshots are written for "
            "offline full-converter timing."
        ),
    )
    parser.add_argument(
        "--sequence-replay-artifact-dir",
        default=None,
        help=(
            "Optional directory where per-batch sequence replay bundles are written for "
            "offline get_indices_dict(...) replay."
        ),
    )
    parser.add_argument(
        "--cleanup-each-minibatch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Run gc.collect() and torch.cuda.empty_cache() after each activation minibatch. "
            "Disabled by default because it slows benchmark generation."
        ),
    )
    parser.add_argument(
        "--correlation-accumulation-device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Policy for correlation accumulator placement.",
    )
    parser.add_argument(
        "--rolling-coefficient-num-threads",
        type=int,
        default=None,
        help=(
            "Optional torch intra-op thread count override applied only during rolling correlation updates. "
            "Leave unset to use the process default."
        ),
    )
    parser.add_argument(
        "--activation-significance-floor",
        type=float,
        default=0.0,
        help="Minimum activation value to consider significant. Values <= floor are excluded from histograms.",
    )
    parser.add_argument(
        "--feature-statistics-backend",
        choices=("object", "arrow"),
        default="arrow",
        help="Backend for columnar feature statistics packaging.",
    )
    parser.add_argument(
        "--logits-histogram-backend",
        choices=("object", "arrow"),
        default="arrow",
        help="Backend for columnar logits histogram packaging.",
    )
    parser.add_argument(
        "--activation-histogram-backend",
        choices=("torch",),
        default="torch",
        help="Backend for positive-only activation histogram packaging in columnar mode.",
    )
    parser.add_argument(
        "--defer-component-construction",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Avoid rebuilding the legacy nested component graph when columnar output is requested.",
    )
    parser.add_argument(
        "--sequence-selection-backend",
        choices=("legacy", "columnar_gpu"),
        default="legacy",
        help="Sequence candidate-selection backend.",
    )
    parser.add_argument(
        "--sequence-top-acts-positive-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Opt-in (columnar backend only): restrict TOP-ACTIVATIONS candidates to strictly "
            "positive activations instead of zero/negative tie-fill."
        ),
    )
    parser.add_argument(
        "--sequence-dedup-across-groups",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Opt-in (columnar backend only): exclude already-selected coordinates from later "
            "group sampling so a sequence position appears in at most one group."
        ),
    )
    parser.add_argument(
        "--sequence-skip-dead-features",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Opt-in (columnar backend only): emit no sequence rows for features with no "
            "positive activation on the eval set instead of degenerate zero-row groups."
        ),
    )
    parser.add_argument(
        "--columnar-max-device-staged-acts-bytes",
        type=int,
        default=None,
        help=(
            "Opt-in (columnar path only): byte budget capping device retention/staging of the "
            "(prompts, seq, feats) activation matrix (default: the historical fixed 4 GiB; 0 "
            "forces host staging). Outputs are bit-identical; only peak GPU memory and speed move."
        ),
    )
    parser.add_argument(
        "--columnar-row-chunk-size",
        type=int,
        default=None,
        help=(
            "Opt-in (columnar path only): feature-row chunk size for the arrow statistics/"
            "histogram packaging loops (defaults 256/128) and the batched sequence selector "
            "(default 64), bounding density/token-count-scaled per-chunk GPU transients on "
            "dense layers. Outputs are bit-identical."
        ),
    )
    parser.add_argument(
        "--sequence-half-open-interval-bins",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Opt-in: numpy-histogram-style interval membership ([lower, upper), highest "
            "interval closed) so each value belongs to exactly one quantile interval. "
            "Applies to the in-tree legacy and columnar selectors."
        ),
    )
    parser.add_argument(
        "--logits-table-mask-token-pattern",
        type=str,
        default=None,
        help=(
            "Optional regex over token strings; matching vocab rows are excluded from logits "
            "tables on the columnar path (e.g. '^<(0x[0-9A-Fa-f]{2}|unused\\d+)>$')."
        ),
    )
    parser.add_argument(
        "--dashboard-output-format",
        choices=("legacy_json", "columnar"),
        default="legacy_json",
        help="Dashboard output format.",
    )
    parser.add_argument(
        "--columnar-artifact-format",
        choices=("arrow", "parquet"),
        default="arrow",
        help="On-disk format for columnar dashboard tables.",
    )
    parser.add_argument(
        "--columnar-write-page-index",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Write a Parquet page index (default: on). Required for page-granular range reads when "
            "artifacts are streamed over HTTP. It cannot be added to existing files, so disabling "
            "it means a later fix requires regenerating the whole corpus."
        ),
    )
    parser.add_argument(
        "--columnar-parquet-row-group-size",
        type=int,
        default=DEFAULT_PARQUET_ROW_GROUP_SIZE,
        help=(
            f"Rows per Parquet row group (default: {DEFAULT_PARQUET_ROW_GROUP_SIZE}). Readers prune "
            "at ROW GROUP granularity, so one row group per file makes fetching a single feature "
            "cost the whole file -- a page index does not change that. Like the page index it is "
            "fixed at write time. Pass 0 for the pyarrow default (a single row group)."
        ),
    )
    parser.add_argument(
        "--columnar-emit-sequence-rows",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Emit raw sequence_rows tables in columnar mode.",
    )
    parser.add_argument(
        "--columnar-emit-activation-rows",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Emit semantic activation_rows tables in columnar mode.",
    )
    parser.add_argument(
        "--columnar-emit-activation-copy-rows",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Emit Neuronpedia Activation COPY-shaped activation_copy_rows in columnar mode.",
    )
    parser.add_argument(
        "--columnar-activation-copy-model-id",
        type=str,
        default=None,
        help="Optional modelId override used when emitting activation_copy_rows.",
    )
    parser.add_argument(
        "--overlap-batch-packaging",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Overlap each columnar batch's CPU packaging tail and artifact writes with "
            "the next batch's forward/encode (single ordered background writer; "
            "batch-level resume semantics unchanged)."
        ),
    )
    parser.add_argument(
        "--torch-profile",
        action="store_true",
        help="Capture a torch.profiler Chrome trace for each generated batch.",
    )
    parser.add_argument(
        "--torch-profile-dir",
        default=None,
        help="Optional directory for torch.profiler trace files. Defaults to an output-local torch_profiles directory.",
    )
    parser.add_argument(
        "--use-cached-activations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse cached model activations across batches and reruns when available.",
    )
    parser.add_argument(
        "--prefix-str",
        type=str,
        default=None,
        help="Optional string to prepend to each prompt. Example: --prefix-str '<|im_start|>user\n'",
    )
    parser.add_argument(
        "--no-prepend-bos",
        action="store_false",
        dest="prepend_bos",
        default=None,
        help="Don't prepend BOS token to sequences (overrides SAE default)",
    )
    parser.add_argument(
        "--prepend-bos",
        action="store_true",
        dest="prepend_bos",
        default=None,
        help="Prepend BOS token to sequences (overrides SAE default)",
    )
    parser.add_argument(
        "--use-transcoder",
        action="store_true",
        help="If set, load a Transcoder instead of a standard SAE",
    )
    parser.add_argument(
        "--use-skip-transcoder",
        action="store_true",
        help="If set, load a SkipTranscoder instead of a Transcoder/SAE",
    )
    parser.add_argument(
        "--use-clt",
        action="store_true",
        help="If set, load a CrossLayerTranscoder instead of a standard SAE/Transcoder",
    )
    parser.add_argument(
        "--clt-layer-idx",
        type=int,
        default=None,
        help="Layer index to use for CLT encoder (required if --use-clt)",
    )
    parser.add_argument(
        "--clt-dtype",
        type=str,
        default="",
        help="Optional override for CLT data type (e.g., 'float16')",
    )
    parser.add_argument(
        "--clt-weights-filename",
        type=str,
        default="",
        help=(
            "Filename of the CLT weights file (supports .safetensors / .pt). If omitted, "
            "script will search for a suitable file automatically."
        ),
    )
    parser.add_argument(
        "--sae-loader",
        "--sae-converter-name",
        dest="sae_converter_name",
        type=str,
        default=None,
        help=(
            "Name of the sae_lens loader/converter to use when loading an SAE "
            "from HuggingFace (passed to SAE.from_pretrained's `converter` arg). "
            "Accepts short registry names from sae_lens' "
            "NAMED_PRETRAINED_SAE_LOADERS (e.g. 'dictionary_learning_1', "
            "'gemma_2', 'sparsify', 'connor_rob_hook_z') as well as the full "
            "function name (e.g. 'dictionary_learning_sae_huggingface_loader_1'). "
            "If omitted, sae_lens infers the loader from the release."
        ),
    )
    parser.add_argument(
        "--ignore-high-activation-norm-multiple",
        type=float,
        default=None,
        help=(
            "If set, filter out activations at token positions whose hidden-state "
            "norm exceeds `median_norm * MULTIPLE` (computed per forward-pass "
            "minibatch). Useful for models like Qwen that have random high-norm "
            "activation sinks. A typical value is 10. Defaults to no filtering."
        ),
    )
    parser.add_argument(
        "--free-unused-model-layers",
        action="store_true",
        help=(
            "If set, replace transformer blocks above the SAE's hook layer "
            "with nn.Identity() after loading to free VRAM. The forward pass "
            "already stops at the hook layer, so those blocks are unused. "
            "W_U / ln_final are preserved for logit-direction calculations. "
            "Works uniformly for any TransformerLens-supported architecture."
        ),
    )
    parser.add_argument(
        "--capture-hook-name",
        type=str,
        default=None,
        help="Capture activations at this hook instead of the one the SAE's metadata declares. Use "
        "where the declared TransformerLens name denotes a different tensor from the one the SAE was "
        "trained on (e.g. 'blocks.5.ln2.hook_out' for a Gemma Scope 2 transcoder on a TransformerBridge, "
        "whose metadata declares the pre-norm 'blocks.5.hook_mlp_in'). Leave unset to trust the SAE.",
    )
    parser.add_argument(
        "--huggingface",
        action="store_true",
        help="Use HuggingFace Transformers directly instead of TransformerLens. "
        "This enables support for models not available in TransformerLens.",
    )
    parser.add_argument(
        "--layer-num",
        type=int,
        default=None,
        help=(
            "Explicit layer index for the SAE's hook. Overrides any value "
            "auto-detected from the SAE config. Required when the SAE's "
            "hook_name does not match the TransformerLens "
            "'blocks.<N>.<hook>' pattern (e.g. HuggingFace-style hook_names "
            "like 'model.language_model.layers.17')."
        ),
    )

    # ------------------------------------------------------------------
    # Neuronpedia bulk-import export flags
    # ------------------------------------------------------------------
    parser.add_argument(
        "--output-neuronpedia-exports",
        action="store_true",
        help=(
            "After generating batch-*.json files, also convert them to the "
            "Neuronpedia bulk-import directory layout (release.jsonl, "
            "model.jsonl, sourceset.jsonl, source.jsonl, plus gzipped "
            "per-batch features and activations). Requires "
            "--neuronpedia-creator-name, --neuronpedia-release-id, "
            "--neuronpedia-release-title, --neuronpedia-release-url, "
            "--neuronpedia-source-set-description, --neuronpedia-model-name, "
            "and --np-set-name."
        ),
    )
    parser.add_argument(
        "--neuronpedia-exports-dir",
        type=str,
        default=None,
        help=(
            "Where to write Neuronpedia bulk-import exports. Defaults to "
            "'<output-dir>/../neuronpedia_exports'. Files end up under "
            "{exports-dir}/{neuronpedia-model-name}/{layer-source_set}/."
        ),
    )
    parser.add_argument(
        "--neuronpedia-creator-name",
        type=str,
        default=None,
        help="Display name of the creator (e.g. your team / org name).",
    )
    parser.add_argument(
        "--neuronpedia-creator-id",
        type=str,
        default=None,
        help=(
            "Neuronpedia creator ID. If omitted, falls back to the "
            "DEFAULT_CREATOR_ID env var, then to a hardcoded default."
        ),
    )
    parser.add_argument(
        "--neuronpedia-release-id",
        type=str,
        default=None,
        help=(
            "Release ID (alphanumeric + dashes). The release is served at "
            "https://[neuronpedia_domain]/[release_id]."
        ),
    )
    parser.add_argument(
        "--neuronpedia-release-title",
        type=str,
        default=None,
        help="Human-readable release title (e.g. 'Gemma 4 SAEs').",
    )
    parser.add_argument(
        "--neuronpedia-release-url",
        type=str,
        default=None,
        help="URL for the release (paper / HuggingFace repo / etc).",
    )
    parser.add_argument(
        "--neuronpedia-source-set-description",
        type=str,
        default=None,
        help=(
            "Human-readable source-set description shown on Neuronpedia "
            "(e.g. 'Residual Stream - 65k')."
        ),
    )
    parser.add_argument(
        "--neuronpedia-model-name",
        type=str,
        default=None,
        help=(
            "Override the model name written to the export. Used as Model.id, "
            "the export '{model_name}/' subdirectory, and modelId on every "
            "feature/activation/source/sourceset row. Required with "
            "--output-neuronpedia-exports."
        ),
    )
    parser.add_argument(
        "--neuronpedia-hf-weights-repo-id",
        type=str,
        default=None,
        help=("HuggingFace repo ID for the SAE weights. Defaults to --sae-set."),
    )
    parser.add_argument(
        "--neuronpedia-hf-weights-path",
        type=str,
        default=None,
        help=("HuggingFace path to the SAE weights folder. Defaults to " "--sae-path."),
    )
    parser.add_argument(
        "--neuronpedia-zero-out-bos-token",
        action="store_true",
        help=(
            "Zero out activations on '<bos>' / '<|endoftext|>' tokens in the "
            "exported activations. Useful for models like Gemma 2 that "
            "weren't trained with a BOS token."
        ),
    )

    args = parser.parse_args()

    if args.dataset_path is None and args.pretokenized_dataset_path is None:
        parser.error(
            "Provide --dataset-path/--prompt-dataset-path or --pretokenized-dataset-path."
        )

    prompt_bucket_ceilings: tuple[int, ...] = ()
    if args.prompt_bucket_ceilings:
        prompt_bucket_ceilings = tuple(
            int(value.strip())
            for value in str(args.prompt_bucket_ceilings).split(",")
            if value.strip()
        )

    cfg = NeuronpediaRunnerConfig(
        sae_set=args.sae_set,
        sae_path=args.sae_path,
        np_set_name=args.np_set_name,
        np_sae_id_suffix=args.np_sae_id_suffix,
        from_local_sae=args.from_local_sae,
        huggingface_dataset_path=args.dataset_path or "",
        huggingface_dataset_config_name=args.dataset_config_name,
        huggingface_dataset_split=args.dataset_split,
        huggingface_dataset_text_field=args.dataset_text_field,
        prompt_dataset_mode=args.prompt_dataset_mode,
        prompt_dataset_path=args.dataset_path,
        prompt_dataset_name=args.dataset_config_name,
        prompt_dataset_split=args.dataset_split,
        prompt_dataset_text_field=args.dataset_text_field,
        prompt_dataset_data_files=tuple(args.prompt_dataset_data_files or ()),
        prompt_dataset_data_dir=args.prompt_dataset_data_dir,
        prompt_dataset_metadata_path=args.prompt_dataset_metadata_path,
        prompt_dataset_trust_remote_code=args.prompt_dataset_trust_remote_code,
        pretokenized_dataset_path=args.pretokenized_dataset_path,
        shared_tokens_file=args.shared_tokens_file,
        deduplicate_shared_prompt_tokens=args.deduplicate_shared_prompt_tokens,
        strict_shared_prompt_count=args.strict_shared_prompt_count,
        prompt_bucket_schedule_file=args.prompt_bucket_schedule_file,
        auto_prompt_bucket_schedule=args.auto_prompt_bucket_schedule,
        prompt_bucket_ceilings=prompt_bucket_ceilings,
        prompt_bucket_scale_limit=args.prompt_bucket_scale_limit,
        prompt_primary_acts_scale_limit=args.prompt_primary_acts_scale_limit,
        prompt_batch_size_round_to=args.prompt_batch_size_round_to,
        dataset_streaming=args.dataset_streaming,
        sae_dtype=args.sae_dtype,
        model_dtype=args.model_dtype,
        outputs_dir=args.output_dir,
        sparsity_threshold=args.sparsity_threshold,
        prefix_str=args.prefix_str,
        prepend_bos=args.prepend_bos,
        n_prompts_total=args.n_prompts,
        n_tokens_in_prompt=args.n_tokens_in_prompt,
        n_prompts_in_forward_pass=args.n_prompts_in_forward_pass,
        primary_acts_batch_size=args.primary_acts_batch_size,
        n_features_at_a_time=args.n_features_per_batch,
        start_batch=args.start_batch,
        end_batch=args.end_batch,
        use_wandb=args.use_wandb,
        shuffle_tokens=args.shuffle_tokens,
        hf_model_path=args.hf_model_path,
        model_wrapper=args.model_wrapper,
        bridge_enable_compatibility_mode=args.bridge_enable_compatibility_mode,
        bridge_compatibility_mode_kwargs=parse_json_dict_arg(
            args.bridge_compatibility_mode_kwargs_json,
            "--bridge-compatibility-mode-kwargs-json",
        )
        or dict(DEFAULT_BRIDGE_COMPATIBILITY_KWARGS),
        log_resource_snapshots=args.log_resource_snapshots,
        log_hook_aliases=args.log_hook_aliases,
        log_performance=args.log_performance,
        profile_rolling_substages=args.profile_rolling_substages,
        cleanup_each_minibatch=args.cleanup_each_minibatch,
        correlation_accumulation_device=args.correlation_accumulation_device,
        rolling_coefficient_num_threads=args.rolling_coefficient_num_threads,
        activation_significance_floor=args.activation_significance_floor,
        converter_input_artifact_dir=args.converter_input_artifact_dir,
        sequence_replay_artifact_dir=args.sequence_replay_artifact_dir,
        feature_statistics_backend=args.feature_statistics_backend,
        logits_histogram_backend=args.logits_histogram_backend,
        activation_histogram_backend=args.activation_histogram_backend,
        defer_component_construction=args.defer_component_construction,
        sequence_selection_backend=args.sequence_selection_backend,
        sequence_top_acts_positive_only=args.sequence_top_acts_positive_only,
        sequence_dedup_across_groups=args.sequence_dedup_across_groups,
        sequence_skip_dead_features=args.sequence_skip_dead_features,
        sequence_half_open_interval_bins=args.sequence_half_open_interval_bins,
        columnar_max_device_staged_acts_bytes=args.columnar_max_device_staged_acts_bytes,
        columnar_row_chunk_size=args.columnar_row_chunk_size,
        logits_table_mask_token_pattern=args.logits_table_mask_token_pattern,
        dashboard_output_format=args.dashboard_output_format,
        columnar_artifact_format=args.columnar_artifact_format,
        columnar_write_page_index=args.columnar_write_page_index,
        columnar_parquet_row_group_size=args.columnar_parquet_row_group_size or None,
        columnar_emit_sequence_rows=args.columnar_emit_sequence_rows,
        columnar_emit_activation_rows=args.columnar_emit_activation_rows,
        overlap_batch_packaging=args.overlap_batch_packaging,
        columnar_emit_activation_copy_rows=args.columnar_emit_activation_copy_rows,
        columnar_activation_copy_model_id=args.columnar_activation_copy_model_id,
        torch_profile=args.torch_profile,
        torch_profile_dir=args.torch_profile_dir,
        use_cached_activations=args.use_cached_activations,
        use_transcoder=args.use_transcoder,
        use_skip_transcoder=args.use_skip_transcoder,
        use_clt=args.use_clt,
        clt_layer_idx=args.clt_layer_idx,
        clt_dtype=args.clt_dtype,
        clt_weights_filename=args.clt_weights_filename,
        sae_converter_name=args.sae_converter_name,
        use_huggingface=args.huggingface,
        capture_hook_name=args.capture_hook_name,
        layer=args.layer_num,
        ignore_high_activation_norm_multiple=args.ignore_high_activation_norm_multiple,
        free_unused_model_layers=args.free_unused_model_layers,
        output_neuronpedia_exports=args.output_neuronpedia_exports,
        neuronpedia_exports_dir=args.neuronpedia_exports_dir,
        neuronpedia_creator_name=args.neuronpedia_creator_name,
        neuronpedia_creator_id=args.neuronpedia_creator_id,
        neuronpedia_release_id=args.neuronpedia_release_id,
        neuronpedia_release_title=args.neuronpedia_release_title,
        neuronpedia_release_url=args.neuronpedia_release_url,
        neuronpedia_source_set_description=args.neuronpedia_source_set_description,
        neuronpedia_model_name=args.neuronpedia_model_name,
        neuronpedia_hf_weights_repo_id=args.neuronpedia_hf_weights_repo_id,
        neuronpedia_hf_weights_path=args.neuronpedia_hf_weights_path,
        neuronpedia_zero_out_bos_token=args.neuronpedia_zero_out_bos_token,
    )

    runner = NeuronpediaRunner(cfg)
    runner.run()


if __name__ == "__main__":
    main()
