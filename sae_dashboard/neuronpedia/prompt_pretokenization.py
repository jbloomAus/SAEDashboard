from __future__ import annotations

import argparse
import importlib
import json
import sys
import warnings
from datetime import datetime, timezone
from time import perf_counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, Protocol, cast, runtime_checkable

import torch
from datasets import Dataset, DatasetDict, IterableDataset, load_dataset
from sae_lens.config import PretokenizeRunnerConfig, special_token
from sae_lens.pretokenize_runner import (
    PretokenizedDatasetMetadata,
    get_special_token_from_cfg,
    metadata_from_config,
    pretokenize_dataset,
)
from sae_lens.tokenization_and_batching import concat_and_batch_sequences
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from sae_dashboard.neuronpedia.prompt_datasets import (
    write_pretokenized_prompt_artifacts,
)

WindowingMode = Literal[
    "concatenate",
    "filter-truncate",
    "max-prompt-pad",
    "fixed-context-pad",
]

WINDOWING_MODES: tuple[WindowingMode, ...] = (
    "concatenate",
    "filter-truncate",
    "max-prompt-pad",
    "fixed-context-pad",
)
PACKED_WINDOWING_MODES: tuple[WindowingMode, ...] = ("concatenate", "filter-truncate")
EXAMPLE_ALIGNED_WINDOWING_MODES: tuple[WindowingMode, ...] = (
    "max-prompt-pad",
    "fixed-context-pad",
)
DEFAULT_WINDOWING_MODE: WindowingMode = "concatenate"


@dataclass(frozen=True)
class PretokenizationResult:
    """Materialized prompt artifact plus windowing metadata.

    ``effective_context_size`` is the saved row width after windowing.
    ``prompt_lengths`` records the per-prompt pre-padding lengths only for example-aligned modes whose rows still map
    1:1 to prompts. Packed modes return ``None`` because their rows no longer align with individual prompts.
    """

    tokenized_dataset: Dataset
    effective_context_size: int | None = None
    prompt_lengths: tuple[int, ...] | None = None
    windowing_mode: WindowingMode | None = None
    disable_concat_sequences: bool | None = None
    pad_token_id: int | None = None


@runtime_checkable
class DashboardPretokenizationModule(Protocol):
    def pretokenize_custom_dataset(
        self,
        dataset: Dataset | IterableDataset,
        tokenizer: PreTrainedTokenizerBase,
        cfg: PretokenizeRunnerConfig,
        settings: Any | None,
    ) -> PretokenizationResult: ...


def add_windowing_arguments(
    parser: argparse.ArgumentParser,
    *,
    default: WindowingMode = DEFAULT_WINDOWING_MODE,
) -> None:
    parser.add_argument(
        "--windowing-mode",
        choices=WINDOWING_MODES,
        default=default,
        help=(
            "Prompt windowing policy. 'concatenate' preserves SAELens packed-window behavior; 'filter-truncate' "
            "keeps only prompts that already meet context_size and truncates them to that width; 'max-prompt-pad' "
            "and 'fixed-context-pad' preserve prompt-aligned rows and emit attention_mask sidecars for dashboard "
            "bucketing."
        ),
    )


def set_default_windowing_mode(
    parser: argparse.ArgumentParser,
    *,
    default: WindowingMode,
) -> None:
    parser.set_defaults(windowing_mode=default)


def windowing_mode_disables_concat_sequences(windowing_mode: WindowingMode) -> bool:
    return windowing_mode != "concatenate"


def is_example_aligned_windowing(windowing_mode: WindowingMode) -> bool:
    return windowing_mode in EXAMPLE_ALIGNED_WINDOWING_MODES


def windowing_mode_supports_streaming(windowing_mode: WindowingMode) -> bool:
    return windowing_mode in PACKED_WINDOWING_MODES


def build_base_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pretokenize datasets for Neuronpedia dashboard generation."
    )
    parser.add_argument(
        "--custom-dataset-module",
        help=(
            "Optional module for task-specific prompt rendering or metadata. Single-column datasets can use the "
            "built-in tokenization path directly, including example-aligned windowing modes."
        ),
    )
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--dataset-name", "--dataset-config-name", dest="dataset_name")
    parser.add_argument("--dataset-split", "--split", dest="split", default="train")
    parser.add_argument(
        "--dataset-trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--data-files", nargs="+")
    parser.add_argument("--data-dir")
    parser.add_argument("--tokenizer-name", required=True)
    parser.add_argument("--context-size", type=int, default=128)
    parser.add_argument("--column-name", default="text")
    parser.add_argument(
        "--use-chat-formatting",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--shuffle",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--streaming",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--num-proc", type=int, default=4)
    parser.add_argument("--pretokenize-batch-size", type=int, default=1000)
    parser.add_argument("--begin-batch-token", type=special_token, default="bos")
    parser.add_argument("--begin-sequence-token", type=special_token, default=None)
    parser.add_argument("--sequence-separator-token", type=special_token, default="bos")
    add_windowing_arguments(parser)
    parser.add_argument(
        "--disable-concat-sequences",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Advanced override for the packed SAELens pretokenizer. Windowing modes also control this value; all "
            "modes except 'concatenate' force it to True."
        ),
    )
    parser.add_argument("--save-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--legacy-output-dir",
        type=Path,
        help=(
            "Optional deprecated load_dataset-compatible JSONL export directory for legacy Neuronpedia runners. "
            "Writes <split>.jsonl plus sae_lens.json."
        ),
    )
    parser.add_argument("--hf-repo-id")
    parser.add_argument("--hf-num-shards", type=int, default=64)
    parser.add_argument("--hf-revision", default="main")
    parser.add_argument(
        "--hf-is-private-repo",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--max-tokenized-rows", type=int)
    parser.add_argument("--force", action="store_true")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    base_parser = build_base_parser()
    known_args, _ = base_parser.parse_known_args(argv)
    custom_module = None
    if known_args.custom_dataset_module:
        custom_module = load_custom_pretokenization_module(
            known_args.custom_dataset_module
        )
    maybe_configure_custom_parser(custom_module, base_parser)
    args = base_parser.parse_args(argv)
    args.custom_module = custom_module
    return args


def build_pretokenize_config(args: argparse.Namespace) -> PretokenizeRunnerConfig:
    save_path = args.save_path or args.output_dir
    windowing_mode = cast(WindowingMode, args.windowing_mode)
    disable_concat_sequences = (
        args.disable_concat_sequences
        or windowing_mode_disables_concat_sequences(windowing_mode)
    )
    return PretokenizeRunnerConfig(
        tokenizer_name=args.tokenizer_name,
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name,
        dataset_trust_remote_code=args.dataset_trust_remote_code,
        split=args.split,
        data_files=args.data_files,
        data_dir=args.data_dir,
        num_proc=args.num_proc,
        context_size=args.context_size,
        column_name=args.column_name,
        use_chat_formatting=args.use_chat_formatting,
        shuffle=args.shuffle,
        seed=args.seed,
        streaming=args.streaming,
        pretokenize_batch_size=args.pretokenize_batch_size,
        begin_batch_token=args.begin_batch_token,
        begin_sequence_token=args.begin_sequence_token,
        sequence_separator_token=args.sequence_separator_token,
        disable_concat_sequences=disable_concat_sequences,
        save_path=str(save_path) if save_path is not None else None,
        hf_repo_id=args.hf_repo_id,
        hf_num_shards=args.hf_num_shards,
        hf_revision=args.hf_revision,
        hf_is_private_repo=args.hf_is_private_repo,
    )


def run_dashboard_pretokenization(
    args: argparse.Namespace,
) -> tuple[PretokenizationResult, PretokenizeRunnerConfig, dict[str, Any]]:
    cfg = build_pretokenize_config(args)
    dataset = load_dashboard_dataset(cfg, max_rows=args.max_rows)
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
    tokenizer.model_max_length = sys.maxsize
    windowing_mode = cast(WindowingMode, args.windowing_mode)

    if args.custom_module is not None:
        settings = maybe_load_custom_pretokenization_settings(args.custom_module, args)
        result = args.custom_module.pretokenize_custom_dataset(
            dataset, tokenizer, cfg, settings
        )
        custom_metadata = maybe_build_custom_metadata(
            args.custom_module,
            args=args,
            settings=settings,
            result=result,
            tokenizer=tokenizer,
            cfg=cfg,
        )
    else:
        if is_example_aligned_windowing(windowing_mode):
            result = pretokenize_prompt_token_sequences(
                _iter_dataset_prompt_token_sequences(
                    dataset, tokenizer=tokenizer, cfg=cfg
                ),
                tokenizer=tokenizer,
                cfg=cfg,
                windowing_mode=windowing_mode,
            )
        else:
            tokenized_dataset = pretokenize_dataset(
                cast(Dataset, dataset), tokenizer, cfg
            )
            result = PretokenizationResult(
                tokenized_dataset=materialize_tokenized_dataset(
                    tokenized_dataset,
                    max_tokenized_rows=args.max_tokenized_rows,
                ),
                effective_context_size=cfg.context_size,
                prompt_lengths=None,
                windowing_mode=windowing_mode,
                disable_concat_sequences=cfg.disable_concat_sequences,
                pad_token_id=None,
            )
        custom_metadata = {}

    if (
        args.max_tokenized_rows is not None
        and len(result.tokenized_dataset) > args.max_tokenized_rows
    ):
        result = replace(
            result,
            tokenized_dataset=result.tokenized_dataset.select(
                range(args.max_tokenized_rows)
            ),
            prompt_lengths=(
                result.prompt_lengths[: args.max_tokenized_rows]
                if result.prompt_lengths is not None
                else None
            ),
        )

    metadata_cfg = replace(
        cfg,
        context_size=result.effective_context_size or cfg.context_size,
        disable_concat_sequences=(
            result.disable_concat_sequences
            if result.disable_concat_sequences is not None
            else cfg.disable_concat_sequences
        ),
    )
    metadata = build_dashboard_metadata(
        metadata_cfg,
        result=result,
        tokenizer=tokenizer,
        custom_metadata=custom_metadata,
    )
    return result, metadata_cfg, metadata


def pretokenize_prompt_token_sequences(
    token_sequences: Iterable[torch.Tensor],
    *,
    tokenizer: PreTrainedTokenizerBase,
    cfg: PretokenizeRunnerConfig,
    windowing_mode: WindowingMode,
) -> PretokenizationResult:
    if is_example_aligned_windowing(windowing_mode):
        return _pretokenize_example_aligned_prompt_sequences(
            token_sequences,
            tokenizer=tokenizer,
            cfg=cfg,
            windowing_mode=windowing_mode,
        )
    return _pretokenize_packed_prompt_sequences(
        token_sequences,
        tokenizer=tokenizer,
        cfg=cfg,
        windowing_mode=windowing_mode,
    )


def build_windowing_metadata(
    result: PretokenizationResult,
    *,
    cfg: PretokenizeRunnerConfig,
) -> dict[str, Any]:
    prompt_lengths = result.prompt_lengths or ()
    windowing_mode = result.windowing_mode
    return {
        "rows": len(result.tokenized_dataset),
        "windowing_mode": windowing_mode,
        "prompt_windowing_family": (
            "example_aligned_pad_enabled"
            if windowing_mode is not None
            and is_example_aligned_windowing(windowing_mode)
            else "packed_legacy"
        ),
        "effective_context_size": result.effective_context_size or cfg.context_size,
        "prompt_lengths_available": result.prompt_lengths is not None,
        "prompt_length_min": min(prompt_lengths) if prompt_lengths else None,
        "prompt_length_max": max(prompt_lengths) if prompt_lengths else None,
        "prompt_length_mean": (
            sum(prompt_lengths) / len(prompt_lengths) if prompt_lengths else None
        ),
        "pad_token_id": result.pad_token_id,
        "disable_concat_sequences": (
            result.disable_concat_sequences
            if result.disable_concat_sequences is not None
            else cfg.disable_concat_sequences
        ),
        "streaming_supported": (
            windowing_mode_supports_streaming(windowing_mode)
            if windowing_mode is not None
            else None
        ),
    }


def load_dashboard_dataset(
    cfg: PretokenizeRunnerConfig,
    *,
    max_rows: int | None,
) -> Dataset | IterableDataset:
    dataset = load_dataset(  # type: ignore[call-overload]
        cfg.dataset_path,
        name=cfg.dataset_name,
        data_dir=cfg.data_dir,
        data_files=cfg.data_files,
        split=cfg.split,  # type: ignore[arg-type]
        streaming=cfg.streaming,  # type: ignore[arg-type]
        trust_remote_code=cfg.dataset_trust_remote_code,
    )
    if isinstance(dataset, DatasetDict):
        raise ValueError("Dataset has multiple splits. Must provide a 'split' param.")
    dataset = cast("Dataset | IterableDataset", dataset)
    if max_rows is None:
        return dataset
    if isinstance(dataset, Dataset):
        return dataset.select(range(min(len(dataset), max_rows)))
    return dataset.take(max_rows)


def _iter_dataset_prompt_token_sequences(
    dataset: Dataset | IterableDataset,
    *,
    tokenizer: PreTrainedTokenizerBase,
    cfg: PretokenizeRunnerConfig,
) -> Iterable[torch.Tensor]:
    for row in dataset:
        row_dict = dict(row)
        if cfg.column_name not in row_dict:
            raise ValueError(
                f"Column '{cfg.column_name}' was not present in prompt dataset row. "
                f"Available columns: {sorted(row_dict.keys())}."
            )
        yield _tokenize_prompt_value(
            row_dict[cfg.column_name],
            tokenizer=tokenizer,
            use_chat_formatting=cfg.use_chat_formatting,
        )


def persist_dashboard_dataset(
    dataset: Dataset,
    *,
    cfg: PretokenizeRunnerConfig,
    metadata: dict[str, Any],
    force: bool,
    legacy_output_dir: Path | None = None,
) -> None:
    write_pretokenized_prompt_artifacts(
        dataset,
        metadata=metadata,
        save_to_disk_path=cfg.save_path,
        legacy_output_dir=legacy_output_dir,
        split=str(cfg.split or "train"),
        force=force,
        hf_repo_id=cfg.hf_repo_id,
        hf_num_shards=cfg.hf_num_shards,
        hf_revision=cfg.hf_revision,
        hf_is_private_repo=cfg.hf_is_private_repo,
    )


def load_custom_pretokenization_module(module_path: str) -> ModuleType:
    return importlib.import_module(module_path)


def maybe_configure_custom_parser(
    module: ModuleType | None, parser: argparse.ArgumentParser
) -> None:
    configure_parser = getattr(module, "configure_parser", None)
    if callable(configure_parser):
        configure_parser(parser)


def maybe_load_custom_pretokenization_settings(
    module: ModuleType | None, args: argparse.Namespace
) -> Any | None:
    load_settings = getattr(module, "load_custom_pretokenization_settings", None)
    if callable(load_settings):
        return load_settings(args)
    return None


def maybe_build_custom_metadata(
    module: ModuleType | None,
    *,
    args: argparse.Namespace,
    settings: Any | None,
    result: PretokenizationResult,
    tokenizer: PreTrainedTokenizerBase,
    cfg: PretokenizeRunnerConfig,
) -> dict[str, Any]:
    build_metadata = getattr(module, "build_custom_metadata", None)
    if callable(build_metadata):
        return cast(
            "dict[str, Any]", build_metadata(args, settings, result, tokenizer, cfg)
        )
    return {}


def build_dashboard_metadata(
    cfg: PretokenizeRunnerConfig,
    *,
    result: PretokenizationResult | None = None,
    tokenizer: PreTrainedTokenizerBase | None = None,
    custom_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = asdict(metadata_from_config(cfg))
    if result is not None:
        metadata.update(build_windowing_metadata(result, cfg=cfg))
        if (
            result.pad_token_id is None
            and tokenizer is not None
            and result.prompt_lengths is not None
        ):
            metadata["pad_token_id"] = getattr(tokenizer, "pad_token_id", None)
    if custom_metadata:
        metadata["custom"] = custom_metadata
    return metadata


def materialize_tokenized_dataset(
    tokenized_dataset: Dataset | IterableDataset,
    *,
    max_tokenized_rows: int | None = None,
) -> Dataset:
    if isinstance(tokenized_dataset, Dataset):
        if (
            max_tokenized_rows is not None
            and len(tokenized_dataset) > max_tokenized_rows
        ):
            tokenized_dataset = tokenized_dataset.select(range(max_tokenized_rows))
        return ensure_torch_dataset_format(tokenized_dataset)

    rows: list[dict[str, Any]] = []
    for index, row in enumerate(tokenized_dataset):
        if max_tokenized_rows is not None and index >= max_tokenized_rows:
            break
        rows.append({key: _to_python_value(value) for key, value in row.items()})

    if not rows:
        return Dataset.from_dict({"input_ids": []})

    materialized = Dataset.from_list(rows)
    return ensure_torch_dataset_format(materialized)


def ensure_torch_dataset_format(dataset: Dataset) -> Dataset:
    tensor_columns = [
        column_name
        for column_name in ("input_ids", "tokens", "attention_mask")
        if column_name in dataset.column_names
    ]
    if tensor_columns:
        dataset.set_format(type="torch", columns=tensor_columns)
    return dataset


def metadata_as_upstream_type(metadata: dict[str, Any]) -> PretokenizedDatasetMetadata:
    upstream_fields = {field.name for field in fields(PretokenizedDatasetMetadata)}
    upstream_metadata = {
        key: value for key, value in metadata.items() if key in upstream_fields
    }
    return PretokenizedDatasetMetadata(**upstream_metadata)


def as_1d_token_tensor(tokens: Any) -> torch.Tensor:
    if isinstance(tokens, Mapping) and "input_ids" in tokens:
        tokens = tokens["input_ids"]
    if isinstance(tokens, torch.Tensor):
        return tokens[0] if tokens.ndim == 2 else tokens
    token_tensor = torch.tensor(tokens, dtype=torch.long)
    return token_tensor[0] if token_tensor.ndim == 2 else token_tensor


def _tokenize_prompt_value(
    value: Any,
    *,
    tokenizer: PreTrainedTokenizerBase,
    use_chat_formatting: bool,
) -> torch.Tensor:
    if use_chat_formatting:
        if isinstance(value, str):
            warnings.warn(
                "use_chat_formatting is True but column contains strings. Wrapping as single user messages.",
                stacklevel=2,
            )
            value = [{"role": "user", "content": value}]
        return as_1d_token_tensor(
            tokenizer.apply_chat_template(
                value,
                tokenize=True,
                return_tensors="pt",
            )
        )
    if isinstance(value, str):
        return as_1d_token_tensor(tokenizer.encode(value, return_tensors="pt"))
    return as_1d_token_tensor(value)


def resolve_pad_token_id(tokenizer: PreTrainedTokenizerBase) -> int:
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError(
            "Tokenizer must define pad_token_id for example-aligned dashboard pretokenization."
        )
    return int(cast(int, pad_token_id))


def pad_to_context_size(
    tokens: torch.Tensor,
    *,
    context_size: int,
    pad_token_id: int,
) -> torch.Tensor:
    if tokens.numel() > context_size:
        raise ValueError(
            f"Tokenized prompt length {tokens.numel()} exceeds context_size={context_size}; "
            "example-aligned dashboard pretokenization does not truncate prompts."
        )
    if tokens.numel() == context_size:
        return tokens.to(dtype=torch.long).cpu()
    padding = torch.full(
        (context_size - tokens.numel(),),
        pad_token_id,
        dtype=torch.long,
        device=tokens.device,
    )
    return torch.cat([tokens.to(dtype=torch.long), padding]).cpu()


def attention_mask_for_prompt(*, length: int, context_size: int) -> list[int]:
    return [1] * length + [0] * (context_size - length)


def _pretokenize_example_aligned_prompt_sequences(
    token_sequences: Iterable[torch.Tensor],
    *,
    tokenizer: PreTrainedTokenizerBase,
    cfg: PretokenizeRunnerConfig,
    windowing_mode: WindowingMode,
) -> PretokenizationResult:
    prompt_tokens = [as_1d_token_tensor(tokens) for tokens in token_sequences]
    if not prompt_tokens:
        raise ValueError("Prompt dataset did not yield any prompts to pretokenize.")

    prompt_lengths = tuple(int(tokens.numel()) for tokens in prompt_tokens)
    effective_context_size = (
        max(prompt_lengths) if windowing_mode == "max-prompt-pad" else cfg.context_size
    )
    pad_token_id = resolve_pad_token_id(tokenizer)
    tokenized_dataset = Dataset.from_dict(
        {
            "input_ids": [
                pad_to_context_size(
                    tokens,
                    context_size=effective_context_size,
                    pad_token_id=pad_token_id,
                ).tolist()
                for tokens in prompt_tokens
            ],
            "attention_mask": [
                attention_mask_for_prompt(
                    length=int(tokens.numel()), context_size=effective_context_size
                )
                for tokens in prompt_tokens
            ],
        }
    )
    return PretokenizationResult(
        tokenized_dataset=ensure_torch_dataset_format(tokenized_dataset),
        effective_context_size=effective_context_size,
        prompt_lengths=prompt_lengths,
        windowing_mode=windowing_mode,
        disable_concat_sequences=True,
        pad_token_id=pad_token_id,
    )


def _pretokenize_packed_prompt_sequences(
    token_sequences: Iterable[torch.Tensor],
    *,
    tokenizer: PreTrainedTokenizerBase,
    cfg: PretokenizeRunnerConfig,
    windowing_mode: WindowingMode,
) -> PretokenizationResult:
    token_rows = list(
        concat_and_batch_sequences(
            tokens_iterator=iter(
                as_1d_token_tensor(tokens) for tokens in token_sequences
            ),
            context_size=cfg.context_size,
            begin_batch_token_id=get_special_token_from_cfg(
                cfg.begin_batch_token, tokenizer
            ),
            begin_sequence_token_id=get_special_token_from_cfg(
                cfg.begin_sequence_token, tokenizer
            ),
            sequence_separator_token_id=get_special_token_from_cfg(
                cfg.sequence_separator_token, tokenizer
            ),
            disable_concat_sequences=windowing_mode_disables_concat_sequences(
                windowing_mode
            ),
        )
    )
    tokenized_dataset = Dataset.from_dict(
        {"input_ids": [tokens.tolist() for tokens in token_rows]}
    )
    return PretokenizationResult(
        tokenized_dataset=ensure_torch_dataset_format(tokenized_dataset),
        effective_context_size=cfg.context_size,
        prompt_lengths=None,
        windowing_mode=windowing_mode,
        disable_concat_sequences=windowing_mode_disables_concat_sequences(
            windowing_mode
        ),
        pad_token_id=None,
    )


def _to_python_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        return value.tolist()
    if isinstance(value, dict):
        return {
            key: _to_python_value(inner_value) for key, inner_value in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_to_python_value(inner_value) for inner_value in value]
    return value


def _write_pretokenization_run_record(
    destination: str, *, duration_s: float, rows: int, context_size: int | None
) -> None:
    """Record the one-time build cost beside the dataset so downstream artifacts
    (e.g. the dashboard benchmark flow diagram) can report the pretokenization
    runtime and cardinality instead of TBD placeholders."""
    dest_path = Path(destination)
    if not dest_path.is_dir():
        return
    record = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "duration_s": round(duration_s, 3),
        "rows": rows,
        "context_size": context_size,
    }
    (dest_path / "pretokenization_run.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if (
        args.save_path is None
        and args.output_dir is None
        and args.legacy_output_dir is None
        and args.hf_repo_id is None
    ):
        raise ValueError(
            "Provide --save-path/--output-dir, --legacy-output-dir, and/or --hf-repo-id."
        )

    start_time = perf_counter()
    result, cfg, metadata = run_dashboard_pretokenization(args)
    persist_dashboard_dataset(
        result.tokenized_dataset,
        cfg=cfg,
        metadata=metadata,
        force=args.force,
        legacy_output_dir=args.legacy_output_dir,
    )
    duration_s = perf_counter() - start_time
    destination = cfg.save_path or cfg.hf_repo_id or "<in-memory>"
    effective_context = result.effective_context_size or cfg.context_size
    if cfg.save_path:
        _write_pretokenization_run_record(
            cfg.save_path,
            duration_s=duration_s,
            rows=len(result.tokenized_dataset),
            context_size=effective_context,
        )
    print(
        f"Saved {len(result.tokenized_dataset)} tokenized prompts to {destination} "
        f"with context_size={effective_context} in {duration_s:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
