"""Prompt dataset loading contract for Neuronpedia dashboard generation.

The mode names intentionally mirror Hugging Face Datasets APIs:

- ``load_dataset`` uses ``datasets.load_dataset(...)`` for Hub datasets and local/file-backed builders.
    In the dashboard flow, use this for raw prompt rows and for tokenized datasets whose source of truth is still a
    builder-backed dataset repo or file collection.
- ``load_from_disk`` uses ``datasets.load_from_disk(...)`` for directories written by ``Dataset.save_to_disk()``.
    In the dashboard flow, use this for local prompt caches produced by the pretokenization step when you want a stable
    on-disk artifact together with local sidecars such as ``sae_lens.json`` and shared token caches.
- Deprecated ``legacy_jsonl`` uses ``datasets.load_dataset("json", data_files=...)`` for legacy dashboard JSONL
    exports that contain ``<split>.jsonl`` plus ``sae_lens.json`` metadata.
"""

from __future__ import annotations

import io
import json
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence, cast

from datasets import Dataset, DatasetDict, IterableDataset, load_dataset, load_from_disk
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

PromptDatasetMode = Literal["load_dataset", "load_from_disk", "legacy_jsonl"]
PromptDatasetDataFiles = str | Sequence[str] | dict[str, str | Sequence[str]]

PROMPT_DATASET_MODES: tuple[PromptDatasetMode, ...] = (
    "load_dataset",
    "load_from_disk",
    "legacy_jsonl",
)


@dataclass(frozen=True)
class PromptDatasetConfig:
    dataset_path: str
    mode: PromptDatasetMode = "load_dataset"
    dataset_name: str | None = None
    split: str | None = None
    text_field: str | None = None
    data_files: PromptDatasetDataFiles | None = None
    data_dir: str | None = None
    streaming: bool = True
    trust_remote_code: bool | None = None
    metadata_path: str | None = None


@dataclass(frozen=True)
class PromptDatasetResolution:
    mode: Literal["load_dataset", "load_from_disk", "legacy_jsonl"]
    loader_api: str
    dataset_path: str
    dataset_name: str | None
    split: str
    text_field: str | None
    data_files: PromptDatasetDataFiles | None
    data_dir: str | None
    streaming: bool
    trust_remote_code: bool | None
    metadata_path: str | None


@dataclass(frozen=True)
class PromptDatasetMaterialization:
    dataset_source: Dataset | IterableDataset
    resolution: PromptDatasetResolution
    metadata: dict[str, Any] | None
    metadata_path: str | None
    token_column: str | None
    text_column: str | None
    attention_mask_column: str | None
    pad_token_id: int | None

    @property
    def is_tokenized(self) -> bool:
        return self.token_column is not None


def resolve_prompt_dataset(config: PromptDatasetConfig) -> PromptDatasetResolution:
    if config.mode not in PROMPT_DATASET_MODES:
        raise ValueError(
            f"Unsupported prompt dataset mode {config.mode!r}. Expected one of {PROMPT_DATASET_MODES}."
        )
    dataset_path = config.dataset_path.strip()
    if not dataset_path:
        raise ValueError("A prompt dataset path is required.")

    resolved_mode: Literal["load_dataset", "load_from_disk", "legacy_jsonl"] = (
        config.mode
    )

    split = config.split or "train"
    loader_api = {
        "load_dataset": "load_dataset",
        "load_from_disk": "load_from_disk",
        "legacy_jsonl": 'load_dataset("json", data_files=...)',
    }[resolved_mode]
    resolved_data_files = config.data_files
    if resolved_mode == "legacy_jsonl":
        warnings.warn(
            "Prompt dataset mode 'legacy_jsonl' is deprecated and exists only for legacy JSONL dashboard "
            "compatibility. Prefer 'load_dataset' for raw or tokenized Hugging Face datasets, or "
            "'load_from_disk' for save_to_disk() prompt caches.",
            DeprecationWarning,
            stacklevel=2,
        )
        resolved_data_files = _resolve_legacy_jsonl_data_files(
            dataset_path=dataset_path,
            split=split,
            data_files=config.data_files,
        )

    return PromptDatasetResolution(
        mode=resolved_mode,
        loader_api=loader_api,
        dataset_path=dataset_path,
        dataset_name=config.dataset_name,
        split=split,
        text_field=config.text_field,
        data_files=resolved_data_files,
        data_dir=config.data_dir,
        streaming=config.streaming,
        trust_remote_code=config.trust_remote_code,
        metadata_path=config.metadata_path,
    )


def load_prompt_dataset(
    resolution: PromptDatasetResolution,
    *,
    max_rows: int | None = None,
) -> PromptDatasetMaterialization:
    if resolution.mode == "load_from_disk":
        dataset_source = _coerce_saved_dataset(
            load_from_disk(resolution.dataset_path),
            split=resolution.split,
        )
    elif resolution.mode == "legacy_jsonl":
        dataset_source = load_dataset(
            "json",
            data_files=resolution.data_files,
            split=resolution.split,
            streaming=resolution.streaming,
        )
    else:
        dataset_source = load_dataset(  # type: ignore[call-overload]
            resolution.dataset_path,
            name=resolution.dataset_name,
            data_dir=resolution.data_dir,
            data_files=resolution.data_files,
            split=resolution.split,
            streaming=resolution.streaming,
            trust_remote_code=resolution.trust_remote_code,
        )

    if (
        isinstance(dataset_source, Dataset)
        and max_rows is not None
        and len(dataset_source) > max_rows
    ):
        dataset_source = dataset_source.select(range(max_rows))

    metadata, metadata_path = _load_prompt_dataset_metadata(resolution)
    token_column, attention_mask_column, text_column = _infer_columns(
        dataset_source, resolution.text_field
    )

    if (
        token_column is None
        and resolution.text_field
        and resolution.text_field != "text"
    ):
        dataset_source = _map_text_column(dataset_source, resolution.text_field)
        token_column, attention_mask_column, text_column = _infer_columns(
            dataset_source, "text"
        )

    pad_token_id_raw = None if metadata is None else metadata.get("pad_token_id")
    pad_token_id = int(pad_token_id_raw) if pad_token_id_raw is not None else None

    return PromptDatasetMaterialization(
        dataset_source=dataset_source,
        resolution=resolution,
        metadata=metadata,
        metadata_path=metadata_path,
        token_column=token_column,
        text_column=text_column,
        attention_mask_column=attention_mask_column,
        pad_token_id=pad_token_id,
    )


def write_pretokenized_prompt_artifacts(
    dataset: Dataset,
    *,
    metadata: dict[str, Any],
    save_to_disk_path: str | Path | None = None,
    legacy_output_dir: str | Path | None = None,
    split: str = "train",
    force: bool = False,
    hf_repo_id: str | None = None,
    hf_num_shards: int = 64,
    hf_revision: str = "main",
    hf_is_private_repo: bool = False,
) -> None:
    if save_to_disk_path is not None:
        output_dir = Path(save_to_disk_path)
        _prepare_output_dir(output_dir, force=force)
        dataset.save_to_disk(str(output_dir))
        write_prompt_dataset_metadata(output_dir, metadata)

    if legacy_output_dir is not None:
        output_dir = Path(legacy_output_dir)
        _prepare_output_dir(output_dir, force=force)
        data_path = output_dir / f"{split}.jsonl"
        dataset_for_export = dataset.with_format(type=None)
        with data_path.open("w", encoding="utf-8") as handle:
            for row in dataset_for_export:
                serialized_row = {
                    key: _coerce_json_value(value)
                    for key, value in cast("dict[str, Any]", row).items()
                }
                handle.write(json.dumps(serialized_row, ensure_ascii=False))
                handle.write("\n")
        write_prompt_dataset_metadata(output_dir, metadata)

    if hf_repo_id is not None:
        dataset.push_to_hub(
            repo_id=hf_repo_id,
            num_shards=hf_num_shards,
            private=hf_is_private_repo,
            revision=hf_revision,
        )
        _upload_prompt_dataset_metadata(
            repo_id=hf_repo_id,
            metadata=metadata,
        )


def write_prompt_dataset_metadata(
    output_dir: str | Path, metadata: dict[str, Any]
) -> Path:
    metadata_dir = Path(output_dir)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = metadata_dir / "sae_lens.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return metadata_path


def _prepare_output_dir(output_dir: Path, *, force: bool) -> None:
    if output_dir.exists():
        if not force:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. Pass force=True to replace it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _upload_prompt_dataset_metadata(*, repo_id: str, metadata: dict[str, Any]) -> None:
    metadata_io = io.BytesIO()
    metadata_io.write(
        json.dumps(metadata, indent=2, ensure_ascii=False).encode("utf-8")
    )
    metadata_io.seek(0)
    HfApi().upload_file(
        path_or_fileobj=metadata_io,
        path_in_repo="sae_lens.json",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Add sae_lens metadata",
    )


def _coerce_saved_dataset(
    dataset: Dataset | DatasetDict,
    *,
    split: str,
) -> Dataset:
    if isinstance(dataset, Dataset):
        return dataset
    if split in dataset:
        selected = dataset[split]
        if not isinstance(selected, Dataset):
            raise ValueError(f"Saved split {split!r} is not a Hugging Face Dataset.")
        return selected
    if len(dataset) == 1:
        selected = next(iter(dataset.values()))
        if not isinstance(selected, Dataset):
            raise ValueError("Saved dataset dict entry is not a Hugging Face Dataset.")
        return selected
    raise ValueError(
        f"load_from_disk returned a DatasetDict without split {split!r}. Available splits: {sorted(dataset.keys())}."
    )


def _resolve_legacy_jsonl_data_files(
    *,
    dataset_path: str,
    split: str,
    data_files: PromptDatasetDataFiles | None,
) -> dict[str, str | Sequence[str]]:
    if data_files is not None:
        return _normalize_data_files(split=split, data_files=data_files)

    local_path = Path(dataset_path)
    if local_path.is_dir():
        split_path = local_path / f"{split}.jsonl"
        return {split: str(split_path)}
    return {split: dataset_path}


def _normalize_data_files(
    *,
    split: str,
    data_files: PromptDatasetDataFiles,
) -> dict[str, str | Sequence[str]]:
    if isinstance(data_files, dict):
        normalized: dict[str, str | Sequence[str]] = {}
        for key, value in data_files.items():
            if isinstance(value, (list, tuple)):
                normalized[key] = [str(path) for path in value]
            else:
                normalized[key] = str(value)
        return normalized
    if isinstance(data_files, (list, tuple)):
        return {split: [str(path) for path in data_files]}
    return {split: str(data_files)}


def _load_prompt_dataset_metadata(
    resolution: PromptDatasetResolution,
) -> tuple[dict[str, Any] | None, str | None]:
    candidate_paths: list[str | Path] = []
    if resolution.metadata_path:
        candidate_paths.append(resolution.metadata_path)
    candidate_paths.extend(_local_metadata_candidates(resolution))

    seen_candidates: set[str] = set()
    for candidate in candidate_paths:
        candidate_path = Path(str(candidate)).expanduser()
        candidate_key = str(candidate_path)
        if candidate_key in seen_candidates:
            continue
        seen_candidates.add(candidate_key)
        if candidate_path.is_file():
            return _read_json_file(candidate_path), str(candidate_path)

    if _looks_like_hub_dataset_id(resolution.dataset_path):
        for filename in _hub_metadata_filenames(resolution):
            try:
                metadata_path = hf_hub_download(
                    repo_id=resolution.dataset_path,
                    repo_type="dataset",
                    filename=filename,
                )
            except (HfHubHTTPError, OSError):
                continue
            return _read_json_file(Path(metadata_path)), metadata_path

    return None, None


def _local_metadata_candidates(resolution: PromptDatasetResolution) -> list[Path]:
    candidates: list[Path] = []
    dataset_path = Path(resolution.dataset_path).expanduser()

    if dataset_path.is_dir():
        if resolution.data_dir:
            candidates.append(dataset_path / resolution.data_dir / "sae_lens.json")
        candidates.append(dataset_path / "sae_lens.json")
    elif dataset_path.is_file():
        candidates.append(dataset_path.parent / "sae_lens.json")

    if resolution.data_dir:
        data_dir_path = Path(resolution.data_dir).expanduser()
        if data_dir_path.is_dir():
            candidates.append(data_dir_path / "sae_lens.json")

    if resolution.data_files is None:
        return candidates

    normalized = _normalize_data_files(
        split=resolution.split,
        data_files=resolution.data_files,
    )
    for value in normalized.values():
        values = value if isinstance(value, list) else [value]
        for entry in values:
            entry_path = Path(str(entry)).expanduser()
            if entry_path.is_file():
                candidates.append(entry_path.parent / "sae_lens.json")
            elif entry_path.is_dir():
                candidates.append(entry_path / "sae_lens.json")

    return candidates


def _hub_metadata_filenames(resolution: PromptDatasetResolution) -> list[str]:
    filenames: list[str] = []
    if resolution.metadata_path:
        filenames.append(resolution.metadata_path)
    if resolution.data_dir:
        filenames.append(f"{resolution.data_dir.rstrip('/')}/sae_lens.json")
    filenames.append("sae_lens.json")
    return filenames


def _looks_like_hub_dataset_id(dataset_path: str) -> bool:
    path = Path(dataset_path)
    if path.is_absolute() or dataset_path.startswith((".", "~")):
        return False
    return (
        not path.exists()
        and "/" in dataset_path
        and not dataset_path.endswith(".jsonl")
    )


def _read_json_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _infer_columns(
    dataset_source: Dataset | IterableDataset,
    preferred_text_field: str | None,
) -> tuple[str | None, str | None, str | None]:
    column_names = list(getattr(dataset_source, "column_names", []) or [])
    token_column = next(
        (column for column in ("input_ids", "tokens") if column in column_names), None
    )
    attention_mask_column = (
        "attention_mask" if "attention_mask" in column_names else None
    )
    if preferred_text_field and preferred_text_field in column_names:
        text_column = preferred_text_field
    elif "text" in column_names:
        text_column = "text"
    elif preferred_text_field:
        text_column = preferred_text_field
    else:
        text_column = None
    return token_column, attention_mask_column, text_column


def _map_text_column(
    dataset_source: Dataset | IterableDataset,
    text_field: str,
) -> Dataset | IterableDataset:
    def select_text_field(example: dict[str, Any]) -> dict[str, str]:
        return {"text": str(example[text_field])}

    return dataset_source.map(select_text_field)


def _coerce_json_value(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return _coerce_json_value(value.tolist())
    if isinstance(value, dict):
        return {key: _coerce_json_value(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [_coerce_json_value(inner) for inner in value]
    if isinstance(value, Path):
        return str(value)
    return value
