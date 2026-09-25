# pyright: basic, reportPrivateImportUsage=false
import json
from pathlib import Path

import pytest
from datasets import Dataset

from sae_dashboard.neuronpedia.prompt_datasets import (
    PROMPT_DATASET_MODES,
    PromptDatasetConfig,
    load_prompt_dataset,
    resolve_prompt_dataset,
    write_pretokenized_prompt_artifacts,
)


def test_prompt_dataset_modes_are_canonical_only() -> None:
    assert PROMPT_DATASET_MODES == ("load_dataset", "load_from_disk", "legacy_jsonl")


def test_prompt_dataset_config_defaults_to_load_dataset() -> None:
    resolution = resolve_prompt_dataset(
        PromptDatasetConfig(dataset_path="aps/super_glue", dataset_name="rte")
    )

    assert resolution.mode == "load_dataset"
    assert resolution.loader_api == "load_dataset"


def test_load_dataset_with_text_field_materializes_text_column(monkeypatch) -> None:
    dataset = Dataset.from_dict(
        {"prompt": ["Already rendered prompt."], "other": ["ignored"]}
    )

    def fake_load_dataset(
        path: str,
        *,
        name: str | None,
        data_dir: str | None,
        data_files,
        split: str,
        streaming: bool,
        trust_remote_code: bool | None,
    ) -> Dataset:
        assert path == "aps/super_glue"
        assert name == "rte"
        assert data_dir is None
        assert data_files is None
        assert split == "train"
        assert streaming is True
        assert trust_remote_code is None
        return dataset

    monkeypatch.setattr(
        "sae_dashboard.neuronpedia.prompt_datasets.load_dataset",
        fake_load_dataset,
    )

    materialized = load_prompt_dataset(
        resolve_prompt_dataset(
            PromptDatasetConfig(
                dataset_path="aps/super_glue",
                dataset_name="rte",
                split="train",
                text_field="prompt",
            )
        )
    )

    assert isinstance(materialized.dataset_source, Dataset)
    assert materialized.text_column == "text"
    assert materialized.dataset_source[0]["text"] == "Already rendered prompt."


def test_load_dataset_tokenized_local_sidecar_sets_token_metadata(
    monkeypatch,
    tmp_path: Path,
) -> None:
    metadata_path = tmp_path / "sae_lens.json"
    metadata_path.write_text(json.dumps({"pad_token_id": 0}), encoding="utf-8")
    dataset = Dataset.from_dict(
        {
            "input_ids": [[1, 2, 0, 0], [3, 4, 5, 0]],
            "attention_mask": [[1, 1, 0, 0], [1, 1, 1, 0]],
        }
    )

    def fake_load_dataset(
        path: str,
        *,
        name: str | None,
        data_dir: str | None,
        data_files,
        split: str,
        streaming: bool,
        trust_remote_code: bool | None,
    ) -> Dataset:
        assert path == str(tmp_path)
        assert name is None
        assert data_dir is None
        assert data_files is None
        assert split == "train"
        assert streaming is False
        assert trust_remote_code is None
        return dataset

    monkeypatch.setattr(
        "sae_dashboard.neuronpedia.prompt_datasets.load_dataset",
        fake_load_dataset,
    )

    materialized = load_prompt_dataset(
        resolve_prompt_dataset(
            PromptDatasetConfig(
                dataset_path=str(tmp_path),
                mode="load_dataset",
                split="train",
                streaming=False,
            )
        )
    )

    assert isinstance(materialized.dataset_source, Dataset)
    assert materialized.token_column == "input_ids"
    assert materialized.attention_mask_column == "attention_mask"
    assert materialized.pad_token_id == 0
    assert materialized.metadata_path == str(metadata_path)


def test_legacy_jsonl_loads_local_split_file_directly(
    monkeypatch,
    tmp_path: Path,
) -> None:
    export_dir = tmp_path / "legacy_export"
    export_dir.mkdir()
    split_path = export_dir / "train.jsonl"
    split_path.write_text('{"input_ids":[1,2,3]}\n', encoding="utf-8")

    def fake_load_dataset(
        path: str,
        *,
        data_files,
        split: str,
        streaming: bool,
    ) -> Dataset:
        assert path == "json"
        assert data_files == {"train": str(split_path)}
        assert split == "train"
        assert streaming is False
        return Dataset.from_dict({"input_ids": [[1, 2, 3]]})

    monkeypatch.setattr(
        "sae_dashboard.neuronpedia.prompt_datasets.load_dataset",
        fake_load_dataset,
    )

    with pytest.deprecated_call(match="legacy_jsonl"):
        resolution = resolve_prompt_dataset(
            PromptDatasetConfig(
                dataset_path=str(export_dir),
                mode="legacy_jsonl",
                split="train",
                streaming=False,
            )
        )

    materialized = load_prompt_dataset(resolution)

    assert isinstance(materialized.dataset_source, Dataset)
    assert materialized.resolution.mode == "legacy_jsonl"


def test_write_pretokenized_prompt_artifacts_writes_modern_and_legacy_outputs(
    tmp_path: Path,
) -> None:
    dataset = Dataset.from_dict(
        {
            "input_ids": [[1, 2, 3, 0]],
            "attention_mask": [[1, 1, 1, 0]],
        }
    )
    metadata = {"pad_token_id": 0, "tokenizer_name": "google/gemma-3-1b-it"}
    save_to_disk_path = tmp_path / "pretokenized"
    legacy_output_dir = tmp_path / "legacy"

    write_pretokenized_prompt_artifacts(
        dataset,
        metadata=metadata,
        save_to_disk_path=save_to_disk_path,
        legacy_output_dir=legacy_output_dir,
        split="train",
        force=False,
    )

    assert (save_to_disk_path / "sae_lens.json").is_file()
    assert (legacy_output_dir / "sae_lens.json").is_file()
    legacy_rows = (
        (legacy_output_dir / "train.jsonl")
        .read_text(encoding="utf-8")
        .strip()
        .splitlines()
    )
    assert len(legacy_rows) == 1
    assert json.loads(legacy_rows[0])["input_ids"] == [1, 2, 3, 0]
