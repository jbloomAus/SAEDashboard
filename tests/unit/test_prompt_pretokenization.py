# pyright: basic, reportPrivateImportUsage=false
from __future__ import annotations

import argparse
import sys
from types import ModuleType
from typing import Any

import pytest
import torch
from datasets import Dataset
from sae_lens.config import PretokenizeRunnerConfig

from sae_dashboard.neuronpedia import prompt_pretokenization
from sae_dashboard.neuronpedia.prompt_pretokenization import (
    DEFAULT_WINDOWING_MODE,
    PretokenizationResult,
    materialize_tokenized_dataset,
    pretokenize_prompt_token_sequences,
)


class DummyTokenizer:
    model_max_length = 0
    pad_token_id = 0

    def encode(self, text: Any, return_tensors: str | None = None) -> torch.Tensor:
        del return_tensors
        length = max(len(str(text)), 1)
        return torch.arange(1, length + 1, dtype=torch.long).unsqueeze(0)

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        return_tensors: str | None = None,
    ) -> torch.Tensor:
        del tokenize, return_tensors
        total_length = max(sum(len(message["content"]) for message in messages), 1)
        return torch.arange(1, total_length + 1, dtype=torch.long).unsqueeze(0)


def test_parse_args_has_no_default_custom_dataset_module() -> None:
    args = prompt_pretokenization.parse_args(
        [
            "--dataset-path",
            "monology/pile-uncopyrighted",
            "--tokenizer-name",
            "google/gemma-3-1b-it",
            "--output-dir",
            "/tmp/monology_tokens",
        ]
    )

    assert args.custom_dataset_module is None
    assert args.custom_module is None
    assert args.windowing_mode == DEFAULT_WINDOWING_MODE


def test_build_pretokenize_config_uses_windowing_mode_to_disable_concat_sequences() -> (
    None
):
    args = prompt_pretokenization.parse_args(
        [
            "--dataset-path",
            "monology/pile-uncopyrighted",
            "--tokenizer-name",
            "google/gemma-3-1b-it",
            "--output-dir",
            "/tmp/monology_tokens",
            "--windowing-mode",
            "filter-truncate",
        ]
    )

    cfg = prompt_pretokenization.build_pretokenize_config(args)

    assert cfg.disable_concat_sequences is True


def test_custom_pretokenization_module_hooks_are_optional(monkeypatch) -> None:
    module_name = "tests.fake_dashboard_pretokenizer"
    fake_module = ModuleType(module_name)
    seen: dict[str, Any] = {}

    def configure_parser(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--custom-flag", default="default")

    def load_custom_pretokenization_settings(
        args: argparse.Namespace,
    ) -> dict[str, str]:
        return {"custom_flag": args.custom_flag}

    def pretokenize_custom_dataset(
        dataset, tokenizer, cfg, settings
    ) -> PretokenizationResult:
        seen["dataset"] = dataset
        seen["settings"] = settings
        seen["tokenizer_max_length"] = tokenizer.model_max_length
        seen["context_size"] = cfg.context_size
        return PretokenizationResult(
            tokenized_dataset=Dataset.from_dict(
                {"input_ids": [[1, 2, 0]], "attention_mask": [[1, 1, 0]]}
            ),
            effective_context_size=3,
            prompt_lengths=(2,),
            windowing_mode="max-prompt-pad",
            disable_concat_sequences=True,
            pad_token_id=0,
        )

    def build_custom_metadata(args, settings, result, tokenizer, cfg) -> dict[str, Any]:
        return {
            "custom_flag": settings["custom_flag"],
            "rows": len(result.tokenized_dataset),
        }

    fake_module.configure_parser = configure_parser  # pyright: ignore
    fake_module.load_custom_pretokenization_settings = (  # pyright: ignore
        load_custom_pretokenization_settings
    )
    fake_module.pretokenize_custom_dataset = (  # pyright: ignore
        pretokenize_custom_dataset
    )
    fake_module.build_custom_metadata = build_custom_metadata  # pyright: ignore
    monkeypatch.setitem(sys.modules, module_name, fake_module)
    monkeypatch.setattr(
        prompt_pretokenization,
        "load_dashboard_dataset",
        lambda cfg, max_rows: Dataset.from_dict({"text": ["alpha"]}),
    )
    monkeypatch.setattr(
        prompt_pretokenization.AutoTokenizer,
        "from_pretrained",
        lambda _: DummyTokenizer(),
    )

    args = prompt_pretokenization.parse_args(
        [
            "--dataset-path",
            "local/prompts",
            "--tokenizer-name",
            "fake-tokenizer",
            "--output-dir",
            "/tmp/custom_tokens",
            "--custom-dataset-module",
            module_name,
            "--custom-flag",
            "custom-value",
        ]
    )

    result, cfg, metadata = prompt_pretokenization.run_dashboard_pretokenization(args)

    assert result.effective_context_size == 3
    assert cfg.context_size == 3
    assert seen["settings"] == {"custom_flag": "custom-value"}
    assert seen["tokenizer_max_length"] == sys.maxsize
    assert metadata["windowing_mode"] == "max-prompt-pad"
    assert metadata["prompt_windowing_family"] == "example_aligned_pad_enabled"
    assert metadata["pad_token_id"] == 0
    assert metadata["custom"] == {"custom_flag": "custom-value", "rows": 1}


def test_materialize_tokenized_dataset_truncates_iterable_rows() -> None:
    dataset = Dataset.from_dict(
        {
            "input_ids": [[1, 2], [3, 4], [5, 6]],
            "attention_mask": [[1, 1], [1, 1], [1, 1]],
        }
    ).to_iterable_dataset()

    materialized = materialize_tokenized_dataset(dataset, max_tokenized_rows=2)

    assert isinstance(materialized, Dataset)
    assert len(materialized) == 2
    assert materialized[0]["input_ids"].tolist() == [1, 2]
    assert materialized[1]["attention_mask"].tolist() == [1, 1]


def test_pretokenize_prompt_token_sequences_max_prompt_pad_materializes_attention_masks() -> (
    None
):
    cfg = _build_cfg(context_size=8)

    result = pretokenize_prompt_token_sequences(
        [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6, 7, 8])],
        tokenizer=DummyTokenizer(),  # pyright: ignore
        cfg=cfg,
        windowing_mode="max-prompt-pad",
    )

    assert result.effective_context_size == 5
    assert result.prompt_lengths == (3, 5)
    assert result.disable_concat_sequences is True
    assert result.tokenized_dataset[0]["input_ids"].tolist() == [1, 2, 3, 0, 0]
    assert result.tokenized_dataset[0]["attention_mask"].tolist() == [1, 1, 1, 0, 0]


def test_pretokenize_prompt_token_sequences_fixed_context_pad_uses_requested_context() -> (
    None
):
    cfg = _build_cfg(context_size=6)

    result = pretokenize_prompt_token_sequences(
        [torch.tensor([1, 2, 3]), torch.tensor([4, 5])],
        tokenizer=DummyTokenizer(),  # pyright: ignore
        cfg=cfg,
        windowing_mode="fixed-context-pad",
    )

    assert result.effective_context_size == 6
    assert result.prompt_lengths == (3, 2)
    assert result.tokenized_dataset[1]["input_ids"].tolist() == [4, 5, 0, 0, 0, 0]
    assert result.tokenized_dataset[1]["attention_mask"].tolist() == [1, 1, 0, 0, 0, 0]


def test_pretokenize_prompt_token_sequences_concatenate_preserves_packed_behavior() -> (
    None
):
    cfg = _build_cfg(context_size=4)

    result = pretokenize_prompt_token_sequences(
        [
            torch.tensor([1, 2]),
            torch.tensor([3, 4]),
            torch.tensor([5, 6]),
            torch.tensor([7, 8]),
        ],
        tokenizer=DummyTokenizer(),  # pyright: ignore
        cfg=cfg,
        windowing_mode="concatenate",
    )

    assert result.effective_context_size == 4
    assert result.prompt_lengths is None
    assert result.disable_concat_sequences is False
    assert len(result.tokenized_dataset) == 2
    assert result.tokenized_dataset[0]["input_ids"].tolist() == [1, 2, 3, 4]
    assert result.tokenized_dataset[1]["input_ids"].tolist() == [5, 6, 7, 8]


def test_pretokenize_prompt_token_sequences_filter_truncate_drops_short_rows() -> None:
    cfg = _build_cfg(context_size=4)

    result = pretokenize_prompt_token_sequences(
        [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6, 7, 8])],
        tokenizer=DummyTokenizer(),  # pyright: ignore
        cfg=cfg,
        windowing_mode="filter-truncate",
    )

    assert result.effective_context_size == 4
    assert result.prompt_lengths is None
    assert result.disable_concat_sequences is True
    assert len(result.tokenized_dataset) == 1
    assert result.tokenized_dataset[0]["input_ids"].tolist() == [4, 5, 6, 7]


def test_run_dashboard_pretokenization_allows_example_aligned_modes_without_custom_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        prompt_pretokenization,
        "load_dashboard_dataset",
        lambda cfg, max_rows: Dataset.from_dict({"text": ["alpha", "go"]}),
    )
    monkeypatch.setattr(
        prompt_pretokenization.AutoTokenizer,
        "from_pretrained",
        lambda _: DummyTokenizer(),
    )

    args = prompt_pretokenization.parse_args(
        [
            "--dataset-path",
            "monology/pile-uncopyrighted",
            "--tokenizer-name",
            "google/gemma-3-1b-it",
            "--output-dir",
            "/tmp/monology_tokens",
            "--windowing-mode",
            "max-prompt-pad",
        ]
    )

    result, cfg, metadata = prompt_pretokenization.run_dashboard_pretokenization(args)

    assert cfg.context_size == 5
    assert result.effective_context_size == 5
    assert result.prompt_lengths == (5, 2)
    assert result.tokenized_dataset[0]["attention_mask"].tolist() == [1, 1, 1, 1, 1]
    assert result.tokenized_dataset[1]["input_ids"].tolist() == [1, 2, 0, 0, 0]
    assert metadata["windowing_mode"] == "max-prompt-pad"
    assert metadata["prompt_lengths_available"] is True
    assert metadata["streaming_supported"] is False


def test_run_dashboard_pretokenization_allows_example_aligned_chat_formatting_without_custom_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        prompt_pretokenization,
        "load_dashboard_dataset",
        lambda cfg, max_rows: Dataset.from_dict({"prompt": ["hi", "hello"]}),
    )
    monkeypatch.setattr(
        prompt_pretokenization.AutoTokenizer,
        "from_pretrained",
        lambda _: DummyTokenizer(),
    )

    args = prompt_pretokenization.parse_args(
        [
            "--dataset-path",
            "monology/pile-uncopyrighted",
            "--tokenizer-name",
            "google/gemma-3-1b-it",
            "--output-dir",
            "/tmp/monology_tokens",
            "--column-name",
            "prompt",
            "--use-chat-formatting",
            "--windowing-mode",
            "fixed-context-pad",
            "--context-size",
            "6",
        ]
    )

    with pytest.warns(
        UserWarning, match="use_chat_formatting is True but column contains strings"
    ):
        result, _, metadata = prompt_pretokenization.run_dashboard_pretokenization(args)

    assert result.effective_context_size == 6
    assert result.prompt_lengths == (2, 5)
    assert result.tokenized_dataset[0]["attention_mask"].tolist() == [1, 1, 0, 0, 0, 0]
    assert result.tokenized_dataset[1]["input_ids"].tolist() == [1, 2, 3, 4, 5, 0]
    assert metadata["windowing_mode"] == "fixed-context-pad"
    assert metadata["prompt_windowing_family"] == "example_aligned_pad_enabled"


def _build_cfg(*, context_size: int) -> PretokenizeRunnerConfig:
    return PretokenizeRunnerConfig(
        tokenizer_name="fake-tokenizer",
        dataset_path="local/prompts",
        split="train",
        num_proc=1,
        context_size=context_size,
        column_name="text",
        shuffle=False,
        streaming=False,
        pretokenize_batch_size=1,
        begin_batch_token=None,
        begin_sequence_token=None,
        sequence_separator_token=None,
        disable_concat_sequences=False,
    )
