# pyright: basic, reportPrivateImportUsage=false
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from datasets import Dataset
from transformer_lens import HookedTransformer

import sae_dashboard.neuronpedia.neuronpedia_runner as neuronpedia_runner_module
from sae_dashboard.neuronpedia import neuronpedia_runner as runner_module
from sae_dashboard.neuronpedia.legacy import runner as legacy_runner
from sae_dashboard.neuronpedia.neuronpedia_runner import NeuronpediaRunner
from sae_dashboard.neuronpedia.neuronpedia_runner_config import (
    NeuronpediaRunnerConfig,
    is_legacy_dashboard_path,
    warn_if_deprecated_legacy_dashboard_path,
)
from sae_dashboard.neuronpedia.prompt_datasets import (
    PromptDatasetConfig,
    load_prompt_dataset,
    resolve_prompt_dataset,
)
from sae_dashboard.tl_compat import hooked_sae_transformer_cls

LEGACY_BASELINE_CONTRACT_PATH = (
    Path(__file__).parents[1]
    / "fixtures"
    / "legacy_dashboard_gen_baseline"
    / "preserved_baseline_contract.json"
)


def _load_legacy_baseline_contract() -> dict[str, Any]:
    return json.loads(LEGACY_BASELINE_CONTRACT_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def runner_config() -> NeuronpediaRunnerConfig:
    return NeuronpediaRunnerConfig(
        sae_set="gpt2-small-res-jb",
        sae_path="blocks.5.hook_resid_pre",
        outputs_dir="test_outputs",
        n_prompts_total=256,
        n_tokens_in_prompt=128,
        huggingface_dataset_path="monology/pile-uncopyrighted",
    )


@pytest.fixture
def neuronpedia_runner(
    runner_config: NeuronpediaRunnerConfig, model: HookedTransformer
) -> NeuronpediaRunner:
    runner = NeuronpediaRunner(runner_config)
    runner.model = model  # type: ignore
    return runner


def test_generate_tokens_no_duplicates(neuronpedia_runner: NeuronpediaRunner) -> None:
    tokens = neuronpedia_runner.generate_tokens(
        neuronpedia_runner.activations_store, n_prompts=256
    )
    assert tokens.shape == (256, neuronpedia_runner.cfg.n_tokens_in_prompt)
    # Move to CPU before checking uniqueness
    tokens_cpu = tokens.cpu()
    assert len(torch.unique(tokens_cpu, dim=0)) == 256


def test_generate_tokens_works_without_the_activations_store_device_seam(
    neuronpedia_runner: NeuronpediaRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`generate_tokens` must run against a RELEASED sae-lens, which lacks `move_to_model_device`.

    The kwarg is an opt-in optimization added by the coordinated SAELens PR. Passing it
    unconditionally raised ``TypeError`` on every batch against released sae-lens, which the
    ``sae-lens = "^6.43.0"`` pin happily allows -- so the runner was unusable and nothing warned.
    """
    store = neuronpedia_runner.activations_store
    real_get_batch_tokens = store.get_batch_tokens

    def released_signature_get_batch_tokens() -> torch.Tensor:
        # No `move_to_model_device` parameter, exactly like released sae-lens.
        return real_get_batch_tokens()

    monkeypatch.setattr(store, "get_batch_tokens", released_signature_get_batch_tokens)
    monkeypatch.setattr(
        neuronpedia_runner_module,
        "_ACTIVATIONS_STORE_SUPPORTS_DEVICE_SEAM",
        False,
    )

    tokens = neuronpedia_runner.generate_tokens(store, n_prompts=256)

    assert tokens.shape == (256, neuronpedia_runner.cfg.n_tokens_in_prompt)


def test_get_tokens_no_duplicates(
    neuronpedia_runner: NeuronpediaRunner, tmp_path: Path
) -> None:
    neuronpedia_runner.cfg.outputs_dir = str(tmp_path)
    tokens = neuronpedia_runner.get_tokens()
    assert tokens.shape == (
        neuronpedia_runner.cfg.n_prompts_total,
        neuronpedia_runner.cfg.n_tokens_in_prompt,
    )
    # Move to CPU before checking uniqueness
    tokens_cpu = tokens.cpu()
    assert (
        len(torch.unique(tokens_cpu, dim=0)) == neuronpedia_runner.cfg.n_prompts_total
    )


def test_run_neuronpedia_export_uses_neuronpedia_model_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = NeuronpediaRunnerConfig(
        sae_set="sae/repo",
        sae_path="blocks.5.hook_resid_pre",
        np_set_name="resid-pre",
        outputs_dir=str(tmp_path / "dashboards"),
        output_neuronpedia_exports=True,
        neuronpedia_exports_dir=str(tmp_path / "exports"),
        neuronpedia_creator_name="Test Creator",
        neuronpedia_release_id="test-release",
        neuronpedia_release_title="Test Release",
        neuronpedia_release_url="https://example.com/release",
        neuronpedia_source_set_description="Residual stream",
        neuronpedia_model_name="neuronpedia-model",
    )
    runner = NeuronpediaRunner.__new__(NeuronpediaRunner)
    runner.cfg = cfg
    runner.layer = 5
    runner.model_id = "hf-org/hf-model-path"
    runner.hook_name = "blocks.5.hook_resid_pre"

    captured: dict[str, runner_module.NeuronpediaExportConfig] = {}

    def fake_export_neuronpedia_dashboards(
        export_cfg: runner_module.NeuronpediaExportConfig,
    ) -> str:
        captured["export_cfg"] = export_cfg
        return str(tmp_path / "exports" / "neuronpedia-model" / "5-resid-pre")

    monkeypatch.setattr(
        runner_module,
        "export_neuronpedia_dashboards",
        fake_export_neuronpedia_dashboards,
    )

    runner._run_neuronpedia_export()

    assert captured["export_cfg"].exports_dir == str(tmp_path / "exports")
    assert captured["export_cfg"].model_name == "neuronpedia-model"


def test_legacy_get_tokens_uses_explicit_shared_tokens_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    shared_tokens = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    shared_tokens_file = tmp_path / "shared_tokens.pt"
    torch.save(shared_tokens, shared_tokens_file)

    runner = NeuronpediaRunner.__new__(NeuronpediaRunner)
    runner.cfg = NeuronpediaRunnerConfig(
        sae_set="gpt2-small-res-jb",
        sae_path="blocks.5.hook_resid_pre",
        outputs_dir=str(tmp_path / "outputs"),
        n_prompts_total=2,
        n_tokens_in_prompt=3,
        dashboard_output_format="legacy_json",
        sequence_selection_backend="legacy",
        shared_tokens_file=str(shared_tokens_file),
    )
    runner._log_token_snapshot = lambda *_args, **_kwargs: None

    def fail_generate_tokens(*_args: Any, **_kwargs: Any) -> torch.Tensor:
        raise AssertionError(
            "legacy explicit shared tokens should be staged before generation"
        )

    monkeypatch.setattr(runner, "generate_tokens", fail_generate_tokens)

    tokens = runner.get_tokens()

    assert torch.equal(tokens, shared_tokens)
    assert torch.equal(
        torch.load(tmp_path / "outputs" / "tokens_2.pt", map_location="cpu"),
        shared_tokens,
    )


def test_legacy_vis_config_forces_cpu_correlation_accumulation(
    runner_config: NeuronpediaRunnerConfig, tmp_path: Path
) -> None:
    runner_config.dashboard_output_format = "legacy_json"
    runner_config.sequence_selection_backend = "legacy"
    runner_config.outputs_dir = str(tmp_path / "outputs")
    runner_config.correlation_accumulation_device = "cuda"

    runner = SimpleNamespace(
        cfg=runner_config,
        cached_activations_dir=tmp_path / "cached_activations",
        hook_name="blocks.5.hook_resid_pre",
        tokenizer=SimpleNamespace(
            pad_token_id=None,
            bos_token_id=1,
            eos_token_id=2,
        ),
    )

    vis_cfg = legacy_runner._build_legacy_vis_config(
        runner,
        features_to_process=[0, 1],
    )

    assert vis_cfg.correlation_accumulation_device == "cpu"


def test_materialize_pretokenized_dataset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset = Dataset.from_dict(
        {"input_ids": [list(range(128)), list(range(128, 256)), list(range(256, 384))]}
    )
    pretokenized_path = tmp_path / "rte_tokens"

    def fake_load_from_disk(path: str) -> Dataset:
        assert path == str(pretokenized_path)
        return dataset

    monkeypatch.setattr(
        "sae_dashboard.neuronpedia.prompt_datasets.load_from_disk",
        fake_load_from_disk,
    )

    runner = NeuronpediaRunner.__new__(NeuronpediaRunner)
    runner.cfg = NeuronpediaRunnerConfig(
        sae_set="gpt2-small-res-jb",
        sae_path="blocks.5.hook_resid_pre",
        outputs_dir="test_outputs",
        n_prompts_total=2,
        huggingface_dataset_path="aps/super_glue",
        prompt_dataset_mode="load_from_disk",
        prompt_dataset_path=str(pretokenized_path),
        pretokenized_dataset_path=str(pretokenized_path),
    )

    materialized_dataset = runner._materialize_prompt_dataset()

    assert isinstance(materialized_dataset, Dataset)
    assert len(materialized_dataset) == 2
    assert materialized_dataset.column_names == ["input_ids"]


def test_legacy_preserved_baseline_contract_maps_to_deprecated_legacy_runner() -> None:
    baseline = _load_legacy_baseline_contract()

    assert baseline["preserved_baseline_lineage"] == {
        "saedashboard": "7886eaa",
        "saelens": "3eea6552",
        "neuronpedia": "5a33f17",
    }

    for scenario_name, scenario in baseline["scenarios"].items():
        prompt_contract = scenario["prompt_contract"]
        legacy_contract = scenario["legacy_contract"]
        shape = scenario["shape"]
        prompt_dataset_path = (
            f"/preserved-baseline/{scenario_name}/{prompt_contract['split']}.jsonl"
        )
        cfg = NeuronpediaRunnerConfig(
            sae_set="gemma-scope-2-1b-it-transcoders-all",
            sae_path="layer_9_width_262k_l0_small_affine",
            outputs_dir="test_outputs",
            prompt_dataset_mode=prompt_contract["mode"],
            prompt_dataset_path=prompt_dataset_path,
            prompt_dataset_split=prompt_contract["split"],
            n_tokens_in_prompt=prompt_contract["context_size"],
            n_features_at_a_time=shape["n_features_per_batch"],
            n_prompts_in_forward_pass=shape["n_prompts_in_forward_pass"],
            dashboard_output_format=legacy_contract["dashboard_output_format"],
            sequence_selection_backend=legacy_contract["sequence_selection_backend"],
        )

        assert is_legacy_dashboard_path(cfg)
        with pytest.deprecated_call(match="legacy JSON dashboard path"):
            warn_if_deprecated_legacy_dashboard_path(cfg)
        with pytest.deprecated_call(match="legacy_jsonl"):
            resolution = resolve_prompt_dataset(
                PromptDatasetConfig(
                    dataset_path=prompt_dataset_path,
                    mode=prompt_contract["mode"],
                    split=prompt_contract["split"],
                    streaming=False,
                )
            )
        assert resolution.loader_api == 'load_dataset("json", data_files=...)'
        assert resolution.data_files == {prompt_contract["split"]: prompt_dataset_path}
        assert prompt_contract["required_files"] == [
            "train.jsonl",
            "sae_lens.json",
        ]


def test_initialize_model_hooked_uses_no_processing_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    try:
        hooked_cls = hooked_sae_transformer_cls()
    except ImportError as err:  # transformer-lens >= 4 has no hooked wrapper to test
        pytest.skip(str(err))
    calls: dict[str, Any] = {}
    fake_model = SimpleNamespace(tokenizer=object())

    def fake_from_pretrained_no_processing(
        *,
        model_name: str,
        device: str,
        n_devices: int,
        hf_model: Any,
        dtype: torch.dtype,
        **kwargs: Any,
    ) -> Any:
        calls.update(
            {
                "model_name": model_name,
                "device": device,
                "n_devices": n_devices,
                "hf_model": hf_model,
                "dtype": dtype,
                "kwargs": kwargs,
            }
        )
        return fake_model

    def fail_from_pretrained(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("hooked runner should not use from_pretrained")

    monkeypatch.setattr(
        hooked_cls,
        "from_pretrained_no_processing",
        fake_from_pretrained_no_processing,
    )
    monkeypatch.setattr(
        hooked_cls,
        "from_pretrained",
        fail_from_pretrained,
    )

    runner = NeuronpediaRunner.__new__(NeuronpediaRunner)
    runner.cfg = NeuronpediaRunnerConfig(
        sae_set="gpt2-small-res-jb",
        sae_path="blocks.5.hook_resid_pre",
        outputs_dir="test_outputs",
        huggingface_dataset_path="monology/pile-uncopyrighted",
        model_wrapper="hooked",
        model_dtype="bfloat16",
        model_device="cpu",
        model_n_devices=1,
        free_unused_model_layers=False,
    )
    runner.sae = SimpleNamespace(
        cfg=SimpleNamespace(metadata={"hook_name": "blocks.5.hook_resid_pre"})
    )
    runner.sae_from_pretrained_kwargs = {"fold_ln": False}
    runner.model_id = "google/gemma-3-1b-it"
    runner._log_resource_snapshot = lambda *_args, **_kwargs: None
    runner._log_hook_alias_summary = lambda *_args, **_kwargs: None

    runner._initialize_model()

    assert runner.model is fake_model
    assert runner.tokenizer is fake_model.tokenizer
    assert calls == {
        "model_name": "google/gemma-3-1b-it",
        "device": "cpu",
        "n_devices": 1,
        "hf_model": None,
        "dtype": torch.bfloat16,
        "kwargs": {"fold_ln": False},
    }


def test_materialize_structured_dataset_uses_supplied_text_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = Dataset.from_dict(
        {"prompt": ["Already rendered prompt."], "other": ["ignored"]}
    )

    def fake_load_dataset(
        path: str,
        *,
        name: str | None,
        data_dir: str | None,
        data_files: Any,
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

    runner = NeuronpediaRunner.__new__(NeuronpediaRunner)
    runner.cfg = NeuronpediaRunnerConfig(
        sae_set="gpt2-small-res-jb",
        sae_path="blocks.5.hook_resid_pre",
        outputs_dir="test_outputs",
        n_prompts_total=1,
        huggingface_dataset_path="aps/super_glue",
        huggingface_dataset_config_name="rte",
        huggingface_dataset_split="train",
        huggingface_dataset_text_field="prompt",
        prompt_dataset_mode="load_dataset",
        prompt_dataset_path="aps/super_glue",
        prompt_dataset_name="rte",
        prompt_dataset_split="train",
        prompt_dataset_text_field="prompt",
    )

    materialized_dataset = runner._materialize_prompt_dataset()

    assert isinstance(materialized_dataset, Dataset)
    assert materialized_dataset[0]["text"] == "Already rendered prompt."


def test_prepare_shared_tokens_from_tokenized_load_dataset_uses_sidecar_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset = Dataset.from_dict(
        {
            "input_ids": [[1, 2, 0, 0], [3, 4, 5, 0]],
            "attention_mask": [[1, 1, 0, 0], [1, 1, 1, 0]],
        }
    )
    metadata_path = tmp_path / "sae_lens.json"
    metadata_path.write_text(json.dumps({"pad_token_id": 0}), encoding="utf-8")

    def fake_load_dataset(
        path: str,
        *,
        name: str | None,
        data_dir: str | None,
        data_files: Any,
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

    runner = NeuronpediaRunner.__new__(NeuronpediaRunner)
    runner.cfg = NeuronpediaRunnerConfig(
        sae_set="gpt2-small-res-jb",
        sae_path="blocks.5.hook_resid_pre",
        outputs_dir=str(tmp_path / "outputs"),
        n_prompts_total=2,
        n_tokens_in_prompt=4,
        huggingface_dataset_path=str(tmp_path),
        prompt_dataset_mode="load_dataset",
        prompt_dataset_path=str(tmp_path),
        dataset_streaming=False,
    )
    runner._prompt_dataset_materialization = load_prompt_dataset(
        resolve_prompt_dataset(
            PromptDatasetConfig(
                dataset_path=str(tmp_path),
                mode="load_dataset",
                split="train",
                streaming=False,
            )
        )
    )

    target_tokens_file = tmp_path / "outputs" / "tokens_2.pt"
    runner._prepare_shared_tokens_from_prompt_dataset(target_tokens_file)

    tokens = torch.load(target_tokens_file)
    effective_lengths = torch.load(
        target_tokens_file.with_suffix(".effective_lengths.pt")
    )

    assert torch.equal(
        tokens,
        torch.tensor([[1, 2, 0, 0], [3, 4, 5, 0]], dtype=torch.long),
    )
    assert torch.equal(
        effective_lengths,
        torch.tensor([2, 3], dtype=torch.int32),
    )


def _make_legacy_runner_stub(tmp_path: Path) -> NeuronpediaRunner:
    runner = NeuronpediaRunner.__new__(NeuronpediaRunner)
    runner.cfg = NeuronpediaRunnerConfig(
        sae_set="gpt2-small-res-jb",
        sae_path="blocks.5.hook_resid_pre",
        outputs_dir=str(tmp_path),
        n_prompts_total=1,
        n_tokens_in_prompt=4,
        n_prompts_in_forward_pass=2,
        n_features_at_a_time=2,
        quantile_feature_batch_size=3,
        prompt_dataset_mode="legacy_jsonl",
        prompt_dataset_path=str(tmp_path / "train.jsonl"),
        prompt_dataset_split="train",
        dashboard_output_format="legacy_json",
        sequence_selection_backend="legacy",
        use_wandb=False,
        output_neuronpedia_exports=False,
        use_cached_activations=False,
        primary_acts_batch_size=17,
    )
    runner.sae = SimpleNamespace(
        cfg=SimpleNamespace(
            d_sae=4,
            to_dict=lambda: {},
            metadata=SimpleNamespace(),
        )
    )
    runner.model = SimpleNamespace()  # pyright: ignore
    runner.tokenizer = SimpleNamespace(pad_token_id=0, bos_token_id=1, eos_token_id=2)
    runner.model_id = "google/gemma-3-1b-it"
    runner.hook_name = "blocks.5.hook_resid_pre"
    runner.layer = 5
    runner.vocab_dict = {0: "<pad>"}
    runner.cached_activations_dir = tmp_path / "cached_activations"
    runner.record_skipped_features = lambda: None
    runner.get_feature_batches = lambda: [[0, 1]]
    runner.get_tokens = lambda: torch.tensor([[1, 2, 3, 0]], dtype=torch.long)
    runner.add_prefix_suffix_to_tokens = lambda tokens: tokens
    runner._run_neuronpedia_export = lambda: None
    runner._log_resource_snapshot = lambda *_args, **_kwargs: None
    runner._log_batch_boundary_snapshot = lambda *_args, **_kwargs: None
    runner._log_token_snapshot = lambda *_args, **_kwargs: None
    runner._write_converter_input_artifact = lambda *_args, **_kwargs: None
    runner._release_unused_host_memory = lambda: None
    runner.activations_store = object()  # pyright: ignore
    return runner


def test_run_routes_legacy_through_compatibility_module(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _make_legacy_runner_stub(tmp_path)
    calls: dict[str, Any] = {}

    def fail_schedule(_tokens: torch.Tensor) -> None:
        raise AssertionError("legacy runner should bypass prompt bucket scheduling")

    def fake_run_legacy_batch_loop(
        runner_arg: NeuronpediaRunner,
        *,
        feature_idx: list[list[int]],
        tokens: torch.Tensor,
    ) -> None:
        calls["runner"] = runner_arg
        calls["feature_idx"] = feature_idx
        calls["tokens"] = tokens.clone()

    runner._load_prompt_bucket_schedule = fail_schedule  # pyright: ignore
    monkeypatch.setattr(
        "sae_dashboard.neuronpedia.neuronpedia_runner.legacy_runner.run_legacy_batch_loop",
        fake_run_legacy_batch_loop,
    )

    runner.run()

    assert calls["runner"] is runner
    assert calls["feature_idx"] == [[0, 1]]
    assert torch.equal(
        calls["tokens"],
        torch.tensor([[1, 2, 3, 0]], dtype=torch.long),
    )
    assert (tmp_path / "run_settings.json").is_file()


def test_legacy_batch_loop_uses_compatibility_vis_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _make_legacy_runner_stub(tmp_path)
    captured: dict[str, Any] = {}

    def fake_run_feature_batch_with_optional_profile(
        feature_vis_config_gpt: Any,
        tokens: torch.Tensor,
        feature_batch_count: int,
    ) -> object:
        captured["config"] = feature_vis_config_gpt
        captured["tokens"] = tokens.clone()
        captured["batch"] = feature_batch_count
        return object()

    monkeypatch.setattr(
        "sae_dashboard.neuronpedia.legacy.runner.NeuronpediaConverter.convert_to_np_json",
        lambda *args, **kwargs: '{"ok": true}',
    )
    runner._run_feature_batch_with_optional_profile = (  # pyright: ignore
        fake_run_feature_batch_with_optional_profile
    )

    legacy_runner.run_legacy_batch_loop(
        runner,
        feature_idx=[[0, 1]],
        tokens=torch.tensor([[1, 2, 3, 0]], dtype=torch.long),
    )

    feature_vis_config = captured["config"]
    assert feature_vis_config.prompt_minibatch_schedule is None
    assert feature_vis_config.primary_acts_batch_size is None
    assert feature_vis_config.correlation_accumulation_device == "cpu"
    assert feature_vis_config.sequence_selection_backend == "legacy"
    assert feature_vis_config.dashboard_output_format == "legacy_json"
    assert feature_vis_config.cache_dir == tmp_path / "_activation_cache"
    assert captured["batch"] == 0
    assert torch.equal(
        captured["tokens"],
        torch.tensor([[1, 2, 3, 0]], dtype=torch.long),
    )
    assert (tmp_path / "batch-0.json").read_text(encoding="utf-8") == '{"ok": true}'


def test_create_output_directory_sanitizes_model_id(tmp_path: Path) -> None:
    runner = NeuronpediaRunner.__new__(NeuronpediaRunner)
    runner.cfg = NeuronpediaRunnerConfig(
        sae_set="gemma-scope-2-1b-it-transcoders-all",
        sae_path="layer_10_width_262k_l0_small_affine",
        outputs_dir=str(tmp_path / "layer_10"),
    )
    runner.model_id = "google/gemma-3-1b-it"
    runner.hook_name = "blocks.10.hook_mlp_in"
    runner.sae = SimpleNamespace(cfg=SimpleNamespace(d_sae=262144))
    runner.np_sae_id_suffix = None

    output_dir = Path(runner.create_output_directory())

    assert output_dir.parent == tmp_path / "layer_10"
    assert (
        output_dir.name
        == "google_gemma-3-1b-it_gemma-scope-2-1b-it-transcoders-all_blocks.10.hook_mlp_in_262144"
    )


def test_resolved_neuronpedia_set_name_appends_suffix() -> None:
    runner = NeuronpediaRunner.__new__(NeuronpediaRunner)
    runner.cfg = NeuronpediaRunnerConfig(
        sae_set="gemma-scope-2-1b-it-transcoders-all",
        sae_path="layer_9_width_262k_l0_small_affine",
        outputs_dir="test_outputs",
        np_set_name="gemmascope-2-transcoder-262k-rte",
    )
    runner.np_sae_id_suffix = "phase1-pr-clean-lazy-parquet-importwarmfix-l9-20260514"

    resolved_name = runner._resolved_neuronpedia_set_name()

    assert (
        resolved_name
        == "gemmascope-2-transcoder-262k-rte__phase1-pr-clean-lazy-parquet-importwarmfix-l9-20260514"
    )


def test_resolved_columnar_activation_copy_ids_include_suffix() -> None:
    runner = NeuronpediaRunner.__new__(NeuronpediaRunner)
    runner.cfg = NeuronpediaRunnerConfig(
        sae_set="gemma-scope-2-1b-it-transcoders-all",
        sae_path="layer_9_width_262k_l0_small_affine",
        outputs_dir="test_outputs",
        np_set_name="gemmascope-2-transcoder-262k-rte",
    )
    runner.np_sae_id_suffix = "phase1-pr-clean-lazy-parquet-importwarmfix-l9-20260514"
    runner.layer = 9

    assert runner._resolved_columnar_activation_copy_layer() == (
        "9-gemmascope-2-transcoder-262k-rte__phase1-pr-clean-lazy-parquet-importwarmfix-l9-20260514"
    )
    assert runner._resolved_columnar_activation_copy_id_prefix() == (
        "9-gemmascope-2-transcoder-262k-rte__phase1-pr-clean-lazy-parquet-importwarmfix-l9-20260514-activation"
    )


def test_setup_output_directory_stages_shared_tokens_file(tmp_path: Path) -> None:
    shared_tokens_file = tmp_path / "shared_tokens.pt"
    expected_tokens = torch.tensor([[1, 2, 3], [1, 2, 3]], dtype=torch.long)
    torch.save(expected_tokens, shared_tokens_file)

    runner = NeuronpediaRunner.__new__(NeuronpediaRunner)
    runner.cfg = NeuronpediaRunnerConfig(
        sae_set="gemma-scope-2-1b-it-transcoders-all",
        sae_path="layer_10_width_262k_l0_small_affine",
        outputs_dir=str(tmp_path / "layer_10"),
        n_prompts_total=2,
        shared_tokens_file=str(shared_tokens_file),
        dashboard_output_format="columnar",
        sequence_selection_backend="columnar_gpu",
    )
    runner.model_id = "google/gemma-3-1b-it"
    runner.hook_name = "blocks.10.hook_mlp_in"
    runner.sae = SimpleNamespace(cfg=SimpleNamespace(d_sae=262144))

    runner._setup_output_directory()
    tokens = runner.get_tokens()

    assert torch.equal(tokens, expected_tokens)
    assert Path(runner.cfg.outputs_dir, "tokens_2.pt").exists()


def test_legacy_get_tokens_without_shared_sidecar_generates_tokens(
    tmp_path: Path,
) -> None:
    generated_tokens = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)

    runner = NeuronpediaRunner.__new__(NeuronpediaRunner)
    runner.cfg = NeuronpediaRunnerConfig(
        sae_set="gemma-scope-2-1b-it-transcoders-all",
        sae_path="layer_10_width_262k_l0_small_affine",
        outputs_dir=str(tmp_path / "layer_10"),
        n_prompts_total=2,
        n_tokens_in_prompt=3,
        dashboard_output_format="legacy_json",
        sequence_selection_backend="legacy",
    )
    runner.model_id = "google/gemma-3-1b-it"
    runner.hook_name = "blocks.10.hook_mlp_in"
    runner.sae = SimpleNamespace(cfg=SimpleNamespace(d_sae=262144))
    runner.activations_store = object()  # pyright: ignore
    runner._log_token_snapshot = lambda *_args, **_kwargs: None
    runner._stage_shared_tokens_file = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError(
            "default legacy runs should not auto-stage shared token sidecars"
        )
    )

    def fake_generate_tokens(activations_store: object, n_prompts: int) -> torch.Tensor:
        assert activations_store is runner.activations_store
        assert n_prompts == 2
        return generated_tokens.clone()

    runner.generate_tokens = fake_generate_tokens  # pyright: ignore

    runner._setup_output_directory()
    tokens_path = Path(runner.cfg.outputs_dir) / "tokens_2.pt"
    assert not tokens_path.exists()

    tokens = runner.get_tokens()

    assert torch.equal(tokens, generated_tokens)
    assert torch.equal(torch.load(tokens_path), generated_tokens)


def test_load_prompt_bucket_schedule_uses_selected_bucket_configs(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "layer_10"
    output_dir.mkdir(parents=True)
    tokens = torch.tensor(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12],
            [13, 14, 15],
            [16, 17, 18],
        ],
        dtype=torch.long,
    )
    torch.save(tokens, output_dir / "tokens_6.pt")
    torch.save(
        torch.tensor([32, 64, 65, 100, 128, 200], dtype=torch.int32),
        output_dir / "tokens_6.effective_lengths.pt",
    )
    schedule_path = tmp_path / "selected_bucket_configs.json"
    schedule_path.write_text(
        json.dumps(
            {
                "buckets": [
                    {
                        "bucket_ceiling": 64,
                        "prompt_count": 2,
                        "selected_config": {
                            "n_prompts_in_forward_pass": 2,
                            "primary_acts_batch_size": 2,
                        },
                    },
                    {
                        "bucket_ceiling": 128,
                        "prompt_count": 3,
                        "selected_config": {
                            "n_prompts_in_forward_pass": 2,
                            "primary_acts_batch_size": 1,
                        },
                    },
                    {
                        "bucket_ceiling": 319,
                        "prompt_count": 1,
                        "selected_config": {
                            "n_prompts_in_forward_pass": 4,
                            "primary_acts_batch_size": 4,
                        },
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    runner = NeuronpediaRunner.__new__(NeuronpediaRunner)
    runner.cfg = NeuronpediaRunnerConfig(
        sae_set="gemma-scope-2-1b-it-transcoders-all",
        sae_path="layer_10_width_262k_l0_small_affine",
        outputs_dir=str(output_dir),
        n_prompts_total=6,
        shared_tokens_file=str(tmp_path / "shared_tokens.pt"),
        prompt_bucket_schedule_file=str(schedule_path),
    )

    schedule = runner._load_prompt_bucket_schedule(tokens)

    assert schedule == [
        {"prompt_indices": [0, 1], "seq_length": 64, "primary_acts_batch_size": 2},
        {"prompt_indices": [2, 3], "seq_length": 128, "primary_acts_batch_size": 1},
        {"prompt_indices": [4], "seq_length": 128, "primary_acts_batch_size": 1},
        {"prompt_indices": [5], "seq_length": 319, "primary_acts_batch_size": 4},
    ]


def test_load_prompt_bucket_schedule_can_auto_bucket_from_effective_lengths(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "layer_10"
    output_dir.mkdir(parents=True)
    tokens = torch.tensor(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12],
            [13, 14, 15],
            [16, 17, 18],
            [19, 20, 21],
            [22, 23, 24],
            [25, 26, 27],
            [28, 29, 30],
            [31, 32, 33],
            [34, 35, 36],
        ],
        dtype=torch.long,
    )
    torch.save(tokens, output_dir / "tokens_12.pt")
    torch.save(
        torch.tensor(
            [40, 50, 60, 64, 64, 64, 65, 70, 80, 90, 110, 120], dtype=torch.int32
        ),
        output_dir / "tokens_12.effective_lengths.pt",
    )

    runner = NeuronpediaRunner.__new__(NeuronpediaRunner)
    runner.cfg = NeuronpediaRunnerConfig(
        sae_set="gemma-scope-2-1b-it-transcoders-all",
        sae_path="layer_10_width_262k_l0_small_affine",
        outputs_dir=str(output_dir),
        n_prompts_total=12,
        n_tokens_in_prompt=128,
        n_prompts_in_forward_pass=2,
        primary_acts_batch_size=1,
        shared_tokens_file=str(tmp_path / "shared_tokens.pt"),
        auto_prompt_bucket_schedule=True,
        prompt_bucket_ceilings=(64,),
        prompt_bucket_scale_limit=2.0,
        prompt_primary_acts_scale_limit=2.0,
        prompt_batch_size_round_to=2,
    )

    schedule = runner._load_prompt_bucket_schedule(tokens)

    assert schedule == [
        {
            "prompt_indices": [0, 1, 2, 3],
            "seq_length": 64,
            "primary_acts_batch_size": 2,
        },
        {"prompt_indices": [4, 5], "seq_length": 64, "primary_acts_batch_size": 2},
        {"prompt_indices": [6, 7], "seq_length": 120, "primary_acts_batch_size": 1},
        {"prompt_indices": [8, 9], "seq_length": 120, "primary_acts_batch_size": 1},
        {"prompt_indices": [10, 11], "seq_length": 120, "primary_acts_batch_size": 1},
    ]


def test_auto_bucket_schedule_can_generate_shared_sidecars_from_pretokenized_dataset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset = Dataset.from_dict(
        {
            "input_ids": [
                [11, 12, 0, 0],
                [21, 22, 0, 0],
                [31, 32, 33, 34],
                [41, 42, 43, 44],
            ],
            "attention_mask": [
                [1, 1, 0, 0],
                [1, 1, 0, 0],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
            ],
        }
    )
    pretokenized_path = tmp_path / "rte_tokens"
    pretokenized_path.mkdir(parents=True)

    def fake_load_from_disk(path: str) -> Dataset:
        assert path == str(pretokenized_path)
        return dataset

    monkeypatch.setattr(
        "sae_dashboard.neuronpedia.prompt_datasets.load_from_disk",
        fake_load_from_disk,
    )

    runner = NeuronpediaRunner.__new__(NeuronpediaRunner)
    output_dir = tmp_path / "layer_9"
    runner.cfg = NeuronpediaRunnerConfig(
        sae_set="gemma-scope-2-1b-it-transcoders-all",
        sae_path="layer_9_width_262k_l0_small_affine",
        outputs_dir=str(output_dir),
        n_prompts_total=4,
        n_tokens_in_prompt=4,
        n_prompts_in_forward_pass=2,
        primary_acts_batch_size=1,
        huggingface_dataset_path="aps/super_glue",
        pretokenized_dataset_path=str(pretokenized_path),
        auto_prompt_bucket_schedule=True,
        prompt_bucket_ceilings=(2,),
        prompt_bucket_scale_limit=2.0,
        prompt_primary_acts_scale_limit=2.0,
        prompt_batch_size_round_to=2,
    )
    runner.model_id = "google/gemma-3-1b-it"
    runner.hook_name = "blocks.9.hook_mlp_in"
    runner.sae = SimpleNamespace(cfg=SimpleNamespace(d_sae=262144))

    runner._setup_output_directory()
    tokens = runner.get_tokens()
    schedule = runner._load_prompt_bucket_schedule(tokens)

    shared_tokens_file = pretokenized_path / "tokens_4.pt"
    assert runner.cfg.shared_tokens_file == str(shared_tokens_file)
    assert shared_tokens_file.is_file()
    assert (pretokenized_path / "tokens_4.effective_lengths.pt").is_file()
    assert (
        output_dir
        / "google_gemma-3-1b-it_gemma-scope-2-1b-it-transcoders-all_blocks.9.hook_mlp_in_262144"
        / "tokens_4.pt"
    ).is_file()
    assert tokens.tolist() == [
        [11, 12, 0, 0],
        [21, 22, 0, 0],
        [31, 32, 33, 34],
        [41, 42, 43, 44],
    ]
    metadata = json.loads(
        (pretokenized_path / "tokens_4.metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["deduplicate"] is True
    assert schedule == [
        {"prompt_indices": [0, 1], "seq_length": 2, "primary_acts_batch_size": 2},
        {"prompt_indices": [2, 3], "seq_length": 4, "primary_acts_batch_size": 1},
    ]


def test_auto_bucket_schedule_can_preserve_duplicate_rows_from_pretokenized_dataset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset = Dataset.from_dict(
        {
            "input_ids": [
                [11, 12, 0, 0],
                [11, 12, 0, 0],
                [31, 32, 33, 34],
                [41, 42, 43, 44],
            ],
            "attention_mask": [
                [1, 1, 0, 0],
                [1, 1, 0, 0],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
            ],
        }
    )
    pretokenized_path = tmp_path / "rte_tokens"
    pretokenized_path.mkdir(parents=True)

    def fake_load_from_disk(path: str) -> Dataset:
        assert path == str(pretokenized_path)
        return dataset

    monkeypatch.setattr(
        "sae_dashboard.neuronpedia.prompt_datasets.load_from_disk",
        fake_load_from_disk,
    )

    runner = NeuronpediaRunner.__new__(NeuronpediaRunner)
    runner.cfg = NeuronpediaRunnerConfig(
        sae_set="gemma-scope-2-1b-it-transcoders-all",
        sae_path="layer_9_width_262k_l0_small_affine",
        outputs_dir=str(tmp_path / "layer_9"),
        n_prompts_total=4,
        n_tokens_in_prompt=4,
        n_prompts_in_forward_pass=2,
        primary_acts_batch_size=1,
        huggingface_dataset_path="aps/super_glue",
        pretokenized_dataset_path=str(pretokenized_path),
        deduplicate_shared_prompt_tokens=False,
        strict_shared_prompt_count=True,
        auto_prompt_bucket_schedule=True,
        prompt_bucket_ceilings=(2,),
        prompt_bucket_scale_limit=2.0,
        prompt_primary_acts_scale_limit=2.0,
        prompt_batch_size_round_to=2,
    )
    runner.model_id = "google/gemma-3-1b-it"
    runner.hook_name = "blocks.9.hook_mlp_in"
    runner.sae = SimpleNamespace(cfg=SimpleNamespace(d_sae=262144))

    runner._setup_output_directory()
    tokens = runner.get_tokens()
    schedule = runner._load_prompt_bucket_schedule(tokens)

    assert tokens.tolist() == [
        [11, 12, 0, 0],
        [11, 12, 0, 0],
        [31, 32, 33, 34],
        [41, 42, 43, 44],
    ]
    metadata = json.loads(
        (pretokenized_path / "tokens_4.metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["tensor_shape"] == [4, 4]
    assert metadata["unique_rows"] == 3
    assert metadata["deduplicate"] is False
    assert schedule == [
        {"prompt_indices": [0, 1], "seq_length": 2, "primary_acts_batch_size": 2},
        {"prompt_indices": [2, 3], "seq_length": 4, "primary_acts_batch_size": 1},
    ]


def test_auto_bucket_schedule_uses_model_tokenizer_pad_token_when_metadata_omits_top_level_pad_token(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset = Dataset.from_dict(
        {
            "input_ids": [
                [11, 12, 0, 0],
                [21, 22, 0, 0],
                [31, 32, 33, 34],
                [41, 42, 43, 44],
            ]
        }
    )
    pretokenized_path = tmp_path / "rte_tokens"
    pretokenized_path.mkdir(parents=True)
    (pretokenized_path / "sae_lens.json").write_text(
        json.dumps({"custom": {"pad_token_id": 0}}),
        encoding="utf-8",
    )

    def fake_load_from_disk(path: str) -> Dataset:
        assert path == str(pretokenized_path)
        return dataset

    monkeypatch.setattr(
        "sae_dashboard.neuronpedia.prompt_datasets.load_from_disk",
        fake_load_from_disk,
    )

    runner = NeuronpediaRunner.__new__(NeuronpediaRunner)
    output_dir = tmp_path / "layer_9"
    runner.cfg = NeuronpediaRunnerConfig(
        sae_set="gemma-scope-2-1b-it-transcoders-all",
        sae_path="layer_9_width_262k_l0_small_affine",
        outputs_dir=str(output_dir),
        n_prompts_total=4,
        n_tokens_in_prompt=4,
        n_prompts_in_forward_pass=2,
        primary_acts_batch_size=1,
        huggingface_dataset_path="aps/super_glue",
        pretokenized_dataset_path=str(pretokenized_path),
        auto_prompt_bucket_schedule=True,
        prompt_bucket_ceilings=(2,),
        prompt_bucket_scale_limit=2.0,
        prompt_primary_acts_scale_limit=2.0,
        prompt_batch_size_round_to=2,
    )
    runner.model = cast(Any, SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0)))
    runner.model_id = "google/gemma-3-1b-it"
    runner.hook_name = "blocks.9.hook_mlp_in"
    runner.sae = SimpleNamespace(cfg=SimpleNamespace(d_sae=262144))

    runner._setup_output_directory()
    tokens = runner.get_tokens()
    schedule = runner._load_prompt_bucket_schedule(tokens)

    metadata = json.loads(
        (pretokenized_path / "tokens_4.metadata.json").read_text(encoding="utf-8")
    )

    assert tokens.tolist() == [
        [11, 12, 0, 0],
        [21, 22, 0, 0],
        [31, 32, 33, 34],
        [41, 42, 43, 44],
    ]
    assert metadata["pad_token_id"] == 0
    assert schedule == [
        {"prompt_indices": [0, 1], "seq_length": 2, "primary_acts_batch_size": 2},
        {"prompt_indices": [2, 3], "seq_length": 4, "primary_acts_batch_size": 1},
    ]


def test_auto_bucket_schedule_strict_count_errors_on_deduped_shortfall(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset = Dataset.from_dict(
        {
            "input_ids": [[11, 12, 0, 0], [11, 12, 0, 0], [31, 32, 33, 34]],
            "attention_mask": [[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1]],
        }
    )
    pretokenized_path = tmp_path / "rte_tokens"
    pretokenized_path.mkdir(parents=True)

    def fake_load_from_disk(path: str) -> Dataset:
        assert path == str(pretokenized_path)
        return dataset

    monkeypatch.setattr(
        "sae_dashboard.neuronpedia.prompt_datasets.load_from_disk",
        fake_load_from_disk,
    )

    runner = NeuronpediaRunner.__new__(NeuronpediaRunner)
    runner.cfg = NeuronpediaRunnerConfig(
        sae_set="gemma-scope-2-1b-it-transcoders-all",
        sae_path="layer_9_width_262k_l0_small_affine",
        outputs_dir=str(tmp_path / "layer_9"),
        n_prompts_total=3,
        n_tokens_in_prompt=4,
        huggingface_dataset_path="aps/super_glue",
        prompt_dataset_mode="load_from_disk",
        prompt_dataset_path=str(pretokenized_path),
        pretokenized_dataset_path=str(pretokenized_path),
        strict_shared_prompt_count=True,
        auto_prompt_bucket_schedule=True,
    )
    runner._prompt_dataset_materialization = load_prompt_dataset(
        resolve_prompt_dataset(
            PromptDatasetConfig(
                dataset_path=str(pretokenized_path),
                mode="load_from_disk",
            )
        )
    )

    with pytest.raises(ValueError, match="did not satisfy the requested prompt count"):
        runner._prepare_shared_tokens_from_prompt_dataset(
            pretokenized_path / "tokens_3.pt"
        )


def test_load_prompt_bucket_schedule_can_auto_bucket_from_quantile_ceilings(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "layer_10"
    output_dir.mkdir(parents=True)
    tokens = torch.tensor(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12],
            [13, 14, 15],
            [16, 17, 18],
            [19, 20, 21],
            [22, 23, 24],
            [25, 26, 27],
            [28, 29, 30],
            [31, 32, 33],
            [34, 35, 36],
        ],
        dtype=torch.long,
    )
    torch.save(tokens, output_dir / "tokens_12.pt")
    torch.save(
        torch.tensor(
            [40, 50, 60, 64, 64, 64, 65, 70, 80, 90, 110, 120], dtype=torch.int32
        ),
        output_dir / "tokens_12.effective_lengths.pt",
    )

    runner = NeuronpediaRunner.__new__(NeuronpediaRunner)
    runner.cfg = NeuronpediaRunnerConfig(
        sae_set="gemma-scope-2-1b-it-transcoders-all",
        sae_path="layer_10_width_262k_l0_small_affine",
        outputs_dir=str(output_dir),
        n_prompts_total=12,
        n_tokens_in_prompt=128,
        n_prompts_in_forward_pass=2,
        primary_acts_batch_size=1,
        shared_tokens_file=str(tmp_path / "shared_tokens.pt"),
        auto_prompt_bucket_schedule=True,
        prompt_bucket_scale_limit=2.0,
        prompt_primary_acts_scale_limit=2.0,
        prompt_batch_size_round_to=2,
    )

    schedule = runner._load_prompt_bucket_schedule(tokens)

    assert sorted(
        index
        for batch in schedule  # pyright: ignore[reportOptionalIterable]
        for index in batch["prompt_indices"]
    ) == list(range(12))
    assert {
        batch["seq_length"]
        for batch in schedule  # pyright: ignore[reportOptionalIterable]
    } == {
        60,
        64,
        80,
        120,
    }


def test_generate_tokens_requests_cpu_batches() -> None:
    class FakeActivationsStore:
        store_batch_size_prompts = 2

        def __init__(self) -> None:
            self.move_to_model_device_args: list[bool] = []

        def get_batch_tokens(self, move_to_model_device: bool = True):
            self.move_to_model_device_args.append(move_to_model_device)
            return torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)

    runner = NeuronpediaRunner.__new__(NeuronpediaRunner)
    runner.cfg = NeuronpediaRunnerConfig(
        sae_set="gpt2-small-res-jb",
        sae_path="blocks.5.hook_resid_pre",
        outputs_dir="test_outputs",
        n_prompts_total=2,
    )
    runner.cfg.shuffle_tokens = False

    fake_store = FakeActivationsStore()
    tokens = runner.generate_tokens(fake_store, n_prompts=2)  # pyright: ignore

    # The runner only passes move_to_model_device when the INSTALLED sae-lens exposes the seam
    # (module-level probe against the real ActivationsStore). Against a released sae-lens without
    # SAELens#721 the kwarg is correctly omitted and the fake records its own default, so asserting
    # [False] unconditionally would fail for anyone not using the coordinated fork. Either way the
    # tokens must come back on CPU, which is what this test actually guards.
    from sae_dashboard.neuronpedia.neuronpedia_runner import (
        _ACTIVATIONS_STORE_SUPPORTS_DEVICE_SEAM,
    )

    expected_device_args = (
        [False] if _ACTIVATIONS_STORE_SUPPORTS_DEVICE_SEAM else [True]
    )
    assert fake_store.move_to_model_device_args == expected_device_args
    assert tokens.device.type == "cpu"
    assert tokens.tolist() == [[1, 2, 3], [4, 5, 6]]


def test_write_converter_input_artifact(tmp_path: Path) -> None:
    runner = NeuronpediaRunner.__new__(NeuronpediaRunner)
    artifact_dir = tmp_path / "converter_inputs"
    runner.cfg = NeuronpediaRunnerConfig(
        sae_set="gpt2-small-res-jb",
        sae_path="blocks.5.hook_resid_pre",
        outputs_dir="test_outputs",
        converter_input_artifact_dir=str(artifact_dir),
    )
    runner.vocab_dict = {1: "token1"}
    runner.model = SimpleNamespace(  # pyright: ignore
        cfg=SimpleNamespace(d_vocab=50257)
    )
    runner.model_id = "gpt2-small"
    runner.layer = 5
    runner.hook_name = "blocks.5.hook_resid_pre"

    feature_data = SimpleNamespace(feature_data_dict={1: {"feature_index": 1}})

    artifact_path = runner._write_converter_input_artifact(feature_data, batch_num=3)

    assert artifact_path == artifact_dir / "converter_input_batch_3.pt"
    assert artifact_path.is_file()  # pyright: ignore[reportOptionalMemberAccess]
    snapshot = torch.load(artifact_path, weights_only=False)  # pyright: ignore
    assert snapshot["feature_data_dict"] == feature_data.feature_data_dict
    assert snapshot["runner_cfg"].converter_input_artifact_dir == str(artifact_dir)
    assert snapshot["vocab_dict"] == {1: "token1"}
    assert snapshot["model_d_vocab"] == 50257
    assert snapshot["batch_num"] == 3


def test_run_feature_batch_with_optional_profile_writes_trace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    perf_events: list[dict[str, object]] = []

    class _FakeSaeVisRunner:
        def __init__(self, cfg) -> None:
            self.cfg = cfg

        def adopt_token_string_caches(self, other) -> None:
            del other

        def run(self, encoder, model, tokens):
            assert encoder is runner.sae
            assert model is runner.model
            assert tokens.shape == (1, 2)
            return {"result": "profiled"}

    def _record(event: str, /, **fields: object) -> None:
        perf_events.append({"event": event, **fields})

    monkeypatch.setattr(
        "sae_dashboard.neuronpedia.neuronpedia_runner.SaeVisRunner",
        _FakeSaeVisRunner,
    )
    monkeypatch.setattr(
        "sae_dashboard.neuronpedia.neuronpedia_runner.log_perf_event",
        _record,
    )

    runner = NeuronpediaRunner.__new__(NeuronpediaRunner)
    trace_dir = tmp_path / "torch_profiles"
    runner.cfg = NeuronpediaRunnerConfig(
        sae_set="gpt2-small-res-jb",
        sae_path="blocks.5.hook_resid_pre",
        outputs_dir=str(tmp_path),
        torch_profile=True,
        torch_profile_dir=str(trace_dir),
    )
    runner.sae = SimpleNamespace()
    runner.model = SimpleNamespace()  # pyright: ignore

    result = runner._run_feature_batch_with_optional_profile(
        feature_vis_config_gpt=SimpleNamespace(),  # pyright: ignore
        tokens=torch.ones(1, 2, dtype=torch.long),
        feature_batch_count=4,
    )

    assert result == {"result": "profiled"}
    trace_event = next(
        event for event in perf_events if event.get("event") == "torch_profile_trace"
    )
    trace_path = Path(str(trace_event["path"]))
    assert trace_path == trace_dir / "batch-4.trace.json"
    assert trace_path.is_file()


# def test_add_prefix_suffix_to_tokens(neuronpedia_runner: NeuronpediaRunner) -> None:
#     # modify the config to add a prefix / suffix
#     neuronpedia_runner.cfg.prefix_tokens = [101, 102, 103]  # Example prefix tokens
#     neuronpedia_runner.cfg.suffix_tokens = [104, 105, 106]  # Example suffix tokens

#     # get the tokens
#     tokens = neuronpedia_runner.get_tokens()
#     tokens = neuronpedia_runner.add_prefix_suffix_to_tokens(tokens)

#     # check that the tokens have the prefix and suffix
#     assert torch.allclose(tokens[:, 1:4].cpu(), torch.tensor([101, 102, 103]))
#     assert torch.allclose(tokens[:, -3:].cpu(), torch.tensor([104, 105, 106]))

#     # assert the first token position is still the bos
#     assert torch.allclose(
#         tokens[:, 0].cpu(),
#         torch.tensor(
#             [neuronpedia_runner.model.to_single_token("<|endoftext|>")],
#             dtype=torch.int64,
#         ),
#     )


# def test_add_prefix_suffix_to_tokens_prepend_bos_false(
#     neuronpedia_runner: NeuronpediaRunner,
# ) -> None:
#     # modify the config to add a prefix / suffix
#     neuronpedia_runner.cfg.prefix_tokens = [101, 102, 103]  # Example prefix tokens
#     neuronpedia_runner.cfg.suffix_tokens = [104, 105, 106]  # Example suffix tokens

#     # get the tokens
#     neuronpedia_runner.sae.cfg.prepend_bos = False
#     tokens = neuronpedia_runner.get_tokens()
#     tokens = neuronpedia_runner.add_prefix_suffix_to_tokens(tokens)

#     # check that the tokens have the prefix and suffix
#     assert torch.allclose(tokens[:, 0:3].cpu(), torch.tensor([101, 102, 103]))
#     assert torch.allclose(tokens[:, -3:].cpu(), torch.tensor([104, 105, 106]))

#     # assert the first token position is still the bos
#     assert not torch.allclose(
#         tokens[:, 0].cpu(),
#         torch.tensor(
#             [neuronpedia_runner.model.to_single_token("<|endoftext|>")],
#             dtype=torch.int64,
#         ),
#     )
