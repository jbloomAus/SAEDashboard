# pyright: basic, reportPrivateImportUsage=false
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from sae_dashboard.neuronpedia import neuronpedia_converter as converter_module
from sae_dashboard.neuronpedia.neuronpedia_dashboard import NeuronpediaDashboardFeature
from sae_dashboard.neuronpedia.neuronpedia_runner_config import NeuronpediaRunnerConfig

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


def test_convert_to_np_json_matches_reference_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature = NeuronpediaDashboardFeature(
        feature_index=np.int64(7),  # pyright: ignore
        neg_values=[np.float32(-1.25)],  # pyright: ignore
        pos_values=[np.float32(2.5)],  # pyright: ignore
        logits_hist_data_bar_heights=[1, 2],
        logits_hist_data_bar_values=[
            np.float32(-0.5),
            np.float32(0.5),
        ],  # pyright: ignore
        freq_hist_data_bar_heights=[2, 1],
        freq_hist_data_bar_values=[
            np.float32(0.25),
            np.float32(0.75),
        ],  # pyright: ignore
        activations=[
            {
                "bin_min": np.float32(0.0),
                "bin_max": np.float32(1.0),
                "bin_contains": np.float32(0.5),
                "tokens": ["Alpha", "Beta"],
                "values": [np.float32(0.125), np.float32(0.875)],
                "qualifying_token_index": np.int64(1),
            }
        ],
        vector=np.array([np.float32(1.25), np.float32(2.5)]),  # pyright: ignore
    )
    runner_cfg = NeuronpediaRunnerConfig(
        sae_set="gemma-scope-2-1b-it-transcoders-all",
        sae_path="layer_3_width_262k_l0_small_affine",
        outputs_dir="/tmp/neuronpedia_converter_test",
        model_id="gemma-3-1b-it",
        layer=3,
        np_set_name="gemmascope-2-transcoder-262k-rte",
    )

    monkeypatch.setattr(
        converter_module.NeuronpediaConverter,
        "_process_features",
        staticmethod(lambda *_args, **_kwargs: [feature]),
    )
    fake_vis_data = cast(Any, SimpleNamespace(feature_data_dict={7: object()}))
    payload = converter_module.NeuronpediaConverter.convert_to_np_json(
        model=None,  # pyright: ignore
        vis_data=fake_vis_data,
        np_cfg=runner_cfg,
        vocab_dict={},
    )

    assert payload == (FIXTURE_DIR / "neuronpedia_reference_batch.json").read_text(
        encoding="utf-8"
    )


def test_convert_to_np_json_deterministic_matches_reference_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature = NeuronpediaDashboardFeature(
        feature_index=np.int64(7),  # pyright: ignore
        neg_values=[np.float32(-1.25)],  # pyright: ignore
        pos_values=[np.float32(2.5)],  # pyright: ignore
        logits_hist_data_bar_heights=[1, 2],
        logits_hist_data_bar_values=[
            np.float32(-0.5),
            np.float32(0.5),
        ],  # pyright: ignore
        freq_hist_data_bar_heights=[2, 1],
        freq_hist_data_bar_values=[
            np.float32(0.25),
            np.float32(0.75),
        ],  # pyright: ignore
        activations=[
            {
                "bin_min": np.float32(0.0),
                "bin_max": np.float32(1.0),
                "bin_contains": np.float32(0.5),
                "tokens": ["Alpha", "Beta"],
                "values": [np.float32(0.125), np.float32(0.875)],
                "qualifying_token_index": np.int64(1),
            }
        ],
        vector=np.array([np.float32(1.25), np.float32(2.5)]),  # pyright: ignore
    )
    runner_cfg = NeuronpediaRunnerConfig(
        sae_set="gemma-scope-2-1b-it-transcoders-all",
        sae_path="layer_3_width_262k_l0_small_affine",
        outputs_dir="/tmp/neuronpedia_converter_test",
        model_id="gemma-3-1b-it",
        layer=3,
        np_set_name="gemmascope-2-transcoder-262k-rte",
    )

    monkeypatch.setattr(
        converter_module.NeuronpediaConverter,
        "_process_features",
        staticmethod(lambda *_args, **_kwargs: [feature]),
    )
    fake_vis_data = cast(Any, SimpleNamespace(feature_data_dict={7: object()}))

    payload = converter_module.NeuronpediaConverter.convert_to_np_json(
        model=None,  # pyright: ignore
        vis_data=fake_vis_data,
        np_cfg=runner_cfg,
        vocab_dict={},
        deterministic_json=True,
    )

    assert payload == (FIXTURE_DIR / "neuronpedia_reference_batch.json").read_text(
        encoding="utf-8"
    )


def test_encode_batch_payload_deterministic_matches_reference_batch() -> None:
    fixture_text = (FIXTURE_DIR / "neuronpedia_reference_batch.json").read_text(
        encoding="utf-8"
    )
    batch_payload = json.loads(fixture_text)

    payload = converter_module.NeuronpediaConverter.encode_batch_payload(
        batch_payload,
        deterministic_json=True,
    )

    assert payload == fixture_text


def test_encode_batch_payload_matches_reference_batch() -> None:
    fixture_text = (FIXTURE_DIR / "neuronpedia_reference_batch.json").read_text(
        encoding="utf-8"
    )
    batch_payload = json.loads(fixture_text)

    payload = converter_module.NeuronpediaConverter.encode_batch_payload(batch_payload)

    assert payload == fixture_text


def test_convert_preserved_snapshot_to_np_json_reuses_full_converter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature = NeuronpediaDashboardFeature(
        feature_index=np.int64(7),  # pyright: ignore
        neg_values=[np.float32(-1.25)],  # pyright: ignore
        pos_values=[np.float32(2.5)],  # pyright: ignore
        logits_hist_data_bar_heights=[1, 2],
        logits_hist_data_bar_values=[
            np.float32(-0.5),
            np.float32(0.5),
        ],  # pyright: ignore
        freq_hist_data_bar_heights=[2, 1],
        freq_hist_data_bar_values=[
            np.float32(0.25),
            np.float32(0.75),
        ],  # pyright: ignore
        activations=[],
    )
    runner_cfg = NeuronpediaRunnerConfig(
        sae_set="gemma-scope-2-1b-it-transcoders-all",
        sae_path="layer_3_width_262k_l0_small_affine",
        outputs_dir="/tmp/neuronpedia_converter_test",
        model_id="gemma-3-1b-it",
        layer=3,
        np_set_name="gemmascope-2-transcoder-262k-rte",
    )
    fake_vis_data = cast(Any, SimpleNamespace(feature_data_dict={7: object()}))

    def fake_process_features(
        model: Any, data_dict: Any, np_cfg: Any, vocab_dict: Any, original_vectors: Any
    ) -> list[Any]:
        assert model.cfg.d_vocab == 256000
        assert data_dict == fake_vis_data.feature_data_dict
        assert np_cfg is runner_cfg
        assert vocab_dict == {7: "token7"}
        assert original_vectors is None
        return [feature]

    monkeypatch.setattr(
        converter_module.NeuronpediaConverter,
        "_process_features",
        staticmethod(fake_process_features),
    )

    snapshot = {
        "feature_data_dict": fake_vis_data.feature_data_dict,
        "runner_cfg": runner_cfg,
        "vocab_dict": {7: "token7"},
        "model_d_vocab": 256000,
    }

    payload = (
        converter_module.NeuronpediaConverter.convert_preserved_snapshot_to_np_json(
            snapshot
        )
    )

    assert json.loads(payload)["features"][0]["feature_index"] == 7


def test_create_activation_trims_trailing_pad_tokens() -> None:
    fake_model = cast(
        Any,
        SimpleNamespace(
            cfg=SimpleNamespace(d_vocab=8),
            tokenizer=SimpleNamespace(pad_token_id=0),
        ),
    )
    sequence = SimpleNamespace(
        original_index=5,
        token_ids=[1, 2, 0, 0],
        feat_acts=[0.125, 0.875, 0.0, 0.0],
        qualifying_token_index=2,
    )
    feature_data = SimpleNamespace(
        dfa_data={
            5: {
                "dfaValues": [0.0, 0.25, 0.75, 0.0, 0.0],
                "dfaTargetIndex": 2,
            }
        }
    )

    activation = converter_module.NeuronpediaConverter._create_activation(
        sequence,
        0.0,
        1.0,
        0.5,
        feature_data,  # pyright: ignore
        fake_model,
        {0: "<pad>", 1: "Alpha", 2: "Beta"},
        feature_index=7,
    )

    assert activation.tokens == ["Alpha", "Beta"]
    assert activation.values == [0.125, 0.875]
    assert activation.qualifying_token_index == 1
    assert activation.dfa_values == [0.25, 0.75]
    assert activation.dfa_maxValue == 0.75
    assert activation.dfa_targetIndex == 1


def test_create_activation_can_preserve_trailing_pad_tokens() -> None:
    fake_model = cast(
        Any,
        SimpleNamespace(
            cfg=SimpleNamespace(d_vocab=8),
            tokenizer=SimpleNamespace(pad_token_id=0),
        ),
    )
    sequence = SimpleNamespace(
        original_index=5,
        token_ids=[1, 2, 0, 0],
        feat_acts=[0.125, 0.875, 0.0, 0.0],
        qualifying_token_index=2,
    )
    feature_data = SimpleNamespace(
        dfa_data={
            5: {
                "dfaValues": [0.0, 0.25, 0.75, 0.0, 0.0],
                "dfaTargetIndex": 2,
            }
        }
    )

    activation = converter_module.NeuronpediaConverter._create_activation(
        sequence,
        0.0,
        1.0,
        0.5,
        feature_data,  # pyright: ignore
        fake_model,
        {0: "<pad>", 1: "Alpha", 2: "Beta"},
        feature_index=7,
        trim_trailing_pad_tokens=False,
    )

    assert activation.tokens == ["Alpha", "Beta", "<pad>", "<pad>"]
    assert activation.values == [0.125, 0.875, 0.0, 0.0]
    assert activation.dfa_values == [0.25, 0.75, 0.0, 0.0]
