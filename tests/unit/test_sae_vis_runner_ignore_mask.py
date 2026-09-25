# pyright: basic, reportPrivateImportUsage=false
from types import SimpleNamespace
from typing import cast

import torch

from sae_dashboard.components import FeatureTablesData, LogitsTableData
from sae_dashboard.feature_data import FeatureData
from sae_dashboard.sae_vis_data import SaeVisConfig, SaeVisData
from sae_dashboard.sae_vis_runner import SaeVisRunner


class _FakeFeatureDataGenerator:
    def get_feature_data(self, features, progress):
        all_feat_acts = torch.tensor([[[1.0], [1.0]]], dtype=torch.float32)
        feature_resid_dir = torch.tensor([[1.0]], dtype=torch.float32)
        return all_feat_acts, None, feature_resid_dir, None, None, None, None


class _FakeSequenceDataGenerator:
    def __init__(self, cfg, tokens, W_U):
        self.cfg = cfg
        self.tokens = tokens
        self.W_U = W_U

    def get_sequences_data(
        self,
        feat_acts,
        feat_logits,
        resid_post,
        feature_resid_dir,
        selection_mask=None,
        selection_backend=None,
    ):
        return []


class _FakeEncoder:
    def __init__(self) -> None:
        self.hook_z_reshaping_mode = False

    def fold_W_dec_norm(self) -> None:
        return None


def test_legacy_ignore_mask_preserves_baseline_histogram_density(
    monkeypatch,
) -> None:
    cfg = SaeVisConfig(
        hook_point="blocks.0.hook_resid_pre",
        features=[0],
        minibatch_size_features=1,
        minibatch_size_tokens=1,
        quantile_feature_batch_size=1,
        device="cpu",
        dtype="float32",
        ignore_tokens={0},
    )
    tokens = torch.tensor([[7, 0]], dtype=torch.long)

    monkeypatch.setattr(
        "sae_dashboard.sae_vis_runner.FeatureDataGeneratorFactory.create",
        lambda cfg, model, encoder, tokens: _FakeFeatureDataGenerator(),
    )
    monkeypatch.setattr(
        "sae_dashboard.sae_vis_runner.SequenceDataGenerator",
        _FakeSequenceDataGenerator,
    )
    monkeypatch.setattr(
        "sae_dashboard.sae_vis_runner.LegacySequenceDataGenerator",
        _FakeSequenceDataGenerator,
    )
    monkeypatch.setattr(
        "sae_dashboard.sae_vis_runner.get_features_table_data",
        lambda **kwargs: {
            name: [value] for name, value in FeatureTablesData().__dict__.items()
        },
    )
    monkeypatch.setattr(
        "sae_dashboard.neuronpedia.legacy.sae_vis_runner.get_features_table_data",
        lambda **kwargs: {
            name: [value] for name, value in FeatureTablesData().__dict__.items()
        },
    )
    monkeypatch.setattr(
        "sae_dashboard.sae_vis_runner.get_logits_table_data",
        lambda **kwargs: LogitsTableData(),
    )
    monkeypatch.setattr(
        "sae_dashboard.neuronpedia.legacy.sae_vis_runner.get_logits_table_data",
        lambda **kwargs: LogitsTableData(),
    )

    fake_model = SimpleNamespace(W_U=torch.tensor([[1.0]], dtype=torch.float32))
    fake_encoder = _FakeEncoder()

    sae_vis_data = cast(
        SaeVisData,
        SaeVisRunner(cfg).run(
            encoder=fake_encoder,  # pyright: ignore
            model=fake_model,  # pyright: ignore
            tokens=tokens,
        ),
    )

    assert sae_vis_data.feature_stats.max == [1.0]
    assert sae_vis_data.feature_stats.frac_nonzero == [1.0]
    assert (
        sae_vis_data.feature_data_dict[0].acts_histogram_data.title
        == "ACTIVATIONS<br>DENSITY = 50.000%"
    )


def test_SaeVisRunner_routes_legacy_through_compatibility_module(
    monkeypatch,
) -> None:
    cfg = SaeVisConfig(
        hook_point="blocks.0.hook_resid_pre",
        features=[0],
        minibatch_size_features=1,
        minibatch_size_tokens=1,
        quantile_feature_batch_size=1,
        device="cpu",
        dtype="float32",
    )
    tokens = torch.tensor([[7, 0]], dtype=torch.long)
    fake_model = SimpleNamespace(W_U=torch.tensor([[1.0]], dtype=torch.float32))
    fake_encoder = _FakeEncoder()
    calls: list[object] = []

    monkeypatch.setattr(
        "sae_dashboard.sae_vis_runner.FeatureDataGeneratorFactory.create",
        lambda cfg, model, encoder, tokens: object(),
    )

    class _FakeLegacySequenceDataGenerator:
        def __init__(self, cfg, tokens, W_U) -> None:
            del cfg, tokens, W_U

    def _fake_run_legacy_object_feature_batch(runner, **kwargs):
        calls.append(kwargs["sequence_data_generator"])
        return SaeVisData(
            cfg=runner.cfg,
            feature_data_dict={99: FeatureData()},
        )

    monkeypatch.setattr(
        "sae_dashboard.sae_vis_runner.LegacySequenceDataGenerator",
        _FakeLegacySequenceDataGenerator,
    )
    monkeypatch.setattr(
        "sae_dashboard.neuronpedia.legacy.sae_vis_runner.run_object_feature_batch",
        _fake_run_legacy_object_feature_batch,
    )

    sae_vis_data = cast(
        SaeVisData,
        SaeVisRunner(cfg).run(
            encoder=fake_encoder,  # pyright: ignore
            model=fake_model,  # pyright: ignore
            tokens=tokens,
        ),
    )

    assert list(sae_vis_data.feature_data_dict) == [99]
    assert len(calls) == 1
    assert isinstance(calls[0], _FakeLegacySequenceDataGenerator)
