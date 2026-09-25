# pyright: basic, reportPrivateImportUsage=false
import pytest
import torch

import sae_dashboard.feature_data_generator as feature_data_generator
from sae_dashboard.feature_data_generator import FeatureDataGenerator
from sae_dashboard.neuronpedia.legacy import utils_fns as legacy_utils_fns
from sae_dashboard.neuronpedia.legacy.feature_data_generator import (
    LegacyFeatureDataGenerator,
)
from sae_dashboard.sae_vis_data import SaeVisConfig
from sae_dashboard.sae_vis_runner import FeatureDataGeneratorFactory
from sae_dashboard.utils_fns import resolve_correlation_accumulation_device


def test_resolve_correlation_accumulation_device_cpu_policy() -> None:
    assert resolve_correlation_accumulation_device("cuda", "cpu") == torch.device("cpu")


def test_resolve_correlation_accumulation_device_rejects_unavailable_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(ValueError, match="requires CUDA"):
        resolve_correlation_accumulation_device("cpu", "cuda")


def test_get_feature_data_uses_configured_correlation_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_devices: list[torch.device] = []

    class CapturingRollingCorrCoef:
        def __init__(
            self,
            indices: list[int] | None = None,
            with_self: bool = False,
            dtype: torch.dtype = torch.float32,
            device: torch.device = torch.device("cpu"),
            **_: object,
        ) -> None:
            captured_devices.append(device)

    generator = FeatureDataGenerator.__new__(FeatureDataGenerator)
    generator.cfg = SaeVisConfig(
        hook_point="blocks.0.hook_resid_pre",
        features=[0],
        device="cpu",
        correlation_accumulation_device="cpu",
    )
    generator.token_minibatches = []
    generator.full_sequence_length = 0
    generator.encoder = type(  # pyright: ignore
        "Encoder", (), {"W_dec": torch.ones((1, 3))}
    )()
    generator.model = object()  # pyright: ignore

    monkeypatch.setattr(
        feature_data_generator, "RollingCorrCoef", CapturingRollingCorrCoef
    )
    monkeypatch.setattr(
        feature_data_generator,
        "to_resid_direction",
        lambda feature_out_dir, model: feature_out_dir,
    )

    generator.get_feature_data([0])

    assert captured_devices == [torch.device("cpu"), torch.device("cpu")]


def test_batch_tokens_uses_prompt_minibatch_schedule() -> None:
    tokens = torch.tensor(
        [
            [1, 2, 3, 4, 0, 0],
            [5, 6, 7, 8, 9, 0],
            [10, 11, 12, 0, 0, 0],
            [13, 14, 15, 16, 17, 18],
        ],
        dtype=torch.long,
    )
    generator = FeatureDataGenerator.__new__(FeatureDataGenerator)
    generator.cfg = SaeVisConfig(
        hook_point="blocks.0.hook_resid_pre",
        features=[0],
        prompt_minibatch_schedule=[
            {
                "prompt_indices": [2, 3],
                "seq_length": 4,
                "primary_acts_batch_size": 1,
            },
            {
                "prompt_indices": [0, 1],
                "seq_length": 5,
                "primary_acts_batch_size": 2,
            },
        ],
    )

    minibatches = FeatureDataGenerator.batch_tokens(generator, tokens)

    assert [minibatch.prompt_indices for minibatch in minibatches] == [(2, 3), (0, 1)]
    assert [tuple(minibatch.tokens.shape) for minibatch in minibatches] == [
        (2, 4),
        (2, 5),
    ]
    assert [minibatch.primary_acts_batch_size for minibatch in minibatches] == [1, 2]
    assert minibatches[0].tokens.tolist() == [[10, 11, 12, 0], [13, 14, 15, 16]]
    assert minibatches[1].tokens.tolist() == [[1, 2, 3, 4, 0], [5, 6, 7, 8, 9]]


def test_scatter_feature_act_chunk_restores_original_prompt_order() -> None:
    destination = torch.zeros((4, 6, 2), dtype=torch.bfloat16)
    shorter_bucket_chunk = torch.full((2, 6, 2), 1, dtype=torch.bfloat16)
    longer_bucket_chunk = torch.full((2, 6, 2), 2, dtype=torch.bfloat16)

    FeatureDataGenerator._scatter_feature_act_chunk(
        destination,
        shorter_bucket_chunk,
        prompt_indices=(2, 3),
    )
    FeatureDataGenerator._scatter_feature_act_chunk(
        destination,
        longer_bucket_chunk,
        prompt_indices=(0, 1),
    )

    assert destination[0].eq(2).all()
    assert destination[1].eq(2).all()
    assert destination[2].eq(1).all()
    assert destination[3].eq(1).all()


def test_pad_sequence_tensor_zero_fills_trimmed_tail() -> None:
    sequence_tensor = torch.tensor(
        [[[1.0], [2.0], [3.0]]],
        dtype=torch.float32,
    )

    padded = FeatureDataGenerator._pad_sequence_tensor(
        sequence_tensor,
        target_seq_len=5,
    )

    assert tuple(padded.shape) == (1, 5, 1)
    assert padded[:, :3].tolist() == sequence_tensor.tolist()
    assert padded[:, 3:].tolist() == [[[0.0], [0.0]]]


def test_transfer_feature_acts_for_output_preserves_legacy_precision() -> None:
    generator = LegacyFeatureDataGenerator.__new__(LegacyFeatureDataGenerator)
    generator.cfg = SaeVisConfig(
        hook_point="blocks.0.hook_resid_pre",
        features=[0],
        dashboard_output_format="legacy_json",
        sequence_selection_backend="legacy",
    )

    feature_acts = torch.tensor([5091.77587890625], dtype=torch.float32)

    transferred = generator._transfer_feature_acts_for_output(feature_acts)

    assert transferred.device == feature_acts.device
    assert transferred.dtype == torch.float32
    assert transferred.tolist() == pytest.approx(feature_acts.tolist())


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Requires CUDA to verify legacy device preservation.",
)
def test_transfer_feature_acts_for_output_preserves_legacy_cuda_device() -> None:
    generator = LegacyFeatureDataGenerator.__new__(LegacyFeatureDataGenerator)
    generator.cfg = SaeVisConfig(
        hook_point="blocks.0.hook_resid_pre",
        features=[0],
        dashboard_output_format="legacy_json",
        sequence_selection_backend="legacy",
        device="cuda",
    )

    feature_acts = torch.tensor([1.0], dtype=torch.float32, device="cuda")

    transferred = generator._transfer_feature_acts_for_output(feature_acts)

    assert transferred.device == feature_acts.device
    assert transferred.dtype == feature_acts.dtype
    assert transferred.tolist() == pytest.approx(feature_acts.tolist())


def test_feature_data_generator_factory_routes_legacy_to_legacy_subclass() -> None:
    cfg = SaeVisConfig(
        hook_point="blocks.0.hook_resid_pre",
        features=[0],
        dashboard_output_format="legacy_json",
        sequence_selection_backend="legacy",
    )

    assert (
        FeatureDataGeneratorFactory.resolve_feature_data_generator_cls(cfg)
        is LegacyFeatureDataGenerator
    )


def test_legacy_feature_data_generator_scatter_matches_detached_baseline() -> None:
    generator = LegacyFeatureDataGenerator.__new__(LegacyFeatureDataGenerator)
    generator.cfg = SaeVisConfig(
        hook_point="blocks.0.hook_resid_pre",
        features=[0],
        dashboard_output_format="legacy_json",
        sequence_selection_backend="legacy",
    )

    destination = torch.zeros(2, 1, 1, dtype=torch.float32)
    chunk = torch.tensor([[[1.0]], [[2.0]]], dtype=torch.float32)
    generator._scatter_feature_act_chunk(destination, chunk, prompt_indices=(0, 1))

    assert destination[0][0][0].item() == pytest.approx(1.0)
    assert destination[1][0][0].item() == pytest.approx(2.0)


def test_legacy_feature_data_generator_keeps_float32_output_dtype() -> None:
    generator = LegacyFeatureDataGenerator.__new__(LegacyFeatureDataGenerator)
    generator.cfg = SaeVisConfig(
        hook_point="blocks.0.hook_resid_pre",
        features=[0],
        dashboard_output_format="legacy_json",
        sequence_selection_backend="legacy",
    )
    x = torch.randn(2, 4, 256)
    result = generator._transfer_feature_acts_for_output(x)
    assert result.dtype == torch.float32
    assert result is x


def test_legacy_feature_data_generator_uses_legacy_corrcoef() -> None:
    generator = LegacyFeatureDataGenerator.__new__(LegacyFeatureDataGenerator)
    generator.cfg = SaeVisConfig(
        hook_point="blocks.0.hook_resid_pre",
        features=[0],
        dashboard_output_format="legacy_json",
        sequence_selection_backend="legacy",
    )

    corrcoef_neurons = generator._create_corrcoef_neurons(
        correlation_device=torch.device("cpu"),
    )
    corrcoef_encoder = generator._create_corrcoef_encoder(
        feature_indices=[0],
        correlation_device=torch.device("cpu"),
    )

    assert isinstance(corrcoef_neurons, legacy_utils_fns.RollingCorrCoef)
    assert isinstance(corrcoef_encoder, legacy_utils_fns.RollingCorrCoef)


def test_legacy_feature_encode_path_matches_baseline_architecture_rule() -> None:
    class FakeEncoderConfig:
        @staticmethod
        def architecture() -> str:
            return "skip_transcoder"

    fake_encoder = type(
        "FakeEncoder",
        (),
        {
            "cfg": FakeEncoderConfig(),
            "activation_fn": feature_data_generator.TopK(k=1),
        },
    )()

    current_generator = FeatureDataGenerator.__new__(FeatureDataGenerator)
    setattr(current_generator, "encoder", fake_encoder)

    legacy_generator = LegacyFeatureDataGenerator.__new__(LegacyFeatureDataGenerator)
    setattr(legacy_generator, "encoder", fake_encoder)
    legacy_generator.cfg = SaeVisConfig(
        hook_point="blocks.0.hook_resid_pre",
        features=[0],
        dashboard_output_format="legacy_json",
        sequence_selection_backend="legacy",
    )

    assert current_generator._uses_full_feature_encode_path()
    assert not legacy_generator._uses_full_feature_encode_path()


def test_transfer_feature_acts_for_output_downcasts_non_legacy_paths() -> None:
    generator = FeatureDataGenerator.__new__(FeatureDataGenerator)
    generator.cfg = SaeVisConfig(
        hook_point="blocks.0.hook_resid_pre",
        features=[0],
        dashboard_output_format="columnar",
        sequence_selection_backend="columnar_gpu",
    )

    feature_acts = torch.tensor([5091.77587890625], dtype=torch.float32)

    transferred = generator._transfer_feature_acts_for_output(feature_acts)

    assert transferred.device.type == "cpu"
    assert transferred.dtype == torch.bfloat16
    assert transferred.to(torch.float32).tolist() == pytest.approx([5088.0])
