"""Unit tests for the run-persistent feature-acts accumulation buffer.

The generator reuses one accumulation buffer across feature batches (identical shapes)
so host-staged runs at large total prompt counts do not free a multi-GiB buffer and
immediately allocate an identical one at every feature-batch transition (allocator
arenas retain the freed pages, transiently double-buffering the run's largest
allocation — the 24,576-prompt host OOM-kill signature).
"""

import torch

from sae_dashboard.feature_data_generator import FeatureDataGenerator


def _bare_generator() -> FeatureDataGenerator:
    generator = FeatureDataGenerator.__new__(FeatureDataGenerator)
    generator._feature_acts_buffer = None
    return generator


def test_acquire_reuses_buffer_for_matching_shape() -> None:
    generator = _bare_generator()
    first = generator._acquire_feature_acts_buffer(
        8, 16, 4, dtype=torch.bfloat16, device="cpu"
    )
    second = generator._acquire_feature_acts_buffer(
        8, 16, 4, dtype=torch.bfloat16, device="cpu"
    )
    assert (
        first.data_ptr() == second.data_ptr()
    ), "matching shape must reuse the same storage"
    assert second.shape == (8, 16, 4)


def test_acquire_releases_before_reallocating_on_shape_change() -> None:
    generator = _bare_generator()
    first = generator._acquire_feature_acts_buffer(
        8, 16, 4, dtype=torch.bfloat16, device="cpu"
    )
    first_ptr = first.data_ptr()
    del first  # drop the test's own reference so the generator holds the only one
    second = generator._acquire_feature_acts_buffer(
        8, 16, 2, dtype=torch.bfloat16, device="cpu"
    )
    assert second.shape == (8, 16, 2)
    assert generator._feature_acts_buffer is second
    # NOTE: the new buffer MAY land at the old buffer's address — that address reuse is
    # exactly the point of releasing before reallocating (no transient double residency),
    # so no data_ptr inequality is asserted here.
    del first_ptr


def test_acquire_reallocates_on_dtype_or_device_change() -> None:
    generator = _bare_generator()
    first = generator._acquire_feature_acts_buffer(
        4, 8, 2, dtype=torch.bfloat16, device="cpu"
    )
    second = generator._acquire_feature_acts_buffer(
        4, 8, 2, dtype=torch.float32, device="cpu"
    )
    assert second.dtype == torch.float32
    assert first.data_ptr() != second.data_ptr()


def test_reused_buffer_is_fully_overwritten_between_batches() -> None:
    """Simulate two feature batches scattering into the reused buffer: the second
    batch's scatter of every prompt row must leave no first-batch values behind."""
    generator = _bare_generator()
    buffer = generator._acquire_feature_acts_buffer(
        6, 4, 3, dtype=torch.float32, device="cpu"
    )
    buffer.fill_(1.0)  # batch 0 contents

    again = generator._acquire_feature_acts_buffer(
        6, 4, 3, dtype=torch.float32, device="cpu"
    )
    assert again.data_ptr() == buffer.data_ptr()
    # batch 1: scatter per token-minibatch covering ALL prompt rows, as get_feature_data does
    for prompt_indices in ((0, 1, 2), (3, 4, 5)):
        chunk = torch.full((len(prompt_indices), 4, 3), 2.0)
        again[list(prompt_indices)] = chunk
    assert torch.equal(
        again, torch.full((6, 4, 3), 2.0)
    ), "reuse must not leak prior-batch values"
