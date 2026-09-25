from __future__ import annotations

import math
from collections.abc import Sequence

DEFAULT_PROMPT_BUCKET_TARGET_COUNT = 4


def derive_prompt_bucket_ceilings(
    effective_lengths: Sequence[int],
    *,
    max_context_size: int,
    explicit_bucket_ceilings: Sequence[int] = (),
    target_bucket_count: int = DEFAULT_PROMPT_BUCKET_TARGET_COUNT,
) -> tuple[int, ...]:
    """Return inclusive prompt-length ceilings for dynamic batch scaling.

    Explicit ceilings still win when provided. Otherwise the runner derives a small set of ceilings from the staged
    effective-length distribution so auto prompt bucketing adapts to the current prompt cache instead of relying on
    hardcoded global defaults.
    """

    valid_lengths = sorted(
        int(length)
        for length in effective_lengths
        if 0 < int(length) <= max_context_size
    )
    if not valid_lengths:
        return (max_context_size,)

    max_effective_length = valid_lengths[-1]
    explicit = {
        int(value)
        for value in explicit_bucket_ceilings
        if 0 < int(value) <= max_effective_length
    }
    if explicit:
        explicit.add(max_effective_length)
        return tuple(sorted(explicit))

    bucket_count = min(max(1, target_bucket_count), len(valid_lengths))
    if bucket_count <= 1:
        return (max_effective_length,)

    ceilings = {
        _higher_quantile(valid_lengths, quantile=step / bucket_count)
        for step in range(1, bucket_count)
    }
    ceilings.add(max_effective_length)
    return tuple(sorted(ceilings))


def _higher_quantile(sorted_values: Sequence[int], *, quantile: float) -> int:
    if not sorted_values:
        raise ValueError("sorted_values must not be empty")
    rank = max(1, math.ceil(quantile * len(sorted_values)))
    return int(sorted_values[rank - 1])
