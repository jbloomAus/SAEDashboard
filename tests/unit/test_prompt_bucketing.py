from __future__ import annotations

from sae_dashboard.neuronpedia.prompt_bucketing import derive_prompt_bucket_ceilings


def test_derive_prompt_bucket_ceilings_uses_quantiles_when_not_explicit() -> None:
    ceilings = derive_prompt_bucket_ceilings(
        [40, 50, 60, 64, 64, 64, 65, 70, 80, 90, 110, 120],
        max_context_size=128,
    )

    assert ceilings == (60, 64, 80, 120)


def test_derive_prompt_bucket_ceilings_preserves_explicit_values_and_max_effective_length() -> (
    None
):
    ceilings = derive_prompt_bucket_ceilings(
        [40, 50, 60, 64, 64, 64, 65, 70, 80, 90, 110, 120],
        max_context_size=128,
        explicit_bucket_ceilings=(64,),
    )

    assert ceilings == (64, 120)
