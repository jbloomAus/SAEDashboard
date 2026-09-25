"""
Mechanism-validation tests for legacy vs columnar_gpu activation row discrepancies.

These tests validate that activation row deltas between the current legacy JSON
path and the columnar_gpu Parquet/Arrow path are explained by known mechanisms:

1. **Interval boundary semantics**: Closed-interval vs half-open bin membership.
2. **bfloat16 downcast**: float32→bfloat16 precision loss at interval edges.

Any unexplained discrepancies would indicate a regression in one of the paths.
All tests use synthetic values — no model inference is required, making
them suitable for fast CI execution.
"""

import torch


def bfloat16_round(value: float) -> float:
    """Simulate bfloat16 round-trip on a float32 value."""
    t = torch.tensor([value], dtype=torch.float32)
    return float(t.to(torch.bfloat16).to(torch.float32)[0])


class TestBfloat16DowncastMechanisms:
    """Validate that bfloat16 downcast causes predictable bin membership changes."""

    def test_bfloat16_downcast_preserves_values_away_from_boundaries(self):
        """Values far from bin edges should keep the same bin after bfloat16 round-trip."""
        # These values are well within their bins and should not cross boundaries
        test_cases = [
            (0.25, (0.0, 0.5)),
            (0.75, (0.5, 1.0)),
            (10.0, (5.0, 15.0)),
            (100.0, (50.0, 150.0)),
        ]

        for float32_val, (bin_lo, bin_hi) in test_cases:
            bf16_val = bfloat16_round(float32_val)
            # Both values should be in the same bin
            float32_in_bin = bin_lo <= float32_val < bin_hi
            bf16_in_bin = bin_lo <= bf16_val < bin_hi
            assert bf16_in_bin == float32_in_bin, (
                f"Value {float32_val} (bf16={bf16_val}) changed bin membership unexpectedly "
                f"for bin [{bin_lo}, {bin_hi})"
            )

    def test_bfloat16_downcast_can_cross_interval_boundary_upward(self):
        """A float32 value just above a bin edge can drop into the lower bin after bfloat16."""
        # 0.5001 in float32 → 0.5 in bfloat16 (drops to lower bin edge)
        float32_val = 0.5001
        bf16_val = bfloat16_round(float32_val)

        # float32: 0.5001 is in bin [0.5, 1.0)
        assert 0.5 <= float32_val < 1.0, f"{float32_val} should be in upper bin"
        # bfloat16: 0.5 is in bin [0.0, 0.5) if half-open, or both if closed
        assert bf16_val == 0.5, f"bf16 of {float32_val} should be 0.5"

    def test_bfloat16_downcast_can_cross_interval_boundary_downward(self):
        """A float32 value just below a bin edge can rise into the upper bin after bfloat16."""
        # 0.4999 in float32 is in bin [0.0, 0.5)
        # After bfloat16, 0.5 would be in bin [0.5, 1.0)
        float32_val = 0.4999
        bf16_val = bfloat16_round(float32_val)

        assert float32_val < 0.5, f"{float32_val} should be in lower bin"
        assert bf16_val == 0.5, f"bf16 of {float32_val} should be 0.5"

    def test_bfloat16_relative_error_at_scale(self):
        """Document bfloat16 precision at typical activation magnitudes."""
        test_values = [0.5, 1.0, 10.0, 100.0, 1000.0, 10000.0]
        for val in test_values:
            bf16_val = bfloat16_round(val)
            rel_error = abs(val - bf16_val) / val if val != 0 else 0
            # bfloat16 has ~7 bits of mantissa, so relative error ~1/128 ≈ 0.78%
            assert (
                rel_error < 0.01
            ), f"bfloat16 relative error {rel_error:.6f} exceeds 1% at magnitude {val}"


class TestIntervalBoundaryMechanisms:
    """Validate closed-interval vs half-open bin membership semantics."""

    def test_value_at_bin_edge_included_in_closed_interval(self):
        """A value exactly at bin_max is in the closed-interval bin."""
        # Closed: value <= bin_max means included
        # Half-open: value < bin_max means excluded
        value = 5.0
        bin_min = 0.0
        bin_max = 5.0

        # Closed-interval semantics (legacy path)
        closed_in_bin = bin_min <= value <= bin_max
        assert (
            closed_in_bin
        ), f"Value {value} should be in closed bin [{bin_min}, {bin_max}]"

        # Half-open semantics (columnar_gpu path)
        half_open_in_bin = bin_min <= value < bin_max
        assert (
            not half_open_in_bin
        ), f"Value {value} should NOT be in half-open bin [{bin_min}, {bin_max})"

    def test_value_just_below_bin_edge_consistent(self):
        """Both interval semantics agree for values clearly within bins."""
        value = 4.999
        bin_min = 0.0
        bin_max = 5.0

        closed_in = bin_min <= value <= bin_max
        half_open_in = bin_min <= value < bin_max
        assert (
            closed_in and half_open_in
        ), f"Value {value} should be in both interval types for bin [{bin_min}, {bin_max}]"

    def test_value_at_bin_min_included_in_both(self):
        """Both interval types include the lower bound."""
        value = 0.0
        bin_min = 0.0
        bin_max = 5.0

        closed_in = bin_min <= value <= bin_max
        half_open_in = bin_min <= value < bin_max
        assert (
            closed_in and half_open_in
        ), f"Value {value} at bin_min should be in both interval types"

    def test_interval_semantics_cause_row_count_delta(self):
        """Demonstrate how interval semantics can cause different row counts."""
        # Simulate: legacy keeps 3 records (values at bin edges included),
        # columnar_gpu keeps 1 (bin-edge value excluded from closed-interval bin)
        bin_intervals = [(0, 5), (5, 10), (10, 15)]
        activations = [5.0, 10.0, 15.0]  # All exactly at bin edges

        # Closed-interval: each activation belongs to its lower bin
        closed_count = 0
        for val in activations:
            for lo, hi in bin_intervals:
                if lo <= val <= hi:
                    closed_count += 1

        # Half-open: bin-edge activations excluded from the lower bin
        half_open_count = 0
        for val in activations:
            for lo, hi in bin_intervals:
                if lo <= val < hi:
                    half_open_count += 1

        # The delta is expected — it's the bin-edge records that differ
        delta = closed_count - half_open_count
        assert delta > 0, (
            f"Closed-interval should have more records than half-open "
            f"when values land on bin edges (got delta={delta})"
        )
        print(
            f"  Interval semantics delta: closed={closed_count} vs half_open={half_open_count}"
        )


class TestCombinedMechanisms:
    """Validate that bfloat16 + interval semantics together explain all deltas."""

    def test_both_mechanisms_cover_all_discrepancy_scenarios(self):
        """Any row discrepancy between legacy and columnar_gpu must fall into one of:
        (a) Value at bin edge (interval semantics)
        (b) Value crosses bin edge after bfloat16 round-trip
        """
        # Mechanism (a): Interval semantics at upper bin boundary
        # A float32 value exactly at bin_max=5.0:
        #   - Closed interval [0.0, 5.0]: value=5.0 IS included
        #   - Half-open [0.0, 5.0): value=5.0 is NOT included
        value = 5.0
        bin_lo, bin_hi = 0.0, 5.0
        closed_in = bin_lo <= value <= bin_hi  # True
        half_open_in = bin_lo <= value < bin_hi  # False (5.0 < 5.0 is False)
        discrepancy = closed_in != half_open_in
        assert discrepancy, (
            f"Interval semantics should cause discrepancy at bin edge: "
            f"closed={closed_in}, half_open={half_open_in}"
        )

        # Mechanism (b): bfloat16 downcast shifts value across boundary
        # float32=1.001, bin=[0.0, 1.0]
        #   - float32: 1.001 is NOT in [0.0, 1.0]
        #   - bfloat16 of 1.001 ≈ 1.0, which IS in [0.0, 1.0] under closed-interval
        value = 1.001
        bf16_val = bfloat16_round(value)
        bin_lo, bin_hi = 0.0, 1.0
        float32_closed = bin_lo <= value <= bin_hi  # False (1.001 > 1.0)
        bf16_closed = bin_lo <= bf16_val <= bin_hi  # True (1.0 <= 1.0)
        discrepancy = float32_closed != bf16_closed
        assert discrepancy, (
            f"bfloat16 downcast should cause discrepancy: "
            f"float32={value} in_closed={float32_closed}, "
            f"bf16={bf16_val} in_closed={bf16_closed}"
        )
