import numpy as np
import pytest
import torch

from sae_dashboard.utils_fns import (
    FeatureStatistics,
    RollingCorrCoef,
    TopK,
    sample_unique_indices,
)

# from torch.profiler import ProfilerActivity, profile, record_function


SYMMETRIC_RANGES_AND_PRECISIONS: list[tuple[list[float], int]] = [
    ([0.0, 0.01], 5),
    ([0.01, 0.05], 4),
    ([0.05, 0.95], 3),
    ([0.95, 0.99], 4),
    ([0.99, 1.0], 5),
]
ASYMMETRIC_RANGES_AND_PRECISIONS: list[tuple[list[float], int]] = [
    ([0.0, 0.95], 3),
    ([0.95, 0.99], 4),
    ([0.99, 1.0], 5),
]


@pytest.fixture(params=[torch.float16, torch.float32, torch.float64])
def precision_data(request):  # type:ignore
    # Create some sample data
    data = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [2.0, 4.0, 6.0, 8.0, 10.0],
            [0.0, 0.1, 0.2, 0.3, 0.4],
        ],
        dtype=request.param,
    )
    return data, request.param


@pytest.fixture(params=[torch.float16, torch.float32, torch.float64])
def large_precision_data(request):  # type:ignore
    # check if cuda is available, and if so, set device to this
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    # Create some sample data
    data = torch.randn(100, 1000, device=device, dtype=request.param)
    return data, request.param


def test_sample_unique_indices():
    large_number = 1000
    small_number = 10

    sampled_indices = sample_unique_indices(large_number, small_number)

    # Test that the function returns a list of the correct length
    assert len(sampled_indices) == small_number

    # assert type is Tensor of Ints
    assert isinstance(sampled_indices, torch.Tensor)
    assert sampled_indices.dtype == torch.int64

    large_number = 1000
    small_number = 990

    sampled_indices = sample_unique_indices(large_number, small_number)

    # test that no indices are repeated
    assert len(sampled_indices) == len(set(sampled_indices.tolist()))


def test_RollingCorrCoef_corrcoef():
    xs = torch.randn(10, 100)
    ys = torch.randn(10, 100)

    rcc = RollingCorrCoef()
    rcc.update(xs[:, :30], ys[:, :30])
    rcc.update(xs[:, 30:70], ys[:, 30:70])
    rcc.update(xs[:, 70:], ys[:, 70:])

    pearson, cossim = rcc.corrcoef()

    norm_xs = xs / xs.norm(dim=1, keepdim=True)
    norm_ys = ys / ys.norm(dim=1, keepdim=True)
    assert torch.allclose(cossim, norm_xs @ norm_ys.T, atol=1e-5)

    full_corrcoef = torch.corrcoef(torch.cat([xs, ys]))
    assert torch.allclose(pearson, full_corrcoef[:10, 10:], atol=1e-5)


def test_TopK_without_mask():
    topk = TopK(
        tensor=torch.arange(10) + 1,
        k=3,
        tensor_mask=None,
    )

    assert topk.values.tolist() == [10, 9, 8]
    assert topk.indices.tolist() == [9, 8, 7]


def test_TopK_without_mask_smallest():
    topk = TopK(tensor=torch.arange(10) + 1, k=3, tensor_mask=None, largest=False)

    assert topk.values.tolist() == [1, 2, 3]
    assert topk.indices.tolist() == [0, 1, 2]


def test_feature_statistics_batched_vs_unbatched(
    large_precision_data: tuple[torch.Tensor, torch.dtype],
):
    data, dtype = large_precision_data

    # Create unbatched FeatureStatistics object
    unbatched_stats = FeatureStatistics.create(data)

    # Create batched FeatureStatistics object
    batched_stats = FeatureStatistics.create(
        data, batch_size=10
    )  # Process 10 features at a time

    # Compare max values
    assert np.allclose(
        unbatched_stats.max, batched_stats.max, atol=1e-3
    ), "Max values do not match"

    # Compare fraction of non-zero values
    assert np.allclose(
        unbatched_stats.frac_nonzero, batched_stats.frac_nonzero, atol=1e-3
    ), "Fraction of non-zero values do not match"

    # Compare quantiles
    assert (
        unbatched_stats.quantiles == batched_stats.quantiles
    ), "Quantiles do not match"

    # Compare quantile data
    assert len(unbatched_stats.quantile_data) == len(
        batched_stats.quantile_data
    ), "Quantile data lengths do not match"
    for unbatched_qd, batched_qd in zip(
        unbatched_stats.quantile_data, batched_stats.quantile_data
    ):
        assert len(unbatched_qd) == len(
            batched_qd
        ), "Quantile data sub-lengths do not match"
        assert np.allclose(
            unbatched_qd, batched_qd, atol=1e-3
        ), "Quantile data values do not match"

    # Compare ranges_and_precisions
    assert (
        unbatched_stats.ranges_and_precisions == batched_stats.ranges_and_precisions
    ), "Ranges and precisions do not match"

    print(f"Test completed for dtype: {dtype}")
    print("Batched and unbatched results match within tolerance.")


@pytest.mark.parametrize("n_features,batch_size", [(100, 10), (100, 30), (100, 7)])
def test_feature_statistics_batched_vs_unbatched_uneven_sizes(
    large_precision_data: tuple[torch.Tensor, torch.dtype],
    n_features: int,
    batch_size: int,
):
    data, dtype = large_precision_data

    # Slice the data to the specified number of features
    data = data[:n_features]

    # Create unbatched FeatureStatistics object
    unbatched_stats = FeatureStatistics.create(data)

    # Create batched FeatureStatistics object
    batched_stats = FeatureStatistics.create(data, batch_size=batch_size)

    # Compare max values
    assert np.allclose(
        unbatched_stats.max, batched_stats.max, atol=1e-3
    ), f"Max values do not match for n_features={n_features}, batch_size={batch_size}"

    # Compare fraction of non-zero values
    assert np.allclose(
        unbatched_stats.frac_nonzero, batched_stats.frac_nonzero, atol=1e-3
    ), f"Fraction of non-zero values do not match for n_features={n_features}, batch_size={batch_size}"

    # Compare quantiles
    assert (
        unbatched_stats.quantiles == batched_stats.quantiles
    ), f"Quantiles do not match for n_features={n_features}, batch_size={batch_size}"

    # Compare quantile data
    assert len(unbatched_stats.quantile_data) == len(
        batched_stats.quantile_data
    ), f"Quantile data lengths do not match for n_features={n_features}, batch_size={batch_size}"
    for unbatched_qd, batched_qd in zip(
        unbatched_stats.quantile_data, batched_stats.quantile_data
    ):
        assert len(unbatched_qd) == len(
            batched_qd
        ), f"Quantile data sub-lengths do not match for n_features={n_features}, batch_size={batch_size}"
        assert np.allclose(
            unbatched_qd, batched_qd, atol=1e-3
        ), f"Quantile data values do not match for n_features={n_features}, batch_size={batch_size}"

    # Compare ranges_and_precisions
    assert (
        unbatched_stats.ranges_and_precisions == batched_stats.ranges_and_precisions
    ), f"Ranges and precisions do not match for n_features={n_features}, batch_size={batch_size}"

    print(
        f"Test completed for dtype: {dtype}, n_features: {n_features}, batch_size: {batch_size}"
    )
    print("Batched and unbatched results match within tolerance.")


def test_feature_statistics_create(precision_data: tuple[torch.Tensor, torch.dtype]):
    data, dtype = precision_data

    # Create FeatureStatistics object
    feature_stats = FeatureStatistics.create(data)

    # Test max values
    assert np.allclose(feature_stats.max, [5.0, 10.0, 0.4], atol=1e-3)

    # Test fraction of non-zero values
    assert np.allclose(feature_stats.frac_nonzero, [1.0, 1.0, 0.8], atol=1e-3)

    # Test quantiles
    assert len(feature_stats.quantiles) > 0
    assert feature_stats.quantiles[0] == 0.0
    assert feature_stats.quantiles[-1] == 1.0

    # Test quantile data
    assert len(feature_stats.quantile_data) == 3  # One for each row in the input data
    assert all(len(qd) > 0 for qd in feature_stats.quantile_data)

    print(f"Test completed for dtype: {dtype}")


def test_feature_statistics_update():
    data1 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    data2 = torch.tensor([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])

    stats1 = FeatureStatistics.create(data1)
    stats2 = FeatureStatistics.create(data2)

    stats1.update(stats2)

    assert stats1.max == [3.0, 6.0, 9.0, 12.0]
    assert len(stats1.quantile_data) == 4


# def test_feature_statistics_benchmark(large_precision_data):
#     # Check if CUDA is available
#     if not torch.cuda.is_available():
#         pytest.skip("CUDA not available, skipping benchmark test")

#     # Create a large dataset
#     data, _ = large_precision_data

#     # Run the benchmark
#     with profile(
#         activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
#         record_shapes=True,
#         profile_memory=True,
#         with_stack=True
#     ) as prof:
#         with record_function("FeatureStatistics.create"):
#             feature_stats = FeatureStatistics.create(data)

#     # Print the profiler results
#     print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

#     # You can add more specific assertions here if needed
#     assert feature_stats is not None
#     assert len(feature_stats.max) == data.shape[0]

#     # Optionally, you can save the profiler results to a file
#     prof.export_chrome_trace("feature_statistics_benchmark_trace.json")
