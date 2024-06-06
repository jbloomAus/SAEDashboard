import torch
from sae_dashboard.utils_fns import RollingCorrCoef, TopK, sample_unique_indices


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
