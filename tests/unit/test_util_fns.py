import torch

from sae_vis.utils_fns import sample_unique_indices


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
