import torch
from sae_lens import SAE

from sae_dashboard.feature_data_generator import FeatureMaskingContext


@torch.no_grad()
def test_feature_mask_context(autoencoder: SAE):  # type: ignore
    feature_indices = list(range(10))

    sae_in_mock = torch.randn(10, 64, autoencoder.cfg.d_in)
    features = autoencoder.encode(sae_in_mock)
    original_feature_acts = features[:, :, feature_indices]

    with FeatureMaskingContext(autoencoder, feature_indices):
        new_feature_acts = autoencoder.encode(sae_in_mock)

    assert (original_feature_acts - new_feature_acts).max() <= 1e-5
