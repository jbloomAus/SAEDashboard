from typing import Any

from sae_dashboard.sae_vis_data import SaeVisConfig


def round_floats_deep(obj: Any, ndigits: int = 3) -> Any:
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {k: round_floats_deep(v, ndigits=ndigits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_floats_deep(v, ndigits=ndigits) for v in obj]
    return obj


def build_sae_vis_cfg(**kwargs: Any) -> SaeVisConfig:
    """
    Helper to create a mock instance of SaeVisConfig for testing
    """
    mock_cfg = SaeVisConfig(
        hook_point="blocks.0.hook_resid_pre",
        features=list(range(32)),
    )

    for key, val in kwargs.items():
        setattr(mock_cfg, key, val)
    return mock_cfg
