"""TransformerLens and SAELens names that moved or were removed across versions.

TransformerLens 4.0 removed ``HookedTransformer`` (and with it SAELens' ``HookedSAETransformer``) and the
``transformer_lens.utils`` module. Importing any of them at module level made ``import sae_dashboard.<module>``
fail on 4.0 even for callers that only use the bridge wrapper. Only the hooked model wrapper needs the removed
classes, so it asks for them here at the point of use, and a missing class is refused by name with the fix.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

try:  # TransformerLens >= 3 keeps the helper in `utilities`; 4.0 removed the old `utils` module
    from transformer_lens.utilities.tensors import to_numpy
except ImportError:  # TransformerLens 2.x
    from transformer_lens.utils import to_numpy  # type: ignore[no-redef]

if TYPE_CHECKING:
    from sae_lens import HookedSAETransformer
    from transformer_lens import HookedTransformer

__all__ = ["hooked_sae_transformer_cls", "hooked_transformer_cls", "to_numpy"]

_HOOKED_UNAVAILABLE = (
    "{name} is unavailable with the installed transformer-lens {tl_version}: TransformerLens 4.0 removed "
    "HookedTransformer, so the hooked model wrapper cannot run. Use the bridge wrapper "
    "(--model-wrapper bridge / model_wrapper='bridge'), or install transformer-lens<4 for the hooked wrapper."
)


def _tl_version() -> str:
    try:
        return version("transformer-lens")
    except PackageNotFoundError:
        return "(not installed)"


def hooked_transformer_cls() -> type[HookedTransformer]:
    """``transformer_lens.HookedTransformer``, or an ImportError naming the bridge wrapper as the alternative."""
    try:
        from transformer_lens import HookedTransformer
    except ImportError as err:
        raise ImportError(
            _HOOKED_UNAVAILABLE.format(
                name="HookedTransformer", tl_version=_tl_version()
            )
        ) from err
    return HookedTransformer


def hooked_sae_transformer_cls() -> type[HookedSAETransformer]:
    """``sae_lens.HookedSAETransformer``, or an ImportError naming the bridge wrapper as the alternative."""
    try:
        from sae_lens import HookedSAETransformer
    except ImportError as err:
        raise ImportError(
            _HOOKED_UNAVAILABLE.format(
                name="sae_lens.HookedSAETransformer", tl_version=_tl_version()
            )
        ) from err
    return HookedSAETransformer
