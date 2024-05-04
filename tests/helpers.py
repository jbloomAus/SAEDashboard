from typing import Any


def round_floats_deep(obj: Any, ndigits: int = 3) -> Any:
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {k: round_floats_deep(v, ndigits=ndigits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_floats_deep(v, ndigits=ndigits) for v in obj]
    return obj
