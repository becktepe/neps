from __future__ import annotations

from typing import Any


def get_loss(result: str | dict | float) -> float | Any:
    if result == "error":
        return float("inf")
    elif isinstance(result, dict):
        return result["loss"]
    else:
        return result