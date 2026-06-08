"""Small token estimation helpers."""

from __future__ import annotations

from math import ceil


def estimate_tokens(text: str) -> int:
    """Estimate tokens as ceil(characters / 4)."""
    if not text:
        return 0
    return ceil(len(text) / 4)
