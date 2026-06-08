"""Shared filesystem and JSON helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CACHE_DIR_NAME = ".repotrim"


def cache_dir(base_path: Path | None = None) -> Path:
    root = base_path or Path.cwd()
    return root / CACHE_DIR_NAME


def ensure_cache_dir(base_path: Path | None = None) -> Path:
    path = cache_dir(base_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def percent_savings(selected_tokens: int, full_tokens: int) -> float:
    if full_tokens <= 0:
        return 0.0
    return max(0.0, 100 * (1 - selected_tokens / full_tokens))
