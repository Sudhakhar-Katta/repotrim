"""Repository-specific vocabulary and ranking preferences."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_LOW_SIGNAL_FILES = {"notfound", "404", "error", "errorpage", "layout", "main", "index"}


@dataclass(frozen=True)
class RepoConfig:
    aliases: dict[str, list[str]] = field(default_factory=dict)
    low_signal_files: set[str] = field(default_factory=lambda: set(DEFAULT_LOW_SIGNAL_FILES))
    warning: str | None = None


def load_repo_config(repo_path: Path) -> RepoConfig:
    path = repo_path / ".repotrimrc.json"
    if not path.exists():
        return RepoConfig()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return RepoConfig(warning=f"Could not load .repotrimrc.json: {exc}")

    aliases: dict[str, list[str]] = {}
    raw_aliases = data.get("aliases", {})
    if isinstance(raw_aliases, dict):
        for key, values in raw_aliases.items():
            if isinstance(key, str) and isinstance(values, list):
                aliases[key.lower().strip()] = [str(value).lower().strip() for value in values if str(value).strip()]

    low_signal = set(DEFAULT_LOW_SIGNAL_FILES)
    values = data.get("lowSignalFiles", [])
    if isinstance(values, list):
        low_signal.update(str(value).lower().strip() for value in values if str(value).strip())
    return RepoConfig(aliases=aliases, low_signal_files=low_signal)
