"""Central ignore rules shared by all repository features."""

from __future__ import annotations

import fnmatch
from pathlib import Path

from . import config


IGNORED_DIRECTORIES = {
    ".git", ".claude", ".repotrim", ".venv", "venv", "env", "__pycache__",
    "node_modules", "dist", "build", ".next", ".nuxt", "coverage",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", ".idea", ".vscode",
    ".install-test", ".pytest-tmp",
    "bin", "obj", "target", "out", ".cache",
}


def has_ignored_path_part(relative_path: str | Path) -> bool:
    """Return whether any component belongs to an ignored directory."""
    parts = {part.lower() for part in Path(relative_path).parts}
    return bool(parts & IGNORED_DIRECTORIES)


def ignore_reason(path: Path, repo_path: Path, *, is_dir: bool | None = None) -> str | None:
    """Return why a path is ignored, checking every relative path component."""
    try:
        relative = path.relative_to(repo_path)
    except ValueError:
        return "outside repository"

    parts = [part.lower() for part in relative.parts]
    ignored_part = next((part for part in parts if part in IGNORED_DIRECTORIES), None)
    if ignored_part:
        return f'inside ignored directory "{ignored_part}"'
    generated_part = next((part for part in parts if part.endswith(".egg-info")), None)
    if generated_part:
        return f'inside generated metadata directory "{generated_part}"'

    name = path.name
    lower_name = name.lower()
    if lower_name.startswith(".env") or lower_name in {"credentials.json", "secret.json", "secrets.json"} or path.suffix.lower() in {".key", ".pem"}:
        return "secret-looking path"

    if is_dir is True or (is_dir is None and path.is_dir()):
        return None
    if any(fnmatch.fnmatch(name, pattern) for pattern in config.IGNORED_FILE_PATTERNS):
        return "ignored file pattern"
    if path.suffix.lower() not in config.SUPPORTED_EXTENSIONS:
        return "unsupported extension"
    try:
        if path.stat().st_size > config.MAX_FILE_SIZE_BYTES:
            return "file larger than 500 KB"
    except OSError:
        return "could not stat file"
    return None
