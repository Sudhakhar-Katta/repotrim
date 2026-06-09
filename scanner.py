"""Repository scanning and lightweight symbol extraction."""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path

from . import config
from .ignore_rules import ignore_reason
from .models import FileInfo, IgnoredPath, ScanResult
from .tokenizer import estimate_tokens
from .utils import ensure_cache_dir, write_json


SYMBOL_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "python": [re.compile(r"^\s*(def|class)\s+([A-Za-z_][\w]*)")],
    "typescript": [
        re.compile(r"^\s*(export\s+)?(class|function)\s+([A-Za-z_][\w]*)"),
        re.compile(r"^\s*const\s+([A-Za-z_][\w]*)\s*="),
    ],
    "javascript": [
        re.compile(r"^\s*(export\s+)?(class|function)\s+([A-Za-z_][\w]*)"),
        re.compile(r"^\s*const\s+([A-Za-z_][\w]*)\s*="),
    ],
    "csharp": [
        re.compile(r"\b(class|interface)\s+([A-Za-z_][\w]*)"),
        re.compile(r"\b(public|private|protected|internal)\s+[\w<>,\[\]\s]+\s+([A-Za-z_][\w]*)\s*\("),
    ],
    "java": [
        re.compile(r"\b(class|interface)\s+([A-Za-z_][\w]*)"),
        re.compile(r"\b(public|private|protected)\s+[\w<>,\[\]\s]+\s+([A-Za-z_][\w]*)\s*\("),
    ],
}


def should_ignore_path(path: Path, repo_path: Path) -> str | None:
    return ignore_reason(path, repo_path)


def categorize_file(relative_path: str, extension: str) -> str:
    parts = {part.lower() for part in Path(relative_path).parts}
    name = Path(relative_path).name.lower()
    if "test" in name or "tests" in parts or "test" in parts or name.startswith("spec"):
        return "test"
    if name in {"requirements.txt", "pyproject.toml", "setup.py", "setup.cfg", "tox.ini"} or extension in {".json", ".yaml", ".yml", ".toml", ".ini", ".cfg"}:
        return "config"
    if extension == ".md":
        return "docs"
    return "source"


def read_text_safely(path: Path) -> str | None:
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if b"\x00" in data[:4096]:
        return None
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return None


def extract_symbols(text: str, language: str) -> list[str]:
    symbols: list[str] = []
    patterns = SYMBOL_PATTERNS.get(language, [])
    for line in text.splitlines():
        for pattern in patterns:
            match = pattern.search(line)
            if not match:
                continue
            symbol = match.group(match.lastindex or 1)
            if symbol in {"class", "interface", "function", "public", "private", "protected", "internal", "export"}:
                symbol = match.group((match.lastindex or 1) + 1)
            symbols.append(symbol)
            break
    return list(dict.fromkeys(symbols))[:80]


def make_file_info(path: Path, repo_path: Path, text: str) -> FileInfo:
    extension = path.suffix.lower()
    language = config.LANGUAGE_BY_EXTENSION.get(extension, "text")
    lines = text.splitlines()
    relative_path = path.relative_to(repo_path).as_posix()
    return FileInfo(
        path=str(path.resolve()),
        relative_path=relative_path,
        extension=extension,
        language=language,
        size_bytes=path.stat().st_size,
        char_count=len(text),
        estimated_tokens=estimate_tokens(text),
        first_lines=lines[:30],
        symbols=extract_symbols(text, language),
        modified_time=path.stat().st_mtime,
        category=categorize_file(relative_path, extension),
        reason="supported readable text file",
    )


def scan_repo(repo_path: Path) -> ScanResult:
    repo_path = repo_path.resolve()
    files: list[FileInfo] = []
    ignored: list[IgnoredPath] = []

    for root, dirnames, filenames in os.walk(repo_path):
        root_path = Path(root)
        kept_dirs: list[str] = []
        for dirname in dirnames:
            dir_path = root_path / dirname
            reason = should_ignore_path(dir_path, repo_path)
            if reason:
                relative = dir_path.relative_to(repo_path).as_posix()
                ignored.append(IgnoredPath(path=str(dir_path.resolve()), relative_path=relative, reason=reason))
            else:
                kept_dirs.append(dirname)
        dirnames[:] = kept_dirs

        for filename in filenames:
            path = root_path / filename
            reason = should_ignore_path(path, repo_path)
            if reason:
                relative = path.relative_to(repo_path).as_posix()
                ignored.append(IgnoredPath(path=str(path.resolve()), relative_path=relative, reason=reason))
                continue
            text = read_text_safely(path)
            if text is None:
                relative = path.relative_to(repo_path).as_posix()
                ignored.append(IgnoredPath(path=str(path.resolve()), relative_path=relative, reason="binary or unreadable"))
                continue
            files.append(make_file_info(path, repo_path, text))

    result = ScanResult(
        repo_path=str(repo_path),
        created_at=datetime.now(timezone.utc).isoformat(),
        total_files_scanned=len(files),
        total_files_ignored=len(ignored),
        total_estimated_tokens=sum(file.estimated_tokens for file in files),
        files=files,
        ignored=ignored,
    )
    write_json(ensure_cache_dir(repo_path) / "scan.json", result.to_dict())
    return result
