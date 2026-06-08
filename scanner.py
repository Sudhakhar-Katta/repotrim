"""Repository scanning and lightweight symbol extraction."""

from __future__ import annotations

import fnmatch
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from . import config
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
    name = path.name
    lower_name = name.lower()
    parts = {part.lower() for part in path.relative_to(repo_path).parts}

    if path.is_dir() and lower_name in config.IGNORED_DIRECTORIES:
        return f"ignored directory: {name}"
    if any(marker in lower_name for marker in config.SECRET_NAME_MARKERS):
        return "secret-looking path"
    if path.is_file():
        if any(part in config.IGNORED_DIRECTORIES for part in parts):
            return "inside ignored directory"
        if any(fnmatch.fnmatch(name, pattern) for pattern in config.IGNORED_FILE_PATTERNS):
            return "ignored file pattern"
        if path.suffix.lower() not in config.SUPPORTED_EXTENSIONS:
            return "unsupported extension"
        try:
            size = path.stat().st_size
        except OSError:
            return "could not stat file"
        if size > config.MAX_FILE_SIZE_BYTES:
            return "file larger than 500 KB"
    return None


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
    return FileInfo(
        path=str(path.resolve()),
        relative_path=path.relative_to(repo_path).as_posix(),
        extension=extension,
        language=language,
        size_bytes=path.stat().st_size,
        char_count=len(text),
        estimated_tokens=estimate_tokens(text),
        first_lines=lines[:30],
        symbols=extract_symbols(text, language),
        modified_time=path.stat().st_mtime,
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
                ignored.append(IgnoredPath(path=dir_path.relative_to(repo_path).as_posix(), reason=reason))
            else:
                kept_dirs.append(dirname)
        dirnames[:] = kept_dirs

        for filename in filenames:
            path = root_path / filename
            reason = should_ignore_path(path, repo_path)
            if reason:
                ignored.append(IgnoredPath(path=path.relative_to(repo_path).as_posix(), reason=reason))
                continue
            text = read_text_safely(path)
            if text is None:
                ignored.append(IgnoredPath(path=path.relative_to(repo_path).as_posix(), reason="binary or unreadable"))
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
    write_json(ensure_cache_dir() / "scan.json", result.to_dict())
    return result
