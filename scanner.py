"""Repository scanning and lightweight symbol extraction."""

from __future__ import annotations

import os
import re
import subprocess
from collections import Counter
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
    if name in {"requirements.txt", "pyproject.toml", "setup.py", "setup.cfg", "tox.ini", ".env.example", "dockerfile", "makefile"} or extension in {".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".csproj"}:
        return "config"
    if extension == ".md":
        return "docs"
    return "source"


def find_repo_root(start_path: Path) -> Path:
    """Find the containing Git root, falling back to the requested directory."""
    start_path = start_path.resolve()
    if start_path.is_file():
        start_path = start_path.parent
    try:
        completed = subprocess.run(
            ["git", "-C", str(start_path), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return start_path
    root = Path(completed.stdout.strip())
    return root.resolve() if root.is_dir() else start_path


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


def extract_repo_hints(text: str) -> tuple[list[str], list[str]]:
    imports: list[str] = []
    comments: list[str] = []
    for line in text.splitlines()[:120]:
        stripped = line.strip()
        if re.match(r"^(from\s+\S+\s+import|import\s+|using\s+|require\(|const\s+.+?=\s*require\(|import\s+.+?from\s+)", stripped):
            imports.append(stripped[:240])
        if stripped.startswith(("#", "//", "/*", "*", "<!--")):
            comments.append(stripped[:240])
    return imports[:30], comments[:20]


def make_file_info(path: Path, repo_path: Path, text: str) -> FileInfo:
    extension = path.suffix.lower()
    language = config.LANGUAGE_BY_EXTENSION.get(extension, "text")
    lines = text.splitlines()
    imports, comments = extract_repo_hints(text)
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
        imports=imports,
        comments=comments,
    )


def scan_repo(repo_path: Path) -> ScanResult:
    repo_path = repo_path.resolve()
    if not repo_path.exists():
        raise ValueError(f"Repository path does not exist: {repo_path}")
    if not repo_path.is_dir():
        raise ValueError(f"Repository path is not a directory: {repo_path}")
    files: list[FileInfo] = []
    ignored: list[IgnoredPath] = []
    files_discovered = 0
    files_unreadable = 0
    extensions: Counter[str] = Counter()

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
            files_discovered += 1
            path = root_path / filename
            extension = path.suffix.lower() or "(none)"
            extensions[extension] += 1
            reason = should_ignore_path(path, repo_path)
            if reason:
                relative = path.relative_to(repo_path).as_posix()
                ignored.append(IgnoredPath(path=str(path.resolve()), relative_path=relative, reason=reason))
                continue
            text = read_text_safely(path)
            if text is None:
                files_unreadable += 1
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
        current_working_directory=str(Path.cwd().resolve()),
        files_discovered=files_discovered,
        extensions_found=dict(sorted(extensions.items())),
        files_unreadable=files_unreadable,
    )
    write_json(ensure_cache_dir(repo_path) / "scan.json", result.to_dict())
    return result
