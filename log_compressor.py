"""Rule-based compression for noisy build/runtime logs."""

from __future__ import annotations

import re
from pathlib import Path

from . import config
from .models import LogSummary
from .utils import ensure_cache_dir, write_json

FILE_RE = re.compile(r"(?:(?:src|app|backend|frontend)[/\\][\w./\\-]+|[\w./\\-]+\.(?:cs|ts|tsx|js|jsx|py|java|go|rs))")
CODE_RE = re.compile(r"\b(?:TS|CS|NG)\d{3,5}\b|\b(?:400|401|403|404|500)\b")
SYMBOL_RE = re.compile(r"\b[A-Z][A-Za-z0-9_]*(?:Exception|Error|Controller|Service|Dto|DTO|Component)?\b")
TIMESTAMP_ONLY_RE = re.compile(r"^\s*(?:\d{4}-\d{2}-\d{2}|\d{2}:\d{2}:\d{2})[\d:.\-TZ\s]*$")


def _is_noise(line: str) -> bool:
    stripped = line.strip()
    lower = stripped.lower()
    if not stripped or TIMESTAMP_ONLY_RE.match(stripped):
        return True
    if lower.startswith(("added ", "audited ", "found 0 vulnerabilities", "up to date")):
        return True
    if re.match(r"^\s+at\s+[\w.$<>]+\s*\(", stripped) and "error" not in lower:
        return True
    return False


def _is_important(line: str) -> bool:
    lower = line.lower()
    if any(pattern in lower for pattern in config.LOG_IMPORTANT_PATTERNS):
        return True
    if CODE_RE.search(line):
        return True
    return bool(FILE_RE.search(line))


def compress_log(log_path: Path) -> LogSummary:
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    seen: set[str] = set()
    important: list[str] = []
    warnings: list[str] = []
    noise_notes: set[str] = set()

    for line in lines:
        normalized = re.sub(r"\s+", " ", line.strip())
        if _is_noise(normalized):
            noise_notes.add("timestamps/progress/repeated stack frames")
            continue
        if normalized in seen:
            noise_notes.add("duplicate lines")
            continue
        seen.add(normalized)
        if _is_important(normalized):
            important.append(normalized)
        elif "warning" in normalized.lower():
            warnings.append(normalized)
            noise_notes.add("warnings omitted while errors were present")

    if not important and warnings:
        important = warnings[:80]
        noise_notes.discard("warnings omitted while errors were present")

    truncated = len(important) > 80
    if truncated:
        important = important[:40] + important[-40:]
        noise_notes.add("log truncated to first 40 and last 40 important lines")

    mentioned_files = sorted({Path(match.replace("\\", "/")).name for line in important for match in FILE_RE.findall(line)})
    mentioned_symbols = sorted({match for line in important for match in SYMBOL_RE.findall(line) if len(match) > 2})[:50]
    error_codes = sorted({match for line in important for match in CODE_RE.findall(line)})

    summary = LogSummary(
        original_line_count=len(lines),
        kept_line_count=len(important),
        important_lines=important,
        mentioned_files=mentioned_files,
        mentioned_symbols=mentioned_symbols,
        error_codes=error_codes,
        noise_removed_notes=sorted(noise_notes) or ["blank and non-error lines"],
        truncated=truncated,
    )
    write_json(ensure_cache_dir() / "log_summary.json", summary.to_dict())
    return summary
