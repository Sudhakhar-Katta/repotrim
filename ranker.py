"""Rule-based file ranking for a coding task."""

from __future__ import annotations

import re
from pathlib import Path

from . import config
from .models import FileInfo, LogSummary, RankedFile, ScanResult, TaskResult
from .scanner import read_text_safely
from .utils import ensure_cache_dir, write_json


def normalize_task(task_description: str) -> list[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9_]+", task_description.lower())
    keywords: set[str] = set()
    for word in words:
        if word in config.STOP_WORDS or len(word) < 2:
            continue
        variants = config.KEYWORD_VARIANTS.get(word, {word})
        keywords.update(variants)
    return sorted(keywords)


def _content_for_file(file: FileInfo) -> str:
    text = read_text_safely(Path(file.path))
    return text.lower() if text else "\n".join(file.first_lines).lower()


def _role_boosts(task_words: set[str], file: FileInfo, reasons: list[str]) -> int:
    rel = file.relative_path.lower()
    name = Path(file.relative_path).name.lower()
    score = 0
    if task_words & {"api", "endpoint", "route"} and any(term in rel for term in ("controller", "router", "route")):
        score += 10
        reasons.append("API task matched controller/router file")
    if task_words & {"database", "db", "migration", "schema"} and any(term in rel for term in ("model", "entity", "migration", "schema")):
        score += 10
        reasons.append("database task matched model/entity/migration file")
    if task_words & {"frontend", "button", "modal", "component", "ui"} and (file.extension in {".ts", ".tsx", ".jsx", ".html", ".css", ".scss"} or "component" in name):
        score += 10
        reasons.append("frontend/UI task matched component-style file")
    if "service" in task_words and "service" in rel:
        score += 10
        reasons.append("service task matched service file")
    if task_words & {"test", "failing", "unit"} and any(term in rel for term in ("test", "spec")):
        score += 10
        reasons.append("test task matched test/spec file")
    return score


def _extension_boosts(task_words: set[str], file: FileInfo, reasons: list[str]) -> int:
    frontend = {"frontend", "button", "modal", "component", "ui"}
    backend = {"api", "endpoint", "server", "backend", "controller", "service"}
    database = {"database", "db", "migration", "schema", "sql"}
    if task_words & frontend and file.extension in {".ts", ".tsx", ".jsx", ".html", ".css", ".scss"}:
        reasons.append("extension matched frontend task")
        return 5
    if task_words & backend and file.extension in {".cs", ".py", ".java", ".go", ".rs", ".js", ".ts"}:
        reasons.append("extension matched backend task")
        return 5
    if task_words & database and (file.extension == ".sql" or any(term in file.relative_path.lower() for term in ("migration", "model", "entity"))):
        reasons.append("extension/path matched database task")
        return 5
    return 0


def _penalties(task_words: set[str], file: FileInfo, reasons: list[str]) -> int:
    rel = file.relative_path.lower()
    penalty = 0
    if Path(rel).name.startswith("readme") and not (task_words & {"docs", "readme", "documentation"}):
        penalty -= 30
        reasons.append("README penalty")
    if file.extension in {".json", ".yaml", ".yml"} and not (task_words & {"config", "settings", "env"}):
        penalty -= 20
        reasons.append("config file penalty")
    if file.estimated_tokens > 20000:
        penalty -= 50
        reasons.append("very large file penalty")
    if any(term in rel for term in ("generated", ".min.", "designer", "dist/")):
        penalty -= 20
        reasons.append("generated-looking file penalty")
    return penalty


def rank_files(scan: ScanResult, task_description: str, log_summary: LogSummary | None = None) -> TaskResult:
    keywords = normalize_task(task_description)
    task_words = set(keywords)
    log_files = {Path(item).name.lower() for item in (log_summary.mentioned_files if log_summary else [])}
    log_symbols = {item.lower() for item in (log_summary.mentioned_symbols if log_summary else [])}

    ranked: list[RankedFile] = []
    for file in scan.files:
        rel = file.relative_path.lower()
        filename = Path(file.relative_path).name.lower()
        folder_path = str(Path(file.relative_path).parent).lower()
        first_lines = "\n".join(file.first_lines).lower()
        symbols = " ".join(file.symbols).lower()
        content = _content_for_file(file)
        score = 0
        reasons: list[str] = []

        for keyword in keywords:
            if keyword in filename:
                score += 20
                reasons.append(f'filename matched "{keyword}"')
            if keyword in folder_path:
                score += 12
                reasons.append(f'path matched "{keyword}"')
            elif keyword in rel:
                score += 8
                reasons.append(f'relative path matched "{keyword}"')
            count = content.count(keyword)
            if count:
                score += min(30, count * 3)
                reasons.append(f'content matched "{keyword}"')
            if keyword in symbols:
                score += 10
                reasons.append(f'symbol matched "{keyword}"')
            if keyword in first_lines:
                score += 5
                reasons.append(f'first lines matched "{keyword}"')

        if filename in log_files or any(symbol in symbols for symbol in log_symbols):
            score += 40
            reasons.append("error log mentioned this file or symbol")
        score += _role_boosts(task_words, file, reasons)
        score += _extension_boosts(task_words, file, reasons)
        score += _penalties(task_words, file, reasons)
        if score > 0:
            ranked.append(RankedFile(file=file, score=score, reasons=list(dict.fromkeys(reasons))))

    ranked.sort(key=lambda item: item.score, reverse=True)
    result = TaskResult(task_description=task_description, keywords=keywords, ranked_files=ranked[: config.MAX_RANKED_FILES])
    write_json(ensure_cache_dir() / "task.json", result.to_dict())
    return result
