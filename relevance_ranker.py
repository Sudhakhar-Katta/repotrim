"""Task-aware ranking over files produced by the shared scanner."""

from __future__ import annotations

from pathlib import Path

from .ignore_rules import has_ignored_path_part
from .models import LogSummary, RankedFile, ScanResult, TaskResult
from .scanner import read_text_safely
from .task_analyzer import TaskAnalysis, analyze_task
from .utils import ensure_cache_dir, write_json


ML_ROLE_TERMS = {"model", "detector", "inference", "predict", "train", "dataset", "preprocess", "ensemble", "config", "requirements"}
SUPPORT_TERMS = {"config", "requirements", "pyproject", "setup", "ensemble", "test", "spec", "readme", "docs"}


def _score_file(file, analysis: TaskAnalysis, log_summary: LogSummary | None) -> RankedFile | None:
    relative = file.relative_path.lower()
    filename = Path(relative).name
    first_lines = "\n".join(file.first_lines).lower()
    symbols = " ".join(file.symbols).lower()
    text = read_text_safely(Path(file.path))
    content = text.lower() if text else first_lines
    score = 0
    reasons: list[str] = []

    for keyword in analysis.keywords:
        if keyword in filename:
            score += 30
            reasons.append(f'filename matched "{keyword}"')
        if keyword in relative and keyword not in filename:
            score += 14
            reasons.append(f'relative path matched "{keyword}"')
        count = content.count(keyword)
        if count:
            score += min(24, count * 3)
            reasons.append(f'content contains "{keyword}" related symbols')
        if keyword in first_lines:
            score += 8
            reasons.append(f'first lines matched "{keyword}"')
        if keyword in symbols:
            score += 10
            reasons.append(f'symbol matched "{keyword}"')

    if file.category == "source":
        score += 5
    elif file.category in {"config", "test"}:
        score += 3

    if analysis.intent == "ml_model_debug":
        matched_roles = sorted(term for term in ML_ROLE_TERMS if term in relative)
        if matched_roles:
            score += 16 + min(12, len(matched_roles) * 3)
            reasons.append(f'likely ML {matched_roles[0]} file')
        ml_symbols = [term for term in ("torch", "tensorflow", "model.eval", "cuda", "predict", "forward") if term in content]
        if ml_symbols:
            score += 12
            reasons.append("content contains CNN/model inference symbols")

    if analysis.intent == "frontend_feature" and (file.extension in {".ts", ".tsx", ".js", ".jsx", ".css", ".html"} or "component" in relative):
        score += 15
        reasons.append("likely frontend component or style file")
    if analysis.intent == "backend_feature" and any(term in relative for term in ("api", "route", "server", "controller", "service", "database")):
        score += 15
        reasons.append("likely backend route or service file")

    if log_summary:
        mentions = {Path(item).name.lower() for item in log_summary.mentioned_files}
        if filename in mentions:
            score += 40
            reasons.append("error log mentioned this file")
    if filename.startswith("readme") and analysis.intent != "general_code_task":
        score -= 10
    if file.estimated_tokens > 20000:
        score -= 20
    if score <= 0:
        return None
    return RankedFile(file=file, score=score, reasons=list(dict.fromkeys(reasons)))


def _deduplicate(ranked: list[RankedFile]) -> list[RankedFile]:
    selected: list[RankedFile] = []
    seen_names: set[str] = set()
    for item in sorted(ranked, key=lambda candidate: (-candidate.score, len(Path(candidate.file.relative_path).parts), candidate.file.relative_path)):
        name = Path(item.file.relative_path).name.lower()
        if name in seen_names:
            continue
        seen_names.add(name)
        selected.append(item)
    return selected


def rank_scanned_files(scan: ScanResult, task_description: str, log_summary: LogSummary | None = None) -> TaskResult:
    analysis = analyze_task(task_description)
    candidates = (
        file for file in scan.scanned_files
        if not has_ignored_path_part(file.relative_path)
    )
    ranked = _deduplicate([item for file in candidates if (item := _score_file(file, analysis, log_summary))])
    primary = [item for item in ranked if item.file.category == "source" and not any(term in item.file.relative_path.lower() for term in SUPPORT_TERMS)][:5]
    primary_paths = {item.file.relative_path for item in primary}
    supporting = [item for item in ranked if item.file.relative_path not in primary_paths][:5]
    result = TaskResult(
        task_description=task_description,
        keywords=analysis.keywords,
        ranked_files=(primary + supporting)[:8],
        intent=analysis.intent,
        primary_files=primary,
        supporting_files=supporting,
    )
    write_json(ensure_cache_dir(Path(scan.repo_path)) / "task.json", result.to_dict())
    return result
