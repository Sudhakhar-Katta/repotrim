"""Deterministic agentic context planning workflow."""

from __future__ import annotations

import re
from pathlib import Path

from ..models import FileInfo, RankedFile, ScanResult, TaskResult
from ..packet_generator import generate_packet
from ..relevance_ranker import rank_scanned_files
from ..scanner import scan_repo
from ..task_analyzer import analyze_task
from .report import write_agent_json, write_agent_report
from .reviewer import review_context, task_interpretation
from .state import AgentState


RELATED_NAME_TERMS = {
    "auth", "guard", "role", "roles", "permission", "permissions", "admin", "route", "routes",
    "router", "middleware", "service", "schema", "model", "types", "interface", "client",
    "api", "controller", "test", "spec", "config", "readme",
}


def _selected_ranked_files(task: TaskResult) -> list[RankedFile]:
    selected: list[RankedFile] = []
    seen: set[str] = set()
    for item in [*task.primary_files, *task.supporting_files, *task.ranked_files]:
        if item.file.relative_path in seen:
            continue
        seen.add(item.file.relative_path)
        selected.append(item)
    return selected


def _tokens(value: str) -> set[str]:
    expanded = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value)
    return {token.lower() for token in re.findall(r"[A-Za-z0-9]+", expanded)}


def _import_targets(file: FileInfo) -> set[str]:
    targets: set[str] = set()
    for line in file.imports:
        for match in re.findall(r"from\s+['\"]([^'\"]+)['\"]|import\s+['\"]([^'\"]+)['\"]|from\s+([A-Za-z0-9_\.]+)\s+import|import\s+([A-Za-z0-9_\.]+)", line):
            target = next((part for part in match if part), "")
            if not target:
                continue
            targets.update(_tokens(target.replace(".", "/")))
    return targets


def _related_score(candidate: FileInfo, selected: FileInfo, task_tokens: set[str], import_terms: set[str]) -> int:
    if candidate.relative_path == selected.relative_path:
        return 0
    candidate_path = Path(candidate.relative_path)
    selected_path = Path(selected.relative_path)
    candidate_tokens = _tokens(candidate.relative_path)
    selected_tokens = _tokens(selected.relative_path)
    score = 0

    if candidate_path.parent == selected_path.parent:
        score += 8
    if candidate.category == "test" and (candidate_tokens & selected_tokens or candidate_tokens & task_tokens):
        score += 18
    if candidate_tokens & import_terms:
        score += 18
    if candidate_tokens & RELATED_NAME_TERMS:
        score += 6
    if candidate_tokens & task_tokens:
        score += min(14, 4 * len(candidate_tokens & task_tokens))
    if {"route", "routes", "router"} & candidate_tokens:
        score += 8
    if {"schema", "model", "service", "middleware", "types", "interface"} & candidate_tokens:
        score += 6
    if candidate.category == "docs" and (candidate_path.name.lower().startswith("readme") or candidate_tokens & task_tokens):
        score += 4
    if candidate.estimated_tokens > 8000:
        score -= 12
    return score


def find_related_files(scan: ScanResult, task: TaskResult, max_files: int, token_budget: int) -> list[RankedFile]:
    selected = _selected_ranked_files(task)
    selected_paths = {item.file.relative_path for item in selected}
    task_tokens = set(task.keywords) | _tokens(task.task_description)
    candidates: dict[str, RankedFile] = {}

    for ranked in selected:
        import_terms = _import_targets(ranked.file)
        for file in scan.scanned_files:
            if file.relative_path in selected_paths:
                continue
            score = _related_score(file, ranked.file, task_tokens, import_terms)
            if score < 12:
                continue
            reason = f"related to {ranked.file.relative_path}"
            existing = candidates.get(file.relative_path)
            if existing is None or score > existing.score:
                candidates[file.relative_path] = RankedFile(file=file, score=score, reasons=[reason], matched_terms=sorted(task_tokens & _tokens(file.relative_path)))

    related = sorted(candidates.values(), key=lambda item: (-item.score, item.file.estimated_tokens, item.file.relative_path))
    base_tokens = sum(item.file.estimated_tokens for item in selected)
    remaining_slots = max(0, max_files - len(selected))
    accepted: list[RankedFile] = []
    used_tokens = base_tokens
    for item in related:
        if len(accepted) >= remaining_slots:
            break
        if used_tokens + item.file.estimated_tokens > token_budget:
            continue
        accepted.append(item)
        used_tokens += item.file.estimated_tokens
    return accepted


def _fit_to_budget(primary: list[RankedFile], supporting: list[RankedFile], token_budget: int, max_files: int) -> tuple[list[RankedFile], list[RankedFile]]:
    selected_primary: list[RankedFile] = []
    selected_supporting: list[RankedFile] = []
    used = 0
    for item in primary:
        if len(selected_primary) + len(selected_supporting) >= max_files:
            break
        selected_primary.append(item)
        used += item.file.estimated_tokens
    for item in supporting:
        if len(selected_primary) + len(selected_supporting) >= max_files:
            break
        if used + item.file.estimated_tokens > token_budget:
            continue
        selected_supporting.append(item)
        used += item.file.estimated_tokens
    return selected_primary, selected_supporting


def run_agent_plan(repo_path: Path, task_description: str, token_budget: int = 30000, max_files: int = 25) -> tuple[AgentState, Path, Path, Path]:
    repo_path = repo_path.resolve()
    state = AgentState.create(repo_path, task_description, token_budget)

    scan = scan_repo(repo_path)
    state.steps_taken.append("Scanned repository")

    analysis = analyze_task(task_description)
    state.task_type = analysis.intent
    state.interpretation = task_interpretation(analysis.intent)
    state.steps_taken.append("Analyzed and classified task")

    task = rank_scanned_files(scan, task_description)
    state.steps_taken.append("Ranked task-relevant files")

    base_selected = _selected_ranked_files(task)
    related = find_related_files(scan, task, max_files=max_files, token_budget=token_budget)
    state.steps_taken.append("Expanded related files")

    primary, supporting = _fit_to_budget(task.primary_files, [*task.supporting_files, *related], token_budget, max_files)
    related_paths = {item.file.relative_path for item in related}
    state.added_related_files = [item.file.relative_path for item in supporting if item.file.relative_path in related_paths]
    agent_task = TaskResult(
        task_description=task.task_description,
        keywords=task.keywords,
        ranked_files=[*primary, *supporting],
        intent=task.intent,
        primary_files=primary,
        supporting_files=supporting,
        glossary_terms=task.glossary_terms,
        confidence=task.confidence,
        context_only_files=task.context_only_files,
        semantic_chunks=task.semantic_chunks,
    )
    state.selected_files = [item.file.relative_path for item in [*primary, *supporting]]
    state.estimated_tokens = sum(item.file.estimated_tokens for item in [*primary, *supporting])
    state.steps_taken.append("Fit final file list inside token budget")

    packet_path, packet_tokens = generate_packet(scan, agent_task, max_tokens=token_budget)
    state.estimated_tokens = packet_tokens
    state.steps_taken.append("Generated context packet")

    state = review_context(state, base_selected)
    state.steps_taken.append("Reviewed missing context")
    state.steps_taken.append("Wrote agent report and JSON")
    report_path = write_agent_report(repo_path, state, primary, supporting)
    json_path = write_agent_json(repo_path, state)
    return state, packet_path, report_path, json_path
