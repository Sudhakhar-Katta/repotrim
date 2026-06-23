"""Report writers for the agentic context builder."""

from __future__ import annotations

from pathlib import Path

from ..models import RankedFile
from ..utils import ensure_cache_dir, write_json
from .state import AgentState


def _why_lines(items: list[RankedFile]) -> list[str]:
    lines: list[str] = []
    for item in items:
        reason = item.reasons[0] if item.reasons else "selected by task relevance"
        lines.append(f"- `{item.file.relative_path}`: {reason}")
    return lines or ["- No files were selected."]


def write_agent_report(repo_path: Path, state: AgentState, primary: list[RankedFile], related: list[RankedFile]) -> Path:
    output_path = ensure_cache_dir(repo_path) / "agent_report.md"
    lines = [
        "# RepoTrim Agent Report",
        "",
        "## Task",
        state.task,
        "",
        "## Task Interpretation",
        state.interpretation or state.task_type,
        "",
        "## Agent Steps",
    ]
    lines.extend(f"{index}. {step}" for index, step in enumerate(state.steps_taken, start=1))
    lines.extend(["", "## Selected Files", "### Primary"])
    lines.extend(f"- {item.file.relative_path}" for item in primary)
    if not primary:
        lines.append("- None")
    lines.extend(["", "### Related / Supporting"])
    lines.extend(f"- {item.file.relative_path}" for item in related)
    if not related:
        lines.append("- None")
    lines.extend(["", "## Why These Files Matter"])
    lines.extend(_why_lines([*primary, *related]))
    lines.extend(["", "## Missing Context / Warnings"])
    lines.extend(f"- {warning}" for warning in state.warnings)
    if not state.warnings:
        lines.append("- None")
    lines.extend([
        "",
        "## Context Quality Score",
        f"{state.quality_score}/100",
        "",
        "## Suggested Prompt For Coding Agent",
        "Use the selected context to implement the task. Pay special attention to the warnings above. Keep the patch minimal and add/update tests if possible.",
        "",
    ])
    output_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return output_path


def write_agent_json(repo_path: Path, state: AgentState) -> Path:
    output_path = ensure_cache_dir(repo_path) / "agent.json"
    write_json(output_path, state.to_dict())
    return output_path
