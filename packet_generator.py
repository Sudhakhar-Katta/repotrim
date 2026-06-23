"""Generate AI-ready context packets from ranked task results."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from . import config
from .models import LogSummary, RankedFile, ScanResult, TaskResult
from .scanner import read_text_safely
from .tokenizer import estimate_tokens
from .utils import ensure_cache_dir


@dataclass
class PacketFile:
    ranked: RankedFile
    content: str
    truncated: bool = False


def _code_fence_language(language: str) -> str:
    return {
        "csharp": "csharp",
        "typescript": "typescript",
        "javascript": "javascript",
        "python": "python",
        "java": "java",
        "html": "html",
        "css": "css",
        "scss": "scss",
        "json": "json",
        "yaml": "yaml",
        "markdown": "markdown",
        "sql": "sql",
    }.get(language, "")


def _code_fence(content: str) -> str:
    longest = max((len(match.group(0)) for match in re.finditer(r"`+", content)), default=0)
    return "`" * max(3, longest + 1)


def _selected_ranked_files(task: TaskResult) -> list[RankedFile]:
    selected: list[RankedFile] = []
    seen: set[str] = set()
    for ranked in [*task.primary_files, *task.supporting_files]:
        if ranked.file.relative_path in seen:
            continue
        seen.add(ranked.file.relative_path)
        selected.append(ranked)
    return selected


def _entry_reason(ranked: RankedFile) -> str:
    if ranked.reasons:
        return ranked.reasons[0].rstrip(".")
    name = Path(ranked.file.relative_path).name
    return f"{name} is the highest-ranked direct match"


def _entry_points(task: TaskResult) -> list[str]:
    selected = _selected_ranked_files(task)
    if not selected:
        return ["- No strong entry points were found by the ranking pass."]
    lines: list[str] = []
    for ranked in selected[:3]:
        lines.append(f"- `{ranked.file.relative_path}` — score {ranked.score}; {_entry_reason(ranked)}.")
    return lines


def _file_section(packet_file: PacketFile) -> list[str]:
    file = packet_file.ranked.file
    language = _code_fence_language(file.language)
    fence = _code_fence(packet_file.content)
    suffix = " (truncated to fit packet budget)" if packet_file.truncated else ""
    return [
        f"### {file.relative_path}{suffix}",
        "",
        f"{fence}{language}",
        packet_file.content.rstrip(),
        fence,
        "",
    ]


def _base_lines(task: TaskResult) -> list[str]:
    return [
        "# RepoTrim Context Packet",
        "",
        "## Task",
        "",
        task.task_description,
        "",
        "## Entry points",
        "",
        *_entry_points(task),
        "",
        "## Relevant files with full contents",
        "",
    ]


def _semantic_chunk_lines(task: TaskResult) -> list[str]:
    if not task.semantic_chunks:
        return []
    lines = ["## Top semantic chunks", ""]
    for match in task.semantic_chunks[:8]:
        fence = _code_fence(match.text)
        label = match.symbol or f"lines {match.start_line}-{match.end_line}"
        lines.extend(
            [
                f"### {match.file_path}::{label}",
                "",
                f"- Kind: {match.kind}",
                f"- Lines: {match.start_line}-{match.end_line}",
                f"- Similarity: {match.similarity:.3f}",
                "",
                fence,
                match.text.rstrip(),
                fence,
                "",
            ]
        )
    return lines


def _closing_lines(task: TaskResult) -> list[str]:
    return [
        "## One-shot prompt",
        "",
        f"Given the context above, implement the following: {task.task_description}. Make minimal changes. Return only the modified files with their full contents.",
    ]


def _render(task: TaskResult, packet_files: list[PacketFile]) -> str:
    lines = _base_lines(task)
    if packet_files:
        for packet_file in packet_files:
            lines.extend(_file_section(packet_file))
    else:
        lines.extend(["No primary or supporting files were selected.", ""])
    lines.extend(_semantic_chunk_lines(task))
    lines.extend(_closing_lines(task))
    return "\n".join(lines).rstrip() + "\n"


def _truncate_to_tokens(content: str, max_tokens: int, packet_budget: int) -> str:
    if max_tokens <= 0:
        return ""
    max_chars = max(0, max_tokens * 4)
    if len(content) <= max_chars:
        return content
    marker = f"\n\n[... truncated to keep context_packet.md under {packet_budget:,} tokens ...]"
    keep_chars = max(0, max_chars - len(marker))
    return content[:keep_chars].rstrip() + marker


def _build_packet_files(task: TaskResult, max_tokens: int | None = None) -> list[PacketFile]:
    packet_budget = max_tokens or config.MAX_CONTEXT_PACKET_TOKENS

    def file_content(ranked: RankedFile) -> str:
        text = read_text_safely(Path(ranked.file.path)) or ""
        return text.replace("\r\n", "\n").replace("\r", "\n")

    primary = [PacketFile(ranked, file_content(ranked)) for ranked in task.primary_files]
    supporting = [PacketFile(ranked, file_content(ranked)) for ranked in task.supporting_files]
    packet_files = [*primary]
    rendered = _render(task, packet_files)
    remaining = packet_budget - estimate_tokens(rendered)

    for packet_file in supporting:
        full_section_tokens = estimate_tokens("\n".join(_file_section(packet_file)))
        if full_section_tokens <= remaining:
            packet_files.append(packet_file)
            remaining -= full_section_tokens
            continue
        if remaining <= 40:
            continue
        truncated_content = _truncate_to_tokens(packet_file.content, remaining - 40, packet_budget)
        truncated = PacketFile(packet_file.ranked, truncated_content, truncated=True)
        if estimate_tokens("\n".join(_file_section(truncated))) <= remaining:
            packet_files.append(truncated)
            remaining = 0
        break
    return packet_files


def generate_packet(
    scan: ScanResult,
    task: TaskResult,
    log_summary: LogSummary | None = None,
    output_path: Path | None = None,
    max_tokens: int | None = None,
) -> tuple[Path, int]:
    """Write .repotrim/context_packet.md for the ranked task result."""
    del log_summary
    output_path = output_path or ensure_cache_dir(Path(scan.repo_path)) / "context_packet.md"
    packet_files = _build_packet_files(task, max_tokens=max_tokens)
    text = _render(task, packet_files)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="\n")
    return output_path, estimate_tokens(text)
