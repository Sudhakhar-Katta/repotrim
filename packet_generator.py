"""Generate AI-ready context packets from scan, task, and optional log data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import config
from .models import FileInfo, LogSummary, ScanResult, TaskResult
from .scanner import read_text_safely
from .tokenizer import estimate_tokens
from .utils import percent_savings


@dataclass
class ExtractedContext:
    relative_path: str
    language: str
    estimated_tokens: int
    mode: str
    content: str


def _merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not ranges:
        return []
    ranges.sort()
    merged = [ranges[0]]
    for start, end in ranges[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + 1:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


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


def extract_context(file: FileInfo, keywords: list[str], log_summary: LogSummary | None = None) -> ExtractedContext:
    text = read_text_safely(Path(file.path)) or ""
    if file.estimated_tokens <= config.SMALL_FILE_TOKEN_LIMIT:
        return ExtractedContext(file.relative_path, file.language, estimate_tokens(text), "full file", text)

    lines = text.splitlines()
    ranges = [(0, min(40, len(lines)))]
    match_terms = [item.lower() for item in keywords]
    if log_summary:
        match_terms.extend(item.lower() for item in log_summary.mentioned_files)
        match_terms.extend(item.lower() for item in log_summary.mentioned_symbols)

    for index, line in enumerate(lines):
        lower = line.lower()
        if any(term and term in lower for term in match_terms):
            start = max(0, index - config.SNIPPET_LINE_WINDOW)
            end = min(len(lines), index + config.SNIPPET_LINE_WINDOW + 1)
            ranges.append((start, end))

    chunks: list[str] = []
    token_budget = 0
    for start, end in _merge_ranges(ranges):
        chunk = "\n".join(lines[start:end])
        chunk_tokens = estimate_tokens(chunk)
        if token_budget + chunk_tokens > config.MAX_SNIPPET_TOKENS_PER_FILE and chunks:
            break
        chunks.append(f"# Lines {start + 1}-{end}\n{chunk}")
        token_budget += chunk_tokens

    content = "\n\n# ...\n\n".join(chunks)
    return ExtractedContext(file.relative_path, file.language, estimate_tokens(content), "snippets", content)


def investigation_steps(keywords: list[str]) -> list[str]:
    words = set(keywords)
    if words & {"update", "save", "saved", "saving"}:
        return [
            "Check where the frontend sends the update request.",
            "Check the request payload/DTO.",
            "Check the backend update endpoint.",
            "Check model/entity mapping.",
            "Check persistence/database save call.",
        ]
    if words & {"login", "signin", "auth", "authentication"}:
        return [
            "Check auth route/controller.",
            "Check token/session handling.",
            "Check frontend login form.",
            "Check redirect or route guard.",
            "Check error response handling.",
        ]
    if words & {"database", "db", "foreign", "key", "constraint", "schema"}:
        return [
            "Check entity relationships.",
            "Check migration/schema.",
            "Check insert/update order.",
            "Check foreign key values.",
            "Check transaction/save logic.",
        ]
    return [
        "Start with the highest-ranked file.",
        "Trace the flow through related services/components.",
        "Use the compressed error log to identify the failure point.",
        "Modify the smallest number of files needed.",
        "Add or update tests if available.",
    ]


def generate_packet(scan: ScanResult, task: TaskResult, log_summary: LogSummary | None = None, output_path: Path | None = None) -> tuple[Path, int]:
    selected = task.ranked_files[:5]
    contexts = [extract_context(item.file, task.keywords, log_summary) for item in selected]
    packet_tokens = sum(context.estimated_tokens for context in contexts)
    savings = percent_savings(packet_tokens, scan.total_estimated_tokens)
    output_path = output_path or Path.cwd() / "context_packet.md"

    lines: list[str] = [
        "# RepoTrim Context Packet",
        "",
        "## Task",
        "",
        task.task_description,
        "",
        "## Purpose",
        "",
        "Use this packet as focused context for an AI coding agent. The goal is to solve the task without sending the entire repository.",
        "",
        "## Token Estimate",
        "",
        f"- Full repo estimate: {scan.total_estimated_tokens:,} tokens",
        f"- Selected packet estimate: {packet_tokens:,} tokens",
        f"- Estimated savings: {savings:.1f}%",
        "",
        "## Relevant Files",
        "",
        "| Rank | File | Score | Why Included |",
        "| --- | --- | ---: | --- |",
    ]
    for index, ranked in enumerate(selected, start=1):
        why = "; ".join(ranked.reasons[:4]) or "ranked by task relevance"
        lines.append(f"| {index} | `{ranked.file.relative_path}` | {ranked.score} | {why} |")

    lines.extend(["", "## Error Log Summary", ""])
    if log_summary:
        lines.append("### Main Important Lines")
        lines.extend(f"- {line}" for line in log_summary.important_lines[:20])
        lines.extend(["", "### Mentioned Files"])
        lines.extend(f"- {file}" for file in log_summary.mentioned_files or ["None detected"])
        lines.extend(["", "### Error Codes"])
        lines.extend(f"- {code}" for code in log_summary.error_codes or ["None detected"])
    else:
        lines.append("No error log was provided.")

    lines.extend(["", "## Relevant Code Context", ""])
    for context in contexts:
        fence = _code_fence_language(context.language)
        lines.extend(
            [
                f"### File: {context.relative_path}",
                "",
                f"- Estimated tokens: {context.estimated_tokens:,}",
                f"- Included as: {context.mode}",
                "",
                f"```{fence}",
                context.content.rstrip(),
                "```",
                "",
            ]
        )

    lines.extend(["## Suggested Investigation Steps", ""])
    lines.extend(f"- {step}" for step in investigation_steps(task.keywords))
    lines.extend(
        [
            "",
            "## Constraints for the AI Coding Agent",
            "",
            "- Do not modify unrelated files unless necessary.",
            "- Prefer minimal changes.",
            "- Preserve existing public APIs unless the task requires changing them.",
            "- Explain any extra files you need before using them.",
            "- If context is missing, ask for the specific missing file instead of guessing.",
            "",
            "## Excluded Context",
            "",
            "- node_modules",
            "- dist",
            "- build",
            "- package-lock.json",
            "- unrelated low-score files",
        ]
    )

    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return output_path, packet_tokens
