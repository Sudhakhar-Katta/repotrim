"""Typer CLI entrypoint for RepoTrim."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .log_compressor import compress_log
from .agents.workflow import run_agent_plan
from .models import LogSummary, ScanResult, TaskResult
from .packet_generator import generate_packet
from .relevance_ranker import rank_scanned_files
from .reporter import render_task_report
from .scanner import find_repo_root, scan_repo
from .semantic import DEFAULT_MODEL_NAME, index_repo, semantic_search
from .token_savings import (
    calculate_token_savings,
    default_packet_path,
    render_token_savings_report,
    save_token_savings,
)
from .utils import cache_dir, percent_savings, read_json

app = typer.Typer(help="Create minimal context packets for AI coding agents.")
agent_app = typer.Typer(help="Plan agent-ready context without editing code.")
app.add_typer(agent_app, name="agent")
console = Console()
CONTEXT_PACKET_WRITTEN_MESSAGE = "✓ context_packet.md written — paste into Claude or Codex to one-shot this."
CONTEXT_PACKET_WRITTEN_ASCII_MESSAGE = "context_packet.md written - paste into Claude or Codex to one-shot this."


def _print_context_packet_written() -> None:
    encoding = getattr(console.file, "encoding", None) or "utf-8"
    try:
        CONTEXT_PACKET_WRITTEN_MESSAGE.encode(encoding)
    except UnicodeEncodeError:
        message = CONTEXT_PACKET_WRITTEN_ASCII_MESSAGE
    else:
        message = CONTEXT_PACKET_WRITTEN_MESSAGE
    console.print(message)


def _load_scan() -> ScanResult:
    path = cache_dir() / "scan.json"
    if not path.exists():
        raise typer.BadParameter("Missing .repotrim/scan.json. Run: repotrim scan .")
    return ScanResult.from_dict(read_json(path))


def _load_task() -> TaskResult:
    path = cache_dir() / "task.json"
    if not path.exists():
        raise typer.BadParameter('Missing .repotrim/task.json. Run: repotrim task "your task"')
    return TaskResult.from_dict(read_json(path))


def _load_log_optional() -> LogSummary | None:
    path = cache_dir() / "log_summary.json"
    if not path.exists():
        return None
    return LogSummary.from_dict(read_json(path))


@app.command()
def scan(repo_path: Path = typer.Argument(Path("."), help="Repository path to scan.")) -> None:
    """Scan a repository and cache file metadata."""
    result = scan_repo(repo_path)
    console.print("\n[bold]RepoTrim Scan Report[/bold]\n")
    console.print(f"Repo path: {result.repo_path}")
    console.print(f"Files scanned: {result.files_scanned}")
    console.print(f"Files ignored: {result.files_ignored}")
    console.print(f"Files discovered: {result.files_discovered}")
    console.print(f"Files unreadable: {result.files_unreadable}")
    console.print(f"Extensions found: {', '.join(result.extensions_found) or 'None'}")
    console.print(f"Estimated full repo tokens: {result.total_tokens:,}")

    largest = result.largest_files(5)
    if largest:
        console.print("\n[bold]Largest included files:[/bold]")
        for index, file in enumerate(largest, start=1):
            console.print(f"{index}. {file.relative_path} - {file.estimated_tokens:,} tokens")

    ignored_folders = sorted({item.relative_path for item in result.ignored if "directory" in item.reason})
    if ignored_folders:
        console.print("\n[bold]Ignored folders:[/bold]")
        for folder in ignored_folders[:20]:
            console.print(f"- {folder}")

    if any("lock" in item.path.lower() for item in result.ignored):
        console.print("\n[yellow]Large lock files were ignored by default.[/yellow]")
    if any("secret-looking" in item.reason for item in result.ignored):
        console.print("[yellow]Secret-looking files were skipped and excluded from cache.[/yellow]")


@app.command()
def index(
    repo_path: Path = typer.Argument(Path("."), help="Repository path to index."),
    model: str = typer.Option(DEFAULT_MODEL_NAME, "--model", help="Sentence-transformers model name."),
    strict: bool = typer.Option(False, "--strict", help="Require sentence-transformers instead of using the local fallback."),
) -> None:
    """Build or refresh the semantic chunk embedding index."""
    scan_result, semantic_index, output_path = index_repo(repo_path, model_name=model, allow_fallback=not strict)
    console.print("\n[bold]RepoTrim Semantic Index[/bold]\n")
    console.print(f"Repo path: {scan_result.repo_path}")
    console.print(f"Files indexed: {scan_result.files_scanned}")
    console.print(f"Chunks indexed: {len(semantic_index.chunks)}")
    console.print(f"Embedding provider: {semantic_index.provider}")
    console.print(f"Embedding model: {semantic_index.model_name}")
    console.print(f"Output: {output_path}")


@app.command()
def task(
    task_description: str = typer.Argument(..., help="Coding task to optimize context for."),
    path: Path | None = typer.Option(None, "--path", help="Repository path to scan. Defaults to the containing Git repository."),
    semantic: bool = typer.Option(False, "--semantic", help="Blend local semantic chunk search into file ranking."),
    model: str = typer.Option(DEFAULT_MODEL_NAME, "--model", help="Sentence-transformers model name for semantic search."),
    semantic_limit: int = typer.Option(12, "--semantic-limit", help="Number of semantic chunks to retrieve."),
    strict_semantic: bool = typer.Option(False, "--strict-semantic", help="Require sentence-transformers instead of using the local fallback."),
) -> None:
    """Scan, rank, and report files relevant to a coding task."""
    scan_path = path if path is not None else find_repo_root(Path.cwd())
    semantic_matches = []
    semantic_index = None
    if semantic:
        scan_result, semantic_index, semantic_matches, _ = semantic_search(
            scan_path,
            task_description,
            model_name=model,
            limit=semantic_limit,
            allow_fallback=not strict_semantic,
        )
    else:
        scan_result = scan_repo(scan_path)
    log_summary = _load_log_optional()
    result = rank_scanned_files(scan_result, task_description, log_summary, semantic_matches=semantic_matches)
    console.print("\n" + render_task_report(scan_result, result))
    if semantic and semantic_index:
        console.print(
            f"\nSemantic search: {len(semantic_matches)} chunks from {semantic_index.provider} "
            f"({semantic_index.model_name})"
        )
    packet_path, _ = generate_packet(scan_result, result, log_summary)
    _print_context_packet_written()
    token_savings = calculate_token_savings(scan_result, packet_path)
    save_token_savings(Path(scan_result.repo_path), token_savings)
    console.print("\n" + render_token_savings_report(token_savings))


@agent_app.command("plan")
def agent_plan(
    task_description: str = typer.Argument(..., help="Coding task to plan context for."),
    repo_path: Path = typer.Option(Path("."), "--repo-path", help="Repository path to scan."),
    token_budget: int = typer.Option(30000, "--token-budget", help="Maximum estimated context packet tokens."),
    max_files: int = typer.Option(25, "--max-files", help="Maximum selected files."),
) -> None:
    """Build an agent-ready context plan and report without editing code."""
    state, packet_path, report_path, json_path = run_agent_plan(
        repo_path=repo_path,
        task_description=task_description,
        token_budget=token_budget,
        max_files=max_files,
    )

    console.print("\n[bold]RepoTrim Agent Plan Complete[/bold]\n")
    console.print(f"Task: {state.task}")
    console.print(f"Task Interpretation: {state.interpretation}")
    console.print(f"Context Quality: {state.quality_score}/100")
    console.print(f"Selected Files: {len(state.selected_files)}")
    console.print(f"Estimated Tokens: {state.estimated_tokens:,} / {state.token_budget:,}")
    console.print("\n[bold]Warnings:[/bold]")
    for warning in state.warnings or ["None"]:
        console.print(f"- {warning}")
    console.print("\n[bold]Written:[/bold]")
    for path in (packet_path, report_path, json_path):
        console.print(f"- {path.relative_to(Path(state.repo_path)).as_posix()}")


@app.command()
def savings(
    repo_path: Path = typer.Argument(Path("."), help="Repository path to scan."),
    packet: Path | None = typer.Option(None, "--packet", help="Path to a custom context packet."),
) -> None:
    """Compare full repository tokens against a context packet."""
    scan_result = scan_repo(repo_path)
    resolved_repo_path = Path(scan_result.repo_path)
    packet_path = packet if packet is not None else default_packet_path(resolved_repo_path)
    if not packet_path.is_absolute():
        packet_path = (Path.cwd() / packet_path).resolve()
    try:
        token_savings = calculate_token_savings(scan_result, packet_path)
    except FileNotFoundError as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(1) from exc
    save_token_savings(resolved_repo_path, token_savings)
    console.print("\n" + render_token_savings_report(token_savings))


@app.command()
def log(log_path: Path = typer.Argument(..., help="Path to a build/runtime error log.")) -> None:
    """Compress an error log into useful lines and mentions."""
    summary = compress_log(log_path)
    console.print("\n[bold]Compressed Log Summary[/bold]\n")
    console.print("[bold]Important lines:[/bold]")
    for line in summary.important_lines[:20]:
        console.print(f"- {line}")
    console.print("\n[bold]Likely mentioned files:[/bold]")
    for file in summary.mentioned_files or ["None detected"]:
        console.print(f"- {file}")
    console.print("\n[bold]Noise removed:[/bold]")
    for note in summary.noise_removed_notes:
        console.print(f"- {note}")


@app.command()
def packet() -> None:
    """Generate context_packet.md from cached scan, task, and optional log summary."""
    scan_result = _load_scan()
    task_result = _load_task()
    log_summary = _load_log_optional()
    output_path, packet_tokens = generate_packet(scan_result, task_result, log_summary)
    savings = percent_savings(packet_tokens, scan_result.total_estimated_tokens)

    console.print(f"\n[bold]Generated {output_path.name}[/bold]\n")
    console.print(f"Full repo estimate: {scan_result.total_estimated_tokens:,} tokens")
    console.print(f"Packet estimate: {packet_tokens:,} tokens")
    console.print(f"Estimated savings: {savings:.1f}%")

    included = task_result.ranked_files[:5]
    console.print("\n[bold]Included:[/bold]")
    console.print(f"- {len(included)} files")
    console.print("- full selected file contents")
    console.print("- entry points")
    console.print("- one-shot prompt")


if __name__ == "__main__":
    app()
