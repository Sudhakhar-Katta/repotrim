"""Typer CLI entrypoint for RepoTrim."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .log_compressor import compress_log
from .models import LogSummary, ScanResult, TaskResult
from .packet_generator import generate_packet
from .relevance_ranker import rank_scanned_files
from .reporter import render_task_report
from .scanner import scan_repo
from .utils import cache_dir, percent_savings, read_json

app = typer.Typer(help="Create minimal context packets for AI coding agents.")
console = Console()


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
def task(
    task_description: str = typer.Argument(..., help="Coding task to optimize context for."),
    path: Path = typer.Option(Path("."), "--path", help="Repository path to scan."),
) -> None:
    """Scan, rank, and report files relevant to a coding task."""
    scan_result = scan_repo(path)
    log_summary = _load_log_optional()
    result = rank_scanned_files(scan_result, task_description, log_summary)
    console.print("\n" + render_task_report(scan_result, result))


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
    console.print("- relevant snippets")
    console.print("- compressed error log" if log_summary else "- no error log")
    console.print("- suggested investigation steps")


if __name__ == "__main__":
    app()
