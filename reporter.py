"""Human-readable deterministic task reports."""

from __future__ import annotations

from .models import RankedFile, ScanResult, TaskResult
from .task_analyzer import INTENT_LABELS
from .utils import percent_savings


INTERPRETATIONS = {
    "ml_model_debug": "This looks like a broad CNN/model debugging task. The exact failure is unknown, so start by checking model loading, preprocessing, device handling, and inference behavior.",
    "frontend_feature": "This looks like a frontend change. Start with the component or page that owns the behavior, then trace its state, styles, and API interactions.",
    "backend_feature": "This looks like a backend change. Start with the route or endpoint, then trace validation, service logic, persistence, and focused tests.",
    "general_code_task": "This is a general code task. Start with the strongest direct match, trace its callers and tests, and keep the change focused.",
}

WORKFLOWS = {
    "ml_model_debug": [
        "Reproduce the CNN failure with the smallest command or script.",
        "Open the primary detector/model file first.",
        "Check the model loading path and checkpoint compatibility.",
        "Check image preprocessing: size, tensor shape, color channel order, and normalization.",
        "Check CPU/GPU device handling.",
        "Check whether model.eval() is called before inference.",
        "Check confidence thresholds and output interpretation.",
        "Inspect ensemble code only if CNN output is combined with other detectors.",
        "Run or add a focused test for one input image.",
    ],
    "frontend_feature": [
        "Reproduce the behavior on the smallest page or component.",
        "Open the primary component and trace its props and state.",
        "Check event handling, API calls, and loading/error states.",
        "Verify related styles and responsive behavior.",
        "Run or add a focused component test.",
    ],
    "backend_feature": [
        "Reproduce the request with the smallest API call.",
        "Open the primary route or endpoint first.",
        "Trace validation, service logic, and persistence.",
        "Check error handling and response contracts.",
        "Run or add a focused endpoint test.",
    ],
    "general_code_task": [
        "Reproduce the issue with the smallest command or test.",
        "Open the highest-ranked source file first.",
        "Trace direct callers and configuration.",
        "Make the smallest coherent change.",
        "Run or add a focused test.",
    ],
}


def _file_lines(items: list[RankedFile]) -> list[str]:
    lines: list[str] = []
    for index, ranked in enumerate(items, start=1):
        lines.extend([
            f"{index}. {ranked.file.relative_path}",
            f"   Score: {ranked.score}",
            f"   Tokens: {ranked.file.estimated_tokens:,}",
            "   Why:",
        ])
        lines.extend(f"   - {reason}" for reason in (ranked.reasons[:4] or ["ranked by task relevance"]))
        lines.append("")
    if not items:
        lines.append("No strong matches found.\n")
    return lines


def render_task_report(scan: ScanResult, task: TaskResult) -> str:
    selected = task.primary_files + task.supporting_files
    selected_tokens = sum(item.file.estimated_tokens for item in selected)
    ignored_junk = [item.relative_path for item in scan.ignored_files if any(term in item.relative_path.lower() for term in (".claude", "worktree", "cache", "node_modules", "dist", "build"))]
    lines = [
        "RepoTrim Task Report", "", "Task:", task.task_description, "",
        "Task type:", INTENT_LABELS[task.intent], "", "Task interpretation:", INTERPRETATIONS[task.intent], "",
        "Primary files:",
    ]
    lines.extend(_file_lines(task.primary_files))
    lines.append("Supporting files:")
    lines.extend(_file_lines(task.supporting_files))
    lines.append("Ignored duplicate/junk files:")
    lines.append(f"- {len(ignored_junk)} ignored path(s) excluded from ranking")
    lines.extend(["", "Suggested fix workflow:"])
    lines.extend(f"{index}. {step}" for index, step in enumerate(WORKFLOWS[task.intent], start=1))
    lines.extend([
        "", "Estimated selected context:",
        f"Selected files: {selected_tokens:,} tokens",
        f"Full repo: {scan.total_tokens:,} tokens",
        f"Estimated savings: {percent_savings(selected_tokens, scan.total_tokens):.1f}%",
    ])
    return "\n".join(lines)
