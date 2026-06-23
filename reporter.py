"""Human-readable deterministic task reports."""

from __future__ import annotations

from .models import RankedFile, ScanResult, TaskResult
from .task_analyzer import INTENT_LABELS
from .utils import percent_savings


INTERPRETATIONS = {
    "access_control": "This is an RBAC/access-control task. Identify identity and role sources, permission enforcement, protected routes, and restricted UI actions.",
    "authentication": "This is an authentication task. Trace login/session or token creation, validation middleware, user lookup, and client auth state.",
    "frontend_feature": "This looks like a frontend UI change. Start with the owning page/component, then trace state, forms, styles, and API interactions.",
    "backend_feature": "This looks like a backend API change. Start with the endpoint/controller, then trace validation, services, persistence, and tests.",
    "database_feature": "This is a database/schema task. Trace models, migrations, repositories, queries, and compatibility with existing data.",
    "travel_planning": "This is a travel/departure planning task. Trace origin and destination input, route duration or traffic data, ETA/departure calculations, and result display.",
    "ml_model_debug": "This is an ML/debugging task. Check model loading, preprocessing, device handling, inference behavior, and focused reproduction.",
    "config_deployment": "This is a configuration/deployment task. Trace environment settings, build/container files, deployment manifests, and startup behavior.",
    "testing_task": "This is a testing task. Identify the behavior under test, existing fixtures/mocks, and the smallest focused regression test.",
    "general_code_task": "This is a general code task. Start with the strongest direct match, trace callers and tests, and keep the change focused.",
}

WORKFLOWS = {
    "access_control": ["Identify the identity and role source.", "Open auth, guard, policy, and permission files.", "Map roles to permissions.", "Enforce permissions on backend routes/services.", "Guard restricted frontend routes/actions.", "Add focused tests for allowed and denied roles."],
    "authentication": ["Trace login or token issuance.", "Check credential and session validation.", "Inspect auth middleware and user lookup.", "Check client auth state and protected navigation.", "Test success, expiry, and rejection paths."],
    "frontend_feature": ["Find the owning page/component.", "Trace state and user input.", "Check API interactions and error states.", "Verify styles and accessibility.", "Add a focused UI test."],
    "backend_feature": ["Find the endpoint/controller.", "Trace validation and service logic.", "Check persistence and response contracts.", "Verify error handling.", "Add a focused API test."],
    "database_feature": ["Open models and schema definitions.", "Check migrations and existing data assumptions.", "Trace repositories and queries.", "Verify transactions and constraints.", "Add a focused persistence test."],
    "travel_planning": ["Find origin/destination input state.", "Trace route, maps, or geocoding calls.", "Locate duration, traffic, ETA, and departure calculations.", "Check time-zone and fallback behavior.", "Verify the result display with a focused route example."],
    "ml_model_debug": ["Reproduce the failure minimally.", "Open the primary model/detector file.", "Check model loading and preprocessing.", "Check CPU/GPU device handling.", "Check evaluation mode and output interpretation.", "Add a focused inference test."],
    "config_deployment": ["Identify the target environment.", "Open configuration and deployment files.", "Trace environment-variable usage.", "Check build/startup commands.", "Validate with the smallest deployment or build check."],
    "testing_task": ["Identify the exact behavior to verify.", "Open related tests and fixtures.", "Reuse existing test helpers.", "Add the smallest regression case.", "Run the focused test and nearby suite."],
    "general_code_task": ["Reproduce the issue minimally.", "Open the highest-ranked source file.", "Trace callers and configuration.", "Make the smallest coherent change.", "Run or add a focused test."],
}


def _file_lines(items: list[RankedFile]) -> list[str]:
    lines: list[str] = []
    for index, ranked in enumerate(items, start=1):
        lines.extend([f"{index}. {ranked.file.relative_path}", f"   Score: {ranked.score}", f"   Tokens: {ranked.file.estimated_tokens:,}", "   Why:"])
        lines.extend(f"   - {reason}" for reason in (ranked.reasons[:5] or ["ranked by task relevance"]))
        lines.append("")
    if not items:
        lines.append("No strong matches found.\n")
    return lines


def _scan_status(scan: ScanResult, task: TaskResult) -> str:
    if scan.files_scanned == 0:
        return "RepoTrim did not index any files. Check repo root, extension filters, ignore rules, or file reading errors."
    if scan.total_tokens == 0:
        return "RepoTrim indexed files, but they contained no tokenizable text. Check empty/generated files and file reading errors."
    if task.confidence == "none":
        return "RepoTrim indexed the repo, but no relevant matches were found. Consider adding aliases to .repotrimrc.json."
    if task.confidence == "weak":
        return "RepoTrim indexed the repo, but only weak keyword matches were found. Consider adding aliases to .repotrimrc.json."
    return "RepoTrim found strong task-specific matches."


def render_task_report(scan: ScanResult, task: TaskResult) -> str:
    selected = task.primary_files + task.supporting_files
    selected_tokens = sum(item.file.estimated_tokens for item in selected)
    extensions = ", ".join(f"{extension}: {count}" for extension, count in scan.extensions_found.items()) or "None"
    ignored_junk = [item for item in scan.ignored_files if any(term in item.relative_path.lower() for term in (".claude", "worktree", "cache", "node_modules", "dist", "build"))]
    lines = [
        "RepoTrim Task Report", "", "Task:", task.task_description, "",
        "Task type:", INTENT_LABELS[task.intent], "", "Task interpretation:", INTERPRETATIONS[task.intent], "",
        "Scan diagnostics:",
        f"- Repo root: {scan.repo_path}",
        f"- Current working directory: {scan.current_working_directory or scan.repo_path}",
        f"- Files discovered: {scan.files_discovered}",
        f"- Files ignored: {scan.files_ignored}",
        f"- Files indexed: {scan.files_scanned}",
        f"- Files unreadable: {scan.files_unreadable}",
        f"- Extensions found: {extensions}",
        f"- Full repo tokens: {scan.total_tokens:,}",
        f"- Selected file tokens: {selected_tokens:,}",
        "", "Result quality:", _scan_status(scan, task), "",
    ]
    if task.glossary_terms:
        lines.extend(["Repo glossary expansion:", f"- {', '.join(task.glossary_terms)}", ""])
    if task.context_only_files:
        lines.extend([
            "Ranking exclusions:",
            f"- context-only match: tool/tools or similar project nouns, not enough RBAC evidence ({len(task.context_only_files)} file(s) excluded)",
            "",
        ])
    lines.append("Primary files:")
    lines.extend(_file_lines(task.primary_files))
    lines.append("Supporting files:")
    lines.extend(_file_lines(task.supporting_files))
    lines.extend(["Ignored duplicate/junk files:", f"- {len(ignored_junk)} ignored path(s) excluded from ranking", "", "Suggested fix workflow:"])
    lines.extend(f"{index}. {step}" for index, step in enumerate(WORKFLOWS[task.intent], start=1))
    lines.extend(["", "Estimated selected context:", f"Selected files: {selected_tokens:,} tokens", f"Full repo: {scan.total_tokens:,} tokens", f"Estimated savings: {percent_savings(selected_tokens, scan.total_tokens):.1f}%"])
    return "\n".join(lines)
