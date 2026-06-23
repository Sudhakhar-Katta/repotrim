"""Explainable task-aware ranking over files produced by the shared scanner."""

from __future__ import annotations

import re
from pathlib import Path

from .ignore_rules import has_ignored_path_part
from .models import LogSummary, RankedFile, ScanResult, SemanticChunkMatch, TaskResult
from .repo_config import RepoConfig, load_repo_config
from .scanner import read_text_safely
from .task_analyzer import TaskAnalysis, analyze_task
from .utils import ensure_cache_dir, write_json


SUPPORT_TERMS = {"config", "requirements", "pyproject", "setup", "ensemble", "test", "spec", "readme", "docs", "migration"}
LOW_SIGNAL_TERMS = {"home", "user", "access", "feature", "main", "index", "page"}
RBAC_ANCHOR_TERMS = {
    "rbac", "role", "permission", "auth", "authentication", "authorize",
    "authorization", "guard", "protected", "claim", "policy", "admin",
}
RBAC_CONTEXT_TERMS = {
    "tool", "dashboard", "route", "component", "page", "user", "employee", "module",
}
RBAC_STRUCTURAL_CONTEXT_TERMS = {"dashboard", "user", "employee", "module"}

ROLE_TERMS = {
    "access_control": {"auth", "authorization", "rbac", "role", "permission", "guard", "policy", "claims", "admin", "protected", "access"},
    "authentication": {"auth", "authentication", "login", "signin", "logout", "session", "token", "oauth", "jwt", "password"},
    "frontend_feature": {"component", "page", "view", "form", "modal", "button", "frontend", "ui"},
    "backend_feature": {"api", "route", "controller", "service", "server", "endpoint"},
    "database_feature": {"database", "schema", "migration", "model", "entity", "repository", "query", "sql"},
    "travel_planning": {"departure", "arrival", "eta", "route", "directions", "traffic", "transit", "duration", "location", "origin", "destination", "map", "geocode", "result", "form"},
    "ml_model_debug": {"model", "detector", "inference", "predict", "train", "dataset", "preprocess", "ensemble", "cuda", "torch", "tensorflow", "config", "requirements"},
    "config_deployment": {"config", "docker", "deploy", "deployment", "pipeline", "environment", "kubernetes", "compose"},
    "testing_task": {"test", "spec", "fixture", "mock", "coverage", "assert"},
}

COMBINATION_RULES = {
    "access_control": [
        ({"role", "permission"}, "role and permission evidence"),
        ({"auth", "guard"}, "boosted because auth + guard evidence"),
        ({"route", "guard"}, "boosted because route + guard evidence"),
        ({"route", "role"}, "boosted because route + role evidence"),
        ({"authorization", "policy"}, "authorization policy evidence"),
        ({"dashboard", "permission"}, "boosted because dashboard + permission evidence"),
        ({"dashboard", "role"}, "boosted because dashboard + role evidence"),
        ({"protected", "route"}, "protected route evidence"),
        ({"tool", "role"}, "boosted because tool + role evidence"),
        ({"tool", "permission"}, "boosted because tool + permission evidence"),
        ({"tool", "protected"}, "boosted because tool + protected evidence"),
        ({"user", "role"}, "boosted because user + role evidence"),
        ({"employee", "role"}, "boosted because employee + role evidence"),
    ],
    "authentication": [
        ({"login", "token"}, "login and token evidence"),
        ({"session", "user"}, "session and user evidence"),
        ({"oauth", "callback"}, "OAuth callback evidence"),
    ],
    "travel_planning": [
        ({"origin", "destination"}, "origin and destination evidence"),
        ({"route", "eta"}, "route and ETA evidence"),
        ({"arrival", "duration"}, "arrival and duration evidence"),
        ({"departure", "time"}, "departure-time evidence"),
        ({"traffic", "duration"}, "traffic-duration evidence"),
        ({"location", "input"}, "location input evidence"),
    ],
    "database_feature": [
        ({"schema", "migration"}, "schema migration evidence"),
        ({"model", "repository"}, "model repository evidence"),
    ],
    "testing_task": [
        ({"test", "fixture"}, "test fixture evidence"),
        ({"test", "mock"}, "test mock evidence"),
    ],
}


def _rbac_admin_path_priority(relative_path: str) -> tuple[int, str]:
    path = relative_path.replace("\\", "/").lower()
    name = Path(path).name
    if path == "lib/rbac.ts" or path.endswith("/lib/rbac.ts"):
        return 70, "prioritized core RBAC policy file"
    if name == "middleware.ts":
        return 62, "prioritized middleware/protected route file"
    if path == "lib/auth.ts" or path.endswith("/lib/auth.ts"):
        return 56, "prioritized auth/session role source"
    if path == "components/layout/sidebar.tsx" or path.endswith("/components/layout/sidebar.tsx"):
        return 50, "prioritized sidebar/navigation visibility file"
    if path.startswith("app/app/") and name in {"page.tsx", "page.jsx", "layout.tsx", "layout.jsx"}:
        return 44, "prioritized protected app page/layout file"
    if path.startswith("actions/") and path.endswith((".ts", ".tsx")):
        return 34, "prioritized server action enforcement file"
    if path.startswith("app/api/") and path.endswith((".ts", ".tsx")):
        return 30, "prioritized API enforcement file"
    if ("/__tests__/" in f"/{path}" or path.startswith("__tests__/")) and "rbac" in path:
        return 28, "prioritized RBAC test file"
    if (path.startswith("tests/") or "/tests/" in f"/{path}") and ("auth" in path or "admin" in path):
        return 24, "prioritized access-control test file"
    return 0, ""


def _tokens(value: str) -> set[str]:
    expanded = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value)
    tokens = {token.lower() for token in re.findall(r"[A-Za-z0-9]+", expanded)}
    tokens.update(token[:-1] for token in list(tokens) if len(token) > 3 and token.endswith("s"))
    return tokens


def _ordered_tokens(value: str) -> list[str]:
    expanded = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value)
    return [token.lower() for token in re.findall(r"[A-Za-z0-9]+", expanded)]


def _term_matches(term: str, value: str) -> bool:
    term_tokens = _tokens(term)
    if not term_tokens:
        return False
    value_tokens = _tokens(value)
    return term_tokens <= value_tokens


def _content_count(term: str, content: str) -> int:
    term_tokens = _ordered_tokens(term)
    if not term_tokens:
        return 0
    if len(term_tokens) == 1:
        return len(re.findall(rf"(?<![A-Za-z0-9]){re.escape(term_tokens[0])}(?![A-Za-z0-9])", content, flags=re.IGNORECASE))
    normalized = " ".join(re.findall(r"[A-Za-z0-9]+", content.lower()))
    phrase = " ".join(term_tokens)
    return normalized.count(phrase)


def _is_low_signal_file(file, repo_config: RepoConfig) -> bool:
    stem = Path(file.relative_path).stem.lower()
    compact = re.sub(r"[^a-z0-9]", "", stem)
    return any(re.sub(r"[^a-z0-9]", "", name) == compact for name in repo_config.low_signal_files)


def _is_route_config(file, structural_tokens: set[str]) -> bool:
    name = Path(file.relative_path).name.lower()
    return bool(
        {"route", "router", "routing"} & structural_tokens
        and (name.startswith("app.") or "route" in name or "routing" in name)
    )


def _is_rbac_context_only(file) -> bool:
    evidence = _tokens(" ".join([
        file.relative_path,
        " ".join(file.symbols),
        " ".join(file.first_lines),
        " ".join(file.imports),
        " ".join(file.comments),
    ]))
    return bool(RBAC_CONTEXT_TERMS & evidence) and not bool(RBAC_ANCHOR_TERMS & evidence) and not _is_route_config(file, _tokens(file.relative_path))


def _score_file(file, analysis: TaskAnalysis, repo_config: RepoConfig, log_summary: LogSummary | None) -> RankedFile | None:
    relative = file.relative_path
    filename = Path(relative).name
    first_lines = "\n".join(file.first_lines)
    symbols = " ".join(file.symbols)
    hints = "\n".join([*file.imports, *file.comments])
    text = read_text_safely(Path(file.path))
    content = text or first_lines
    combined_tokens = _tokens(" ".join([relative, symbols, first_lines, hints, content]))
    structural_tokens = _tokens(" ".join([relative, symbols]))
    matched_terms: set[str] = set()
    score = 0
    reasons: list[str] = []

    for keyword in analysis.keywords:
        keyword_tokens = _tokens(keyword)
        if analysis.intent == "access_control" and keyword_tokens & (RBAC_ANCHOR_TERMS | RBAC_CONTEXT_TERMS):
            continue
        weak = bool(keyword_tokens & LOW_SIGNAL_TERMS)
        if _term_matches(keyword, filename):
            score += 8 if weak else 28
            matched_terms.add(keyword)
            reasons.append(f'filename matched "{keyword}"')
        elif _term_matches(keyword, relative):
            score += 4 if weak else 13
            matched_terms.add(keyword)
            reasons.append(f'path matched "{keyword}"')
        if _term_matches(keyword, symbols):
            score += 5 if weak else 16
            matched_terms.add(keyword)
            reasons.append(f'symbol matched "{keyword}"')
        count = _content_count(keyword, content)
        if count:
            score += min(5 if weak else 18, count * (1 if weak else 3))
            matched_terms.add(keyword)
            reasons.append(f'content matched "{keyword}"')
        if _term_matches(keyword, first_lines):
            score += 3 if weak else 7
            matched_terms.add(keyword)
        if _term_matches(keyword, hints):
            score += 2 if weak else 6
            matched_terms.add(keyword)
            reasons.append(f'import/comment hint matched "{keyword}"')

    role_matches = ROLE_TERMS.get(analysis.intent, set()) & combined_tokens
    structural_role_matches = ROLE_TERMS.get(analysis.intent, set()) & structural_tokens
    route_config_match = False
    if analysis.intent == "access_control":
        anchor_matches = RBAC_ANCHOR_TERMS & combined_tokens
        structural_anchor_matches = RBAC_ANCHOR_TERMS & structural_tokens
        context_matches = RBAC_CONTEXT_TERMS & combined_tokens
        structural_context_matches = RBAC_CONTEXT_TERMS & structural_tokens
        useful_structural_context = RBAC_STRUCTURAL_CONTEXT_TERMS & structural_context_matches
        route_config_match = _is_route_config(file, structural_tokens)

        if structural_anchor_matches:
            score += 36 + min(28, len(structural_anchor_matches) * 7)
            reasons.append(f"RBAC anchor in file structure: {', '.join(sorted(structural_anchor_matches)[:3])}")
        elif anchor_matches:
            score += 24 + min(24, len(anchor_matches) * 6)
            reasons.append(f"RBAC anchor evidence: {', '.join(sorted(anchor_matches)[:3])}")

        if route_config_match:
            score += 30
            reasons.append("boosted because route file matched RBAC task type")

        if context_matches and (anchor_matches or route_config_match):
            score += min(12, len(context_matches) * 3)
            reasons.append(f"RBAC context paired with anchor: {', '.join(sorted(context_matches)[:3])}")
        elif context_matches:
            score += min(6, len(context_matches) * 2)
            reasons.append(f"context-only match: {', '.join(sorted(context_matches)[:3])}, not enough RBAC evidence")
            if useful_structural_context:
                score += 30
                reasons.append(f"supporting structural RBAC context: {', '.join(sorted(useful_structural_context)[:3])}")

        role_matches = anchor_matches | context_matches
        structural_role_matches = structural_anchor_matches | structural_context_matches

        path_priority, priority_reason = _rbac_admin_path_priority(relative)
        if path_priority:
            score += path_priority
            reasons.append(priority_reason)

    if structural_role_matches:
        if analysis.intent != "access_control":
            score += 12 + min(18, len(structural_role_matches) * 4)
            reasons.append(f"{analysis.intent.replace('_', ' ')} structural role matched: {', '.join(sorted(structural_role_matches)[:3])}")
    elif len(role_matches) >= 2 or (role_matches and matched_terms):
        if analysis.intent != "access_control":
            score += 8 + min(18, len(role_matches) * 3)
            reasons.append(f"{analysis.intent.replace('_', ' ')} role matched: {', '.join(sorted(role_matches)[:3])}")
    else:
        role_matches = set()

    evidence_tokens = set().union(*(_tokens(term) for term in matched_terms)) if matched_terms else set()
    evidence_tokens.update(role_matches)
    evidence_tokens.update(structural_role_matches)
    combination_count = 0
    for required, reason in COMBINATION_RULES.get(analysis.intent, []):
        if required <= evidence_tokens:
            score += 24
            combination_count += 1
            reasons.append(reason)

    if score > 0 and file.category == "source":
        score += 4
    elif score > 0 and file.category in {"config", "test"}:
        score += 2

    if file.category == "test" and analysis.intent != "testing_task" and len(evidence_tokens) >= 2:
        score += 5
        reasons.append("test can verify the requested behavior")

    if _is_low_signal_file(file, repo_config):
        if len(matched_terms) < 2 and combination_count == 0:
            score -= 45
            reasons.append("low-signal filename with only one weak match")
        else:
            score -= 12
            reasons.append("low-signal filename penalty")

    normalized_relative = file.relative_path.replace("\\", "/").lower()
    if "/components/ui/" in f"/{normalized_relative}" and len(matched_terms) < 2:
        score -= 30
        reasons.append("generic UI primitive without direct task evidence")

    if analysis.intent == "access_control":
        anchor_evidence = RBAC_ANCHOR_TERMS & evidence_tokens
        context_evidence = RBAC_CONTEXT_TERMS & evidence_tokens
        useful_structural_context = RBAC_STRUCTURAL_CONTEXT_TERMS & structural_tokens
        if context_evidence and not anchor_evidence and not route_config_match and not useful_structural_context:
            score -= 18
        if file.estimated_tokens < 100 and not anchor_evidence and not route_config_match:
            score -= 18
            reasons.append("very small placeholder without RBAC anchor evidence")
        if normalized_relative.startswith("app/api/"):
            score = min(score, 96)
            reasons.append("API enforcement ranked after core RBAC screen-control files")

    if log_summary:
        mentions = {Path(item).name.lower() for item in log_summary.mentioned_files}
        if filename.lower() in mentions:
            score += 40
            reasons.append("error log mentioned this file")
    if file.estimated_tokens > 20000:
        score -= 20
        reasons.append("very large file penalty")
    if score <= 0:
        return None
    return RankedFile(
        file=file,
        score=score,
        reasons=list(dict.fromkeys(reasons)),
        matched_terms=sorted(matched_terms | role_matches | structural_role_matches),
    )


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


def _confidence(ranked: list[RankedFile]) -> str:
    if not ranked:
        return "none"
    strongest = ranked[0]
    if strongest.score >= 55 and len(strongest.matched_terms) >= 2:
        return "strong"
    return "weak"


def _semantic_file_scores(matches: list[SemanticChunkMatch]) -> dict[str, tuple[float, list[SemanticChunkMatch]]]:
    grouped: dict[str, list[SemanticChunkMatch]] = {}
    for match in matches:
        if match.similarity <= 0:
            continue
        grouped.setdefault(match.file_path, []).append(match)
    scores: dict[str, tuple[float, list[SemanticChunkMatch]]] = {}
    for path, items in grouped.items():
        ordered = sorted(items, key=lambda item: item.similarity, reverse=True)
        top = ordered[:3]
        score = top[0].similarity + sum(item.similarity for item in top[1:]) * 0.25
        scores[path] = (score, top)
    return scores


def _blend_semantic_matches(ranked: list[RankedFile], scan: ScanResult, matches: list[SemanticChunkMatch]) -> list[RankedFile]:
    if not matches:
        return ranked
    by_path = {item.file.relative_path: item for item in ranked}
    files_by_path = {file.relative_path: file for file in scan.scanned_files if not has_ignored_path_part(file.relative_path)}
    semantic_scores = _semantic_file_scores(matches)
    for path, (semantic_score, file_matches) in semantic_scores.items():
        if semantic_score < 0.12:
            continue
        top = file_matches[0]
        boost = max(8, round(semantic_score * 80))
        reason = f"semantic chunk matched task ({top.kind} {top.symbol or top.start_line}, similarity {top.similarity:.2f})"
        existing = by_path.get(path)
        if existing:
            existing.score += boost
            existing.reasons = list(dict.fromkeys([*existing.reasons, reason]))
            existing.matched_terms = list(dict.fromkeys([*existing.matched_terms, "semantic"]))
            continue
        file = files_by_path.get(path)
        if file is None:
            continue
        ranked_file = RankedFile(
            file=file,
            score=boost,
            reasons=[reason],
            matched_terms=["semantic"],
        )
        ranked.append(ranked_file)
        by_path[path] = ranked_file
    return _deduplicate(ranked)


def rank_scanned_files(
    scan: ScanResult,
    task_description: str,
    log_summary: LogSummary | None = None,
    semantic_matches: list[SemanticChunkMatch] | None = None,
) -> TaskResult:
    repo_path = Path(scan.repo_path)
    repo_config = load_repo_config(repo_path)
    analysis = analyze_task(task_description, repo_config.aliases)
    candidates = [file for file in scan.scanned_files if not has_ignored_path_part(file.relative_path)]
    ranked = _deduplicate([
        item for file in candidates
        if (item := _score_file(file, analysis, repo_config, log_summary))
    ])
    ranked = _blend_semantic_matches(ranked, scan, semantic_matches or [])
    ranked_paths = {item.file.relative_path for item in ranked}
    context_only_files = [
        file.relative_path for file in candidates
        if analysis.intent == "access_control"
        and file.relative_path not in ranked_paths
        and _is_rbac_context_only(file)
    ]
    primary = [
        item for item in ranked
        if item.score >= 18 and item.file.category == "source"
        and not (_tokens(item.file.relative_path) & SUPPORT_TERMS)
        and not (analysis.intent == "access_control" and any(reason.startswith("context-only match:") for reason in item.reasons))
    ][:5]
    primary_paths = {item.file.relative_path for item in primary}
    supporting = [item for item in ranked if item.file.relative_path not in primary_paths and item.score >= 12][:5]
    result = TaskResult(
        task_description=task_description,
        keywords=analysis.keywords,
        ranked_files=(primary + supporting)[:8],
        intent=analysis.intent,
        primary_files=primary,
        supporting_files=supporting,
        glossary_terms=analysis.glossary_terms,
        confidence=_confidence(ranked),
        context_only_files=context_only_files,
        semantic_chunks=semantic_matches or [],
    )
    write_json(ensure_cache_dir(repo_path) / "task.json", result.to_dict())
    return result
