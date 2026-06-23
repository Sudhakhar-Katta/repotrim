"""Rule-based context review and quality scoring."""

from __future__ import annotations

from pathlib import Path

from ..models import RankedFile
from ..reporter import INTERPRETATIONS
from ..scanner import read_text_safely
from ..task_analyzer import INTENT_LABELS
from .state import AgentState


CATEGORY_TERMS = {
    "test": {"test", "tests", "spec", "fixture"},
    "ui": {"component", "page", "view", "screen", "ui", "tsx", "jsx", "vue", "svelte", "css", "scss"},
    "backend": {"api", "route", "routes", "controller", "service", "server", "middleware", "handler"},
    "auth": {"auth", "role", "roles", "permission", "permissions", "rbac", "guard", "admin", "policy", "user"},
    "route": {"route", "routes", "router", "routing", "page"},
    "model": {"model", "schema", "entity", "repository", "migration", "prisma"},
    "config": {"config", "settings", "env", "toml", "yaml", "json", "dockerfile"},
    "docs": {"readme", "docs", "md"},
}

REQUIRED_BY_INTENT = {
    "access_control": {"auth", "route", "test"},
    "authentication": {"auth", "route", "test"},
    "frontend_feature": {"ui", "route", "test"},
    "backend_feature": {"backend", "model", "test"},
    "database_feature": {"model", "config", "test"},
    "config_deployment": {"config", "test"},
    "testing_task": {"test"},
}

RBAC_REQUIRED_CATEGORIES = {
    "rbac_rules",
    "auth_role_source",
    "middleware_routes",
    "navigation_visibility",
    "protected_pages",
    "backend_enforcement",
    "tests",
}


def _path_tokens(path: str) -> set[str]:
    value = path.replace("\\", "/").lower()
    name = Path(value).name
    raw = value.replace(".", "/").replace("-", "/").replace("_", "/")
    return {part for part in raw.split("/") if part} | {name}


def covered_categories(paths: list[str]) -> set[str]:
    covered: set[str] = set()
    for path in paths:
        tokens = _path_tokens(path)
        for category, terms in CATEGORY_TERMS.items():
            if tokens & terms or any(term in path.lower() for term in terms):
                covered.add(category)
    return covered


def _selected_text(state: AgentState, ranked_files: list[RankedFile]) -> dict[str, str]:
    text_by_path: dict[str, str] = {}
    for item in ranked_files:
        relative = item.file.relative_path
        text = read_text_safely(Path(item.file.path))
        if text is not None:
            text_by_path[relative] = text
    repo_path = Path(state.repo_path)
    for relative in state.selected_files:
        if relative in text_by_path:
            continue
        text = read_text_safely(repo_path / relative)
        if text is not None:
            text_by_path[relative] = text
    return text_by_path


def _is_rbac_review_task(state: AgentState) -> bool:
    task = state.task.lower()
    task_terms = {"rbac", "admin", "access", "access-control", "permission", "permissions", "role", "roles", "protected"}
    return state.task_type == "access_control" or any(term in task for term in task_terms)


def _has_app_page_path(path: str) -> bool:
    parts = path.split("/")
    return (
        len(parts) >= 3
        and parts[0] == "app"
        and parts[1] == "app"
        and parts[-1] in {"page.tsx", "page.jsx", "layout.tsx", "layout.jsx"}
    )


def _detect_rbac_categories(state: AgentState, ranked_files: list[RankedFile]) -> set[str]:
    categories: set[str] = set()
    text_by_path = _selected_text(state, ranked_files)
    selected = state.selected_files or [item.file.relative_path for item in ranked_files]

    for relative in selected:
        path = relative.replace("\\", "/").lower()
        name = Path(path).name
        text = text_by_path.get(relative, "")
        combined = f"{path}\n{text}".lower()

        if name.endswith((".test.ts", ".test.tsx", ".spec.ts", ".spec.tsx", "_test.py")) or "/tests/" in f"/{path}" or "/__tests__/" in f"/{path}":
            categories.add("tests")
        if path.endswith("lib/rbac.ts") or "rbac" in path or "permission" in path or "role" in path or "canaccess" in combined or "policy" in combined:
            categories.add("rbac_rules")
        if "lib/auth" in path or "/auth/" in path or "session" in combined or "user.role" in combined or "getserversession" in combined:
            categories.add("auth_role_source")
        if name == "middleware.ts" or "middleware" in path or "protected route" in combined or "canaccesspath" in combined:
            categories.add("middleware_routes")
        if "sidebar" in path or "navigation" in path or "/nav" in path or "/menu" in path or "menuitem" in combined:
            categories.add("navigation_visibility")
        if _has_app_page_path(path) or (name in {"page.tsx", "page.jsx", "layout.tsx", "layout.jsx"} and any(term in path for term in ("/admin", "/dashboard", "/settings", "/tools"))):
            categories.add("protected_pages")
        is_server_surface = path.startswith("actions/") or "/actions/" in path or path.startswith("app/api/") or "/app/api/" in path
        calls_server_guard = "requiretoolaccess" in combined or "canaccesspath" in combined or "canaccesstool" in combined
        if is_server_surface or "use server" in combined or (calls_server_guard and name not in {"auth.ts", "rbac.ts"}):
            categories.add("backend_enforcement")

    return categories


def _apply_rbac_review(state: AgentState, warnings: list[str], missing: list[str], score: int, ranked_files: list[RankedFile]) -> tuple[int, list[str], list[str]]:
    categories = _detect_rbac_categories(state, ranked_files)
    missing_categories = RBAC_REQUIRED_CATEGORIES - categories

    if "protected_pages" in missing_categories:
        warnings.append("Actual protected page/screen files may be missing.")
        missing.append("actual protected page/screen file")
        score = min(score, 85)
    if "backend_enforcement" in missing_categories:
        warnings.append("Backend/server action enforcement may be missing.")
        missing.append("backend/server action/API enforcement file")
        score = min(score, 85)
    if "navigation_visibility" in categories and "backend_enforcement" in missing_categories:
        warnings.append("Sidebar/UI hiding exists, but server-side enforcement should also be checked.")
    if "tests" in missing_categories:
        warnings.append("No test file found for this access-control task.")
        missing.append("test or spec file")
        score = min(score, 90)

    ui_categories = categories & {"navigation_visibility", "protected_pages"}
    backend_categories = categories & {"rbac_rules", "auth_role_source", "middleware_routes", "backend_enforcement"}
    if ui_categories and not backend_categories:
        score = min(score, 75)
    if backend_categories and not ui_categories:
        score = min(score, 85)
    if missing_categories:
        score = min(score, 99)
    else:
        score += 15

    return score, warnings, missing


def review_context(state: AgentState, ranked_files: list[RankedFile]) -> AgentState:
    paths = state.selected_files
    covered = covered_categories(paths)
    warnings: list[str] = []
    missing: list[str] = []

    if not ranked_files:
        warnings.append("No strong primary file was found for this task.")
        missing.append("primary task-relevant source file")

    required = REQUIRED_BY_INTENT.get(state.task_type, set())
    for category in sorted(required - covered):
        if category == "test":
            warnings.append("No test file found for this task.")
            missing.append("test or spec file")
        elif category == "auth":
            warnings.append("Auth, role, or permission file may be missing.")
            missing.append("auth/role/permission file")
        elif category == "backend":
            warnings.append("Backend route/controller/service context may be missing.")
            missing.append("backend route/controller/service file")
        elif category == "route":
            warnings.append("Route or navigation context may be missing.")
            missing.append("route/navigation file")
        elif category == "model":
            warnings.append("Model/schema/repository context may be missing.")
            missing.append("model/schema/repository file")
        elif category == "config":
            warnings.append("Configuration context may be missing.")
            missing.append("config file")
        elif category == "ui":
            warnings.append("Frontend component/page context may be missing.")
            missing.append("frontend component/page file")

    if state.task_type in {"access_control", "authentication"}:
        if "ui" in covered and "backend" not in covered:
            warnings.append("Only UI files found; security-sensitive tasks usually need backend checks too.")
        if "backend" not in covered:
            warnings.append("Backend enforcement file may be missing.")

    score = 0
    if state.interpretation:
        score += 30
    if ranked_files:
        score += 25
    if state.added_related_files:
        score += 15
    if "test" in covered:
        score += 10
    if state.estimated_tokens <= state.token_budget:
        score += 10
    if required and len(required & covered) >= max(1, len(required) - 1):
        score += 10
    score -= 10 * len(warnings)

    if _is_rbac_review_task(state):
        score, warnings, missing = _apply_rbac_review(state, warnings, missing, score, ranked_files)

    state.missing_context = list(dict.fromkeys(missing))
    state.warnings = list(dict.fromkeys(warnings))
    state.quality_score = max(0, min(100, score))
    return state


def task_interpretation(intent: str) -> str:
    label = INTENT_LABELS.get(intent, "General code task")
    detail = INTERPRETATIONS.get(intent, INTERPRETATIONS["general_code_task"])
    return f"{label}. {detail}"
