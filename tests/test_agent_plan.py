from pathlib import Path
import json

from typer.testing import CliRunner

from repotrim.agents.reviewer import review_context
from repotrim.agents.state import AgentState
from repotrim.agents.workflow import find_related_files
from repotrim.main import app
from repotrim.models import TaskResult
from repotrim.relevance_ranker import rank_scanned_files
from repotrim.scanner import scan_repo


def _make_auth_repo(root: Path) -> None:
    files = {
        "src/auth/AuthGuard.tsx": "import { PermissionService } from './PermissionService';\nexport function AuthGuard({ user }) { return PermissionService.canAccess(user.role, 'admin'); }\n",
        "src/auth/PermissionService.ts": "export const PermissionService = { canAccess(role, area) { return role === 'admin' && area === 'admin'; } };\n",
        "src/auth/auth-config.ts": "export const AUTH_CONFIG = { deniedRedirect: '/login' };\n",
        "src/auth/UserRole.ts": "export type UserRole = 'admin' | 'member';\n",
        "src/routes.ts": "import { AuthGuard } from './auth/AuthGuard';\nexport const routes = [{ path: 'admin', guard: AuthGuard }];\n",
        "tests/auth.spec.ts": "test('blocks non-admin users from admin routes', () => {});\n",
        ".env": "SECRET=ignored\n",
    }
    for relative_path, content in files.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def _write_files(root: Path, files: dict[str, str]) -> None:
    for relative_path, content in files.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def _make_next_rbac_repo(root: Path, include_pages: bool = False, include_backend: bool = False, include_tests: bool = False) -> None:
    files = {
        "lib/rbac.ts": "export const TOOL_ACCESS = { admin: ['settings'] };\nexport function canAccessTool(role, tool) { return role === 'admin' || TOOL_ACCESS[role]?.includes(tool); }\nexport function canAccessPath(role, path) { return canAccessTool(role, 'settings'); }\n",
        "middleware.ts": "import { canAccessPath } from './lib/rbac';\nexport function middleware(request) { return canAccessPath(request.user.role, request.nextUrl.pathname); }\n",
        "lib/auth.ts": "export async function getSession() { return { user: { role: 'admin' } }; }\n",
        "components/layout/sidebar.tsx": "import { canAccessTool } from '../../lib/rbac';\nexport function Sidebar({ role }) { return canAccessTool(role, 'settings') ? <a href='/app/settings'>Settings</a> : null; }\n",
    }
    if include_pages:
        files["app/app/settings/page.tsx"] = "import { getSession } from '../../../lib/auth';\nexport default async function SettingsPage() { const session = await getSession(); return <main>{session.user.role}</main>; }\n"
        files["app/app/layout.tsx"] = "import { Sidebar } from '../../components/layout/sidebar';\nexport default function AppLayout({ children }) { return <><Sidebar role='admin' />{children}</>; }\n"
    if include_backend:
        files["actions/settings.ts"] = "\"use server\";\nimport { requireToolAccess } from '../lib/auth';\nexport async function updateSettings() { await requireToolAccess('settings'); }\n"
        files["app/api/settings/route.ts"] = "import { canAccessTool } from '../../../lib/rbac';\nexport async function POST(request) { return canAccessTool('admin', 'settings') ? Response.json({ ok: true }) : new Response('', { status: 403 }); }\n"
    if include_tests:
        files["__tests__/rbac.test.ts"] = "import { canAccessTool } from '../lib/rbac';\ntest('blocks non-admin settings access', () => expect(canAccessTool('member', 'settings')).toBe(false));\n"
    _write_files(root, files)


def _review_next_rbac_repo(root: Path) -> AgentState:
    scan = scan_repo(root)
    task = rank_scanned_files(scan, "restrict admin screens")
    state = AgentState.create(root, "restrict admin screens", 30000)
    state.task_type = task.intent
    state.interpretation = "Access control task"
    state.selected_files = [item.file.relative_path for item in task.ranked_files]
    state.estimated_tokens = sum(item.file.estimated_tokens for item in task.ranked_files)
    return review_context(state, task.ranked_files)


def test_agent_state_creation(tmp_path: Path) -> None:
    state = AgentState.create(tmp_path, "restrict admin screens", 30000)

    assert state.repo_path == str(tmp_path.resolve())
    assert state.task == "restrict admin screens"
    assert state.token_budget == 30000
    assert state.selected_files == []


def test_reviewer_scores_access_control_context(tmp_path: Path) -> None:
    _make_auth_repo(tmp_path)
    scan = scan_repo(tmp_path)
    task = rank_scanned_files(scan, "restrict admin screens")
    state = AgentState.create(tmp_path, "restrict admin screens", 30000)
    state.task_type = task.intent
    state.interpretation = "Access control task"
    state.selected_files = [item.file.relative_path for item in task.ranked_files]
    state.estimated_tokens = sum(item.file.estimated_tokens for item in task.ranked_files)

    reviewed = review_context(state, task.ranked_files)

    assert reviewed.quality_score > 50
    assert not any("No test file found" in warning for warning in reviewed.warnings)


def test_reviewer_caps_rbac_context_without_protected_pages(tmp_path: Path) -> None:
    _make_next_rbac_repo(tmp_path, include_backend=True, include_tests=True)

    reviewed = _review_next_rbac_repo(tmp_path)

    assert reviewed.quality_score < 100
    assert reviewed.quality_score <= 85
    assert "Actual protected page/screen files may be missing." in reviewed.warnings


def test_reviewer_warns_when_rbac_tests_are_missing(tmp_path: Path) -> None:
    _make_next_rbac_repo(tmp_path, include_pages=True, include_backend=True, include_tests=False)

    reviewed = _review_next_rbac_repo(tmp_path)

    assert "No test file found for this access-control task." in reviewed.warnings
    assert reviewed.quality_score <= 90


def test_reviewer_scores_complete_rbac_context_higher(tmp_path: Path) -> None:
    incomplete = tmp_path / "incomplete"
    complete = tmp_path / "complete"
    _make_next_rbac_repo(incomplete, include_backend=True, include_tests=True)
    _make_next_rbac_repo(complete, include_pages=True, include_backend=True, include_tests=True)

    incomplete_review = _review_next_rbac_repo(incomplete)
    complete_review = _review_next_rbac_repo(complete)

    assert "Actual protected page/screen files may be missing." in incomplete_review.warnings
    assert "Actual protected page/screen files may be missing." not in complete_review.warnings
    assert complete_review.quality_score > incomplete_review.quality_score


def test_admin_screen_ranking_keeps_api_only_file_below_core_rbac_files(tmp_path: Path) -> None:
    _make_next_rbac_repo(tmp_path, include_pages=True, include_backend=True, include_tests=True)
    _write_files(
        tmp_path,
        {
            "app/api/admin/route.ts": "export async function POST() { const admin = 'admin'; const role = 'admin'; const permission = 'settings'; const protectedRoute = true; return Response.json({ admin, role, permission, protectedRoute }); }\n",
        },
    )

    task = rank_scanned_files(scan_repo(tmp_path), "restrict admin screens")
    ranked_paths = [item.file.relative_path for item in task.ranked_files]

    assert ranked_paths.index("app/api/admin/route.ts") > ranked_paths.index("lib/rbac.ts")
    assert ranked_paths.index("app/api/admin/route.ts") > ranked_paths.index("middleware.ts")
    assert ranked_paths.index("app/api/admin/route.ts") > ranked_paths.index("components/layout/sidebar.tsx")


def test_related_file_expansion_finds_routes_tests_and_services(tmp_path: Path) -> None:
    _make_auth_repo(tmp_path)
    scan = scan_repo(tmp_path)
    task = rank_scanned_files(scan, "restrict admin screens")
    seed_task = TaskResult(
        task_description=task.task_description,
        keywords=task.keywords,
        ranked_files=task.ranked_files[:1],
        intent=task.intent,
        primary_files=task.ranked_files[:1],
        supporting_files=[],
    )

    related = find_related_files(scan, seed_task, max_files=25, token_budget=30000)
    related_paths = {item.file.relative_path for item in related}

    assert "src/auth/auth-config.ts" in related_paths
    assert ".env" not in related_paths


def test_agent_plan_cli_writes_context_report_and_json(tmp_path: Path) -> None:
    _make_auth_repo(tmp_path)

    result = CliRunner().invoke(app, ["agent", "plan", "restrict admin screens", "--repo-path", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "RepoTrim Agent Plan Complete" in result.output
    assert "Task Interpretation:" in result.output
    assert "Context Quality:" in result.output
    assert "Selected Files:" in result.output
    assert "Estimated Tokens:" in result.output
    assert ".repotrim/context_packet.md" in result.output
    assert ".repotrim/agent_report.md" in result.output
    assert ".repotrim/agent.json" in result.output

    packet = tmp_path / ".repotrim" / "context_packet.md"
    report = tmp_path / ".repotrim" / "agent_report.md"
    data_path = tmp_path / ".repotrim" / "agent.json"
    assert packet.exists()
    assert report.exists()
    assert data_path.exists()

    report_text = report.read_text(encoding="utf-8")
    assert "# RepoTrim Agent Report" in report_text
    assert "## Suggested Prompt For Coding Agent" in report_text

    data = json.loads(data_path.read_text(encoding="utf-8"))
    assert data["task"] == "restrict admin screens"
    assert data["selected_files"]
    assert data["quality_score"] >= 0
