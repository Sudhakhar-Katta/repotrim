from pathlib import Path
import json
import subprocess

from typer.testing import CliRunner

from repotrim.ignore_rules import ignore_reason
from repotrim.models import FileInfo, ScanResult
from repotrim.main import app
from repotrim.relevance_ranker import rank_scanned_files
from repotrim.reporter import render_task_report
from repotrim.scanner import scan_repo
from repotrim.semantic import build_semantic_index, chunk_file
from repotrim.task_analyzer import analyze_task


def _make_ml_repo(root: Path) -> None:
    (root / "cnn_detector.py").write_text(
        "import torch\nclass CNNDetector:\n    def predict(self, image):\n        self.model.eval()\n",
        encoding="utf-8",
    )
    (root / "ensemble.py").write_text("def combine(cnn_output, other_output):\n    return cnn_output\n", encoding="utf-8")
    (root / "requirements.txt").write_text("torch==2.4.0\nopencv-python\n", encoding="utf-8")
    duplicate = root / ".claude" / "worktrees" / "fix-cnn-cuda"
    duplicate.mkdir(parents=True)
    (duplicate / "cnn_detector.py").write_text("# stale duplicate\n", encoding="utf-8")


def test_claude_worktree_path_is_ignored(tmp_path: Path) -> None:
    path = tmp_path / ".claude" / "worktrees" / "fix-cnn-cuda" / "cnn_detector.py"
    path.parent.mkdir(parents=True)
    path.write_text("duplicate", encoding="utf-8")

    assert ignore_reason(path, tmp_path) == 'inside ignored directory ".claude"'
    scan = scan_repo(tmp_path)
    assert path.name not in [Path(file.path).name for file in scan.scanned_files]
    assert any(item.relative_path == ".claude" for item in scan.ignored_files)


def test_pytest_generated_directories_are_ignored(tmp_path: Path) -> None:
    generated = tmp_path / ".pytest-rbac-final" / "fixture" / "permissions.py"
    generated.parent.mkdir(parents=True)
    generated.write_text("ROLES = {'admin': ['tools:read']}\n", encoding="utf-8")
    source = tmp_path / "src" / "permissions.py"
    source.parent.mkdir()
    source.write_text("ROLES = {'admin': ['tools:read']}\n", encoding="utf-8")

    scan = scan_repo(tmp_path)
    scanned = {file.relative_path for file in scan.scanned_files}

    assert "src/permissions.py" in scanned
    assert not any(path.startswith(".pytest-rbac-final/") for path in scanned)
    assert any(item.relative_path == ".pytest-rbac-final" for item in scan.ignored_files)


def test_source_directories_are_not_ignored(tmp_path: Path) -> None:
    for folder in ("src", "backend", "frontend", "tests"):
        path = tmp_path / folder / f"{folder}_module.py"
        path.parent.mkdir()
        path.write_text("value = 1\n", encoding="utf-8")

    scan = scan_repo(tmp_path)
    scanned = {file.relative_path for file in scan.scanned_files}
    assert scanned == {
        "src/src_module.py",
        "backend/backend_module.py",
        "frontend/frontend_module.py",
        "tests/tests_module.py",
    }


def test_ml_task_ranks_canonical_detector_and_builds_report(tmp_path: Path) -> None:
    _make_ml_repo(tmp_path)
    scan = scan_repo(tmp_path)
    task = rank_scanned_files(scan, "fix cnn model")
    ranked_paths = [item.file.relative_path for item in task.ranked_files]

    assert "cnn_detector.py" in ranked_paths
    assert not any(path.startswith(".claude/") for path in ranked_paths)
    assert task.primary_files[0].file.relative_path == "cnn_detector.py"
    supporting_paths = {item.file.relative_path for item in task.supporting_files}
    assert {"ensemble.py", "requirements.txt"} <= supporting_paths

    report = render_task_report(scan, task)
    assert "Task type:\nML/debugging feature" in report
    assert "Primary files:" in report
    assert "Supporting files:" in report
    assert "Suggested fix workflow:" in report
    assert "Check CPU/GPU device handling." in report
    assert ".claude" not in report


def test_ranker_defensively_rejects_ignored_scanned_file(tmp_path: Path) -> None:
    canonical = tmp_path / "cnn_detector.py"
    ignored = tmp_path / ".claude" / "worktrees" / "fix-cnn-cuda" / "cnn_detector.py"
    ignored.parent.mkdir(parents=True)
    canonical.write_text("class CNNDetector:\n    pass\n", encoding="utf-8")
    ignored.write_text("class CNNDetector:\n    pass\n", encoding="utf-8")

    def file_info(path: Path, relative_path: str) -> FileInfo:
        return FileInfo(
            path=str(path), relative_path=relative_path, extension=".py", language="python",
            size_bytes=path.stat().st_size, char_count=20, estimated_tokens=5,
            first_lines=["class CNNDetector:"], symbols=["CNNDetector"], modified_time=0,
        )

    scan = ScanResult(
        repo_path=str(tmp_path), created_at="test", total_files_scanned=2,
        total_files_ignored=0, total_estimated_tokens=10,
        files=[
            file_info(canonical, "cnn_detector.py"),
            file_info(ignored, ".claude/worktrees/fix-cnn-cuda/cnn_detector.py"),
        ],
    )

    task = rank_scanned_files(scan, "fix cnn model")
    assert [item.file.relative_path for item in task.ranked_files] == ["cnn_detector.py"]


def test_task_cli_scans_requested_path_and_excludes_worktree_candidate(tmp_path: Path) -> None:
    _make_ml_repo(tmp_path)
    result = CliRunner().invoke(app, ["task", "fix cnn model", "--path", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "RepoTrim Task Report" in result.output
    assert "cnn_detector.py" in result.output
    primary_section = result.output.split("Primary files:", 1)[1].split("Supporting files:", 1)[0]
    assert ".claude/worktrees" not in primary_section
    assert "✓ context_packet.md written — paste into Claude or Codex to one-shot this." in result.output
    assert "RepoTrim Token Savings" in result.output
    assert "Files counted:" in result.output
    assert "Compression ratio:" in result.output

    packet = tmp_path / ".repotrim" / "context_packet.md"
    assert packet.exists()
    savings = tmp_path / ".repotrim" / "token_savings.json"
    assert savings.exists()
    savings_data = json.loads(savings.read_text(encoding="utf-8"))
    assert savings_data["files_counted"] == 3
    assert savings_data["full_repo_tokens"] > 0
    assert savings_data["context_packet_tokens"] > 0
    assert savings_data["tokens_saved"] >= 0
    text = packet.read_text(encoding="utf-8")
    assert "## Task\n\nfix cnn model" in text
    assert "## Entry points" in text
    assert "## Relevant files with full contents" in text
    assert "### cnn_detector.py" in text
    assert "import torch\nclass CNNDetector" in text
    assert "### ensemble.py" in text
    assert "def combine(cnn_output, other_output):" in text
    assert "Given the context above, implement the following: fix cnn model. Make minimal changes. Return only the modified files with their full contents." in text


def test_savings_cli_reports_and_writes_json_for_default_packet(tmp_path: Path) -> None:
    _make_ml_repo(tmp_path)
    task_result = CliRunner().invoke(app, ["task", "fix cnn model", "--path", str(tmp_path)])
    assert task_result.exit_code == 0, task_result.output

    result = CliRunner().invoke(app, ["savings", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "RepoTrim Token Savings" in result.output
    assert "Files counted:" in result.output
    assert "Full repo estimate:" in result.output
    assert "Context packet estimate:" in result.output
    assert "Tokens saved:" in result.output
    assert "Reduction:" in result.output
    assert "Compression ratio:" in result.output

    savings = json.loads((tmp_path / ".repotrim" / "token_savings.json").read_text(encoding="utf-8"))
    assert savings["files_counted"] == 3
    assert savings["context_packet_tokens"] > 0
    assert savings["compression_ratio"] > 0


def test_savings_cli_accepts_custom_packet(tmp_path: Path) -> None:
    _make_ml_repo(tmp_path)
    custom_packet = tmp_path / "packet.md"
    custom_packet.write_text("# Custom\n\nsmall packet\n", encoding="utf-8")

    result = CliRunner().invoke(app, ["savings", str(tmp_path), "--packet", str(custom_packet)])

    assert result.exit_code == 0, result.output
    savings = json.loads((tmp_path / ".repotrim" / "token_savings.json").read_text(encoding="utf-8"))
    assert savings["packet_path"] == str(custom_packet.resolve())


def test_savings_cli_errors_when_packet_is_missing(tmp_path: Path) -> None:
    _make_ml_repo(tmp_path)

    result = CliRunner().invoke(app, ["savings", str(tmp_path)])

    assert result.exit_code != 0
    assert 'Run: repotrim task "..." first.' in result.output


def test_task_cli_returns_only_canonical_file_for_exact_duplicate_fixture(tmp_path: Path) -> None:
    canonical = tmp_path / "cnn_detector.py"
    duplicate = tmp_path / ".claude" / "worktrees" / "fix-cnn-cuda" / "cnn_detector.py"
    duplicate.parent.mkdir(parents=True)
    canonical.write_text("class CNNDetector:\n    pass\n", encoding="utf-8")
    duplicate.write_text("class CNNDetector:\n    pass\n", encoding="utf-8")

    result = CliRunner().invoke(app, ["task", "fix cnn model", "--path", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert ".claude" not in result.output
    assert "1. cnn_detector.py" in result.output
    assert "Supporting files:\nNo strong matches found." in result.output


def test_rbac_task_discovers_and_ranks_auth_tool_files(tmp_path: Path) -> None:
    files = {
        "backend/auth/permissions.py": "ROLES = {'admin': ['tools:read']}\ndef authorize(role, permission):\n    return permission in ROLES.get(role, [])\n",
        "backend/api/routes/tools.py": "def list_tools(user):\n    return authorize(user.role, 'tools:read')\n",
        "frontend/src/ToolGuard.tsx": "export function ToolGuard({ permissions, children }) { return permissions.includes('tools:read') ? children : null }\n",
        "tests/test_rbac.py": "def test_user_without_permission_cannot_access_tools():\n    assert True\n",
        "backend/unrelated.py": "",
        "backend/api/middleware/rate_limit.py": "",
        "backend/llm/guardrails.py": "",
        "Dockerfile": "FROM python:3.13-slim\n",
        "frontend/src/App.vue": "<template><ToolGuard /></template>\n",
    }
    for relative_path, content in files.items():
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    scan = scan_repo(tmp_path)
    task = rank_scanned_files(scan, "implement a rbac access for accessing the tools")
    ranked_paths = {item.file.relative_path for item in task.ranked_files}

    assert scan.total_tokens > 0
    assert {"Dockerfile", "frontend/src/App.vue"} <= {item.relative_path for item in scan.scanned_files}
    assert analyze_task(task.task_description).intent == "access_control"
    assert analyze_task("add authentication guards").intent == "access_control"
    assert "backend/auth/permissions.py" in ranked_paths
    assert "backend/api/routes/tools.py" in ranked_paths
    assert "frontend/src/ToolGuard.tsx" in ranked_paths
    assert "tests/test_rbac.py" in ranked_paths
    assert "backend/unrelated.py" not in ranked_paths
    assert "backend/api/middleware/rate_limit.py" not in ranked_paths
    assert "backend/llm/guardrails.py" not in ranked_paths

    report = render_task_report(scan, task)
    assert "Task type:\nAccess control / RBAC" in report
    assert "Full repo: 0 tokens" not in report
    assert "Add focused tests for allowed and denied roles." in report


def test_task_defaults_to_containing_git_root(tmp_path: Path, monkeypatch) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    auth_file = tmp_path / "backend" / "auth" / "permissions.py"
    nested = tmp_path / "backend" / "api" / "routes"
    auth_file.parent.mkdir(parents=True)
    nested.mkdir(parents=True)
    auth_file.write_text("def authorize(role, permission):\n    return role == 'admin'\n", encoding="utf-8")
    monkeypatch.chdir(nested)

    result = CliRunner().invoke(app, ["task", "implement rbac access for tools"])

    assert result.exit_code == 0, result.output
    assert "backend/auth/permissions.py" in result.output
    assert "Full repo: 0 tokens" not in result.output


def test_empty_repo_reports_zero_scan_diagnostics(tmp_path: Path) -> None:
    scan = scan_repo(tmp_path)
    task = rank_scanned_files(scan, "implement a feature")
    report = render_task_report(scan, task)

    assert scan.files_discovered == 0
    assert scan.files_scanned == 0
    assert scan.total_tokens == 0
    assert "Files discovered: 0" in report
    assert "Files indexed: 0" in report
    assert "RepoTrim did not index any files" in report


def test_departure_task_classification_and_low_signal_not_found_penalty(tmp_path: Path) -> None:
    files = {
        "src/components/RoutePlanner.tsx": "const origin = home; const destination = work; const eta = route.duration; const departureTime = arrivalTime - duration;",
        "src/components/ResultCard.tsx": "export function ResultCard({ arrivalTime, departureTime, duration }) { return departureTime }",
        "src/pages/NotFound.tsx": "export function NotFound() { return <a>Go home</a> }",
    }
    for relative_path, content in files.items():
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    scan = scan_repo(tmp_path)
    task = rank_scanned_files(scan, "implement the when to leave from home feature")
    scores = {item.file.relative_path: item.score for item in task.ranked_files}

    assert task.intent == "travel_planning"
    assert task.primary_files[0].file.relative_path == "src/components/RoutePlanner.tsx"
    assert "src/components/ResultCard.tsx" in scores
    assert "src/pages/NotFound.tsx" not in scores or scores["src/pages/NotFound.tsx"] < scores["src/components/ResultCard.tsx"]


def test_repo_glossary_expands_task_vocabulary(tmp_path: Path) -> None:
    config_data = {
        "aliases": {
            "tools": ["dashboard", "modules", "tool cards", "features"],
            "rbac": ["roles", "permissions", "access control", "guards", "protected routes"],
            "when to leave": ["departure time", "arrival time", "eta", "route duration", "traffic", "origin", "destination"],
        },
        "lowSignalFiles": ["NotFound", "404", "ErrorPage", "main", "index", "Layout"],
    }
    (tmp_path / ".repotrimrc.json").write_text(json.dumps(config_data), encoding="utf-8")
    dashboard = tmp_path / "src" / "Dashboard.tsx"
    permissions = tmp_path / "backend" / "Permissions.py"
    dashboard.parent.mkdir(parents=True)
    permissions.parent.mkdir(parents=True)
    dashboard.write_text("export const ToolCards = () => permissions.includes('tools:read') ? modules.map(renderTool) : []", encoding="utf-8")
    permissions.write_text("ROLES = {'admin': ['dashboard:read']}", encoding="utf-8")

    scan = scan_repo(tmp_path)
    task = rank_scanned_files(scan, "implement a rbac access for accessing the tools")

    assert {"dashboard", "modules", "tool cards", "roles", "permissions", "protected routes"} <= set(task.glossary_terms)
    assert "src/Dashboard.tsx" in {item.file.relative_path for item in task.ranked_files}
    assert "Repo glossary expansion:" in render_task_report(scan, task)


def test_nonempty_repo_reports_index_and_token_diagnostics(tmp_path: Path) -> None:
    source = tmp_path / "src" / "location.ts"
    source.parent.mkdir()
    source.write_text("export const origin = 'home'; export const destination = 'office';", encoding="utf-8")

    scan = scan_repo(tmp_path)
    task = rank_scanned_files(scan, "calculate route from origin to destination")
    report = render_task_report(scan, task)

    assert scan.files_discovered == 1
    assert scan.files_scanned == 1
    assert scan.total_tokens > 0
    assert scan.extensions_found == {".ts": 1}
    assert "Full repo tokens: 0" not in report


def test_rbac_anchors_outrank_context_only_tool_placeholders(tmp_path: Path) -> None:
    files = {
        "src/app/auth/auth-guard.ts": "export class AuthGuard { canActivate(user) { return user.roles.includes('admin'); } }",
        "src/app/app.routes.ts": "export const routes = [{ path: 'tools', canActivate: [AuthGuard] }];",
        "src/app/dashboard/dashboard.component.ts": "export class DashboardComponent { permission = 'tools:read'; }",
        "src/app/dashboard/dashboard.component.html": "<section>Dashboard modules for employees and users. " + ("Select a module from the dashboard. " * 20) + "</section>",
        "src/app/users/user-role.component.ts": "export class UserRoleComponent { role = 'admin'; }",
        "src/app/tools/tool1/tool1.component.html": "<p>tool1 works</p>",
        "src/app/tools/tool2/tool2.component.html": "<p>tool2 works</p>",
        "src/app/tools/tool3/tool3.component.html": "<p>tool3 works</p>",
    }
    for relative_path, content in files.items():
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    scan = scan_repo(tmp_path)
    task = rank_scanned_files(scan, "implement a rbac access for accessing the tools")
    ranked_paths = [item.file.relative_path for item in task.ranked_files]
    reasons = {item.file.relative_path: item.reasons for item in task.ranked_files}

    assert set(ranked_paths[:2]) == {"src/app/auth/auth-guard.ts", "src/app/app.routes.ts"}
    assert "src/app/app.routes.ts" in ranked_paths
    assert "src/app/dashboard/dashboard.component.ts" in ranked_paths
    assert "src/app/dashboard/dashboard.component.html" in ranked_paths
    assert "src/app/dashboard/dashboard.component.html" not in {item.file.relative_path for item in task.primary_files}
    assert "src/app/users/user-role.component.ts" in ranked_paths
    assert not any("tool1.component.html" in path or "tool2.component.html" in path or "tool3.component.html" in path for path in ranked_paths)
    assert "boosted because auth + guard evidence" in reasons["src/app/auth/auth-guard.ts"]
    assert "boosted because route file matched RBAC task type" in reasons["src/app/app.routes.ts"]


def test_rbac_context_only_reason_is_explainable(tmp_path: Path) -> None:
    tool_file = tmp_path / "src" / "tools" / "catalog.component.ts"
    tool_file.parent.mkdir(parents=True)
    tool_file.write_text("export class ToolCatalogComponent { tools = []; }", encoding="utf-8")

    scan = scan_repo(tmp_path)
    task = rank_scanned_files(scan, "implement rbac access for tools")

    assert task.ranked_files == []
    report = render_task_report(scan, task)
    assert "context-only match: tool/tools or similar project nouns, not enough RBAC evidence" in report


def test_semantic_chunking_splits_supported_languages(tmp_path: Path) -> None:
    files = {
        "auth.py": "class AuthService:\n    def login(self):\n        pass\n\ndef helper():\n    pass\n",
        "guard.ts": "export class AuthGuard {\n  canActivate() { return true }\n}\nexport function routeGuard() { return true }\n",
        "roles.cs": "public class UserRole {\n    public bool CanAccess() { return true; }\n}\n",
    }
    for relative_path, content in files.items():
        path = tmp_path / relative_path
        path.write_text(content, encoding="utf-8")

    scan = scan_repo(tmp_path)
    chunks = [chunk for file in scan.scanned_files for chunk in chunk_file(file)]
    chunk_ids = {chunk.chunk_id for chunk in chunks}

    assert any("auth.py::class::AuthService" in chunk_id for chunk_id in chunk_ids)
    assert any("auth.py::function::helper" in chunk_id for chunk_id in chunk_ids)
    assert any("guard.ts::class::AuthGuard" in chunk_id for chunk_id in chunk_ids)
    assert any("roles.cs::class::UserRole" in chunk_id for chunk_id in chunk_ids)


def test_index_cli_builds_semantic_index_with_fallback_provider(tmp_path: Path) -> None:
    _make_ml_repo(tmp_path)

    result = CliRunner().invoke(app, ["index", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "RepoTrim Semantic Index" in result.output
    assert "Chunks indexed:" in result.output
    index_path = tmp_path / ".repotrim" / "semantic_index.json"
    assert index_path.exists()
    data = json.loads(index_path.read_text(encoding="utf-8"))
    assert data["chunks"]
    assert data["provider"] in {"sentence-transformers", "lexical-fallback"}


def test_semantic_index_reuses_cached_embeddings_by_file_hash(tmp_path: Path) -> None:
    class CountingProvider:
        provider_name = "test-provider"
        model_name = "test-model"

        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def embed(self, texts: list[str]) -> list[list[float]]:
            self.calls.append(texts)
            return [[1.0, 0.0] for _ in texts]

    first = tmp_path / "first.py"
    second = tmp_path / "second.py"
    first.write_text("def first():\n    return 1\n", encoding="utf-8")
    second.write_text("def second():\n    return 2\n", encoding="utf-8")

    provider = CountingProvider()
    scan = scan_repo(tmp_path)
    initial = build_semantic_index(scan, provider)
    assert sum(len(call) for call in provider.calls) == 2

    provider.calls.clear()
    second.write_text("def second():\n    return 3\n", encoding="utf-8")
    updated = build_semantic_index(scan_repo(tmp_path), provider, initial)

    assert sum(len(call) for call in provider.calls) == 1
    assert len(updated.chunks) == 2


def test_task_semantic_blends_matches_and_packet_includes_chunks(tmp_path: Path) -> None:
    files = {
        "src/security/AuthGuard.ts": "export class AuthGuard { canActivate(user) { return user.role === 'admin'; } }",
        "src/security/PermissionService.cs": "public class PermissionService { public bool CanAccess(string role) { return role == \"admin\"; } }",
        "src/routes.ts": "export const routes = [{ path: 'admin', guard: AuthGuard }];",
        "src/pages/Home.tsx": "export function Home() { return <main>Welcome</main> }",
    }
    for relative_path, content in files.items():
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    result = CliRunner().invoke(app, ["task", "restrict admin screens", "--path", str(tmp_path), "--semantic"])

    assert result.exit_code == 0, result.output
    assert "Semantic search:" in result.output
    packet = (tmp_path / ".repotrim" / "context_packet.md").read_text(encoding="utf-8")
    assert "## Top semantic chunks" in packet
    assert "AuthGuard" in packet or "PermissionService" in packet
    task_data = json.loads((tmp_path / ".repotrim" / "task.json").read_text(encoding="utf-8"))
    assert task_data["semantic_chunks"]
