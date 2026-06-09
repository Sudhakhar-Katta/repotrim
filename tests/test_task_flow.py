from pathlib import Path

from typer.testing import CliRunner

from repotrim.ignore_rules import ignore_reason
from repotrim.models import FileInfo, ScanResult
from repotrim.main import app
from repotrim.relevance_ranker import rank_scanned_files
from repotrim.reporter import render_task_report
from repotrim.scanner import scan_repo


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
    assert "Task type:\nML model debugging" in report
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
