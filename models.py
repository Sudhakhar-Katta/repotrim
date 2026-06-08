"""Dataclasses used by RepoTrim."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class FileInfo:
    path: str
    relative_path: str
    extension: str
    language: str
    size_bytes: int
    char_count: int
    estimated_tokens: int
    first_lines: list[str]
    symbols: list[str]
    modified_time: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FileInfo":
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class IgnoredPath:
    path: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass
class ScanResult:
    repo_path: str
    created_at: str
    total_files_scanned: int
    total_files_ignored: int
    total_estimated_tokens: int
    files: list[FileInfo] = field(default_factory=list)
    ignored: list[IgnoredPath] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScanResult":
        files = [FileInfo.from_dict(item) for item in data.get("files", [])]
        ignored = [IgnoredPath(**item) for item in data.get("ignored", [])]
        return cls(
            repo_path=data["repo_path"],
            created_at=data["created_at"],
            total_files_scanned=data["total_files_scanned"],
            total_files_ignored=data["total_files_ignored"],
            total_estimated_tokens=data["total_estimated_tokens"],
            files=files,
            ignored=ignored,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo_path": self.repo_path,
            "created_at": self.created_at,
            "total_files_scanned": self.total_files_scanned,
            "total_files_ignored": self.total_files_ignored,
            "total_estimated_tokens": self.total_estimated_tokens,
            "files": [file.to_dict() for file in self.files],
            "ignored": [item.to_dict() for item in self.ignored],
        }


@dataclass
class RankedFile:
    file: FileInfo
    score: int
    reasons: list[str]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RankedFile":
        return cls(
            file=FileInfo.from_dict(data["file"]),
            score=data["score"],
            reasons=list(data.get("reasons", [])),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"file": self.file.to_dict(), "score": self.score, "reasons": self.reasons}


@dataclass
class TaskResult:
    task_description: str
    keywords: list[str]
    ranked_files: list[RankedFile]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskResult":
        return cls(
            task_description=data["task_description"],
            keywords=list(data.get("keywords", [])),
            ranked_files=[RankedFile.from_dict(item) for item in data.get("ranked_files", [])],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_description": self.task_description,
            "keywords": self.keywords,
            "ranked_files": [item.to_dict() for item in self.ranked_files],
        }


@dataclass
class LogSummary:
    original_line_count: int
    kept_line_count: int
    important_lines: list[str]
    mentioned_files: list[str]
    mentioned_symbols: list[str]
    error_codes: list[str]
    noise_removed_notes: list[str]
    truncated: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LogSummary":
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
