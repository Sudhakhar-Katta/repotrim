"""State tracked by the deterministic agent context builder."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AgentState:
    repo_path: str
    task: str
    task_type: str = "general_code_task"
    interpretation: str = ""
    selected_files: list[str] = field(default_factory=list)
    added_related_files: list[str] = field(default_factory=list)
    missing_context: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    quality_score: int = 0
    token_budget: int = 30000
    estimated_tokens: int = 0
    steps_taken: list[str] = field(default_factory=list)

    @classmethod
    def create(cls, repo_path: Path, task: str, token_budget: int) -> "AgentState":
        return cls(repo_path=str(repo_path.resolve()), task=task, token_budget=token_budget)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
