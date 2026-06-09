"""Backward-compatible task ranking API."""

from .relevance_ranker import rank_scanned_files
from .task_analyzer import extract_keywords as normalize_task

rank_files = rank_scanned_files

__all__ = ["normalize_task", "rank_files"]
