"""Deterministic task keyword extraction and intent classification."""

from __future__ import annotations

import re
from dataclasses import dataclass

from . import config


INTENT_LABELS = {
    "ml_model_debug": "ML model debugging",
    "frontend_feature": "Frontend feature",
    "backend_feature": "Backend feature",
    "general_code_task": "General code task",
}

INTENT_TERMS = {
    "ml_model_debug": {"cnn", "model", "train", "inference", "accuracy", "cuda", "torch", "tensorflow", "predict"},
    "frontend_feature": {"ui", "button", "page", "component", "frontend", "style"},
    "backend_feature": {"api", "endpoint", "route", "backend", "server", "database"},
}


@dataclass(frozen=True)
class TaskAnalysis:
    description: str
    keywords: list[str]
    intent: str

    @property
    def intent_label(self) -> str:
        return INTENT_LABELS[self.intent]


def extract_keywords(description: str) -> list[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9_]+", description.lower())
    keywords: set[str] = set()
    for word in words:
        if word in config.STOP_WORDS or len(word) < 2:
            continue
        keywords.update(config.KEYWORD_VARIANTS.get(word, {word}))
    return sorted(keywords)


def analyze_task(description: str) -> TaskAnalysis:
    words = set(re.findall(r"[a-zA-Z][a-zA-Z0-9_]+", description.lower()))
    intent = "general_code_task"
    for candidate in ("ml_model_debug", "frontend_feature", "backend_feature"):
        if words & INTENT_TERMS[candidate]:
            intent = candidate
            break
    return TaskAnalysis(description=description, keywords=extract_keywords(description), intent=intent)
