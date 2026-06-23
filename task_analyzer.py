"""Deterministic task classification and vocabulary expansion."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import config


INTENT_LABELS = {
    "access_control": "Access control / RBAC",
    "authentication": "Authentication",
    "frontend_feature": "Frontend UI feature",
    "backend_feature": "Backend API feature",
    "database_feature": "Database/schema feature",
    "travel_planning": "Travel/departure/location-planning feature",
    "ml_model_debug": "ML/debugging feature",
    "config_deployment": "Config/deployment task",
    "testing_task": "Testing task",
    "general_code_task": "General code task",
}

INTENT_TERMS = {
    "access_control": {"rbac", "access", "accessing", "access control", "role", "roles", "permission", "permissions", "authorize", "authorization", "guard", "guards", "route guard", "protected route", "admin", "claims", "policy", "policies"},
    "authentication": {"auth", "authentication", "login", "signin", "sign in", "logout", "session", "token", "oauth", "jwt", "password"},
    "travel_planning": {"leave", "departure", "depart", "arrive", "arrival", "eta", "route", "routes", "directions", "commute", "traffic", "transit", "duration", "location", "origin", "destination", "home", "maps", "geocode"},
    "database_feature": {"database", "schema", "migration", "table", "column", "query", "sql", "entity", "repository"},
    "testing_task": {"test", "tests", "testing", "spec", "coverage", "fixture", "mock", "assert"},
    "config_deployment": {"config", "configuration", "deploy", "deployment", "docker", "kubernetes", "environment", "pipeline", "ci", "cd"},
    "ml_model_debug": {"cnn", "model", "train", "training", "inference", "accuracy", "cuda", "torch", "tensorflow", "predict", "debug", "failure"},
    "frontend_feature": {"ui", "button", "page", "component", "frontend", "style", "form", "modal", "view"},
    "backend_feature": {"api", "endpoint", "backend", "server", "controller", "service", "request", "response"},
}

INTENT_ORDER = (
    "access_control", "authentication", "travel_planning", "database_feature",
    "testing_task", "config_deployment", "ml_model_debug", "frontend_feature", "backend_feature",
)

TRAVEL_STRONG_TERMS = {"leave", "departure", "depart", "arrive", "arrival", "eta", "directions", "commute", "traffic", "transit", "origin", "destination", "maps", "geocode"}


@dataclass(frozen=True)
class TaskAnalysis:
    description: str
    keywords: list[str]
    intent: str
    glossary_terms: list[str] = field(default_factory=list)

    @property
    def intent_label(self) -> str:
        return INTENT_LABELS[self.intent]


def _normalize(value: str) -> str:
    return " ".join(re.findall(r"[a-zA-Z0-9]+", value.lower()))


def extract_keywords(description: str) -> list[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9_]+", description.lower())
    keywords: set[str] = set()
    for word in words:
        if word in config.STOP_WORDS or len(word) < 2:
            continue
        keywords.update(config.KEYWORD_VARIANTS.get(word, {word}))
    return sorted(keywords)


def expand_aliases(description: str, aliases: dict[str, list[str]] | None) -> list[str]:
    if not aliases:
        return []
    normalized = _normalize(description)
    expanded: set[str] = set()
    for key, values in aliases.items():
        normalized_key = _normalize(key)
        if normalized_key and re.search(rf"\b{re.escape(normalized_key)}\b", normalized):
            expanded.update(value for value in values if value)
    return sorted(expanded)


def analyze_task(description: str, aliases: dict[str, list[str]] | None = None) -> TaskAnalysis:
    normalized = _normalize(description)
    glossary_terms = expand_aliases(description, aliases)
    classification_text = " ".join([normalized, *glossary_terms])
    intent = "general_code_task"
    for candidate in INTENT_ORDER:
        matched = {term for term in INTENT_TERMS[candidate] if re.search(rf"\b{re.escape(term)}\b", classification_text)}
        if candidate == "travel_planning" and matched and not (matched & TRAVEL_STRONG_TERMS) and len(matched) < 2:
            continue
        if matched:
            intent = candidate
            break
    keywords = set(extract_keywords(description))
    keywords.update(glossary_terms)
    return TaskAnalysis(
        description=description,
        keywords=sorted(keywords),
        intent=intent,
        glossary_terms=glossary_terms,
    )
