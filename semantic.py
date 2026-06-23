"""Local semantic chunking, embedding cache, and vector search."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from .models import FileInfo, ScanResult, SemanticChunkMatch
from .scanner import read_text_safely, scan_repo
from .tokenizer import estimate_tokens
from .utils import ensure_cache_dir


DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_FILENAME = "semantic_index.json"


@dataclass(frozen=True)
class CodeChunk:
    chunk_id: str
    file_path: str
    absolute_path: str
    kind: str
    symbol: str
    start_line: int
    end_line: int
    text: str
    tokens: int
    file_hash: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "CodeChunk":
        return cls(
            chunk_id=str(data["chunk_id"]),
            file_path=str(data["file_path"]),
            absolute_path=str(data["absolute_path"]),
            kind=str(data["kind"]),
            symbol=str(data.get("symbol", "")),
            start_line=int(data["start_line"]),
            end_line=int(data["end_line"]),
            text=str(data["text"]),
            tokens=int(data["tokens"]),
            file_hash=str(data["file_hash"]),
        )


@dataclass(frozen=True)
class EmbeddedChunk:
    chunk: CodeChunk
    embedding: list[float]

    def to_dict(self) -> dict[str, object]:
        return {"chunk": self.chunk.to_dict(), "embedding": self.embedding}

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "EmbeddedChunk":
        return cls(chunk=CodeChunk.from_dict(data["chunk"]), embedding=[float(value) for value in data["embedding"]])


@dataclass(frozen=True)
class SemanticIndex:
    repo_path: str
    model_name: str
    provider: str
    embedding_dim: int
    chunks: list[EmbeddedChunk]

    def to_dict(self) -> dict[str, object]:
        return {
            "repo_path": self.repo_path,
            "model_name": self.model_name,
            "provider": self.provider,
            "embedding_dim": self.embedding_dim,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "SemanticIndex":
        return cls(
            repo_path=str(data["repo_path"]),
            model_name=str(data["model_name"]),
            provider=str(data["provider"]),
            embedding_dim=int(data["embedding_dim"]),
            chunks=[EmbeddedChunk.from_dict(item) for item in data.get("chunks", [])],
        )


class EmbeddingProvider(Protocol):
    provider_name: str
    model_name: str

    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


class SentenceTransformersEmbeddingProvider:
    provider_name = "sentence-transformers"

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. Install semantic support with: "
                "py -m pip install -e .[semantic]"
            ) from exc
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return [[float(value) for value in vector] for vector in vectors]


class LexicalEmbeddingProvider:
    """Deterministic local fallback used when sentence-transformers is absent."""

    provider_name = "lexical-fallback"

    def __init__(self, model_name: str = "repotrim-lexical-fallback", dimensions: int = 256) -> None:
        self.model_name = model_name
        self.dimensions = dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [_normalize(_hash_tokens(text, self.dimensions)) for text in texts]


def make_embedding_provider(model_name: str = DEFAULT_MODEL_NAME, allow_fallback: bool = True) -> EmbeddingProvider:
    try:
        return SentenceTransformersEmbeddingProvider(model_name)
    except RuntimeError:
        if not allow_fallback:
            raise
        return LexicalEmbeddingProvider()


def _tokens(text: str) -> list[str]:
    expanded = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    return re.findall(r"[A-Za-z][A-Za-z0-9_]+", expanded.lower())


def _hash_tokens(text: str, dimensions: int) -> list[float]:
    vector = [0.0] * dimensions
    for token in _tokens(text):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[bucket] += sign
    return vector


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return sum(a * b for a, b in zip(left, right))


def file_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _find_blocks(lines: list[str], patterns: list[tuple[str, re.Pattern[str]]]) -> list[tuple[str, str, int, int]]:
    starts: list[tuple[str, str, int]] = []
    for index, line in enumerate(lines):
        for kind, pattern in patterns:
            match = pattern.search(line)
            if match:
                symbol = next((group for group in reversed(match.groups()) if group), "")
                starts.append((kind, symbol, index))
                break
    blocks: list[tuple[str, str, int, int]] = []
    for offset, (kind, symbol, start) in enumerate(starts):
        end = starts[offset + 1][2] if offset + 1 < len(starts) else len(lines)
        if end > start:
            blocks.append((kind, symbol, start, end))
    return blocks


def _patterns_for_language(language: str) -> list[tuple[str, re.Pattern[str]]]:
    if language == "python":
        return [
            ("class", re.compile(r"^\s*class\s+([A-Za-z_][\w]*)")),
            ("function", re.compile(r"^\s*def\s+([A-Za-z_][\w]*)")),
        ]
    if language in {"typescript", "javascript"}:
        return [
            ("class", re.compile(r"^\s*(?:export\s+)?class\s+([A-Za-z_][\w]*)")),
            ("function", re.compile(r"^\s*(?:export\s+)?function\s+([A-Za-z_][\w]*)")),
            ("function", re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_][\w]*)\s*=\s*(?:async\s*)?(?:\([^)]*\)|[A-Za-z_][\w]*)\s*=>")),
        ]
    if language == "csharp":
        return [
            ("class", re.compile(r"\b(?:class|interface)\s+([A-Za-z_][\w]*)")),
            ("method", re.compile(r"\b(?:public|private|protected|internal)\s+(?:static\s+)?[\w<>,\[\]\?]+\s+([A-Za-z_][\w]*)\s*\(")),
        ]
    return []


def chunk_file(file: FileInfo, max_fallback_lines: int = 80) -> list[CodeChunk]:
    text = read_text_safely(Path(file.path))
    if text is None:
        return []
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.splitlines()
    digest = file_hash(normalized)
    chunks: list[CodeChunk] = []
    blocks = _find_blocks(lines, _patterns_for_language(file.language))
    if not blocks:
        blocks = [("file", Path(file.relative_path).name, start, min(len(lines), start + max_fallback_lines)) for start in range(0, len(lines), max_fallback_lines)]
    for index, (kind, symbol, start, end) in enumerate(blocks):
        chunk_text = "\n".join(lines[start:end]).strip()
        if not chunk_text:
            continue
        symbol_part = symbol or f"chunk_{index + 1}"
        chunk_id = f"{file.relative_path}::{kind}::{symbol_part}::{start + 1}-{end}"
        chunks.append(
            CodeChunk(
                chunk_id=chunk_id,
                file_path=file.relative_path,
                absolute_path=file.path,
                kind=kind,
                symbol=symbol,
                start_line=start + 1,
                end_line=end,
                text=chunk_text,
                tokens=estimate_tokens(chunk_text),
                file_hash=digest,
            )
        )
    return chunks


def chunk_scan(scan: ScanResult) -> list[CodeChunk]:
    return [chunk for file in scan.scanned_files for chunk in chunk_file(file)]


def index_path(repo_path: Path) -> Path:
    return ensure_cache_dir(repo_path) / INDEX_FILENAME


def load_semantic_index(repo_path: Path) -> SemanticIndex | None:
    path = index_path(repo_path)
    if not path.exists():
        return None
    return SemanticIndex.from_dict(json.loads(path.read_text(encoding="utf-8")))


def build_semantic_index(
    scan: ScanResult,
    provider: EmbeddingProvider,
    previous: SemanticIndex | None = None,
) -> SemanticIndex:
    previous_by_id = {item.chunk.chunk_id: item for item in (previous.chunks if previous else [])}
    chunks = chunk_scan(scan)
    embedded: list[EmbeddedChunk] = []
    pending: list[CodeChunk] = []
    for chunk in chunks:
        cached = previous_by_id.get(chunk.chunk_id)
        if cached and cached.chunk.file_hash == chunk.file_hash and len(cached.embedding) > 0:
            embedded.append(cached)
        else:
            pending.append(chunk)
    if pending:
        vectors = provider.embed([_embedding_text(chunk) for chunk in pending])
        embedded.extend(EmbeddedChunk(chunk=chunk, embedding=vector) for chunk, vector in zip(pending, vectors))
    dimension = len(embedded[0].embedding) if embedded else 0
    return SemanticIndex(
        repo_path=scan.repo_path,
        model_name=provider.model_name,
        provider=provider.provider_name,
        embedding_dim=dimension,
        chunks=embedded,
    )


def save_semantic_index(repo_path: Path, index: SemanticIndex) -> Path:
    path = index_path(repo_path)
    path.write_text(json.dumps(index.to_dict(), indent=2), encoding="utf-8")
    return path


def index_repo(repo_path: Path, model_name: str = DEFAULT_MODEL_NAME, allow_fallback: bool = True) -> tuple[ScanResult, SemanticIndex, Path]:
    scan = scan_repo(repo_path)
    provider = make_embedding_provider(model_name, allow_fallback=allow_fallback)
    previous = load_semantic_index(Path(scan.repo_path))
    index = build_semantic_index(scan, provider, previous)
    path = save_semantic_index(Path(scan.repo_path), index)
    return scan, index, path


def _embedding_text(chunk: CodeChunk) -> str:
    return "\n".join(
        [
            f"file: {chunk.file_path}",
            f"kind: {chunk.kind}",
            f"symbol: {chunk.symbol}",
            chunk.text,
        ]
    )


def search_semantic_index(index: SemanticIndex, query: str, provider: EmbeddingProvider, limit: int = 12) -> list[SemanticChunkMatch]:
    query_embedding = provider.embed([query])[0]
    matches: list[SemanticChunkMatch] = []
    for item in index.chunks:
        similarity = cosine_similarity(query_embedding, item.embedding)
        matches.append(
            SemanticChunkMatch(
                chunk_id=item.chunk.chunk_id,
                file_path=item.chunk.file_path,
                kind=item.chunk.kind,
                symbol=item.chunk.symbol,
                start_line=item.chunk.start_line,
                end_line=item.chunk.end_line,
                text=item.chunk.text,
                similarity=similarity,
            )
        )
    return sorted(matches, key=lambda match: match.similarity, reverse=True)[:limit]


def semantic_search(
    repo_path: Path,
    query: str,
    model_name: str = DEFAULT_MODEL_NAME,
    limit: int = 12,
    allow_fallback: bool = True,
) -> tuple[ScanResult, SemanticIndex, list[SemanticChunkMatch], Path]:
    scan = scan_repo(repo_path)
    provider = make_embedding_provider(model_name, allow_fallback=allow_fallback)
    previous = load_semantic_index(Path(scan.repo_path))
    if previous and (previous.model_name != provider.model_name or previous.provider != provider.provider_name):
        previous = None
    index = build_semantic_index(scan, provider, previous)
    path = save_semantic_index(Path(scan.repo_path), index)
    matches = search_semantic_index(index, query, provider, limit=limit)
    return scan, index, matches, path
