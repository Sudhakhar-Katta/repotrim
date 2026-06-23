"""Token savings calculations and reporting."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from .models import ScanResult
from .tokenizer import estimate_tokens
from .utils import ensure_cache_dir, percent_savings, write_json


@dataclass(frozen=True)
class TokenSavings:
    repo_path: str
    packet_path: str
    files_counted: int
    full_repo_tokens: int
    context_packet_tokens: int
    tokens_saved: int
    reduction_percentage: float
    compression_ratio: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def default_packet_path(repo_path: Path) -> Path:
    return repo_path / ".repotrim" / "context_packet.md"


def calculate_token_savings(scan: ScanResult, packet_path: Path) -> TokenSavings:
    packet_path = packet_path.resolve()
    if not packet_path.exists():
        raise FileNotFoundError(f'Context packet not found: {packet_path}. Run: repotrim task "..." first.')

    packet_tokens = estimate_tokens(packet_path.read_text(encoding="utf-8"))
    full_tokens = scan.total_tokens
    tokens_saved = max(0, full_tokens - packet_tokens)
    compression_ratio = (full_tokens / packet_tokens) if packet_tokens > 0 else 0.0

    return TokenSavings(
        repo_path=scan.repo_path,
        packet_path=str(packet_path),
        files_counted=scan.files_scanned,
        full_repo_tokens=full_tokens,
        context_packet_tokens=packet_tokens,
        tokens_saved=tokens_saved,
        reduction_percentage=percent_savings(packet_tokens, full_tokens),
        compression_ratio=compression_ratio,
    )


def save_token_savings(repo_path: Path, savings: TokenSavings) -> Path:
    output_path = ensure_cache_dir(repo_path) / "token_savings.json"
    write_json(output_path, savings.to_dict())
    return output_path


def render_token_savings_report(savings: TokenSavings) -> str:
    return "\n".join(
        [
            "RepoTrim Token Savings",
            "----------------------",
            f"Files counted:              {savings.files_counted:,}",
            f"Full repo estimate:         {savings.full_repo_tokens:,} tokens",
            f"Context packet estimate:    {savings.context_packet_tokens:,} tokens",
            f"Tokens saved:               {savings.tokens_saved:,} tokens",
            f"Reduction:                  {savings.reduction_percentage:.1f}%",
            f"Compression ratio:          {savings.compression_ratio:.1f}x smaller",
        ]
    )
