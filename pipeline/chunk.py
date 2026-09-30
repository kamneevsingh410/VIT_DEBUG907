from __future__ import annotations

from dataclasses import dataclass

from .parse import parse
from .tokens import split_identifier

STRATEGIES = ("truncate", "windows", "summary")


@dataclass(frozen=True)
class ChunkConfig:
    strategy: str = "truncate"
    budget_tokens: int = 512
    window_overlap: float = 0.25

    def __post_init__(self) -> None:
        if self.strategy not in STRATEGIES:
            raise ValueError(f"unknown chunk strategy {self.strategy!r}; "
                             f"expected one of {STRATEGIES}")


def _words(text: str) -> list[str]:
    return text.split()


def truncate(text: str, budget: int) -> list[str]:
    return [" ".join(_words(text)[:budget])]


def windows(text: str, budget: int, overlap: float) -> list[str]:
    words = _words(text)
    if len(words) <= budget:
        return [" ".join(words)]
    step = max(1, int(budget * (1.0 - overlap)))
    out: list[str] = []
    for start in range(0, len(words), step):
        piece = words[start:start + budget]
        if not piece:
            break
        out.append(" ".join(piece))
        if start + budget >= len(words):
            break
    return out


def summarise(text: str, budget: int) -> list[str]:
    parsed = parse(text)
    bits: list[str] = []
    bits += parsed.names
    for name in parsed.names + parsed.class_names:
        bits += split_identifier(name)
    bits += parsed.signatures
    bits += parsed.docstrings[:4]
    bits += list(dict.fromkeys(parsed.calls))[:30]
    bits += parsed.strings[:15]
    bits += parsed.control
    summary = " ".join(b for b in bits if b)
    return [" ".join(_words(summary)[:budget])] or [""]


def chunk(text: str, config: ChunkConfig | None = None) -> list[str]:
    cfg = config or ChunkConfig()
    if cfg.strategy == "windows":
        return windows(text, cfg.budget_tokens, cfg.window_overlap)
    if cfg.strategy == "summary":
        return summarise(text, cfg.budget_tokens)
    return truncate(text, cfg.budget_tokens)
