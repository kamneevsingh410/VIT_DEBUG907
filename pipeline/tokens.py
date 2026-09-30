from __future__ import annotations

import re

WORD_RE = re.compile(r"[A-Za-z_$][\w$]*|\d+")
SUBWORD_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z]+|[a-z]+|\d+")

SCAFFOLDING = {
    "a", "about", "an", "and", "any", "are", "as", "at", "be", "been", "by",
    "can", "could", "did", "do", "does", "doing", "done", "ever", "for",
    "from", "give", "given", "gives", "how", "i", "if", "in", "into", "is",
    "it", "its", "just", "me", "my", "of", "on", "or", "our", "out", "please",
    "should", "so", "some", "somebody", "someone", "that", "the", "their",
    "them", "then", "there", "these", "they", "this", "those", "to", "up",
    "want", "was", "we", "were", "what", "when", "where", "which", "while",
    "who", "why", "will", "with", "would", "you", "your",
}

LOW_SIGNAL = {"code", "function", "method", "program", "snippet", "solution", "way"}


def split_identifier(token: str) -> list[str]:
    return [p.lower() for p in SUBWORD_RE.findall(token) if p]


def tokenize(text: str, *, split_identifiers: bool = True) -> list[str]:
    out: list[str] = []
    for raw in WORD_RE.findall(text or ""):
        lowered = raw.lower()
        out.append(lowered)
        if split_identifiers:
            parts = split_identifier(raw)
            if len(parts) > 1 or parts and parts[0] != lowered:
                out.extend(parts)
    return [t for t in out if t]


def content_tokens(text: str, *, split_identifiers: bool = True,
                   strip_scaffolding: bool = True) -> list[str]:
    toks = tokenize(text, split_identifiers=split_identifiers)
    if not strip_scaffolding:
        return toks
    kept = [t for t in toks if t not in SCAFFOLDING and t not in LOW_SIGNAL and len(t) > 1]
    return kept or toks


def bigrams(tokens: list[str]) -> list[str]:
    return [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]


def stable_hash(text: str) -> int:
    value = 2166136261
    for char in text:
        value ^= ord(char)
        value = (value * 16777619) & 0xFFFFFFFF
    return value
