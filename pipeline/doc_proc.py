from __future__ import annotations

import re
from dataclasses import dataclass, field, replace

from .parse import ParsedSnippet, parse
from .tokens import split_identifier

SNIPPET_CATEGORIES = ("algorithm", "io", "utility", "data_structure", "class_method")

ALGO_MARKERS = {
    "sort", "sorted", "search", "binary", "dfs", "bfs", "dijkstra", "dp",
    "memo", "recursion", "recursive", "permutation", "combination", "gcd",
    "prime", "fibonacci", "matrix", "graph", "heap", "min", "max", "sum",
    "count", "longest", "shortest", "palindrome", "subsequence",
}
IO_MARKERS = {
    "input", "stdin", "stdout", "print", "read", "write", "open", "file",
    "readline", "scanf", "console", "log", "fetch", "request", "response",
    "send", "receive", "socket", "http",
}
DS_MARKERS = {
    "node", "tree", "list", "linkedlist", "stack", "queue", "deque", "heap",
    "map", "dict", "set", "array", "buffer", "table", "trie", "graph",
}


@dataclass(frozen=True)
class DocConfig:

    use_name: bool = True
    use_split_name: bool = True
    use_signature: bool = True
    use_docstring: bool = True
    use_calls: bool = True
    use_strings: bool = True
    use_control: bool = True
    use_imports: bool = True
    use_raw_code: bool = True
    categorise: bool = True

    @classmethod
    def baseline(cls) -> "DocConfig":
        return cls(use_name=False, use_split_name=False, use_signature=False,
                   use_docstring=False, use_calls=False, use_strings=False,
                   use_control=False, use_imports=False, use_raw_code=True,
                   categorise=False)

    def without(self, component: str) -> "DocConfig":
        return replace(self, **{component: False})


@dataclass(frozen=True)
class EnrichedDoc:
    text: str
    category: str
    names: list[str]
    identifiers: list[str]
    language: str
    raw_length: int
    views: dict[str, str] = field(default_factory=dict)


VIEW_NAMES = ("nl", "code", "full", "raw")
DEFAULT_VIEWS = ("nl", "code")


def _nl_view(parsed: ParsedSnippet, category: str, cfg: "DocConfig") -> str:
    bits: list[str] = []
    if cfg.use_name and parsed.names:
        bits.append(" ".join(parsed.names))
    if cfg.use_split_name:
        name_words: list[str] = []
        for name in parsed.names + parsed.class_names:
            name_words.extend(split_identifier(name))
        if name_words:
            bits.append(" ".join(dict.fromkeys(name_words)))
    if cfg.use_signature and parsed.signatures:
        bits.append(" ".join(parsed.signatures[:4]))
    if cfg.use_docstring and parsed.docstrings:
        bits.append(" ".join(parsed.docstrings[:6]))
    if cfg.use_control and parsed.control:
        bits.append(", ".join(parsed.control))
    if cfg.use_calls:
        called_words: list[str] = []
        for name in list(dict.fromkeys(parsed.calls))[:25]:
            called_words.extend(split_identifier(name))
        if called_words:
            bits.append("uses " + " ".join(dict.fromkeys(called_words)))
    if cfg.use_strings and parsed.strings:
        bits.append(" ".join(parsed.strings[:10]))
    if cfg.categorise:
        bits.append(f"a {category.replace('_', ' ')} routine")
    return ". ".join(b for b in bits if b and b.strip())


def _code_view(parsed: ParsedSnippet, source: str, cfg: "DocConfig") -> str:
    head = " ".join(parsed.signatures[:4]) if cfg.use_signature else ""
    body = source if cfg.use_raw_code else ""
    joined = f"{head}\n{body}".strip()
    return joined or source


def build_views(source: str, doc: "EnrichedDoc",
                names: tuple[str, ...] = DEFAULT_VIEWS,
                language: str | None = None,
                config: "DocConfig | None" = None) -> dict[str, str]:
    cfg = config or DocConfig()
    parsed = parse(source, language)
    built: dict[str, str] = {}
    for view in names:
        if view == "nl":
            built[view] = _nl_view(parsed, doc.category, cfg)
        elif view == "code":
            built[view] = _code_view(parsed, source, cfg)
        elif view == "full":
            built[view] = doc.text
        elif view == "raw":
            built[view] = source.strip()
        else:
            raise ValueError(f"unknown view {view!r}; expected one of {VIEW_NAMES}")
    return {k: v for k, v in built.items() if v and v.strip()}


def _tokens_of(names: list[str]) -> set[str]:
    out: set[str] = set()
    for name in names:
        out.add(name.lower())
        out.update(split_identifier(name))
    return out


def categorise(parsed: ParsedSnippet, source: str) -> str:
    call_tokens = _tokens_of(parsed.calls)
    name_tokens = _tokens_of(parsed.names + parsed.class_names)

    if parsed.class_names:
        if _tokens_of(parsed.class_names) & DS_MARKERS:
            return "data_structure"
        return "class_method"

    io_hits = len(call_tokens & IO_MARKERS)
    algo_hits = len(call_tokens & ALGO_MARKERS) + len(name_tokens & ALGO_MARKERS)
    if io_hits and io_hits >= algo_hits:
        return "io"
    if algo_hits:
        return "algorithm"
    if (call_tokens | name_tokens) & DS_MARKERS:
        return "data_structure"
    return "utility"


def enrich(source: str, config: DocConfig | None = None,
           language: str | None = None,
           views: tuple[str, ...] | None = None) -> EnrichedDoc:
    cfg = config or DocConfig()
    parsed = parse(source, language)

    parts: list[str] = []

    if cfg.use_name and parsed.names:
        parts.append(" ".join(parsed.names))
    if cfg.use_split_name:
        split: list[str] = []
        for name in parsed.names + parsed.class_names:
            split.extend(split_identifier(name))
        if split:
            parts.append(" ".join(split))
    if cfg.use_signature and parsed.signatures:
        parts.append(" ".join(parsed.signatures))
    if cfg.use_docstring and parsed.docstrings:
        parts.append(" ".join(parsed.docstrings))
    if cfg.use_calls and parsed.calls:
        called = list(dict.fromkeys(parsed.calls))
        split_calls: list[str] = []
        for name in called:
            split_calls.extend(split_identifier(name))
        parts.append(" ".join(called + split_calls))
    if cfg.use_strings and parsed.strings:
        parts.append(" ".join(parsed.strings[:40]))
    if cfg.use_control and parsed.control:
        parts.append(", ".join(parsed.control))
    if cfg.use_imports and parsed.imports:
        parts.append(" ".join(parsed.imports))
    if cfg.use_raw_code:
        parts.append(source)

    text = "\n".join(p for p in parts if p and p.strip())
    category = categorise(parsed, source) if cfg.categorise else "utility"

    identifiers = list(dict.fromkeys(parsed.names + parsed.class_names + parsed.calls))

    doc = EnrichedDoc(
        text=text or source,
        category=category,
        names=parsed.names,
        identifiers=identifiers,
        language=parsed.language,
        raw_length=len(source),
    )
    if views:
        return replace(doc, views=build_views(source, doc, tuple(views),
                                              language, cfg))
    return doc
