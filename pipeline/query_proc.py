from __future__ import annotations

import re
from dataclasses import dataclass, field, replace

from .tokens import bigrams, content_tokens, tokenize


IDENTIFIER_RE = re.compile(r"\b(?:[a-z]+[A-Z]\w*|\w+_\w+|[A-Z]{2,}(?:_\w+)*)\b")
CALL_RE = re.compile(r"\b(\w+)\s*\(")

ALGORITHMIC = {
    "sort", "sorted", "search", "binary", "dynamic", "programming", "greedy",
    "recursive", "recursion", "dfs", "bfs", "graph", "tree", "heap", "queue",
    "stack", "matrix", "permutation", "combination", "subsequence", "substring",
    "palindrome", "fibonacci", "prime", "gcd", "modulo", "maximum", "minimum",
    "sum", "product", "distance", "path", "cycle", "shortest", "longest",
}
STRUCTURAL = {
    "class", "method", "inherit", "inherits", "subclass", "interface",
    "module", "import", "imports", "constructor", "decorator", "wrapper",
    "calls", "called", "caller", "defined", "definition", "signature",
    "returns", "parameter", "argument", "before", "after", "inside",
}
BEHAVIOURAL = {
    "handle", "handles", "handling", "validate", "validates", "check",
    "checks", "parse", "parses", "convert", "converts", "read", "reads",
    "write", "writes", "fetch", "send", "receive", "process", "processes",
    "preprocess", "preprocessed", "format", "formats", "filter", "filters",
    "initialise", "initialize", "connect", "retry", "fail", "failure", "error",
}

QUERY_CATEGORIES = ("identifier_led", "structural", "algorithmic", "behavioural")

EXPANSIONS: dict[str, tuple[str, ...]] = {
    "check": ("validate", "assert", "is", "has", "verify"),
    "checks": ("validate", "assert", "is", "has", "verify"),
    "validate": ("check", "assert", "valid"),
    "make": ("create", "build", "new", "init"),
    "create": ("make", "build", "new", "init"),
    "build": ("create", "make", "construct"),
    "get": ("fetch", "read", "load", "retrieve", "return"),
    "fetch": ("get", "load", "request", "read"),
    "set": ("assign", "update", "write", "put"),
    "remove": ("delete", "drop", "pop", "discard", "clear"),
    "delete": ("remove", "drop", "pop", "clear"),
    "add": ("insert", "append", "push", "put"),
    "find": ("search", "locate", "index", "lookup", "match"),
    "search": ("find", "lookup", "match", "scan"),
    "sort": ("order", "sorted", "rank", "arrange"),
    "count": ("len", "length", "size", "total", "num"),
    "convert": ("cast", "parse", "transform"),
    "parse": ("read", "decode", "convert", "load"),
    "print": ("output", "write", "display", "show"),
    "input": ("stdin", "read", "arg", "argument", "param"),
    "output": ("stdout", "print", "write", "result", "return"),
    "preprocess": ("clean", "normalize", "normalise", "sanitize", "strip", "trim", "prepare"),
    "preprocessed": ("clean", "normalize", "normalise", "sanitize", "strip", "trim", "prepare"),
    "string": ("str", "text", "char", "chars"),
    "array": ("list", "arr", "vector", "sequence"),
    "list": ("array", "arr", "sequence"),
    "dictionary": ("dict", "map", "hash", "table"),
    "error": ("exception", "raise", "throw", "fail", "err"),
    "failure": ("error", "exception", "fail", "raise"),
    "loop": ("for", "while", "iterate", "each"),
    "maximum": ("max", "largest", "highest", "greatest"),
    "minimum": ("min", "smallest", "lowest", "least"),
    "first": ("head", "front", "initial", "start"),
    "last": ("tail", "back", "final", "end"),
    "number": ("int", "num", "digit", "integer"),
    "file": ("path", "open", "read", "write", "stream"),
    "main": ("entry", "run", "start", "init"),
}


BOILERPLATE_RE = re.compile(
    r"-{3,}\s*(?:Input|Output|Examples?|Note|Constraints|Sample)", re.I)


def strip_boilerplate(text: str) -> str:
    match = BOILERPLATE_RE.search(text)
    if not match or match.start() < 40:
        return text
    return text[:match.start()].strip()


@dataclass(frozen=True)
class QueryConfig:

    categorise: bool = True
    split_identifiers: bool = True
    strip_scaffolding: bool = True
    expand_vocabulary: bool = True
    use_bigrams: bool = True
    expansion_weight: int = 1
    strip_boilerplate: bool = False

    @classmethod
    def baseline(cls) -> "QueryConfig":
        return cls(categorise=False, split_identifiers=False,
                   strip_scaffolding=False, expand_vocabulary=False,
                   use_bigrams=False, strip_boilerplate=False)

    def without(self, step: str) -> "QueryConfig":
        return replace(self, **{step: False})


@dataclass(frozen=True)
class ProcessedQuery:
    raw: str
    category: str
    tokens: list[str] = field(default_factory=list)
    expansions: list[str] = field(default_factory=list)
    identifiers: list[str] = field(default_factory=list)
    embedding_text: str = ""

    @property
    def text(self) -> str:
        return " ".join(self.tokens)

    def fts_terms(self) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for tok in self.tokens:
            clean = re.sub(r"[^a-z0-9_]", "", tok)
            if len(clean) > 1 and clean not in seen:
                seen.add(clean)
                out.append(clean)
        return out


def categorise(query: str) -> str:
    if IDENTIFIER_RE.search(query) or CALL_RE.search(query):
        return "identifier_led"
    words = set(tokenize(query, split_identifiers=False))
    scores = {
        "structural": len(words & STRUCTURAL),
        "algorithmic": len(words & ALGORITHMIC),
        "behavioural": len(words & BEHAVIOURAL),
    }
    best = max(scores, key=lambda key: scores[key])
    return best if scores[best] else "behavioural"


def extract_identifiers(query: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for match in IDENTIFIER_RE.finditer(query):
        tok = match.group(0)
        if tok.lower() not in seen:
            seen.add(tok.lower())
            out.append(tok)
    for match in CALL_RE.finditer(query):
        tok = match.group(1)
        if tok and tok.lower() not in seen:
            seen.add(tok.lower())
            out.append(tok)
    return out


def expand(tokens: list[str], weight: int = 1) -> list[str]:
    out: list[str] = []
    seen = set(tokens)
    for tok in tokens:
        for extra in EXPANSIONS.get(tok, ()):
            if extra not in seen:
                seen.add(extra)
                out.extend([extra] * max(1, weight))
    return out


def process(query: str, config: QueryConfig | None = None) -> ProcessedQuery:
    cfg = config or QueryConfig()

    query = strip_boilerplate(query) if cfg.strip_boilerplate else query
    embedding_text = query

    if cfg.strip_scaffolding:
        toks = content_tokens(query, split_identifiers=cfg.split_identifiers,
                              strip_scaffolding=True)
    else:
        toks = tokenize(query, split_identifiers=cfg.split_identifiers)

    expansions = expand(toks, cfg.expansion_weight) if cfg.expand_vocabulary else []
    category = categorise(query) if cfg.categorise else "behavioural"
    idents = extract_identifiers(query)

    combined = list(toks) + expansions
    if cfg.use_bigrams:
        combined += bigrams(toks)

    return ProcessedQuery(raw=query, category=category, tokens=combined,
                          expansions=expansions, identifiers=idents,
                          embedding_text=embedding_text)
