from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

MAX_WORDS = 14

_X = r"(?P<x>.+?)"
_A = r"(?P<a>.+?)"
_B = r"(?P<b>.+?)"
_ORDER = r"(?P<order>before|after)"
_WHO = r"(?:(?:which|what)\s+(?:files?|functions?|methods?|code|modules?|places?)|who|what)"

PATTERNS: list[tuple[str, str]] = [
    ("order", rf"^{_WHO}\s+(?:calls?|invokes?|runs?)\s+{_A}\s+{_ORDER}\s+(?:calling\s+|invoking\s+)?{_B}$"),
    ("order", rf"^(?:does|do|is|are)\s+{_A}\s+(?:gets?\s+|being\s+)?(?:called|invoked|run)\s+{_ORDER}\s+{_B}$"),
    ("order", rf"^where\s+(?:is|are)\s+{_A}\s+(?:called|invoked)\s+{_ORDER}\s+{_B}$"),
    ("order", rf"^(?:find|show|list)(?:\s+me)?\s+(?:all\s+)?(?:the\s+)?(?:functions?|files?|places?)\s+"
              rf"(?:that|which|where)\s+(?:call|calls|invoke)\s+{_A}\s+{_ORDER}\s+{_B}$"),
    ("usage", rf"^(?:where|how)\s+(?:is|are)\s+{_X}\s+(?:used|referenced|read|accessed|mentioned)(?:\s+anywhere)?$"),
    ("usage", rf"^{_WHO}\s+(?:uses?|references?|reads?|mentions?|accesses?)\s+{_X}$"),
    ("usage", rf"^(?:find|show|list|get|search)(?:\s+me)?\s+(?:all\s+)?(?:the\s+)?"
              rf"(?:usages?|uses|references?|occurrences?|mentions?)\s+(?:of|to|for)\s+{_X}$"),
    ("usage", rf"^where\s+(?:do|does)\s+(?:we|i|you|they|the\s+code)\s+(?:use|reference|read|access)\s+{_X}$"),
    ("usage", rf"^(?:usages?|references?|occurrences?)\s+of\s+{_X}$"),
    ("callers", rf"^{_WHO}\s+(?:calls?|invokes?)\s+{_X}$"),
    ("callers", rf"^where\s+(?:is|are)\s+{_X}\s+(?:called|invoked)(?:\s+from)?$"),
    ("callers", rf"^(?:find|show|list)(?:\s+me)?\s+(?:all\s+)?(?:the\s+)?(?:callers?|call\s+sites?)\s+(?:of|for|to)\s+{_X}$"),
    ("callers", rf"^(?:callers?|call\s+sites?)\s+of\s+{_X}$"),
    ("definition", rf"^where\s+(?:is|are)\s+{_X}\s+(?:defined|implemented|declared|written)$"),
    ("definition", rf"^(?:find|show|go\s+to|open)(?:\s+me)?\s+(?:the\s+)?(?:definition|implementation|declaration|source)\s+of\s+{_X}$"),
    ("definition", rf"^(?:definition|implementation|declaration)\s+of\s+{_X}$"),
    ("definition", rf"^where\s+(?:does|do)\s+{_X}\s+(?:live|come\s+from|get\s+defined)$"),
]
_COMPILED = [(k, re.compile(p, re.IGNORECASE)) for k, p in PATTERNS]


@dataclass
class Route:
    kind: str
    phrases: list[str]
    order: str = "before"


def route(text: str) -> Route | None:
    if "\n" in text.strip():
        return None
    q = text.strip().rstrip("?.! ").strip()
    if not q or len(q.split()) > MAX_WORDS:
        return None
    for kind, rx in _COMPILED:
        m = rx.match(q)
        if not m:
            continue
        if kind == "order":
            a, b = clean(m.group("a")), clean(m.group("b"))
            if a and b:
                return Route("order", [a, b], m.group("order").lower())
        else:
            x = clean(m.group("x"))
            if x:
                return Route(kind, [x])
    return None


FILLER = {"the", "a", "an", "our", "my", "this", "that", "function", "functions", "method",
          "methods", "class", "variable", "constant", "value", "field", "property", "call",
          "calls", "called", "module", "file", "helper", "code", "thing"}


def clean(phrase: str) -> str:
    p = phrase.strip().strip(",;:")
    if len(p) >= 2 and p[0] == p[-1] and p[0] in "'\"`":
        return p
    words = [w for w in p.split() if w.lower() not in FILLER]
    return " ".join(words).strip()


_IDENT = re.compile(r"[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*")
_STRING = re.compile(r"""(['"`])((?:(?!\1)[^\\\n]|\\.){2,200})\1""")


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def variants(words: list[str]) -> list[str]:
    w = [x for x in words if x]
    if not w:
        return []
    low = [x.lower() for x in w]
    out = [low[0] + "".join(x.capitalize() for x in low[1:]),
           "".join(x.capitalize() for x in low),
           "_".join(low), "_".join(low).upper(), "-".join(low), "".join(low), " ".join(w)]
    return list(dict.fromkeys(out))


@dataclass
class Resolution:
    phrase: str
    best: str | None
    others: list[str] = field(default_factory=list)
    how: str = ""


class Vocabulary:

    def __init__(self, texts: list[str]) -> None:
        from collections import Counter
        self.count: Counter[str] = Counter()
        self.strings: Counter[str] = Counter()
        for t in texts:
            self.count.update(_IDENT.findall(t))
            self.strings.update(m.group(2) for m in _STRING.finditer(t))
        self.by_norm: dict[str, list[str]] = {}
        for name in list(self.count) + list(self.strings):
            self.by_norm.setdefault(norm(name), []).append(name)

    def freq(self, name: str) -> int:
        return self.count.get(name, 0) + self.strings.get(name, 0)

    def resolve(self, phrase: str, limit: int = 6, identifiers_only: bool = False) -> Resolution:
        p = phrase.strip()
        if len(p) >= 2 and p[0] == p[-1] and p[0] in "'\"`":
            inner = p[1:-1]
            return Resolution(phrase, inner, [], "quoted")
        ok = (lambda n: n in self.count) if identifiers_only else (lambda n: True)
        if p in self.count or (p in self.strings and not identifiers_only):
            return Resolution(phrase, p, self._near(p, limit), "exact")
        words = re.findall(r"[A-Za-z0-9]+", p)
        if not words:
            return Resolution(phrase, None, [], "none")
        ranked: dict[str, float] = {}
        for v in variants(words):
            for name in self.by_norm.get(norm(v), []):
                if ok(name):
                    ranked[name] = max(ranked.get(name, 0), 3.0)
        key = norm("".join(words))
        if not ranked and len(key) >= 4:
            for nkey, names in self.by_norm.items():
                if all(w.lower() in nkey for w in words) and len(nkey) <= len(key) * 3:
                    for name in names:
                        if ok(name):
                            ranked[name] = max(ranked.get(name, 0), 2.0 - (len(nkey) - len(key)) / 100)
        if not ranked and len(key) >= 4:
            for nkey in difflib.get_close_matches(key, list(self.by_norm), n=limit, cutoff=0.8):
                for name in self.by_norm[nkey]:
                    if ok(name):
                        ranked[name] = max(ranked.get(name, 0),
                                           1.0 + difflib.SequenceMatcher(None, key, nkey).ratio() / 10)
        if not ranked:
            return Resolution(phrase, None, [], "none")
        order = sorted(ranked, key=lambda n: (-ranked[n], -self.freq(n), n))
        how = {3.0: "variant"}.get(ranked[order[0]], "contains" if ranked[order[0]] >= 1.5 else "fuzzy")
        return Resolution(phrase, order[0], order[1:limit], how)

    def _near(self, name: str, limit: int) -> list[str]:
        return [n for n in self.by_norm.get(norm(name), []) if n != name][:limit]
