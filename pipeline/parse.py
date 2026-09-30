from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ParsedSnippet:
    language: str
    names: list[str] = field(default_factory=list)
    signatures: list[str] = field(default_factory=list)
    docstrings: list[str] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)
    strings: list[str] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    control: list[str] = field(default_factory=list)
    class_names: list[str] = field(default_factory=list)


STRING_RE = re.compile(r"""(["'])((?:\\.|(?!\1).)*?)\1""")
TRIPLE_RE = re.compile(r'"""(.*?)"""|\'\'\'(.*?)\'\'\'', re.DOTALL)
CALL_RE = re.compile(r"\b([A-Za-z_$][\w$]*)\s*\(")

NOT_CALLS = {
    "if", "for", "while", "switch", "catch", "return", "with", "elif",
    "except", "print", "def", "function", "class", "lambda", "await",
    "yield", "assert", "del", "raise", "and", "or", "not", "in", "is",
}

PY_HINTS = (re.compile(r"^\s*def\s+\w+\s*\(", re.M), re.compile(r"^\s*import\s+\w", re.M),
            re.compile(r"^\s*from\s+[\w.]+\s+import", re.M), re.compile(r":\s*$", re.M))
JS_HINTS = (re.compile(r"\bfunction\s+\w*\s*\("), re.compile(r"=>"),
            re.compile(r"\b(?:const|let|var)\s+\w+\s*="), re.compile(r";\s*$", re.M))


def detect_language(source: str) -> str:
    py = sum(1 for pattern in PY_HINTS if pattern.search(source))
    js = sum(1 for pattern in JS_HINTS if pattern.search(source))
    if py > js:
        return "python"
    if js > py:
        return "javascript"
    return "python" if "def " in source else "unknown"


def strip_strings(text: str) -> str:
    return STRING_RE.sub('""', text)


def _string_literals(source: str) -> list[str]:
    out: list[str] = []
    for match in STRING_RE.finditer(source):
        value = match.group(2).strip()
        if 1 < len(value) < 200:
            out.append(value)
    return out


def _calls(source: str, own_names: set[str]) -> list[str]:
    out: list[str] = []
    for match in CALL_RE.finditer(strip_strings(source)):
        name = match.group(1)
        if name in NOT_CALLS or name in own_names:
            continue
        out.append(name)
    return out


def _control_summary(source: str, language: str) -> list[str]:
    body = strip_strings(source)
    out: list[str] = []
    if re.search(r"\bfor\b", body):
        out.append("loops over sequence")
    if re.search(r"\bwhile\b", body):
        out.append("while loop")
    if re.search(r"\bif\b", body):
        out.append("conditional branch")
    if re.search(r"\b(?:try|except|catch|finally)\b", body):
        out.append("error handling")
    if re.search(r"\b(?:return|yield)\b", body):
        out.append("returns value")
    if re.search(r"\b(?:sorted|sort)\s*\(", body):
        out.append("sorts data")
    if re.search(r"\b(?:input|stdin|readline|scanf|prompt)\b", body):
        out.append("reads input")
    if re.search(r"\b(?:print|stdout|console\.log|write)\b", body):
        out.append("writes output")
    if re.search(r"\b(?:append|push|insert|add)\s*\(", body):
        out.append("builds collection")
    nesting = max((len(m) // 4 for m in re.findall(r"^( +)", body, re.M)), default=0)
    if nesting >= 3:
        out.append("deeply nested")
    if language == "python" and re.search(r"^\s*class\s+\w", body, re.M):
        out.append("defines class")
    return out


def parse_python(source: str) -> ParsedSnippet:
    names = re.findall(r"^\s*(?:async\s+)?def\s+([A-Za-z_]\w*)\s*\(", source, re.M)
    classes = re.findall(r"^\s*class\s+([A-Za-z_]\w*)", source, re.M)
    signatures = [m.group(0).strip().rstrip(":")
                  for m in re.finditer(r"^\s*(?:async\s+)?def\s+[A-Za-z_]\w*\s*\([^)]*\)[^:]*:",
                                       source, re.M)]
    docs: list[str] = []
    for match in TRIPLE_RE.finditer(source):
        text = (match.group(1) or match.group(2) or "").strip()
        if text:
            docs.append(text)
    docs += [c.strip() for c in re.findall(r"^\s*#\s?(.+)$", source, re.M) if c.strip()]
    imports = re.findall(r"^\s*(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))", source, re.M)
    flat_imports = [a or b for a, b in imports]
    without_docs = TRIPLE_RE.sub('""', source)
    return ParsedSnippet(
        language="python",
        names=names,
        class_names=classes,
        signatures=signatures,
        docstrings=docs,
        calls=_calls(source, set(names) | set(classes)),
        strings=_string_literals(without_docs),
        imports=flat_imports,
        control=_control_summary(source, "python"),
    )


def parse_javascript(source: str) -> ParsedSnippet:
    names = re.findall(r"\bfunction\s+([A-Za-z_$][\w$]*)\s*\(", source)
    names += re.findall(r"\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*=>", source)
    classes = re.findall(r"\bclass\s+([A-Za-z_$][\w$]*)", source)
    signatures = [m.group(0).strip()
                  for m in re.finditer(r"\bfunction\s+[A-Za-z_$][\w$]*\s*\([^)]*\)", source)]
    docs = [c.strip() for c in re.findall(r"^\s*//\s?(.+)$", source, re.M) if c.strip()]
    docs += [re.sub(r"^\s*\*\s?", "", line).strip()
             for block in re.findall(r"/\*\*(.*?)\*/", source, re.DOTALL)
             for line in block.splitlines() if line.strip(" *")]
    imports = re.findall(r"""require\s*\(\s*["']([^"']+)["']\s*\)""", source)
    imports += re.findall(r"""\bfrom\s+["']([^"']+)["']""", source)
    return ParsedSnippet(
        language="javascript",
        names=names,
        class_names=classes,
        signatures=signatures,
        docstrings=[d for d in docs if d],
        calls=_calls(source, set(names) | set(classes)),
        strings=_string_literals(source),
        imports=imports,
        control=_control_summary(source, "javascript"),
    )


def parse_generic(source: str) -> ParsedSnippet:
    return ParsedSnippet(
        language="unknown",
        names=[],
        signatures=[],
        docstrings=[],
        calls=_calls(source, set()),
        strings=_string_literals(source),
        imports=[],
        control=_control_summary(source, "unknown"),
    )


EXTRACTORS = {
    "python": parse_python,
    "javascript": parse_javascript,
    "unknown": parse_generic,
}


def parse(source: str, language: str | None = None) -> ParsedSnippet:
    lang = language or detect_language(source)
    return EXTRACTORS.get(lang, parse_generic)(source)
