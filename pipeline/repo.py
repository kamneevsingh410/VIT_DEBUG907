from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

SOURCE_SUFFIXES = {".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".py"}

SKIP_DIR_REASONS = {
    "vendored dependencies": {"node_modules", "vendor", "bower_components", "site-packages",
                              "jspm_packages", ".yarn", ".pnpm-store"},
    "virtual environment": {".venv", "venv", "virtualenv", ".tox", ".nox", "conda-env"},
    "build output": {"dist", "build", "out", ".next", ".nuxt", ".svelte-kit", ".turbo",
                     ".output", ".vercel", ".parcel-cache", "storybook-static", "target",
                     ".expo", ".docusaurus"},
    "caches and tooling": {".git", "__pycache__", ".mypy_cache", ".pytest_cache",
                           ".ruff_cache", ".ipynb_checkpoints", ".cache", ".gradle",
                           ".idea", ".vscode", "coverage", ".nyc_output",
                           ".eggs", "htmlcov"},
    "generated test snapshots": {"tap-snapshots", "__snapshots__"},
}
SKIP_DIRS = set().union(*SKIP_DIR_REASONS.values())

WINDOW_LANGUAGES = {".java": "java", ".go": "go", ".rb": "ruby", ".php": "php",
                    ".c": "c", ".h": "c", ".cpp": "cpp", ".cc": "cpp", ".hpp": "cpp",
                    ".cs": "csharp", ".rs": "rust", ".kt": "kotlin", ".swift": "swift",
                    ".scala": "scala", ".lua": "lua", ".dart": "dart", ".m": "objective-c",
                    ".sh": "shell", ".sql": "sql"}

LANGUAGES = {".py": "python", ".js": "javascript", ".jsx": "javascript",
             ".mjs": "javascript", ".cjs": "javascript", ".ts": "typescript",
             ".tsx": "typescript"}

SECRET_SUFFIXES = {".pem", ".key", ".p12", ".pfx", ".jks", ".keystore", ".crt", ".cer",
                   ".der", ".gpg", ".asc", ".ppk", ".kdbx"}
SECRET_NAME_RE = re.compile(
    r"^(\.env(\..*)?|.*\.env|\.npmrc|\.pypirc|\.netrc|\.git-credentials|\.htpasswd"
    r"|id_(rsa|dsa|ecdsa|ed25519)(\.pub)?"
    r"|(credentials?|secrets?)([._-].*)?"
    r"|service[-_]?account.*\.json|.*[-_]key\.json)$", re.I)
SECRET_CONTENT_RE = re.compile(
    r"-----BEGIN (?:[A-Z]+ )*PRIVATE KEY-----"
    r"|\bAKIA[0-9A-Z]{16}\b"
    r"|\bgh[pousr]_[A-Za-z0-9]{36,}\b"
    r"|\bxox[baprs]-[A-Za-z0-9-]{10,}"
    r"|\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}"
    r"|\bsk_live_[0-9A-Za-z]{20,}"
    r"|\bAIza[0-9A-Za-z_-]{35}\b"
    r"|(?i:(?:api[_-]?key|secret|passw(?:or)?d|auth[_-]?token|access[_-]?token|private[_-]?key)"
    r"[\"']?\s*[:=]\s*[\"'][^\"'\s]{16,}[\"'])")
MINIFIED_NAME_RE = re.compile(r"\.(min|bundle|chunk)\.(js|mjs|cjs)$|[-.]bundle\.js$", re.I)
MAX_LINE_CHARS = 2000


def is_secret_file(name: str) -> bool:
    lower = name.lower()
    return (Path(lower).suffix in SECRET_SUFFIXES or bool(SECRET_NAME_RE.match(name))
            or any(f"{s}." in lower for s in (".pem", ".key")))


def contains_secret(text: str) -> bool:
    return bool(SECRET_CONTENT_RE.search(text))

MIN_CHARS = 40
MAX_CHARS = 6000
WINDOW_LINES = 60
MAX_FILE_BYTES = 400_000

JS_FUNC_RE = re.compile(
    r"^[ \t]*(?:export\s+(?:default\s+)?)?"
    r"(?:"
    r"(?:async\s+)?function\s*\*?\s*(?P<f>[A-Za-z_$][\w$]*)\s*\("
    r"|(?:const|let|var)\s+(?P<a>[A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?"
    r"(?:function\b|\([^)]*\)\s*=>|[A-Za-z_$][\w$]*\s*=>)"
    r"|(?:static\s+)?(?:async\s+)?(?:get\s+|set\s+)?(?P<m>[A-Za-z_$][\w$]*)\s*\([^)]*\)\s*\{"
    r"|class\s+(?P<c>[A-Za-z_$][\w$]*)"
    r")",
    re.M,
)
PY_FUNC_RE = re.compile(r"^(?P<indent>[ \t]*)(?:async\s+)?(?:def|class)\s+(?P<n>\w+)", re.M)
NOT_METHODS = {"if", "for", "while", "switch", "catch", "return", "function", "with"}


@dataclass(frozen=True)
class Snippet:
    id: str
    path: str
    name: str
    start_line: int
    text: str


def _dir_reason(dirpath: Path, name: str, include_vendored: bool) -> str | None:
    if name in SKIP_DIR_REASONS["virtual environment"] or (dirpath / name / "pyvenv.cfg").exists():
        return "virtual environment"
    for reason, names in SKIP_DIR_REASONS.items():
        if name in names:
            return None if include_vendored and reason == "vendored dependencies" else reason
    if name.startswith("."):
        return "hidden directory"
    return None


def _file_reason(path: Path, all_languages: bool = False) -> str | None:
    name = path.name
    if is_secret_file(name):
        return "secret file"
    if path.suffix not in SOURCE_SUFFIXES and not (all_languages
                                                   and path.suffix in WINDOW_LANGUAGES):
        return "not source code (docs, data, lockfiles, notebooks)"
    if MINIFIED_NAME_RE.search(name):
        return "minified or generated"
    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            return "too large (generated)"
        with path.open(encoding="utf-8", errors="replace") as handle:
            if any(len(line) > MAX_LINE_CHARS for line in handle):
                return "minified or generated"
    except OSError:
        return "unreadable"
    return None


def iter_source_files(root: Path, include_vendored: bool = False,
                      skipped_dirs=None, skipped_files=None, all_languages: bool = False):
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        here = Path(dirpath)
        keep = []
        for d in dirnames:
            reason = "symbolic link" if (here / d).is_symlink() else _dir_reason(here, d, include_vendored)
            if reason is None:
                keep.append(d)
            elif skipped_dirs is not None:
                skipped_dirs[reason] += 1
        dirnames[:] = keep
        for name in filenames:
            path = here / name
            reason = "symbolic link" if path.is_symlink() else _file_reason(path, all_languages)
            if reason is None:
                yield path
            elif skipped_files is not None:
                skipped_files[reason] += 1


@dataclass
class ScanResult:
    corpus: dict[str, str]
    files_indexed: int
    languages: dict[str, int]
    skipped_dirs: "Counter[str]"
    skipped_files: "Counter[str]"
    skipped_snippets: "Counter[str]"


def scan_repo(root: Path, include_vendored: bool = False,
              limit_files: int | None = None, all_languages: bool = False) -> ScanResult:
    from collections import Counter

    root = root.resolve()
    dirs, files, snippets, langs = Counter(), Counter(), Counter(), Counter()
    corpus: dict[str, str] = {}
    indexed = 0
    for i, path in enumerate(iter_source_files(root, include_vendored, dirs, files,
                                                  all_languages)):
        if limit_files is not None and i >= limit_files:
            break
        indexed += 1
        langs[LANGUAGES.get(path.suffix) or WINDOW_LANGUAGES.get(path.suffix, path.suffix)] += 1
        for snip in extract(path, root):
            if contains_secret(snip.text):
                snippets["contains a secret"] += 1
                continue
            corpus[snip.id] = snip.text
    return ScanResult(corpus, indexed, dict(langs), dirs, files, snippets)


def _block_end_js(lines: list[str], start: int) -> int:
    depth, parens, opened, seen_arrow = 0, 0, False, False
    for i in range(start, len(lines)):
        line = re.sub(r"(['\"`])(?:\\.|(?!\1).)*\1", '""', lines[i])
        line = re.sub(r"(?<![\\:])//.*$", "", line)
        for ch in line:
            if ch == "(":
                parens += 1
            elif ch == ")":
                parens = max(0, parens - 1)
            elif parens:
                continue
            elif ch == "{":
                depth += 1
                opened = True
            elif ch == "}":
                depth -= 1
                if opened and depth <= 0:
                    return i
        if not opened and i > start and lines[i].rstrip().endswith(";"):
            return i
        seen_arrow = seen_arrow or "=>" in line
        if not opened and seen_arrow and parens == 0 and _expression_ends(lines, i, line):
            return i
    return min(len(lines) - 1, start + WINDOW_LINES)


_CONTINUES_AT_END = ("=>", ",", "(", "[", "{", "+", "-", "*", "/", "&&", "||", "?", ":",
                     "=", ".", "??")
_CONTINUES_AT_START = (".", "?", ":", "&&", "||", "+", "-", "*", "/", ")", "]", "}", "??")


def _expression_ends(lines: list[str], i: int, stripped: str) -> bool:
    body = stripped.rstrip()
    if not body or body.endswith(_CONTINUES_AT_END):
        return False
    nxt = next((ln.strip() for ln in lines[i + 1:] if ln.strip()), "")
    return not nxt.startswith(_CONTINUES_AT_START)


def _block_end_py(lines: list[str], start: int, indent: int) -> int:
    end = start
    for i in range(start + 1, len(lines)):
        stripped = lines[i].strip()
        if not stripped:
            continue
        if len(lines[i]) - len(lines[i].lstrip()) <= indent:
            break
        end = i
    return end


def _python_definitions(source: str) -> list[tuple[str, int, int]] | None:
    import ast
    import warnings
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return None
    out = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            first = min([node.lineno] + [d.lineno for d in node.decorator_list])
            out.append((node.name, first - 1, (node.end_lineno or node.lineno) - 1))
    return sorted(out, key=lambda item: item[1])


def _windows(rel: str, lines: list[str]) -> list[Snippet]:
    out = []
    for lo in range(0, len(lines), WINDOW_LINES):
        text = "\n".join(lines[lo:lo + WINDOW_LINES]).strip()
        if len(text) >= MIN_CHARS:
            out.append(Snippet(f"{rel}::window#{lo + 1}", rel, "window", lo + 1, text))
    return out


def _cap(snippet: Snippet) -> list[Snippet]:
    if len(snippet.text) <= MAX_CHARS:
        return [snippet]
    lines = snippet.text.splitlines()
    out = []
    for lo in range(0, len(lines), WINDOW_LINES):
        text = "\n".join(lines[lo:lo + WINDOW_LINES]).strip()
        if len(text) >= MIN_CHARS:
            line = snippet.start_line + lo
            out.append(Snippet(f"{snippet.path}::{snippet.name}#{line}",
                               snippet.path, snippet.name, line, text))
    return out


def extract(path: Path, root: Path) -> list[Snippet]:
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    rel = path.relative_to(root).as_posix()
    if path.suffix in WINDOW_LANGUAGES:
        lines = source.splitlines()
        return [s for snip in _windows(rel, lines) for s in _cap(snip)]
    lines = source.splitlines()
    found: list[Snippet] = []

    if path.suffix == ".py" and (py := _python_definitions(source)) is not None:
        for name, start, end in py:
            text = "\n".join(lines[start:end + 1]).strip()
            if len(text) >= MIN_CHARS:
                found.append(Snippet(f"{rel}::{name}#{start + 1}", rel, name, start + 1, text))
    elif path.suffix == ".py":
        for m in PY_FUNC_RE.finditer(source):
            start = source.count("\n", 0, m.start())
            end = _block_end_py(lines, start, len(m.group("indent")))
            text = "\n".join(lines[start:end + 1]).strip()
            if len(text) >= MIN_CHARS:
                found.append(Snippet(f"{rel}::{m.group('n')}#{start + 1}",
                                     rel, m.group("n"), start + 1, text))
    else:
        covered_until = -1
        for m in JS_FUNC_RE.finditer(source):
            name = m.group("f") or m.group("a") or m.group("m") or m.group("c")
            if not name or name in NOT_METHODS:
                continue
            start = source.count("\n", 0, m.start())
            if start <= covered_until and not m.group("c"):
                continue
            end = _block_end_js(lines, start)
            text = "\n".join(lines[start:end + 1]).strip()
            if len(text) < MIN_CHARS:
                continue
            found.append(Snippet(f"{rel}::{name}#{start + 1}", rel, name,
                                 start + 1, text))
            if not m.group("c"):
                covered_until = end

    snippets = found or _windows(rel, lines)
    return [s for snip in snippets for s in _cap(snip)]


def extract_repo(root: Path, include_vendored: bool = False,
                 limit_files: int | None = None, all_languages: bool = False) -> dict[str, str]:
    return scan_repo(root, include_vendored, limit_files, all_languages).corpus
