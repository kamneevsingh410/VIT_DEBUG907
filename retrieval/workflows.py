from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from retrieval.embed import Embedder
from retrieval.index import BuildStats, Index, build, connect
from retrieval.rerank import RerankConfig
from retrieval.search import SearchConfig, SearchResult, search

CODE_VIEW = {"code": 1.0, "nl": 0.0, "raw": 1.0}

CONF_HIGH = 0.03
CONF_LOW = 0.01


def shipped_config(collapse_versions: bool = False, prefer_newest: bool = False,
                   hubness: bool = False) -> SearchConfig:
    return SearchConfig(use_sparse=False, rerank=RerankConfig.off(),
                        view_weights=dict(CODE_VIEW),
                        collapse_versions=collapse_versions, prefer_newest=prefer_newest,
                        hubness=hubness)


def hubness_default(index) -> bool:
    return getattr(index, "corpus_kind", "") == "apps"


_TEST_PATH = re.compile(r"(^|/)(tests?|__tests__|spec|specs)/|(^|/)test_[^/]*\.py$"
                        r"|_test\.py$|\.(test|spec)\.[cm]?[jt]sx?$")


def is_test_path(path: str) -> bool:
    return bool(_TEST_PATH.search(path.replace("\\", "/")))


def shipped_search(index: Index, text: str, embedder: Embedder,
                   top_k: int = 10, collapse: bool | None = None,
                   include_tests: bool = True, prefer_newest: bool = False,
                   hubness: bool | None = None) -> SearchResult:
    if collapse is None:
        collapse = len(getattr(index, "loaded_versions", [])) > 1
    if hubness is None:
        hubness = hubness_default(index)
    cfg = shipped_config(collapse_versions=collapse, prefer_newest=prefer_newest,
                         hubness=hubness)
    if include_tests:
        return search(index, text, cfg, top_k=top_k, embedder=embedder)
    result = search(index, text, cfg, top_k=top_k * 4, embedder=embedder)
    ident = getattr(index, "identity_of", {})
    result.hits = [(sid, s) for sid, s in result.hits
                   if not is_test_path(ident.get(sid, sid).split("::", 1)[0])][:top_k]
    return result


@dataclass
class EvalRun:
    scores: "Scores"
    run: dict[str, list[str]]
    latencies_ms: list[float]
    seconds: float

    @property
    def p50_ms(self) -> float:
        ordered = sorted(self.latencies_ms)
        return ordered[len(ordered) // 2] if ordered else 0.0


def evaluate_shipped(index: Index, queries: dict[str, str],
                     qrels: dict[str, dict[str, int]], embedder: Embedder,
                     show_progress: bool = False) -> EvalRun:
    from bench.metrics import evaluate

    started = time.perf_counter()
    embedder.encode(list(queries.values()), show_progress=show_progress)
    run: dict[str, list[str]] = {}
    latencies: list[float] = []
    for qid, text in queries.items():
        t = time.perf_counter()
        run[qid] = shipped_search(index, text, embedder, top_k=100).ids
        latencies.append((time.perf_counter() - t) * 1000)
    return EvalRun(scores=evaluate(run, qrels), run=run, latencies_ms=latencies,
                   seconds=time.perf_counter() - started)


def confidence(hits: list[tuple[str, float]],
               identity: dict[str, str] | None = None) -> tuple[str, float]:
    if identity:
        top = identity.get(hits[0][0], hits[0][0]) if hits else None
        hits = hits[:1] + [h for h in hits[1:] if identity.get(h[0], h[0]) != top]
    if len(hits) < 2:
        return ("low", 0.0) if not hits else ("medium", 0.0)
    margin = float(hits[0][1] - hits[1][1])
    if margin >= CONF_HIGH:
        return "high", margin
    if margin < CONF_LOW:
        return "low", margin
    return "medium", margin


@dataclass
class IndexReport:
    db: Path
    snippets: int
    version: str
    stats: BuildStats
    seconds: float
    source: str

    @property
    def encoded(self) -> int:
        return self.stats.encoded_vectors

    @property
    def reused(self) -> int:
        return self.stats.reused_vectors


def verify_index(db: Path, version: str | None = None) -> tuple[int, int | None]:
    if not Path(db).is_file():
        raise FileNotFoundError(f"no index at {db}")
    conn = sqlite3.connect(f"{Path(db).resolve().as_uri()}?mode=ro", uri=True)
    try:
        if version is None:
            rows = conn.execute("SELECT COUNT(*) FROM snippets").fetchone()[0]
            doc = conn.execute("SELECT value FROM meta WHERE key='doc_count'").fetchone()
        else:
            rows = conn.execute("SELECT COUNT(*) FROM snippets WHERE version = ?",
                                [version]).fetchone()[0]
            doc = conn.execute("SELECT snippets FROM versions WHERE version = ?",
                               [version]).fetchone()
        return rows, (int(doc[0]) if doc else None)
    finally:
        conn.close()


def index_is_usable(db: Path) -> bool:
    if not db.exists():
        return False
    try:
        rows, doc = verify_index(db)
    except sqlite3.DatabaseError:
        return False
    return rows > 0 and doc == rows


def _build(corpus: dict[str, str], db: Path, version: str, embedder: Embedder | None,
           show_progress: bool, source: str, *, reset: bool = False,
           drop_vectors: bool = False, corpus_kind: str = "", view: str = "code") -> IndexReport:
    started = time.perf_counter()
    stats: BuildStats = build(corpus, db, version=version, views=(view,),
                              embedder=embedder, reset=reset, drop_vectors=drop_vectors,
                              show_progress=show_progress, corpus_kind=corpus_kind)
    rows, recorded = verify_index(db, version)
    if rows != len(corpus) or recorded != len(corpus):
        raise RuntimeError(f"index verification failed: {rows} rows of version {version} "
                           f"on disk, {recorded} recorded, expected {len(corpus)}")
    return IndexReport(db=db, snippets=rows, version=version, stats=stats,
                       seconds=time.perf_counter() - started, source=source)


def index_folder(root: Path, db: Path, *, version: str = "v1",
                 include_vendored: bool = False, embedder: Embedder | None = None,
                 show_progress: bool = False, reset: bool = False,
                 drop_vectors: bool = False, all_languages: bool = False,
                 view: str = "code") -> IndexReport:
    from pipeline.repo import extract_repo

    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"not a folder: {root}")
    corpus = extract_repo(root, include_vendored=include_vendored,
                          all_languages=all_languages)
    if not corpus:
        raise ValueError(f"no JS/TS/Python functions found under {root}")
    return _build(corpus, db, version, embedder, show_progress, str(root),
                  reset=reset, drop_vectors=drop_vectors, corpus_kind="repo", view=view)


def index_corpus(path: Path, db: Path, *, embedder: Embedder | None = None,
                 show_progress: bool = False, version: str = "v1",
                 view: str = "code") -> IndexReport:
    import json
    from retrieval.batch import read_corpus

    corpus, paths = read_corpus(Path(path))
    report = _build(corpus, db, version, embedder, show_progress, str(path),
                    corpus_kind="corpus", view=view)
    conn = connect(db)
    try:
        conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                     ("corpus_paths", json.dumps(paths)))
        conn.commit()
    finally:
        conn.close()
    return report


def corpus_paths(index: Index) -> dict[str, str]:
    cached = getattr(index, "_corpus_paths", None)
    if cached is None:
        import json
        row = index.conn.execute("SELECT value FROM meta WHERE key = 'corpus_paths'").fetchone()
        cached = json.loads(row[0]) if row else {}
        index._corpus_paths = cached
    return cached


APPS_VERSION = "test"


def index_apps(db: Path, *, embedder: Embedder | None = None, limit: int | None = None,
               show_progress: bool = False, reset: bool = False,
               drop_vectors: bool = False, view: str = "raw") -> IndexReport:
    from bench.dataset import load_appsretrieval

    corpus, _queries, _qrels = load_appsretrieval(limit=limit)
    return _build(corpus, db, APPS_VERSION, embedder, show_progress,
                  "CoIR AppsRetrieval (test)", reset=reset, drop_vectors=drop_vectors,
                  corpus_kind="apps", view=view)


@dataclass(frozen=True)
class HitInfo:
    snippet_id: str
    path: str
    name: str
    start: int | None
    end: int | None
    language: str


def describe_hit(index: Index, snippet_id: str, text: str | None = None) -> HitInfo:
    text = index.content(snippet_id) if text is None else text
    n_lines = max(1, len(text.splitlines()))
    ident = getattr(index, "identity_of", {}).get(snippet_id, snippet_id)
    if "::" in ident:
        path, rest = ident.split("::", 1)
        name, _, line = rest.partition("#")
        start = int(line) if line.isdigit() else None
        end = start + n_lines - 1 if start is not None else None
    elif getattr(index, "corpus_kind", "") == "corpus":
        path, name, start, end = corpus_paths(index).get(ident, "corpus"), ident, 1, n_lines
    else:
        path, name, start, end = "AppsRetrieval", ident, 1, n_lines
    suffix = Path(path).suffix.lower()
    language = {".py": "python", ".js": "javascript", ".jsx": "jsx", ".mjs": "javascript",
                ".cjs": "javascript", ".ts": "typescript", ".tsx": "tsx"}.get(suffix, "python")
    return HitInfo(snippet_id, path, name, start, end, language)


@dataclass(frozen=True)
class UseHit:
    uid: str
    path: str
    name: str
    line: int
    text: str
    unit: str = ""


@dataclass(frozen=True)
class OrderHit:
    uid: str
    path: str
    name: str
    first_line: int
    second_line: int


def _term_pattern(term: str):
    import re
    escaped = re.escape(term)
    return re.compile(rf"\b{escaped}\b" if re.fullmatch(r"\w+", term) else escaped)


def _all_snippets(index: Index) -> list[tuple[HitInfo, list[str]]]:
    versions = index.loaded_versions or [""]
    marks = ",".join("?" * len(versions))
    rows = index.conn.execute(
        f"SELECT id, version, content FROM snippets WHERE version IN ({marks})", versions)
    out = []
    for sid, version, content in rows:
        uid = index.uid(sid, version)
        out.append((describe_hit(index, uid, content), content.splitlines()))
    return sorted(out, key=lambda pair: len(pair[1]))


def _source_unit(index: Index, info: HitInfo) -> str:
    ident = getattr(index, "identity_of", {}).get(info.snippet_id, info.snippet_id)
    return info.path if "::" in ident else ident


def uses(index: Index, term: str) -> list[UseHit]:
    pattern = _term_pattern(term)
    seen: dict[tuple[str, int], UseHit] = {}
    for info, lines in _all_snippets(index):
        for offset, text in enumerate(lines):
            if pattern.search(text):
                line = (info.start or 1) + offset
                unit = _source_unit(index, info)
                seen.setdefault((unit, line),
                                UseHit(info.snippet_id, info.path, info.name, line, text.strip(),
                                       unit))
    return sorted(seen.values(), key=lambda h: (h.unit, h.line))


def calls_before(index: Index, first: str, second: str) -> list[OrderHit]:
    import re

    call = lambda name: re.compile(rf"(?<![\w$]){re.escape(name)}\s*\(")
    a, b = call(first), call(second)
    seen: dict[tuple[str, int, int], OrderHit] = {}
    for info, lines in _all_snippets(index):
        la = next((i for i, t in enumerate(lines) if a.search(t)), None)
        lb = next((i for i, t in enumerate(lines) if b.search(t)), None)
        if la is not None and lb is not None and la < lb:
            start = info.start or 1
            key = (_source_unit(index, info), start + la, start + lb)
            seen.setdefault(key, OrderHit(info.snippet_id, info.path, info.name, *key[1:]))
    return sorted(seen.values(), key=lambda h: (h.path, h.first_line))


def _definition_pattern(name: str):
    import re
    n = re.escape(name.rsplit(".", 1)[-1])
    return re.compile(
        rf"^\s*(?:async\s+)?def\s+{n}\b"
        rf"|^\s*class\s+{n}\b"
        rf"|^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s*\*?\s*{n}\b"
        rf"|^\s*(?:export\s+)?(?:const|let|var)\s+{n}\s*="
        rf"|^\s*(?:static\s+)?(?:async\s+)?\*?{n}\s*\([^)]*\)\s*\{{"
        rf"|^\s*{n}\s*[:=]\s*(?:async\s+)?(?:function\b|\()"
        rf"|^{n}\s*=")


def callers(index: Index, name: str) -> list[UseHit]:
    import re
    last = name.rsplit(".", 1)[-1]
    call = re.compile(rf"(?<![\w$]){re.escape(last)}\s*\(")
    define = _definition_pattern(name)
    seen: dict[tuple[str, int], UseHit] = {}
    for info, lines in _all_snippets(index):
        for offset, text in enumerate(lines):
            if call.search(text) and not define.search(text):
                line = (info.start or 1) + offset
                unit = _source_unit(index, info)
                seen.setdefault((unit, line), UseHit(info.snippet_id, info.path, info.name, line,
                                                     text.strip(), unit))
    return sorted(seen.values(), key=lambda h: (h.unit, h.line))


def definitions(index: Index, name: str) -> list[UseHit]:
    define = _definition_pattern(name)
    seen: dict[tuple[str, int], UseHit] = {}
    for info, lines in _all_snippets(index):
        for offset, text in enumerate(lines):
            if define.search(text):
                line = (info.start or 1) + offset
                unit = _source_unit(index, info)
                seen.setdefault((unit, line), UseHit(info.snippet_id, info.path, info.name, line,
                                                     text.strip(), unit))
    return sorted(seen.values(), key=lambda h: (h.unit, h.line))


def vocabulary(index: Index):
    cached = getattr(index, "_vocabulary", None)
    if cached is None:
        from retrieval.router import Vocabulary
        cached = Vocabulary(["\n".join(lines) for _, lines in _all_snippets(index)])
        index._vocabulary = cached
    return cached
