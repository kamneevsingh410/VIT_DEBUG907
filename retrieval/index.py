from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from pipeline.chunk import ChunkConfig, chunk
from pipeline.doc_proc import DEFAULT_VIEWS, DocConfig, enrich
from pipeline.tokens import tokenize
from retrieval.embed import Embedder, EmbedderUnavailable, mean_pool, pack, unpack

SCHEMA = """
CREATE TABLE IF NOT EXISTS snippets (
  id          TEXT NOT NULL,
  version     TEXT NOT NULL,
  content     TEXT NOT NULL,
  hash        TEXT NOT NULL,
  enriched    TEXT NOT NULL,
  category    TEXT,
  language    TEXT,
  identifiers TEXT,
  vkeys       TEXT,  -- JSON {view: vector key}; see vector_key()
  PRIMARY KEY (id, version)
);

CREATE TABLE IF NOT EXISTS vectors (
  hash    TEXT NOT NULL,   -- vector_key(): sha256 of the exact encoded text
  view    TEXT NOT NULL,
  vector  BLOB NOT NULL,
  PRIMARY KEY (hash, view)
);

CREATE VIRTUAL TABLE IF NOT EXISTS search_text USING fts5(
  snippet_id UNINDEXED,
  version UNINDEXED,
  enriched
);

CREATE TABLE IF NOT EXISTS df (
  version TEXT NOT NULL,
  term    TEXT NOT NULL,
  n       INTEGER NOT NULL,
  PRIMARY KEY (version, term)
);

CREATE TABLE IF NOT EXISTS versions (
  version  TEXT PRIMARY KEY,
  built_at REAL NOT NULL,
  snippets INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS meta (
  key   TEXT PRIMARY KEY,
  value TEXT
);

CREATE INDEX IF NOT EXISTS idx_snippets_hash ON snippets(hash);
CREATE INDEX IF NOT EXISTS idx_snippets_version ON snippets(version);
"""


_RESETTABLE = frozenset({"snippets", "search_text", "df", "versions"})


def _check_table(table: str) -> None:
    if table not in _RESETTABLE:
        raise ValueError(f"refusing SQL on unknown table {table!r}")

SCHEMA_VERSION = "3: text-keyed vectors, versioned snippets"
_TEXT_KEYED = {"sha256-of-encoded-text/v2", SCHEMA_VERSION}


class IndexUnavailable(SystemExit):
    pass


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def vector_key(pieces: list[str]) -> str:
    return content_hash("\x1e".join(pieces))


def doc_tokens(enriched: str) -> list[str]:
    return [t for t in tokenize(enriched) if len(t) > 1]


@dataclass
class BuildStats:
    snippets: int = 0
    reused_vectors: int = 0
    encoded_vectors: int = 0
    text_cache_hits: int = 0
    model_encoded: int = 0
    views: int = 0
    elapsed_ms: int = 0
    encode_ms: int = 0
    embedder: str = "none"

    def as_dict(self) -> dict[str, object]:
        total = self.reused_vectors + self.encoded_vectors
        return {
            "snippets": self.snippets,
            "views": self.views,
            "vectors_new": self.encoded_vectors,
            "vectors_reused": self.reused_vectors,
            "reuse_pct": round(100.0 * self.reused_vectors / max(1, total), 1),
            "texts_from_cache": self.text_cache_hits,
            "texts_model_encoded": self.model_encoded,
            "encode_ms": self.encode_ms,
            "elapsed_ms": self.elapsed_ms,
            "embedder": self.embedder,
        }


@dataclass
class Index:

    conn: sqlite3.Connection
    ids: list[str] = field(default_factory=list)
    vectors: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    categories: dict[str, str] = field(default_factory=dict)
    identifiers: dict[str, set[str]] = field(default_factory=dict)
    lengths: dict[str, int] = field(default_factory=dict)
    document_frequency: dict[str, int] = field(default_factory=dict)
    doc_count: int = 0
    has_vectors: bool = False
    matrices: dict[str, object] = field(default_factory=dict)
    matrix_ids: dict[str, list[str]] = field(default_factory=dict)
    loaded_versions: list[str] = field(default_factory=list)
    available_versions: list[str] = field(default_factory=list)
    corpus_kind: str = ""
    identity_of: dict[str, str] = field(default_factory=dict)
    version_of: dict[str, str] = field(default_factory=dict)
    hash_of: dict[str, str] = field(default_factory=dict)
    uids_of: dict[str, list[str]] = field(default_factory=dict)

    def uid(self, snippet_id: str, version: str) -> str:
        return f"{snippet_id}@{version}" if len(self.loaded_versions) > 1 else snippet_id

    def _row(self, uid: str) -> tuple[str, str]:
        return (self.identity_of.get(uid, uid),
                self.version_of.get(uid, self.loaded_versions[0] if self.loaded_versions else ""))

    def idf(self, term: str) -> float:
        df = self.document_frequency.get(term, 0)
        return max(0.0, math.log(1.0 + (self.doc_count - df + 0.5) / (df + 0.5)))

    def content(self, uid: str) -> str:
        row = self.conn.execute("SELECT content FROM snippets WHERE id = ? AND version = ?",
                                list(self._row(uid))).fetchone()
        return row[0] if row else ""

    def enriched(self, uid: str) -> str:
        row = self.conn.execute("SELECT enriched FROM snippets WHERE id = ? AND version = ?",
                                list(self._row(uid))).fetchone()
        return row[0] if row else ""

    def enriched_many(self, uids: list[str], chunk: int = 400) -> dict[str, str]:
        rows = {self._row(u): u for u in uids}
        keys = list(rows)
        found: dict[str, str] = {}
        for lo in range(0, len(keys), chunk):
            part = keys[lo:lo + chunk]
            marks = ",".join("(?, ?)" for _ in part)
            params = [x for key in part for x in key]
            for sid, version, text in self.conn.execute(
                    f"SELECT id, version, enriched FROM snippets WHERE (id, version) IN "
                    f"(VALUES {marks})", params):
                found[rows[(sid, version)]] = text
        return {u: found.get(u, "") for u in uids}

    def close(self) -> None:
        self.conn.close()


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=DELETE")
    _migrate(conn)
    conn.executescript(SCHEMA)
    return conn


def _schema_of(conn: sqlite3.Connection) -> str | None:
    try:
        row = conn.execute("SELECT value FROM meta WHERE key IN ('schema', 'vector_key') "
                           "ORDER BY key = 'schema' DESC LIMIT 1").fetchone()
    except sqlite3.OperationalError:
        return None
    return row[0] if row else ""


def _migrate(conn: sqlite3.Connection) -> None:
    found = _schema_of(conn)
    if found is None or found == SCHEMA_VERSION:
        return
    for table in ("snippets", "search_text", "df", "versions"):
        _check_table(table)
        conn.execute(f"DROP TABLE IF EXISTS {table}")
    if found not in _TEXT_KEYED:
        conn.execute("DROP TABLE IF EXISTS vectors")
    conn.execute("DELETE FROM meta WHERE key IN ('vector_key', 'doc_count')")
    conn.commit()


def _open_readonly(db_path: Path) -> sqlite3.Connection:
    missing = f"No index found at {db_path}. Build one: debug907 index (or index-repo <folder>)."
    if not Path(db_path).is_file():
        raise IndexUnavailable(missing)
    conn = sqlite3.connect(f"{Path(db_path).resolve().as_uri()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("SELECT 1 FROM meta LIMIT 1")
    except sqlite3.DatabaseError:
        conn.close()
        raise IndexUnavailable(missing) from None
    return conn


def build(
    corpus: dict[str, str],
    db_path: Path,
    *,
    version: str = "v1",
    doc_config: DocConfig | None = None,
    chunk_config: ChunkConfig | None = None,
    views: tuple[str, ...] = DEFAULT_VIEWS,
    embedder: Embedder | None = None,
    use_embeddings: bool = True,
    reset: bool = False,
    drop_vectors: bool = False,
    show_progress: bool = False,
    corpus_kind: str = "",
) -> BuildStats:
    started = time.perf_counter()
    cfg_doc = doc_config or DocConfig()
    cfg_chunk = chunk_config or ChunkConfig()

    conn = connect(db_path)
    stats = BuildStats(views=len(views))
    try:
        for table in ("snippets", "search_text", "df", "versions"):
            _check_table(table)
            if reset:
                conn.execute(f"DELETE FROM {table}")
            else:
                conn.execute(f"DELETE FROM {table} WHERE version = ?", [version])
        if drop_vectors:
            conn.execute("DELETE FROM vectors")

        emb_for_sig = embedder or (Embedder() if use_embeddings else None)
        signature = "none" if emb_for_sig is None else emb_for_sig.signature
        previous_row = conn.execute(
            "SELECT value FROM meta WHERE key = 'embedder_sig'").fetchone()
        previous = previous_row[0] if previous_row else None
        if previous and previous != signature and signature != "none":
            conn.execute("DELETE FROM vectors")
            if show_progress:
                print(f"  encoder changed ({previous} -> {signature}); "
                      "stored vectors dropped")

        cached: set[tuple[str, str]] = {
            (row[0], row[1]) for row in conn.execute("SELECT hash, view FROM vectors")
        }

        rows: list[tuple] = []
        fts_rows: list[tuple[str, str, str]] = []
        pending: dict[tuple[str, str], list[str]] = {}
        df_counter: Counter[str] = Counter()
        reused = 0

        for snippet_id, source in corpus.items():
            digest = content_hash(source)
            doc = enrich(source, cfg_doc, views=views)
            enriched_text = "\n".join(chunk(doc.text, cfg_chunk))

            df_counter.update(set(doc_tokens(enriched_text)))
            idents = json.dumps(sorted({i.lower() for i in doc.identifiers}))
            fts_rows.append((snippet_id, version, enriched_text))

            vkeys: dict[str, str] = {}
            for view_name, view_text in doc.views.items():
                pieces = [view_text] if view_name == "raw" else chunk(view_text, cfg_chunk)
                key = (vector_key(pieces), view_name)
                vkeys[view_name] = key[0]
                if key in cached:
                    reused += 1
                    continue
                if key in pending:
                    continue
                pending[key] = pieces
            rows.append((snippet_id, version, source, digest, enriched_text,
                         doc.category, doc.language, idents, json.dumps(vkeys)))

        stats.reused_vectors = reused

        if use_embeddings and pending:
            emb = emb_for_sig or Embedder()
            flat: list[str] = []
            spans: list[tuple[tuple[str, str], int, int]] = []
            for key, pieces in pending.items():
                spans.append((key, len(flat), len(flat) + len(pieces)))
                flat.extend(pieces)
            if show_progress:
                print(f"  encoding {len(flat)} text(s) for "
                      f"{len(pending)} (snippet, view) pair(s)...")
            mark = time.perf_counter()
            encoded = emb.encode(flat, show_progress=show_progress)
            stats.encode_ms = int((time.perf_counter() - mark) * 1000)
            stats.embedder = emb.name
            stats.text_cache_hits = int(getattr(emb, "last_cache_hits", 0) or 0)
            stats.model_encoded = int(getattr(emb, "last_model_encoded", len(flat)))

            to_store: list[tuple[str, str, bytes]] = []
            for (vkey, view_name), lo, hi in spans:
                window = encoded[lo:hi]
                vector = window[0] if len(window) == 1 else mean_pool(window)
                to_store.append((vkey, view_name, pack(vector)))
            conn.executemany(
                "INSERT OR REPLACE INTO vectors (hash, view, vector) VALUES (?,?,?)",
                to_store)
            stats.encoded_vectors = len(to_store)

        conn.executemany(
            "INSERT OR REPLACE INTO snippets "
            "(id, version, content, hash, enriched, category, language, identifiers, vkeys) "
            "VALUES (?,?,?,?,?,?,?,?,?)", rows)
        conn.executemany(
            "INSERT INTO search_text (snippet_id, version, enriched) VALUES (?,?,?)",
            fts_rows)
        conn.executemany("INSERT OR REPLACE INTO df (version, term, n) VALUES (?,?,?)",
                         [(version, term, n) for term, n in df_counter.items()])
        conn.execute("INSERT OR REPLACE INTO versions (version, built_at, snippets) "
                     "VALUES (?,?,?)", [version, time.time(), len(rows)])
        total = conn.execute("SELECT COUNT(*) FROM snippets").fetchone()[0]

        stats.snippets = len(rows)
        stats.elapsed_ms = int((time.perf_counter() - started) * 1000)
        conn.executemany("INSERT OR REPLACE INTO meta (key, value) VALUES (?,?)", [
            ("version", version),
            ("built_at", str(int(time.time()))),
            ("doc_count", str(total)),
            ("embedder_sig", signature),
            ("schema", SCHEMA_VERSION),
            ("corpus_kind", corpus_kind),
            ("views", json.dumps(list(views))),
            ("build_stats", json.dumps(stats.as_dict())),
            ("doc_config", json.dumps(cfg_doc.__dict__)),
            ("chunk_config", json.dumps(cfg_chunk.__dict__)),
        ])
        conn.commit()
        return stats
    finally:
        conn.close()


def load(db_path: Path, versions: str | list[str] | None = None) -> Index:
    conn = _open_readonly(db_path)
    row = conn.execute("SELECT value FROM meta WHERE key = 'doc_count'").fetchone()
    if not row:
        conn.close()
        raise IndexUnavailable(f"No index found at {db_path} (the file has no doc_count; "
                         "an interrupted or empty build). Rebuild it.")
    if _schema_of(conn) != SCHEMA_VERSION:
        conn.close()
        raise IndexUnavailable(f"{db_path} was built by an older debug907 (before the vector-key "
                         "and version fixes); its vectors cannot be trusted. Please "
                         "rebuild it: debug907 index / index-repo. Cached text vectors "
                         "make the rebuild fast.")

    available = [r[0] for r in conn.execute("SELECT version FROM versions ORDER BY built_at")]
    if versions is None:
        selected = available[-1:]
    elif versions == "all":
        selected = list(available)
    else:
        wanted = [versions] if isinstance(versions, str) else list(versions)
        unknown = [v for v in wanted if v not in available]
        if unknown:
            conn.close()
            raise IndexUnavailable(f"version {', '.join(unknown)} is not in {db_path}; "
                             f"it has: {', '.join(available) or 'none'}")
        selected = wanted

    kind = conn.execute("SELECT value FROM meta WHERE key = 'corpus_kind'").fetchone()
    index = Index(conn=conn, loaded_versions=selected, available_versions=available,
                  corpus_kind=kind[0] if kind else "")
    marks = ",".join("?" * len(selected))
    index.document_frequency = {
        r[0]: r[1] for r in conn.execute(
            f"SELECT term, SUM(n) FROM df WHERE version IN ({marks}) GROUP BY term", selected)}

    stored: dict[tuple[str, str], list[float]] = {
        (r["hash"], r["view"]): unpack(r["vector"])
        for r in conn.execute("SELECT hash, view, vector FROM vectors")}

    for r in conn.execute("SELECT id, version, hash, vkeys, category, identifiers, "
                          f"LENGTH(content) AS n FROM snippets WHERE version IN ({marks}) "
                          "ORDER BY rowid", selected):
        uid = index.uid(r["id"], r["version"])
        index.ids.append(uid)
        index.identity_of[uid] = r["id"]
        index.version_of[uid] = r["version"]
        index.hash_of[uid] = r["hash"]
        index.uids_of.setdefault(r["id"], []).append(uid)
        keys = json.loads(r["vkeys"] or "{}")
        index.vectors[uid] = {view: stored[(key, view)] for view, key in keys.items()
                              if (key, view) in stored}
        index.categories[uid] = r["category"] or "utility"
        index.lengths[uid] = r["n"] or 0
        try:
            index.identifiers[uid] = set(json.loads(r["identifiers"] or "[]"))
        except json.JSONDecodeError:
            index.identifiers[uid] = set()

    index.doc_count = len(index.ids)
    index.has_vectors = any(index.vectors.values())
    _build_matrices(index)
    return index


def _build_matrices(index: Index) -> None:
    if not index.has_vectors:
        return
    try:
        import numpy as np
    except ImportError:
        return
    by_view: dict[str, list[tuple[str, list[float]]]] = {}
    for snippet_id, views in index.vectors.items():
        for view_name, vector in views.items():
            by_view.setdefault(view_name, []).append((snippet_id, vector))
    for view_name, pairs in by_view.items():
        index.matrix_ids[view_name] = [sid for sid, _ in pairs]
        index.matrices[view_name] = np.asarray([v for _, v in pairs],
                                               dtype="float32")


__all__ = ["Index", "BuildStats", "IndexUnavailable", "build", "load", "connect",
           "content_hash",
           "doc_tokens", "EmbedderUnavailable"]
