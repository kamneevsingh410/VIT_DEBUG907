from __future__ import annotations

import gzip
import hashlib
import json
import os
import random
import shutil
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "prebuilt_index.json"
FILE_NAME = "appsretrieval_index.db.gz"
DOWNLOAD_DIR = Path("out/downloads")
SHIPPED_VIEWS = ["raw"]
ALLOWED_TABLES = {"snippets", "vectors", "search_text", "search_text_data", "search_text_idx",
                  "search_text_content", "search_text_docsize", "search_text_config", "df",
                  "versions", "meta"}
CHUNK = 1 << 20


class PrebuiltError(Exception):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def load_manifest(path: Path = MANIFEST) -> dict:
    if not path.exists():
        raise PrebuiltError(f"no pinned manifest at {path}: this checkout does not ship a "
                            "pre-built index. Build it: debug907 index")
    return json.loads(path.read_text(encoding="utf-8"))


def _tables(conn: sqlite3.Connection) -> set[str]:
    return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _meta(conn: sqlite3.Connection) -> dict[str, str]:
    return dict(conn.execute("SELECT key, value FROM meta"))


def export_index(src: Path, out_file: Path) -> dict:
    src, out_file = Path(src), Path(out_file)
    if not src.exists():
        raise PrebuiltError(f"no index at {src}. Build it first: debug907 index")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_db = out_file.with_name(out_file.name + ".export.db")
    tmp_db.unlink(missing_ok=True)
    conn = sqlite3.connect(f"file:{src.resolve().as_posix()}?mode=ro", uri=True)
    try:
        conn.execute("VACUUM INTO ?", [str(tmp_db)])
    finally:
        conn.close()
    conn = sqlite3.connect(tmp_db)
    try:
        extra = _tables(conn) - ALLOWED_TABLES
        if extra:
            raise PrebuiltError(f"refusing to export: unexpected tables {sorted(extra)}")
        meta = _meta(conn)
        if json.loads(meta.get("views", "[]")) != SHIPPED_VIEWS or meta.get("corpus_kind") != "apps":
            raise PrebuiltError("refusing to export: not the shipped AppsRetrieval index "
                                f"(views {meta.get('views')}, kind {meta.get('corpus_kind')})")
        keep = set()
        for (vkeys,) in conn.execute("SELECT vkeys FROM snippets"):
            keep.update((h, v) for v, h in json.loads(vkeys or "{}").items())
        conn.execute("CREATE TEMP TABLE keep (hash TEXT, view TEXT, PRIMARY KEY (hash, view))")
        conn.executemany("INSERT INTO keep VALUES (?, ?)", sorted(keep))
        conn.execute("DELETE FROM vectors WHERE (hash, view) NOT IN (SELECT hash, view FROM keep)")
        conn.execute("DROP TABLE keep")
        conn.commit()
        snippets = conn.execute("SELECT COUNT(*) FROM snippets").fetchone()[0]
        vectors = conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
        conn.execute("VACUUM")
    finally:
        conn.close()
    with open(tmp_db, "rb") as raw, open(out_file, "wb") as dst:
        with gzip.GzipFile(filename="", mode="wb", fileobj=dst, mtime=0, compresslevel=9) as gz:
            shutil.copyfileobj(raw, gz, CHUNK)
    entry = {"file": out_file.name, "sha256": sha256_file(out_file), "size": out_file.stat().st_size,
             "db_sha256": sha256_file(tmp_db), "db_size": tmp_db.stat().st_size,
             "snippets": snippets, "vectors": vectors, "views": SHIPPED_VIEWS,
             "corpus_kind": "apps", "embedder_sig": meta.get("embedder_sig"),
             "contains": "the 8,765 public AppsRetrieval corpus snippets, their search text and "
                         "their vectors; no queries, relevance labels or rankings"}
    tmp_db.unlink()
    return entry


def fetch(source: str, manifest: dict, download_dir: Path = DOWNLOAD_DIR) -> tuple[Path, bool]:
    if "://" not in source:
        path = Path(source)
        if not path.is_file():
            raise PrebuiltError(f"not a file: {source}")
        return path, False
    if not source.lower().startswith("https://"):
        raise PrebuiltError("only https:// URLs are accepted")
    import urllib.request
    download_dir.mkdir(parents=True, exist_ok=True)
    target = download_dir / FILE_NAME
    partial = target.with_name(target.name + ".part")
    limit = int(manifest["size"])
    got = 0
    with urllib.request.urlopen(source, timeout=60) as resp, open(partial, "wb") as out:
        for block in iter(lambda: resp.read(CHUNK), b""):
            got += len(block)
            if got > limit:
                out.close()
                partial.unlink(missing_ok=True)
                raise PrebuiltError(f"download is larger than the pinned {limit:,} bytes: refused")
            out.write(block)
    os.replace(partial, target)
    return target, True


def import_index(source: str, dest: Path, *, expected_sig: str, manifest: dict | None = None,
                 force: bool = False, download_dir: Path = DOWNLOAD_DIR, say=print) -> dict:
    manifest = manifest or load_manifest()
    dest = Path(dest)
    if dest.exists() and not force:
        raise PrebuiltError(f"{dest} already exists (your own build?). Pass --force to replace it, "
                            "or choose another --db")
    path, downloaded = fetch(source, manifest, download_dir)
    try:
        digest = sha256_file(path)
        if digest != manifest["sha256"]:
            raise PrebuiltError(f"SHA-256 mismatch: the file is {digest[:16]}..., the repository "
                                f"pins {manifest['sha256'][:16]}... - refused, nothing was opened")
        say(f"  sha256    {digest}  (matches the pinned hash)")
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".importing")
        cap = int(manifest["db_size"])
        written = 0
        with gzip.open(path, "rb") as src, open(tmp, "wb") as out:
            for block in iter(lambda: src.read(CHUNK), b""):
                written += len(block)
                if written > cap:
                    out.close()
                    tmp.unlink(missing_ok=True)
                    raise PrebuiltError("decompressed size exceeds the pinned size: refused")
                out.write(block)
        if sha256_file(tmp) != manifest["db_sha256"]:
            tmp.unlink(missing_ok=True)
            raise PrebuiltError("the decompressed database does not match its pinned hash: refused")
        try:
            check_configuration(tmp, expected_sig, manifest)
        except PrebuiltError:
            tmp.unlink(missing_ok=True)
            raise
        os.replace(tmp, dest)
    finally:
        if downloaded:
            path.unlink(missing_ok=True)
    say(f"  installed {dest}  ({manifest['snippets']:,} snippets, view {manifest['views'][0]}, "
        f"{manifest['embedder_sig']})")
    return manifest


def check_configuration(db: Path, expected_sig: str, manifest: dict) -> None:
    conn = sqlite3.connect(f"file:{Path(db).resolve().as_posix()}?mode=ro", uri=True)
    try:
        extra = _tables(conn) - ALLOWED_TABLES
        if extra:
            raise PrebuiltError(f"unexpected tables {sorted(extra)}: refused")
        meta = _meta(conn)
        count = conn.execute("SELECT COUNT(*) FROM snippets").fetchone()[0]
    finally:
        conn.close()
    problems = []
    if meta.get("embedder_sig") != expected_sig:
        problems.append(f"encoder signature {meta.get('embedder_sig')!r}, this code uses {expected_sig!r}")
    if json.loads(meta.get("views", "[]")) != SHIPPED_VIEWS:
        problems.append(f"view {meta.get('views')}, the shipped view is {SHIPPED_VIEWS}")
    if meta.get("corpus_kind") != "apps":
        problems.append(f"corpus kind {meta.get('corpus_kind')!r}, expected 'apps'")
    if count != manifest.get("snippets"):
        problems.append(f"{count} snippets, expected {manifest.get('snippets')}")
    if problems:
        raise PrebuiltError("not the shipped configuration: " + "; ".join(problems))


def verify_index(db: Path, *, sample: int = 200, seed: int = 907, embedder=None,
                 say=print) -> dict:
    from pipeline.chunk import ChunkConfig, chunk
    from pipeline.doc_proc import DocConfig, enrich
    from retrieval.embed import Embedder, mean_pool, unpack
    from retrieval.index import vector_key

    conn = sqlite3.connect(f"file:{Path(db).resolve().as_posix()}?mode=ro", uri=True)
    try:
        meta = _meta(conn)
        views = tuple(json.loads(meta.get("views", "[]")))
        doc_cfg = DocConfig(**json.loads(meta.get("doc_config", "{}")))
        chunk_cfg = ChunkConfig(**json.loads(meta.get("chunk_config", "{}")))
        rows = conn.execute("SELECT id, version, content, vkeys FROM snippets ORDER BY id, version").fetchall()
        picked = rows if sample <= 0 or sample >= len(rows) else random.Random(seed).sample(rows, sample)
        stored = {}
        for sid, version, content, vkeys in picked:
            for view, key in json.loads(vkeys or "{}").items():
                blob = conn.execute("SELECT vector FROM vectors WHERE hash = ? AND view = ?",
                                    [key, view]).fetchone()
                stored[(sid, version, view)] = (key, unpack(blob[0]) if blob else None)
    finally:
        conn.close()
    emb = embedder or Embedder(cache=False)
    if getattr(emb, "signature", None) != meta.get("embedder_sig"):
        raise PrebuiltError(f"index encoder {meta.get('embedder_sig')!r} is not this code's "
                            f"{getattr(emb, 'signature', '?')!r}")
    jobs, key_mismatch, missing = [], [], []
    for sid, version, content, _ in picked:
        doc = enrich(content, doc_cfg, views=views)
        for view, text in doc.views.items():
            pieces = [text] if view == "raw" else chunk(text, chunk_cfg)
            key, vec = stored.get((sid, version, view), (None, None))
            if key != vector_key(pieces):
                key_mismatch.append(sid)
            elif vec is None:
                missing.append(sid)
            else:
                jobs.append((sid, pieces, vec))
    flat = [p for _, pieces, _ in jobs for p in pieces]
    say(f"  re-encoding {len(flat)} text(s) for {len(jobs)} snippet(s) from scratch (no cache) ...")
    encoded = emb.encode(flat) if flat else []
    cosines, at = [], 0
    worst = None
    for sid, pieces, vec in jobs:
        window = encoded[at:at + len(pieces)]
        at += len(pieces)
        fresh = window[0] if len(window) == 1 else mean_pool(window)
        c = sum(a * b for a, b in zip(fresh, vec))
        cosines.append(c)
        if worst is None or c < worst[1]:
            worst = (sid, c)
    ok = not key_mismatch and not missing and bool(cosines) and min(cosines) >= 0.9999
    return {"db": str(db), "checked": len(jobs), "sampled": len(picked),
            "min_cosine": min(cosines) if cosines else None,
            "mean_cosine": sum(cosines) / len(cosines) if cosines else None,
            "worst": worst[0] if worst else None, "key_mismatch": key_mismatch,
            "missing_vectors": missing, "ok": ok}
