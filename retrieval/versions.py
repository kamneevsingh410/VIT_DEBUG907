from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from pipeline.doc_proc import DocConfig
from retrieval.index import BuildStats, Index, build


@dataclass
class ReindexReport:
    full_ms: int
    incremental_ms: int
    total_snippets: int
    changed_snippets: int
    reused_vectors: int
    new_vectors: int
    views: int = 1
    full_encode_ms: int = 0
    incremental_encode_ms: int = 0

    @property
    def speedup(self) -> float:
        return self.full_ms / max(1, self.incremental_ms)

    @property
    def total_vectors(self) -> int:
        return self.total_snippets * max(1, self.views)

    def summary(self) -> str:
        pct = 100.0 * self.changed_snippets / max(1, self.total_snippets)
        lines = [
            f"full index     {self.full_ms} ms  ({self.total_snippets} snippets "
            f"x {self.views} views = {self.total_vectors} vectors)",
            f"incremental    {self.incremental_ms} ms  "
            f"({self.changed_snippets} changed, {pct:.1f}% of corpus)",
            f"vectors reused {self.reused_vectors} / {self.total_vectors}",
            f"re-encoded     {self.new_vectors} vector(s)",
            f"speedup        {self.speedup:.1f}x",
        ]
        if self.full_encode_ms == 0:
            lines.append(
                "note           encoding was served entirely from the text cache, "
                "so `full index` here excludes model time. Delete "
                "out/embed_cache.db for a cold-start number.")
        else:
            lines.append(
                f"encode time    full {self.full_encode_ms} ms  ->  "
                f"incremental {self.incremental_encode_ms} ms")
        return "\n".join(lines)


def reindex(corpus_v1: dict[str, str], corpus_v2: dict[str, str], db_path: Path,
            doc_config: DocConfig | None = None,
            embedder=None) -> ReindexReport:
    if db_path.exists():
        db_path.unlink()

    started = time.perf_counter()
    first: BuildStats = build(corpus_v1, db_path, version="v1", views=("code",),
                              doc_config=doc_config, embedder=embedder)
    full_ms = int((time.perf_counter() - started) * 1000)

    started = time.perf_counter()
    second: BuildStats = build(corpus_v2, db_path, version="v2", views=("code",),
                               doc_config=doc_config, embedder=embedder)
    incremental_ms = int((time.perf_counter() - started) * 1000)

    views = max(1, second.views)
    return ReindexReport(
        full_ms=full_ms,
        incremental_ms=incremental_ms,
        total_snippets=second.snippets,
        changed_snippets=second.encoded_vectors // views,
        reused_vectors=second.reused_vectors,
        new_vectors=second.encoded_vectors,
        views=views,
        full_encode_ms=first.encode_ms,
        incremental_encode_ms=second.encode_ms,
    )


def lineage(index: Index) -> tuple[dict[str, str], dict[str, list[str]]]:
    cached = getattr(index, "_lineage", None)
    if cached is not None:
        return cached
    per_version: dict[tuple[str, str], set[str]] = {}
    for uid, ident in index.identity_of.items():
        base = ident.rsplit("#", 1)[0]
        per_version.setdefault((index.version_of.get(uid, ""), base), set()).add(ident)
    ambiguous = {base for (_, base), idents in per_version.items() if len(idents) > 1}
    key_of: dict[str, str] = {}
    members: dict[str, list[str]] = {}
    for uid, ident in index.identity_of.items():
        base = ident.rsplit("#", 1)[0]
        key = ident if base in ambiguous else base
        key_of[uid] = key
        members.setdefault(key, []).append(uid)
    index._lineage = (key_of, members)
    return index._lineage


def collapse_versions(index: Index, hits: list[tuple[str, float]], top_k: int = 10,
                      prefer_newest: bool = False
                      ) -> tuple[list[tuple[str, float]], dict[str, list[str]]]:
    order = {v: i for i, v in enumerate(index.available_versions)}
    key_of, lineage_members = lineage(index)
    kept: list[tuple[str, float]] = []
    seen: set[str] = set()
    versions: dict[str, list[str]] = {}
    for uid, score in hits:
        key = key_of.get(uid, uid)
        if key in seen:
            continue
        seen.add(key)
        members = lineage_members.get(key, [uid])
        if prefer_newest:
            same = [u for u in members if index.hash_of.get(u) == index.hash_of.get(uid)]
            uid = max(same or [uid], key=lambda u: order.get(index.version_of.get(u, ""), 0))
        kept.append((uid, score))
        versions[uid] = sorted({index.version_of.get(u, "") for u in members},
                               key=lambda v: order.get(v, 0))
        if len(kept) >= top_k:
            break
    return kept, versions


@dataclass
class Appearance:
    version: str
    uid: str
    hash: str
    changed: bool


def evolution(index: Index, uid: str) -> list[Appearance]:
    order = {v: i for i, v in enumerate(index.available_versions)}
    key_of, lineage_members = lineage(index)
    members = sorted(lineage_members.get(key_of.get(uid, uid), [uid]),
                     key=lambda u: order.get(index.version_of.get(u, ""), 0))
    out: list[Appearance] = []
    for u in members:
        h = index.hash_of.get(u, "")
        out.append(Appearance(index.version_of.get(u, ""), u, h,
                              changed=bool(out) and h != out[-1].hash))
    return out


def evolution_label(appearances: list[Appearance]) -> str:
    if not appearances:
        return ""
    parts = [appearances[0].version]
    for a in appearances[1:]:
        parts.append(("!= " if a.changed else "= ") + a.version)
    return " ".join(parts)


def history_diffs(index: Index, uid: str, max_lines: int = 30
                  ) -> list[tuple[str, str, list[str]]]:
    import difflib
    appearances = evolution(index, uid)
    out = []
    for prev, cur in zip(appearances, appearances[1:]):
        if not cur.changed:
            continue
        diff = list(difflib.unified_diff(index.content(prev.uid).splitlines(),
                                         index.content(cur.uid).splitlines(),
                                         fromfile=prev.version, tofile=cur.version,
                                         n=2, lineterm=""))
        if len(diff) > max_lines:
            diff = diff[:max_lines] + [f"... {len(diff) - max_lines} more diff lines"]
        out.append((prev.version, cur.version, diff))
    return out
