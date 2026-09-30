from __future__ import annotations

import re
import sqlite3

from retrieval.index import Index

SAFE_TERM = re.compile(r"^[a-z0-9_]{2,}$")


def build_match_query(terms: list[str], limit_terms: int = 64) -> str:
    safe = [t for t in dict.fromkeys(terms) if SAFE_TERM.match(t)][:limit_terms]
    return " OR ".join(f'"{t}"' for t in safe)


def search(index: Index, terms: list[str], top_k: int = 100) -> list[tuple[str, float]]:
    match = build_match_query(terms)
    if not match:
        return []
    versions = index.loaded_versions
    marks = ",".join("?" * len(versions))
    sql = (
        "SELECT snippet_id, version, bm25(search_text) AS rank "
        f"FROM search_text WHERE search_text MATCH ? AND version IN ({marks}) "
        "ORDER BY rank LIMIT ?"
    )
    try:
        rows = index.conn.execute(sql, [match, *versions, top_k]).fetchall()
    except sqlite3.OperationalError:
        return []
    return [(index.uid(row["snippet_id"], row["version"]), -float(row["rank"]))
            for row in rows]
