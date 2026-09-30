from __future__ import annotations

from collections import Counter

import numpy as np


def _topk_mean(sims: np.ndarray, k: int) -> np.ndarray:
    k = max(1, min(k, sims.shape[1]))
    part = np.partition(sims, sims.shape[1] - k, axis=1)[:, -k:]
    return part.mean(axis=1)


def hub_scores(docs: np.ndarray, refs: np.ndarray, k: int,
               exclude: dict[int, set[int]] | None = None,
               chunk: int = 2048) -> np.ndarray:
    out = np.empty(len(docs), dtype="float32")
    for lo in range(0, len(docs), chunk):
        sims = docs[lo:lo + chunk] @ refs.T
        if exclude:
            for i in range(sims.shape[0]):
                for j in exclude.get(lo + i, ()):
                    sims[i, j] = -np.inf
        out[lo:lo + chunk] = _topk_mean(sims, k)
    return out


def doc_doc_hub(docs: np.ndarray, k: int, chunk: int = 2048) -> np.ndarray:
    out = np.empty(len(docs), dtype="float32")
    for lo in range(0, len(docs), chunk):
        sims = docs[lo:lo + chunk] @ docs.T
        for i in range(sims.shape[0]):
            sims[i, lo + i] = -np.inf
        out[lo:lo + chunk] = _topk_mean(sims, k)
    return out


def penalised(queries: np.ndarray, docs: np.ndarray, hub: np.ndarray,
              beta: float) -> np.ndarray:
    return queries @ docs.T - beta * hub[None, :]


def extend_queries(queries: np.ndarray) -> np.ndarray:
    return np.hstack([queries, np.ones((len(queries), 1), dtype=queries.dtype)])


def extend_documents(docs: np.ndarray, hub: np.ndarray, beta: float) -> np.ndarray:
    return np.hstack([docs, (-beta * hub)[:, None].astype(docs.dtype)])


def separation_auc(scores: np.ndarray, group: np.ndarray) -> float:
    pos, neg = np.sort(scores[group]), np.sort(scores[~group])
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    lo = np.searchsorted(neg, pos, side="left")
    hi = np.searchsorted(neg, pos, side="right")
    return float((lo + 0.5 * (hi - lo)).sum() / (len(pos) * len(neg)))


def hub_share(wrong_top1: list[str], min_times: int = 5) -> float:
    if not wrong_top1:
        return 0.0
    counts = Counter(wrong_top1)
    return sum(c for c in counts.values() if c >= min_times) / len(wrong_top1)


TABLE = "data/apps_hubness.json"


_TABLES: dict = {}


def load_table(path: str | None = None) -> dict | None:
    import json
    from pathlib import Path
    p = Path(path or TABLE)
    if not p.is_absolute() and not p.exists():
        p = Path(__file__).resolve().parent.parent / p
    if not p.exists():
        return None
    key = (str(p.resolve()), p.stat().st_mtime_ns)
    if key not in _TABLES:
        _TABLES.clear()
        _TABLES[key] = json.loads(p.read_text(encoding="utf-8"))
    return _TABLES[key]


class TableMismatch(ValueError):
    pass


def table_view(index, table: dict) -> str:
    view = table.get("document_text", "code")
    if view not in getattr(index, "matrices", {}):
        have = ", ".join(sorted(getattr(index, "matrices", {}))) or "none"
        raise TableMismatch(f"the hub table is for {view!r} document vectors but this index has "
                            f"{have}; rebuild it: debug907 index --db <this index>")
    return view


def index_bias(index, table: dict, view: str = "code"):
    import numpy as np
    cache = getattr(index, "_hub_bias", None)
    if cache is not None and cache[0] is table and cache[2] == view:
        return cache[1]
    by_id = table["by_id"]
    ids = index.matrix_ids[view]
    missing = [i for i in ids if index.identity_of.get(i, i) not in by_id]
    if missing:
        raise KeyError(f"{len(missing)} indexed documents are not in the hub table "
                       f"(first: {missing[0]})")
    bias = -table["beta"] * np.asarray([by_id[index.identity_of.get(i, i)] for i in ids],
                                       dtype="float32")
    index._hub_bias = (table, bias, view)
    return bias
