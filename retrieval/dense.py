from __future__ import annotations

from retrieval.embed import Embedder, cosine
from retrieval.index import Index


def search_vector(index: Index, query_vec: list[float], top_k: int = 100,
                  view_weights: dict[str, float] | None = None, bias: dict | None = None
                  ) -> list[tuple[str, float]]:
    if not query_vec:
        return []
    if index.matrices:
        return _search_numpy(index, query_vec, top_k, view_weights, bias)
    if bias:
        raise ValueError("a score bias needs numpy")
    return _search_python(index, query_vec, top_k, view_weights)


def _search_numpy(index: Index, query_vec: list[float], top_k: int,
                  view_weights: dict[str, float] | None,
                  bias: dict | None = None) -> list[tuple[str, float]]:
    import numpy as np

    if not query_vec:
        return []
    weights = view_weights or {}
    best: dict[str, float] = {}
    q = np.asarray(query_vec, dtype="float32")
    floor = float("-inf") if bias else 0.0

    for view_name, matrix in index.matrices.items():
        weight = weights.get(view_name, 1.0)
        if weight == 0.0:
            continue
        scores = (matrix @ q) * weight
        if bias and view_name in bias:
            scores = scores + bias[view_name]
        ids = index.matrix_ids[view_name]
        if len(scores) > top_k:
            cut = np.argpartition(-scores, top_k)[:top_k]
        else:
            cut = np.arange(len(scores))
        for i in cut:
            score = float(scores[i])
            if score <= floor:
                continue
            sid = ids[i]
            if score > best.get(sid, floor):
                best[sid] = score

    ranked = sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))
    return ranked[:top_k]


def _search_python(index: Index, query_vec: list[float], top_k: int,
                   view_weights: dict[str, float] | None) -> list[tuple[str, float]]:
    weights = view_weights or {}
    scores: dict[str, float] = {}
    for snippet_id, views in index.vectors.items():
        best = 0.0
        for view_name, vector in views.items():
            score = cosine(query_vec, vector) * weights.get(view_name, 1.0)
            if score > best:
                best = score
        if best > 0.0:
            scores[snippet_id] = best
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]


def search(index: Index, query_text: str, top_k: int = 100,
           embedder: Embedder | None = None,
           view_weights: dict[str, float] | None = None
           ) -> list[tuple[str, float]]:
    emb = embedder or Embedder()
    return search_vector(index, emb.encode_one(query_text), top_k, view_weights)


def best_view(index: Index, query_vec: list[float], snippet_id: str) -> tuple[str, float]:
    best_name, best_score = "", 0.0
    for view_name, vector in index.vectors.get(snippet_id, {}).items():
        score = cosine(query_vec, vector)
        if score > best_score:
            best_name, best_score = view_name, score
    return best_name, best_score
