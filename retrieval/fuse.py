from __future__ import annotations

Ranked = list[tuple[str, float]]


def rrf(lists: list[Ranked], k: int = 60, weights: list[float] | None = None) -> Ranked:
    if weights is None:
        weights = [1.0] * len(lists)
    if len(weights) != len(lists):
        raise ValueError("weights must match the number of lists")

    scores: dict[str, float] = {}
    for ranked, weight in zip(lists, weights):
        for rank, (doc_id, _score) in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
