from __future__ import annotations

import math
from dataclasses import dataclass


def dcg(relevances: list[int]) -> float:
    return sum((2 ** rel - 1) / math.log2(rank + 2)
               for rank, rel in enumerate(relevances))


def ndcg_at_k(ranked_ids: list[str], relevant: dict[str, int], k: int = 10) -> float:
    gains = [relevant.get(doc_id, 0) for doc_id in ranked_ids[:k]]
    ideal = sorted(relevant.values(), reverse=True)[:k]
    idcg = dcg(ideal)
    if idcg == 0.0:
        return 0.0
    return dcg(gains) / idcg


def reciprocal_rank(ranked_ids: list[str], relevant: dict[str, int],
                    k: int | None = None) -> float:
    ids = ranked_ids if k is None else ranked_ids[:k]
    for rank, doc_id in enumerate(ids, start=1):
        if relevant.get(doc_id, 0) > 0:
            return 1.0 / rank
    return 0.0


def recall_at_k(ranked_ids: list[str], relevant: dict[str, int], k: int = 100) -> float:
    total = sum(1 for rel in relevant.values() if rel > 0)
    if not total:
        return 0.0
    found = sum(1 for doc_id in ranked_ids[:k] if relevant.get(doc_id, 0) > 0)
    return found / total


@dataclass
class Scores:
    ndcg_at_10: float
    mrr: float
    recall_at_100: float
    queries: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "ndcg_at_10": round(self.ndcg_at_10, 5),
            "mrr": round(self.mrr, 5),
            "recall_at_100": round(self.recall_at_100, 5),
            "queries": self.queries,
        }

    def __str__(self) -> str:
        return (f"NDCG@10 {self.ndcg_at_10:.4f}  MRR {self.mrr:.4f}  "
                f"R@100 {self.recall_at_100:.4f}  ({self.queries} queries)")


def evaluate(run: dict[str, list[str]], qrels: dict[str, dict[str, int]],
             k: int = 10) -> Scores:
    ndcgs: list[float] = []
    rrs: list[float] = []
    recalls: list[float] = []
    for query_id, relevant in qrels.items():
        ranked = run.get(query_id, [])
        ndcgs.append(ndcg_at_k(ranked, relevant, k))
        rrs.append(reciprocal_rank(ranked, relevant))
        recalls.append(recall_at_k(ranked, relevant, 100))
    count = max(1, len(ndcgs))
    return Scores(
        ndcg_at_10=sum(ndcgs) / count,
        mrr=sum(rrs) / count,
        recall_at_100=sum(recalls) / count,
        queries=len(qrels),
    )
