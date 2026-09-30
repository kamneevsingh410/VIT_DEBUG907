from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ARTIFACT = Path(__file__).resolve().parent.parent / "appsretrieval_results.json"


def expected(path: Path = ARTIFACT) -> dict[str, float]:
    import json
    s = json.loads(Path(path).read_text(encoding="utf-8"))["scores"]["test"][0]
    return {"ndcg_at_10": round(s["ndcg_at_10"], 5), "mrr_at_10": round(s["mrr_at_10"], 5)}


TOLERANCE = 0.001


def read_rankings(path: Path) -> dict[str, list[str]]:
    ranked: dict[str, list[tuple[int, str]]] = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = {"query_id", "corpus_id", "rank"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing column(s) {sorted(missing)}")
        for line, row in enumerate(reader, start=2):
            try:
                rank = int(row["rank"])
            except (TypeError, ValueError):
                raise ValueError(f"{path} line {line}: rank {row['rank']!r} is not an integer") from None
            ranked.setdefault(row["query_id"], []).append((rank, row["corpus_id"]))
    return {q: [d for _, d in sorted(pairs)] for q, pairs in ranked.items()}


def score(run: dict[str, list[str]], qrels: dict[str, dict[str, int]], k: int = 10) -> dict:
    ndcg = mrr = 0.0
    for qid, rels in qrels.items():
        docs = run.get(qid, [])[:k]
        dcg = sum((2 ** rels.get(d, 0) - 1) / math.log2(i + 2) for i, d in enumerate(docs))
        ideal = sorted(rels.values(), reverse=True)[:k]
        idcg = sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(ideal))
        ndcg += dcg / idcg if idcg else 0.0
        first = next((i for i, d in enumerate(docs) if rels.get(d, 0) > 0), None)
        mrr += 1.0 / (first + 1) if first is not None else 0.0
    n = max(1, len(qrels))
    return {"ndcg_at_10": ndcg / n, "mrr_at_10": mrr / n, "queries": len(qrels)}


def recall(run: dict[str, list[str]], qrels: dict[str, dict[str, int]], k: int = 100) -> float:
    total = 0.0
    for qid, rels in qrels.items():
        relevant = {d for d, r in rels.items() if r > 0}
        if relevant:
            total += len(relevant & set(run.get(qid, [])[:k])) / len(relevant)
    return total / max(1, len(qrels))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="score a rankings CSV")
    parser.add_argument("csv", type=Path, nargs="?", default=Path("appsretrieval_rankings.csv"))
    parser.add_argument("--check", action="store_true",
                        help=f"exit 1 unless both numbers are within {TOLERANCE} of the artifact")
    args = parser.parse_args(argv)

    from bench.dataset import load_appsretrieval
    _corpus, _queries, qrels = load_appsretrieval()
    run = read_rankings(args.csv)
    got = score(run, qrels)
    print(f"file      {args.csv}  ({len(run):,} ranked queries, {got['queries']:,} judged)")
    ok = True
    for key, target in expected().items():
        delta = got[key] - target
        within = abs(delta) <= TOLERANCE
        ok &= within
        print(f"{key:<12}{got[key]:.5f}   artifact {target:.5f}   {delta:+.5f}  "
              f"{'OK' if within else 'MISMATCH'}")
    return 0 if ok or not args.check else 1


if __name__ == "__main__":
    raise SystemExit(main())
