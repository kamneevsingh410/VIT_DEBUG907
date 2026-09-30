from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def convert(predictions: Path, out: Path, top_k: int = 100,
            subset: str = "default", split: str = "test") -> int:
    data = json.loads(Path(predictions).read_text(encoding="utf-8"))
    run: dict[str, dict[str, float]] = data[subset][split]
    rows = 0
    with Path(out).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query_id", "corpus_id", "rank", "score"])
        for qid in sorted(run):
            ranked = sorted(run[qid].items(), key=lambda kv: (kv[1], kv[0]),
                            reverse=True)[:top_k]
            for rank, (doc_id, score) in enumerate(ranked, start=1):
                writer.writerow([qid, doc_id, rank, f"{score:.6f}"])
                rows += 1
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MTEB's native per-query predictions -> appsretrieval_rankings.csv.")
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--out", type=Path, default=Path("appsretrieval_rankings.csv"))
    parser.add_argument("--top-k", type=int, default=100)
    args = parser.parse_args(argv)
    rows = convert(args.predictions, args.out, args.top_k)
    print(f"wrote {args.out}  ({rows:,} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
