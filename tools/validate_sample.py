from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.dataset import load_appsretrieval
from bench.metrics import evaluate
from bench.significance import compare_all, report
from bench.stratified import build_sample, describe, save, validate
from retrieval.embed import Embedder, set_threads
from retrieval.index import load as load_index
from retrieval.rerank import RerankConfig
from retrieval.search import SearchConfig, search

RESULTS = Path("bench/results")

CONFIGS = {
    "dense+rerank": SearchConfig(use_sparse=False),
    "hybrid_0.15": SearchConfig(sparse_weight=0.15),
    "hybrid_0.3": SearchConfig(sparse_weight=0.3),
    "dense_only": SearchConfig(use_sparse=False, rerank=RerankConfig.off()),
}


def main() -> int:
    print(f"threads={set_threads()}")
    corpus, queries, qrels = load_appsretrieval()
    emb = Embedder(quantize=True, batch_size=32)

    picked = build_sample(queries, 1000, seed=0)
    print()
    print(describe(queries, picked))

    index = load_index(Path("out/real.db"))
    try:
        print(f"\nscoring {len(CONFIGS)} configs on the full split "
              f"({len(queries)} queries)...")
        runs_full: dict[str, dict[str, list[str]]] = {}
        full_scores: dict[str, float] = {}
        for name, cfg in CONFIGS.items():
            started = time.perf_counter()
            run = {qid: search(index, q, cfg, top_k=100, embedder=emb).ids
                   for qid, q in queries.items()}
            runs_full[name] = run
            s = evaluate(run, qrels)
            full_scores[name] = s.ndcg_at_10
            print(f"  {name:<16} NDCG@10 {s.ndcg_at_10:.4f}  "
                  f"({time.perf_counter()-started:.0f}s)")

        sample_set = set(picked)
        sample_qrels = {q: qrels[q] for q in picked if q in qrels}
        sample_scores = {
            name: evaluate({q: r for q, r in run.items() if q in sample_set},
                           sample_qrels).ndcg_at_10
            for name, run in runs_full.items()
        }
        print(f"\nsame configs on the {len(picked)}-query stratified sample:")
        for name, value in sample_scores.items():
            print(f"  {name:<16} NDCG@10 {value:.4f}  "
                  f"(delta {value - full_scores[name]:+.4f})")

        print("\n" + "=" * 62)
        sig = compare_all(runs_full, qrels, iterations=500)
        print(report(sig))

        separable = [(c.name_a, c.name_b) for c in sig["pairs"] if c.significant]
        result = validate(full_scores, sample_scores, separable_pairs=separable)
        print("\n" + "=" * 62)
        print(f"sample validation, judged on {len(separable)} separable pair(s):")
        for a, b in separable:
            print(f"    {a} > {b}")
        print(result)
        inverted = result.details["inverted_separable_pairs"]
        print("  inverted pairs:", inverted if inverted else "none")

        save(picked, result)
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / "significance.json").write_text(json.dumps({
            "marginals": {k: {"mean": v.mean, "low": v.low, "high": v.high}
                          for k, v in sig["marginals"].items()},
            "pairs": [{"a": c.name_a, "b": c.name_b, "diff": c.diff.mean,
                       "low": c.diff.low, "high": c.diff.high,
                       "p": c.p_value, "significant": c.significant}
                      for c in sig["pairs"]],
            "sample_validation": {
                "ranking_preserved": result.preserved,
                "kendall_tau": result.kendall_tau,
                "max_abs_delta": result.max_abs_delta,
            },
        }, indent=2), encoding="utf-8")
        print(f"\nwrote {RESULTS / 'significance.json'}")
    finally:
        index.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
