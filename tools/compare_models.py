from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.dataset import load_appsretrieval
from bench.metrics import evaluate
from bench.significance import paired_bootstrap, per_query_scores
from bench.stratified import load as load_sample
from retrieval.embed import MODELS, Embedder, set_threads
from retrieval.index import build, load as load_index
from retrieval.rerank import RerankConfig
from retrieval.search import SearchConfig, search

RESULTS = Path("bench/results")
CODE_ONLY = {"code": 1.0, "nl": 0.0}


def evaluate_model(key: str, corpus: dict, queries: dict, qrels: dict,
                   quantize: bool, batch_size: int) -> dict:
    spec = MODELS[key]
    print(f"\n{'=' * 66}\n{key}  ({spec.name})\n  {spec.note}", flush=True)

    doc_emb = Embedder(name=spec.name, quantize=quantize,
                       batch_size=batch_size, prefix=spec.doc_prefix)
    query_emb = Embedder(name=spec.name, quantize=quantize,
                         batch_size=batch_size, prefix=spec.query_prefix)

    db = Path(f"out/model_{key}.db")
    started = time.perf_counter()
    stats = build(corpus, db, version="test", embedder=doc_emb,
                  views=("code",), show_progress=True)
    encode_s = time.perf_counter() - started
    print(f"  corpus encoded in {encode_s/60:.1f} min "
          f"({stats.encoded_vectors} new, {stats.reused_vectors} cached)",
          flush=True)

    t = time.perf_counter()
    query_emb.encode(list(queries.values()), show_progress=True)
    print(f"  queries encoded in {(time.perf_counter()-t)/60:.1f} min", flush=True)

    index = load_index(db)
    cfg = SearchConfig(use_sparse=False, rerank=RerankConfig.off(),
                       view_weights=CODE_ONLY)
    try:
        t = time.perf_counter()
        run = {qid: search(index, q, cfg, top_k=100, embedder=query_emb).ids
               for qid, q in queries.items()}
        search_s = time.perf_counter() - t
        scores = evaluate(run, qrels)
    finally:
        index.close()

    print(f"  {scores}", flush=True)
    return {
        "model": spec.name,
        "ndcg_at_10": scores.ndcg_at_10,
        "mrr": scores.mrr,
        "recall_at_100": scores.recall_at_100,
        "corpus_encode_s": round(encode_s, 1),
        "search_s": round(search_s, 1),
        "quantize": quantize,
        "run": run,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*",
                        default=["jina", "gte-modernbert", "granite-r2"])
    parser.add_argument("--quantize", action="store_true",
                        help="int8; NOT validated on real data, off by default")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--full", action="store_true",
                        help="full 3,765-query split instead of the sample")
    args = parser.parse_args(argv)

    print(f"threads={set_threads()}")
    corpus, queries, qrels = load_appsretrieval()
    if not args.full:
        picked = set(load_sample())
        queries = {q: t for q, t in queries.items() if q in picked}
        qrels = {q: r for q, r in qrels.items() if q in picked}
    print(f"corpus {len(corpus)}  queries {len(queries)}  "
          f"precision={'int8' if args.quantize else 'fp32'}")

    results: dict[str, dict] = {}
    for key in args.models:
        if key not in MODELS:
            print(f"unknown model {key!r}; known: {list(MODELS)}")
            continue
        try:
            results[key] = evaluate_model(key, corpus, queries, qrels,
                                          args.quantize, args.batch_size)
        except Exception:
            print(f"  {key} FAILED:\n{traceback.format_exc()}", flush=True)

    if not results:
        return 1

    print(f"\n{'=' * 66}\nSUMMARY  (dense only, code view, seq 512, "
          f"{'int8' if args.quantize else 'fp32'})")
    print(f"  {'model':<16}{'NDCG@10':>10}{'MRR':>9}{'R@100':>9}"
          f"{'encode':>10}{'search':>9}")
    order = sorted(results, key=lambda k: -results[k]["ndcg_at_10"])
    for key in order:
        r = results[key]
        print(f"  {key:<16}{r['ndcg_at_10']:>10.4f}{r['mrr']:>9.4f}"
              f"{r['recall_at_100']:>9.4f}{r['corpus_encode_s']/60:>9.1f}m"
              f"{r['search_s']:>8.0f}s")

    if "jina" in results and len(results) > 1:
        print("\npaired bootstrap against jina (500 resamples):")
        base = per_query_scores(results["jina"]["run"], qrels)
        for key in order:
            if key == "jina":
                continue
            other = per_query_scores(results[key]["run"], qrels)
            print(f"  {paired_bootstrap(other, base, key, 'jina', iterations=500)}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    payload = {k: {kk: vv for kk, vv in r.items() if kk != "run"}
               for k, r in results.items()}
    suffix = "full" if args.full else "sample1000"
    (RESULTS / f"model_comparison_{suffix}.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {RESULTS / f'model_comparison_{suffix}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
