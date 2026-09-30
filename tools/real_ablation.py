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
from pipeline.doc_proc import DocConfig
from pipeline.query_proc import QueryConfig
from retrieval.embed import MODELS, Embedder, set_threads
from retrieval.index import build, load as load_index
from retrieval.rerank import RerankConfig
from retrieval.search import SearchConfig, search

RESULTS = Path("bench/results")
CODE = {"code": 1.0, "nl": 0.0}
NL = {"nl": 1.0, "code": 0.0}
OFF = RerankConfig.off()


def rows_spec(slow: bool) -> list[dict]:
    gte, jina, gran = "gte-modernbert", "jina", "granite-r2"
    base = dict(model=gte, quantize=False, seq=512, views=("code",),
                doc=DocConfig(), search=SearchConfig(use_sparse=False,
                                                     rerank=OFF, view_weights=CODE))
    spec: list[dict] = [
        dict(base, label="BM25 only, no encoder", group="control",
             search=SearchConfig(use_dense=False, rerank=OFF,
                                 query=QueryConfig.baseline()),
             doc=DocConfig.baseline()),
        dict(base, label="BM25 only, enriched + preprocessed", group="control",
             search=SearchConfig(use_dense=False, rerank=OFF)),

        dict(base, label="granite-embedding-english-r2, fp32", group="encoder",
             model=gran),
        dict(base, label="jina-v2-base-code, int8", group="encoder",
             model=jina, quantize=True),
        dict(base, label="jina-v2-base-code, fp32", group="encoder", model=jina),
        dict(base, label="gte-modernbert-base, fp32  [SHIPPED]", group="encoder"),

        dict(base, label="+ interaction rerank", group="pipeline",
             search=SearchConfig(use_sparse=False, view_weights=CODE)),
        dict(base, label="+ BM25 fusion (RRF, w=0.15)", group="pipeline",
             search=SearchConfig(sparse_weight=0.15, view_weights=CODE)),
        dict(base, label="+ BM25 fusion (RRF, w=0.30)", group="pipeline",
             search=SearchConfig(sparse_weight=0.3, view_weights=CODE)),
        dict(base, label="+ rerank, candidate pool 500", group="pipeline",
             search=SearchConfig(use_sparse=False, view_weights=CODE,
                                 shortlist=500)),

        dict(base, label="query: strip -----Input----- sections", group="query",
             search=SearchConfig(use_sparse=False, rerank=OFF, view_weights=CODE,
                                 query=QueryConfig(strip_boilerplate=True))),
        dict(base, label="query: embed processed tokens, not raw", group="query",
             search=SearchConfig(use_sparse=False, rerank=OFF, view_weights=CODE,
                                 embed_raw_query=False)),
    ]
    if slow:
        spec += [
            dict(base, label="document view: nl instead of code  [SLOW]",
                 group="document", views=("nl", "code"),
                 search=SearchConfig(use_sparse=False, rerank=OFF,
                                     view_weights=NL)),
            dict(base, label="document view: multi-view max  [SLOW]",
                 group="document", views=("nl", "code"),
                 search=SearchConfig(use_sparse=False, rerank=OFF)),
            dict(base, label="sequence length 1024  [SLOW]", group="document",
                 seq=1024),
        ]
    return spec


def run_row(row: dict, corpus, queries, qrels) -> tuple[dict, dict]:
    spec = MODELS[row["model"]]
    emb = Embedder(name=spec.name, quantize=row["quantize"],
                   max_seq_length=row["seq"], batch_size=32,
                   prefix=spec.doc_prefix)
    tag = f"{row['model']}_{'int8' if row['quantize'] else 'fp32'}_s{row['seq']}"
    db = Path(f"out/abl_{tag}_{'-'.join(row['views'])}.db")

    needs_vectors = row["search"].use_dense
    started = time.perf_counter()
    build(corpus, db, version="test", embedder=emb, views=row["views"],
          doc_config=row["doc"], show_progress=False,
          use_embeddings=needs_vectors, corpus_kind="apps")
    if needs_vectors:
        emb.encode(list(queries.values()))
    encode_s = time.perf_counter() - started

    index = load_index(db)
    try:
        run = {qid: search(index, q, row["search"], top_k=100, embedder=emb).ids
               for qid, q in queries.items()}
        scores = evaluate(run, qrels)
    finally:
        index.close()
    return ({"label": row["label"], "group": row["group"],
             **scores.as_dict(), "setup_s": round(encode_s, 1)}, run)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slow", action="store_true",
                        help="include rows that need a fresh encoder pass")
    parser.add_argument("--threads", type=int, default=8,
                        help="CPU threads (8 by day; the machine is shared)")
    args = parser.parse_args(argv)

    print(f"threads={set_threads(args.threads)}")
    corpus, queries, qrels = load_appsretrieval()
    picked = set(load_sample())
    queries = {q: t for q, t in queries.items() if q in picked}
    qrels = {q: r for q, r in qrels.items() if q in picked}
    print(f"corpus {len(corpus)}  queries {len(queries)} (stratified sample)\n")

    rows, runs = [], {}
    for row in rows_spec(args.slow):
        try:
            result, run = run_row(row, corpus, queries, qrels)
            rows.append(result)
            runs[result["label"]] = run
            print(f"  {result['label']:<44} NDCG@10 {result['ndcg_at_10']:.4f}  "
                  f"R@100 {result['recall_at_100']:.4f}", flush=True)
        except Exception:
            print(f"  {row['label']} FAILED:\n{traceback.format_exc()}", flush=True)

    shipped = next((r["label"] for r in rows if "SHIPPED" in r["label"]), None)
    pairs = []
    if shipped and shipped in runs:
        base = per_query_scores(runs[shipped], qrels)
        for row in rows:
            if row["label"] == shipped:
                continue
            cmp = paired_bootstrap(per_query_scores(runs[row["label"]], qrels),
                                   base, row["label"], "SHIPPED", iterations=500)
            pairs.append({"label": row["label"], "diff": cmp.diff.mean,
                          "low": cmp.diff.low, "high": cmp.diff.high,
                          "p": cmp.p_value, "significant": cmp.significant})

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "ablation_real.json").write_text(
        json.dumps({"rows": rows, "vs_shipped": pairs}, indent=2), encoding="utf-8")
    write_markdown(rows, pairs, len(queries))
    print(f"\nwrote {RESULTS / 'ablation.md'}")
    return 0


GROUP_TITLES = {
    "control": "Controls: no encoder",
    "encoder": "Embedding model (guideline: which embedding model)",
    "pipeline": "Multiple retrieval passes (axis 5)",
    "query": "Query pre-processing (axes 1-2)",
    "document": "Document pre/post-processing (axes 3-4)",
}


def write_markdown(rows: list[dict], pairs: list[dict], n_queries: int) -> None:
    by_label = {p["label"]: p for p in pairs}
    lines = [
        "# Ablation table",
        "",
        f"CoIR AppsRetrieval, full 8,765-snippet corpus, {n_queries}-query "
        "stratified sample (validated rank-preserving against the full split).",
        "Base configuration: **gte-modernbert-base, fp32, seq 512, code view, "
        "dense only**: the shipped system.",
        "Deltas are paired bootstrap against the shipped row, 500 resamples; "
        "**bold** = significant at 95%.",
        "",
    ]
    for group, title in GROUP_TITLES.items():
        group_rows = [r for r in rows if r["group"] == group]
        if not group_rows:
            continue
        lines += [f"## {title}", "",
                  "| Variant | NDCG@10 | MRR | R@100 | vs shipped |",
                  "|---|---:|---:|---:|---|"]
        for r in sorted(group_rows, key=lambda x: -x["ndcg_at_10"]):
            cmp = by_label.get(r["label"])
            if cmp is None:
                delta = "(this is the baseline)"
            else:
                mark = "**" if cmp["significant"] else ""
                delta = (f"{mark}{cmp['diff']:+.4f}{mark} "
                         f"[{cmp['low']:+.4f}, {cmp['high']:+.4f}] p~{cmp['p']:.3f}")
            lines.append(f"| {r['label']} | {r['ndcg_at_10']:.4f} | {r['mrr']:.4f} "
                         f"| {r['recall_at_100']:.4f} | {delta} |")
        lines.append("")
    (RESULTS / "ablation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
