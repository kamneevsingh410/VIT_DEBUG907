from __future__ import annotations

import argparse
import json
import random
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.dataset import load_appsretrieval
from bench.metrics import evaluate
from retrieval.embed import Embedder, set_threads
from retrieval.index import build, load as load_index
from retrieval.search import SearchConfig, search

RESULTS = Path("bench/results")
LOG = RESULTS / "batch_run_screening.log"
SAMPLE_N = 300


def log(message: str) -> None:
    stamp = time.strftime("%H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def sample(queries: dict, qrels: dict, n: int = SAMPLE_N) -> tuple[dict, dict]:
    random.seed(0)
    keys = random.sample(sorted(queries), min(n, len(queries)))
    return {k: queries[k] for k in keys}, {k: qrels[k] for k in keys}


def score(index, queries, qrels, cfg, emb, top_k: int = 100):
    run = {qid: search(index, q, cfg, top_k=top_k, embedder=emb).ids
           for qid, q in queries.items()}
    return evaluate(run, qrels)


def step1_pick_view(corpus, queries, qrels, emb) -> str:
    log("STEP 1  view selection (300-query sample, cached vectors)")
    q, r = sample(queries, qrels)
    emb.encode(list(q.values()))
    index = load_index(Path("out/real.db"))
    options = {
        "multi_view": None,
        "code_only": {"code": 1.0, "nl": 0.0},
        "nl_only": {"nl": 1.0, "code": 0.0},
    }
    out = {}
    try:
        for name, weights in options.items():
            s = score(index, q, r, SearchConfig(use_sparse=False,
                                                view_weights=weights), emb)
            out[name] = s.as_dict()
            log(f"  {name:<12} NDCG@10 {s.ndcg_at_10:.4f}  R@100 {s.recall_at_100:.4f}")
    finally:
        index.close()
    (RESULTS / "step1_views.json").write_text(json.dumps(out, indent=2))
    best = max(out, key=lambda k: out[k]["ndcg_at_10"])
    log(f"  -> best view config: {best}")
    return {"multi_view": "nl", "code_only": "code", "nl_only": "nl"}[best]


def step2_mteb(view: str) -> None:
    log(f"STEP 2  MTEB artifact (view={view}) -- the screening deliverable")
    import mteb
    from encoder import PrePostPipelineEncoder

    model = PrePostPipelineEncoder(view=view, raw_documents=False, hub_table=None)
    model.embedder.quantize = True
    model.embedder.batch_size = 32

    started = time.perf_counter()
    task = mteb.get_task("AppsRetrieval")
    result = mteb.evaluate(model, [task], overwrite_strategy="always")
    task_result = list(getattr(result, "task_results", result))[0]
    payload = (task_result.to_dict() if hasattr(task_result, "to_dict")
               else task_result)
    Path("appsretrieval_results.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8")
    log(f"  wrote appsretrieval_results.json in "
        f"{(time.perf_counter()-started)/60:.1f} min")
    scores = payload.get("scores", {}) if isinstance(payload, dict) else {}
    for split, entries in scores.items():
        for entry in entries if isinstance(entries, list) else [entries]:
            log(f"  {split}: ndcg@10={entry.get('ndcg_at_10')} "
                f"mrr@10={entry.get('mrr_at_10')}")
    log(f"  {json.dumps(payload, default=str)[:300]}")


def step3_full_eval(corpus, queries, qrels, emb) -> None:
    log("STEP 3  full test split, all configurations")
    t = time.perf_counter()
    emb.encode(list(queries.values()), show_progress=False)
    log(f"  encoded {len(queries)} queries in {(time.perf_counter()-t)/60:.1f} min")

    index = load_index(Path("out/real.db"))
    out = {}
    try:
        for name, cfg in [("track_a_dense", SearchConfig.track_a()),
                          ("dense_plus_rerank", SearchConfig(use_sparse=False)),
                          ("hybrid_0.15", SearchConfig(sparse_weight=0.15)),
                          ("hybrid_0.3", SearchConfig(sparse_weight=0.3)),
                          ("hybrid_0.5", SearchConfig(sparse_weight=0.5)),
                          ("bm25_only", SearchConfig.lexical_only())]:
            s = score(index, queries, qrels, cfg, emb)
            out[name] = s.as_dict()
            log(f"  {name:<20} NDCG@10 {s.ndcg_at_10:.4f}  MRR {s.mrr:.4f}  "
                f"R@100 {s.recall_at_100:.4f}")
    finally:
        index.close()
    (RESULTS / "full_eval.json").write_text(json.dumps(out, indent=2))
    best = max(out, key=lambda k: out[k]["ndcg_at_10"])
    log(f"  -> best overall: {best} at NDCG@10 {out[best]['ndcg_at_10']:.4f}")


def step4_seq_length(corpus, queries, qrels) -> None:
    log("STEP 4  sequence-length sweep (re-encodes the corpus, slow)")
    q, r = sample(queries, qrels)
    out = {}
    for seq in (1024, 512):
        emb = Embedder(quantize=True, batch_size=16, max_seq_length=seq)
        db = Path(f"out/real_seq{seq}.db")
        t = time.perf_counter()
        build(corpus, db, version="test", embedder=emb, show_progress=False)
        emb.encode(list(q.values()))
        log(f"  seq{seq}: indexed in {(time.perf_counter()-t)/60:.1f} min")
        index = load_index(db)
        try:
            s = score(index, q, r, SearchConfig.track_a(), emb)
            out[f"seq{seq}"] = s.as_dict()
            log(f"  seq{seq:<5} NDCG@10 {s.ndcg_at_10:.4f}  R@100 {s.recall_at_100:.4f}")
        finally:
            index.close()
    (RESULTS / "step4_seqlen.json").write_text(json.dumps(out, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip", nargs="*", type=int, default=[])
    args = parser.parse_args()

    log("=" * 64)
    log(f"unattended run starting, threads={set_threads()}")
    corpus, queries, qrels = load_appsretrieval()
    log(f"corpus {len(corpus)}  queries {len(queries)}")
    emb = Embedder(quantize=True, batch_size=32)

    view = "nl"
    steps = [
        (1, "view selection", lambda: step1_pick_view(corpus, queries, qrels, emb)),
        (2, "MTEB artifact", lambda: step2_mteb(view)),
        (3, "full evaluation", lambda: step3_full_eval(corpus, queries, qrels, emb)),
        (4, "sequence length", lambda: step4_seq_length(corpus, queries, qrels)),
    ]
    for number, name, fn in steps:
        if number in args.skip:
            log(f"STEP {number}  {name} -- skipped")
            continue
        try:
            started = time.perf_counter()
            got = fn()
            if number == 1 and isinstance(got, str):
                view = got
            log(f"STEP {number} done in {(time.perf_counter()-started)/60:.1f} min")
        except Exception:
            log(f"STEP {number} FAILED:\n{traceback.format_exc()}")

    log("unattended run complete")
    log("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
