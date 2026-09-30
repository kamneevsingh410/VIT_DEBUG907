from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

SHIPPED_SEQ = 1024
CODE = {"code": 1.0, "nl": 0.0}


def cosines(a: list[list[float]], b: list[list[float]]) -> np.ndarray:
    x, y = np.asarray(a, dtype="float32"), np.asarray(b, dtype="float32")
    return np.sum(x * y, axis=1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--n", type=int, default=100)
    args = parser.parse_args(argv)

    import transformers

    from bench.dataset import load_appsretrieval
    from bench.metrics import evaluate
    from bench.stratified import load as load_sample
    from pipeline.chunk import ChunkConfig, chunk
    from pipeline.doc_proc import DocConfig, enrich
    from retrieval.embed import MODELS, Embedder, TextCache, get_cache, set_threads
    from retrieval.index import content_hash, load as load_index
    from retrieval.rerank import RerankConfig
    from retrieval.search import SearchConfig, search

    print(f"transformers {transformers.__version__}   threads {set_threads(args.threads)}")
    name = MODELS["gte-modernbert"].name
    fresh = Embedder(name=name, max_seq_length=SHIPPED_SEQ, batch_size=16, cache=False)
    cached = Embedder(name=name, max_seq_length=SHIPPED_SEQ, batch_size=16)
    rng = random.Random(0)
    verdicts: dict[str, bool] = {}

    corpus, queries, qrels = load_appsretrieval()

    index = load_index(Path("out/real.db"))
    sig = index.conn.execute("SELECT value FROM meta WHERE key='embedder_sig'").fetchone()[0]
    print(f"\n[1] documents vs out/real.db  (index signature: {sig})")
    doc_ids = rng.sample(sorted(corpus), args.n)
    texts, stored = [], []
    for did in doc_ids:
        view = enrich(corpus[did], DocConfig(), views=("code",)).views["code"]
        pieces = chunk(view, ChunkConfig())
        if len(pieces) != 1:
            continue
        vec = index.vectors.get(did, {}).get("code")
        if vec is None:
            continue
        texts.append(pieces[0])
        stored.append(vec)
    index.close()
    t = time.perf_counter()
    cos = cosines(fresh.encode(texts), stored)
    print(f"    {len(texts)} docs   cos mean {cos.mean():.6f}  min {cos.min():.6f}"
          f"   ({time.perf_counter()-t:.0f}s)")
    verdicts["documents"] = bool(cos.min() > 0.9999)

    print("\n[2] queries vs text cache")
    sample_ids = sorted(load_sample())
    qtexts = [queries[q] for q in rng.sample(sample_ids, args.n)]
    store = get_cache()
    hits = store.get_many(cached.signature, [TextCache.key(q) for q in qtexts])
    qtexts = [q for q in qtexts if TextCache.key(q) in hits]
    t = time.perf_counter()
    cos = cosines(fresh.encode(qtexts), [hits[TextCache.key(q)] for q in qtexts])
    print(f"    {len(qtexts)} queries   cos mean {cos.mean():.6f}  min {cos.min():.6f}"
          f"   ({time.perf_counter()-t:.0f}s)")
    verdicts["queries"] = bool(len(qtexts) > 0 and cos.min() > 0.9999)

    print("\n[3] npm JS vectors vs text cache (encoded under transformers 4.53.3)")
    npm_lib = Path("C:/Program Files/nodejs/node_modules/npm/lib")
    if npm_lib.is_dir():
        from pipeline.repo import extract_repo
        js = list(extract_repo(npm_lib).values())
        rng.shuffle(js)
        js_hits = store.get_many(cached.signature, [TextCache.key(s) for s in js])
        js = [s for s in js if TextCache.key(s) in js_hits][: args.n // 2]
        if js:
            cos = cosines(fresh.encode(js), [js_hits[TextCache.key(s)] for s in js])
            print(f"    {len(js)} snippets   cos mean {cos.mean():.6f}  min {cos.min():.6f}")
            verdicts["npm_cache"] = bool(cos.min() > 0.9999)
        else:
            print("    no cached npm vectors found")
    else:
        print("    npm lib/ not found; skipped")

    print("\n[4] shipped config on the stratified sample")
    if not (verdicts["documents"] and verdicts["queries"]):
        print("    SKIPPED: fresh vectors do not match stored ones; the cached "
              "number would not reflect this environment.")
        return 1
    picked = set(sample_ids)
    sq = {q: t_ for q, t_ in queries.items() if q in picked}
    sr = {q: r for q, r in qrels.items() if q in picked}
    index = load_index(Path("out/real.db"))
    cfg = SearchConfig(use_sparse=False, rerank=RerankConfig.off(), view_weights=CODE)
    try:
        run = {q: search(index, t_, cfg, top_k=100, embedder=cached).ids
               for q, t_ in sq.items()}
    finally:
        index.close()
    scores = evaluate(run, sr)
    print(f"    {scores}")
    print(f"    expected NDCG@10 ~0.5092 (plain cosine, ablation row 'sequence length 1024'; "
          f"the shipped config embeds raw documents and adds the hubness correction: 0.6206)")

    print("\nverdicts:", verdicts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
