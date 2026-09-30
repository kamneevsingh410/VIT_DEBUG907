from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

RESULTS = Path("bench/results")
MODELS = (("gte-modernbert", 1024), ("granite-r2", 512), ("jina", 512))
BAND = (0.15, 0.35)
ITERATIONS = 2000


def ndcg(rank: int) -> float:
    return 1.0 / math.log2(rank + 1) if rank <= 10 else 0.0


def probe(key: str, seq: int, corpus: dict, variants: dict, sample: dict, gold: dict) -> dict:
    from pipeline.query_proc import process
    from retrieval.embed import MODELS as SPECS, Embedder
    from retrieval.index import build, load

    spec = SPECS[key]
    doc_emb = Embedder(name=spec.name, max_seq_length=seq, prefix=spec.doc_prefix)
    query_emb = Embedder(name=spec.name, max_seq_length=seq, prefix=spec.query_prefix)
    db = Path("out") / f"rename4b_{key}.db"
    if db.exists():
        db.unlink()
    t = time.perf_counter()
    first = build(corpus, db, version="original", views=("code",), embedder=doc_emb)
    second = build(variants, db, version="renamed", views=("code",), embedder=doc_emb)
    build_s = time.perf_counter() - t
    index = load(db, versions="all")
    try:
        mat, uids = index.matrices["code"], index.matrix_ids["code"]
        row = {(index.identity_of[u], index.version_of[u]): i for i, u in enumerate(uids)}
        ids = sorted(corpus)
        D = np.asarray(mat[[row[(d, "original")] for d in ids]], dtype="float32")
        V = {d: np.asarray(mat[row[(d, "renamed")]], dtype="float32") for d in variants}
    finally:
        index.close()
    col = {d: i for i, d in enumerate(ids)}
    qids = [q for q in sample if gold[q] in variants]
    Q = np.asarray(query_emb.encode([process(sample[q]).embedding_text or sample[q] for q in qids]),
                   dtype="float32")
    orig, ren = {}, {}
    for i, q in enumerate(qids):
        s = D @ Q[i]
        a = col[gold[q]]
        other = np.delete(s, a)
        orig[q] = ndcg(1 + int((other > s[a]).sum()))
        ren[q] = ndcg(1 + int((other > float(V[gold[q]] @ Q[i])).sum()))
    return {"model": spec.name, "seq": seq, "orig": orig, "renamed": ren,
            "build_seconds": round(build_s, 1),
            "texts_through_model": first.model_encoded + second.model_encoded}


def summary(orig: dict, ren: dict, qs: list[str]) -> dict:
    o = float(np.mean([orig[q] for q in qs]))
    r = float(np.mean([ren[q] for q in qs]))
    return {"queries": len(qs), "original": round(o, 4), "renamed": round(r, 4),
            "abs_drop": round(o - r, 4), "rel_drop": round((o - r) / o, 3) if o else None}


def compare_drops(a: dict, b: dict, qs: list[str], seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    ao = np.array([a["orig"][q] for q in qs]); ar = np.array([a["renamed"][q] for q in qs])
    bo = np.array([b["orig"][q] for q in qs]); br = np.array([b["renamed"][q] for q in qs])
    abs_d, rel_d = [], []
    n = len(qs)
    for _ in range(ITERATIONS):
        i = rng.integers(0, n, n)
        da, db_ = ao[i].mean() - ar[i].mean(), bo[i].mean() - br[i].mean()
        abs_d.append(da - db_)
        if ao[i].mean() > 0 and bo[i].mean() > 0:
            rel_d.append(da / ao[i].mean() - db_ / bo[i].mean())
    point_abs = (ao.mean() - ar.mean()) - (bo.mean() - br.mean())
    point_rel = ((ao.mean() - ar.mean()) / ao.mean() - (bo.mean() - br.mean()) / bo.mean()
                 if ao.mean() > 0 and bo.mean() > 0 else float("nan"))
    ci = lambda xs: (round(float(np.percentile(xs, 2.5)), 4), round(float(np.percentile(xs, 97.5)), 4))
    return {"abs_diff": round(float(point_abs), 4), "abs_ci": ci(abs_d),
            "rel_diff": round(float(point_rel), 3), "rel_ci": ci(rel_d)}


def main(argv: list[str] | None = None) -> int:
    from bench.dataset import load_appsretrieval
    from bench.stratified import load as load_stratified
    from retrieval.embed import set_threads
    from tools.evolutionary import rename_variables
    from tools.memorisation_probe import changed_fraction

    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args(argv)
    if args.render:
        r = json.loads((RESULTS / "rename_models.json").read_text(encoding="utf-8"))
        (RESULTS / "rename_models.md").write_text(render(r), encoding="utf-8")
        return 0
    set_threads(args.threads)
    started = time.perf_counter()
    corpus, queries, qrels = load_appsretrieval()
    picked = set(load_stratified())
    sample = {q: t for q, t in queries.items() if q in picked and q in qrels}
    gold = {q: next(iter(qrels[q])) for q in sample}
    variants, touched = {}, {}
    for d in sorted(set(gold.values())):
        new = rename_variables(corpus[d], style="short")
        if new and new != corpus[d]:
            variants[d] = new
            touched[d] = changed_fraction(corpus[d], new)

    per_model = {}
    for key, seq in MODELS:
        print(f"== {key} (seq {seq})", flush=True)
        per_model[key] = probe(key, seq, corpus, variants, sample, gold)
    qs_all = [q for q in sample if gold[q] in variants]
    qs_band = [q for q in qs_all if BAND[0] <= touched[gold[q]] <= BAND[1]]
    report = {"variants": len(variants), "band": BAND, "models": {}, "gte_vs": {},
              "seconds": None}
    for key, _ in MODELS:
        m = per_model[key]
        report["models"][key] = {"model": m["model"], "seq": m["seq"],
                                 "texts_through_model": m["texts_through_model"],
                                 "all": summary(m["orig"], m["renamed"], qs_all),
                                 "matched": summary(m["orig"], m["renamed"], qs_band)}
    for key, _ in MODELS[1:]:
        report["gte_vs"][key] = {"all": compare_drops(per_model["gte-modernbert"], per_model[key], qs_all),
                                 "matched": compare_drops(per_model["gte-modernbert"], per_model[key], qs_band)}
    report["seconds"] = round(time.perf_counter() - started, 1)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "rename_models.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    (RESULTS / "rename_models.md").write_text(render(report), encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 0


def render(r: dict) -> str:
    lines = ["# Rename probe across three encoders", "",
             f"The identical probe on each model: the same {r['variants']} renamed gold solutions "
             "(style-preserving short names), the same validated 1,000-query sample, plain cosine, "
             "code view, each model at its ablation-row sequence length. Only the renamed documents "
             "are new encodes. `tools/rename_models.py`.", "",
             "| model | seq | queries | NDCG@10 original | renamed | absolute drop | relative drop |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for part, label in (("all", "all renamable"), ("matched", "dose-matched")):
        for key, m in r["models"].items():
            s = m[part]
            lines.append(f"| {key} ({label}) | {m['seq']} | {s['queries']} | {s['original']:.4f} | "
                         f"{s['renamed']:.4f} | {s['abs_drop']:+.4f} | "
                         + (f"{s['rel_drop']:.0%}" if s["rel_drop"] is not None else "-") + " |")
    lines += ["", "## gte's drop minus each other model's drop (paired bootstrap, 95% CI)", "",
              "| vs | queries | absolute | relative |", "|---|---|---|---|"]
    for key, parts in r["gte_vs"].items():
        for part, c in parts.items():
            lines.append(f"| {key} ({part}) | {r['models'][key][part]['queries']} | "
                         f"{c['abs_diff']:+.4f} [{c['abs_ci'][0]:+.4f}, {c['abs_ci'][1]:+.4f}] | "
                         f"{c['rel_diff']:+.3f} [{c['rel_ci'][0]:+.3f}, {c['rel_ci'][1]:+.3f}] |")
    lines += ["", f"Total {r['seconds']} s.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
