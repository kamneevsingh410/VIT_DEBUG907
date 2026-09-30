from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from retrieval import hubness
from tools.hubness_eval import compare, hub_stats, load_train, runs_from_scores, score_run

RESULTS = Path("bench/results")


def normalise(m: np.ndarray) -> np.ndarray:
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("raw", "canonical"), required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args(argv)
    out_json = RESULTS / f"doc_variant_{args.variant}.json"
    if args.render:
        r = json.loads(out_json.read_text(encoding="utf-8"))
        (RESULTS / f"doc_variant_{args.variant}.md").write_text(render(r), encoding="utf-8")
        return 0
    from bench.dataset import load_appsretrieval
    from bench.stratified import load as load_stratified
    from encoder import PrePostPipelineEncoder
    from retrieval.embed import Embedder, set_threads
    from retrieval.index import load
    from tools.evolutionary import rename_variables

    set_threads(args.threads)
    started = time.perf_counter()
    corpus, queries, qrels = load_appsretrieval()
    index = load(Path("out/real.db"))
    D = np.asarray(index.matrices["code"], dtype="float32")
    ids = list(index.matrix_ids["code"])
    index.close()
    emb = Embedder()
    enc = PrePostPipelineEncoder(raw_documents=False, hub_table=None)

    def canonical_text(src: str) -> str:
        return enc.process_document(rename_variables(src, style="var") or src)

    t = time.perf_counter()
    if args.variant == "raw":
        texts = [corpus[d].strip() for d in ids]
        D_new = normalise(np.asarray(emb.encode(texts, show_progress=True), dtype="float32"))
    else:
        texts = [canonical_text(corpus[d]) for d in ids]
        C = np.asarray(emb.encode(texts, show_progress=True), dtype="float32")
        D_new = normalise(D + C)
    encode_s = time.perf_counter() - t
    changed = sum(1 for d, x in zip(ids, texts) if x != enc.process_document(corpus[d]))

    table = hubness.load_table()
    row = {d: i for i, d in enumerate(ids)}
    train_texts, train_gold = load_train()
    train_ids = sorted(train_texts)
    T = np.asarray(emb.encode([train_texts[i] for i in train_ids]), dtype="float32")
    exclude: dict[int, set[int]] = {}
    for j, qid in enumerate(train_ids):
        if train_gold.get(qid) in row:
            exclude.setdefault(row[train_gold[qid]], set()).add(j)
    hub_old = np.asarray([table["by_id"][d] for d in ids], dtype="float32")
    hub_new = hubness.hub_scores(D_new, T, table["k"], exclude=exclude)
    beta = table["beta"]

    test = sorted(q for q in queries if q in qrels)
    Q = np.asarray(emb.encode([queries[q] for q in test]), dtype="float32")
    runs = {"plain": runs_from_scores(Q @ D.T, test, ids),
            "shipped": runs_from_scores(Q @ D.T - beta * hub_old[None, :], test, ids),
            f"{args.variant}": runs_from_scores(Q @ D_new.T, test, ids),
            f"{args.variant} + hubness": runs_from_scores(Q @ D_new.T - beta * hub_new[None, :],
                                                          test, ids)}
    sample = set(load_stratified())
    groups = {"sample": [q for q in test if q in sample], "full": test,
              "held-out": [q for q in test if q not in sample]}
    table_rows = {}
    for g, qs in groups.items():
        qr = {q: qrels[q] for q in qs}
        sub = lambda name: {q: runs[name][q] for q in qs}
        table_rows[g] = {"queries": len(qs),
                         **{name: {**score_run(sub(name), qr), **hub_stats(sub(name), qr)}
                            for name in runs},
                         "variant_vs_plain": compare(sub(args.variant), sub("plain"), qr),
                         "variant_hub_vs_shipped": compare(sub(f"{args.variant} + hubness"),
                                                           sub("shipped"), qr)}
    report = {"variant": args.variant, "documents": len(ids), "texts_changed": changed,
              "encode_seconds": round(encode_s, 1), "groups": table_rows}
    if args.variant == "canonical":
        report["rename_probe"] = rename_probe(corpus, queries, qrels, ids, D, D_new, emb, enc,
                                              canonical_text)
    report["seconds"] = round(time.perf_counter() - started, 1)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=1), encoding="utf-8")
    (RESULTS / f"doc_variant_{args.variant}.md").write_text(render(report), encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 0


def rename_probe(corpus, queries, qrels, ids, D, D_new, emb, enc, canonical_text) -> dict:
    from bench.stratified import load as load_stratified
    from tools.evolutionary import rename_variables
    from tools.memorisation_probe import ndcg

    picked = set(load_stratified())
    qids = sorted(q for q in queries if q in picked and q in qrels)
    col = {d: i for i, d in enumerate(ids)}
    gold = {q: next(iter(qrels[q])) for q in qids}
    renamed = {}
    for d in sorted(set(gold.values())):
        new = rename_variables(corpus[d], style="short")
        if new and new != corpus[d]:
            renamed[d] = new
    golds = sorted(renamed)
    V_code = np.asarray(emb.encode([enc.process_document(renamed[d]) for d in golds]), dtype="float32")
    V_can = np.asarray(emb.encode([canonical_text(renamed[d]) for d in golds]), dtype="float32")
    V_both = normalise(V_code + V_can)
    vrow = {d: i for i, d in enumerate(golds)}
    qs = [q for q in qids if gold[q] in renamed]
    Q = np.asarray(emb.encode([queries[q] for q in qs]), dtype="float32")
    out = {}
    for name, M, V in (("shipped code view", D, V_code), ("with canonical view", D_new, V_both)):
        o, r = [], []
        for i, q in enumerate(qs):
            s = M @ Q[i]
            a = col[gold[q]]
            other = np.delete(s, a)
            o.append(ndcg(1 + int((other > s[a]).sum())))
            r.append(ndcg(1 + int((other > float(V[vrow[gold[q]]] @ Q[i])).sum())))
        mo, mr = float(np.mean(o)), float(np.mean(r))
        out[name] = {"queries": len(qs), "original": round(mo, 4), "renamed": round(mr, 4),
                     "abs_drop": round(mo - mr, 4), "rel_drop": round((mo - mr) / mo, 3)}
    return out


def render(r: dict) -> str:
    v = r["variant"]
    title = {"raw": "Raw document text",
             "canonical": "Canonical view: identifiers renamed, averaged with the code view"}[v]
    lines = [f"# {title}", "",
             f"`tools/doc_variants.py --variant {v}`. {r['texts_changed']:,} of {r['documents']:,} "
             f"document texts differ from the shipped ones; encoding took {r['encode_seconds']} s. "
             "Hub scores are recomputed for the new vectors from the same 5,000 training queries "
             "(k and β unchanged).", "",
             f"| queries | plain | shipped (hubness) | {v} | {v} + hubness | {v} vs plain | "
             f"{v} + hubness vs shipped |", "|---|---:|---:|---:|---:|---|---|"]
    for g, x in r["groups"].items():
        a, b = x["variant_vs_plain"], x["variant_hub_vs_shipped"]
        lines.append(f"| {g} ({x['queries']:,}) | {x['plain']['ndcg_at_10']:.4f} | "
                     f"{x['shipped']['ndcg_at_10']:.4f} | {x[v]['ndcg_at_10']:.4f} | "
                     f"{x[v + ' + hubness']['ndcg_at_10']:.4f} | {a['diff']:+.4f} [{a['low']:+.4f}, "
                     f"{a['high']:+.4f}]{' sig.' if a['significant'] else ''} | {b['diff']:+.4f} "
                     f"[{b['low']:+.4f}, {b['high']:+.4f}]{' sig.' if b['significant'] else ''} |")
    full = r["groups"]["full"]
    lines += ["", f"MRR@10 on the full split: plain {full['plain']['mrr']:.4f}, shipped "
              f"{full['shipped']['mrr']:.4f}, {v} {full[v]['mrr']:.4f}, {v} + hubness "
              f"{full[v + ' + hubness']['mrr']:.4f}."]
    rp = r.get("rename_probe")
    if rp:
        lines += ["", "## Rename probe (validated sample, plain cosine)", "",
                  "| document vectors | queries | NDCG@10 original | renamed | drop | relative |",
                  "|---|---:|---:|---:|---:|---:|"]
        for name, s in rp.items():
            lines.append(f"| {name} | {s['queries']} | {s['original']:.4f} | {s['renamed']:.4f} | "
                         f"{s['abs_drop']:+.4f} | {s['rel_drop']:.0%} |")
    lines += ["", f"Total {r['seconds']} s.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
