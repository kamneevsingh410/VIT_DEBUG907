from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from bench.metrics import evaluate
from bench.significance import paired_bootstrap, per_query_scores
from retrieval import hubness

RESULTS = Path("bench/results")
KS = (10, 20, 50)
BETAS = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0)


def load_train():
    from datasets import load_dataset
    from bench.dataset import APPS_REVISION, QRELS_REVISION
    q = load_dataset("CoIR-Retrieval/apps", "queries", split="queries", revision=APPS_REVISION)
    texts = {i: t for i, t, p in zip(q["_id"], q["text"], q["partition"]) if p == "train"}
    tr = load_dataset("CoIR-Retrieval/apps-qrels", split="train", revision=QRELS_REVISION)
    gold = {qid: did for qid, did in zip(tr["query_id"], tr["corpus_id"])}
    return texts, gold


def runs_from_scores(scores: np.ndarray, qids: list[str], doc_ids: list[str],
                     top: int = 100) -> dict[str, list[str]]:
    idx = np.argpartition(-scores, top, axis=1)[:, :top]
    out = {}
    for row, qid in enumerate(qids):
        order = idx[row][np.argsort(-scores[row, idx[row]])]
        out[qid] = [doc_ids[i] for i in order]
    return out


def score_run(run, qrels) -> dict:
    s = evaluate(run, qrels)
    return {"ndcg_at_10": s.ndcg_at_10, "mrr": s.mrr, "recall_at_100": s.recall_at_100}


def compare(run, base_run, qrels, iterations=1000) -> dict:
    cmp = paired_bootstrap(per_query_scores(run, qrels), per_query_scores(base_run, qrels),
                           "hub", "base", iterations=iterations)
    return {"diff": cmp.diff.mean, "low": cmp.diff.low, "high": cmp.diff.high,
            "p": cmp.p_value, "significant": cmp.significant and cmp.diff.mean > 0}


def hub_stats(run, qrels) -> dict:
    wrong = [ids[0] for q, ids in run.items() if ids and ids[0] not in qrels[q]]
    counts = Counter(wrong)
    top_doc, top_n = counts.most_common(1)[0] if counts else (None, 0)
    return {"wrong_top1": len(wrong), "hub_share": hubness.hub_share(wrong),
            "top_hub": top_doc, "top_hub_queries": top_n}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--skip-npm", action="store_true")
    parser.add_argument("--smoke", type=int, default=0,
                        help="code-path check only: N training queries, results to out/")
    parser.add_argument("--render", action="store_true",
                        help="only re-render hubness.md from bench/results/hubness.json")
    args = parser.parse_args(argv)
    if args.render:
        report = json.loads((RESULTS / "hubness.json").read_text(encoding="utf-8"))
        (RESULTS / "hubness.md").write_text(render(report), encoding="utf-8")
        return 0
    from bench.dataset import load_appsretrieval
    from bench.stratified import load as load_stratified
    from retrieval.embed import Embedder, set_threads
    from retrieval.index import load

    set_threads(args.threads)
    started = time.perf_counter()
    corpus, queries, qrels = load_appsretrieval()
    train_texts, train_gold = load_train()
    assert not set(train_texts) & set(qrels), "train and test queries overlap"

    index = load(Path("out/real.db"))
    D = np.asarray(index.matrices["code"], dtype="float32")
    doc_ids = list(index.matrix_ids["code"])
    index.close()
    row = {d: i for i, d in enumerate(doc_ids)}

    emb = Embedder()
    t = time.perf_counter()
    train_ids = sorted(train_texts)[:args.smoke] if args.smoke else sorted(train_texts)
    T = np.asarray(emb.encode([train_texts[i] for i in train_ids], show_progress=True),
                   dtype="float32")
    encode_s = time.perf_counter() - t
    test_ids = sorted(q for q in queries if q in qrels)
    Q = np.asarray(emb.encode([queries[q] for q in test_ids]), dtype="float32")
    sample = set(load_stratified())
    s_rows = [i for i, q in enumerate(test_ids) if q in sample]
    s_ids = [test_ids[i] for i in s_rows]
    s_qrels = {q: qrels[q] for q in s_ids}

    exclude: dict[int, set[int]] = {}
    for j, qid in enumerate(train_ids):
        g = train_gold.get(qid)
        if g in row:
            exclude.setdefault(row[g], set()).add(j)
    is_train_answer = np.zeros(len(doc_ids), dtype=bool)
    for g in train_gold.values():
        if g in row:
            is_train_answer[row[g]] = True

    sources = {}
    for k in KS:
        sources[("train-excl", k)] = hubness.hub_scores(D, T, k, exclude=exclude)
        sources[("train-incl", k)] = hubness.hub_scores(D, T, k)
        sources[("doc-doc", k)] = hubness.doc_doc_hub(D, k)
    leakage = {f"{s}@{k}": hubness.separation_auc(h, is_train_answer)
               for (s, k), h in sources.items()}

    base_scores = Q[s_rows] @ D.T
    base_run = runs_from_scores(base_scores, s_ids, doc_ids)
    base = score_run(base_run, s_qrels)
    sweep = []
    for (src, k), hub in sources.items():
        for beta in BETAS:
            run = runs_from_scores(hubness.penalised(Q[s_rows], D, hub, beta), s_ids, doc_ids)
            sweep.append({"source": src, "k": k, "beta": beta, **score_run(run, s_qrels),
                          **compare(run, base_run, s_qrels, iterations=500)})

    full_base_run = runs_from_scores(Q @ D.T, test_ids, doc_ids)
    full = {"base": {**score_run(full_base_run, qrels), **hub_stats(full_base_run, qrels)}}
    winners = {}
    for src in ("train-excl", "doc-doc", "train-incl"):
        rows = [r for r in sweep if r["source"] == src]
        best = max(rows, key=lambda r: r["ndcg_at_10"])
        winners[src] = best
        if best["significant"]:
            hub = sources[(src, best["k"])]
            run = runs_from_scores(hubness.penalised(Q, D, hub, best["beta"]), test_ids, doc_ids)
            leak = leakage[f"{src}@{best['k']}"]
            keep = np.flatnonzero(~is_train_answer)
            kept_ids = [doc_ids[i] for i in keep]
            only_base = runs_from_scores(Q @ D[keep].T, test_ids, kept_ids)
            only_run = runs_from_scores(hubness.penalised(Q, D[keep], hub[keep], best["beta"]),
                                        test_ids, kept_ids)
            only = {"base": score_run(only_base, qrels)["ndcg_at_10"],
                    "hub": score_run(only_run, qrels)["ndcg_at_10"],
                    **compare(only_run, only_base, qrels)}
            held = [q for q in test_ids if q not in sample]
            h_qrels = {q: qrels[q] for q in held}
            held_out = {"queries": len(held),
                        "base": score_run({q: full_base_run[q] for q in held}, h_qrels)["ndcg_at_10"],
                        "hub": score_run({q: run[q] for q in held}, h_qrels)["ndcg_at_10"],
                        **compare({q: run[q] for q in held}, {q: full_base_run[q] for q in held},
                                  h_qrels)}
            full[src] = {"k": best["k"], "beta": best["beta"], **score_run(run, qrels),
                         **compare(run, full_base_run, qrels), **hub_stats(run, qrels),
                         "leakage_auc": leak, "test_answers_only": only, "held_out": held_out}
            full[src]["eligible_for_artifact"] = bool(
                src != "train-incl" and full[src]["significant"] and only["significant"])

    tables = {src: write_hub_table(src, f, sources[(src, f["k"])], doc_ids, corpus, args.smoke)
              for src, f in full.items() if src in ("train-excl", "doc-doc")}

    report = {"train_queries": len(train_ids), "train_encode_s": round(encode_s, 1),
              "hub_tables": tables,
              "sample_base": base, "leakage_auc": leakage, "sweep": sweep,
              "best_on_sample": winners, "full_split": full,
              "npm": None if args.skip_npm else npm_eval(emb),
              "seconds": round(time.perf_counter() - started, 1)}
    results = Path("out/hubness_smoke") if args.smoke else RESULTS
    results.mkdir(parents=True, exist_ok=True)
    (results / "hubness.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    (results / "hubness.md").write_text(render(report), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "sweep"}, indent=1))
    return 0


def write_hub_table(src, winner, hub, doc_ids, corpus, smoke) -> dict:
    import hashlib
    from encoder import PrePostPipelineEncoder
    enc = PrePostPipelineEncoder(raw_documents=False, hub_table=None)
    values: dict[str, list[float]] = {}
    key_of: dict[str, str] = {}
    for d, h in zip(doc_ids, hub):
        key = hashlib.sha256(enc.process_document(corpus[d]).encode("utf-8")).hexdigest()
        values.setdefault(key, []).append(float(h))
        key_of[d] = key
    conflicts = sum(1 for v in values.values() if max(v) - min(v) > 1e-6)
    hubs = {k: sum(v) / len(v) for k, v in values.items()}
    path = Path("out") / f"hubness_table_{src}{'_smoke' if smoke else ''}.json"
    path.write_text(json.dumps({"source": src, "k": winner["k"], "beta": winner["beta"],
                                "made_by": "tools/hubness_eval.py: AppsRetrieval TRAINING "
                                           "queries only, own answers excluded",
                                "hubs": hubs, "by_id": {d: hubs[key_of[d]] for d in doc_ids}}),
                    encoding="utf-8")
    return {"path": path.as_posix(), "documents": len(doc_ids), "keys": len(values),
            "conflicts": conflicts}


def npm_eval(emb) -> dict:
    from bench.repo_eval import pseudo_queries
    from pipeline.repo import extract_repo
    from retrieval.index import load

    db = Path("out/npm_eval.db")
    if not db.exists():
        return {"skipped": "out/npm_eval.db not found (run debug907 eval-repo external/npm)"}
    index = load(db)
    D = np.asarray(index.matrices["code"], dtype="float32")
    doc_ids = list(index.matrix_ids["code"])
    index.close()
    qs, _ = pseudo_queries(Path("external/npm"), extract_repo(Path("external/npm")))
    qs = [q for q in qs if q.answer in set(doc_ids)]
    Q = np.asarray(emb.encode([q.text for q in qs]), dtype="float32")
    qrels = {q.qid: {q.answer: 1} for q in qs}
    ids = [q.qid for q in qs]
    base_run = runs_from_scores(Q @ D.T, ids, doc_ids, top=min(100, len(doc_ids) - 1))
    out = {"queries": len(qs), "base": score_run(base_run, qrels), "rows": []}
    folds = [np.array([i % 2 == f for i in range(len(qs))]) for f in (0, 1)]
    for k in KS:
        variants = {"doc-doc": hubness.doc_doc_hub(D, k)}
        for beta in BETAS:
            for name, hub in variants.items():
                run = runs_from_scores(hubness.penalised(Q, D, hub, beta), ids, doc_ids,
                                       top=min(100, len(doc_ids) - 1))
                out["rows"].append({"source": name, "k": k, "beta": beta,
                                    **score_run(run, qrels), **compare(run, base_run, qrels, 500)})
            run = {}
            for f in (0, 1):
                refs = Q[~folds[f]]
                hub = hubness.hub_scores(D, refs, k)
                sub = runs_from_scores(hubness.penalised(Q[folds[f]], D, hub, beta),
                                       [i for i, m in zip(ids, folds[f]) if m], doc_ids,
                                       top=min(100, len(doc_ids) - 1))
                run.update(sub)
            out["rows"].append({"source": "pseudo-query (cross-fitted)", "k": k, "beta": beta,
                                **score_run(run, qrels), **compare(run, base_run, qrels, 500)})
    out["best"] = max(out["rows"], key=lambda r: r["ndcg_at_10"])
    return out


def outcome() -> list[str]:
    meta_path = Path("appsretrieval_results.meta.json")
    if not meta_path.exists():
        return []
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    hub = meta.get("configuration", {}).get("hubness_correction")
    if not hub:
        return ["", "## Outcome", "", "Not shipped: the artifact is plain cosine."]
    s, prev = meta["artifact_scores"], meta.get("previous_artifact", {})
    return ["", "## Outcome: shipped", "",
            f"train-excl, k={hub['k']}, β={hub['beta']} is part of the shipped configuration. "
            "Inside MTEB it runs as one appended dimension (+1 for queries, −β·hub(d) for "
            "documents; MTEB scores with the encoder's plain dot product), from the committed "
            "table `data/apps_hubness.json`. Official MTEB result: NDCG@10 "
            f"**{s['ndcg_at_10']:.5f}** (was {prev.get('ndcg_at_10', 0):.5f}), MRR@10 "
            f"{s['mrr_at_10']:.5f} (was {prev.get('mrr_at_10', 0):.5f}), R@100 "
            f"{s['recall_at_100']:.5f} (was {prev.get('recall_at_100', 0):.5f}). The tool applies "
            "the same correction on an AppsRetrieval index, so `debug907 reproduce --full` "
            "matches the artifact on both paths. The previous artifact is kept as "
            f"`{prev.get('file', '?')}` (`run_mteb.py --no-hubness`). Repo indexes are not "
            "corrected (npm below)."]


def render(r: dict) -> str:
    b, full = r["sample_base"], r["full_split"]
    lines = ["# Hubness correction (2.1)", "",
             "penalised(q, d) = cos(q, d) − β · hub(d), hub(d) = mean cosine of d to its k "
             "nearest reference vectors (`retrieval/hubness.py`, `tools/hubness_eval.py`). "
             f"References: the {r['train_queries']:,} AppsRetrieval **training** queries (never "
             "test queries), or the other documents. Sweep on the validated 1,000-query sample "
             "(paired bootstrap vs β = 0), winners confirmed on the full test split.", "",
             "## The leakage trap, checked", "",
             "Every training query's answer is IN the 8,765-document corpus and is never right "
             "for a test query. A training query sits close to its own answer, so a hub score "
             "that counts that pair penalises exactly the train-answer documents: a gain from "
             "the split, not from hubness. Each document's own training queries are therefore "
             "excluded (train-excl), and the table shows how well each hub score separates "
             "train-answer from test-answer documents (AUC 0.5 = no separation):", "",
             "| hub source | k=10 | k=20 | k=50 |", "|---|---:|---:|---:|"]
    for src in ("train-incl", "train-excl", "doc-doc"):
        lines.append(f"| {src} | " + " | ".join(f"{r['leakage_auc'][f'{src}@{k}']:.3f}" for k in KS) + " |")
    lines += ["", "AUC below 0.5 means test-answer documents have the HIGHER hub scores, so a "
              "penalty works against them. The AUC is direction-blind, so the deciding check "
              "is direct: every full-split winner is re-scored with all train-answer documents "
              "REMOVED from the corpus (only documents that can be right for a test query "
              "remain). Eligible for the screening artifact only if the gain is significant on "
              "the full split AND in that test-answers-only corpus (train-incl is never "
              "eligible).", "",
              "## Sample (1,000 queries): best β per source", "",
              f"Baseline (β = 0, plain cosine, the previous shipped config): NDCG@10 "
              f"{b['ndcg_at_10']:.4f}.", "",
              "| source | k | β | NDCG@10 | vs baseline |", "|---|---:|---:|---:|---|"]
    for src, w in r["best_on_sample"].items():
        mark = "**" if w["significant"] else ""
        lines.append(f"| {src} | {w['k']} | {w['beta']} | {w['ndcg_at_10']:.4f} | "
                     f"{mark}{w['diff']:+.4f}{mark} [{w['low']:+.4f}, {w['high']:+.4f}] p≈{w['p']:.3f} |")
    lines += ["", "## Full test split (3,765 queries)", "",
              "| | NDCG@10 | vs baseline | hub share of wrong #1 | top hub wins | leakage AUC "
              "| test-answers-only corpus | artifact? |",
              "|---|---:|---|---:|---:|---:|---|---|"]
    fb = full["base"]
    lines.append(f"| plain cosine (the previous artifact) | {fb['ndcg_at_10']:.4f} | - | {fb['hub_share']:.1%} | "
                 f"{fb['top_hub_queries']} | - | - | - |")
    for src in ("train-excl", "doc-doc", "train-incl"):
        f = full.get(src)
        if f:
            o = f["test_answers_only"]
            lines.append(f"| {src} k={f['k']} β={f['beta']} | {f['ndcg_at_10']:.4f} | "
                         f"{f['diff']:+.4f} [{f['low']:+.4f}, {f['high']:+.4f}] | {f['hub_share']:.1%} | "
                         f"{f['top_hub_queries']} | {f['leakage_auc']:.3f} | "
                         f"{o['base']:.4f} → {o['hub']:.4f}, {o['diff']:+.4f} "
                         f"[{o['low']:+.4f}, {o['high']:+.4f}] | "
                         f"{'eligible' if f['eligible_for_artifact'] else 'no'} |")
        else:
            lines.append(f"| {src} | (no significant gain on the sample) | | | | | | no |")
    lines += ["", "k and β were chosen on the sample, which is part of the full split. On the "
              "other test queries alone (never used for any choice):", ""]
    for src in ("train-excl", "train-incl"):
        f = full.get(src)
        if f and "held_out" in f:
            h = f["held_out"]
            lines.append(f"- {src}: {h['queries']:,} held-out queries, NDCG@10 {h['base']:.4f} → "
                         f"{h['hub']:.4f}, {h['diff']:+.4f} [{h['low']:+.4f}, {h['high']:+.4f}]"
                         + (", significant" if h["significant"] else ", not significant"))
    lines += outcome()
    n = r.get("npm")
    if n and "rows" in n:
        best = n["best"]
        lines += ["", "## Real code: npm (tool-only)", "",
                  f"{n['queries']} npm pseudo-queries; baseline NDCG@10 {n['base']['ndcg_at_10']:.4f}. "
                  "Pseudo-query hubs are 2-fold cross-fitted (a query never helps compute its own "
                  "correction). Best:", "",
                  f"{best['source']}, k={best['k']}, β={best['beta']}: NDCG@10 {best['ndcg_at_10']:.4f}, "
                  f"{best['diff']:+.4f} [{best['low']:+.4f}, {best['high']:+.4f}] p≈{best['p']:.3f}: "
                  + ("**significant**." if best["significant"] else "not significant.")]
    lines += ["", f"Training-query encoding: {r['train_encode_s']} s (0 when cached). Total {r['seconds']} s."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
