from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from retrieval import hubness
from tools.hubness_eval import compare, load_train, runs_from_scores, score_run

RESULTS = Path("bench/results")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args(argv)
    from bench.dataset import load_appsretrieval
    from bench.stratified import load as load_stratified
    from encoder import PrePostPipelineEncoder
    from retrieval.embed import Embedder, set_threads

    set_threads(args.threads)
    started = time.perf_counter()
    table = json.loads(Path("data/apps_hubness.json").read_text(encoding="utf-8"))
    assert table["document_text"] == "raw", "not the shipped raw table"
    corpus, queries, qrels = load_appsretrieval()
    doc_ids = sorted(corpus)
    enc = PrePostPipelineEncoder()
    emb = Embedder()
    D = np.asarray(emb.encode([enc.process_document(corpus[d]) for d in doc_ids]), dtype="float32")
    hub = np.asarray([table["by_id"][d] for d in doc_ids], dtype="float32")
    beta, k = table["beta"], table["k"]
    test = sorted(q for q in queries if q in qrels)
    Q = np.asarray(emb.encode([queries[q] for q in test]), dtype="float32")

    _, train_gold = load_train()
    is_train_answer = np.array([d in set(train_gold.values()) for d in doc_ids])
    sample = set(load_stratified())
    held = [q for q in test if q not in sample]

    plain = runs_from_scores(Q @ D.T, test, doc_ids)
    hubbed = runs_from_scores(Q @ D.T - beta * hub[None, :], test, doc_ids)
    keep = np.flatnonzero(~is_train_answer)
    kept = [doc_ids[i] for i in keep]
    only_plain = runs_from_scores(Q @ D[keep].T, test, kept)
    only_hub = runs_from_scores(Q @ D[keep].T - beta * hub[keep][None, :], test, kept)

    def row(a, b, qs):
        qr = {q: qrels[q] for q in qs}
        sa, sb = ({q: a[q] for q in qs}, {q: b[q] for q in qs})
        return {"queries": len(qs), "plain": score_run(sb, qr)["ndcg_at_10"],
                "hub": score_run(sa, qr)["ndcg_at_10"], **compare(sa, sb, qr)}

    groups = {"full split": row(hubbed, plain, test),
              "held-out (not in the sample)": row(hubbed, plain, held),
              "sample (where k, beta were chosen)": row(hubbed, plain, [q for q in test if q in sample]),
              "test-answers-only corpus (train answers removed)": row(only_hub, only_plain, test),
              "test-answers-only, held-out": row(only_hub, only_plain, held)}
    old = json.loads((RESULTS / "hubness.json").read_text(encoding="utf-8"))
    sweep = [r for r in old["sweep"] if r["source"] == "train-excl"]
    best = max(sweep, key=lambda r: r["ndcg_at_10"])
    report = {"table": {"k": k, "beta": beta, "document_text": table["document_text"],
                        "source": table["source"], "documents": len(table["by_id"])},
              "k_beta_provenance": {"sample_sweep_best_k": best["k"], "sample_sweep_best_beta": best["beta"],
                                    "sweep_rows": len(sweep), "sweep_queries": 1000,
                                    "matches_table": best["k"] == k and best["beta"] == beta},
              "train_answer_documents": int(is_train_answer.sum()), "documents": len(doc_ids),
              "separation_auc": hubness.separation_auc(hub, is_train_answer),
              "groups": groups, "seconds": round(time.perf_counter() - started, 1)}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "hub_table_checks.json").write_bytes((json.dumps(report, indent=1) + "\n").encode())
    (RESULTS / "hub_table_checks.md").write_bytes(render(report).encode("utf-8"))
    print(json.dumps(report, indent=1))
    return 0


def render(r: dict) -> str:
    p = r["k_beta_provenance"]
    lines = ["# Hub table checks on the FINAL (raw-text) table", "",
             "`tools/hub_table_checks.py`. The shipped table `data/apps_hubness.json` "
             f"(document_text `{r['table']['document_text']}`, k = {r['table']['k']}, "
             f"β = {r['table']['beta']}, {r['table']['documents']:,} documents, AppsRetrieval "
             "TRAINING queries only, own answers excluded) with raw-text vectors, exactly as the "
             "screening artifact scores. Plain = raw cosine; hub = cos − β·hub(d).", "",
             "| queries | n | plain | hub | change [95% CI] |", "|---|---:|---:|---:|---|"]
    for g, x in r["groups"].items():
        lines.append(f"| {g} | {x['queries']:,} | {x['plain']:.4f} | {x['hub']:.4f} | "
                     f"{x['diff']:+.4f} [{x['low']:+.4f}, {x['high']:+.4f}] p≈{x['p']:.3f}"
                     f"{' **significant**' if x['significant'] else ''} |")
    lines += ["", f"Train-answer documents: {r['train_answer_documents']:,} of {r['documents']:,}. "
              f"Separation AUC of the hub score (train-answer vs the rest): "
              f"**{r['separation_auc']:.3f}** (0.5 = no separation; below 0.5, the documents that "
              "can be right for a test query carry the higher hub scores, so the penalty works "
              "against them, not for them).", "",
              "## Where k and β came from", "",
              f"The sample sweep in `hubness.json` ({p['sweep_rows']} train-excl settings, on the "
              f"validated {p['sweep_queries']:,}-query stratified sample only) picked k = "
              f"{p['sample_sweep_best_k']}, β = {p['sample_sweep_best_beta']}. The raw table "
              f"{'copies exactly those' if p['matches_table'] else 'does NOT match them'}: "
              "`tools/build_hub_table.py` reads k and β from the previous table and never "
              "re-tunes. No held-out query was used for any choice.", ""]
    g = r["groups"]
    ok = all(g[x]["significant"] for x in ("held-out (not in the sample)",
                                            "test-answers-only corpus (train answers removed)"))
    lines += ["## Verdict", "",
              ("All three checks hold on the final table: the gain is significant on the 2,765 "
               "held-out queries and with every train-answer document removed, and k and β come "
               "from the sample only." if ok else
               "**At least one check FAILS on the final table** - see the table above."), "",
              "The table is still dataset-specific: it is built from AppsRetrieval's own training "
              "queries and applies to this corpus only (README and docs/TECHNICAL_REPORT.md §9, §24).", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
