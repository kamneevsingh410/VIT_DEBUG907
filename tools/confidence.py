from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.dataset import load_appsretrieval
from bench.stratified import load as load_stratified
from retrieval import workflows
from retrieval.embed import Embedder, set_threads
from retrieval.index import load

RESULTS = Path("bench/results")
LABELS = ("high", "medium", "low")


def outcomes(index, queries, qrels, emb) -> list[dict]:
    emb.encode(list(queries.values()))
    rows = []
    for qid, text in queries.items():
        hits = workflows.shipped_search(index, text, emb, top_k=10).hits
        label, margin = workflows.confidence(hits)
        ids = [h[0] for h in hits]
        rel = qrels[qid]
        rows.append({"qid": qid, "label": label, "margin": margin,
                     "top1": bool(ids and ids[0] in rel),
                     "top10": any(i in rel for i in ids)})
    return rows


def auc(rows: list[dict]) -> float:
    pos = sorted(r["margin"] for r in rows if r["top1"])
    neg = sorted(r["margin"] for r in rows if not r["top1"])
    if not pos or not neg:
        return float("nan")
    import bisect
    wins = sum(bisect.bisect_left(neg, p) + 0.5 * (bisect.bisect_right(neg, p)
                                                   - bisect.bisect_left(neg, p)) for p in pos)
    return wins / (len(pos) * len(neg))


def summarise(rows: list[dict]) -> dict:
    out = {"n": len(rows), "auc": auc(rows)}
    for label in LABELS:
        group = [r for r in rows if r["label"] == label]
        n = len(group)
        out[label] = {"n": n, "share": n / max(1, len(rows)),
                      "top1": sum(r["top1"] for r in group) / max(1, n),
                      "top10": sum(r["top10"] for r in group) / max(1, n)}
    ordered = sorted(rows, key=lambda r: r["margin"])
    size = max(1, len(ordered) // 10)
    out["deciles"] = [
        {"decile": d + 1,
         "margin_lo": ordered[d * size]["margin"],
         "margin_hi": ordered[min(len(ordered), (d + 1) * size) - 1]["margin"],
         "top1": sum(r["top1"] for r in ordered[d * size:(d + 1) * size]) / size}
        for d in range(10)]
    return out


def main(argv: list[str] | None = None) -> int:
    set_threads(8)
    corpus, queries, qrels = load_appsretrieval()
    picked = set(load_stratified())
    sample = {q: t for q, t in queries.items() if q in picked and q in qrels}
    held_out = {q: t for q, t in queries.items() if q not in picked and q in qrels}
    index = load(Path("out/real.db"))
    emb = Embedder()
    try:
        if len(index.ids) != len(corpus):
            print("out/real.db is not the full AppsRetrieval index; run: debug907 index")
            return 1
        report = {"thresholds": {"high": workflows.CONF_HIGH, "low": workflows.CONF_LOW},
                  "calibration_sample": summarise(outcomes(index, sample, qrels, emb)),
                  "held_out": summarise(outcomes(index, held_out, qrels, emb))}
    finally:
        index.close()
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "confidence.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_md(report)
    for part in ("calibration_sample", "held_out"):
        s = report[part]
        print(part, s["n"], {k: (s[k]["n"], round(s[k]["top1"], 3)) for k in LABELS})
    print(f"wrote {RESULTS / 'confidence.md'}")
    return 0


def write_md(r: dict) -> None:
    t = r["thresholds"]
    lines = ["# Confidence labels: calibration and held-out check", "",
             f"Label = cosine margin between results #1 and #2 of the shipped search: "
             f"**high** ≥ {t['high']}, **low** < {t['low']}, medium in between "
             "(retrieval/workflows.py). Thresholds were set on the 1,000-query stratified "
             "sample; the held-out column is the other test queries, never used for tuning.", "",
             "| label | sample: share | sample: top-1 | sample: in top 10 "
             "| held-out: share | held-out: top-1 | held-out: in top 10 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    a, b = r["calibration_sample"], r["held_out"]
    for label in LABELS:
        x, y = a[label], b[label]
        lines.append(f"| {label} | {x['share']:.0%} ({x['n']}) | {x['top1']:.0%} | {x['top10']:.0%} "
                     f"| {y['share']:.0%} ({y['n']}) | {y['top1']:.0%} | {y['top10']:.0%} |")
    lines += ["", f"Queries: sample {a['n']}, held-out {b['n']}.", "",
              f"**AUC** of the margin as a predictor of a correct #1: {a['auc']:.3f} on the "
              f"sample, **{b['auc']:.3f} held-out** (0.5 = no signal, 1.0 = perfect).", "",
              "## Top-1 accuracy by margin decile (held-out)", "",
              "| decile | margin range | top-1 |", "|---:|---|---:|"]
    for d in b["deciles"]:
        lines.append(f"| {d['decile']} | {d['margin_lo']:.4f} – {d['margin_hi']:.4f} | {d['top1']:.0%} |")
    lines += ["", "Caveat: calibrated on competitive-programming problems (CoIR AppsRetrieval). "
              "On other code the labels are indicative, not measured."]
    (RESULTS / "confidence.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
