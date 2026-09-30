from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from pipeline.query_proc import BOILERPLATE_RE
from retrieval.workflows import CONF_LOW
from tools.hubness_eval import compare, runs_from_scores, score_run

RESULTS = Path("bench/results")
RRF_K = 60


def segments(text: str) -> tuple[str, str] | None:
    m = BOILERPLATE_RE.search(text)
    if not m or m.start() < 40:
        return None
    desc, io = text[:m.start()].strip(), text[m.start():].strip()
    return (desc, io) if desc and io else None


def margins(scores: np.ndarray) -> np.ndarray:
    top2 = -np.partition(-scores, 1, axis=1)[:, :2]
    return top2[:, 0] - top2[:, 1]


def rrf(*mats: np.ndarray) -> np.ndarray:
    out = np.zeros_like(mats[0])
    for m in mats:
        ranks = np.argsort(np.argsort(-m, axis=1), axis=1)
        out += 1.0 / (RRF_K + 1 + ranks)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--idle", action="store_true")
    args = parser.parse_args(argv)
    from bench.dataset import load_appsretrieval
    from bench.stratified import load as load_stratified
    from retrieval.embed import Embedder, set_threads
    from retrieval.index import load

    set_threads(args.threads)
    started = time.perf_counter()
    _, queries, qrels = load_appsretrieval()
    index = load(Path("out/real.db"))
    D = np.asarray(index.matrices["code"], dtype="float32")
    doc_ids = list(index.matrix_ids["code"])
    index.close()
    emb = Embedder()
    sample = set(load_stratified())
    splits = {"sample": sorted(q for q in queries if q in qrels and q in sample),
              "held-out": sorted(q for q in queries if q in qrels and q not in sample)}
    bias = first_pass_bias(doc_ids)

    report = {"idle": args.idle, "threads": args.threads, "conf_low": CONF_LOW, "splits": {}}
    for split, qids in splits.items():
        if split == "held-out" and not report["splits"]["sample"]["winner"]:
            report["splits"][split] = {"skipped": "no variant won on the sample"}
            continue
        Q = np.asarray(emb.encode([queries[q] for q in qids], show_progress=True), dtype="float32")
        base = Q @ D.T + bias[None, :]
        low = margins(base) < CONF_LOW
        segs = [segments(queries[q]) for q in qids]
        todo = [i for i in range(len(qids)) if segs[i]]
        S_desc = np.array(Q)
        S_io = np.array(Q)
        if todo:
            vd = emb.encode([segs[i][0] for i in todo], show_progress=True)
            vi = emb.encode([segs[i][1] for i in todo], show_progress=True)
            S_desc[todo] = np.asarray(vd, dtype="float32")
            S_io[todo] = np.asarray(vi, dtype="float32")
        desc, io = S_desc @ D.T + bias[None, :], S_io @ D.T + bias[None, :]
        variants = {"seg-mean": (base + desc + io) / 3.0, "seg-rrf": rrf(base, desc, io)}
        q_rels = {q: qrels[q] for q in qids}
        base_run = runs_from_scores(base, qids, doc_ids)
        rows = {"queries": len(qids), "low": int(low.sum()),
                "low_with_sections": int(sum(1 for i in todo if low[i])),
                "base": score_run(base_run, q_rels), "variants": {}}
        low_ids = [q for q, l in zip(qids, low) if l]
        for name, merged in variants.items():
            for mode, mask in (("low only", low), ("always", np.ones_like(low))):
                adaptive = np.where(mask[:, None], merged, base)
                run = runs_from_scores(adaptive, qids, doc_ids)
                low_run = {q: run[q] for q in low_ids}
                low_base = {q: base_run[q] for q in low_ids}
                top1 = lambda r: float(np.mean([r[q][0] in qrels[q] for q in low_ids]))
                rows["variants"][f"{name} ({mode})"] = {
                    **score_run(run, q_rels), **compare(run, base_run, q_rels),
                    "low_top1_base": top1(low_base), "low_top1": top1(low_run),
                    "low_ndcg_base": score_run(low_base, {q: qrels[q] for q in low_ids})["ndcg_at_10"],
                    "low_ndcg": score_run(low_run, {q: qrels[q] for q in low_ids})["ndcg_at_10"]}
        adaptive_rows = {k: v for k, v in rows["variants"].items() if k.endswith("(low only)")}
        best = max(adaptive_rows, key=lambda k: adaptive_rows[k]["ndcg_at_10"])
        if split == "sample":
            rows["winner"] = best if adaptive_rows[best]["significant"] else None
        report["splits"][split] = rows

    report["latency"] = latency(queries, splits["sample"], emb)
    report["seconds"] = round(time.perf_counter() - started, 1)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "adaptive.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    (RESULTS / "adaptive.md").write_text(render(report), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "splits"}, indent=1))
    return 0


def first_pass_bias(doc_ids: list[str]) -> np.ndarray:
    from retrieval import hubness
    table = hubness.load_table()
    return (-table["beta"] * np.asarray([table["by_id"][d] for d in doc_ids], dtype="float32")
            if table else np.zeros(len(doc_ids), dtype="float32"))


def latency(queries, qids, emb, n: int = 40) -> dict:
    from retrieval.embed import Embedder, set_threads
    set_threads(8)
    cold = Embedder(cache=False)
    cold.warm()
    picked = [q for q in qids if segments(queries[q])][:n]
    t = time.perf_counter()
    for q in picked:
        d, i = segments(queries[q])
        cold.encode([d])
        cold.encode([i])
    per_low = (time.perf_counter() - t) * 1000 / max(1, len(picked))
    t = time.perf_counter()
    for q in picked:
        cold.encode([queries[q]])
    first = (time.perf_counter() - t) * 1000 / max(1, len(picked))
    return {"queries_timed": len(picked), "first_pass_encode_ms": round(first, 1),
            "second_pass_ms_per_low_query": round(per_low, 1)}


def render(r: dict) -> str:
    lat = r["latency"]
    lines = ["# Adaptive second pass (2.4)", "",
             f"A second pass runs only when the first pass is unsure (confidence LOW: margin "
             f"< {r['conf_low']}) and is merged with it. The first pass is the shipped search "
             "(with the 2.1 hubness correction). `tools/adaptive_eval.py`. Variants "
             "were chosen on the validated 1,000-query sample (where the confidence "
             "thresholds were set); the best is confirmed on the 2,765 held-out test queries. "
             "Paired bootstrap vs the first pass alone.", ""]
    for split, rows in r["splits"].items():
        if "skipped" in rows:
            lines += [f"## {split}", "", f"Not run: {rows['skipped']}.", ""]
            continue
        b = rows["base"]
        lines += [f"## {split} ({rows['queries']:,} queries, {rows['low']:,} LOW, "
                  f"{rows['low_with_sections']:,} of them with I/O sections)", "",
                  f"First pass alone: NDCG@10 {b['ndcg_at_10']:.4f}.", "",
                  "| variant | NDCG@10 | vs first pass | LOW: top-1 | LOW: NDCG@10 |",
                  "|---|---:|---|---:|---:|"]
        for name, v in rows["variants"].items():
            mark = "**" if v["significant"] else ""
            lines.append(f"| {name} | {v['ndcg_at_10']:.4f} | {mark}{v['diff']:+.4f}{mark} "
                         f"[{v['low']:+.4f}, {v['high']:+.4f}] p≈{v['p']:.3f} | "
                         f"{v['low_top1_base']:.1%} → {v['low_top1']:.1%} | "
                         f"{v['low_ndcg_base']:.4f} → {v['low_ndcg']:.4f} |")
        if split == "sample":
            lines += ["", "Winner on the sample: " + (f"**{rows['winner']}**." if rows["winner"]
                                                     else "none (no LOW-only variant is significant).")]
        lines.append("")
    lines += ["## Added latency", "",
              f"Uncached encodes, {lat['queries_timed']} sample queries with sections, "
              "8 threads (the tool's default), " + ("measured idle" if r["idle"] else
                                             "**measured under load**") + ": the first pass "
              f"encodes the query in {lat['first_pass_encode_ms']} ms; the second pass adds "
              f"{lat['second_pass_ms_per_low_query']} ms per LOW query (two segment encodes; "
              "scoring is a matrix product, negligible). Averaged over all queries that is "
              "about the LOW share times that.", ""]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
