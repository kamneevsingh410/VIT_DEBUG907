from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RESULTS = Path("bench/results")
PUBLISHED = {"CosQA": 43.47, "AppsRetrieval": 57.54}


def run(task: str, raw: bool, threads: int) -> dict:
    import run_mteb
    out = Path("out") / f"mteb_{task.lower()}_{'raw' if raw else 'shipped'}.json"
    argv = ["--task", task, "--out", str(out), "--threads", str(threads), "--no-hubness"]
    argv.append("--raw-docs" if raw else "--code-view")
    started = time.perf_counter()
    code = run_mteb.main(argv)
    if code != 0:
        return {"error": f"run_mteb exited {code}"}
    s = json.loads(out.read_text(encoding="utf-8"))["scores"]["test"][0]
    return {"ndcg_at_10": s["ndcg_at_10"], "mrr_at_10": s["mrr_at_10"],
            "recall_at_100": s.get("recall_at_100"), "minutes": round((time.perf_counter() - started) / 60, 1),
            "file": out.as_posix()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="CosQA")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args(argv)
    r = {"task": args.task, "published_ndcg_at_10": PUBLISHED.get(args.task),
         "shipped": run(args.task, False, args.threads),
         "raw": run(args.task, True, args.threads)}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "second_benchmark.json").write_text(json.dumps(r, indent=1), encoding="utf-8")
    (RESULTS / "second_benchmark.md").write_text(render(r), encoding="utf-8")
    print(json.dumps(r, indent=1))
    return 0


def render(r: dict) -> str:
    t = r["task"]
    rows = []
    for name, label in (("shipped", "shipped config (code view, seq 1024)"),
                        ("raw", "same encoder, raw documents")):
        x = r[name]
        if "error" in x:
            rows.append(f"| {label} | {x['error']} | | |")
        else:
            rows.append(f"| {label} | **{x['ndcg_at_10'] * 100:.2f}** | {x['mrr_at_10'] * 100:.2f} | "
                        f"{x['minutes']} min |")
    pub = r["published_ndcg_at_10"]
    return "\n".join([
        f"# A second benchmark, held out: {t}", "",
        f"CoIR {t} through the official MTEB path (`run_mteb.py --task {t}`), with no tuning "
        "on this task. The hubness table belongs to AppsRetrieval, so both runs are plain cosine. "
        "`tools/second_benchmark.py`.", "",
        "| configuration | NDCG@10 | MRR@10 | time |", "|---|---:|---:|---:|", *rows,
        f"| published: gte-modernbert-base model card (seq 8,192, raw documents) | {pub} | - | - |"
        if pub else "", "", ""])


if __name__ == "__main__":
    raise SystemExit(main())
