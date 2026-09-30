from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

RESULTS = Path("bench/results")


def evaluate(name: str, root: Path, emb) -> tuple[dict, dict, dict]:
    from pipeline.query_proc import process
    from retrieval.index import build, load
    from tools.memorisation_probe import ndcg, questions

    corpus, qs = questions(name, root)
    scores = {}
    for view in ("code", "raw"):
        db = Path("out") / f"rawrepo_{name}_{view}.db"
        if db.exists():
            db.unlink()
        build(corpus, db, version="v1", views=(view,), embedder=emb, corpus_kind="repo")
        index = load(db)
        try:
            ids = [index.identity_of[u] for u in index.matrix_ids[view]]
            D = np.asarray(index.matrices[view], dtype="float32")
        finally:
            index.close()
        col = {d: i for i, d in enumerate(ids)}
        kept = [q for q in qs if q["answer"] in col]
        Q = np.asarray(emb.encode([process(q["text"]).embedding_text or q["text"] for q in kept]),
                       dtype="float32")
        per = {}
        for i, q in enumerate(kept):
            s = D @ Q[i]
            a = col[q["answer"]]
            per[f"{name}:{q['answer']}:{i}"] = ndcg(1 + int((np.delete(s, a) > s[a]).sum()))
        scores[view] = per
    common = sorted(set(scores["code"]) & set(scores["raw"]))
    return ({k: scores["code"][k] for k in common}, {k: scores["raw"][k] for k in common},
            {"functions": len(corpus), "questions": len(common)})


def main(argv: list[str] | None = None) -> int:
    from bench.significance import paired_bootstrap
    from retrieval.embed import Embedder, set_threads
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", action="append", required=True, help="NAME=PATH (read-only)")
    parser.add_argument("--private", action="append", default=[])
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args(argv)
    set_threads(args.threads)
    started = time.perf_counter()
    emb = Embedder()
    rows, pooled_code, pooled_raw = [], {}, {}
    for spec in args.repo:
        name, _, path = spec.partition("=")
        code, raw, info = evaluate(name, Path(path), emb)
        cmp = paired_bootstrap(raw, code, "raw", "code", iterations=2000)
        rows.append({"repo": name, **info, "code": round(float(np.mean(list(code.values()))), 4),
                     "raw": round(float(np.mean(list(raw.values()))), 4),
                     "diff": round(cmp.diff.mean, 4), "low": round(cmp.diff.low, 4),
                     "high": round(cmp.diff.high, 4), "p": round(cmp.p_value, 4)})
        pooled_code.update(code)
        pooled_raw.update(raw)
        print(json.dumps(rows[-1]), flush=True)
    cmp = paired_bootstrap(pooled_raw, pooled_code, "raw", "code", iterations=2000)
    pooled = {"questions": len(pooled_code), "diff": round(cmp.diff.mean, 4),
              "low": round(cmp.diff.low, 4), "high": round(cmp.diff.high, 4),
              "p": round(cmp.p_value, 4), "significant": bool(cmp.significant and cmp.diff.mean > 0)}
    r = {"rows": rows, "pooled": pooled, "private": args.private,
         "seconds": round(time.perf_counter() - started, 1)}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "raw_repos.json").write_text(json.dumps(r, indent=1), encoding="utf-8")
    (RESULTS / "raw_repos.md").write_text(render(r), encoding="utf-8")
    print(json.dumps(pooled, indent=1))
    return 0


def render(r: dict) -> str:
    lines = ["# Raw document view vs code view on real repositories", "",
             "Plain cosine, held-out questions from the code's own documentation (docstrings removed "
             "from their answer; JS/TS comments above the function). Both views built by the real "
             "`build()`. Aggregate numbers only. `tools/raw_repos.py`.", "",
             "| repository | functions | questions | code view | raw view | change [95% CI] |",
             "|---|---:|---:|---:|---:|---|"]
    for row in r["rows"]:
        tag = " (private)" if row["repo"] in r["private"] else ""
        lines.append(f"| {row['repo']}{tag} | {row['functions']:,} | {row['questions']} | "
                     f"{row['code']:.4f} | {row['raw']:.4f} | {row['diff']:+.4f} [{row['low']:+.4f}, "
                     f"{row['high']:+.4f}] p≈{row['p']:.3f} |")
    p = r["pooled"]
    lines += ["", f"Pooled over {p['questions']} questions: {p['diff']:+.4f} [{p['low']:+.4f}, "
              f"{p['high']:+.4f}] p≈{p['p']:.3f}: "
              + ("**significant: the raw view becomes the default for repository indexes**."
                 if p["significant"] else "not significant: repository indexes keep the code view."),
              "", f"Total {r['seconds']} s.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
