from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cli
from retrieval import batch, workflows

WORK = Path("out/batch_check")
RESULTS = Path("bench/results")


def run_cli(argv: list[str]) -> tuple[int, str]:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = cli.main(argv)
    return code, buf.getvalue()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args(argv)
    from bench.dataset import load_appsretrieval
    from retrieval.embed import Embedder, set_threads
    from retrieval.index import load

    set_threads(args.threads)
    WORK.mkdir(parents=True, exist_ok=True)
    corpus, queries, qrels = load_appsretrieval()
    qids = sorted(q for q in queries if q in qrels)[:args.n]
    qfile, rfile = WORK / "queries.jsonl", WORK / "qrels.tsv"
    qfile.write_text("".join(json.dumps({"id": q, "text": queries[q]}) + "\n" for q in qids),
                     encoding="utf-8")
    rfile.write_text("query-id\tcorpus-id\tscore\n" + "".join(
        f"{q}\t{d}\t{r}\n" for q in qids for d, r in qrels[q].items()), encoding="utf-8")
    sub_qrels = {q: qrels[q] for q in qids}

    ranked = WORK / "ranked.csv"
    t = time.perf_counter()
    code, out = run_cli(["rank", "--queries", str(qfile), "--db", "out/real.db", "--top", "100",
                         "--out", str(ranked), "--jsonl", str(WORK / "ranked.jsonl"),
                         "--threads", str(args.threads)])
    rank_s = time.perf_counter() - t
    assert code == 0, out
    code, eval_out = run_cli(["eval", "--qrels", str(rfile), "--rankings", str(ranked)])
    assert code == 0, eval_out
    from tools.score_csv import read_rankings
    cli_scores = batch.evaluate(read_rankings(ranked), sub_qrels)

    index = load(Path("out/real.db"))
    emb = Embedder()
    try:
        shipped = workflows.evaluate_shipped(index, {q: queries[q] for q in qids}, sub_qrels, emb)
        plain = batch.rank(index, {q: queries[q] for q in qids}, emb, top_k=100, hubness=False)
    finally:
        index.close()
    shipped_scores = batch.evaluate(shipped.run, sub_qrels)
    identical = all(abs(cli_scores[k] - shipped_scores[k]) < 1e-12
                    for k in ("ndcg_at_10", "mrr_at_10", "recall_at_100"))
    same_order = read_rankings(ranked) == {q: shipped.run[q][:100] for q in qids}

    cfile = WORK / "corpus.jsonl"
    cfile.write_text("".join(json.dumps({"id": d, "text": t_}) + "\n"
                             for d, t_ in sorted(corpus.items())), encoding="utf-8")
    cdb = WORK / "corpus.db"
    if cdb.exists():
        cdb.unlink()
    t = time.perf_counter()
    code, ic_out = run_cli(["index-corpus", str(cfile), "--out", str(cdb),
                            "--threads", str(args.threads)])
    index_s = time.perf_counter() - t
    assert code == 0, ic_out
    gindex = load(cdb)
    try:
        generic = batch.rank(gindex, {q: queries[q] for q in qids}, emb, top_k=100)
        kind = gindex.corpus_kind
    finally:
        gindex.close()
    generic_same = ({q: [s for s, _ in h] for q, h in generic.items()}
                    == {q: [s for s, _ in h] for q, h in plain.items()})

    report = {"queries": len(qids), "cli": cli_scores, "shipped": shipped_scores,
              "identical_scores": identical, "identical_order": same_order,
              "rank_seconds": round(rank_s, 1), "index_corpus_seconds": round(index_s, 1),
              "generic_corpus_kind": kind, "generic_equals_plain_cosine": generic_same,
              "eval_output": eval_out.strip()}
    (RESULTS / "batch_ranking.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    (RESULTS / "batch_ranking.md").write_text(render(report), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "eval_output"}, indent=1))
    return 0 if identical and same_order and generic_same else 1


def render(r: dict) -> str:
    c, s = r["cli"], r["shipped"]
    return "\n".join([
        "# Batch ranking on your own data: end-to-end check", "",
        "`debug907 index-corpus`, `debug907 rank` and `debug907 eval --qrels` "
        "(`retrieval/batch.py`), checked on real data by `tools/batch_check.py`. "
        f"The first {r['queries']} judged queries of the AppsRetrieval test split and their "
        "labels were exported as files (JSONL queries, TSV qrels), as a judge would supply them, "
        "and run through the real command-line entry point on the real index.", "",
        "| | NDCG@10 | MRR@10 | R@100 |", "|---|---:|---:|---:|",
        f"| `rank` → CSV → `eval --qrels` | {c['ndcg_at_10']:.5f} | {c['mrr_at_10']:.5f} | "
        f"{c['recall_at_100']:.5f} |",
        f"| the shipped path (`evaluate_shipped`, as `reproduce`) | {s['ndcg_at_10']:.5f} | "
        f"{s['mrr_at_10']:.5f} | {s['recall_at_100']:.5f} |", "",
        f"Identical scores: **{'yes' if r['identical_scores'] else 'NO'}**. Identical rankings, "
        f"rank by rank: **{'yes' if r['identical_order'] else 'NO'}**. `rank` took "
        f"{r['rank_seconds']} s for {r['queries']} queries (query vectors cached).", "",
        "**index-corpus.** The whole 8,765-snippet corpus, exported as a plain JSONL file "
        f"(id, text), indexed in {r['index_corpus_seconds']} s (vectors reused from the text "
        f"cache: the same code-view processing). The index is a generic corpus "
        f"(`{r['generic_corpus_kind']}`), so the AppsRetrieval hub table does not apply. Its "
        "rankings equal the shipped index's plain-cosine rankings for every query: "
        f"**{'yes' if r['generic_equals_plain_cosine'] else 'NO'}**.", "",
        "`debug907 eval` output:", "", "```", r["eval_output"], "```", ""])


if __name__ == "__main__":
    raise SystemExit(main())
