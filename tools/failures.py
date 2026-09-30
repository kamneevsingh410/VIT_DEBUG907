from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from bench.dataset import load_appsretrieval
from retrieval import workflows
from retrieval.embed import Embedder, set_threads
from retrieval.index import load

RESULTS = Path("bench/results")
BUCKETS = [("rank 1", 1, 1), ("rank 2-10", 2, 10), ("rank 11-100", 11, 100),
           ("not in top 100", 101, 10**9)]


def bucket(rank: int) -> str:
    return next(name for name, lo, hi in BUCKETS if lo <= rank <= hi)


def quartiles(values: list[float]) -> list[float]:
    return [float(np.percentile(values, p)) for p in (25, 50, 75)]


def ndcg10(rank: int) -> float:
    import math
    return 1.0 / math.log2(rank + 1) if rank <= 10 else 0.0


def by_tercile(rows: list[dict], key: str) -> list[dict]:
    ordered = sorted(rows, key=lambda r: r[key])
    third = len(ordered) // 3
    parts = [ordered[:third], ordered[third:2 * third], ordered[2 * third:]]
    return [{"tercile": i + 1, "lo": p[0][key], "hi": p[-1][key], "n": len(p),
             "ndcg_at_10": sum(ndcg10(r["rank"]) for r in p) / len(p)} for i, p in enumerate(parts)]


MISS_SAMPLE = 50
LABELS_FILE = Path("data/failure_labels.json")


def by_band(rows: list[dict], key: str, edges: list[float]) -> list[dict]:
    out = []
    for lo, hi in zip(edges, edges[1:]):
        group = [r for r in rows if lo <= r[key] < hi]
        if group:
            out.append({"lo": lo, "hi": hi, "n": len(group),
                        "top1": sum(r["rank"] == 1 for r in group) / len(group),
                        "top10": sum(r["rank"] <= 10 for r in group) / len(group)})
    return out


def main(argv: list[str] | None = None) -> int:
    set_threads(8)
    corpus, queries, qrels = load_appsretrieval()
    queries = {q: t for q, t in queries.items() if q in qrels}
    index = load(Path("out/real.db"))
    emb = Embedder()
    rows = []
    try:
        if len(index.ids) != len(corpus):
            print("out/real.db is not the full AppsRetrieval index; run: debug907 index")
            return 1
        emb.encode(list(queries.values()))
        for qid, text in queries.items():
            ids = workflows.shipped_search(index, text, emb, top_k=100).ids
            gold = next(iter(qrels[qid]))
            rank = ids.index(gold) + 1 if gold in ids else 10**9
            top = ids[0] if ids else None
            dup = None
            if top and top != gold:
                a = index.vectors.get(top, {}).get("code")
                b = index.vectors.get(gold, {}).get("code")
                if a and b:
                    dup = float(np.dot(a, b))
            rows.append({"qid": qid, "gold": gold, "top": top, "rank": rank,
                         "cyrillic": sum("Ѐ" <= ch <= "ӿ" for ch in text) > 20,
                         "query_chars": len(text), "gold_chars": len(corpus[gold]),
                         "top_gold_cos": dup, "same_text": top is not None and top != gold
                         and corpus.get(top) == corpus[gold]})
    finally:
        index.close()

    n = len(rows)
    report = {
        "n": n,
        "buckets": {name: sum(bucket(r["rank"]) == name for r in rows) for name, *_ in BUCKETS},
        "by_gold_length": by_band(rows, "gold_chars", [0, 500, 1000, 2000, 4000, 8000, 10**9]),
        "by_query_length": by_band(rows, "query_chars", [0, 800, 1200, 1600, 2400, 3500, 10**9]),
        "by_language": {
            name: {"n": len(g), "top1": sum(r["rank"] == 1 for r in g) / max(1, len(g)),
                   "top10": sum(r["rank"] <= 10 for r in g) / max(1, len(g))}
            for name, g in (("Russian (Cyrillic) statement", [r for r in rows if r["cyrillic"]]),
                            ("English statement", [r for r in rows if not r["cyrillic"]]))},
        "wrong_top1": sum(1 for r in rows if r["rank"] != 1),
        "wrong_top1_same_text": sum(1 for r in rows if r["same_text"]),
        "wrong_top1_near_dup": sum(1 for r in rows if r["top_gold_cos"] is not None
                                   and r["top_gold_cos"] >= 0.95),
        "gold_chars_quartiles_hit": quartiles([r["gold_chars"] for r in rows if r["rank"] <= 10]),
        "gold_chars_quartiles_miss": quartiles([r["gold_chars"] for r in rows if r["rank"] > 100]),
    }
    from collections import Counter
    wrong = Counter(r["top"] for r in rows if r["rank"] != 1 and r["top"])
    report["hubs"] = {"wrong_top1_queries": sum(wrong.values()),
                      "distinct_wrong_top1": len(wrong),
                      "top": [{"doc": d, "times": c, "head": corpus[d][:90].strip()}
                              for d, c in wrong.most_common(5)],
                      "share_from_docs_seen_5plus": sum(c for c in wrong.values() if c >= 5)
                      / max(1, sum(wrong.values()))}
    report["ndcg_by_query_tercile"] = by_tercile(rows, "query_chars")
    report["ndcg_by_doc_tercile"] = by_tercile(rows, "gold_chars")

    import random
    outside = sorted((r for r in rows if r["rank"] > 10), key=lambda r: r["qid"])
    sample = random.Random(0).sample(outside, MISS_SAMPLE)
    report["miss_sample"] = [r["qid"] for r in sample]
    if "--dump" in (argv or sys.argv[1:]):
        for r in sample:
            print(json.dumps({"qid": r["qid"], "rank": r["rank"], "gold_chars": r["gold_chars"],
                              "query": queries[r["qid"]][:500],
                              "gold": corpus[r["gold"]][:500],
                              "top1": corpus[r["top"]][:300] if r["top"] else ""},
                             ensure_ascii=False))
        return 0
    labels = json.loads(LABELS_FILE.read_text(encoding="utf-8")) if LABELS_FILE.exists() else {}
    report["miss_categories"] = labels.get("categories", {})
    report["miss_labels"] = {q: labels.get("labels", {}).get(q) for q in report["miss_sample"]}
    by_qid = {r["qid"]: r for r in rows}
    report["miss_rows"] = {q: {"rank": by_qid[q]["rank"], "gold_chars": by_qid[q]["gold_chars"]}
                           for q in report["miss_sample"]}

    misses = sorted((r for r in rows if r["rank"] > 100), key=lambda r: r["qid"])[:5]
    report["examples"] = [{"qid": r["qid"], "gold": r["gold"], "top": r["top"],
                           "query": queries[r["qid"]][:220].strip(),
                           "gold_head": corpus[r["gold"]][:160].strip(),
                           "top_head": corpus[r["top"]][:160].strip() if r["top"] else ""}
                          for r in misses]
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "failures.json").write_text(json.dumps({**report, "rows": rows}, indent=1),
                                           encoding="utf-8")
    write_md(report)
    print(json.dumps({k: v for k, v in report.items() if k != "examples"}, indent=1))
    print(f"wrote {RESULTS / 'failures.md'}")
    return 0


def write_md(r: dict) -> None:
    n = r["n"]
    lines = ["# Failure analysis: shipped config, full test split", "",
             f"{n:,} CoIR AppsRetrieval test queries, one relevant solution each; shipped "
             "search (gte-modernbert-base, fp32, seq 1024, code view, dense only).", "",
             "## Where the relevant solution lands", "", "| rank | queries | share |",
             "|---|---:|---:|"]
    for name, count in r["buckets"].items():
        lines.append(f"| {name} | {count:,} | {count / n:.1%} |")
    lines += ["", "## By length of the relevant solution", "",
              "The encoder reads at most 1,024 tokens (~4,000 characters of code); anything "
              "after that is invisible to it.", "",
              "| gold solution chars | queries | top-1 | in top 10 |", "|---|---:|---:|---:|"]
    for b in r["by_gold_length"]:
        hi = "∞" if b["hi"] >= 10**9 else f"{b['hi']:,}"
        lines.append(f"| {b['lo']:,} – {hi} | {b['n']:,} | {b['top1']:.0%} | {b['top10']:.0%} |")
    q_hit, q_miss = r["gold_chars_quartiles_hit"], r["gold_chars_quartiles_miss"]
    lines += ["", f"Gold length quartiles (25/50/75%): found in top 10 "
              f"{q_hit[0]:.0f} / {q_hit[1]:.0f} / {q_hit[2]:.0f} chars; not in top 100 "
              f"{q_miss[0]:.0f} / {q_miss[1]:.0f} / {q_miss[2]:.0f} chars.", "",
              "## By query length", "", "| query chars | queries | top-1 | in top 10 |",
              "|---|---:|---:|---:|"]
    for b in r["by_query_length"]:
        hi = "∞" if b["hi"] >= 10**9 else f"{b['hi']:,}"
        lines.append(f"| {b['lo']:,} – {hi} | {b['n']:,} | {b['top1']:.0%} | {b['top10']:.0%} |")
    lines += ["", "## By language of the problem statement", "",
              "gte-modernbert-base is an English model; a few statements are in Russian.", "",
              "| statement | queries | top-1 | in top 10 |", "|---|---:|---:|---:|"]
    for name, b in r["by_language"].items():
        lines.append(f"| {name} | {b['n']:,} | {b['top1']:.0%} | {b['top10']:.0%} |")
    lines += ["", "## NDCG@10 by tercile", "",
              "| tercile | query chars | NDCG@10 | gold solution chars | NDCG@10 |",
              "|---:|---|---:|---|---:|"]
    for q, d in zip(r["ndcg_by_query_tercile"], r["ndcg_by_doc_tercile"]):
        lines.append(f"| {q['tercile']} | {q['lo']:,} – {q['hi']:,} | {q['ndcg_at_10']:.4f} | "
                     f"{d['lo']:,} – {d['hi']:,} | {d['ndcg_at_10']:.4f} |")
    labels, cats = r["miss_labels"], r["miss_categories"]
    if cats and all(labels.values()):
        from collections import Counter
        counts = Counter(labels.values())
        lines += ["", f"## 50 misses, categorised (answer not in the top 10)", "",
                  f"A fixed random sample (seed 0) of {len(labels)} from the "
                  f"{r['buckets']['rank 11-100'] + r['buckets']['not in top 100']:,} queries "
                  "whose answer is outside the top 10; each "
                  "was read (statement, gold solution, returned #1) and given one primary cause. "
                  "Labels: `data/failure_labels.json`: manual judgement, not a measurement.", "",
                  "| cause | count | what it means | examples |", "|---|---:|---|---|"]
        for cat, count in counts.most_common():
            ex = [q for q, c in labels.items() if c == cat][:3]
            lines.append(f"| {cat.replace('_', ' ')} | {count} | {cats[cat]} | "
                         + ", ".join(f"{q} (rank {r['miss_rows'][q]['rank'] if r['miss_rows'][q]['rank'] < 10**9 else '>100'})"
                                     for q in ex) + " |")
    h = r["hubs"]
    lines += ["", "## Hubs: generic snippets that win for unrelated queries", "",
              f"The {h['wrong_top1_queries']:,} wrong #1s are {h['distinct_wrong_top1']:,} distinct "
              f"snippets; {h['share_from_docs_seen_5plus']:.0%} of wrong #1s come from snippets that "
              "are the wrong #1 for 5 or more queries.", "",
              "| snippet | wrong #1 for | starts with |", "|---|---:|---|"]
    for t in h["top"]:
        head = " ".join(t["head"].split())[:70].replace("|", "\\|")
        lines.append(f"| {t['doc']} | {t['times']} queries | `{head}` |")
    w = r["wrong_top1"]
    lines += ["", "## Is the wrong #1 really wrong?", "",
              f"Of {w:,} queries whose #1 is not the judged solution: "
              f"**{r['wrong_top1_same_text']}** have a #1 with *identical* text to the gold "
              f"(a duplicate in the corpus: unwinnable, the qrels judge only one copy), and "
              f"**{r['wrong_top1_near_dup']}** have a #1 whose code vector is ≥ 0.95 cosine to "
              "the gold's (near-identical solutions to closely related problems).", "",
              "## Five complete misses (relevant solution not in the top 100)", ""]
    for e in r["examples"]:
        lines += [f"**{e['qid']}**: gold `{e['gold']}`, returned #1 `{e['top']}`", "",
                  f"> {e['query'][:220]}…".replace("\n", " "), "",
                  "gold starts: `" + e["gold_head"].splitlines()[0][:100] + "`  ",
                  "#1 starts: `" + (e["top_head"].splitlines()[0][:100] if e["top_head"] else "") + "`",
                  ""]
    (RESULTS / "failures.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
