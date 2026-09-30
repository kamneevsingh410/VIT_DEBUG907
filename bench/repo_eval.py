from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from retrieval.rerank import RerankConfig
from retrieval.search import SearchConfig

MIN_WORDS = 5
CONFIGS: dict[str, SearchConfig] = {
    "shipped (code view, dense)": SearchConfig(use_sparse=False, rerank=RerankConfig.off(),
                                               view_weights={"code": 1.0, "nl": 0.0}),
    "nl view, dense": SearchConfig(use_sparse=False, rerank=RerankConfig.off(),
                                   view_weights={"code": 0.0, "nl": 1.0}),
    "multi-view (max of nl, code)": SearchConfig(use_sparse=False, rerank=RerankConfig.off()),
    "code view + BM25 (RRF)": SearchConfig(rerank=RerankConfig.off(), sparse_weight=0.3,
                                           view_weights={"code": 1.0, "nl": 0.0}),
}


@dataclass
class Query:
    qid: str
    text: str
    answer: str
    source: str


def _clean(lines: list[str]) -> str:
    out = []
    for line in lines:
        s = line.strip()
        s = re.sub(r"^(/\*\*?|\*/|\*|//+)", "", s).strip()
        s = re.sub(r"\*/$", "", s).strip()
        if s.startswith("@") or s.startswith("eslint") or not s:
            continue
        out.append(s)
    return " ".join(out)


def leading_comment(lines: list[str], start: int) -> str:
    i = start - 1
    if i >= 0 and not lines[i].strip():
        i -= 1
    if i < 0:
        return ""
    if lines[i].strip().endswith("*/"):
        j = i
        while j >= 0 and "/*" not in lines[j]:
            j -= 1
        return _clean(lines[j:i + 1]) if j >= 0 else ""
    block = []
    while i >= 0 and lines[i].strip().startswith("//"):
        block.insert(0, lines[i])
        i -= 1
    return _clean(block)


def pseudo_queries(root: Path, corpus: dict[str, str]) -> tuple[list[Query], dict[str, int]]:
    stats = {"functions": len(corpus), "with_comment": 0, "too_short": 0, "leaked": 0,
             "duplicate_text": 0}
    cache: dict[str, list[str]] = {}
    found: dict[str, list[str]] = {}
    for sid in corpus:
        path, rest = sid.split("::", 1)
        name, _, line = rest.partition("#")
        if not path.endswith((".js", ".mjs", ".cjs", ".ts")) or not line.isdigit() or name == "window":
            continue
        if path not in cache:
            try:
                cache[path] = (root / path).read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                cache[path] = []
        text = leading_comment(cache[path], int(line) - 1)
        if not text:
            continue
        stats["with_comment"] += 1
        if len(text.split()) < MIN_WORDS:
            stats["too_short"] += 1
            continue
        found.setdefault(text, []).append(sid)

    blob = "\n".join(corpus.values())
    normalise = lambda s: " ".join(s.split()).lower()
    flat = normalise(re.sub(r"(//+|/\*\*?|\*/|^\s*\*)", " ", blob, flags=re.M))
    queries = []
    for text, sids in sorted(found.items()):
        if len(sids) > 1:
            stats["duplicate_text"] += len(sids)
            continue
        if normalise(text) in flat:
            stats["leaked"] += 1
            continue
        queries.append(Query(f"p{len(queries):04d}", text, sids[0], "pseudo"))
    stats["kept"] = len(queries)
    return queries, stats


def handwritten_queries(path: Path, corpus: dict[str, str]) -> tuple[list[Query], list[str]]:
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    by_name: dict[str, list[str]] = {}
    for sid in corpus:
        by_name.setdefault(sid.rsplit("#", 1)[0], []).append(sid)
    out, missing = [], []
    for i, item in enumerate(spec["queries"]):
        hits = by_name.get(item["answer"], [])
        if not hits:
            missing.append(item["answer"])
            continue
        out.append(Query(f"h{i:02d}", item["query"], sorted(hits)[0], "handwritten"))
    return out, missing


def evaluate_repo(root: Path, db: Path, queries_file: Path | None, embedder,
                  results: Path | None = Path("bench/results"),
                  show_progress: bool = False) -> dict:
    import time

    from bench.metrics import evaluate
    from bench.significance import paired_bootstrap, per_query_scores
    from pipeline.repo import extract_repo
    from retrieval.index import build, load
    from retrieval.search import search

    corpus = extract_repo(root)
    pseudo, pseudo_stats = pseudo_queries(root, corpus)
    hand, missing = (handwritten_queries(queries_file, corpus) if queries_file else ([], []))
    queries = pseudo + hand
    started = time.perf_counter()
    stats = build(corpus, db, version="eval", views=("nl", "code"), embedder=embedder,
                  corpus_kind="repo", show_progress=show_progress)
    build_s = time.perf_counter() - started

    index = load(db)
    runs: dict[str, dict[str, list[str]]] = {}
    try:
        embedder.encode([q.text for q in queries])
        for name, cfg in CONFIGS.items():
            runs[name] = {q.qid: search(index, q.text, cfg, top_k=100, embedder=embedder).ids
                          for q in queries}
    finally:
        index.close()

    qrels_all = {q.qid: {q.answer: 1} for q in queries}
    subsets = {"all": qrels_all,
               "pseudo (comments)": {q.qid: {q.answer: 1} for q in pseudo},
               "handwritten": {q.qid: {q.answer: 1} for q in hand}}
    base_name = next(iter(CONFIGS))
    table = {}
    for subset, qrels in subsets.items():
        if not qrels:
            continue
        table[subset] = {}
        base = per_query_scores({q: runs[base_name][q] for q in qrels}, qrels)
        for name in CONFIGS:
            run = {q: runs[name][q] for q in qrels}
            s = evaluate(run, qrels)
            row = {"ndcg_at_10": s.ndcg_at_10, "mrr": s.mrr, "recall_at_100": s.recall_at_100,
                   "n": len(qrels)}
            if name != base_name:
                cmp = paired_bootstrap(per_query_scores(run, qrels), base, name, base_name,
                                       iterations=1000)
                row.update(diff=cmp.diff.mean, low=cmp.diff.low, high=cmp.diff.high,
                           p=cmp.p_value, significant=cmp.significant)
            table[subset][name] = row
    report = {"root": root.as_posix(), "functions": len(corpus), "pseudo_stats": pseudo_stats,
              "handwritten": len(hand), "handwritten_missing": missing,
              "build": {"seconds": round(build_s, 1), **stats.as_dict()}, "table": table,
              "queries": [{"qid": q.qid, "source": q.source, "text": q.text, "answer": q.answer,
                           "shipped_rank": (runs[base_name][q.qid].index(q.answer) + 1
                                            if q.answer in runs[base_name][q.qid] else None)}
                          for q in queries]}
    if results is not None:
        results.mkdir(parents=True, exist_ok=True)
        (results / "repo_eval.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (results / "repo_eval.md").write_text(render_md(report), encoding="utf-8")
    return report


def render_md(r: dict) -> str:
    ps = r["pseudo_stats"]
    lines = ["# Held-out retrieval on real code", "",
             f"Source: `{r['root']}` ({r['functions']:,} functions, index-repo extractor). "
             "Not CoIR AppsRetrieval: nothing here was used to choose the shipped config.", "",
             "Queries:",
             f"- **pseudo**: {ps['kept']} functions whose leading comment becomes the query "
             f"({ps['with_comment']} had one; dropped: {ps['too_short']} under {MIN_WORDS} words, "
             f"{ps['leaked']} whose text still appears inside an indexed snippet, "
             f"{ps['duplicate_text']} shared by several functions). The comment is not part of "
             "the answer's indexed text.",
             f"- **handwritten**: {r['handwritten']} developer questions "
             "(`data/npm_queries.json`)" + (f"; unresolved: {r['handwritten_missing']}"
                                            if r["handwritten_missing"] else "") + ".", "",
             "Paired bootstrap (1,000 resamples) against the shipped configuration; "
             "**bold** = significant at 95%.", ""]
    for subset, rows in r["table"].items():
        n = next(iter(rows.values()))["n"]
        lines += [f"## {subset} ({n} queries)", "",
                  "| configuration | NDCG@10 | MRR | R@100 | vs shipped |", "|---|---:|---:|---:|---|"]
        for name, row in rows.items():
            if "diff" in row:
                mark = "**" if row["significant"] else ""
                delta = (f"{mark}{row['diff']:+.4f}{mark} [{row['low']:+.4f}, {row['high']:+.4f}] "
                         f"p≈{row['p']:.3f}")
            else:
                delta = "(baseline)"
            lines.append(f"| {name} | {row['ndcg_at_10']:.4f} | {row['mrr']:.4f} | "
                         f"{row['recall_at_100']:.4f} | {delta} |")
        lines.append("")
    b = r["build"]
    lines += [f"Index build (wall clock; the committed run was **measured under load**): "
              f"{b['seconds']} s, {b['vectors_new']} new vectors, "
              f"{b['texts_model_encoded']} texts through the model, "
              f"{b['texts_from_cache']} from the text cache.", ""]
    return "\n".join(lines)
