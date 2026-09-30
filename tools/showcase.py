from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.dataset import load_appsretrieval
from bench.stratified import load as load_stratified
from pipeline.tokens import split_identifier
from retrieval import workflows
from retrieval.embed import MODELS, Embedder, set_threads
from retrieval.index import build, load

RESULTS = Path("bench/results")
JINA_DB = Path("out/showcase_jina.db")
STOP = {"the", "a", "an", "of", "to", "and", "in", "is", "for", "on", "you", "it", "that",
        "be", "are", "with", "as", "this", "each", "by", "or", "if", "print", "input", "int",
        "i", "n", "m", "k", "x", "y", "s", "t", "a", "b", "c", "d", "j", "one", "line"}


def words(text: str) -> set[str]:
    out = set()
    for tok in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text):
        for part in split_identifier(tok):
            part = part.lower()
            if len(part) > 1 and part not in STOP:
                out.add(part)
    return out


def overlap(query: str, code: str) -> float:
    q, c = words(query), words(code)
    return len(q & c) / max(1, len(q | c))


def main(argv: list[str] | None = None) -> int:
    set_threads(8)
    corpus, queries, qrels = load_appsretrieval()
    picked = set(load_stratified())
    sample = {q: t for q, t in queries.items() if q in picked and q in qrels}

    spec = MODELS["jina"]
    jina = Embedder(name=spec.name, max_seq_length=512, batch_size=32, prefix=spec.doc_prefix)
    if not JINA_DB.exists():
        stats = build(corpus, JINA_DB, version="test", views=("code",), embedder=jina,
                      corpus_kind="apps")
        print(f"jina index: {stats.model_encoded} texts through the model, "
              f"{stats.text_cache_hits} from the text cache")
    gte = Embedder()
    shipped, jina_index = load(Path("out/real.db")), load(JINA_DB)
    candidates = []
    try:
        gte.encode(list(sample.values()))
        jina.encode(list(sample.values()))
        for qid, text in sample.items():
            gold = next(iter(qrels[qid]))
            ours = workflows.shipped_search(shipped, text, gte, top_k=10)
            if not ours.ids or ours.ids[0] != gold:
                continue
            theirs = workflows.shipped_search(jina_index, text, jina, top_k=100).ids
            jrank = theirs.index(gold) + 1 if gold in theirs else None
            if jrank is not None and jrank <= 10:
                continue
            label, margin = workflows.confidence(ours.hits)
            candidates.append({"qid": qid, "gold": gold, "overlap": overlap(text, corpus[gold]),
                               "shared_words": sorted(words(text) & words(corpus[gold])),
                               "jina_rank": jrank, "jina_top1": theirs[0] if theirs else None,
                               "confidence": label, "margin": margin})
    finally:
        shipped.close(); jina_index.close()

    candidates.sort(key=lambda c: c["overlap"])
    top = candidates[:3]
    for c in top:
        c["query"] = queries[c["qid"]]
        c["gold_code"] = corpus[c["gold"]]
        c["jina_code"] = corpus[c["jina_top1"]] if c["jina_top1"] else ""
    report = {"eligible": len(candidates), "sample": len(sample), "examples": top}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "showcase.json").write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                           encoding="utf-8")
    write_md(report)
    print(f"{len(candidates)} queries: shipped #1, jina outside the top 10")
    for c in top:
        print(c["qid"], round(c["overlap"], 3), c["shared_words"], "jina rank", c["jina_rank"])
    return 0


def first_lines(text: str, n: int) -> str:
    return "\n".join(text.strip().splitlines()[:n])


def write_md(r: dict) -> None:
    lines = ["# Tough-query showcase", "",
             f"Validated 1,000-query sample. **{r['eligible']}** queries where the shipped model "
             "(gte-modernbert-base) ranks the solution #1 **and** the previous encoder "
             "(jina-v2-base-code) leaves it outside its top 10. Below: the three of those with the "
             "least vocabulary in common between problem statement and solution (Jaccard over "
             "identifier sub-tokens, stop words removed).", ""]
    for i, c in enumerate(r["examples"], 1):
        jr = f"at rank {c['jina_rank']}" if c["jina_rank"] else "not in its top 100"
        statement = " ".join(c["query"].split())[:420]
        lines += [f"## {i}. {c['qid']}: word overlap {c['overlap']:.3f}", "",
                  f"> {statement}…", "",
                  f"**Shipped model: #1**, confidence {c['confidence']} (margin {c['margin']:.3f}). "
                  f"Words shared with the solution: {', '.join(c['shared_words']) or 'none'}.", "",
                  "```python", first_lines(c["gold_code"], 12), "```", "",
                  f"**jina-v2-base-code: the solution is {jr}**; its #1 was `{c['jina_top1']}`:",
                  "", "```python", first_lines(c["jina_code"], 6), "```", ""]
    (RESULTS / "showcase.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
