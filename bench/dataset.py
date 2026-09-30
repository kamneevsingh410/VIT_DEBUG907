from __future__ import annotations

import sys

import json
from collections import Counter
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SAMPLE_PATH = DATA_DIR / "sample_dataset.json"

Corpus = dict[str, str]
Queries = dict[str, str]
Qrels = dict[str, dict[str, int]]


def load_sample() -> tuple[Corpus, Queries, Qrels]:
    payload = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
    return payload["corpus"], payload["queries"], payload["qrels"]


APPS_REVISION = "f22508f96b7a36c2415181ed8bb76f76e04ae2d5"
QRELS_REVISION = "a4fb4d92996bcfe1e0b0af9e97ea4b70b80ec9d5"


def load_appsretrieval(split: str = "test", limit: int | None = None
                       ) -> tuple[Corpus, Queries, Qrels]:
    from datasets import load_dataset

    corpus_ds = load_dataset("CoIR-Retrieval/apps", "corpus", split="corpus",
                             revision=APPS_REVISION)
    queries_ds = load_dataset("CoIR-Retrieval/apps", "queries", split="queries",
                              revision=APPS_REVISION)
    qrels_ds = load_dataset("CoIR-Retrieval/apps-qrels", split=split,
                            revision=QRELS_REVISION)

    corpus: Corpus = {}
    for row in corpus_ds:
        doc_id = str(row.get("_id") or row.get("id"))
        text = row.get("text") or ""
        title = row.get("title") or ""
        corpus[doc_id] = f"{title}\n{text}".strip() if title else text

    queries: Queries = {str(row.get("_id") or row.get("id")): row.get("text", "")
                        for row in queries_ds}

    qrels: Qrels = {}
    for row in qrels_ds:
        qid = str(row.get("query-id") or row.get("query_id"))
        did = str(row.get("corpus-id") or row.get("corpus_id"))
        score = int(row.get("score", 1))
        qrels.setdefault(qid, {})[did] = score

    queries = {qid: text for qid, text in queries.items() if qid in qrels}
    if limit:
        keep = list(queries)[:limit]
        queries = {qid: queries[qid] for qid in keep}
        qrels = {qid: qrels[qid] for qid in keep}
        needed = {did for rels in qrels.values() for did in rels}
        corpus = {did: text for did, text in corpus.items() if did in needed} | \
                 dict(list(corpus.items())[:max(2000, len(needed))])
    return corpus, queries, qrels


FIXTURE_LABEL = "FIXTURE (30-snippet bundled sample, not real data)"


def load(limit: int | None = None, *, fixture: bool = False
         ) -> tuple[Corpus, Queries, Qrels, str]:
    if fixture:
        print(f"[dataset] {FIXTURE_LABEL}", file=sys.stderr)
        corpus, queries, qrels = load_sample()
        return corpus, queries, qrels, FIXTURE_LABEL
    corpus, queries, qrels = load_appsretrieval(limit=limit)
    return corpus, queries, qrels, "CoIR AppsRetrieval (test)"


def describe(corpus: Corpus, queries: Queries, qrels: Qrels) -> str:
    from pipeline.parse import detect_language

    langs = Counter(detect_language(text) for text in list(corpus.values())[:500])
    lengths = sorted(len(t) for t in corpus.values())
    judged = sum(len(r) for r in qrels.values())
    median = lengths[len(lengths) // 2] if lengths else 0
    lines = [
        f"corpus size        {len(corpus)}",
        f"queries            {len(queries)}",
        f"judgements         {judged} ({judged / max(1, len(qrels)):.1f} per query)",
        f"snippet chars      min {lengths[0] if lengths else 0} / "
        f"median {median} / max {lengths[-1] if lengths else 0}",
        f"detected language  {dict(langs)}",
    ]
    return "\n".join(lines)
