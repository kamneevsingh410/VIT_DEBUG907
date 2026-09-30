from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.guideline_example import QUERY, SNIPPETS, parse_tool_hits

RECORDED = json.loads(Path("bench/results/guideline_example.json").read_text(encoding="utf-8"))


def test_the_worked_example_routes_to_ranked_search_not_the_order_lookup():
    from retrieval.router import route
    assert "before" in QUERY
    assert route(QUERY) is None
    assert RECORDED["route"] == "ranked search"


def test_the_snippets_are_the_guidelines_three_functions():
    assert list(SNIPPETS) == ["Code#1", "Code#2", "Code#3"]
    assert [s.split("(")[0] for s in SNIPPETS.values()] == [
        "function normalize", "function check", "function perf"]
    assert RECORDED["expected"] == ["Code#1", "Code#2", "Code#3"]


def test_tool_output_parser_reads_the_result_lines():
    text = ("  3 results  |  2.7 ms\n\n   1.  0.5523  corpus:1-5  Code#3  [low]\n"
            "  1 function perf(str) {\n   2.  0.5478  corpus:1-4  Code#2\n")
    assert parse_tool_hits(text) == [("Code#3", 0.5523), ("Code#2", 0.5478)]


def _model_available() -> bool:
    try:
        from retrieval.embed import MODEL_NAME, _find_local, _spec
        spec = _spec(MODEL_NAME)
        return _find_local(MODEL_NAME, spec.revision if spec else None) is not None
    except Exception:
        return False


@pytest.mark.skipif(not _model_available(), reason="needs the downloaded encoder")
def test_the_recorded_ranking_reproduces_with_the_shipped_path(tmp_path):
    from retrieval import workflows
    from retrieval.embed import Embedder
    from retrieval.index import load

    corpus = tmp_path / "snippets.jsonl"
    corpus.write_bytes("".join(json.dumps({"id": k, "text": v}) + "\n"
                               for k, v in SNIPPETS.items()).encode("utf-8"))
    emb = Embedder()
    workflows.index_corpus(corpus, tmp_path / "g.db", embedder=emb)
    index = load(tmp_path / "g.db")
    try:
        hits = workflows.shipped_search(index, QUERY, emb, top_k=3).hits
    finally:
        index.close()
    want = RECORDED["rank"]
    assert [sid for sid, _ in hits] == [d for d, _ in want]
    for (_, s), (_, w) in zip(hits, want):
        assert abs(s - w) < 2e-3
