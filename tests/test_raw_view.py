from __future__ import annotations

from pipeline.doc_proc import DocConfig, enrich
from retrieval import workflows
from retrieval.index import build, load
from tests.test_vector_keys import TextEmbedder

SRC = "def load_config(path):\n    # read it\n    return yaml.safe_load(open(path))\n"


def test_raw_view_is_the_source_verbatim():
    doc = enrich(SRC, DocConfig(), views=("raw", "code"))
    assert doc.views["raw"] == SRC.strip()
    assert doc.views["code"] != SRC


def test_a_raw_view_index_embeds_the_source_unchanged_and_searches(tmp_path):
    seen = []

    class Spy(TextEmbedder):
        def encode(self, texts, show_progress=False):
            seen.extend(texts)
            return super().encode(texts, show_progress)

    db = tmp_path / "raw.db"
    build({"a": SRC, "b": "def gcd(a, b):\n    return a if b == 0 else gcd(b, a % b)\n"},
          db, version="v1", views=("raw",), embedder=Spy())
    assert SRC.strip() in seen
    index = load(db)
    try:
        assert set(index.matrices) == {"raw"}
        hits = workflows.shipped_search(index, "load the yaml config", TextEmbedder(), top_k=2).hits
        assert hits and hits[0][0] == "a"
    finally:
        index.close()
