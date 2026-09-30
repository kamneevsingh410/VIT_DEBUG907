from __future__ import annotations

import json

import numpy as np
import pytest

from retrieval import dense, hubness, workflows
from retrieval.index import build, load
from tests.test_vector_keys import TextEmbedder

CORPUS = {"d1": "def load_config(path):\n    return yaml.safe_load(open(path))\n",
          "d2": "def read_yaml_file(path):\n    return yaml.load(open(path))\n",
          "d3": "def gcd(a, b):\n    return a if b == 0 else gcd(b, a % b)\n"}


@pytest.fixture
def apps_index(tmp_path, monkeypatch):
    db = tmp_path / "apps.db"
    emb = TextEmbedder()
    build(CORPUS, db, version="test", views=("code",), embedder=emb, corpus_kind="apps")
    table = tmp_path / "hub.json"
    table.write_text(json.dumps({"source": "train-excl", "k": 20, "beta": 0.5,
                                 "by_id": {"d1": 0.9, "d2": 0.1, "d3": 0.1}, "hubs": {}}),
                     encoding="utf-8")
    monkeypatch.setattr(hubness, "TABLE", str(table))
    index = load(db)
    yield index, emb
    index.close()


def test_bias_is_minus_beta_hub_aligned_with_the_matrix_rows(apps_index):
    index, _ = apps_index
    bias = hubness.index_bias(index, hubness.load_table())
    expected = {"d1": -0.45, "d2": -0.05, "d3": -0.05}
    for row, sid in enumerate(index.matrix_ids["code"]):
        assert bias[row] == pytest.approx(expected[sid])


def test_biased_scores_below_zero_are_kept(apps_index):
    index, emb = apps_index
    q = emb.encode_one("load the yaml config file")
    bias = {"code": np.full(len(index.matrix_ids["code"]), -5.0, dtype="float32")}
    assert len(dense.search_vector(index, q, 10, bias=bias)) == 3


def test_shipped_search_applies_the_correction_on_an_apps_index(apps_index):
    index, emb = apps_index
    q = "load the yaml config file"
    plain = workflows.shipped_search(index, q, emb, top_k=3, hubness=False).hits
    auto = workflows.shipped_search(index, q, emb, top_k=3).hits
    by = dict(plain)
    for sid, score in auto:
        hub = {"d1": 0.9, "d2": 0.1, "d3": 0.1}[sid]
        assert score == pytest.approx(by[sid] - 0.5 * hub, abs=1e-5)
    assert workflows.hubness_default(index) is True


def test_repo_indexes_are_not_corrected(tmp_path):
    db = tmp_path / "repo.db"
    build({"a.py::f#1": CORPUS["d1"]}, db, version="v1", views=("code",),
          embedder=TextEmbedder(), corpus_kind="repo")
    index = load(db)
    try:
        assert workflows.hubness_default(index) is False
    finally:
        index.close()


def test_a_document_missing_from_the_table_is_an_error(apps_index, tmp_path, monkeypatch):
    index, _ = apps_index
    with pytest.raises(KeyError, match="not in the hub table"):
        hubness.index_bias(index, {"beta": 0.5, "by_id": {"d1": 0.2}})


def test_missing_table_is_a_clear_error(apps_index, monkeypatch, tmp_path):
    index, emb = apps_index
    monkeypatch.setattr(hubness, "TABLE", str(tmp_path / "absent.json"))
    index._hub_bias = None
    with pytest.raises(FileNotFoundError, match="hubness is on"):
        workflows.shipped_search(index, "yaml", emb, top_k=3)


def test_the_committed_table_covers_the_whole_corpus_consistently():
    table = hubness.load_table()
    assert table is not None and table["source"] == "train-excl"
    assert len(table["by_id"]) == 8765 and len(table["hubs"]) >= 8700
    assert set(table["by_id"].values()) <= set(table["hubs"].values())


def test_the_table_is_read_once_not_on_every_query(apps_index, monkeypatch):
    index, emb = apps_index
    reads = []
    import pathlib
    original = pathlib.Path.read_text

    def counting(self, *a, **k):
        if self.name.endswith(".json"):
            reads.append(self.name)
        return original(self, *a, **k)

    monkeypatch.setattr(pathlib.Path, "read_text", counting)
    for _ in range(5):
        workflows.shipped_search(index, "load the yaml config file", emb, top_k=3)
    assert len(reads) <= 1, f"table read {len(reads)} times for 5 queries"
    assert hubness.load_table() is hubness.load_table()


def test_a_table_for_raw_documents_is_refused_on_a_code_view_index(apps_index, tmp_path, monkeypatch):
    index, emb = apps_index
    raw = tmp_path / "raw_table.json"
    raw.write_text(json.dumps({"source": "train-excl", "k": 20, "beta": 0.5, "document_text": "raw",
                               "by_id": {"d1": 0.9, "d2": 0.1, "d3": 0.1}, "hubs": {}}),
                   encoding="utf-8")
    monkeypatch.setattr(hubness, "TABLE", str(raw))
    index._hub_bias = None
    with pytest.raises(hubness.TableMismatch, match="rebuild it"):
        workflows.shipped_search(index, "yaml", emb, top_k=3)
