from __future__ import annotations

from pipeline.query_proc import QueryConfig, process
from retrieval import workflows
from retrieval.index import build, load
from retrieval.rerank import CATEGORY_AFFINITY, RerankConfig, structural_score
from tests.test_tool import FakeEmbedder

CODE = {"a.py::solve#1": "def solve(n):\n    return sorted(range(n))\n"}


def _category_boost(index, sid):
    query = process("sort the numbers", QueryConfig())
    index.categories[sid] = sorted(CATEGORY_AFFINITY[query.category])[0]
    cfg = RerankConfig(boost_identifier=0.0, length_penalty_cap=0.0)
    return structural_score(0.0, index, query, sid, cfg)


def test_category_affinity_applies_to_appsretrieval_indexes(tmp_path):
    db = tmp_path / "apps.db"
    build(CODE, db, views=("code",), embedder=FakeEmbedder(), corpus_kind="apps")
    index = load(db)
    try:
        assert index.corpus_kind == "apps"
        assert _category_boost(index, "a.py::solve#1") > 0
    finally:
        index.close()


def test_category_affinity_is_off_for_a_repo_index(tmp_path):
    folder = tmp_path / "proj"
    folder.mkdir()
    (folder / "a.py").write_text(CODE["a.py::solve#1"], encoding="utf-8")
    db = tmp_path / "repo.db"
    workflows.index_folder(folder, db, embedder=FakeEmbedder())
    index = load(db)
    try:
        assert index.corpus_kind == "repo"
        sid = index.ids[0]
        assert _category_boost(index, sid) == 0.0
    finally:
        index.close()


def test_input_section_stripping_is_off_by_default():
    assert QueryConfig().strip_boilerplate is False
    assert workflows.shipped_config().query.strip_boilerplate is False
    assert workflows.shipped_config().rerank.structural is False
