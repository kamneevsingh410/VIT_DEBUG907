from __future__ import annotations

import pytest

import cli
from retrieval.index import build, load
from retrieval.rerank import RerankConfig
from retrieval.search import SearchConfig, search
from tests.test_vector_keys import TextEmbedder

V1 = {
    "auth.js::checkPassword#1": "function checkPassword(user, pw) {\n  return hash(pw) === user.hash;\n}\n",
    "cfg.py::load_config#1": "def load_config(path):\n    return yaml.safe_load(open(path))\n",
}
V2 = {
    "auth.js::checkPassword#1": "function checkPassword(user, pw) {\n  return bcrypt.compare(pw, user.hash);\n}\n",
    "cfg.py::load_config#1": V1["cfg.py::load_config#1"],
    "net.py::fetch_url#1": "def fetch_url(url):\n    return requests.get(url, timeout=5).text\n",
}
DENSE = SearchConfig(use_sparse=False, rerank=RerankConfig.off(),
                     view_weights={"code": 1.0, "nl": 0.0})


def two_versions(tmp_path):
    db = tmp_path / "repo.db"
    emb = TextEmbedder()
    build(V1, db, version="v1", views=("code",), embedder=emb)
    stats = build(V2, db, version="v2", views=("code",), embedder=emb)
    return db, emb, stats


def test_second_version_is_added_not_replacing(tmp_path):
    db, _, _ = two_versions(tmp_path)
    index = load(db, versions="all")
    try:
        assert len(index.ids) == len(V1) + len(V2)
        assert sorted(index.loaded_versions) == ["v1", "v2"]
    finally:
        index.close()


def test_default_load_is_the_latest_version(tmp_path):
    db, _, _ = two_versions(tmp_path)
    index = load(db)
    try:
        assert index.loaded_versions == ["v2"]
        assert sorted(index.ids) == sorted(V2)
    finally:
        index.close()


def test_one_named_version_can_be_searched(tmp_path):
    db, emb, _ = two_versions(tmp_path)
    index = load(db, versions=["v1"])
    try:
        ids = search(index, "fetch a url over http", DENSE, top_k=10, embedder=emb).ids
        assert "net.py::fetch_url#1" not in ids
        assert "bcrypt" not in index.content("auth.js::checkPassword#1")
    finally:
        index.close()


def test_all_versions_searchable_and_collapse_shows_versions(tmp_path):
    db, emb, _ = two_versions(tmp_path)
    index = load(db, versions="all")
    try:
        cfg = SearchConfig(use_sparse=False, rerank=RerankConfig.off(),
                           view_weights={"code": 1.0, "nl": 0.0}, collapse_versions=True)
        result = search(index, "load the yaml config file", cfg, top_k=10, embedder=emb)
        idents = [index.identity_of[uid] for uid in result.ids]
        assert len(idents) == len(set(idents)), "collapse must keep one row per snippet"
        top = next(uid for uid in result.ids if index.identity_of[uid] == "cfg.py::load_config#1")
        assert sorted(result.versions[top]) == ["v1", "v2"]
    finally:
        index.close()


def test_unchanged_code_across_versions_reuses_the_vector(tmp_path):
    db, emb, stats = two_versions(tmp_path)
    assert stats.encoded_vectors == 2
    assert stats.reused_vectors == 1


def test_rebuilding_a_version_replaces_only_that_version(tmp_path):
    db, emb, _ = two_versions(tmp_path)
    build(V1, db, version="v1", views=("code",), embedder=emb)
    index = load(db, versions="all")
    try:
        assert len(index.ids) == len(V1) + len(V2)
    finally:
        index.close()


def test_sparse_search_respects_the_loaded_version(tmp_path):
    db, emb, _ = two_versions(tmp_path)
    index = load(db, versions=["v1"])
    try:
        from retrieval.sparse import search as bm25
        hits = bm25(index, ["requests", "fetch", "url"], top_k=10)
        assert all(uid in index.identity_of for uid, _ in hits)
        assert "net.py::fetch_url#1" not in [uid for uid, _ in hits]
    finally:
        index.close()


def test_index_is_incremental_by_default(tmp_path, monkeypatch):
    parser = cli.build_parser()
    args = parser.parse_args(["index"])
    assert args.reset is False and args.drop_vectors is False


def test_reset_keeps_stored_vectors(tmp_path):
    db = tmp_path / "i.db"
    emb = TextEmbedder()
    build(V1, db, version="v1", views=("code",), embedder=emb)
    calls = emb.calls
    stats = build(V1, db, version="v1", views=("code",), embedder=emb, reset=True)
    assert emb.calls == calls and stats.reused_vectors == len(V1)


def test_drop_vectors_is_explicit(tmp_path):
    db = tmp_path / "i.db"
    emb = TextEmbedder()
    build(V1, db, version="v1", views=("code",), embedder=emb)
    stats = build(V1, db, version="v1", views=("code",), embedder=emb, drop_vectors=True)
    assert stats.encoded_vectors == len(V1)


def test_stats_separate_model_work_from_cache_hits(tmp_path):
    class CachingEmbedder(TextEmbedder):
        def encode(self, texts, show_progress=False):
            self.last_cache_hits = 1
            self.last_model_encoded = len(texts) - 1
            return super().encode(texts)

    db = tmp_path / "i.db"
    stats = build(V1, db, version="v1", views=("code",), embedder=CachingEmbedder())
    assert stats.encoded_vectors == 2
    assert stats.text_cache_hits == 1
    assert stats.model_encoded == 1


def test_evolution_marks_changed_and_unchanged_versions(tmp_path):
    from retrieval.versions import evolution, evolution_label, history_diffs
    db, _, _ = two_versions(tmp_path)
    index = load(db, versions="all")
    try:
        uid = next(u for u, i in index.identity_of.items() if i == "auth.js::checkPassword#1")
        assert evolution_label(evolution(index, uid)) == "v1 != v2"
        (old, new, diff), = history_diffs(index, uid)
        assert (old, new) == ("v1", "v2")
        assert any(line.startswith("+") and "bcrypt" in line for line in diff)
        cfg = next(u for u, i in index.identity_of.items() if i == "cfg.py::load_config#1")
        assert evolution_label(evolution(index, cfg)) == "v1 = v2"
        assert history_diffs(index, cfg) == []
        fetch = next(u for u, i in index.identity_of.items() if i == "net.py::fetch_url#1")
        assert evolution_label(evolution(index, fetch)) == "v2"
    finally:
        index.close()


def test_prefer_newest_only_breaks_ties_between_identical_versions(tmp_path):
    from retrieval.versions import collapse_versions
    db, _, _ = two_versions(tmp_path)
    index = load(db, versions="all")
    try:
        by = {(index.identity_of[u], index.version_of[u]): u for u in index.ids}
        cfg1, cfg2 = by[("cfg.py::load_config#1", "v1")], by[("cfg.py::load_config#1", "v2")]
        pw1 = by[("auth.js::checkPassword#1", "v1")]
        hits = [(cfg1, 0.9), (cfg2, 0.9), (pw1, 0.5)]
        plain, _ = collapse_versions(index, hits)
        newest, _ = collapse_versions(index, hits, prefer_newest=True)
        assert plain[0][0] == cfg1 and newest[0][0] == cfg2
        assert newest[1][0] == pw1
    finally:
        index.close()


def test_a_function_that_moves_lines_keeps_its_lineage(tmp_path):
    from retrieval.versions import collapse_versions, evolution, evolution_label
    v1 = {"a.py::parse#1": "def parse(s):\n    return s.split(',')\n",
          "big.py::Big#1": "class Big:\n    def one(self): return 1\n",
          "big.py::Big#60": "    def two(self): return 2\n"}
    v2 = {"a.py::parse#9": "def parse(s):\n    return [x.strip() for x in s.split(',')]\n",
          "big.py::Big#1": v1["big.py::Big#1"], "big.py::Big#60": v1["big.py::Big#60"]}
    db = tmp_path / "moved.db"
    emb = TextEmbedder()
    build(v1, db, version="v1", views=("code",), embedder=emb)
    build(v2, db, version="v2", views=("code",), embedder=emb)
    index = load(db, versions="all")
    try:
        by = {(index.identity_of[u], index.version_of[u]): u for u in index.ids}
        moved = by[("a.py::parse#9", "v2")]
        assert evolution_label(evolution(index, moved)) == "v1 != v2"
        hits = [(moved, 0.9), (by[("a.py::parse#1", "v1")], 0.8),
                (by[("big.py::Big#1", "v2")], 0.7), (by[("big.py::Big#60", "v2")], 0.6)]
        kept, _ = collapse_versions(index, hits)
        assert [index.identity_of[u] for u, _ in kept] == ["a.py::parse#9", "big.py::Big#1",
                                                           "big.py::Big#60"]
    finally:
        index.close()
