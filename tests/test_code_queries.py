from __future__ import annotations

from retrieval import workflows
from retrieval.index import load
from tests.test_tool import FakeEmbedder, run_tool, sandbox

SRC_A = """\
function openSettings (router) {
  validateSession()
  router.navigate('app://settings/privacy')
  logEvent('settings')
}
"""
SRC_B = """\
function share (link) {
  logEvent('share')
  validateSession()
  return link + '?ref=app://settings/privacy'
}
"""


def indexed(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "a.js").write_text(SRC_A, encoding="utf-8")
    (root / "b.js").write_text(SRC_B, encoding="utf-8")
    db = tmp_path / "p.db"
    workflows.index_folder(root, db, embedder=FakeEmbedder())
    return db


def test_uses_finds_every_line_with_file_and_line_number(tmp_path):
    index = load(indexed(tmp_path))
    try:
        hits = workflows.uses(index, "app://settings/privacy")
    finally:
        index.close()
    assert [(h.path, h.line) for h in hits] == [("a.js", 3), ("b.js", 4)]


def test_uses_matches_identifiers_as_whole_words(tmp_path):
    index = load(indexed(tmp_path))
    try:
        assert len(workflows.uses(index, "logEvent")) == 2
        assert workflows.uses(index, "Event") == []
    finally:
        index.close()


def test_calls_before_respects_order_within_a_function(tmp_path):
    index = load(indexed(tmp_path))
    try:
        hits = workflows.calls_before(index, "validateSession", "logEvent")
    finally:
        index.close()
    assert [(h.path, h.first_line, h.second_line) for h in hits] == [("a.js", 2, 4)]


def test_tool_commands(tmp_path, sandbox):
    rc, text = run_tool([":uses app://settings/privacy",
                         ":calls validateSession before logEvent", ":quit"],
                        db=indexed(tmp_path))
    assert rc == 0
    assert "a.js:3" in text and "b.js:4" in text and "2 lines" in text
    assert "a.js:2" in text and "1 function" in text


def test_uses_counts_a_line_once_when_snippets_overlap(tmp_path):
    root = tmp_path / "cls"
    root.mkdir()
    (root / "c.js").write_text(
        "class Client {\n"
        "  fetchProfile (user) {\n"
        "    return this.get('https://api.example.com/profile/' + user.id)\n"
        "  }\n"
        "}\n", encoding="utf-8")
    db = tmp_path / "c.db"
    workflows.index_folder(root, db, embedder=FakeEmbedder())
    index = load(db)
    try:
        hits = workflows.uses(index, "api.example.com")
    finally:
        index.close()
    assert [(h.path, h.line) for h in hits] == [("c.js", 3)]


def test_uses_does_not_merge_different_appsretrieval_snippets(tmp_path):
    from retrieval.index import build
    corpus = {"d1": "import sys\nsys.setrecursionlimit(10**6)\n",
              "d2": "import sys\nsys.setrecursionlimit(10**7)\n"}
    db = tmp_path / "apps.db"
    build(corpus, db, views=("code",), embedder=FakeEmbedder(), corpus_kind="apps")
    index = load(db)
    try:
        hits = workflows.uses(index, "setrecursionlimit")
        order = workflows.calls_before(index, "import", "setrecursionlimit")
    finally:
        index.close()
    assert len(hits) == 2 and {h.uid for h in hits} == {"d1", "d2"}
