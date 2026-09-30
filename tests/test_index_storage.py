from __future__ import annotations

import sqlite3

import pytest

from retrieval.index import build, load

CORPUS = {f"s{i}": f"def f{i}(x):\n    return x + {i}\n" for i in range(20)}


def test_load_of_a_missing_path_does_not_create_a_file(tmp_path):
    db = tmp_path / "out" / "real.db"
    with pytest.raises(SystemExit):
        load(db)
    assert not db.exists()
    assert not db.parent.exists()


def test_load_of_a_non_index_file_leaves_it_untouched(tmp_path):
    db = tmp_path / "other.db"
    sqlite3.connect(db).close()
    before = db.read_bytes()
    with pytest.raises(SystemExit):
        load(db)
    assert db.read_bytes() == before, "load() must not write the schema"


def test_build_uses_the_rollback_journal_and_leaves_no_side_files(tmp_path):
    db = tmp_path / "real.db"
    build(CORPUS, db, views=("code",), use_embeddings=False)
    conn = sqlite3.connect(db)
    try:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
    finally:
        conn.close()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["real.db"]


def test_load_reads_what_build_wrote(tmp_path):
    db = tmp_path / "real.db"
    build(CORPUS, db, views=("code",), use_embeddings=False)
    index = load(db)
    try:
        assert len(index.ids) == len(CORPUS)
    finally:
        index.close()


def test_verify_index_does_not_create_a_file(tmp_path):
    from retrieval import workflows
    db = tmp_path / "missing.db"
    with pytest.raises(FileNotFoundError):
        workflows.verify_index(db)
    assert not db.exists()
    assert workflows.index_is_usable(db) is False
