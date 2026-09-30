from __future__ import annotations

import sqlite3

import pytest

import cli
from retrieval import prebuilt, workflows
from retrieval.embed import pack, unpack
from retrieval.index import build, load
from tests.test_tool import FakeEmbedder

CORPUS = {
    "d1": "def add(a, b):\n    return a + b\n",
    "d2": "n = int(input())\nprint(sorted(map(int, input().split())))\n",
    "d3": "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a\n",
}


@pytest.fixture()
def exported(tmp_path):
    db = tmp_path / "real.db"
    build(CORPUS, db, version="test", views=("raw",), embedder=FakeEmbedder(), corpus_kind="apps")
    out = tmp_path / "idx.db.gz"
    entry = prebuilt.export_index(db, out)
    return out, entry


def test_export_keeps_only_used_vectors_and_no_foreign_tables(tmp_path, exported):
    out, entry = exported
    assert entry["snippets"] == 3 and entry["views"] == ["raw"] and entry["corpus_kind"] == "apps"
    assert entry["sha256"] == prebuilt.sha256_file(out)
    again = prebuilt.export_index(tmp_path / "real.db", tmp_path / "again.db.gz")
    assert again["sha256"] == entry["sha256"]


def test_a_hash_mismatch_is_refused_before_the_file_is_opened(tmp_path, exported):
    _, entry = exported
    junk = tmp_path / "junk.db.gz"
    junk.write_bytes(b"not gzip, not sqlite")
    dest = tmp_path / "installed.db"
    with pytest.raises(prebuilt.PrebuiltError, match="SHA-256 mismatch.*nothing was opened"):
        prebuilt.import_index(str(junk), dest, expected_sig=FakeEmbedder().signature,
                              manifest=entry, say=lambda *_: None)
    assert not dest.exists() and not list(tmp_path.glob("*.importing"))


def test_a_wrong_encoder_signature_is_refused(tmp_path, exported):
    out, entry = exported
    dest = tmp_path / "installed.db"
    with pytest.raises(prebuilt.PrebuiltError, match="encoder signature"):
        prebuilt.import_index(str(out), dest, expected_sig="Alibaba-NLP/gte-modernbert-base|seq1024|int8=False",
                              manifest=entry, say=lambda *_: None)
    assert not dest.exists() and not list(tmp_path.glob("*.importing"))


def test_a_wrong_view_is_refused_even_with_matching_hashes(tmp_path):
    db = tmp_path / "code.db"
    build(CORPUS, db, version="test", views=("raw",), embedder=FakeEmbedder(), corpus_kind="apps")
    conn = sqlite3.connect(db)
    conn.execute("UPDATE meta SET value = '[\"code\"]' WHERE key = 'views'")
    conn.commit()
    conn.close()
    with pytest.raises(prebuilt.PrebuiltError, match="not the shipped AppsRetrieval index"):
        prebuilt.export_index(db, tmp_path / "x.db.gz")
    with pytest.raises(prebuilt.PrebuiltError, match="view"):
        prebuilt.check_configuration(db, FakeEmbedder().signature, {"snippets": 3})


def test_a_good_import_is_searchable(tmp_path, exported):
    out, entry = exported
    dest = tmp_path / "installed.db"
    prebuilt.import_index(str(out), dest, expected_sig=FakeEmbedder().signature,
                          manifest=entry, say=lambda *_: None)
    index = load(dest)
    try:
        hits = workflows.shipped_search(index, "greatest common divisor gcd", FakeEmbedder(),
                                        top_k=3, hubness=False).hits
    finally:
        index.close()
    assert hits[0][0] == "d3"


def test_an_existing_index_is_not_replaced_without_force(tmp_path, exported):
    out, entry = exported
    dest = tmp_path / "installed.db"
    dest.write_bytes(b"SQLite format 3\x00 my own build")
    with pytest.raises(prebuilt.PrebuiltError, match="--force"):
        prebuilt.import_index(str(out), dest, expected_sig=FakeEmbedder().signature,
                              manifest=entry, say=lambda *_: None)
    assert dest.read_bytes().endswith(b"my own build")
    prebuilt.import_index(str(out), dest, expected_sig=FakeEmbedder().signature,
                          manifest=entry, force=True, say=lambda *_: None)
    assert prebuilt.sha256_file(dest) == entry["db_sha256"]


def test_verify_index_passes_clean_and_catches_a_tampered_vector(tmp_path, exported):
    out, entry = exported
    dest = tmp_path / "installed.db"
    prebuilt.import_index(str(out), dest, expected_sig=FakeEmbedder().signature,
                          manifest=entry, say=lambda *_: None)
    clean = prebuilt.verify_index(dest, sample=0, embedder=FakeEmbedder(), say=lambda *_: None)
    assert clean["ok"] and clean["checked"] == 3 and clean["min_cosine"] > 0.9999
    conn = sqlite3.connect(dest)
    key, view, blob = conn.execute("SELECT hash, view, vector FROM vectors LIMIT 1").fetchone()
    v = unpack(blob)
    v[0], v[1] = v[1], v[0] + 0.5
    conn.execute("UPDATE vectors SET vector = ? WHERE hash = ? AND view = ?", [pack(v), key, view])
    conn.commit()
    conn.close()
    bad = prebuilt.verify_index(dest, sample=0, embedder=FakeEmbedder(), say=lambda *_: None)
    assert not bad["ok"] and bad["min_cosine"] < 0.9999


def test_verify_index_catches_edited_source_text(tmp_path, exported):
    out, entry = exported
    dest = tmp_path / "installed.db"
    prebuilt.import_index(str(out), dest, expected_sig=FakeEmbedder().signature,
                          manifest=entry, say=lambda *_: None)
    conn = sqlite3.connect(dest)
    conn.execute("UPDATE snippets SET content = 'print(42)' WHERE id = 'd1'")
    conn.commit()
    conn.close()
    bad = prebuilt.verify_index(dest, sample=0, embedder=FakeEmbedder(), say=lambda *_: None)
    assert not bad["ok"] and bad["key_mismatch"] == ["d1"]


def test_only_https_urls_and_a_fixed_download_name(tmp_path, exported):
    _, entry = exported
    for bad in ("http://example.com/idx.db.gz", "file:///etc/passwd", "ftp://x/../../evil"):
        with pytest.raises(prebuilt.PrebuiltError, match="https"):
            prebuilt.fetch(bad, entry, download_dir=tmp_path / "dl")
    assert prebuilt.FILE_NAME == "appsretrieval_index.db.gz"


def test_import_can_never_write_over_a_shipped_file(tmp_path, exported):
    out, _ = exported
    artifact = cli.ROOT / "appsretrieval_results.json"
    before = artifact.read_bytes()
    assert cli.main(["import-index", str(out), "--db", str(artifact), "--force"]) == 1
    assert cli.main(["import-index", str(out), "--db", str(cli.ROOT / "data" / "x.db")]) == 1
    assert artifact.read_bytes() == before


def test_the_pinned_manifest_describes_the_shipped_configuration():
    m = prebuilt.load_manifest()
    assert m["views"] == ["raw"] and m["corpus_kind"] == "apps" and m["snippets"] == 8765
    assert m["embedder_sig"] == "Alibaba-NLP/gte-modernbert-base|seq1024|int8=False"
    assert len(m["sha256"]) == 64 and len(m["db_sha256"]) == 64 and m["file"] == prebuilt.FILE_NAME
    assert "no queries" in m["contains"]
