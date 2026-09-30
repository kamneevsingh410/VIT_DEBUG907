from __future__ import annotations

import hashlib
import math
import re
import sqlite3

import pytest

from pipeline.chunk import ChunkConfig
from retrieval.index import build, load


class TextEmbedder:

    name = "fake/text-encoder"
    max_seq_length = 1024
    quantize = False

    def __init__(self, prefix: str = ""):
        self.prefix = prefix
        self.calls = 0

    @property
    def signature(self) -> str:
        base = f"{self.name}|seq{self.max_seq_length}|int8={self.quantize}"
        return f"{base}|pfx={self.prefix}" if self.prefix else base

    def encode(self, texts, show_progress=False):
        self.calls += len(texts)
        return [self.vec(self.prefix + t) for t in texts]

    def encode_one(self, text):
        return self.encode([text])[0]

    @staticmethod
    def vec(text: str) -> list[float]:
        v = [0.0] * 16
        for tok in re.findall(r"\w+", text.lower()):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % 16] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]


CORPUS = {
    "a": "def load_config(path):\n    data = read_yaml_file(path)\n"
         "    validate_schema(data)\n    return merge_defaults(data)\n",
    "b": "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a\n",
}


def stored_code_vector(db, snippet_id):
    index = load(db)
    try:
        return index.vectors[snippet_id]["code"]
    finally:
        index.close()


def test_chunk_config_change_is_not_served_stale_vectors(tmp_path):
    db = tmp_path / "i.db"
    emb = TextEmbedder()
    build(CORPUS, db, views=("code",), embedder=emb)
    full = stored_code_vector(db, "a")

    small = ChunkConfig(budget_tokens=4)
    build(CORPUS, db, views=("code",), embedder=emb, chunk_config=small)
    after = stored_code_vector(db, "a")
    assert after != full, "a rebuild with new preprocessing reused the old vector"


def test_same_text_is_reused_not_reencoded(tmp_path):
    db = tmp_path / "i.db"
    emb = TextEmbedder()
    build(CORPUS, db, views=("code",), embedder=emb)
    first = emb.calls
    stats = build(CORPUS, db, views=("code",), embedder=emb)
    assert emb.calls == first and stats.encoded_vectors == 0
    assert stats.reused_vectors == len(CORPUS)


def test_prefix_is_part_of_the_encoder_signature(tmp_path):
    db = tmp_path / "i.db"
    build(CORPUS, db, views=("code",), embedder=TextEmbedder())
    with_prefix = TextEmbedder(prefix="passage: ")
    stats = build(CORPUS, db, views=("code",), embedder=with_prefix)
    assert stats.encoded_vectors == len(CORPUS), "prefix change reused old vectors"


def test_index_from_before_the_fix_must_be_rebuilt(tmp_path):
    db = tmp_path / "old.db"
    build(CORPUS, db, views=("code",), embedder=TextEmbedder())
    conn = sqlite3.connect(db)
    conn.execute("DELETE FROM meta WHERE key = 'schema'")
    conn.commit(); conn.close()
    with pytest.raises(SystemExit, match="rebuild"):
        load(db)
