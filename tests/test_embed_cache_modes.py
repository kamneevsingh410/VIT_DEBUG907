from __future__ import annotations

from retrieval import embed
from retrieval.embed import Embedder, TextCache


def fake_model(counter):
    def encode_raw(self, texts, show_progress):
        counter.append(len(texts))
        return [[float(len(t)), 1.0] for t in texts]
    return encode_raw


def test_write_mode_runs_the_model_and_fills_the_cache(tmp_path, monkeypatch):
    store = TextCache(str(tmp_path / "c.db"))
    monkeypatch.setattr(embed, "get_cache", lambda *a, **k: store)
    calls: list[int] = []
    monkeypatch.setattr(Embedder, "_encode_raw", fake_model(calls))

    writer = Embedder(name="fake", cache="write")
    writer.encode(["alpha", "beta"])
    writer.encode(["alpha", "beta"])
    assert calls == [2, 2] and writer.last_model_encoded == 2

    reader = Embedder(name="fake")
    reader.encode(["alpha", "beta"])
    assert calls == [2, 2] and reader.last_cache_hits == 2


def test_cache_off_neither_reads_nor_writes(tmp_path, monkeypatch):
    store = TextCache(str(tmp_path / "c.db"))
    monkeypatch.setattr(embed, "get_cache", lambda *a, **k: store)
    calls: list[int] = []
    monkeypatch.setattr(Embedder, "_encode_raw", fake_model(calls))
    Embedder(name="fake", cache=False).encode(["alpha"])
    Embedder(name="fake").encode(["alpha"])
    assert calls == [1, 1]


def test_an_interrupted_encode_resumes_from_the_cache(tmp_path, monkeypatch):
    import pytest
    store = TextCache(str(tmp_path / "c.db"))
    monkeypatch.setattr(embed, "get_cache", lambda *a, **k: store)
    calls: list[int] = []
    good = fake_model(calls)

    def dies_on_second_step(self, texts, show_progress):
        if len(calls) == 1:
            raise KeyboardInterrupt
        return good(self, texts, show_progress)

    monkeypatch.setattr(Embedder, "_encode_raw", dies_on_second_step)
    texts = [f"text number {i:03d}" + "x" * i for i in range(40)]
    emb = Embedder(name="fake", batch_size=1)
    with pytest.raises(KeyboardInterrupt):
        emb.encode(texts)
    assert calls == [16]
    monkeypatch.setattr(Embedder, "_encode_raw", good)
    out = emb.encode(texts, show_progress=True)
    assert calls == [16, 16, 8]
    assert emb.last_cache_hits == 16 and emb.last_model_encoded == 24
    assert out == [[float(len(t)), 1.0] for t in texts]


def test_steps_follow_the_longest_first_order_and_whole_batches(tmp_path, monkeypatch):
    store = TextCache(str(tmp_path / "c.db"))
    monkeypatch.setattr(embed, "get_cache", lambda *a, **k: store)
    seen: list[list[str]] = []
    monkeypatch.setattr(Embedder, "_encode_raw",
                        lambda self, texts, show_progress: seen.append(list(texts)) or
                        [[float(len(t)), 1.0] for t in texts])
    texts = ["a" * n for n in (3, 9, 1, 7, 5, 9)]
    Embedder(name="fake", batch_size=1).encode(texts)
    flat = [t for step in seen for t in step]
    assert flat == sorted(set(texts), key=len, reverse=True)


def test_progress_line_reports_the_time_left():
    from retrieval.embed import format_duration, progress_line
    line = progress_line(2304, 8754, 740.0)
    assert line.startswith("encoded 2,304 / 8,754 texts (26%)") and "texts/s" in line
    assert "left" in line and progress_line(10, 10, 5.0).endswith("done")
    assert format_duration(42) == "42 s" and format_duration(2100) == "35 min 00 s"
    assert format_duration(3 * 3600 + 60) == "3 h 01 min"
