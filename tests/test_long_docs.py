from __future__ import annotations

import numpy as np

from tools.long_docs import mean_pool, token_windows


class FakeTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}

    def decode(self, ids):
        return " ".join(ids)


def test_short_text_is_one_window():
    assert token_windows("a b c", FakeTokenizer(), window=5, stride=3) == ["a b c"]


def test_long_text_is_covered_by_overlapping_windows():
    text = " ".join(f"t{i}" for i in range(12))
    ws = token_windows(text, FakeTokenizer(), window=5, stride=4)
    assert ws[0] == "t0 t1 t2 t3 t4" and ws[1].startswith("t4")
    assert ws[-1].endswith("t11")
    covered = {tok for w in ws for tok in w.split()}
    assert covered == set(text.split())


def test_mean_pool_is_unit_length():
    v = mean_pool(np.array([[1.0, 0.0], [0.0, 1.0]]))
    assert np.allclose(v, [2 ** -0.5, 2 ** -0.5])
