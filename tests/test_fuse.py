from __future__ import annotations

import pytest

from retrieval.fuse import rrf


def test_single_list_preserves_order():
    ranked = [("a", 9.0), ("b", 5.0), ("c", 0.1)]
    assert [doc for doc, _ in rrf([ranked])] == ["a", "b", "c"]


def test_hand_computed_scores_k60():
    out = dict(rrf([[("a", 0.0), ("b", 0.0), ("c", 0.0)]], k=60))
    assert out["a"] == pytest.approx(1 / 61)
    assert out["b"] == pytest.approx(1 / 62)
    assert out["c"] == pytest.approx(1 / 63)


def test_agreement_beats_single_list_top():
    a = [("x", 1.0), ("y", 0.9)]
    b = [("z", 1.0), ("y", 0.9)]
    out = dict(rrf([a, b], k=1))
    assert out["y"] == pytest.approx(1 / 3 + 1 / 3)
    assert out["x"] == pytest.approx(1 / 2)
    assert out["y"] > out["x"]
    assert [doc for doc, _ in rrf([a, b], k=1)][0] == "y"


def test_raw_scores_are_ignored():
    a = [("a", 1_000_000.0), ("b", 0.0)]
    b = [("b", 0.001), ("a", 0.0)]
    out = dict(rrf([a, b], k=60))
    assert out["a"] == pytest.approx(out["b"])


def test_weights_shift_the_winner():
    a = [("a", 0.0), ("b", 0.0)]
    b = [("b", 0.0), ("a", 0.0)]
    assert [d for d, _ in rrf([a, b], k=60, weights=[2.0, 1.0])][0] == "a"
    assert [d for d, _ in rrf([a, b], k=60, weights=[1.0, 2.0])][0] == "b"


def test_zero_weight_disables_a_list():
    a = [("a", 0.0)]
    b = [("b", 0.0)]
    out = dict(rrf([a, b], k=60, weights=[1.0, 0.0]))
    assert out["b"] == 0.0
    assert out["a"] > 0.0


def test_mismatched_weights_raise():
    with pytest.raises(ValueError):
        rrf([[("a", 0.0)], [("b", 0.0)]], weights=[1.0])


def test_empty_inputs():
    assert rrf([]) == []
    assert rrf([[], []]) == []


def test_ties_break_deterministically_by_id():
    a = [("b", 0.0), ("a", 0.0)]
    b = [("a", 0.0), ("b", 0.0)]
    first = [d for d, _ in rrf([a, b], k=60)]
    assert first == sorted(first)
