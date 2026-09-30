from __future__ import annotations

import random

import pytest

from retrieval.dense import _search_numpy, _search_python
from retrieval.index import Index


def make_index(vectors: dict[str, dict[str, list[float]]]) -> Index:
    import numpy as np

    index = Index(conn=None, doc_count=len(vectors))
    index.vectors = vectors
    index.ids = list(vectors)
    by_view: dict[str, list[tuple[str, list[float]]]] = {}
    for sid, views in vectors.items():
        for view, vec in views.items():
            by_view.setdefault(view, []).append((sid, vec))
    for view, pairs in by_view.items():
        index.matrix_ids[view] = [s for s, _ in pairs]
        index.matrices[view] = np.asarray([v for _, v in pairs], dtype="float32")
    index.has_vectors = True
    return index


def unit(vec):
    import math
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def random_index(n_docs: int, n_views: int, dim: int, seed: int) -> Index:
    rng = random.Random(seed)
    views = ["nl", "code", "full"][:n_views]
    return make_index({
        f"d{i}": {v: unit([rng.gauss(0, 1) for _ in range(dim)]) for v in views}
        for i in range(n_docs)
    })


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("top_k", [1, 5, 25, 100])
def test_pruned_pool_matches_brute_force(seed, top_k):
    index = random_index(120, 2, 32, seed)
    rng = random.Random(seed + 999)
    query = unit([rng.gauss(0, 1) for _ in range(32)])

    fast = _search_numpy(index, query, top_k, None)
    slow = _search_python(index, query, top_k, None)
    assert [d for d, _ in fast] == [d for d, _ in slow]
    for (_, a), (_, b) in zip(fast, slow):
        assert a == pytest.approx(b, abs=1e-5)


def test_adversarial_one_view_dominates():
    index = make_index({
        "hot1":      {"code": unit([1, 0, 0]), "nl": unit([0, 0, 1])},
        "hot2":      {"code": unit([0.9, 0.1, 0]), "nl": unit([0, 0, 1])},
        "d_target":  {"code": unit([0.5, 0.5, 0]), "nl": unit([1, 0, 0])},
    })
    query = unit([1, 0, 0])
    fast = _search_numpy(index, query, 2, None)
    slow = _search_python(index, query, 2, None)
    assert [d for d, _ in fast] == [d for d, _ in slow]
    assert "d_target" in [d for d, _ in fast]


def test_top_k_larger_than_corpus():
    index = random_index(5, 2, 8, seed=1)
    query = unit([1] + [0] * 7)
    assert len(_search_numpy(index, query, 500, None)) <= 5


def test_view_weights_zero_excludes_view():
    index = make_index({
        "a": {"code": unit([1, 0]), "nl": unit([0, 1])},
        "b": {"code": unit([0, 1]), "nl": unit([1, 0])},
    })
    query = unit([1, 0])
    code_only = dict(_search_numpy(index, query, 10, {"code": 1.0, "nl": 0.0}))
    assert code_only["a"] == pytest.approx(1.0, abs=1e-5)
    assert code_only.get("b", 0.0) == pytest.approx(0.0, abs=1e-5)


def test_non_positive_scores_are_dropped():
    index = make_index({
        "same":     {"code": unit([1, 0])},
        "ortho":    {"code": unit([0, 1])},
        "opposite": {"code": unit([-1, 0])},
    })
    got = dict(_search_numpy(index, unit([1, 0]), 10, None))
    assert "same" in got
    assert "opposite" not in got


def test_empty_query_returns_nothing():
    index = random_index(4, 1, 8, seed=2)
    assert _search_numpy(index, [], 10, None) == []


def test_pool_never_exceeds_top_k():
    index = random_index(200, 1, 16, seed=7)
    rng = random.Random(3)
    query = unit([rng.gauss(0, 1) for _ in range(16)])
    for k in (1, 10, 100, 150):
        assert len(_search_numpy(index, query, k, None)) <= min(k, 200)


def test_pool_is_exactly_top_k_when_all_scores_positive():
    rng = random.Random(11)
    docs = {}
    for i in range(50):
        vec = [abs(rng.gauss(0, 1)) for _ in range(8)]
        docs[f"d{i}"] = {"code": unit(vec)}
    index = make_index(docs)
    query = unit([1.0] * 8)
    for k in (1, 10, 25, 50):
        assert len(_search_numpy(index, query, k, None)) == k


def test_non_positive_scores_can_shrink_the_pool():
    index = random_index(200, 1, 16, seed=7)
    rng = random.Random(3)
    query = unit([rng.gauss(0, 1) for _ in range(16)])
    got = _search_numpy(index, query, 200, None)
    assert len(got) < 200
    assert all(score > 0 for _, score in got)
