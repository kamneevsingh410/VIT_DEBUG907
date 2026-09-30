from __future__ import annotations

import pytest

from bench.significance import (
    bootstrap_ci,
    compare_all,
    paired_bootstrap,
    per_query_scores,
)


def constant_scores(n: int, value: float) -> dict[str, float]:
    return {f"q{i}": value for i in range(n)}


def test_zero_variance_gives_zero_width_interval():
    ci = bootstrap_ci(constant_scores(50, 0.4), iterations=200)
    assert ci.mean == pytest.approx(0.4)
    assert ci.low == pytest.approx(0.4)
    assert ci.high == pytest.approx(0.4)


def test_interval_contains_the_mean():
    scores = {f"q{i}": i / 100 for i in range(100)}
    ci = bootstrap_ci(scores, iterations=500)
    assert ci.low <= ci.mean <= ci.high


def test_more_queries_narrows_the_interval():
    import random
    rng = random.Random(0)
    small = {f"q{i}": rng.random() for i in range(50)}
    rng = random.Random(0)
    large = {f"q{i}": rng.random() for i in range(2000)}
    assert bootstrap_ci(large, iterations=400).half_width < \
           bootstrap_ci(small, iterations=400).half_width


def test_deterministic_for_a_fixed_seed():
    scores = {f"q{i}": (i % 7) / 7 for i in range(200)}
    a = bootstrap_ci(scores, iterations=300, seed=42)
    b = bootstrap_ci(scores, iterations=300, seed=42)
    assert (a.low, a.high) == (b.low, b.high)


def test_empty_input_is_safe():
    ci = bootstrap_ci({}, iterations=100)
    assert (ci.mean, ci.low, ci.high, ci.n) == (0.0, 0.0, 0.0, 0)


def test_large_consistent_difference_is_significant():
    a = constant_scores(200, 0.5)
    b = constant_scores(200, 0.1)
    result = paired_bootstrap(a, b, "a", "b", iterations=400)
    assert result.significant
    assert result.diff.mean == pytest.approx(0.4)
    assert result.diff.low > 0


def test_identical_systems_are_not_significant():
    a = {f"q{i}": (i % 5) / 5 for i in range(300)}
    result = paired_bootstrap(a, dict(a), "a", "b", iterations=400)
    assert not result.significant
    assert result.diff.mean == pytest.approx(0.0)


def test_tiny_difference_amid_large_variance_is_not_significant():
    import random
    rng = random.Random(3)
    a, b = {}, {}
    for i in range(300):
        base = rng.random()
        a[f"q{i}"] = base
        b[f"q{i}"] = max(0.0, min(1.0, base + rng.gauss(0, 0.3)))
    assert not paired_bootstrap(a, b, "a", "b", iterations=500).significant


def test_paired_beats_marginal_when_differences_are_consistent():
    import random
    rng = random.Random(7)
    a, b = {}, {}
    for i in range(400):
        base = rng.random()
        a[f"q{i}"] = min(1.0, base + 0.02)
        b[f"q{i}"] = base
    ci_a = bootstrap_ci(a, iterations=500)
    ci_b = bootstrap_ci(b, iterations=500)
    assert ci_a.low < ci_b.high, "marginals should overlap in this construction"
    assert paired_bootstrap(a, b, "a", "b", iterations=500).significant


def test_direction_is_reported_correctly():
    worse = constant_scores(100, 0.1)
    better = constant_scores(100, 0.6)
    assert paired_bootstrap(better, worse, "hi", "lo", iterations=200).diff.mean > 0
    assert paired_bootstrap(worse, better, "lo", "hi", iterations=200).diff.mean < 0


def test_per_query_scores_match_metrics():
    qrels = {"q1": {"d1": 1}, "q2": {"d9": 1}}
    run = {"q1": ["d1", "d2"], "q2": ["d2", "d3"]}
    scores = per_query_scores(run, qrels, k=10)
    assert scores["q1"] == pytest.approx(1.0)
    assert scores["q2"] == pytest.approx(0.0)


def test_compare_all_ranks_by_mean():
    qrels = {f"q{i}": {"gold": 1} for i in range(40)}
    good = {q: ["gold"] for q in qrels}
    bad = {q: ["junk", "gold"] for q in qrels}
    result = compare_all({"good": good, "bad": bad}, qrels, iterations=200)
    assert result["ranking"][0] == "good"
    assert result["pairs"][0].significant
