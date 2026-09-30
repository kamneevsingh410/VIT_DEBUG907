from __future__ import annotations

import random
from dataclasses import dataclass

from bench.metrics import ndcg_at_k, reciprocal_rank

Run = dict[str, list[str]]
Qrels = dict[str, dict[str, int]]


def per_query_scores(run: Run, qrels: Qrels, k: int = 10,
                     metric: str = "ndcg") -> dict[str, float]:
    fn = (lambda r, g: ndcg_at_k(r, g, k)) if metric == "ndcg" else reciprocal_rank
    return {qid: fn(run.get(qid, []), rel) for qid, rel in qrels.items()}


@dataclass
class Interval:
    mean: float
    low: float
    high: float
    n: int

    @property
    def half_width(self) -> float:
        return (self.high - self.low) / 2

    def __str__(self) -> str:
        return f"{self.mean:.4f} [{self.low:.4f}, {self.high:.4f}]"


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(pct * (len(ordered) - 1)))))
    return ordered[idx]


def bootstrap_ci(scores: dict[str, float], iterations: int = 500,
                 confidence: float = 0.95, seed: int = 0) -> Interval:
    qids = list(scores)
    if not qids:
        return Interval(0.0, 0.0, 0.0, 0)
    rng = random.Random(seed)
    n = len(qids)
    values = [scores[q] for q in qids]
    means: list[float] = []
    for _ in range(iterations):
        total = 0.0
        for _ in range(n):
            total += values[rng.randrange(n)]
        means.append(total / n)
    tail = (1.0 - confidence) / 2
    return Interval(mean=sum(values) / n,
                    low=_percentile(means, tail),
                    high=_percentile(means, 1 - tail),
                    n=n)


@dataclass
class Comparison:
    name_a: str
    name_b: str
    diff: Interval
    significant: bool
    p_value: float

    def __str__(self) -> str:
        verdict = "SIGNIFICANT" if self.significant else "not significant"
        return (f"{self.name_a} - {self.name_b}: {self.diff}  "
                f"p~{self.p_value:.3f}  {verdict}")


def paired_bootstrap(a: dict[str, float], b: dict[str, float],
                     name_a: str = "A", name_b: str = "B",
                     iterations: int = 500, confidence: float = 0.95,
                     seed: int = 0) -> Comparison:
    shared = [q for q in a if q in b]
    if not shared:
        return Comparison(name_a, name_b, Interval(0, 0, 0, 0), False, 1.0)

    rng = random.Random(seed)
    n = len(shared)
    diffs = [a[q] - b[q] for q in shared]
    observed = sum(diffs) / n

    means: list[float] = []
    for _ in range(iterations):
        total = 0.0
        for _ in range(n):
            total += diffs[rng.randrange(n)]
        means.append(total / n)

    tail = (1.0 - confidence) / 2
    interval = Interval(mean=observed, low=_percentile(means, tail),
                        high=_percentile(means, 1 - tail), n=n)
    crossings = sum(1 for m in means if (m <= 0) == (observed > 0))
    p_value = min(1.0, 2.0 * crossings / max(1, iterations))
    significant = (interval.low > 0) or (interval.high < 0)
    return Comparison(name_a, name_b, interval, significant, p_value)


def compare_all(runs: dict[str, Run], qrels: Qrels, k: int = 10,
                iterations: int = 500, metric: str = "ndcg",
                seed: int = 0) -> dict:
    scores = {name: per_query_scores(run, qrels, k, metric)
              for name, run in runs.items()}
    marginals = {name: bootstrap_ci(s, iterations, seed=seed)
                 for name, s in scores.items()}

    names = sorted(marginals, key=lambda n: -marginals[n].mean)
    pairs: list[Comparison] = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            pairs.append(paired_bootstrap(scores[a], scores[b], a, b,
                                          iterations, seed=seed))
    return {"marginals": marginals, "pairs": pairs, "ranking": names}


def report(result: dict, metric_name: str = "NDCG@10") -> str:
    lines = [f"{metric_name} with 95% bootstrap confidence intervals",
             f"(resampling the query set {500} times)", ""]
    for name in result["ranking"]:
        lines.append(f"  {name:<26} {result['marginals'][name]}")

    lines += ["", "Marginal intervals overlapping does NOT settle a comparison;",
              "the paired test below does, because per-query difficulty cancels.",
              ""]
    for comparison in result["pairs"]:
        lines.append(f"  {comparison}")

    significant = [c for c in result["pairs"] if c.significant]
    lines += ["", f"{len(significant)} of {len(result['pairs'])} pairs are "
                  f"separable at 95%."]
    if not significant:
        lines.append("None of these configurations is distinguishable from the "
                     "others on this query set. Prefer the simplest or fastest.")
    return "\n".join(lines)
