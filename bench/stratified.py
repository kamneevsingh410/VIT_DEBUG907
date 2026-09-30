from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

SAMPLE_PATH = Path("data/stratified_1000.json")
DEFAULT_SIZE = 1000
N_STRATA = 10


@dataclass
class ValidationResult:
    preserved: bool
    full_ranking: list[str]
    sample_ranking: list[str]
    max_abs_delta: float
    kendall_tau: float
    details: dict

    def __str__(self) -> str:
        verdict = "PRESERVED" if self.preserved else "NOT PRESERVED"
        return (f"ranking {verdict}\n"
                f"  full   {self.full_ranking}\n"
                f"  sample {self.sample_ranking}\n"
                f"  kendall tau {self.kendall_tau:+.3f}   "
                f"max |delta| {self.max_abs_delta:.4f}")


def build_sample(queries: dict[str, str], size: int = DEFAULT_SIZE,
                 strata: int = N_STRATA, seed: int = 0) -> list[str]:
    if size >= len(queries):
        return sorted(queries)
    ordered = sorted(queries, key=lambda q: (len(queries[q]), q))
    per_stratum = len(ordered) // strata
    rng = random.Random(seed)
    picked: list[str] = []
    for i in range(strata):
        lo = i * per_stratum
        hi = len(ordered) if i == strata - 1 else (i + 1) * per_stratum
        block = ordered[lo:hi]
        take = round(size * len(block) / len(ordered))
        picked.extend(rng.sample(block, min(take, len(block))))
    if len(picked) > size:
        picked = rng.sample(picked, size)
    elif len(picked) < size:
        remaining = [q for q in ordered if q not in set(picked)]
        picked.extend(rng.sample(remaining, min(size - len(picked), len(remaining))))
    return sorted(picked)


def kendall_tau(a: list[str], b: list[str]) -> float:
    shared = [x for x in a if x in b]
    if len(shared) < 2:
        return 1.0
    pos_b = {name: i for i, name in enumerate(b)}
    concordant = discordant = 0
    for i in range(len(shared)):
        for j in range(i + 1, len(shared)):
            left = pos_b[shared[i]] - pos_b[shared[j]]
            if left < 0:
                concordant += 1
            elif left > 0:
                discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total if total else 1.0


def validate(full_scores: dict[str, float], sample_scores: dict[str, float],
             tolerance: float = 0.02,
             separable_pairs: list[tuple[str, str]] | None = None
             ) -> ValidationResult:
    full_ranking = sorted(full_scores, key=lambda k: -full_scores[k])
    sample_ranking = sorted(sample_scores, key=lambda k: -sample_scores[k])
    deltas = {k: abs(sample_scores.get(k, 0.0) - v) for k, v in full_scores.items()}
    max_delta = max(deltas.values()) if deltas else 0.0
    tau = kendall_tau(full_ranking, sample_ranking)

    if separable_pairs is None:
        preserved = full_ranking == sample_ranking
        inverted: list[tuple[str, str]] = []
    else:
        inverted = [
            (a, b) for a, b in separable_pairs
            if a in sample_scores and b in sample_scores
            and (full_scores[a] > full_scores[b]) !=
                (sample_scores[a] > sample_scores[b])
        ]
        preserved = not inverted

    return ValidationResult(
        preserved=(preserved and max_delta <= tolerance),
        full_ranking=full_ranking,
        sample_ranking=sample_ranking,
        max_abs_delta=max_delta,
        kendall_tau=tau,
        details={"per_config_delta": deltas,
                 "full": full_scores, "sample": sample_scores,
                 "separable_pairs": separable_pairs,
                 "inverted_separable_pairs": inverted},
    )


def save(query_ids: list[str], validation: ValidationResult | None = None,
         path: Path = SAMPLE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"query_ids": query_ids, "size": len(query_ids)}
    if validation is not None:
        payload["validation"] = {
            "ranking_preserved": validation.preserved,
            "kendall_tau": round(validation.kendall_tau, 4),
            "max_abs_delta": round(validation.max_abs_delta, 5),
            "full_ranking": validation.full_ranking,
            "sample_ranking": validation.sample_ranking,
            "per_config_delta": {k: round(v, 5) for k, v in
                                 validation.details["per_config_delta"].items()},
        }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load(path: Path = SAMPLE_PATH) -> list[str]:
    if not path.exists():
        raise SystemExit(f"No sample at {path}. Run: python -m bench.stratified --build")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("validation", {}).get("ranking_preserved", False):
        print(f"WARNING: {path} has not been validated as rank-preserving; "
              "use the full split for decisions.")
    return payload["query_ids"]


def describe(queries: dict[str, str], picked: list[str]) -> str:
    import statistics
    all_lens = sorted(len(v) for v in queries.values())
    sub_lens = sorted(len(queries[q]) for q in picked)

    def pct(values, p):
        return values[min(len(values) - 1, int(p * len(values)))]

    rows = [("n", len(all_lens), len(sub_lens)),
            ("min", all_lens[0], sub_lens[0]),
            ("p25", pct(all_lens, .25), pct(sub_lens, .25)),
            ("median", statistics.median(all_lens), statistics.median(sub_lens)),
            ("p75", pct(all_lens, .75), pct(sub_lens, .75)),
            ("p95", pct(all_lens, .95), pct(sub_lens, .95)),
            ("max", all_lens[-1], sub_lens[-1])]
    out = ["query-length distribution (characters)",
           f"  {'stat':<8}{'full':>10}{'sample':>10}"]
    for name, a, b in rows:
        out.append(f"  {name:<8}{a:>10.0f}{b:>10.0f}")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stratified query sample")
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    from bench.dataset import load_appsretrieval
    _corpus, queries, _qrels = load_appsretrieval()
    picked = build_sample(queries, args.size, seed=args.seed)
    print(describe(queries, picked))
    save(picked)
    print(f"\nwrote {SAMPLE_PATH} ({len(picked)} queries, UNVALIDATED)")
    print("Validate it with tools/validate_sample.py before relying on it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
