"""Deciding whether one ranker really beats another.

Both systems are scored on the same queries, so the comparison is paired: query
difficulty cancels and much smaller differences become detectable than two
independent samples would allow.

The statistics are the same paired machinery as
[abeval](https://github.com/mohammadi-hadi/abeval) — sign-flip permutation as
the primary test, percentile bootstrap for the interval — applied to per-query
ranking metrics instead of per-item eval scores. `--per-query` writes the scores
out in abeval's input format if you would rather run the tests there.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from math import ceil, erf, sqrt


def normal_cdf(z: float) -> float:
    """Standard normal CDF, via the error function in the standard library."""
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def normal_quantile(p: float) -> float:
    """Inverse of `normal_cdf`, by bisection — exact enough and dependency-free."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"probability must be strictly between 0 and 1, got {p}")
    lo, hi = -12.0, 12.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if normal_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


@dataclass(frozen=True)
class Comparison:
    """The result of pitting two rankers against each other on shared queries."""

    metric: str
    n: int
    mean_a: float
    mean_b: float
    delta: float
    ci_lo: float
    ci_hi: float
    p_permutation: float
    wins: int
    losses: int
    ties: int
    sd_diff: float
    dropped: list[str]
    movers: list[tuple[str, float]]

    @property
    def significant(self) -> bool:
        return self.ci_lo > 0.0 or self.ci_hi < 0.0

    def queries_needed(self, target: float, level: float = 0.95, power: float = 0.80) -> int | None:
        """Queries required to detect a `target` difference at this per-query spread.

        A planning number, not a verdict on the run you already have: it uses
        only the observed variability, not the observed difference.
        """
        if target <= 0.0:
            raise ValueError(f"target must be positive, got {target}")
        if self.sd_diff <= 0.0:
            return None
        z_level = normal_quantile(1.0 - (1.0 - level) / 2.0)
        z_power = normal_quantile(power)
        return ceil((((z_level + z_power) * self.sd_diff) / target) ** 2)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _sd(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mu = _mean(values)
    return sqrt(sum((v - mu) ** 2 for v in values) / (n - 1))


def bootstrap_ci(
    values: list[float],
    *,
    level: float = 0.95,
    reps: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap interval for a mean over queries.

    Used both for a single run's mean metric and for the mean paired difference
    between two runs; resampling queries is the right unit either way.
    """
    if not values:
        raise ValueError("no values to resample")
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be between 0 and 1, got {level}")
    rng = random.Random(seed)
    n = len(values)
    means = sorted(_mean([values[rng.randrange(n)] for _ in range(n)]) for _ in range(reps))
    lo_idx = int((1.0 - level) / 2.0 * reps)
    hi_idx = min(reps - 1, int((1.0 + level) / 2.0 * reps))
    return means[lo_idx], means[hi_idx]


def permutation_p(deltas: list[float], *, reps: int = 10000, seed: int = 0) -> float:
    """Two-sided sign-flip permutation p-value for the mean paired difference.

    Under the null the two systems are interchangeable on each query, so
    flipping the sign of any difference leaves the distribution unchanged. The
    observed statistic is counted in, which keeps the p-value from ever
    reaching an impossible 0.
    """
    if not deltas:
        raise ValueError("no paired differences to permute")
    rng = random.Random(seed)
    observed = abs(_mean(deltas))
    extreme = 0
    for _ in range(reps):
        flipped = _mean([d if rng.random() < 0.5 else -d for d in deltas])
        if abs(flipped) >= observed:
            extreme += 1
    return (extreme + 1) / (reps + 1)


def paired_compare(
    a: dict[str, float],
    b: dict[str, float],
    *,
    metric: str = "metric",
    level: float = 0.95,
    reps: int = 10000,
    boot_reps: int = 2000,
    seed: int = 0,
    top_movers: int = 3,
) -> Comparison:
    """Compare two systems' per-query scores, keyed by query id.

    Queries scored by only one of the two are dropped and reported rather than
    filled in, since a missing score is not a zero.
    """
    shared = [qid for qid in a if qid in b]
    if not shared:
        raise ValueError("the two systems share no scored queries")
    dropped = sorted(set(a) ^ set(b))
    deltas = [b[qid] - a[qid] for qid in shared]
    movers = sorted(
        ((qid, b[qid] - a[qid]) for qid in shared),
        key=lambda pair: abs(pair[1]),
        reverse=True,
    )[:top_movers]
    lo, hi = bootstrap_ci(deltas, level=level, reps=boot_reps, seed=seed)
    return Comparison(
        metric=metric,
        n=len(shared),
        mean_a=_mean([a[qid] for qid in shared]),
        mean_b=_mean([b[qid] for qid in shared]),
        delta=_mean(deltas),
        ci_lo=lo,
        ci_hi=hi,
        p_permutation=permutation_p(deltas, reps=reps, seed=seed),
        wins=sum(1 for d in deltas if d > 0),
        losses=sum(1 for d in deltas if d < 0),
        ties=sum(1 for d in deltas if d == 0),
        sd_diff=_sd(deltas),
        dropped=dropped,
        movers=[(qid, delta) for qid, delta in movers if delta != 0.0],
    )
