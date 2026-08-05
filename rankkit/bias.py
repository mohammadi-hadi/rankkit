"""Position-bias correction for metrics computed from click logs.

A click means the user *examined* the document and found it relevant. Under the
position-based model those two are independent:

    P(click on d) = p(rank d was shown at) * relevance(d)

so counting clicks measures relevance times examination. Documents shown near
the top collect clicks because they were seen, and any metric built straight
out of clicks flatters whichever ranker produced the log.

Inverse propensity scoring removes that. For any metric that adds up a weight
over the relevant documents,

    metric(policy) = sum over relevant d of lam(rank the policy gives d)

an unbiased estimate from logged clicks is

    sum over clicked d of lam(rank the policy gives d) / p(rank d was SHOWN at)

The propensity is always indexed by the rank in the *logged* ranking, because
that is where examination happened. The new policy's rank only ever enters
through `lam`. Getting those two the wrong way round is the classic bug, so
`test_bias.py` pins it.

Where the propensities come from is the user's problem, and deliberately so:
they cannot be recovered from a plain click log, because a document ranked
first gets more clicks both for being seen more *and* for being better. Fitting
a curve to clicks by rank would silently absorb relevance into the propensity.
Estimate them from a swap or randomisation experiment, or accept the
`(1/rank) ** eta` model and read the sensitivity sweep.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from math import log2

from .data import ClickQuery

WEIGHTS = ("dcg", "recall", "rr")

Ranking = Callable[[ClickQuery], Sequence[str]]


def propensity(rank: int, eta: float = 1.0) -> float:
    """Probability a user examines position `rank`, under the power-law model.

    `eta = 0` means no position bias at all, `eta = 1` is the usual starting
    guess, and larger values mean attention drops off faster.
    """
    if rank < 1:
        raise ValueError(f"rank must be at least 1, got {rank}")
    if eta < 0:
        raise ValueError(f"eta must not be negative, got {eta}")
    return (1.0 / rank) ** eta


def rank_weight(kind: str = "dcg", k: int = 10) -> Callable[[int], float]:
    """The per-rank credit `lam` that defines which metric is being estimated."""
    if kind == "dcg":
        return lambda rank: 1.0 / log2(rank + 1)
    if kind == "recall":
        return lambda rank: 1.0 if rank <= k else 0.0
    if kind == "rr":
        return lambda rank: 1.0 / rank
    raise ValueError(f"unknown weight {kind!r}, expected one of {', '.join(WEIGHTS)}")


def logged_policy(query: ClickQuery) -> Sequence[str]:
    """The ranking that produced the log, as a policy to evaluate."""
    return query.ranking


def policy_from(rankings: dict[str, Sequence[str]]) -> Ranking:
    """A policy that looks each query's ranking up by id, empty when missing."""
    return lambda query: rankings.get(query.qid, ())


@dataclass(frozen=True)
class Estimate:
    """One policy's estimated metric, with and without the correction."""

    naive: float
    ips: float
    snips: float
    per_query: dict[str, float]
    naive_per_query: dict[str, float]


@dataclass(frozen=True)
class Diagnostics:
    """Whether the correction can be trusted on this log."""

    n_queries: int
    n_clicks: int
    ess: float
    max_weight_share: float
    clipped: int
    relevant_per_query: float

    @property
    def ess_share(self) -> float:
        return self.ess / self.n_clicks if self.n_clicks else 0.0


def _weight(query: ClickQuery, doc: str, eta: float, clip: float) -> tuple[float, bool]:
    """Inverse propensity for one click, plus whether the clip bound was hit."""
    p = propensity(query.shown_rank(doc), eta)
    if clip > 0.0 and p < clip:
        return 1.0 / clip, True
    return 1.0 / p, False


def estimate(
    queries: Sequence[ClickQuery],
    policy: Ranking = logged_policy,
    *,
    eta: float = 1.0,
    weight: str = "dcg",
    k: int = 10,
    clip: float = 0.0,
) -> Estimate:
    """Estimate a policy's rank metric from clicks logged under another ranking.

    `naive` treats every click as an unbiased relevance label and is what most
    offline click metrics report. `ips` is the corrected version: both are means
    per query, so they are directly comparable. `snips` self-normalises by the
    total weight instead of the query count, which makes it a mean per
    *relevant document* rather than per query — a different quantity, related by
    the exact identity `snips * relevant_per_query == ips`.
    """
    if not queries:
        raise ValueError("no queries to estimate from")
    lam = rank_weight(weight, k)
    per_query: dict[str, float] = {}
    naive_per_query: dict[str, float] = {}
    weighted_sum = 0.0
    weight_sum = 0.0
    for query in queries:
        ranks = {doc: i for i, doc in enumerate(policy(query), start=1)}
        ips_total = 0.0
        naive_total = 0.0
        for doc in query.clicks:
            # A document the new policy did not retrieve earns no credit.
            credit = lam(ranks[doc]) if doc in ranks else 0.0
            w, _clipped = _weight(query, doc, eta, clip)
            ips_total += credit * w
            naive_total += credit
            weighted_sum += credit * w
            weight_sum += w
        per_query[query.qid] = ips_total
        naive_per_query[query.qid] = naive_total
    n = len(queries)
    return Estimate(
        naive=sum(naive_per_query.values()) / n,
        ips=sum(per_query.values()) / n,
        snips=weighted_sum / weight_sum if weight_sum > 0 else 0.0,
        per_query=per_query,
        naive_per_query=naive_per_query,
    )


def diagnose(
    queries: Sequence[ClickQuery],
    *,
    eta: float = 1.0,
    clip: float = 0.0,
) -> Diagnostics:
    """Effective sample size and weight concentration for a log.

    The weights are what the correction runs on, so a handful of deep clicks
    carrying most of the total weight means the estimate rests on those few
    clicks however many queries the file contains.
    """
    weights: list[float] = []
    clipped = 0
    for query in queries:
        for doc in query.clicks:
            w, hit = _weight(query, doc, eta, clip)
            weights.append(w)
            clipped += hit
    total = sum(weights)
    squared = sum(w * w for w in weights)
    return Diagnostics(
        n_queries=len(queries),
        n_clicks=len(weights),
        ess=(total * total / squared) if squared > 0 else 0.0,
        max_weight_share=(max(weights) / total) if total > 0 else 0.0,
        clipped=clipped,
        relevant_per_query=total / len(queries) if queries else 0.0,
    )


def clicks_by_rank(queries: Iterable[ClickQuery], max_rank: int = 10) -> list[float]:
    """Share of impressions at each rank that were clicked.

    Descriptive only. This curve is *not* a propensity estimate: it mixes
    examination with relevance, because better documents are also placed higher.
    """
    shown = [0] * max_rank
    clicked = [0] * max_rank
    for query in queries:
        for rank, doc in enumerate(query.ranking[:max_rank], start=1):
            shown[rank - 1] += 1
            if doc in query.clicks:
                clicked[rank - 1] += 1
    return [c / s if s else 0.0 for c, s in zip(clicked, shown)]


@dataclass(frozen=True)
class SensitivityRow:
    eta: float
    baseline: float
    candidate: float

    @property
    def delta(self) -> float:
        return self.candidate - self.baseline


def sensitivity(
    queries: Sequence[ClickQuery],
    candidate: Ranking,
    *,
    etas: Sequence[float] = (0.0, 0.5, 1.0, 1.5, 2.0),
    weight: str = "dcg",
    k: int = 10,
    clip: float = 0.0,
    baseline: Ranking = logged_policy,
) -> list[SensitivityRow]:
    """Re-estimate both policies across a range of position-bias strengths.

    Since eta has to be assumed rather than measured, the honest report is not
    one number but the range of conclusions the assumption supports.
    """
    rows = []
    for eta in etas:
        base = estimate(queries, baseline, eta=eta, weight=weight, k=k, clip=clip)
        cand = estimate(queries, candidate, eta=eta, weight=weight, k=k, clip=clip)
        rows.append(SensitivityRow(eta=eta, baseline=base.ips, candidate=cand.ips))
    return rows


def _bisect(delta_at: Callable[[float], float], lo: float, hi: float, tol: float) -> float:
    """Narrow a sign change down to `tol`, assuming delta_at(lo) and (hi) differ."""
    lo_positive = delta_at(lo) > 0
    while hi - lo > tol:
        mid = (lo + hi) / 2.0
        if (delta_at(mid) > 0) == lo_positive:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def crossover(
    rows: Sequence[SensitivityRow],
    delta_at: Callable[[float], float] | None = None,
    *,
    tol: float = 1e-4,
) -> float | None:
    """The eta at which the winner changes, or None if one policy leads throughout.

    None is the answer you want: the conclusion does not depend on the
    assumption. A value close to the eta you assumed is the warning.

    The swept rows only bracket the crossing. Straight-line interpolation
    between two grid points would make the answer depend on how coarse the
    sweep was, so passing `delta_at` — a callable turning an eta into the
    candidate-minus-baseline difference — refines the bracket by bisection.
    """
    for before, after in pairwise(rows):
        if before.delta == 0.0:
            return before.eta
        if (before.delta > 0) != (after.delta > 0):
            if delta_at is not None:
                return _bisect(delta_at, before.eta, after.eta, tol)
            span = before.delta - after.delta
            if span == 0:
                return after.eta
            return before.eta + (after.eta - before.eta) * before.delta / span
    return None
