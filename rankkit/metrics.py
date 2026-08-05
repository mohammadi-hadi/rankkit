"""Per-query ranking metrics.

Every metric here returns a value for a *single* query, or `None` when the
metric is undefined for it — which happens when a query has no relevant
documents at all. Those queries are skipped rather than scored 0, matching
trec_eval, and the CLI reports how many were skipped.

Conventions worth knowing, because implementations disagree:

* The ideal ranking behind NDCG is built from *all* labelled grades, not just
  the ones that were retrieved, so failing to retrieve a relevant document
  costs you.
* Average precision divides by the total number of relevant documents, as in
  trec_eval's `map_cut`. AP@k is therefore below 1 when more than k documents
  are relevant.
* Precision@k always divides by k, even when the ranking is shorter than k.
* ERR needs a maximum grade for the grading scale; it is taken from the whole
  file rather than per query, so queries stay comparable.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import log2

from .data import Run

METRICS = ("ndcg", "mrr", "map", "recall", "precision", "hit", "err")

GAINS = ("exp", "linear")


def _gain(grade: float, gain: str) -> float:
    if gain == "exp":
        return 2.0**grade - 1.0
    if gain == "linear":
        return grade
    raise ValueError(f"unknown gain {gain!r}, expected one of {', '.join(GAINS)}")


def _check_k(k: int) -> int:
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    return k


def dcg(gains: Sequence[float], k: int) -> float:
    """Discounted cumulative gain of an already-ordered list of gains."""
    _check_k(k)
    return sum(g / log2(i + 2) for i, g in enumerate(gains[:k]))


def ndcg(run: Run, k: int = 10, gain: str = "exp") -> float | None:
    """DCG@k divided by the DCG of the best ranking the labels allow."""
    _check_k(k)
    ideal_grades = sorted(run.relevance.values(), reverse=True)
    ideal = dcg([_gain(g, gain) for g in ideal_grades], k)
    if ideal <= 0.0:
        return None
    actual = dcg([_gain(run.gain(doc), gain) for doc in run.ranking], k)
    return actual / ideal


def mrr(run: Run, k: int = 10) -> float | None:
    """Reciprocal rank of the first relevant document, 0 if none in the top k."""
    _check_k(k)
    if run.n_relevant == 0:
        return None
    for i, doc in enumerate(run.ranking[:k], start=1):
        if run.gain(doc) > 0:
            return 1.0 / i
    return 0.0


def average_precision(run: Run, k: int = 10) -> float | None:
    """Mean of the precisions measured at each relevant document in the top k."""
    _check_k(k)
    total_relevant = run.n_relevant
    if total_relevant == 0:
        return None
    hits = 0
    running = 0.0
    for i, doc in enumerate(run.ranking[:k], start=1):
        if run.gain(doc) > 0:
            hits += 1
            running += hits / i
    return running / total_relevant


def recall_at_k(run: Run, k: int = 10) -> float | None:
    """Share of the relevant documents that made it into the top k."""
    _check_k(k)
    if run.n_relevant == 0:
        return None
    found = sum(1 for doc in run.ranking[:k] if run.gain(doc) > 0)
    return found / run.n_relevant


def precision_at_k(run: Run, k: int = 10) -> float | None:
    """Share of the top k slots filled by a relevant document."""
    _check_k(k)
    if run.n_relevant == 0:
        return None
    found = sum(1 for doc in run.ranking[:k] if run.gain(doc) > 0)
    return found / k


def hit_rate(run: Run, k: int = 10) -> float | None:
    """1 when at least one relevant document is in the top k, else 0."""
    _check_k(k)
    if run.n_relevant == 0:
        return None
    return 1.0 if any(run.gain(doc) > 0 for doc in run.ranking[:k]) else 0.0


def err(run: Run, k: int = 10, max_grade: float | None = None) -> float | None:
    """Expected reciprocal rank: how far a user reads before they are satisfied.

    Follows Chapelle et al. (2009). `max_grade` is the top of the grading scale;
    pass the same value for every query so the numbers stay comparable.
    """
    _check_k(k)
    if run.n_relevant == 0:
        return None
    top = max_grade if max_grade is not None else max(run.relevance.values(), default=0.0)
    if top <= 0:
        return None
    total = 0.0
    survival = 1.0
    for i, doc in enumerate(run.ranking[:k], start=1):
        stop = (2.0 ** run.gain(doc) - 1.0) / (2.0**top)
        total += survival * stop / i
        survival *= 1.0 - stop
    return total


def max_grade_of(runs: Sequence[Run]) -> float:
    """Highest relevance grade anywhere in the file, used as ERR's scale top."""
    return max((max(r.relevance.values(), default=0.0) for r in runs), default=0.0)


def score(
    run: Run,
    metric: str,
    *,
    k: int = 10,
    gain: str = "exp",
    max_grade: float | None = None,
) -> float | None:
    """Dispatch one named metric for one query."""
    if metric == "ndcg":
        return ndcg(run, k, gain)
    if metric == "mrr":
        return mrr(run, k)
    if metric == "map":
        return average_precision(run, k)
    if metric == "recall":
        return recall_at_k(run, k)
    if metric == "precision":
        return precision_at_k(run, k)
    if metric == "hit":
        return hit_rate(run, k)
    if metric == "err":
        return err(run, k, max_grade)
    raise ValueError(f"unknown metric {metric!r}, expected one of {', '.join(METRICS)}")


def scores(
    runs: Sequence[Run],
    metric: str,
    *,
    k: int = 10,
    gain: str = "exp",
    max_grade: float | None = None,
) -> dict[str, float]:
    """Per-query values keyed by query id, leaving out the undefined ones."""
    if max_grade is None and metric == "err":
        max_grade = max_grade_of(runs)
    out: dict[str, float] = {}
    for run in runs:
        value = score(run, metric, k=k, gain=gain, max_grade=max_grade)
        if value is not None:
            out[run.qid] = value
    return out
