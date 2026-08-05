import random
from math import log2

import pytest

from rankkit.bias import (
    SensitivityRow,
    clicks_by_rank,
    crossover,
    diagnose,
    estimate,
    logged_policy,
    policy_from,
    propensity,
    rank_weight,
    sensitivity,
)
from rankkit.data import ClickQuery


def test_propensity_decays_with_rank():
    assert propensity(1, eta=1.0) == 1.0
    assert propensity(2, eta=1.0) == pytest.approx(0.5)
    assert propensity(2, eta=2.0) == pytest.approx(0.25)


def test_eta_zero_means_no_position_bias():
    assert propensity(9, eta=0.0) == 1.0


def test_propensity_rejects_nonsense_arguments():
    with pytest.raises(ValueError, match="rank must be at least 1"):
        propensity(0)
    with pytest.raises(ValueError, match="eta must not be negative"):
        propensity(1, eta=-0.5)


def test_rank_weights():
    assert rank_weight("dcg")(1) == pytest.approx(1.0)
    assert rank_weight("dcg")(3) == pytest.approx(1.0 / 2.0)
    assert rank_weight("recall", k=2)(2) == 1.0
    assert rank_weight("recall", k=2)(3) == 0.0
    assert rank_weight("rr")(4) == pytest.approx(0.25)
    with pytest.raises(ValueError, match="unknown weight"):
        rank_weight("ndcg")


# The bug this whole module exists to avoid: the propensity belongs to the rank
# the document was SHOWN at, and the new policy's rank only enters through lam.
# Here the click landed at shown rank 3 (weight 3) while the candidate promotes
# it to rank 1 (credit 1). Indexing the propensity by the candidate's rank
# instead would give 1.0.
PINNED = [ClickQuery(qid="q1", ranking=("a", "b", "c"), clicks=("c",))]


def test_propensity_is_indexed_by_the_logged_rank_not_the_new_one():
    est = estimate(PINNED, lambda q: ("c", "a", "b"), eta=1.0, weight="dcg")
    assert est.ips == pytest.approx(3.0)
    assert est.naive == pytest.approx(1.0)


def test_a_document_the_candidate_dropped_earns_nothing():
    est = estimate(PINNED, lambda q: ("a", "b"), eta=1.0, weight="dcg")
    assert est.ips == 0.0
    assert est.naive == 0.0


def test_evaluating_the_logging_policy_itself():
    est = estimate(PINNED, logged_policy, eta=1.0, weight="dcg")
    assert est.naive == pytest.approx(1.0 / log2(4))
    assert est.ips == pytest.approx(3.0 / log2(4))


def test_policy_from_looks_rankings_up_by_query_id():
    est = estimate(PINNED, policy_from({"q1": ("c", "a", "b")}), eta=1.0)
    assert est.ips == pytest.approx(3.0)
    missing = estimate(PINNED, policy_from({"other": ("c",)}), eta=1.0)
    assert missing.ips == 0.0


def test_no_position_bias_makes_the_correction_a_no_op():
    est = estimate(PINNED, lambda q: ("c", "a", "b"), eta=0.0)
    assert est.ips == pytest.approx(est.naive)


TWO = [
    ClickQuery(qid="q1", ranking=("a", "b"), clicks=("a", "b")),
    ClickQuery(qid="q2", ranking=("a", "b"), clicks=("b",)),
]


def test_snips_is_the_per_relevant_document_view_of_ips():
    est = estimate(TWO, logged_policy, eta=1.0, weight="dcg")
    diag = diagnose(TWO, eta=1.0)
    assert est.snips * diag.relevant_per_query == pytest.approx(est.ips)


def test_diagnostics_report_weight_concentration():
    diag = diagnose([TWO[0]], eta=1.0)
    # weights are 1 at rank 1 and 2 at rank 2
    assert diag.n_clicks == 2
    assert diag.ess == pytest.approx(9.0 / 5.0)
    assert diag.max_weight_share == pytest.approx(2.0 / 3.0)
    assert diag.relevant_per_query == pytest.approx(3.0)
    assert diag.ess_share == pytest.approx(0.9)


def test_clipping_caps_the_weights_and_says_how_often():
    diag = diagnose([TWO[0]], eta=1.0, clip=0.6)
    assert diag.clipped == 1
    assert diag.relevant_per_query == pytest.approx(1.0 + 1.0 / 0.6)


def test_clicks_by_rank_is_a_plain_click_through_curve():
    queries = [
        ClickQuery(qid="q1", ranking=("a", "b", "c"), clicks=("a",)),
        ClickQuery(qid="q2", ranking=("a", "b", "c"), clicks=("a", "c")),
    ]
    assert clicks_by_rank(queries, max_rank=3) == pytest.approx([1.0, 0.0, 0.5])


def test_ips_recovers_the_true_metric_where_the_naive_estimate_cannot():
    """The load-bearing test: simulate clicks from a known relevance and check
    that the correction lands on the true metric while counting clicks does not.
    """
    relevance = {"d1": 0, "d2": 1, "d3": 0, "d4": 1, "d5": 0}
    logged = ("d1", "d2", "d3", "d4", "d5")
    candidate = tuple(reversed(logged))
    eta = 1.0

    lam = rank_weight("dcg")
    candidate_rank = {doc: i for i, doc in enumerate(candidate, start=1)}
    truth = sum(lam(candidate_rank[doc]) for doc, rel in relevance.items() if rel)

    rng = random.Random(20260805)
    queries = []
    for i in range(20_000):
        clicks = tuple(
            doc
            for rank, doc in enumerate(logged, start=1)
            if relevance[doc] and rng.random() < propensity(rank, eta)
        )
        queries.append(ClickQuery(qid=f"q{i}", ranking=logged, clicks=clicks))

    est = estimate(queries, lambda q: candidate, eta=eta, weight="dcg")
    assert est.ips == pytest.approx(truth, abs=0.05)
    # Counting clicks under-counts every relevant document by its examination
    # probability, and the error is far larger than the sampling noise above.
    assert est.naive == pytest.approx(0.373, abs=0.02)
    assert abs(est.naive - truth) > 0.5


def test_sensitivity_sweeps_eta_for_both_policies():
    rows = sensitivity(PINNED, lambda q: ("c", "a", "b"), etas=(0.0, 1.0), weight="dcg")
    assert [row.eta for row in rows] == [0.0, 1.0]
    assert rows[0].baseline == pytest.approx(1.0 / log2(4))
    assert rows[0].candidate == pytest.approx(1.0)
    assert rows[1].delta == pytest.approx(3.0 - 3.0 / log2(4))


def test_crossover_interpolates_where_the_winner_changes():
    rows = [
        SensitivityRow(eta=1.0, baseline=1.0, candidate=0.0),
        SensitivityRow(eta=2.0, baseline=0.0, candidate=1.0),
    ]
    assert crossover(rows) == pytest.approx(1.5)


def test_crossover_is_none_when_one_policy_leads_throughout():
    rows = [
        SensitivityRow(eta=0.5, baseline=0.0, candidate=1.0),
        SensitivityRow(eta=1.0, baseline=0.0, candidate=2.0),
    ]
    assert crossover(rows) is None


def test_estimate_refuses_an_empty_file():
    with pytest.raises(ValueError, match="no queries"):
        estimate([], logged_policy)


def test_crossover_refines_the_bracket_when_given_the_real_function():
    """Straight-line interpolation on a coarse grid is only a bracket: with the
    underlying function in hand the answer stops depending on sweep spacing.
    """
    rows = [
        SensitivityRow(eta=0.0, baseline=1.0, candidate=0.0),
        SensitivityRow(eta=2.0, baseline=0.0, candidate=1.0),
    ]

    # delta crosses zero at eta = 1.5, nowhere near the interpolated midpoint
    def delta_at(eta):
        return eta - 1.5

    assert crossover(rows) == pytest.approx(1.0)
    assert crossover(rows, delta_at) == pytest.approx(1.5, abs=1e-3)


def test_a_coarse_and_a_fine_sweep_refine_to_the_same_crossover():
    def delta_at(eta):
        return (eta - 0.8) ** 3

    coarse = [SensitivityRow(eta=e, baseline=0.0, candidate=delta_at(e)) for e in (0.0, 1.0, 2.0)]
    fine = [SensitivityRow(eta=e / 10, baseline=0.0, candidate=delta_at(e / 10)) for e in range(21)]
    assert crossover(coarse, delta_at) == pytest.approx(crossover(fine, delta_at), abs=1e-3)
