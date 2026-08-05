from math import log2

import pytest

from rankkit.data import Run, extract_runs
from rankkit.metrics import (
    average_precision,
    dcg,
    err,
    hit_rate,
    max_grade_of,
    mrr,
    ndcg,
    precision_at_k,
    recall_at_k,
    score,
    scores,
)

# Worked by hand throughout: d1 is highly relevant and first, d2 is irrelevant,
# d3 is mildly relevant and last.
WORKED = Run(qid="q1", ranking=("d1", "d2", "d3"), relevance={"d1": 3.0, "d2": 0.0, "d3": 1.0})
EMPTY = Run(qid="q2", ranking=("d1", "d2"), relevance={})


def test_dcg_discounts_by_log_of_rank_plus_one():
    assert dcg([1.0], 1) == pytest.approx(1.0)
    assert dcg([0.0, 1.0], 2) == pytest.approx(1.0 / log2(3))


def test_dcg_stops_at_k():
    assert dcg([1.0, 1.0, 1.0], 1) == pytest.approx(1.0)


def test_ndcg_matches_the_hand_computation():
    # actual 7/log2(2) + 0 + 1/log2(4) = 7.5, ideal grades [3, 1, 0]
    assert ndcg(WORKED, 3) == pytest.approx(7.5 / (7.0 + 1.0 / log2(3)))


def test_ndcg_with_linear_gain():
    assert ndcg(WORKED, 3, gain="linear") == pytest.approx(3.5 / (3.0 + 1.0 / log2(3)))


def test_ndcg_is_one_for_a_perfect_ranking():
    perfect = Run(qid="q", ranking=("a", "b"), relevance={"a": 2.0, "b": 1.0})
    assert ndcg(perfect, 2) == pytest.approx(1.0)


def test_a_relevant_document_that_was_never_retrieved_still_costs_you():
    partial = Run(qid="q", ranking=("d1",), relevance={"d1": 3.0, "missing": 3.0})
    assert ndcg(partial, 10) == pytest.approx(7.0 / (7.0 + 7.0 / log2(3)))


def test_mrr_is_the_reciprocal_of_the_first_hit():
    assert mrr(WORKED, 3) == pytest.approx(1.0)
    later = Run(qid="q", ranking=("x", "y", "d"), relevance={"d": 1.0})
    assert mrr(later, 3) == pytest.approx(1.0 / 3.0)


def test_mrr_is_zero_when_the_only_hit_sits_below_k():
    later = Run(qid="q", ranking=("x", "y", "d"), relevance={"d": 1.0})
    assert mrr(later, 2) == 0.0


def test_average_precision_matches_the_hand_computation():
    # hits at ranks 1 and 3, so (1/1 + 2/3) / 2 relevant documents
    assert average_precision(WORKED, 3) == pytest.approx((1.0 + 2.0 / 3.0) / 2.0)


def test_average_precision_divides_by_all_relevant_documents():
    # trec_eval's map_cut convention: AP@1 cannot reach 1 when 2 documents are relevant
    assert average_precision(WORKED, 1) == pytest.approx(0.5)


def test_recall_and_precision():
    assert recall_at_k(WORKED, 3) == pytest.approx(1.0)
    assert recall_at_k(WORKED, 1) == pytest.approx(0.5)
    assert precision_at_k(WORKED, 3) == pytest.approx(2.0 / 3.0)


def test_precision_always_divides_by_k_even_past_the_end_of_the_ranking():
    assert precision_at_k(WORKED, 10) == pytest.approx(0.2)


def test_hit_rate():
    assert hit_rate(WORKED, 1) == 1.0
    later = Run(qid="q", ranking=("x", "y", "d"), relevance={"d": 1.0})
    assert hit_rate(later, 2) == 0.0


def test_err_matches_the_hand_computation():
    # stop probabilities 7/8, 0 and 1/8 against a top grade of 3
    expected = 0.875 + (1.0 / 3.0) * 0.125 * 0.125
    assert err(WORKED, 3, max_grade=3.0) == pytest.approx(expected)


def test_err_uses_a_shared_scale_top_so_queries_stay_comparable():
    runs = [WORKED, Run(qid="q3", ranking=("a",), relevance={"a": 1.0})]
    assert max_grade_of(runs) == 3.0
    shared = scores(runs, "err", k=3)
    assert shared["q3"] == pytest.approx((2.0**1.0 - 1.0) / 2.0**3.0)


def test_queries_without_relevant_documents_are_undefined_not_zero():
    for metric in ("ndcg", "mrr", "map", "recall", "precision", "hit", "err"):
        assert score(EMPTY, metric, k=10) is None


def test_scores_drops_the_undefined_queries():
    out = scores([WORKED, EMPTY], "ndcg", k=3)
    assert set(out) == {"q1"}


def test_scores_reads_a_parsed_file():
    runs = extract_runs([{"qid": "q1", "ranking": ["d1", "d2"], "relevance": ["d2"]}])
    assert scores(runs, "recall", k=2) == {"q1": 1.0}


def test_k_must_be_at_least_one():
    with pytest.raises(ValueError, match="k must be at least 1"):
        ndcg(WORKED, 0)


def test_unknown_metric_and_gain_are_rejected():
    with pytest.raises(ValueError, match="unknown metric"):
        score(WORKED, "f1", k=3)
    with pytest.raises(ValueError, match="unknown gain"):
        ndcg(WORKED, 3, gain="quadratic")
