import pytest

from rankkit.compare import (
    bootstrap_ci,
    normal_cdf,
    normal_quantile,
    paired_compare,
    permutation_p,
)


def test_normal_cdf_and_its_inverse_agree():
    assert normal_cdf(0.0) == pytest.approx(0.5)
    assert normal_quantile(0.975) == pytest.approx(1.959964, abs=1e-5)
    assert normal_quantile(0.80) == pytest.approx(0.841621, abs=1e-5)
    assert normal_cdf(normal_quantile(0.3)) == pytest.approx(0.3, abs=1e-9)


def test_normal_quantile_rejects_impossible_probabilities():
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        normal_quantile(1.0)


def test_permutation_p_is_large_when_nothing_is_happening():
    deltas = [0.1, -0.1] * 30
    assert permutation_p(deltas, reps=2000, seed=0) > 0.5


def test_permutation_p_is_small_for_a_consistent_difference():
    deltas = [0.2] * 40
    assert permutation_p(deltas, reps=2000, seed=0) < 0.01


def test_permutation_p_never_reaches_zero():
    assert permutation_p([1.0] * 50, reps=100, seed=0) == pytest.approx(1.0 / 101.0)


def test_bootstrap_ci_brackets_the_observed_mean():
    deltas = [0.05, 0.10, 0.15, 0.20, 0.25] * 20
    lo, hi = bootstrap_ci(deltas, reps=1000, seed=0)
    assert lo < 0.15 < hi


def test_bootstrap_ci_is_reproducible_for_a_seed():
    deltas = [0.1, -0.2, 0.3, 0.05] * 10
    assert bootstrap_ci(deltas, seed=7) == bootstrap_ci(deltas, seed=7)


def test_a_wider_level_gives_a_wider_interval():
    deltas = [0.1, -0.2, 0.3, 0.05] * 10
    narrow = bootstrap_ci(deltas, level=0.80, seed=1)
    wide = bootstrap_ci(deltas, level=0.99, seed=1)
    assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])


def test_paired_compare_counts_wins_losses_and_ties():
    a = {"q1": 0.1, "q2": 0.5, "q3": 0.3}
    b = {"q1": 0.4, "q2": 0.2, "q3": 0.3}
    result = paired_compare(a, b, metric="ndcg@10", seed=0, reps=500, boot_reps=200)
    assert (result.wins, result.losses, result.ties) == (1, 1, 1)
    assert result.n == 3
    assert result.delta == pytest.approx(((0.3) + (-0.3) + 0.0) / 3)


def test_paired_compare_drops_and_reports_unmatched_queries():
    result = paired_compare(
        {"q1": 0.2, "q2": 0.4},
        {"q1": 0.3, "q3": 0.9},
        seed=0,
        reps=200,
        boot_reps=100,
    )
    assert result.n == 1
    assert result.dropped == ["q2", "q3"]


def test_paired_compare_refuses_disjoint_systems():
    with pytest.raises(ValueError, match="share no scored queries"):
        paired_compare({"q1": 0.1}, {"q2": 0.2})


def test_a_real_difference_comes_out_significant():
    a = {f"q{i}": 0.40 for i in range(200)}
    b = {f"q{i}": 0.40 + (0.05 if i % 4 else -0.01) for i in range(200)}
    result = paired_compare(a, b, seed=0, reps=2000, boot_reps=500)
    assert result.delta > 0
    assert result.significant
    assert result.p_permutation < 0.01


def test_noise_does_not_come_out_significant():
    a = {f"q{i}": 0.5 for i in range(60)}
    b = {f"q{i}": 0.5 + (0.02 if i % 2 else -0.02) for i in range(60)}
    result = paired_compare(a, b, seed=0, reps=2000, boot_reps=500)
    assert not result.significant
    assert result.p_permutation > 0.05


def test_biggest_movers_are_sorted_by_size_and_skip_ties():
    a = {"q1": 0.1, "q2": 0.1, "q3": 0.1}
    b = {"q1": 0.9, "q2": 0.1, "q3": 0.4}
    result = paired_compare(a, b, seed=0, reps=200, boot_reps=100)
    assert [qid for qid, _ in result.movers] == ["q1", "q3"]


def test_queries_needed_scales_with_the_square_of_the_spread():
    a = {f"q{i}": 0.0 for i in range(50)}
    b = {f"q{i}": (0.1 if i % 2 else -0.1) for i in range(50)}
    result = paired_compare(a, b, seed=0, reps=200, boot_reps=100)
    coarse = result.queries_needed(0.02)
    fine = result.queries_needed(0.01)
    assert fine == pytest.approx(coarse * 4, rel=0.02)


def test_queries_needed_rejects_a_non_positive_target():
    result = paired_compare({"q1": 0.1, "q2": 0.2}, {"q1": 0.3, "q2": 0.1}, reps=100, boot_reps=50)
    with pytest.raises(ValueError, match="target must be positive"):
        result.queries_needed(0.0)


def test_queries_needed_is_undefined_without_variation():
    # every query moves by exactly the same amount, so the paired spread is zero
    result = paired_compare({"q1": 0.0, "q2": 0.0}, {"q1": 0.5, "q2": 0.5}, reps=100, boot_reps=50)
    assert result.sd_diff == 0.0
    assert result.queries_needed(0.01) is None
