"""Baseline MATCH tests for the existing computational code."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from src import match


def _all_confusion_matrices_with_total(n: int) -> list[np.ndarray]:
    cms: list[np.ndarray] = []
    for tp in range(n + 1):
        for fn in range(n - tp + 1):
            for fp in range(n - tp - fn + 1):
                tn = n - tp - fn - fp
                cms.append(np.array([[tp, fn], [fp, tn]], dtype=np.float64))
    return cms


def _cm_probability(cm: np.ndarray, probs: np.ndarray) -> float:
    counts = [int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])]
    return float(stats.multinomial.pmf(counts, n=sum(counts), p=probs.reshape(-1)))


def _manual_jrm_value(cm: np.ndarray, numerator_cell: tuple[int, int], denominator_cells: tuple[tuple[int, int], tuple[int, int]]) -> float:
    numerator = float(cm[numerator_cell])
    denominator = float(sum(cm[cell] for cell in denominator_cells))
    return float("nan") if denominator == 0.0 else numerator / denominator


def _manual_metric_value(cm: np.ndarray, metric_name: str) -> float:
    if metric_name == "marginal_benefit":
        return float(cm[1, 0] - cm[0, 1]) / float(cm.sum())

    jrm_specs = {
        "true_positive_rate": ((0, 0), ((0, 0), (0, 1))),
        "false_positive_rate": ((1, 0), ((1, 0), (1, 1))),
        "positive_predictive_value": ((0, 0), ((0, 0), (1, 0))),
        "negative_predictive_value": ((1, 1), ((1, 1), (0, 1))),
    }
    numerator_cell, denominator_cells = jrm_specs[metric_name]
    return _manual_jrm_value(cm, numerator_cell, denominator_cells)


def test_reference_probabilities_sum_to_one() -> None:
    ref = match.create_cm(4, 1, 1, 4)
    probs = match.reference_probabilities(ref)
    assert probs.shape == (2, 2)
    assert np.isclose(float(probs.sum()), 1.0)


def test_reference_probabilities_are_invariant_to_reference_cm_scale() -> None:
    base = match.create_cm(4, 1, 1, 4)
    scaled = match.create_cm(40, 10, 10, 40)
    np.testing.assert_allclose(match.reference_probabilities(base), match.reference_probabilities(scaled))


def test_metric_score_accuracy_matches_expected_value() -> None:
    cm = match.create_cm(1, 2, 3, 4)
    assert np.isclose(match.metric_score(cm, "accuracy"), 0.5)


def test_metric_score_supports_alias_names() -> None:
    cm = match.create_cm(8, 2, 1, 9)
    assert match.metric_score(cm, "precision") == pytest.approx(match.metric_score(cm, "positive_predictive_value"))
    assert match.metric_score(cm, "recall") == pytest.approx(match.metric_score(cm, "true_positive_rate"))
    assert match.metric_score(cm, "B") == pytest.approx(match.metric_score(cm, "marginal_benefit"))


def test_match_test_returns_match_result_for_accuracy() -> None:
    observed = match.create_cm(8, 2, 1, 9)
    reference = match.create_cm(9, 1, 2, 8)
    result = match.match_test(observed, reference, "accuracy", method="exact")
    assert result.metric == "accuracy"
    assert result.method == "exact"
    assert result.alternative == "two-sided"
    assert np.isfinite(result.p_value)


def test_match_test_from_reference_probs_matches_reference_cm_accuracy() -> None:
    observed = match.create_cm(8, 2, 1, 9)
    reference = match.create_cm(9, 1, 2, 8)
    reference_probs = match.reference_probabilities(reference)
    result_from_cm = match.match_test(observed, reference, "accuracy", method="exact")
    result_from_probs = match.match_test_from_reference_probs(
        observed,
        reference_probs,
        "accuracy",
        method="exact",
    )
    assert result_from_probs.cdf == pytest.approx(result_from_cm.cdf)
    assert result_from_probs.p_value == pytest.approx(result_from_cm.p_value)


def test_match_result_is_invariant_to_reference_cm_size_when_probabilities_match() -> None:
    observed = match.create_cm(8, 2, 1, 9)
    reference_small = match.create_cm(9, 1, 2, 8)
    reference_large = match.create_cm(90, 10, 20, 80)
    result_small = match.match_test(observed, reference_small, "accuracy", method="exact")
    result_large = match.match_test(observed, reference_large, "accuracy", method="exact")
    assert result_small.cdf == pytest.approx(result_large.cdf)
    assert result_small.p_value == pytest.approx(result_large.p_value)


def test_compare_match_methods_returns_multiple_results() -> None:
    observed = match.create_cm(8, 2, 1, 9)
    reference = match.create_cm(9, 1, 2, 8)
    results = match.compare_match_methods(observed, reference, "accuracy")
    assert len(results) == 3
    assert {result.method for result in results} == {"exact", "normal", "peizer_pratt"}


def test_compare_match_methods_excludes_peizer_pratt_for_marginal_benefit() -> None:
    observed = match.create_cm(8, 2, 1, 9)
    reference = match.create_cm(9, 1, 2, 8)
    results = match.compare_match_methods(observed, reference, "marginal_benefit")
    assert len(results) == 2
    assert {result.method for result in results} == {"exact", "normal"}


def test_match_test_many_preserves_number_of_inputs() -> None:
    observed = np.stack(
        [
            match.create_cm(8, 2, 1, 9),
            match.create_cm(4, 1, 2, 7),
        ],
        axis=0,
    )
    reference = match.create_cm(9, 1, 2, 8)
    results = match.match_test_many(observed, reference, "accuracy")
    assert len(results) == 2


def test_match_test_raises_for_undefined_observed_jrm() -> None:
    observed = match.create_cm(0, 3, 0, 7)
    reference = match.create_cm(9, 1, 2, 8)
    with pytest.raises(ValueError, match="undefined"):
        match.match_test(observed, reference, "positive_predictive_value", method="exact")


def test_match_test_uses_canonical_metric_name_for_alias() -> None:
    observed = match.create_cm(8, 2, 1, 9)
    reference = match.create_cm(9, 1, 2, 8)
    result = match.match_test(observed, reference, "precision", method="exact")
    assert result.metric == "positive_predictive_value"


def test_preferred_match_method_uses_family_specific_approximations() -> None:
    assert match.preferred_match_method("accuracy", "approximate") == "peizer_pratt"
    assert match.preferred_match_method("marginal_benefit", "approximate") == "normal"
    assert match.preferred_match_method("positive_predictive_value", "approximate") == "beta"


def test_match_test_rejects_peizer_pratt_for_marginal_benefit() -> None:
    observed = match.create_cm(8, 2, 1, 9)
    reference = match.create_cm(9, 1, 2, 8)
    with pytest.raises(ValueError, match="marginal_benefit metrics support methods: exact, normal"):
        match.match_test(observed, reference, "marginal_benefit", method="peizer_pratt")


def test_match_test_score_from_reference_probs_rejects_peizer_pratt_for_jrm() -> None:
    reference_probs = np.array([[0.45, 0.05], [0.1, 0.4]], dtype=np.float64)
    with pytest.raises(ValueError, match="jrm metrics support methods: exact, beta"):
        match.match_test_score_from_reference_probs(
            observed_score=0.75,
            n_obs=20,
            reference_probs=reference_probs,
            metric="positive_predictive_value",
            method="peizer_pratt",
            observed_denominator=10,
        )


def test_ppr_greater_tail_below_reference_mean_peizer_pratt_regression() -> None:
    result = match.match_test_score_from_reference_probs(
        observed_score=0.375,
        n_obs=40,
        reference_probs=np.array([[0.45, 0.10], [0.10, 0.35]], dtype=np.float64),
        metric="predicted_positive_rate",
        method="peizer_pratt",
        alternative="greater",
    )

    assert result.p_value > 0.95


def test_ppr_less_tail_below_reference_mean_peizer_pratt_regression() -> None:
    result = match.match_test_score_from_reference_probs(
        observed_score=0.375,
        n_obs=40,
        reference_probs=np.array([[0.45, 0.10], [0.10, 0.35]], dtype=np.float64),
        metric="predicted_positive_rate",
        method="peizer_pratt",
        alternative="less",
    )

    assert result.p_value < 0.05


def test_peizer_pratt_cdf_below_mean_matches_exact_binomial_direction() -> None:
    n = 40
    p = 0.55
    k = 15

    approx = match.peizer_pratt_binom_cdf(k, n, p)
    exact = match.binom_cdf(k, n, p)

    assert approx < 0.05
    assert abs(approx - exact) < 0.02


def test_peizer_pratt_cdf_above_mean_matches_exact_binomial_direction() -> None:
    n = 40
    p = 0.55
    k = 29

    approx = match.peizer_pratt_binom_cdf(k, n, p)
    exact = match.binom_cdf(k, n, p)

    assert approx > 0.95
    assert abs(approx - exact) < 0.02


def test_ppr_exact_tail_direction_regression() -> None:
    probs = np.array([[0.45, 0.10], [0.10, 0.35]], dtype=np.float64)

    less = match.match_test_score_from_reference_probs(
        observed_score=0.375,
        n_obs=40,
        reference_probs=probs,
        metric="predicted_positive_rate",
        method="exact",
        alternative="less",
    )

    greater = match.match_test_score_from_reference_probs(
        observed_score=0.375,
        n_obs=40,
        reference_probs=probs,
        metric="predicted_positive_rate",
        method="exact",
        alternative="greater",
    )

    assert less.p_value < 0.05
    assert greater.p_value > 0.95


def test_match_test_score_from_reference_probs_respects_requested_n() -> None:
    reference_probs = np.array([[0.45, 0.05], [0.1, 0.4]], dtype=np.float64)
    result = match.match_test_score_from_reference_probs(
        observed_score=0.5,
        n_obs=20,
        reference_probs=reference_probs,
        metric="accuracy",
        method="exact",
    )
    expected_cdf = match.binom_cdf(10, 20, 0.85)
    assert result.n_obs == 20
    assert result.cdf == pytest.approx(expected_cdf)


def test_jrm_beta_approximation_uses_observed_count_formula_from_aistats() -> None:
    observed = match.create_cm(8, 2, 1, 9)
    reference_probs = np.array([[0.45, 0.05], [0.1, 0.4]], dtype=np.float64)
    result = match.match_test_from_reference_probs(
        observed,
        reference_probs,
        "positive_predictive_value",
        method="beta",
    )
    observed_score = match.metric_score(observed, "positive_predictive_value")
    observed_denominator = int(observed[0, 0] + observed[1, 0])
    p_pair = float(reference_probs[0, 0] + reference_probs[1, 0])
    theta = float(reference_probs[0, 0] / p_pair)
    expected_cdf = match.regularized_incomplete_beta(
        observed_score,
        observed_denominator * theta + 1.0,
        observed_denominator * (1.0 - theta) + 1.0,
    )
    assert result.details["beta_count"] == "observed"
    assert result.cdf == pytest.approx(expected_cdf)


def test_unconditioned_jrm_exact_matches_aistats_mixture_mass() -> None:
    reference_probs = np.array([[0.2, 0.2], [0.1, 0.5]], dtype=np.float64)
    result = match.match_test_score_from_reference_probs(
        observed_score=1.0,
        n_obs=4,
        reference_probs=reference_probs,
        metric="true_positive_rate",
        method="exact",
        condition_on_defined=False,
    )
    expected_mass = 1.0 - (1.0 - 0.4) ** 4
    assert result.cdf == pytest.approx(expected_mass)


def test_validate_reference_probs_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="shape"):
        match.validate_reference_probs(np.array([0.5, 0.5, 0.0]))
    with pytest.raises(ValueError, match="sum to 1"):
        match.validate_reference_probs(np.array([[0.4, 0.2], [0.1, 0.4]], dtype=np.float64))
    with pytest.raises(ValueError, match="non-negative"):
        match.validate_reference_probs(np.array([[0.5, -0.1], [0.2, 0.4]], dtype=np.float64))


def test_validate_reference_probs_accepts_flat_vector_input() -> None:
    probs = match.validate_reference_probs(np.array([0.4, 0.1, 0.2, 0.3], dtype=np.float64))
    assert probs.shape == (2, 2)
    np.testing.assert_allclose(probs, np.array([[0.4, 0.1], [0.2, 0.3]], dtype=np.float64))


def test_match_test_rejects_fractional_confusion_matrices() -> None:
    observed = np.array([[1.5, 2.0], [3.0, 4.0]], dtype=np.float64)
    reference = match.create_cm(9, 1, 2, 8)
    with pytest.raises(ValueError, match="integer counts"):
        match.match_test(observed, reference, "accuracy", method="exact")


def test_match_test_score_from_reference_probs_rejects_nonpositive_n_obs() -> None:
    reference_probs = np.array([[0.45, 0.05], [0.1, 0.4]], dtype=np.float64)
    with pytest.raises(ValueError, match="n_obs must be positive"):
        match.match_test_score_from_reference_probs(
            observed_score=0.5,
            n_obs=0,
            reference_probs=reference_probs,
            metric="accuracy",
        )


def test_tail_p_value_uses_requested_alternative() -> None:
    assert match._tail_p_value(0.2, 0.8, "less") == pytest.approx(0.2)
    assert match._tail_p_value(0.2, 0.8, "greater") == pytest.approx(0.8)
    assert match._tail_p_value(0.2, 0.8, "two-sided") == pytest.approx(0.4)


@pytest.mark.parametrize(
    "metric_name",
    [
        "true_positive_rate",
        "false_positive_rate",
        "positive_predictive_value",
        "negative_predictive_value",
    ],
)
def test_jrm_exact_conditioned_distribution_matches_bruteforce(metric_name: str) -> None:
    n = 5
    reference_probs = np.array([[0.35, 0.15], [0.1, 0.4]], dtype=np.float64)
    cms = _all_confusion_matrices_with_total(n)
    finite = [
        (_manual_metric_value(cm, metric_name), _cm_probability(cm, reference_probs))
        for cm in cms
        if np.isfinite(_manual_metric_value(cm, metric_name))
    ]
    defined_mass = sum(prob for _, prob in finite)
    support = sorted({value for value, _ in finite})

    for score in support:
        result = match.match_test_score_from_reference_probs(
            observed_score=score,
            n_obs=n,
            reference_probs=reference_probs,
            metric=metric_name,
            method="exact",
            condition_on_defined=True,
        )
        expected_lower = sum(prob for value, prob in finite if value <= score) / defined_mass
        expected_upper = sum(prob for value, prob in finite if value >= score) / defined_mass
        assert result.cdf == pytest.approx(expected_lower)
        assert result.p_upper == pytest.approx(expected_upper)


def test_jrm_exact_unconditioned_distribution_matches_bruteforce() -> None:
    n = 5
    metric_name = "positive_predictive_value"
    reference_probs = np.array([[0.35, 0.15], [0.1, 0.4]], dtype=np.float64)
    cms = _all_confusion_matrices_with_total(n)
    weighted_values = [
        (_manual_metric_value(cm, metric_name), _cm_probability(cm, reference_probs))
        for cm in cms
    ]
    support = sorted({value for value, _ in weighted_values if np.isfinite(value)})

    for score in support:
        result = match.match_test_score_from_reference_probs(
            observed_score=score,
            n_obs=n,
            reference_probs=reference_probs,
            metric=metric_name,
            method="exact",
            condition_on_defined=False,
        )
        expected_lower = sum(prob for value, prob in weighted_values if np.isfinite(value) and value <= score)
        expected_upper = sum(prob for value, prob in weighted_values if np.isfinite(value) and value >= score)
        assert result.cdf == pytest.approx(expected_lower)
        assert result.p_upper == pytest.approx(expected_upper)


def test_marginal_benefit_exact_distribution_matches_bruteforce() -> None:
    n = 5
    reference_probs = np.array([[0.3, 0.2], [0.15, 0.35]], dtype=np.float64)
    cms = _all_confusion_matrices_with_total(n)
    weighted_values = [
        (_manual_metric_value(cm, "marginal_benefit"), _cm_probability(cm, reference_probs))
        for cm in cms
    ]
    support = sorted({value for value, _ in weighted_values})

    for score in support:
        result = match.match_test_score_from_reference_probs(
            observed_score=score,
            n_obs=n,
            reference_probs=reference_probs,
            metric="marginal_benefit",
            method="exact",
        )
        expected_lower = sum(prob for value, prob in weighted_values if value <= score)
        expected_upper = sum(prob for value, prob in weighted_values if value >= score)
        assert result.cdf == pytest.approx(expected_lower)
        assert result.p_upper == pytest.approx(expected_upper)


def test_jrm_match_test_from_cm_matches_score_api_for_exact_method() -> None:
    observed = match.create_cm(2, 1, 1, 1)
    reference_probs = np.array([[0.35, 0.15], [0.1, 0.4]], dtype=np.float64)

    result_from_cm = match.match_test_from_reference_probs(
        observed,
        reference_probs,
        "positive_predictive_value",
        method="exact",
    )
    result_from_score = match.match_test_score_from_reference_probs(
        observed_score=2.0 / 3.0,
        n_obs=5,
        reference_probs=reference_probs,
        metric="positive_predictive_value",
        method="exact",
    )

    assert result_from_cm.cdf == pytest.approx(result_from_score.cdf)
    assert result_from_cm.p_upper == pytest.approx(result_from_score.p_upper)
    assert result_from_cm.p_value == pytest.approx(result_from_score.p_value)


def test_multinomial_reference_samples_has_expected_shape_and_totals() -> None:
    rng = np.random.default_rng(123)
    probs = np.array([[0.4, 0.1], [0.2, 0.3]], dtype=np.float64)
    samples = match.multinomial_reference_samples(probs, n=20, num_samples=25, rng=rng)
    assert samples.shape == (25, 2, 2)
    assert np.all(samples.sum(axis=(1, 2)) == 20)


def test_metric_distribution_preserves_length() -> None:
    cms = np.stack([match.create_cm(8, 2, 1, 9), match.create_cm(4, 1, 2, 7)], axis=0)
    values = match.metric_distribution(cms, lambda cm: cm[0, 0] / cm.sum())
    assert values.shape == (2,)


def test_metric_distribution_rejects_invalid_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        match.metric_distribution(np.array([[1.0, 2.0], [3.0, 4.0]]), lambda cm: cm[0, 0])


def test_match_test_many_rejects_invalid_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        match.match_test_many(np.array([1.0, 2.0, 3.0, 4.0]), match.create_cm(9, 1, 2, 8), "accuracy")


def test_simulation_match_test_returns_required_keys() -> None:
    observed = match.create_cm(8, 2, 1, 9)
    probs = np.array([[0.45, 0.05], [0.1, 0.4]], dtype=np.float64)
    result = match.simulation_match_test(
        observed_cm=observed,
        reference_probs=probs,
        metric_func=lambda cm: cm[0, 0] / cm.sum(),
        num_samples=2000,
        seed=12345,
    )
    required_keys = {
        "observed_value",
        "reference_mean",
        "reference_std",
        "reference_median",
        "q025",
        "q05",
        "q95",
        "q975",
        "p_value",
        "undefined_rate",
        "finite_sample_count",
        "total_sample_count",
        "alternative",
        "n",
        "seed",
        "status",
    }
    assert required_keys.issubset(result)


def test_simulation_match_test_is_deterministic_for_fixed_seed() -> None:
    observed = match.create_cm(8, 2, 1, 9)
    probs = np.array([[0.45, 0.05], [0.1, 0.4]], dtype=np.float64)
    result_1 = match.simulation_match_test(
        observed_cm=observed,
        reference_probs=probs,
        metric_func=lambda cm: cm[0, 0] / cm.sum(),
        num_samples=3000,
        seed=777,
    )
    result_2 = match.simulation_match_test(
        observed_cm=observed,
        reference_probs=probs,
        metric_func=lambda cm: cm[0, 0] / cm.sum(),
        num_samples=3000,
        seed=777,
    )
    assert result_1["p_value"] == result_2["p_value"]


def test_simulation_match_test_batched_uses_requested_sample_size_override() -> None:
    observed = match.create_cm(8, 2, 1, 9)
    probs = np.array([[0.45, 0.05], [0.1, 0.4]], dtype=np.float64)
    result = match.simulation_match_test_batched(
        observed_cm=observed,
        reference_probs=probs,
        metric_func=lambda cm: cm[0, 0] / cm.sum(),
        n=50,
        num_samples=1000,
        seed=321,
        batch_size=250,
    )
    assert result["n"] == 50
    assert result["status"] == "ok"


def test_simulation_match_test_batched_reports_progress_callback() -> None:
    observed = match.create_cm(8, 2, 1, 9)
    probs = np.array([[0.45, 0.05], [0.1, 0.4]], dtype=np.float64)
    progress_updates: list[tuple[int, int]] = []
    result = match.simulation_match_test_batched(
        observed_cm=observed,
        reference_probs=probs,
        metric_func=lambda cm: cm[0, 0] / cm.sum(),
        num_samples=1200,
        seed=123,
        batch_size=500,
        progress_callback=lambda done, total: progress_updates.append((done, total)),
    )
    assert result["status"] == "ok"
    assert progress_updates[-1] == (1200, 1200)
    assert all(total == 1200 for _, total in progress_updates)
    assert all(done_1 < done_2 for (done_1, _), (done_2, _) in zip(progress_updates, progress_updates[1:]))


def test_simulation_match_test_batched_can_cancel_before_sampling() -> None:
    observed = match.create_cm(8, 2, 1, 9)
    probs = np.array([[0.45, 0.05], [0.1, 0.4]], dtype=np.float64)
    result = match.simulation_match_test_batched(
        observed_cm=observed,
        reference_probs=probs,
        metric_func=lambda cm: cm[0, 0] / cm.sum(),
        num_samples=2000,
        seed=12345,
        should_cancel=lambda: True,
    )
    assert result["status"] == "cancelled"


def test_simulation_match_test_returns_observed_undefined_status() -> None:
    observed = match.create_cm(0, 0, 3, 7)
    probs = np.array([[0.0, 0.0], [0.3, 0.7]], dtype=np.float64)
    result = match.simulation_match_test(
        observed_cm=observed,
        reference_probs=probs,
        metric_func=lambda cm: np.nan if (cm[0, 0] + cm[0, 1]) == 0 else cm[0, 0] / (cm[0, 0] + cm[0, 1]),
        num_samples=2000,
        seed=123,
    )
    assert result["status"] == "observed_undefined"


def test_results_to_records_round_trip() -> None:
    observed = match.create_cm(8, 2, 1, 9)
    reference = match.create_cm(9, 1, 2, 8)
    results = match.compare_match_methods(observed, reference, "accuracy")
    records = match.results_to_records(results)
    assert len(records) == len(results)
    assert records[0]["metric"] == results[0].metric
    assert records[0]["p_value"] == pytest.approx(results[0].p_value)
