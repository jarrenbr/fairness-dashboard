"""Behavioral tests for confusion-matrix metrics and fairness metrics."""

from __future__ import annotations

import numpy as np
import pytest

from src import metrics


COUNT_EXPECTED_VALUES = {
    metrics.TRUE_POSITIVE_COUNT: 4.0,
    metrics.FALSE_NEGATIVE_COUNT: 3.0,
    metrics.FALSE_POSITIVE_COUNT: 2.0,
    metrics.TRUE_NEGATIVE_COUNT: 1.0,
    metrics.ACTUAL_POSITIVE_COUNT: 7.0,
    metrics.ACTUAL_NEGATIVE_COUNT: 3.0,
    metrics.PREDICTED_POSITIVE_COUNT: 6.0,
    metrics.PREDICTED_NEGATIVE_COUNT: 4.0,
    metrics.INCORRECT_COUNT: 5.0,
    metrics.CORRECT_COUNT: 5.0,
    metrics.TOTAL_COUNT: 10.0,
}
BINOMIAL_COMPLEMENT_PAIRS = (
    ("accuracy", "inaccuracy"),
    ("prevalence", "negative_prevalence"),
    ("predicted_positive_rate", "predicted_negative_rate"),
)
JRM_COMPLEMENT_PAIRS = (
    ("true_positive_rate", "false_negative_rate"),
    ("false_positive_rate", "true_negative_rate"),
    ("positive_predictive_value", "false_discovery_rate"),
    ("negative_predictive_value", "false_omission_rate"),
    ("true_positive_share_of_correct", "true_negative_share_of_correct"),
    ("false_positive_share_of_errors", "false_negative_share_of_errors"),
)
FAIRNESS_DIFFERENCE_METRICS = tuple(
    name for name in metrics.TWO_CM_FAIRNESS_METRICS if name != metrics.DISPARATE_IMPACT
)


def test_create_cm_returns_shape_2x2() -> None:
    cm = metrics.create_cm(1, 2, 3, 4)
    assert cm.shape == (2, 2)
    assert cm.dtype == np.float64


def test_create_cm_broadcasts_to_stacked_confusion_matrices() -> None:
    cm = metrics.create_cm(
        tp=np.array([1, 2]),
        fn=np.array([3, 4]),
        fp=np.array([5, 6]),
        tn=np.array([7, 8]),
    )
    expected = np.array(
        [
            [[1.0, 3.0], [5.0, 7.0]],
            [[2.0, 4.0], [6.0, 8.0]],
        ]
    )
    assert cm.shape == (2, 2, 2)
    np.testing.assert_allclose(cm, expected)


def test_is_binary_cm_proper_accepts_single_and_stacked_confusion_matrices() -> None:
    assert metrics.is_binary_cm_proper(metrics.create_cm(1, 2, 3, 4))
    assert metrics.is_binary_cm_proper(metrics.create_cm(tp=[1, 2], fn=[3, 4], fp=[5, 6], tn=[7, 8]))
    assert not metrics.is_binary_cm_proper(np.array([1.0, 2.0, 3.0, 4.0]))


@pytest.mark.parametrize("metric_name, expected", COUNT_EXPECTED_VALUES.items())
def test_count_metrics_follow_repository_cm_convention(metric_name: str, expected: float) -> None:
    cm = metrics.create_cm(4, 3, 2, 1)
    value = metrics.COUNT_METRICS[metric_name](cm)
    assert value == pytest.approx(expected)


def test_accuracy_matches_expected_value() -> None:
    cm = metrics.create_cm(1, 2, 3, 4)
    assert np.isclose(float(metrics.accuracy(cm)), 0.5)


def test_metrics_support_stacked_confusion_matrices() -> None:
    cms = np.stack(
        [
            metrics.create_cm(1, 2, 3, 4),
            metrics.create_cm(4, 1, 0, 5),
        ],
        axis=0,
    )
    np.testing.assert_allclose(metrics.total(cms), np.array([10.0, 10.0]))
    np.testing.assert_allclose(metrics.accuracy(cms), np.array([0.5, 0.9]))
    np.testing.assert_allclose(metrics.true_positive_rate(cms), np.array([1.0 / 3.0, 0.8]))


def test_marginal_benefit_matches_expected_value() -> None:
    cm = metrics.create_cm(1, 2, 3, 4)
    assert np.isclose(float(metrics.marginal_benefit(cm)), 0.1)


def test_true_positive_rate_returns_nan_when_denominator_zero() -> None:
    cm = metrics.create_cm(0, 0, 3, 4)
    assert np.isnan(metrics.true_positive_rate(cm))


def test_accuracy_returns_nan_for_zero_total() -> None:
    cm = metrics.create_cm(0, 0, 0, 0)
    assert np.isnan(metrics.accuracy(cm))


def test_safe_divide_returns_nan_where_denominator_is_zero() -> None:
    result = metrics.safe_divide(np.array([1.0, 2.0]), np.array([1.0, 0.0]))
    np.testing.assert_allclose(result[:1], np.array([1.0]))
    assert np.isnan(result[1])


def test_safe_divide_scalar_returns_python_float() -> None:
    value = metrics.safe_divide(3.0, 2.0)
    assert isinstance(value, float)
    assert value == pytest.approx(1.5)


def test_compute_metrics_includes_expected_keys() -> None:
    cm = metrics.create_cm(8, 2, 1, 9)
    result = metrics.compute_metrics(cm)
    assert "accuracy" in result
    assert "true_positive_rate" in result


def test_objective_fairness_index_matches_marginal_benefit_difference() -> None:
    cm_i = metrics.create_cm(40, 10, 5, 45)
    cm_j = metrics.create_cm(35, 15, 10, 40)
    expected = float(metrics.marginal_benefit(cm_i)) - float(metrics.marginal_benefit(cm_j))
    assert np.isclose(metrics.objective_fairness_index(cm_i, cm_j), expected)


def test_disparate_impact_returns_nan_when_reference_rate_zero() -> None:
    cm_i = metrics.create_cm(10, 0, 5, 15)
    cm_j = metrics.create_cm(0, 8, 0, 12)
    assert np.isnan(metrics.disparate_impact(cm_i, cm_j))


def test_mcc_returns_nan_when_denominator_zero() -> None:
    cm = metrics.create_cm(0, 0, 0, 5)
    assert np.isnan(metrics.matthews_correlation_coefficient(cm))


def test_f1_score_matches_manual_formula() -> None:
    cm = metrics.create_cm(8, 2, 1, 9)
    expected = 2.0 * 8.0 / (2.0 * 8.0 + 1.0 + 2.0)
    assert metrics.f1_score(cm) == pytest.approx(expected)


def test_mcc_hits_signed_extremes_for_perfect_and_inverted_predictions() -> None:
    perfect = metrics.create_cm(5, 0, 0, 5)
    inverted = metrics.create_cm(0, 5, 5, 0)
    assert metrics.matthews_correlation_coefficient(perfect) == pytest.approx(1.0)
    assert metrics.matthews_correlation_coefficient(inverted) == pytest.approx(-1.0)


def test_metric_aliases_match_canonical_functions() -> None:
    cm = metrics.create_cm(8, 2, 1, 9)
    assert metrics.error_rate(cm) == metrics.inaccuracy(cm)
    assert metrics.recall(cm) == metrics.true_positive_rate(cm)
    assert metrics.precision(cm) == metrics.positive_predictive_value(cm)
    assert metrics.mcc(cm) == metrics.matthews_correlation_coefficient(cm)


def test_difference_in_conditional_rejection_uses_documented_orientation() -> None:
    cm_i = metrics.create_cm(40, 10, 5, 45)
    cm_j = metrics.create_cm(35, 15, 10, 40)
    value_i = (10.0 + 40.0) / (15.0 + 40.0)
    value_j = (5.0 + 45.0) / (10.0 + 45.0)
    expected = value_j - value_i
    assert metrics.difference_in_conditional_rejection(cm_i, cm_j) == pytest.approx(expected)


def test_compute_metrics_honors_custom_metric_selection() -> None:
    cm = metrics.create_cm(8, 2, 1, 9)
    result = metrics.compute_metrics(cm, {"accuracy": metrics.accuracy})
    assert list(result) == ["accuracy"]
    assert result["accuracy"] == pytest.approx(0.85)


@pytest.mark.parametrize(
    "bad_cm",
    [
        np.array([1.0, 2.0, 3.0, 4.0]),
        np.array([[1.0, -1.0], [2.0, 3.0]]),
        np.array([[1.0, np.nan], [2.0, 3.0]]),
    ],
)
def test_single_cm_metrics_validate_confusion_matrix_inputs(bad_cm: np.ndarray) -> None:
    with pytest.raises(ValueError):
        metrics.accuracy(bad_cm)


@pytest.mark.parametrize("left_name,right_name", BINOMIAL_COMPLEMENT_PAIRS)
def test_binomial_complement_metrics_sum_to_one(left_name: str, right_name: str) -> None:
    cm = metrics.create_cm(8, 2, 1, 9)
    left = getattr(metrics, left_name)(cm)
    right = getattr(metrics, right_name)(cm)
    assert float(left + right) == pytest.approx(1.0)


@pytest.mark.parametrize("left_name,right_name", JRM_COMPLEMENT_PAIRS)
def test_joint_ratio_complement_metrics_sum_to_one_when_defined(left_name: str, right_name: str) -> None:
    cm = metrics.create_cm(8, 2, 1, 9)
    left = getattr(metrics, left_name)(cm)
    right = getattr(metrics, right_name)(cm)
    assert float(left + right) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("left_name", "right_name", "cm"),
    [
        ("true_positive_rate", "false_negative_rate", metrics.create_cm(0, 0, 2, 3)),
        ("false_positive_rate", "true_negative_rate", metrics.create_cm(2, 3, 0, 0)),
        ("positive_predictive_value", "false_discovery_rate", metrics.create_cm(0, 3, 0, 7)),
        ("negative_predictive_value", "false_omission_rate", metrics.create_cm(8, 0, 2, 0)),
        ("true_positive_share_of_correct", "true_negative_share_of_correct", metrics.create_cm(0, 4, 5, 0)),
        ("false_positive_share_of_errors", "false_negative_share_of_errors", metrics.create_cm(3, 0, 0, 7)),
    ],
)
def test_joint_ratio_complement_metrics_return_nan_for_zero_denominator(
    left_name: str,
    right_name: str,
    cm: np.ndarray,
) -> None:
    assert np.isnan(getattr(metrics, left_name)(cm))
    assert np.isnan(getattr(metrics, right_name)(cm))


def test_two_cm_fairness_metrics_have_g_and_m_formula_metadata() -> None:
    for metric_name in metrics.TWO_CM_FAIRNESS_METRICS:
        info = metrics.METRIC_INFO[metric_name]
        assert "g_latex" in info
        assert "M_latex" in info
        assert info["g_latex"]
        assert info["M_latex"]


def test_conditional_rejection_metadata_matches_metric_orientation() -> None:
    info = metrics.METRIC_INFO["difference_in_conditional_rejection"]
    m_formula = str(info["M_latex"])
    assert "g_j" in m_formula
    assert "g_i" in m_formula
    assert "g_j - g_i" in m_formula


@pytest.mark.parametrize("metric_name", FAIRNESS_DIFFERENCE_METRICS)
def test_difference_style_fairness_metrics_are_antisymmetric(metric_name: str) -> None:
    cm_i = metrics.create_cm(40, 10, 5, 45)
    cm_j = metrics.create_cm(35, 15, 10, 40)
    metric_func = metrics.TWO_CM_FAIRNESS_METRICS[metric_name]
    value_ij = metric_func(cm_i, cm_j)
    value_ji = metric_func(cm_j, cm_i)
    assert value_ij == pytest.approx(-value_ji)


def test_disparate_impact_is_reciprocal_when_both_groups_have_positive_selection_rate() -> None:
    cm_i = metrics.create_cm(40, 10, 5, 45)
    cm_j = metrics.create_cm(35, 15, 10, 40)
    forward = metrics.disparate_impact(cm_i, cm_j)
    reverse = metrics.disparate_impact(cm_j, cm_i)
    assert forward * reverse == pytest.approx(1.0)


def test_equal_groups_hit_documented_no_bias_values_for_defined_fairness_metrics() -> None:
    cm = metrics.create_cm(8, 2, 1, 9)
    for metric_name, metric_func in metrics.TWO_CM_FAIRNESS_METRICS.items():
        value = metric_func(cm, cm)
        assert np.isfinite(value), f"{metric_name} unexpectedly undefined for a well-formed identical-group CM"
        expected = float(metrics.METRIC_INFO[metric_name]["no_bias_value"])
        assert value == pytest.approx(expected), metric_name


def test_objective_fairness_index_is_the_unique_objective_testing_metric() -> None:
    objective_testing_metrics = {
        name
        for name, info in metrics.METRIC_INFO.items()
        if info.get("metric_kind") == "two_cm_fairness"
        and info["properties"]["chapter_3_bias_desiderata"]["objective_testing"] is True
    }
    assert objective_testing_metrics == {metrics.OBJECTIVE_FAIRNESS_INDEX}


def test_all_registered_metrics_have_core_display_metadata() -> None:
    required_fields = ("display_name", "abbreviation", "formula_text", "range_text", "interpretation")
    registered_metric_names = (
        set(metrics.COUNT_METRICS)
        | set(metrics.SINGLE_CM_METRICS)
        | set(metrics.TWO_CM_FAIRNESS_METRICS)
    )
    for metric_name in registered_metric_names:
        info = metrics.METRIC_INFO[metric_name]
        for field in required_fields:
            assert isinstance(info.get(field), str), f"{metric_name} missing string metadata for {field}"
            assert str(info[field]).strip(), f"{metric_name} has blank metadata for {field}"
        assert isinstance(info.get("defined_everywhere"), bool)
        assert isinstance(info.get("defined_everywhere_scope"), str)
