"""Pure helper tests for dashboard-specific logic."""

from __future__ import annotations

import numpy as np
import pytest

from src import dashboard_logic
from src import metrics


def test_reference_probabilities_from_cm_normalizes_decimal_reference_values() -> None:
    probs = dashboard_logic._reference_probabilities_from_cm(
        np.array([[4.5, 1.5], [2.0, 2.0]], dtype=np.float64)
    )
    np.testing.assert_allclose(probs.sum(), 1.0)
    np.testing.assert_allclose(probs, np.array([[0.45, 0.15], [0.2, 0.2]], dtype=np.float64))


def test_match_sample_size_from_observed_cm_accepts_integer_counts() -> None:
    observed = metrics.create_cm(10, 5, 5, 30)
    assert dashboard_logic._match_sample_size_from_observed_cm(observed) == 50


def test_match_sample_size_from_observed_cm_rejects_fractional_cell_counts() -> None:
    observed = np.array([[10.5, 5.0], [5.0, 29.5]], dtype=np.float64)
    with pytest.raises(ValueError, match="integer counts"):
        dashboard_logic._match_sample_size_from_observed_cm(observed)


def test_build_match_payload_preserves_repository_cm_order() -> None:
    observed = metrics.create_cm(8, 2, 1, 9)
    reference_probs = np.array([[0.45, 0.05], [0.1, 0.4]], dtype=np.float64)
    payload = dashboard_logic._build_match_payload(
        observed,
        reference_probs,
        metric_name="accuracy",
        match_mode="Approximate",
        sample_n=20,
        num_samples=1000,
        alternative="two-sided",
        seed=123,
    )
    assert payload["observed_values"] == (8.0, 2.0, 1.0, 9.0)
    assert payload["reference_probs_values"] == (0.45, 0.05, 0.1, 0.4)
    assert payload["n"] == 20


def test_default_match_payload_uses_accuracy_and_approximate_mode() -> None:
    payload = dashboard_logic._default_match_payload()
    assert payload["metric_name"] == "accuracy"
    assert payload["match_mode"] == "Approximate"
    assert payload["alternative"] == "two-sided"
    assert payload["observed_values"] == dashboard_logic.DEFAULT_MATCH_OBSERVED_VALUES
    assert np.isclose(sum(payload["reference_probs_values"]), 1.0)


@pytest.mark.parametrize(
    ("metric_name", "match_mode", "sample_n", "expected"),
    [
        ("accuracy", "Approximate", 1000, False),
        ("accuracy", "Exact", 1000, False),
        ("positive_predictive_value", "Exact", 400, False),
        ("positive_predictive_value", "Exact", 600, True),
    ],
)
def test_should_warn_about_exact_runtime_tracks_method_family(
    metric_name: str,
    match_mode: str,
    sample_n: int,
    expected: bool,
) -> None:
    assert dashboard_logic._should_warn_about_exact_runtime(metric_name, match_mode, sample_n) is expected
