"""Support-function tests for Tab 1 UI helpers."""

from __future__ import annotations

from src import metrics
from src import match
from src.ui_components import _metric_latex, _normalize_cm_input_value, cm_summary, metric_display_name


def test_cm_summary_reports_expected_counts() -> None:
    summary = cm_summary(metrics.create_cm(4, 3, 2, 1))
    assert summary["tp"] == 4.0
    assert summary["fn"] == 3.0
    assert summary["fp"] == 2.0
    assert summary["tn"] == 1.0
    assert summary["actual_positive"] == 7.0
    assert summary["actual_negative"] == 3.0
    assert summary["predicted_positive"] == 6.0
    assert summary["predicted_negative"] == 4.0
    assert summary["n"] == 10.0


def test_metric_display_name_uses_metadata() -> None:
    assert metric_display_name("objective_fairness_index") == "Objective Fairness Index"


def test_metric_display_name_falls_back_to_raw_name_for_unknown_metric() -> None:
    assert metric_display_name("unregistered_metric") == "unregistered_metric"


def test_metric_latex_prefers_latex_metadata_for_match_metrics() -> None:
    assert "FP" in str(_metric_latex("false_discovery_rate"))
    assert "TP" in str(_metric_latex("false_discovery_rate"))
    assert "TP" in str(_metric_latex("accuracy"))
    assert "TN" in str(_metric_latex("accuracy"))
    assert "FP" in str(_metric_latex("marginal_benefit"))
    assert "FN" in str(_metric_latex("marginal_benefit"))


def test_match_metrics_have_required_display_metadata() -> None:
    required_fields = (
        "display_name",
        "abbreviation",
        "formula_text",
        "family",
        "range_text",
        "interpretation",
    )
    for metric_name in match.MATCH_METRICS:
        info = metrics.METRIC_INFO[metric_name]
        for field in required_fields:
            assert isinstance(info.get(field), str)
            assert str(info[field]).strip()
        assert _metric_latex(metric_name) is not None


def test_normalize_cm_input_value_snaps_decimal_step_upward() -> None:
    assert _normalize_cm_input_value(4.2, 5.2) == 5.0


def test_normalize_cm_input_value_snaps_decimal_step_downward() -> None:
    assert _normalize_cm_input_value(4.2, 3.2) == 4.0


def test_normalize_cm_input_value_preserves_integer_steps() -> None:
    assert _normalize_cm_input_value(4.0, 5.0) == 5.0
