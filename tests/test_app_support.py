"""Support tests for Tab 2 MATCH app helpers."""

from __future__ import annotations

import numpy as np
from streamlit.testing.v1 import AppTest

from src import app
from src import metrics


def _find_number_input(at: AppTest, label: str):
    return next(widget for widget in at.number_input if widget.label == label)


def _find_radio(at: AppTest, label: str):
    return next(widget for widget in at.radio if widget.label == label)


def _find_selectbox(at: AppTest, label: str, key: str | None = None):
    widgets = [widget for widget in at.selectbox if widget.label == label]
    if key is not None:
        return next(widget for widget in widgets if widget.key == key)
    return next(iter(widgets))


def _find_number_input_by_key(at: AppTest, key: str):
    return next(widget for widget in at.number_input if widget.key == key)


def _caption_values(at: AppTest) -> list[str]:
    return [str(widget.value) for widget in at.caption]


def test_match_metric_names_excludes_non_match_metrics() -> None:
    names = app._match_metric_names()
    assert "accuracy" in names
    assert "positive_predictive_value" in names
    assert "f1_score" not in names
    assert "matthews_correlation_coefficient" not in names


def test_resolve_match_method_matches_metric_family() -> None:
    assert app._resolve_match_method("accuracy", "Approximate") == "peizer_pratt"
    assert app._resolve_match_method("marginal_benefit", "Approximate") == "normal"
    assert app._resolve_match_method("positive_predictive_value", "Approximate") == "beta"
    assert app._resolve_match_method("accuracy", "Exact") == "exact"


def test_intra_group_metric_sections_split_rate_and_signed_metrics() -> None:
    assert [section.title for section in app.INTRA_GROUP_METRIC_SECTIONS] == ["Rate Metrics", "Signed Metrics"]
    assert app.INTRA_GROUP_METRIC_SECTIONS[0].metric_groups == (
        (
            "accuracy",
            "predicted_positive_rate",
            "true_positive_rate",
            "false_positive_rate",
            "positive_predictive_value",
        ),
    )
    assert app.INTRA_GROUP_METRIC_SECTIONS[0].yaxis_range == (0.0, 1.0)
    assert app.INTRA_GROUP_METRIC_SECTIONS[1].metric_groups == (
        ("marginal_benefit",),
        ("matthews_correlation_coefficient",),
    )
    assert app.INTRA_GROUP_METRIC_SECTIONS[1].yaxis_range is None
    assert app.INTRA_GROUP_METRIC_SECTIONS[1].show_zero_line is True
    assert app.INTRA_GROUP_METRIC_SECTIONS[1].horizontal_stack is True


def test_intra_group_metric_values_preserve_group_comparison_by_section() -> None:
    cm_i = metrics.create_cm(40, 10, 5, 45)
    cm_j = metrics.create_cm(35, 15, 10, 40)
    group_i_values, group_j_values = app._intra_group_metric_values(
        cm_i,
        cm_j,
        app.INTRA_GROUP_METRIC_SECTIONS[0].metric_groups[0],
    )
    assert list(group_i_values.keys()) == [
        "Accuracy",
        "Predicted Positive Rate",
        "True Positive Rate",
        "False Positive Rate",
        "Positive Predictive Value",
    ]
    assert list(group_i_values.keys()) == list(group_j_values.keys())


def test_intra_group_chart_title_uses_metric_name_for_single_metric_group() -> None:
    section = app.INTRA_GROUP_METRIC_SECTIONS[1]
    assert app._intra_group_chart_title(section, ("marginal_benefit",)) == "Marginal Benefit"
    assert (
        app._intra_group_chart_title(section, ("matthews_correlation_coefficient",))
        == "Matthews Correlation Coefficient"
    )


def test_analytic_match_result_uses_requested_sample_size() -> None:
    observed = metrics.create_cm(8, 2, 1, 9)
    reference_probs = np.array([[0.45, 0.05], [0.1, 0.4]], dtype=np.float64)
    result = app._analytic_match_result(
        observed,
        reference_probs,
        "accuracy",
        sample_n=50,
        alternative="two-sided",
        match_mode="Exact",
    )
    assert result["n_obs"] == 50
    assert result["method"] == "exact"


def test_analytic_match_result_uses_expected_beta_when_sample_n_changes() -> None:
    observed = metrics.create_cm(8, 2, 1, 9)
    reference_probs = np.array([[0.45, 0.05], [0.1, 0.4]], dtype=np.float64)
    result = app._analytic_match_result(
        observed,
        reference_probs,
        "positive_predictive_value",
        sample_n=50,
        alternative="two-sided",
        match_mode="Approximate",
    )
    assert result["method"] == "beta"
    assert result["details"]["beta_count"] == "expected"


def test_match_summary_table_uses_arrow_safe_string_columns() -> None:
    execution = app._execute_match_payload(
        {
            "observed_values": (8.0, 2.0, 1.0, 9.0),
            "reference_probs_values": (0.45, 0.05, 0.1, 0.4),
            "metric_name": "accuracy",
            "match_mode": "Approximate",
            "n": 20,
            "num_samples": 1000,
            "alternative": "two-sided",
            "seed": 123,
        }
    )
    assert execution["status"] == "completed"
    result = execution["result"]
    assert result is not None
    summary = app._match_summary_table(result)
    assert list(summary.columns) == ["Statistic", "Value"]
    assert summary["Statistic"].dtype.name == "string"
    assert summary["Value"].dtype.name == "string"
    assert "Analytic Method" in set(summary["Statistic"])
    assert "Number of Samples" in set(summary["Statistic"])


def test_match_interpretation_varies_by_alternative() -> None:
    assert (
        app._match_interpretation({"status": "ok", "alternative": "two-sided", "p_value": 0.4})
        == "The observed score is plausibly consistent with the reference distribution."
    )
    assert (
        app._match_interpretation({"status": "ok", "alternative": "less", "p_value": 0.4})
        == "There is not strong evidence that the observed score is unusually low relative to the reference distribution."
    )
    assert (
        app._match_interpretation({"status": "ok", "alternative": "greater", "p_value": 0.4})
        == "There is not strong evidence that the observed score is unusually high relative to the reference distribution."
    )


def test_match_interpretation_greater_large_p_value() -> None:
    msg = app._match_interpretation(
        {
            "status": "ok",
            "alternative": "greater",
            "p_value": 0.99,
        }
    )
    assert "not strong evidence" in msg
    assert "unusually high" in msg


def test_match_interpretation_warns_for_small_p_value() -> None:
    msg = app._match_interpretation({"status": "ok", "alternative": "two-sided", "p_value": 0.01})
    assert "may warrant further investigation" in msg


def test_match_interpretation_uses_directional_messages_for_small_p_value() -> None:
    assert (
        app._match_interpretation({"status": "ok", "alternative": "less", "p_value": 0.01})
        == "The observed score is unusually low relative to the reference distribution."
    )
    assert (
        app._match_interpretation({"status": "ok", "alternative": "greater", "p_value": 0.01})
        == "The observed score is unusually high relative to the reference distribution."
    )


def test_match_alternative_description_returns_expected_caption() -> None:
    assert (
        app._match_alternative_description("two-sided")
        == "Tests whether the observed score is unusual in the reference distribution."
    )
    assert (
        app._match_alternative_description("less")
        == "Tests whether the observed score is unusually low; p-value is the lower-tail probability."
    )
    assert (
        app._match_alternative_description("greater")
        == "Tests whether the observed score is unusually high; p-value is the upper-tail probability."
    )


def test_execute_match_payload_approximate_binomial_metric_completes() -> None:
    observed = metrics.create_cm(8, 2, 1, 9)
    reference_probs = np.array([[0.45, 0.05], [0.1, 0.4]], dtype=np.float64)
    execution = app._execute_match_payload(
        {
            "observed_values": (
                float(observed[0, 0]),
                float(observed[0, 1]),
                float(observed[1, 0]),
                float(observed[1, 1]),
            ),
            "reference_probs_values": (
                float(reference_probs[0, 0]),
                float(reference_probs[0, 1]),
                float(reference_probs[1, 0]),
                float(reference_probs[1, 1]),
            ),
            "metric_name": "accuracy",
            "match_mode": "Approximate",
            "n": 20,
            "num_samples": 1000,
            "alternative": "two-sided",
            "seed": 123,
        }
    )
    assert execution["status"] == "completed"
    result = execution["result"]
    assert result is not None
    assert result["method"] == "peizer_pratt"
    assert np.isfinite(result["analytic"]["p_value"])
    assert result["simulation"]["status"] == "ok"


def test_execute_match_payload_approximate_joint_ratio_metric_completes() -> None:
    observed = metrics.create_cm(8, 2, 1, 9)
    reference_probs = np.array([[0.45, 0.05], [0.1, 0.4]], dtype=np.float64)
    execution = app._execute_match_payload(
        {
            "observed_values": (
                float(observed[0, 0]),
                float(observed[0, 1]),
                float(observed[1, 0]),
                float(observed[1, 1]),
            ),
            "reference_probs_values": (
                float(reference_probs[0, 0]),
                float(reference_probs[0, 1]),
                float(reference_probs[1, 0]),
                float(reference_probs[1, 1]),
            ),
            "metric_name": "positive_predictive_value",
            "match_mode": "Approximate",
            "n": 20,
            "num_samples": 1000,
            "alternative": "two-sided",
            "seed": 456,
        }
    )
    assert execution["status"] == "completed"
    result = execution["result"]
    assert result is not None
    assert result["method"] == "beta"
    assert np.isfinite(result["analytic"]["p_value"])
    assert result["simulation"]["status"] == "ok"


def test_execute_match_payload_approximate_marginal_benefit_completes() -> None:
    observed = metrics.create_cm(8, 2, 1, 9)
    reference_probs = np.array([[0.45, 0.05], [0.1, 0.4]], dtype=np.float64)
    execution = app._execute_match_payload(
        {
            "observed_values": (
                float(observed[0, 0]),
                float(observed[0, 1]),
                float(observed[1, 0]),
                float(observed[1, 1]),
            ),
            "reference_probs_values": (
                float(reference_probs[0, 0]),
                float(reference_probs[0, 1]),
                float(reference_probs[1, 0]),
                float(reference_probs[1, 1]),
            ),
            "metric_name": "marginal_benefit",
            "match_mode": "Approximate",
            "n": 20,
            "num_samples": 1000,
            "alternative": "two-sided",
            "seed": 789,
        }
    )
    assert execution["status"] == "completed"
    result = execution["result"]
    assert result is not None
    assert result["method"] == "normal"
    assert np.isfinite(result["analytic"]["p_value"])
    assert result["simulation"]["status"] == "ok"


def _run_approximate_match_ui(metric_label: str) -> AppTest:
    at = AppTest.from_file("src/app.py", default_timeout=15)
    at.run(timeout=15)
    _find_radio(at, "MATCH mode").set_value("Approximate")
    _find_selectbox(at, "Metric", key="match_metric_selector").set_value(metric_label)
    _find_number_input(at, "Number of simulations").set_value(1000)
    at.button[0].click()
    at.run(timeout=15)
    return at


def _metric_values_by_label(at: AppTest) -> dict[str, str]:
    return {metric.label: metric.value for metric in at.metric}


def test_match_ui_shows_default_accuracy_result_on_first_render() -> None:
    at = AppTest.from_file("src/app.py", default_timeout=15)
    at.run(timeout=15)

    metric_values = _metric_values_by_label(at)
    assert _find_selectbox(at, "Metric", key="match_metric_selector").value == "accuracy"
    assert _find_radio(at, "MATCH mode").value == "Approximate"
    assert "Observed score" in metric_values
    assert "p-value" in metric_values
    assert metric_values["Analytic method"] == "peizer_pratt"
    assert metric_values["Status"] == "ok"
    assert not at.error
    assert not at.exception


def test_match_ui_does_not_restart_default_run_on_rerender() -> None:
    at = AppTest.from_file("src/app.py", default_timeout=15)
    at.run(timeout=15)
    first_metric_values = _metric_values_by_label(at)

    at.run(timeout=15)
    second_metric_values = _metric_values_by_label(at)

    assert first_metric_values == second_metric_values
    assert not any("MATCH run in progress" in str(widget.value) for widget in at.info)
    assert not at.error
    assert not at.exception


def test_approximate_match_ui_renders_result_for_binomial_metric() -> None:
    at = _run_approximate_match_ui("Accuracy")
    metric_values = _metric_values_by_label(at)
    assert "Observed score" in metric_values
    assert "p-value" in metric_values
    assert metric_values["Status"] == "ok"
    assert not at.error
    assert not at.exception


def test_approximate_match_ui_renders_result_for_joint_ratio_metric() -> None:
    at = _run_approximate_match_ui("Positive Predictive Value")
    metric_values = _metric_values_by_label(at)
    assert "Observed score" in metric_values
    assert "p-value" in metric_values
    assert metric_values["Status"] == "ok"
    assert not at.error
    assert not at.exception


def test_approximate_match_ui_renders_result_for_marginal_benefit() -> None:
    at = _run_approximate_match_ui("Marginal Benefit")
    metric_values = _metric_values_by_label(at)
    assert "Observed score" in metric_values
    assert "p-value" in metric_values
    assert metric_values["Status"] == "ok"
    assert not at.error
    assert not at.exception


def test_match_ui_hides_sample_size_input_and_reference_mode_radio() -> None:
    at = AppTest.from_file("src/app.py", default_timeout=15)
    at.run(timeout=15)
    assert not any(widget.label == "Sample size n" for widget in at.number_input)
    assert not any(widget.label == "Reference mode" for widget in at.radio)


def test_match_ui_shows_observed_and_reference_value_captions() -> None:
    at = AppTest.from_file("src/app.py", default_timeout=15)
    at.run(timeout=15)
    captions = _caption_values(at)
    assert "Observed values must be non-negative integers." in captions
    assert "Reference values allow non-negative decimals. For example, you may enter a probability distribution." in captions


def test_reference_cm_mode_hides_reference_total_n_caption() -> None:
    at = AppTest.from_file("src/app.py", default_timeout=15)
    at.run(timeout=15)
    captions = _caption_values(at)
    assert sum(value.startswith("Total n = ") for value in captions) == 3
    assert "Simulations are only done for visualization purposes. The MATCH Test does not use simulations." in captions


def test_reference_cm_mode_uses_observed_subgroup_total_for_n() -> None:
    at = AppTest.from_file("src/app.py", default_timeout=15)
    at.run(timeout=15)
    _find_radio(at, "MATCH mode").set_value("Approximate")
    _find_selectbox(at, "Metric", key="match_metric_selector").set_value("Accuracy")
    _find_number_input_by_key(at, "observed_subgroup_cm_tp").set_value(10)
    _find_number_input_by_key(at, "observed_subgroup_cm_fn").set_value(5)
    _find_number_input_by_key(at, "observed_subgroup_cm_fp").set_value(5)
    _find_number_input_by_key(at, "observed_subgroup_cm_tn").set_value(30)
    _find_number_input_by_key(at, "reference_cm_tp").set_value(20)
    _find_number_input_by_key(at, "reference_cm_fn").set_value(5)
    _find_number_input_by_key(at, "reference_cm_fp").set_value(5)
    _find_number_input_by_key(at, "reference_cm_tn").set_value(20)
    _find_number_input(at, "Number of simulations").set_value(1000)
    at.button[0].click()
    at.run(timeout=15)
    summary = at.dataframe[-1].value
    n_row = summary.loc[summary["Statistic"] == "Observed Sample Size", "Value"]
    assert n_row.iloc[0] == "50"


def test_match_ui_accepts_decimal_reference_confusion_matrix_values() -> None:
    at = AppTest.from_file("src/app.py", default_timeout=15)
    at.run(timeout=15)
    _find_number_input_by_key(at, "reference_cm_tp").set_value(20.5)
    _find_number_input_by_key(at, "reference_cm_fn").set_value(5.25)
    _find_number_input_by_key(at, "reference_cm_fp").set_value(5.25)
    _find_number_input_by_key(at, "reference_cm_tn").set_value(19.0)
    at.run(timeout=15)
    _find_number_input(at, "Number of simulations").set_value(1000)
    at.button[0].click()
    at.run(timeout=15)
    metric_values = _metric_values_by_label(at)
    assert "Observed score" in metric_values
    assert metric_values["Status"] == "ok"
    assert not at.error


def test_match_ui_shows_formula_latex_for_false_discovery_rate() -> None:
    at = AppTest.from_file("src/app.py", default_timeout=15)
    at.run(timeout=15)
    _find_selectbox(at, "Metric", key="match_metric_selector").set_value("False Discovery Rate")
    at.run(timeout=15)
    latex_values = [str(widget.value) for widget in at.latex]
    assert any("FP" in value and "TP" in value for value in latex_values)


def test_match_ui_shows_alternative_hypothesis_caption() -> None:
    at = AppTest.from_file("src/app.py", default_timeout=15)
    at.run(timeout=15)
    assert (
        "Tests whether the observed score is unusual in the reference distribution."
        in _caption_values(at)
    )

    _find_selectbox(at, "Alternative hypothesis", key="match_alternative_selector").set_value("less")
    at.run(timeout=15)
    assert (
        "Tests whether the observed score is unusually low; p-value is the lower-tail probability."
        in _caption_values(at)
    )


def test_all_fairness_metrics_table_uses_pretty_columns_and_formula_metadata() -> None:
    cm_i = metrics.create_cm(40, 10, 5, 45)
    cm_j = metrics.create_cm(35, 15, 10, 40)

    table = app._all_fairness_metrics_table(cm_i, cm_j)

    assert list(table.columns) == [
        "Metric",
        "Value",
        "No-Bias Value",
        "Intra-Group Metric g",
        "Fairness Metric M",
    ]
    assert "Family" not in table.columns

    accuracy_row = table.loc[table["Metric"] == "Accuracy Difference"].iloc[0]
    assert "TP" in accuracy_row["Intra-Group Metric g"]
    assert "TN" in accuracy_row["Intra-Group Metric g"]
    assert accuracy_row["Fairness Metric M"] == r"M(i,j) = g_i - g_j"


def test_all_fairness_metrics_table_preserves_conditional_rejection_orientation() -> None:
    cm_i = metrics.create_cm(40, 10, 5, 45)
    cm_j = metrics.create_cm(35, 15, 10, 40)

    table = app._all_fairness_metrics_table(cm_i, cm_j)
    row = table.loc[table["Metric"] == "Difference in Conditional Rejection"].iloc[0]

    assert row["Fairness Metric M"] == r"M(i,j) = g_j - g_i"
