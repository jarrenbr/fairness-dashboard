"""Plotting support tests for Tab 1."""

from __future__ import annotations

from src import metrics
import numpy as np

from src.plotting import (
    confusion_matrix_heatmap,
    fairness_bias_gauge,
    intra_group_bar_chart,
    metric_distribution_histogram,
    metric_ecdf,
)


def test_confusion_matrix_heatmap_has_expected_axis_titles() -> None:
    fig = confusion_matrix_heatmap(metrics.create_cm(1, 2, 3, 4), "Test")
    assert fig.layout.xaxis.title.text == "Predicted Condition"
    assert fig.layout.yaxis.title.text == "Actual Condition"


def test_intra_group_bar_chart_has_two_traces() -> None:
    fig = intra_group_bar_chart({"Accuracy": 0.8, "TPR": 0.7}, {"Accuracy": 0.6, "TPR": 0.5})
    assert len(fig.data) == 2


def test_intra_group_bar_chart_accepts_fixed_rate_axis_range() -> None:
    fig = intra_group_bar_chart(
        {"Accuracy": 0.8, "TPR": 0.7},
        {"Accuracy": 0.6, "TPR": 0.5},
        title="Rate Metrics",
        yaxis_range=(0.0, 1.0),
    )
    assert fig.layout.title.text == "Rate Metrics"
    assert tuple(fig.layout.yaxis.range) == (0.0, 1.0)
    assert fig.layout.yaxis.zeroline is False


def test_intra_group_bar_chart_supports_signed_zero_line() -> None:
    fig = intra_group_bar_chart(
        {"Marginal Benefit": 0.2},
        {"Marginal Benefit": -0.05},
        title="Marginal Benefit",
        show_zero_line=True,
    )
    assert fig.layout.title.text == "Marginal Benefit"
    assert fig.layout.yaxis.range is None
    assert fig.layout.yaxis.zeroline is True


def test_fairness_bias_gauge_marks_no_bias_point() -> None:
    fig = fairness_bias_gauge(0.2, 0.0, "Bias")
    assert fig.layout.shapes is not None
    assert len(fig.layout.shapes) >= 1


def test_metric_distribution_histogram_adds_observed_marker() -> None:
    fig = metric_distribution_histogram(
        np.array([0.1, 0.2, 0.3]),
        0.25,
        "Histogram",
        metric_display_name="False Discovery Rate",
        p_value=0.935,
        method="beta",
    )
    names = {trace.name for trace in fig.data if getattr(trace, "name", None)}
    assert "Observed score" in names
    assert fig.layout.yaxis.title.text == "Reference probability"
    assert "False Discovery Rate" in fig.layout.title.text
    assert "p = 0.935" in fig.layout.title.text


def test_match_distribution_plot_has_required_legend_items() -> None:
    values = np.linspace(0.0, 1.0, 101)
    fig = metric_distribution_histogram(values, observed_value=0.5, title="Test")
    names = {trace.name for trace in fig.data if getattr(trace, "name", None)}
    assert "Reference distribution" in names
    assert "Observed score" in names
    assert "Reference mean" in names
    assert "5th percentile" in names
    assert "95th percentile" in names
    assert "Central 90% interval" in names


def test_match_distribution_plot_handles_empty_values() -> None:
    fig = metric_distribution_histogram(np.array([np.nan, np.nan]), observed_value=np.nan, title="Empty")
    assert fig.layout.yaxis.title.text == "Reference probability"
    assert len(fig.data) == 0


def test_match_distribution_plot_uses_bar_chart_for_discrete_values() -> None:
    fig = metric_distribution_histogram(np.array([0.1, 0.2, 0.2, 0.3, 0.4]), observed_value=0.25, title="Discrete")
    assert fig.data[0].type == "bar"


def test_match_distribution_plot_uses_histogram_for_many_unique_values() -> None:
    fig = metric_distribution_histogram(np.linspace(0.0, 1.0, 200), observed_value=0.25, title="Continuous")
    assert fig.data[0].type == "histogram"


def test_metric_ecdf_contains_ecdf_and_observed_traces() -> None:
    fig = metric_ecdf(np.array([0.1, 0.2, 0.3]), 0.25, "ECDF")
    assert len(fig.data) == 2
    assert fig.data[0].name == "ECDF"
    assert fig.data[1].name == "Observed score"
    assert fig.layout.yaxis.title.text == "Empirical cumulative probability"
