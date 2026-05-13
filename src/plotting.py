"""Pure Plotly figure factories for the dashboard."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

PRIMARY_BLUE = "#1f77b4"
ALERT_RED = "#C53030"
REFERENCE_GREEN = "#2F855A"
SLATE = "#64748B"
SLATE_DARK = "#2D3748"
REFERENCE_BAND = "rgba(100, 116, 139, 0.10)"
REFERENCE_BAND_LEGEND = "rgba(100, 116, 139, 0.25)"


def _finite_values(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return array[np.isfinite(array)]


def _title_with_subtitle(
    title: str,
    *,
    metric_display_name: str | None = None,
    observed_value: float | None = None,
    p_value: float | None = None,
    method: str | None = None,
) -> str:
    subtitle_parts: list[str] = []
    if metric_display_name:
        subtitle_parts.append(metric_display_name)
    if observed_value is not None and np.isfinite(observed_value):
        subtitle_parts.append(f"observed = {float(observed_value):.3f}")
    if p_value is not None and np.isfinite(p_value):
        subtitle_parts.append(f"p = {float(p_value):.3f}")
    if method:
        subtitle_parts.append(f"analytic = {method}")
    if not subtitle_parts:
        return title
    return f"{title}<br><sup>{' | '.join(subtitle_parts)}</sup>"


def _add_vertical_line_trace(
    fig: go.Figure,
    *,
    x: float,
    y_max: float,
    name: str,
    color: str,
    dash: str = "solid",
    width: int = 2,
) -> None:
    fig.add_trace(
        go.Scatter(
            x=[x, x],
            y=[0.0, y_max],
            mode="lines",
            name=name,
            line={"color": color, "dash": dash, "width": width},
            hovertemplate=f"{name}: {x:.6g}<extra></extra>",
        )
    )


def confusion_matrix_heatmap(cm: np.ndarray, title: str) -> go.Figure:
    """Return a confusion-matrix heatmap preserving the repository CM layout."""

    matrix = np.asarray(cm, dtype=np.float64)
    labels = np.vectorize(lambda value: f"{value:g}")(matrix)
    fig = go.Figure(
        data=[
            go.Heatmap(
                z=matrix,
                x=["Predicted Positive", "Predicted Negative"],
                y=["Actual Positive", "Actual Negative"],
                text=labels,
                texttemplate="%{text}",
                colorscale="Blues",
                hovertemplate="Actual=%{y}<br>Predicted=%{x}<br>Count=%{z}<extra></extra>",
                colorbar={"title": "Count"},
            )
        ]
    )
    fig.update_layout(
        title=title,
        xaxis_title="Predicted Condition",
        yaxis_title="Actual Condition",
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
        height=360,
    )
    fig.update_yaxes(autorange="reversed")
    return fig


def intra_group_bar_chart(
    group_i_values: dict[str, float],
    group_j_values: dict[str, float],
    *,
    title: str = "Within-Group Metric Comparison",
    yaxis_range: tuple[float, float] | None = None,
    show_zero_line: bool = False,
) -> go.Figure:
    """Return a grouped bar chart for within-group metric comparison."""

    metric_names = list(group_i_values.keys())
    fig = go.Figure()
    fig.add_bar(name="Group i", x=metric_names, y=[group_i_values[name] for name in metric_names], marker_color=PRIMARY_BLUE)
    fig.add_bar(name="Group j", x=metric_names, y=[group_j_values[name] for name in metric_names], marker_color="#d62728")
    fig.update_layout(
        title=title,
        barmode="group",
        xaxis_title="Metric",
        yaxis_title="Value",
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
        height=420,
    )
    fig.update_yaxes(
        range=list(yaxis_range) if yaxis_range is not None else None,
        zeroline=show_zero_line,
        zerolinecolor=SLATE_DARK if show_zero_line else None,
        zerolinewidth=2 if show_zero_line else None,
    )
    return fig


def fairness_bias_gauge(value: float, no_bias_value: float, title: str) -> go.Figure:
    """Return a simple horizontal bias indicator centered at a no-bias point."""

    span = 1.0
    if np.isfinite(value):
        span = max(span, abs(value - no_bias_value) * 1.35)
    if no_bias_value == 1.0:
        x_min = min(0.0, no_bias_value - span)
        x_max = no_bias_value + span
    else:
        x_min = no_bias_value - span
        x_max = no_bias_value + span

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[value] if np.isfinite(value) else [],
            y=[0],
            mode="markers",
            marker={"size": 14, "color": ALERT_RED},
            name="Selected value",
            hovertemplate="Value=%{x:.6g}<extra></extra>",
        )
    )
    fig.add_vline(x=no_bias_value, line_dash="dash", line_color=SLATE_DARK, annotation_text="No-bias")
    fig.update_layout(
        title=title,
        xaxis={"range": [x_min, x_max], "title": "Metric value"},
        yaxis={"visible": False, "range": [-1, 1]},
        showlegend=False,
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
        height=220,
    )
    return fig


def metric_distribution_histogram(
    values: np.ndarray,
    observed_value: float,
    title: str,
    *,
    metric_display_name: str | None = None,
    p_value: float | None = None,
    method: str | None = None,
) -> go.Figure:
    """Return a polished distribution plot for finite reference metric values."""

    finite_values = _finite_values(values)
    title_text = _title_with_subtitle(
        title,
        metric_display_name=metric_display_name,
        observed_value=observed_value,
        p_value=p_value,
        method=method,
    )

    fig = go.Figure()
    fig.update_layout(
        title={"text": title_text, "y": 0.97},
        xaxis_title="Metric value",
        yaxis_title="Reference probability",
        template="plotly_white",
        bargap=0.05,
        margin={"l": 30, "r": 30, "t": 105, "b": 40},
        height=430,
        showlegend=True,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "center",
            "x": 0.5,
        },
    )

    if finite_values.size == 0:
        return fig

    mean_value = float(np.mean(finite_values))
    q05, q95 = np.quantile(finite_values, [0.05, 0.95])
    q05 = float(q05)
    q95 = float(q95)

    unique_values, counts = np.unique(np.round(finite_values, 12), return_counts=True)

    if unique_values.size <= 60:
        probs = counts / counts.sum()
        fig.add_bar(
            x=unique_values,
            y=probs,
            name="Reference distribution",
            marker_color=PRIMARY_BLUE,
            opacity=0.85,
            hovertemplate=(
                "Metric value=%{x:.6g}<br>"
                "Simulation proportion=%{y:.4f}<extra></extra>"
            ),
        )
        y_max = float(np.max(probs)) * 1.15 if probs.size else 1.0
    else:
        nbins = min(50, max(15, int(np.sqrt(finite_values.size))))
        fig.add_histogram(
            x=finite_values,
            nbinsx=nbins,
            histnorm="probability",
            name="Reference distribution",
            marker_color=PRIMARY_BLUE,
            opacity=0.85,
            hovertemplate=(
                "Metric value=%{x:.6g}<br>"
                "Simulation proportion=%{y:.4f}<extra></extra>"
            ),
        )
        hist_counts, _ = np.histogram(finite_values, bins=nbins)
        hist_probs = hist_counts / hist_counts.sum()
        y_max = float(np.max(hist_probs)) * 1.15 if hist_probs.size else 1.0

    y_max = max(y_max, 1e-6)

    fig.add_vrect(
        x0=q05,
        x1=q95,
        fillcolor=REFERENCE_BAND,
        line_width=0,
        layer="below",
    )

    _add_vertical_line_trace(fig, x=mean_value, y_max=y_max, name="Reference mean", color=REFERENCE_GREEN, dash="dash", width=2)
    if np.isfinite(observed_value):
        _add_vertical_line_trace(fig, x=float(observed_value), y_max=y_max, name="Observed score", color=ALERT_RED, dash="solid", width=3)
    _add_vertical_line_trace(fig, x=q05, y_max=y_max, name="5th percentile", color=SLATE, dash="dot", width=2)
    _add_vertical_line_trace(fig, x=q95, y_max=y_max, name="95th percentile", color=SLATE, dash="dot", width=2)

    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            name="Central 90% interval",
            marker={"size": 10, "color": REFERENCE_BAND_LEGEND, "symbol": "square"},
            hoverinfo="skip",
        )
    )

    range_values = [float(np.min(finite_values)), float(np.max(finite_values)), mean_value, q05, q95]
    if np.isfinite(observed_value):
        range_values.append(float(observed_value))
    range_min = min(range_values)
    range_max = max(range_values)
    pad = 0.05 * max(1e-8, range_max - range_min)
    fig.update_xaxes(range=[range_min - pad, range_max + pad])
    fig.update_yaxes(range=[0.0, y_max], title="Reference probability")
    return fig


def metric_ecdf(values: np.ndarray, observed_value: float, title: str) -> go.Figure:
    """Return an ECDF-style line plot for finite metric values."""

    finite_values = np.sort(_finite_values(values))
    cumulative = np.arange(1, finite_values.size + 1, dtype=np.float64)
    if finite_values.size > 0:
        cumulative = cumulative / finite_values.size

    fig = go.Figure()
    fig.add_scatter(
        x=finite_values,
        y=cumulative,
        mode="lines",
        line={"color": PRIMARY_BLUE, "width": 2},
        name="ECDF",
    )
    if np.isfinite(observed_value):
        fig.add_scatter(
            x=[float(observed_value), float(observed_value)],
            y=[0.0, 1.0],
            mode="lines",
            line={"color": ALERT_RED, "width": 3},
            name="Observed score",
            hovertemplate=f"Observed score: {float(observed_value):.6g}<extra></extra>",
        )
    fig.update_layout(
        title=title,
        xaxis_title="Metric value",
        yaxis_title="Empirical cumulative probability",
        template="plotly_white",
        margin={"l": 30, "r": 30, "t": 70, "b": 40},
        height=380,
        showlegend=True,
    )
    return fig
