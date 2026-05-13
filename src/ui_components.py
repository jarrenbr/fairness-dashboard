"""Reusable Streamlit UI helpers for confusion-matrix workflows."""

from __future__ import annotations

import inspect
import math
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

try:
    from .metrics import (
        FN,
        FP,
        TN,
        TP,
        METRIC_INFO,
        SINGLE_CM_METRICS,
        TWO_CM_FAIRNESS_METRICS,
        create_cm,
        total_count,
    )
    from .dashboard_logic import metric_display_name
    from .plotting import confusion_matrix_heatmap
except ImportError:  # pragma: no cover - supports `streamlit run src/app.py`
    from metrics import (
        FN,
        FP,
        TN,
        TP,
        METRIC_INFO,
        SINGLE_CM_METRICS,
        TWO_CM_FAIRNESS_METRICS,
        create_cm,
        total_count,
    )
    from dashboard_logic import metric_display_name
    from plotting import confusion_matrix_heatmap


def _format_metric_value(value: object) -> str:
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return "nan"
        return f"{float(value):.6g}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def _format_cm_numeric(value: float) -> str:
    value_float = float(value)
    if np.isclose(value_float, round(value_float)):
        return str(int(round(value_float)))
    return f"{value_float:.6g}"


def _normalize_cm_input_value(previous: float, current: float) -> float:
    """Normalize a stepped CM input value while preserving integer display when possible."""

    normalized = float(current)
    if normalized < 0.0:
        normalized = 0.0
    elif np.isclose(normalized, previous + 1.0) and not np.isclose(previous, round(previous)):
        normalized = float(math.ceil(previous))
    elif np.isclose(normalized, previous - 1.0) and not np.isclose(previous, round(previous)):
        normalized = float(math.floor(previous))

    if np.isclose(normalized, round(normalized)):
        normalized = float(round(normalized))
    return normalized


def _cm_input_on_change(value_key: str, previous_key: str) -> None:
    previous = float(st.session_state.get(previous_key, st.session_state.get(value_key, 0.0)))
    current = float(st.session_state.get(value_key, previous))
    normalized = _normalize_cm_input_value(previous, current)
    st.session_state[value_key] = normalized
    st.session_state[previous_key] = normalized


def cm_input(
    label: str,
    default: tuple[int | float, int | float, int | float, int | float],
    *,
    show_total: bool = True,
    key_prefix: str | None = None,
    allow_decimal: bool = True,
) -> np.ndarray:
    """Render a 2x2 confusion matrix editor and return a float64 CM."""

    st.subheader(label)
    key_prefix = key_prefix or label.lower().replace(" ", "_")
    defaults = {
        "tp": float(default[0]) if allow_decimal else int(round(float(default[0]))),
        "fn": float(default[1]) if allow_decimal else int(round(float(default[1]))),
        "fp": float(default[2]) if allow_decimal else int(round(float(default[2]))),
        "tn": float(default[3]) if allow_decimal else int(round(float(default[3]))),
    }
    for name, default_value in defaults.items():
        value_key = f"{key_prefix}_{name}"
        previous_key = f"{value_key}_previous"
        if value_key not in st.session_state:
            st.session_state[value_key] = default_value
        if allow_decimal and previous_key not in st.session_state:
            st.session_state[previous_key] = float(st.session_state[value_key])

    min_value = 0.0 if allow_decimal else 0
    step = 1.0 if allow_decimal else 1
    value_format = "%g" if allow_decimal else "%d"
    on_change = _cm_input_on_change if allow_decimal else None

    col1, col2 = st.columns(2)
    with col1:
        tp = st.number_input(
            "TP",
            min_value=min_value,
            step=step,
            format=value_format,
            key=f"{key_prefix}_tp",
            on_change=on_change,
            args=(f"{key_prefix}_tp", f"{key_prefix}_tp_previous") if allow_decimal else None,
        )
        fp = st.number_input(
            "FP",
            min_value=min_value,
            step=step,
            format=value_format,
            key=f"{key_prefix}_fp",
            on_change=on_change,
            args=(f"{key_prefix}_fp", f"{key_prefix}_fp_previous") if allow_decimal else None,
        )
    with col2:
        fn = st.number_input(
            "FN",
            min_value=min_value,
            step=step,
            format=value_format,
            key=f"{key_prefix}_fn",
            on_change=on_change,
            args=(f"{key_prefix}_fn", f"{key_prefix}_fn_previous") if allow_decimal else None,
        )
        tn = st.number_input(
            "TN",
            min_value=min_value,
            step=step,
            format=value_format,
            key=f"{key_prefix}_tn",
            on_change=on_change,
            args=(f"{key_prefix}_tn", f"{key_prefix}_tn_previous") if allow_decimal else None,
        )

    cm = create_cm(tp, fn, fp, tn)
    if show_total:
        st.caption(f"Total n = {_format_cm_numeric(float(total_count(cm)))}")
    return cm.astype(np.float64)


def cm_summary(cm: np.ndarray) -> dict[str, float]:
    """Return a count summary for one confusion matrix."""

    matrix = np.asarray(cm, dtype=np.float64)
    return {
        "tp": float(matrix[TP]),
        "fn": float(matrix[FN]),
        "fp": float(matrix[FP]),
        "tn": float(matrix[TN]),
        "actual_positive": float(matrix[TP] + matrix[FN]),
        "actual_negative": float(matrix[FP] + matrix[TN]),
        "predicted_positive": float(matrix[TP] + matrix[FP]),
        "predicted_negative": float(matrix[FN] + matrix[TN]),
        "n": float(matrix.sum()),
    }


def metric_selectbox(
    key: str,
    include_fairness: bool = False,
    metric_names: list[str] | None = None,
) -> str:
    """Render a metric selectbox and return the selected metric key."""

    options = list(metric_names) if metric_names is not None else list(
        TWO_CM_FAIRNESS_METRICS if include_fairness else SINGLE_CM_METRICS
    )
    default_name = "objective_fairness_index" if include_fairness and "objective_fairness_index" in options else options[0]
    default_index = options.index(default_name)
    return st.selectbox(
        "Metric",
        options,
        index=default_index,
        key=key,
        format_func=lambda name: str(METRIC_INFO.get(name, {}).get("display_name", name)),
    )


def show_cm_heatmap(cm: np.ndarray, title: str) -> None:
    """Render a confusion-matrix heatmap."""

    st.plotly_chart(confusion_matrix_heatmap(cm, title), width="stretch")


@lru_cache(maxsize=1)
def _supports_metric_border() -> bool:
    return "border" in inspect.signature(st.metric).parameters


def _metric_latex(metric_name: str) -> str | None:
    """Return the best available formula representation for a metric."""

    info = METRIC_INFO.get(metric_name, {})
    for key in ("latex", "g_latex", "formula_text"):
        value = info.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _metric_uses_latex(metric_name: str) -> bool:
    info = METRIC_INFO.get(metric_name, {})
    for key in ("latex", "g_latex"):
        value = info.get(key)
        if isinstance(value, str) and value.strip():
            return True
    return False


def _short_interpretation(metric_name: str) -> str:
    info = METRIC_INFO.get(metric_name, {})
    interpretation = str(info.get("interpretation", "")).strip()
    if not interpretation:
        return ""
    split_idx = interpretation.find(". ")
    if split_idx >= 0:
        return interpretation[: split_idx + 1]
    return interpretation if interpretation.endswith(".") else f"{interpretation}."


def render_metric_formula_card(metric_name: str) -> None:
    """Render a bordered formula summary card for the selected metric."""

    info = METRIC_INFO.get(metric_name, {})
    display_name = metric_display_name(metric_name)
    abbreviation = str(info.get("abbreviation", "")).strip()
    family = str(info.get("family", "")).strip()
    range_text = str(info.get("range_text", "")).strip()
    formula = _metric_latex(metric_name)
    short_interpretation = _short_interpretation(metric_name)

    with st.container(border=True):
        st.markdown(f"**{display_name}**")

        metadata_parts = [
            f"Abbreviation: {abbreviation}" if abbreviation else "",
            f"Family: {family}" if family else "",
            f"Range: {range_text}" if range_text else "",
        ]
        metadata_line = " | ".join(part for part in metadata_parts if part)
        if metadata_line:
            st.caption(metadata_line)

        if formula:
            if _metric_uses_latex(metric_name):
                st.latex(formula)
            else:
                st.write(f"Formula: {formula}")
        else:
            st.write("Formula metadata unavailable.")

        if short_interpretation:
            st.write(short_interpretation)


def metric_card(label: str, value: object, help_text: str | None = None, *, border: bool = False) -> None:
    """Render a formatted metric card with optional border support."""

    metric_kwargs = {
        "label": label,
        "value": _format_metric_value(value),
        "help": help_text,
    }
    if _supports_metric_border():
        st.metric(border=border, width="stretch", **metric_kwargs)
        return
    if border:
        with st.container(border=True):
            st.metric(**metric_kwargs)
        return
    st.metric(**metric_kwargs)
