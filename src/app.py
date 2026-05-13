"""Streamlit app for the fairness dashboard."""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import streamlit as st

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

try:
    from .dashboard_logic import (
        INTRA_GROUP_METRIC_SECTIONS,
        MATCH_DEFAULT_SESSION_STATE,
        MatchPayload,
        _all_fairness_metrics_table,
        _analytic_match_result,
        _build_match_payload,
        _default_match_payload,
        _execute_match_payload,
        _intra_group_chart_title,
        _intra_group_metric_values,
        _match_alternative_description,
        _match_interpretation,
        _match_metric_names,
        _match_method_description,
        _match_sample_size_from_observed_cm,
        _match_summary_table,
        _reference_probabilities_from_cm,
        _resolve_match_method,
        _should_warn_about_exact_runtime,
        metric_display_name,
    )
    from .metrics import METRIC_INFO, TWO_CM_FAIRNESS_METRICS
    from .plotting import fairness_bias_gauge, intra_group_bar_chart, metric_distribution_histogram, metric_ecdf
    from .ui_components import cm_input, metric_card, metric_selectbox, render_metric_formula_card, show_cm_heatmap
except ImportError:  # pragma: no cover - supports `streamlit run src/app.py`
    from dashboard_logic import (
        INTRA_GROUP_METRIC_SECTIONS,
        MATCH_DEFAULT_SESSION_STATE,
        MatchPayload,
        _all_fairness_metrics_table,
        _analytic_match_result,
        _build_match_payload,
        _default_match_payload,
        _execute_match_payload,
        _intra_group_chart_title,
        _intra_group_metric_values,
        _match_alternative_description,
        _match_interpretation,
        _match_metric_names,
        _match_method_description,
        _match_sample_size_from_observed_cm,
        _match_summary_table,
        _reference_probabilities_from_cm,
        _resolve_match_method,
        _should_warn_about_exact_runtime,
        metric_display_name,
    )
    from metrics import METRIC_INFO, TWO_CM_FAIRNESS_METRICS
    from plotting import fairness_bias_gauge, intra_group_bar_chart, metric_distribution_histogram, metric_ecdf
    from ui_components import cm_input, metric_card, metric_selectbox, render_metric_formula_card, show_cm_heatmap


FAIRNESS_TABLE_WIDTHS = [1.8, 0.8, 0.95, 2.1, 1.6]
FAIRNESS_TABLE_HEADER_LABELS = [
    "Metric",
    "Value",
    "No-Bias Value",
    "Intra-Group Metric $g$",
    "Fairness Metric $M$",
]
MATCH_EXPLANATION_TEXT = """The MATCH Test is from my [AISTATS work](https://proceedings.mlr.press/v258/briscoe25a.html)
and it evaluates whether an observed group's metric estimate is probabilistically
consistent with a reference distribution. It is useful when small group sizes
make confusion-matrix metrics jagged, unstable, or undefined."""
FAIRNESS_EXPLANATION_TEXT = """
Compares two groups using confusion-matrix-based fairness metrics.
Metrics such as my [Objective Fairness Index](https://dl.acm.org/doi/pdf/10.1145/3627673.3679925),
Disparate Impact, and Accuracy Difference compare Group i against Group j.
"""
MATCH_RUN_PROGRESS_CAPTION = (
    "Use `Refresh status` to poll progress or `Cancel MATCH run` to stop the current calculation."
)
MATCH_SIMULATION_CAPTION = (
    "Simulations are only done for visualization purposes. The MATCH Test does not use simulations."
)
INTRA_GROUP_NOTE = (
    "Metrics are separated by natural scale. Rate metrics use [0, 1], while signed metrics use [-1, 1]."
)


@dataclass(frozen=True)
class MatchFormState:
    observed_cm: np.ndarray
    metric_name: str
    match_mode: str
    alternative: str
    reference_probs: np.ndarray | None
    observed_error: str | None
    reference_error: str | None
    sample_n: int | None
    num_samples: int
    seed: int


def _match_job_registry() -> dict[str, dict[str, object]]:
    registry = st.session_state.get("_match_job_registry")
    if registry is None:
        registry = {}
        st.session_state["_match_job_registry"] = registry
    return registry


def _render_latex_formula_cell(col, formula: str) -> None:
    if formula and formula.lower() != "n/a":
        col.markdown(rf"${formula}$")
    else:
        col.markdown("n/a")


def render_fairness_metrics_table(df: pd.DataFrame) -> None:
    header_cols = st.columns(FAIRNESS_TABLE_WIDTHS)
    for col, label in zip(header_cols, FAIRNESS_TABLE_HEADER_LABELS):
        col.markdown(f"**{label}**")

    st.divider()

    for _, row in df.iterrows():
        cols = st.columns(FAIRNESS_TABLE_WIDTHS)
        cols[0].markdown(str(row["Metric"]))
        cols[1].markdown(str(row["Value"]))
        cols[2].markdown(str(row["No-Bias Value"]))
        _render_latex_formula_cell(cols[3], str(row["Intra-Group Metric g"]).strip())
        _render_latex_formula_cell(cols[4], str(row["Fairness Metric M"]).strip())


def _initialize_match_lab_defaults() -> None:
    for key, value in MATCH_DEFAULT_SESSION_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _maybe_start_default_match_job(job_running: bool) -> bool:
    if bool(st.session_state.get("_match_default_started")):
        return False
    if job_running or st.session_state.get("match_result") is not None:
        st.session_state["_match_default_started"] = True
        return False

    st.session_state["_match_default_started"] = True
    _start_match_job(_default_match_payload())
    return True


def _match_job_key() -> str:
    key = st.session_state.get("_match_job_key")
    if not key:
        key = f"match-job-{uuid4().hex}"
        st.session_state["_match_job_key"] = key
    return key


def _current_match_job() -> dict[str, object] | None:
    return _match_job_registry().get(_match_job_key())


def _match_worker(job: dict[str, object], payload: MatchPayload) -> None:
    cancel_event = job["cancel_event"]

    def should_cancel() -> bool:
        return bool(cancel_event.is_set())

    def update_progress(done: int, total: int) -> None:
        job["progress"] = done / total if total else 1.0
        job["message"] = f"Simulating reference distribution: {done:,} / {total:,} draws."

    try:
        job["message"] = "Computing analytic MATCH result."
        execution = _execute_match_payload(
            payload,
            should_cancel=should_cancel,
            progress_callback=update_progress,
        )
        if str(execution["status"]) == "cancelled":
            job["status"] = "cancelled"
            job["message"] = "MATCH run cancelled."
            return

        job["result"] = execution["result"]
        job["progress"] = 1.0
        job["message"] = "MATCH run complete."
        job["status"] = "completed"
    except Exception as exc:  # pragma: no cover - exercised through app flow.
        job["error"] = str(exc)
        job["status"] = "error"
        job["message"] = "MATCH run failed."


def _start_match_job(payload: MatchPayload) -> None:
    registry = _match_job_registry()
    job_key = _match_job_key()
    existing = registry.get(job_key)
    if existing and str(existing.get("status")) == "running":
        return

    cancel_event = threading.Event()
    job: dict[str, object] = {
        "status": "running",
        "progress": 0.0,
        "message": "Preparing MATCH run.",
        "error": None,
        "result": None,
        "cancel_event": cancel_event,
    }
    registry[job_key] = job
    st.session_state["match_result"] = None
    st.session_state["match_error"] = None
    st.session_state["match_notice"] = None

    thread = threading.Thread(target=_match_worker, args=(job, payload), daemon=True, name="match-worker")
    job["thread"] = thread
    thread.start()


def _cancel_match_job() -> None:
    job = _current_match_job()
    if not job or str(job.get("status")) != "running":
        return
    cancel_event = job.get("cancel_event")
    if cancel_event is not None and hasattr(cancel_event, "set"):
        cancel_event.set()
    job["message"] = "Cancelling MATCH run."


def _sync_match_job_state() -> dict[str, object] | None:
    job = _current_match_job()
    if not job:
        return None

    status = str(job.get("status"))
    if status == "completed":
        st.session_state["match_result"] = job.get("result")
        st.session_state["match_error"] = None
        st.session_state["match_notice"] = None
    elif status == "error":
        st.session_state["match_error"] = str(job.get("error") or "MATCH run failed.")
        st.session_state["match_notice"] = None
    elif status == "cancelled":
        st.session_state["match_notice"] = "MATCH run cancelled."
    return job


def _render_fairness_metric_details(
    selected_metric: str,
    info: dict[str, object],
    no_bias_value: object,
    selected_value: float,
) -> None:
    detail_left, detail_right = st.columns([1.2, 1.0])
    with detail_left:
        st.subheader("Formula and interpretation")
        st.markdown(f"**{metric_display_name(selected_metric)}**")
        st.write(f"Formula: `{info.get('formula_text', 'n/a')}`")
        st.write(f"No-bias point: `{no_bias_value}`")
        if "note" in info:
            st.write(f"Note: {info['note']}")
        if "bounded" in info:
            st.write(f"Bounds: {info['bounded']}")
        interpretation = str(info.get("interpretation", "")).strip()
        if interpretation:
            st.write(f"Interpretation: {interpretation}")
    with detail_right:
        st.subheader("Bias gauge")
        st.plotly_chart(
            fairness_bias_gauge(
                selected_value,
                float(no_bias_value) if isinstance(no_bias_value, (int, float)) else 0.0,
                f"{metric_display_name(selected_metric)} Indicator",
            ),
            width="stretch",
        )


def _render_intra_group_metric_sections(cm_i: np.ndarray, cm_j: np.ndarray) -> None:
    st.subheader("Intra-Group Metric Comparison")
    st.caption(INTRA_GROUP_NOTE)

    for section in INTRA_GROUP_METRIC_SECTIONS:
        if section.horizontal_stack:
            cols = st.columns(len(section.metric_groups))
            for col, metric_names in zip(cols, section.metric_groups):
                group_i_values, group_j_values = _intra_group_metric_values(cm_i, cm_j, metric_names)
                with col:
                    st.plotly_chart(
                        intra_group_bar_chart(
                            group_i_values,
                            group_j_values,
                            title=_intra_group_chart_title(section, metric_names),
                            yaxis_range=section.yaxis_range,
                            show_zero_line=section.show_zero_line,
                        ),
                        width="stretch",
                    )
            continue

        metric_names = section.metric_groups[0]
        group_i_values, group_j_values = _intra_group_metric_values(cm_i, cm_j, metric_names)
        st.plotly_chart(
            intra_group_bar_chart(
                group_i_values,
                group_j_values,
                title=_intra_group_chart_title(section, metric_names),
                yaxis_range=section.yaxis_range,
                show_zero_line=section.show_zero_line,
            ),
            width="stretch",
        )


def render_fairness_metric_explorer() -> None:
    st.header("Fairness Metric Explorer")
    with st.expander("What this explains", expanded=True):
        st.markdown(FAIRNESS_EXPLANATION_TEXT)

    input_left, input_right = st.columns(2)
    with input_left:
        cm_i = cm_input("Group i", (40, 10, 5, 45))
    with input_right:
        cm_j = cm_input("Group j", (35, 15, 10, 40))

    heat_left, heat_right = st.columns(2)
    with heat_left:
        show_cm_heatmap(cm_i, "Group i Confusion Matrix")
    with heat_right:
        show_cm_heatmap(cm_j, "Group j Confusion Matrix")

    selected_metric = metric_selectbox("fairness_metric_selector", include_fairness=True)
    selected_value = float(TWO_CM_FAIRNESS_METRICS[selected_metric](cm_i, cm_j))
    info = METRIC_INFO.get(selected_metric, {})
    no_bias_value = info.get("no_bias_value")

    if not np.isfinite(selected_value):
        st.warning("The selected fairness metric is undefined for the current inputs.")

    _render_fairness_metric_details(selected_metric, info, no_bias_value, selected_value)

    st.subheader("All Fairness Metrics")
    render_fairness_metrics_table(_all_fairness_metrics_table(cm_i, cm_j))
    _render_intra_group_metric_sections(cm_i, cm_j)


def _reference_probs_or_error(reference_cm: np.ndarray) -> tuple[np.ndarray | None, str | None]:
    try:
        return _reference_probabilities_from_cm(reference_cm), None
    except ValueError as exc:
        return None, str(exc)


def _observed_sample_size_or_error(observed_cm: np.ndarray) -> tuple[int | None, str | None]:
    try:
        return _match_sample_size_from_observed_cm(observed_cm), None
    except ValueError as exc:
        return None, str(exc)


def _render_match_run_controls(
    *,
    metric_name: str,
    match_mode: str,
    sample_n: int | None,
    job_running: bool,
) -> tuple[int, int, bool, bool]:
    st.subheader("Run Settings")
    st.caption(MATCH_SIMULATION_CAPTION)
    num_samples = int(st.number_input("Number of simulations", min_value=1000, max_value=200000, value=10000, step=1000))
    seed = int(st.number_input("Random seed", value=12345, step=1))

    if _should_warn_about_exact_runtime(metric_name, match_mode, sample_n):
        st.warning("Exact JRM MATCH can be slow for large n. Use cancel if needed.")

    action_cols = st.columns([1.2, 1.0, 1.0])
    with action_cols[0]:
        run_clicked = st.button("Run MATCH test", type="primary", disabled=job_running)
    with action_cols[1]:
        cancel_clicked = st.button("Cancel MATCH run", disabled=not job_running)
    with action_cols[2]:
        st.button("Refresh status", disabled=not job_running)

    return num_samples, seed, run_clicked, cancel_clicked


def _render_match_setup(job_running: bool) -> tuple[MatchFormState, bool, bool]:
    left_col, right_col = st.columns([1.05, 0.95], gap="large")

    with left_col:
        with st.container(border=True):
            observed_cm = cm_input(
                "Observed Group's Confusion Matrix",
                (12, 4, 3, 21),
                key_prefix="observed_subgroup_cm",
                allow_decimal=False,
            )
            st.caption("Observed values must be non-negative integers.")
            sample_n, observed_error = _observed_sample_size_or_error(observed_cm)
            if observed_error:
                st.error(observed_error)

        with st.container(border=True):
            st.subheader("Selected metric")
            metric_name = metric_selectbox(
                "match_metric_selector",
                include_fairness=False,
                metric_names=_match_metric_names(),
            )
            match_mode = st.radio(
                "MATCH mode",
                ["Approximate", "Exact"],
                index=0,
                horizontal=True,
                key="match_mode_selector",
            )
            st.caption(_match_method_description(metric_name, match_mode))
            render_metric_formula_card(metric_name)
            alternative = st.selectbox(
                "Alternative hypothesis",
                ["two-sided", "less", "greater"],
                index=0,
                key="match_alternative_selector",
            )
            st.caption(_match_alternative_description(alternative))

    with right_col:
        with st.container(border=True):
            reference_cm = cm_input(
                "Reference Confusion Matrix",
                (45, 10, 10, 35),
                show_total=False,
                key_prefix="reference_cm",
            )
            st.caption("Reference values allow non-negative decimals. For example, you may enter a probability distribution.")
            reference_probs, reference_error = _reference_probs_or_error(reference_cm)
            if reference_error:
                st.error(reference_error)

        with st.container(border=True):
            num_samples, seed, run_clicked, cancel_clicked = _render_match_run_controls(
                metric_name=metric_name,
                match_mode=match_mode,
                sample_n=sample_n,
                job_running=job_running,
            )

    form_state = MatchFormState(
        observed_cm=observed_cm,
        metric_name=metric_name,
        match_mode=match_mode,
        alternative=alternative,
        reference_probs=reference_probs,
        observed_error=observed_error,
        reference_error=reference_error,
        sample_n=sample_n,
        num_samples=num_samples,
        seed=seed,
    )
    return form_state, run_clicked, cancel_clicked


def _start_match_job_from_form(form_state: MatchFormState) -> None:
    if form_state.observed_error:
        st.session_state["match_result"] = None
        st.session_state["match_error"] = form_state.observed_error
        return
    if form_state.reference_error or form_state.reference_probs is None:
        st.session_state["match_result"] = None
        st.session_state["match_error"] = form_state.reference_error or "Reference input is invalid."
        return
    assert form_state.sample_n is not None

    payload = _build_match_payload(
        form_state.observed_cm,
        form_state.reference_probs,
        metric_name=form_state.metric_name,
        match_mode=form_state.match_mode,
        sample_n=form_state.sample_n,
        num_samples=form_state.num_samples,
        alternative=form_state.alternative,
        seed=form_state.seed,
    )
    _start_match_job(payload)


def _render_match_job_status(job: dict[str, object]) -> None:
    st.info(str(job.get("message", "MATCH run in progress.")))
    st.progress(float(job.get("progress", 0.0)))
    st.caption(MATCH_RUN_PROGRESS_CAPTION)
    time.sleep(0.1)
    st.rerun()


def _render_match_notifications() -> None:
    match_error = st.session_state.get("match_error")
    if match_error:
        st.error(match_error)

    match_notice = st.session_state.get("match_notice")
    if match_notice:
        st.warning(str(match_notice))


def _render_match_result_cards(
    analytic_result: dict[str, object],
    simulation_result: dict[str, object],
    method: str,
) -> None:
    primary_cols = st.columns(4)
    with primary_cols[0]:
        metric_card("Observed score", float(simulation_result["observed_value"]), border=True)
    with primary_cols[1]:
        metric_card("Reference mean", float(simulation_result["reference_mean"]), border=True)
    with primary_cols[2]:
        metric_card("p-value", float(analytic_result["p_value"]), border=True)
    with primary_cols[3]:
        metric_card("Analytic CDF", float(analytic_result["cdf"]), border=True)

    detail_cols = st.columns(4)
    with detail_cols[0]:
        metric_card("Undefined rate", float(simulation_result["undefined_rate"]), border=True)
    with detail_cols[1]:
        metric_card("Finite reference samples", int(simulation_result["finite_sample_count"]), border=True)
    with detail_cols[2]:
        metric_card("Analytic method", method, border=True)
    with detail_cols[3]:
        metric_card("Status", str(analytic_result["status"]), border=True)


def _render_match_result(match_result: dict[str, object]) -> None:
    analytic_result = dict(match_result["analytic"])
    simulation_result = dict(match_result["simulation"])
    undefined_rate = float(simulation_result["undefined_rate"])
    finite_values = np.asarray(simulation_result["finite_reference_values"], dtype=np.float64)
    metric_name = str(match_result["metric_name"])
    metric_label = metric_display_name(metric_name)

    with st.container(border=True):
        st.subheader("MATCH Test Result")
        _render_match_interpretation(analytic_result)

        if int(simulation_result["n"]) < 30:
            st.warning("Small sample warning: metric distributions can be jagged and highly discrete.")
        if undefined_rate > 0.05:
            st.warning("Reference distribution has a nontrivial undefined rate. Interpret p-values carefully.")
        if finite_values.size == 0:
            st.warning("No finite reference metric values were available to plot.")

        distribution_tab, ecdf_tab, summary_tab = st.tabs(["Distribution", "Empirical CDF", "Reference summary"])
        with distribution_tab:
            st.plotly_chart(
                metric_distribution_histogram(
                    finite_values,
                    float(simulation_result["observed_value"]),
                    "MATCH Reference Distribution",
                    metric_display_name=metric_label,
                    p_value=float(analytic_result["p_value"]),
                    method=str(match_result["method"]),
                ),
                width="stretch",
            )
        with ecdf_tab:
            st.plotly_chart(
                metric_ecdf(finite_values, float(simulation_result["observed_value"]), f"{metric_label} Empirical CDF"),
                width="stretch",
            )
        with summary_tab:
            st.dataframe(_match_summary_table(match_result), width="stretch", hide_index=True)
        _render_match_result_cards(analytic_result, simulation_result, str(match_result["method"]))


def _render_match_interpretation(analytic_result: dict[str, object]) -> None:
    message = _match_interpretation(analytic_result)
    status = str(analytic_result.get("status", "ok"))
    p_value = float(analytic_result.get("p_value", float("nan")))
    if status != "ok" or np.isnan(p_value):
        st.error(message)
        return
    if p_value < 0.05:
        st.warning(message)
        return
    st.success(message)


def render_match_test_lab() -> None:
    st.header("MATCH Test Lab")
    with st.expander("What the MATCH Test does", expanded=True):
        st.markdown(MATCH_EXPLANATION_TEXT)

    job = _sync_match_job_state()
    job_running = bool(job and str(job.get("status")) == "running")
    _initialize_match_lab_defaults()
    if _maybe_start_default_match_job(job_running):
        st.rerun()

    form_state, run_clicked, cancel_clicked = _render_match_setup(job_running)

    if cancel_clicked:
        _cancel_match_job()

    if run_clicked:
        _start_match_job_from_form(form_state)
        if st.session_state.get("match_error") is None:
            st.rerun()

    if job_running and job:
        _render_match_job_status(job)

    _render_match_notifications()

    match_result = st.session_state.get("match_result")
    if match_result:
        _render_match_result(match_result)


def main() -> None:
    st.set_page_config(page_title="Fairness Dashboard", page_icon="⚖️", layout="wide")
    st.title("Fairness Dashboard")
    st.caption("Objective fairness, group comparison, and metric reliability for binary classification.")

    tab_fairness, tab_match = st.tabs(["1. Fairness Metric Explorer", "2. MATCH Test Lab"])
    with tab_fairness:
        render_fairness_metric_explorer()
    with tab_match:
        render_match_test_lab()


if __name__ == "__main__":
    main()
