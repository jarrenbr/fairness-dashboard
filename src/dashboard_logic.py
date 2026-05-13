"""Pure helper logic for the Streamlit fairness dashboard."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

try:
    from .match import (
        MATCH_METRICS,
        METRIC_SPECS,
        match_test_score_from_reference_probs,
        preferred_match_method,
        simulation_match_test_batched,
    )
    from .metrics import METRIC_INFO, SINGLE_CM_METRICS, TWO_CM_FAIRNESS_METRICS
except ImportError:  # pragma: no cover - supports `streamlit run src/app.py`
    from match import (
        MATCH_METRICS,
        METRIC_SPECS,
        match_test_score_from_reference_probs,
        preferred_match_method,
        simulation_match_test_batched,
    )
    from metrics import METRIC_INFO, SINGLE_CM_METRICS, TWO_CM_FAIRNESS_METRICS


class MatchPayload(TypedDict):
    observed_values: tuple[float, float, float, float]
    reference_probs_values: tuple[float, float, float, float]
    metric_name: str
    match_mode: str
    n: int
    num_samples: int
    alternative: str
    seed: int


@dataclass(frozen=True)
class IntraGroupMetricSection:
    title: str
    metric_groups: tuple[tuple[str, ...], ...]
    yaxis_range: tuple[float, float] | None = None
    show_zero_line: bool = False
    horizontal_stack: bool = False


MATCH_METRIC_SET = frozenset(MATCH_METRICS)
MATCH_SUMMARY_LABELS: dict[str, str] = {
    "mode": "MATCH Mode",
    "analytic_method": "Analytic Method",
    "analytic_cdf": "Analytic CDF",
    "analytic_lower_tail": "Analytic Lower Tail",
    "analytic_upper_tail": "Analytic Upper Tail",
    "mean": "Reference Mean",
    "std": "Reference Standard Deviation",
    "median": "Reference Median",
    "2.5%": "2.5th Percentile",
    "5%": "5th Percentile",
    "95%": "95th Percentile",
    "97.5%": "97.5th Percentile",
    "undefined_rate": "Undefined Rate",
    "n": "Observed Sample Size",
    "num_samples": "Number of Samples",
}
MATCH_ALTERNATIVE_DESCRIPTIONS: dict[str, str] = {
    "two-sided": "Tests whether the observed score is unusual in the reference distribution.",
    "less": "Tests whether the observed score is unusually low; p-value is the lower-tail probability.",
    "greater": "Tests whether the observed score is unusually high; p-value is the upper-tail probability.",
}
DEFAULT_MATCH_OBSERVED_VALUES = (12.0, 4.0, 3.0, 21.0)
DEFAULT_MATCH_REFERENCE_CM = np.array([[45.0, 10.0], [10.0, 35.0]], dtype=np.float64)
DEFAULT_MATCH_NUM_SAMPLES = 10_000
DEFAULT_MATCH_SEED = 12_345
MATCH_EXACT_RUNTIME_SAFE_METRICS = frozenset(
    {
        "accuracy",
        "prevalence",
        "predicted_positive_rate",
        "inaccuracy",
        "negative_prevalence",
        "predicted_negative_rate",
        "marginal_benefit",
    }
)
MATCH_DEFAULT_SESSION_STATE: dict[str, object] = {
    "match_metric_selector": "accuracy",
    "match_mode_selector": "Approximate",
    "match_alternative_selector": "two-sided",
}
INTRA_GROUP_METRIC_SECTIONS: tuple[IntraGroupMetricSection, ...] = (
    IntraGroupMetricSection(
        title="Rate Metrics",
        metric_groups=(
            (
                "accuracy",
                "predicted_positive_rate",
                "true_positive_rate",
                "false_positive_rate",
                "positive_predictive_value",
            ),
        ),
        yaxis_range=(0.0, 1.0),
    ),
    IntraGroupMetricSection(
        title="Signed Metrics",
        metric_groups=(
            ("marginal_benefit",),
            ("matthews_correlation_coefficient",),
        ),
        show_zero_line=True,
        horizontal_stack=True,
    ),
)


def _is_integer_like(value: float, *, atol: float = 1e-9) -> bool:
    return bool(np.isfinite(value) and np.isclose(value, np.round(value), atol=atol))


def _display_number(value: object, *, none_text: str, nonfinite_text: str) -> str:
    if value is None:
        return none_text
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(value_float):
        return nonfinite_text
    return f"{value_float:.6g}"


def metric_display_name(metric_name: str) -> str:
    """Return a metric display name from metadata when available."""

    return str(METRIC_INFO.get(metric_name, {}).get("display_name", metric_name))


def _metric_value(metric_name: str, cm: np.ndarray) -> float:
    return float(np.asarray(SINGLE_CM_METRICS[metric_name](cm), dtype=np.float64))


def _format_table_number(value: object) -> str:
    return _display_number(value, none_text="n/a", nonfinite_text="Undefined")


def _format_summary_value(value: object) -> str:
    return _display_number(value, none_text="nan", nonfinite_text="nan")


def _all_fairness_metrics_table(cm_i: np.ndarray, cm_j: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric_name, metric_func in TWO_CM_FAIRNESS_METRICS.items():
        value = float(metric_func(cm_i, cm_j))
        info = METRIC_INFO.get(metric_name, {})
        rows.append(
            {
                "Metric": metric_display_name(metric_name),
                "Value": _format_table_number(value),
                "No-Bias Value": _format_table_number(info.get("no_bias_value")),
                "Intra-Group Metric g": str(info.get("g_latex") or "n/a"),
                "Fairness Metric M": str(info.get("M_latex") or "n/a"),
            }
        )
    return pd.DataFrame(rows)


def _intra_group_metric_values(
    cm_i: np.ndarray,
    cm_j: np.ndarray,
    metric_names: tuple[str, ...],
) -> tuple[dict[str, float], dict[str, float]]:
    group_i_values = {metric_display_name(name): _metric_value(name, cm_i) for name in metric_names}
    group_j_values = {metric_display_name(name): _metric_value(name, cm_j) for name in metric_names}
    return group_i_values, group_j_values


def _intra_group_chart_title(section: IntraGroupMetricSection, metric_names: tuple[str, ...]) -> str:
    if len(metric_names) == 1:
        return metric_display_name(metric_names[0])
    return section.title


def _match_metric_names() -> list[str]:
    return [name for name in SINGLE_CM_METRICS if name in MATCH_METRIC_SET]


def _cm_values(cm: np.ndarray) -> tuple[float, float, float, float]:
    matrix = np.asarray(cm, dtype=np.float64)
    if matrix.shape != (2, 2):
        raise ValueError(f"Confusion matrix must have shape (2, 2); got {matrix.shape}.")
    return (
        float(matrix[0, 0]),
        float(matrix[0, 1]),
        float(matrix[1, 0]),
        float(matrix[1, 1]),
    )


def _reference_probabilities_from_cm(reference_cm: np.ndarray) -> np.ndarray:
    matrix = np.asarray(reference_cm, dtype=np.float64)
    if matrix.shape != (2, 2):
        raise ValueError(f"Reference CM must have shape (2, 2); got {matrix.shape}.")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0):
        raise ValueError("Reference CM must contain only finite, non-negative values.")
    total = float(matrix.sum())
    if total <= 0.0:
        raise ValueError("Reference CM must contain at least one positive count.")
    return matrix / total


def _match_sample_size_from_observed_cm(observed_cm: np.ndarray) -> int:
    matrix = np.asarray(observed_cm, dtype=np.float64)
    if matrix.shape != (2, 2):
        raise ValueError(f"Observed subgroup CM must have shape (2, 2); got {matrix.shape}.")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0):
        raise ValueError("Observed subgroup CM must contain only finite, non-negative values.")
    if not np.allclose(matrix, np.round(matrix), atol=1e-9):
        raise ValueError("Observed subgroup CM must contain non-negative integer counts for MATCH.")
    total = float(matrix.sum())
    if total <= 0.0:
        raise ValueError("Observed subgroup CM must contain at least one positive count.")
    if not _is_integer_like(total):
        raise ValueError("Observed subgroup CM must sum to an integer sample size for MATCH.")
    return int(round(total))


def _build_match_payload(
    observed_cm: np.ndarray,
    reference_probs: np.ndarray,
    *,
    metric_name: str,
    match_mode: str,
    sample_n: int,
    num_samples: int,
    alternative: str,
    seed: int,
) -> MatchPayload:
    reference = np.asarray(reference_probs, dtype=np.float64)
    if reference.shape != (2, 2):
        raise ValueError(f"Reference probabilities must have shape (2, 2); got {reference.shape}.")
    return {
        "observed_values": _cm_values(observed_cm),
        "reference_probs_values": _cm_values(reference),
        "metric_name": metric_name,
        "match_mode": match_mode,
        "n": int(sample_n),
        "num_samples": int(num_samples),
        "alternative": alternative,
        "seed": int(seed),
    }


def _default_match_payload() -> MatchPayload:
    reference_probs = _reference_probabilities_from_cm(DEFAULT_MATCH_REFERENCE_CM)
    return _build_match_payload(
        np.array(
            [
                [DEFAULT_MATCH_OBSERVED_VALUES[0], DEFAULT_MATCH_OBSERVED_VALUES[1]],
                [DEFAULT_MATCH_OBSERVED_VALUES[2], DEFAULT_MATCH_OBSERVED_VALUES[3]],
            ],
            dtype=np.float64,
        ),
        reference_probs,
        metric_name="accuracy",
        match_mode="Approximate",
        sample_n=40,
        num_samples=DEFAULT_MATCH_NUM_SAMPLES,
        alternative="two-sided",
        seed=DEFAULT_MATCH_SEED,
    )


def _resolve_match_method(metric_name: str, match_mode: str) -> str:
    return preferred_match_method(metric_name, "exact" if match_mode == "Exact" else "approximate")


def _match_method_description(metric_name: str, match_mode: str) -> str:
    method = _resolve_match_method(metric_name, match_mode)
    display_name = metric_display_name(metric_name)
    if method == "peizer_pratt":
        return f"`{match_mode}` uses the Peizer-Pratt approximation for {display_name}."
    if method == "normal":
        return f"`{match_mode}` uses the normal approximation for {display_name}."
    if method == "beta":
        return (
            f"`{match_mode}` uses the beta approximation for {display_name}, aligned to the "
            "AISTATS beta-approximation derivation for joint-ratio metrics."
        )
    return f"`{match_mode}` uses the exact MATCH distribution for {display_name}."


def _match_alternative_description(alternative: str) -> str:
    return MATCH_ALTERNATIVE_DESCRIPTIONS.get(
        alternative,
        "Tests whether the observed score is unusual in the reference distribution.",
    )


def _match_interpretation(analytic_result: dict[str, object]) -> str:
    status = str(analytic_result.get("status", "ok"))
    if status == "observed_undefined":
        return (
            "MATCH could not produce a valid p-value because the observed metric is undefined "
            "for the current confusion matrix."
        )

    alternative = str(analytic_result.get("alternative", "two-sided"))
    p_value = float(analytic_result.get("p_value", float("nan")))
    if np.isnan(p_value):
        return "MATCH could not produce a valid p-value for the selected inputs."
    if alternative == "less":
        if p_value < 0.05:
            return "The observed score is unusually low relative to the reference distribution."
        return (
            "There is not strong evidence that the observed score is unusually low relative "
            "to the reference distribution."
        )
    if alternative == "greater":
        if p_value < 0.05:
            return "The observed score is unusually high relative to the reference distribution."
        return (
            "There is not strong evidence that the observed score is unusually high relative "
            "to the reference distribution."
        )
    if p_value < 0.05:
        return "The observed score is unusual under the reference distribution and may warrant further investigation."
    return "The observed score is plausibly consistent with the reference distribution."


def _match_summary_table(match_result: dict[str, object]) -> pd.DataFrame:
    analytic_result = dict(match_result["analytic"])
    simulation_result = dict(match_result["simulation"])
    rows = [
        ("mode", match_result["match_mode"]),
        ("analytic_method", match_result["method"]),
        ("analytic_cdf", analytic_result["cdf"]),
        ("analytic_lower_tail", analytic_result["p_lower"]),
        ("analytic_upper_tail", analytic_result["p_upper"]),
        ("mean", simulation_result["reference_mean"]),
        ("std", simulation_result["reference_std"]),
        ("median", simulation_result["reference_median"]),
        ("2.5%", simulation_result["q025"]),
        ("5%", simulation_result["q05"]),
        ("95%", simulation_result["q95"]),
        ("97.5%", simulation_result["q975"]),
        ("undefined_rate", simulation_result["undefined_rate"]),
        ("n", simulation_result["n"]),
        ("num_samples", simulation_result["total_sample_count"]),
    ]
    return pd.DataFrame(
        {
            "Statistic": pd.Series([MATCH_SUMMARY_LABELS.get(name, name) for name, _ in rows], dtype="string"),
            "Value": pd.Series([_format_summary_value(value) for _, value in rows], dtype="string"),
        }
    )


def _simulation_batch_size(num_samples: int) -> int:
    return max(500, min(5000, max(1, num_samples // 20)))


def _should_warn_about_exact_runtime(metric_name: str, match_mode: str, sample_n: int | None) -> bool:
    return bool(
        match_mode == "Exact"
        and sample_n is not None
        and sample_n > 500
        and metric_name not in MATCH_EXACT_RUNTIME_SAFE_METRICS
    )


def _analytic_match_result(
    observed_cm: np.ndarray,
    reference_probs: np.ndarray,
    metric_name: str,
    sample_n: int,
    alternative: str,
    match_mode: str,
) -> dict[str, object]:
    method = _resolve_match_method(metric_name, match_mode)
    observed_value = _metric_value(metric_name, observed_cm)
    observed_total = int(np.asarray(observed_cm, dtype=np.float64).sum())

    if not np.isfinite(observed_value):
        return {
            "metric": metric_name,
            "method": method,
            "alternative": alternative,
            "observed_score": float("nan"),
            "n_obs": sample_n,
            "cdf": float("nan"),
            "p_lower": float("nan"),
            "p_upper": float("nan"),
            "p_value": float("nan"),
            "reference_probability": float("nan"),
            "details": {},
            "status": "observed_undefined",
        }

    observed_denominator: int | None = None
    metric_spec = METRIC_SPECS[metric_name]
    beta_count = "observed"
    if metric_spec.family == "jrm":
        observed_denominator = int(sum(float(observed_cm[cell]) for cell in metric_spec.denominator_cells))
        if sample_n != observed_total:
            beta_count = "expected"

    result = match_test_score_from_reference_probs(
        observed_score=observed_value,
        n_obs=sample_n,
        reference_probs=reference_probs,
        metric=metric_name,
        method=method,
        alternative=alternative,
        observed_denominator=observed_denominator,
        beta_count=beta_count,
    )
    result_dict = result.asdict()
    result_dict["status"] = "ok"
    return result_dict


def _execute_match_payload(
    payload: MatchPayload,
    *,
    should_cancel=None,
    progress_callback=None,
) -> dict[str, object]:
    observed_values = payload["observed_values"]
    reference_probs_values = payload["reference_probs_values"]

    observed_cm = np.array(
        [[observed_values[0], observed_values[1]], [observed_values[2], observed_values[3]]],
        dtype=np.float64,
    )
    reference_probs = np.array(
        [
            [reference_probs_values[0], reference_probs_values[1]],
            [reference_probs_values[2], reference_probs_values[3]],
        ],
        dtype=np.float64,
    )
    metric_name = str(payload["metric_name"])
    alternative = str(payload["alternative"])
    match_mode = str(payload["match_mode"])
    sample_n = int(payload["n"])
    num_samples = int(payload["num_samples"])
    seed = int(payload["seed"])

    analytic_result = _analytic_match_result(
        observed_cm,
        reference_probs,
        metric_name,
        sample_n,
        alternative,
        match_mode,
    )
    if should_cancel is not None and should_cancel():
        return {
            "status": "cancelled",
            "result": None,
        }

    simulation_result = simulation_match_test_batched(
        observed_cm=observed_cm,
        reference_probs=reference_probs,
        metric_func=SINGLE_CM_METRICS[metric_name],
        n=sample_n,
        num_samples=num_samples,
        alternative=alternative,
        seed=seed,
        batch_size=_simulation_batch_size(num_samples),
        should_cancel=should_cancel,
        progress_callback=progress_callback,
    )
    if str(simulation_result.get("status")) == "cancelled":
        return {
            "status": "cancelled",
            "result": None,
        }

    return {
        "status": "completed",
        "result": {
            "analytic": analytic_result,
            "simulation": simulation_result,
            "match_mode": match_mode,
            "method": _resolve_match_method(metric_name, match_mode),
            "metric_name": metric_name,
        },
    }
