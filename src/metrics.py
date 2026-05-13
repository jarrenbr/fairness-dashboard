"""Confusion-matrix metrics for binary classification and fairness comparison.

The confusion matrix convention is fixed throughout the project:

    [[TP, FN],
     [FP, TN]]

Undefined metrics return ``np.nan`` rather than silently coercing to zero.
Single-matrix metric functions support either one CM with shape ``(2, 2)`` or a
stack with shape ``(..., 2, 2)``.
"""

from __future__ import annotations

from math import sqrt
from typing import Any, Callable

import numpy as np

TP = (0, 0)
FN = (0, 1)
FP = (1, 0)
TN = (1, 1)

SingleMetricFunc = Callable[[np.ndarray], np.ndarray]
FairnessMetricFunc = Callable[[np.ndarray, np.ndarray], float]


def create_cm(tp: Any, fn: Any, fp: Any, tn: Any) -> np.ndarray:
    """Create one CM or a broadcasted stack of CMs."""

    tp_arr, fn_arr, fp_arr, tn_arr = np.broadcast_arrays(tp, fn, fp, tn)
    if tp_arr.shape == ():
        return np.array([[tp_arr.item(), fn_arr.item()], [fp_arr.item(), tn_arr.item()]], dtype=np.float64)
    return np.stack(
        [
            np.stack([tp_arr, fn_arr], axis=-1),
            np.stack([fp_arr, tn_arr], axis=-1),
        ],
        axis=-2,
    ).astype(np.float64)


def is_binary_cm_proper(cm: np.ndarray) -> bool:
    """Return True for one CM or a stack of CMs in the required layout."""

    return isinstance(cm, np.ndarray) and (2 <= cm.ndim <= 3) and cm.shape[-2:] == (2, 2)


def _require_cm(cm: np.ndarray) -> np.ndarray:
    arr = np.asarray(cm, dtype=np.float64)
    if not is_binary_cm_proper(arr):
        raise ValueError("cm must be a np.ndarray with shape (2, 2) or (..., 2, 2)")
    if not np.all(np.isfinite(arr)):
        raise ValueError("cm must contain only finite values")
    if np.any(arr < 0):
        raise ValueError("cm must contain non-negative counts")
    return arr


def _require_single_cm(cm: np.ndarray) -> np.ndarray:
    arr = _require_cm(cm)
    if arr.shape != (2, 2):
        raise ValueError(f"Expected one confusion matrix with shape (2, 2), got {arr.shape}")
    return arr


def safe_divide(num: Any, den: Any) -> np.ndarray | float:
    """Divide and return ``np.nan`` when the denominator is zero."""

    num_arr = np.asarray(num, dtype=np.float64)
    den_arr = np.asarray(den, dtype=np.float64)
    out = np.full(np.broadcast_shapes(num_arr.shape, den_arr.shape), np.nan, dtype=np.float64)
    result = np.divide(num_arr, den_arr, out=out, where=(den_arr != 0))
    if result.shape == ():
        return float(result)
    return result


def true_positive_count(cm: np.ndarray) -> float:
    return float(_require_single_cm(cm)[TP])


def false_negative_count(cm: np.ndarray) -> float:
    return float(_require_single_cm(cm)[FN])


def false_positive_count(cm: np.ndarray) -> float:
    return float(_require_single_cm(cm)[FP])


def true_negative_count(cm: np.ndarray) -> float:
    return float(_require_single_cm(cm)[TN])


def actual_positive_count(cm: np.ndarray) -> float:
    matrix = _require_single_cm(cm)
    return float(matrix[TP] + matrix[FN])


def actual_negative_count(cm: np.ndarray) -> float:
    matrix = _require_single_cm(cm)
    return float(matrix[FP] + matrix[TN])


def predicted_positive_count(cm: np.ndarray) -> float:
    matrix = _require_single_cm(cm)
    return float(matrix[TP] + matrix[FP])


def predicted_negative_count(cm: np.ndarray) -> float:
    matrix = _require_single_cm(cm)
    return float(matrix[FN] + matrix[TN])


def incorrect_count(cm: np.ndarray) -> float:
    matrix = _require_single_cm(cm)
    return float(matrix[FN] + matrix[FP])


def correct_count(cm: np.ndarray) -> float:
    matrix = _require_single_cm(cm)
    return float(matrix[TP] + matrix[TN])


def total_count(cm: np.ndarray) -> float:
    return float(_require_single_cm(cm).sum())


def total(cm: np.ndarray) -> np.ndarray:
    """Vectorized total count helper for one CM or a stack."""

    return _require_cm(cm).sum(axis=(-2, -1))


def accuracy(cm: np.ndarray) -> np.ndarray:
    cm_arr = _require_cm(cm)
    return safe_divide(cm_arr[..., *TP] + cm_arr[..., *TN], total(cm_arr))


def prevalence(cm: np.ndarray) -> np.ndarray:
    cm_arr = _require_cm(cm)
    return safe_divide(cm_arr[..., *TP] + cm_arr[..., *FN], total(cm_arr))


def predicted_positive_rate(cm: np.ndarray) -> np.ndarray:
    cm_arr = _require_cm(cm)
    return safe_divide(cm_arr[..., *TP] + cm_arr[..., *FP], total(cm_arr))


def inaccuracy(cm: np.ndarray) -> np.ndarray:
    cm_arr = _require_cm(cm)
    return safe_divide(cm_arr[..., *FP] + cm_arr[..., *FN], total(cm_arr))


def negative_prevalence(cm: np.ndarray) -> np.ndarray:
    cm_arr = _require_cm(cm)
    return safe_divide(cm_arr[..., *TN] + cm_arr[..., *FP], total(cm_arr))


def predicted_negative_rate(cm: np.ndarray) -> np.ndarray:
    cm_arr = _require_cm(cm)
    return safe_divide(cm_arr[..., *TN] + cm_arr[..., *FN], total(cm_arr))


def true_positive_rate(cm: np.ndarray) -> np.ndarray:
    cm_arr = _require_cm(cm)
    return safe_divide(cm_arr[..., *TP], cm_arr[..., *TP] + cm_arr[..., *FN])


def false_negative_rate(cm: np.ndarray) -> np.ndarray:
    cm_arr = _require_cm(cm)
    return safe_divide(cm_arr[..., *FN], cm_arr[..., *FN] + cm_arr[..., *TP])


def false_positive_rate(cm: np.ndarray) -> np.ndarray:
    cm_arr = _require_cm(cm)
    return safe_divide(cm_arr[..., *FP], cm_arr[..., *FP] + cm_arr[..., *TN])


def true_negative_rate(cm: np.ndarray) -> np.ndarray:
    cm_arr = _require_cm(cm)
    return safe_divide(cm_arr[..., *TN], cm_arr[..., *TN] + cm_arr[..., *FP])


def positive_predictive_value(cm: np.ndarray) -> np.ndarray:
    cm_arr = _require_cm(cm)
    return safe_divide(cm_arr[..., *TP], cm_arr[..., *TP] + cm_arr[..., *FP])


def negative_predictive_value(cm: np.ndarray) -> np.ndarray:
    cm_arr = _require_cm(cm)
    return safe_divide(cm_arr[..., *TN], cm_arr[..., *TN] + cm_arr[..., *FN])


def false_discovery_rate(cm: np.ndarray) -> np.ndarray:
    cm_arr = _require_cm(cm)
    return safe_divide(cm_arr[..., *FP], cm_arr[..., *FP] + cm_arr[..., *TP])


def false_omission_rate(cm: np.ndarray) -> np.ndarray:
    cm_arr = _require_cm(cm)
    return safe_divide(cm_arr[..., *FN], cm_arr[..., *FN] + cm_arr[..., *TN])


def true_positive_share_of_correct(cm: np.ndarray) -> np.ndarray:
    cm_arr = _require_cm(cm)
    return safe_divide(cm_arr[..., *TP], cm_arr[..., *TP] + cm_arr[..., *TN])


def true_negative_share_of_correct(cm: np.ndarray) -> np.ndarray:
    cm_arr = _require_cm(cm)
    return safe_divide(cm_arr[..., *TN], cm_arr[..., *TN] + cm_arr[..., *TP])


def false_positive_share_of_errors(cm: np.ndarray) -> np.ndarray:
    cm_arr = _require_cm(cm)
    return safe_divide(cm_arr[..., *FP], cm_arr[..., *FP] + cm_arr[..., *FN])


def false_negative_share_of_errors(cm: np.ndarray) -> np.ndarray:
    cm_arr = _require_cm(cm)
    return safe_divide(cm_arr[..., *FN], cm_arr[..., *FN] + cm_arr[..., *FP])


def f1_score(cm: np.ndarray) -> np.ndarray:
    cm_arr = _require_cm(cm)
    tp = cm_arr[..., *TP]
    fn = cm_arr[..., *FN]
    fp = cm_arr[..., *FP]
    return safe_divide(2.0 * tp, 2.0 * tp + fp + fn)


def matthews_correlation_coefficient(cm: np.ndarray) -> np.ndarray:
    cm_arr = _require_cm(cm)
    tp = cm_arr[..., *TP]
    fn = cm_arr[..., *FN]
    fp = cm_arr[..., *FP]
    tn = cm_arr[..., *TN]
    numerator = tp * tn - fp * fn
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return safe_divide(numerator, denominator)


def marginal_benefit(cm: np.ndarray) -> np.ndarray:
    cm_arr = _require_cm(cm)
    return safe_divide(cm_arr[..., *FP] - cm_arr[..., *FN], total(cm_arr))


error_rate = misclassification_rate = inaccuracy
actual_positive_rate = prevalence
actual_negative_rate = negative_prevalence
recall = sensitivity = hit_rate = true_positive_rate
miss_rate = false_negative_rate
fall_out = false_positive_rate
specificity = true_negative_rate
precision = positive_predictive_value
false_reassurance_rate = false_omission_rate
mcc = matthews_correlation_coefficient


# Metric key constants.  Keep dictionaries keyed by these variables rather than
# hard-coded strings so display names and internal names can be changed in one
# place without hunting through METRIC_INFO.
UNKNOWN = "Unknown"

TRUE_POSITIVE_COUNT = "true_positive_count"
FALSE_NEGATIVE_COUNT = "false_negative_count"
FALSE_POSITIVE_COUNT = "false_positive_count"
TRUE_NEGATIVE_COUNT = "true_negative_count"
ACTUAL_POSITIVE_COUNT = "actual_positive_count"
ACTUAL_NEGATIVE_COUNT = "actual_negative_count"
PREDICTED_POSITIVE_COUNT = "predicted_positive_count"
PREDICTED_NEGATIVE_COUNT = "predicted_negative_count"
INCORRECT_COUNT = "incorrect_count"
CORRECT_COUNT = "correct_count"
TOTAL_COUNT = "total_count"

ACCURACY = "accuracy"
PREVALENCE = "prevalence"
PREDICTED_POSITIVE_RATE = "predicted_positive_rate"
INACCURACY = "inaccuracy"
NEGATIVE_PREVALENCE = "negative_prevalence"
PREDICTED_NEGATIVE_RATE = "predicted_negative_rate"
TRUE_POSITIVE_RATE = "true_positive_rate"
FALSE_NEGATIVE_RATE = "false_negative_rate"
FALSE_POSITIVE_RATE = "false_positive_rate"
TRUE_NEGATIVE_RATE = "true_negative_rate"
POSITIVE_PREDICTIVE_VALUE = "positive_predictive_value"
NEGATIVE_PREDICTIVE_VALUE = "negative_predictive_value"
FALSE_DISCOVERY_RATE = "false_discovery_rate"
FALSE_OMISSION_RATE = "false_omission_rate"
TRUE_POSITIVE_SHARE_OF_CORRECT = "true_positive_share_of_correct"
TRUE_NEGATIVE_SHARE_OF_CORRECT = "true_negative_share_of_correct"
FALSE_POSITIVE_SHARE_OF_ERRORS = "false_positive_share_of_errors"
FALSE_NEGATIVE_SHARE_OF_ERRORS = "false_negative_share_of_errors"
F1_SCORE = "f1_score"
MATTHEWS_CORRELATION_COEFFICIENT = "matthews_correlation_coefficient"
MARGINAL_BENEFIT = "marginal_benefit"

OBJECTIVE_FAIRNESS_INDEX = "objective_fairness_index"
DISPARATE_IMPACT = "disparate_impact"
ACCURACY_DIFFERENCE = "accuracy_difference"
MCC_DIFFERENCE = "mcc_difference"
PREDICTIVE_PARITY_DIFFERENCE = "predictive_parity_difference"
TREATMENT_EQUALITY = "treatment_equality"
DIFFERENCE_IN_CONDITIONAL_ACCEPTANCE = "difference_in_conditional_acceptance"
DIFFERENCE_IN_CONDITIONAL_REJECTION = "difference_in_conditional_rejection"
DIFFERENCE_IN_POSITIVE_PROPORTION_AND_LABELS = "difference_in_positive_proportion_and_labels"


COUNT_METRICS: dict[str, Callable[[np.ndarray], float]] = {
    TRUE_POSITIVE_COUNT: true_positive_count,
    FALSE_NEGATIVE_COUNT: false_negative_count,
    FALSE_POSITIVE_COUNT: false_positive_count,
    TRUE_NEGATIVE_COUNT: true_negative_count,
    ACTUAL_POSITIVE_COUNT: actual_positive_count,
    ACTUAL_NEGATIVE_COUNT: actual_negative_count,
    PREDICTED_POSITIVE_COUNT: predicted_positive_count,
    PREDICTED_NEGATIVE_COUNT: predicted_negative_count,
    INCORRECT_COUNT: incorrect_count,
    CORRECT_COUNT: correct_count,
    TOTAL_COUNT: total_count,
}

BINOMIAL_METRICS: dict[str, SingleMetricFunc] = {
    ACCURACY: accuracy,
    PREVALENCE: prevalence,
    PREDICTED_POSITIVE_RATE: predicted_positive_rate,
    INACCURACY: inaccuracy,
    NEGATIVE_PREVALENCE: negative_prevalence,
    PREDICTED_NEGATIVE_RATE: predicted_negative_rate,
}

JOINT_RATIO_METRICS: dict[str, SingleMetricFunc] = {
    TRUE_POSITIVE_RATE: true_positive_rate,
    FALSE_NEGATIVE_RATE: false_negative_rate,
    FALSE_POSITIVE_RATE: false_positive_rate,
    TRUE_NEGATIVE_RATE: true_negative_rate,
    POSITIVE_PREDICTIVE_VALUE: positive_predictive_value,
    FALSE_DISCOVERY_RATE: false_discovery_rate,
    NEGATIVE_PREDICTIVE_VALUE: negative_predictive_value,
    FALSE_OMISSION_RATE: false_omission_rate,
    TRUE_POSITIVE_SHARE_OF_CORRECT: true_positive_share_of_correct,
    TRUE_NEGATIVE_SHARE_OF_CORRECT: true_negative_share_of_correct,
    FALSE_POSITIVE_SHARE_OF_ERRORS: false_positive_share_of_errors,
    FALSE_NEGATIVE_SHARE_OF_ERRORS: false_negative_share_of_errors,
}

OTHER_SINGLE_CM_METRICS: dict[str, SingleMetricFunc] = {
    F1_SCORE: f1_score,
    MATTHEWS_CORRELATION_COEFFICIENT: matthews_correlation_coefficient,
    MARGINAL_BENEFIT: marginal_benefit,
}

SINGLE_CM_METRICS: dict[str, SingleMetricFunc] = {
    **BINOMIAL_METRICS,
    **JOINT_RATIO_METRICS,
    **OTHER_SINGLE_CM_METRICS,
}

ALL_METRICS = SINGLE_CM_METRICS


def _scalar_metric_value(metric_func: SingleMetricFunc, cm: np.ndarray) -> float:
    value = metric_func(_require_single_cm(cm))
    return float(np.asarray(value, dtype=np.float64))


def accuracy_difference(cm_i: np.ndarray, cm_j: np.ndarray) -> float:
    return _scalar_metric_value(accuracy, cm_i) - _scalar_metric_value(accuracy, cm_j)


def mcc_difference(cm_i: np.ndarray, cm_j: np.ndarray) -> float:
    return _scalar_metric_value(matthews_correlation_coefficient, cm_i) - _scalar_metric_value(
        matthews_correlation_coefficient, cm_j
    )


def disparate_impact(cm_i: np.ndarray, cm_j: np.ndarray) -> float:
    return float(
        safe_divide(
            _scalar_metric_value(predicted_positive_rate, cm_i),
            _scalar_metric_value(predicted_positive_rate, cm_j),
        )
    )


def predictive_parity_difference(cm_i: np.ndarray, cm_j: np.ndarray) -> float:
    return _scalar_metric_value(positive_predictive_value, cm_i) - _scalar_metric_value(
        positive_predictive_value, cm_j
    )


def treatment_equality(cm_i: np.ndarray, cm_j: np.ndarray) -> float:
    ratio_i = float(safe_divide(false_negative_count(cm_i), false_positive_count(cm_i)))
    ratio_j = float(safe_divide(false_negative_count(cm_j), false_positive_count(cm_j)))
    return ratio_i - ratio_j


def difference_in_conditional_acceptance(cm_i: np.ndarray, cm_j: np.ndarray) -> float:
    value_i = float(safe_divide(actual_positive_count(cm_i), predicted_positive_count(cm_i)))
    value_j = float(safe_divide(actual_positive_count(cm_j), predicted_positive_count(cm_j)))
    return value_i - value_j


def difference_in_conditional_rejection(cm_i: np.ndarray, cm_j: np.ndarray) -> float:
    value_j = float(safe_divide(actual_negative_count(cm_j), predicted_negative_count(cm_j)))
    value_i = float(safe_divide(actual_negative_count(cm_i), predicted_negative_count(cm_i)))
    return value_j - value_i


def difference_in_positive_proportion_and_labels(cm_i: np.ndarray, cm_j: np.ndarray) -> float:
    return _scalar_metric_value(predicted_positive_rate, cm_i) - _scalar_metric_value(predicted_positive_rate, cm_j)


def objective_fairness_index(cm_i: np.ndarray, cm_j: np.ndarray) -> float:
    return _scalar_metric_value(marginal_benefit, cm_i) - _scalar_metric_value(marginal_benefit, cm_j)


TWO_CM_FAIRNESS_METRICS: dict[str, FairnessMetricFunc] = {
    OBJECTIVE_FAIRNESS_INDEX: objective_fairness_index,
    DISPARATE_IMPACT: disparate_impact,
    ACCURACY_DIFFERENCE: accuracy_difference,
    MCC_DIFFERENCE: mcc_difference,
    PREDICTIVE_PARITY_DIFFERENCE: predictive_parity_difference,
    TREATMENT_EQUALITY: treatment_equality,
    DIFFERENCE_IN_CONDITIONAL_ACCEPTANCE: difference_in_conditional_acceptance,
    DIFFERENCE_IN_CONDITIONAL_REJECTION: difference_in_conditional_rejection,
    DIFFERENCE_IN_POSITIVE_PROPORTION_AND_LABELS: difference_in_positive_proportion_and_labels,
}


def compute_metrics(
    cm: np.ndarray,
    metrics: dict[str, Callable[[np.ndarray], np.ndarray]] | None = None,
) -> dict[str, np.ndarray]:
    """Compute a dictionary of metric_name -> metric_value."""

    metrics = SINGLE_CM_METRICS if metrics is None else metrics
    return {name: fn(cm) for name, fn in metrics.items()}


# Metadata helpers.  The dissertation defines Universal Domain / Defined
# Everywhere for every non-empty confusion matrix (n > 0).  The implementation
# still returns np.nan for zero-count edge cases.
_BASELINE_UNKNOWN = {
    "objective_testing": UNKNOWN,
    "real_valued": UNKNOWN,
    "directed": UNKNOWN,
    "symmetric": UNKNOWN,
    "bounded": UNKNOWN,
    "defined_everywhere": UNKNOWN,
}

_SCT_UNKNOWN = {
    "direction": UNKNOWN,
    "skew_symmetry": UNKNOWN,
    "consistency": UNKNOWN,
    "participation": UNKNOWN,
    "non_imposition": UNKNOWN,
    "non_dictatorship": UNKNOWN,
    "condorcet_consistent": UNKNOWN,
    "independence_of_irrelevant_alternatives": UNKNOWN,
    "anonymity": UNKNOWN,
}

_INTRA_GROUP_PROPORTION_PROPERTIES = {
    "pi_involution_symmetry": True,
    "participation": True,
    "universal_domain": True,
    "non_dictatorship": True,
}

_JRM_PROPERTIES = {
    "pi_involution_symmetry": True,
    "participation": True,
    "universal_domain": False,
    "non_dictatorship": True,
}

_UNKNOWN_INTRA_PROPERTIES = {
    "pi_involution_symmetry": UNKNOWN,
    "participation": UNKNOWN,
    "universal_domain": UNKNOWN,
    "non_dictatorship": UNKNOWN,
}

_SEP_D_GP_SCT = {
    "direction": True,
    "skew_symmetry": True,
    "consistency": False,
    "participation": True,
    "non_imposition": True,
    "non_dictatorship": False,
    "condorcet_consistent": True,
    "independence_of_irrelevant_alternatives": True,
    "anonymity": True,
}

_SEP_R_GP_SCT = {
    "direction": True,
    "skew_symmetry": False,
    "consistency": False,
    "participation": True,
    "non_imposition": True,
    "non_dictatorship": False,
    "condorcet_consistent": True,
    "independence_of_irrelevant_alternatives": True,
    "anonymity": True,
}

_SEP_D_UNKNOWN_G_SCT = {
    "direction": True,
    "skew_symmetry": True,
    "consistency": UNKNOWN,
    "participation": UNKNOWN,
    "non_imposition": True,
    "non_dictatorship": False,
    "condorcet_consistent": True,
    "independence_of_irrelevant_alternatives": True,
    "anonymity": True,
}


def _single_metric_info(
    *,
    display_name: str,
    abbreviation: str,
    family: str,
    formula_text: str,
    latex: str,
    direction_text: str,
    interpretation: str,
    range_text: str,
    defined_everywhere: bool,
    intra_group_metric_class: str,
    intra_group_properties: dict[str, object],
    aliases: tuple[str, ...] = (),
    note: str | None = None,
) -> dict[str, object]:
    info: dict[str, object] = {
        "display_name": display_name,
        "abbreviation": abbreviation,
        "metric_kind": "single_cm",
        "family": family,
        "intra_group_metric_class": intra_group_metric_class,
        "formula_text": formula_text,
        "latex": latex,
        "g_latex": latex,
        "M_latex": None,
        "no_bias_value": None,
        "range_text": range_text,
        "direction_text": direction_text,
        "defined_everywhere": defined_everywhere,
        "defined_everywhere_scope": "For non-empty confusion matrices (n > 0).",
        "interpretation": interpretation,
        "aliases": aliases,
        "properties": {
            "chapter_3_bias_desiderata": _BASELINE_UNKNOWN.copy(),
            "chapter_4_intra_group_sct": intra_group_properties.copy(),
            "chapter_4_inter_group_sct": _SCT_UNKNOWN.copy(),
        },
    }
    if note is not None:
        info["note"] = note
    return info


def _fairness_metric_info(
    *,
    display_name: str,
    abbreviation: str,
    family: str,
    formula_text: str,
    g_latex: str,
    M_latex: str,
    no_bias_value: float,
    direction_text: str,
    interpretation: str,
    range_text: str,
    defined_everywhere: bool,
    objective_testing: bool,
    real_valued: bool,
    directed: bool,
    symmetric: bool,
    bounded: bool,
    fairness_metric_class: str,
    intra_group_metric_class: str,
    sct_properties: dict[str, object],
    aliases: tuple[str, ...] = (),
    note: str | None = None,
) -> dict[str, object]:
    info: dict[str, object] = {
        "display_name": display_name,
        "abbreviation": abbreviation,
        "metric_kind": "two_cm_fairness",
        "family": family,
        "fairness_metric_class": fairness_metric_class,
        "intra_group_metric_class": intra_group_metric_class,
        "formula_text": formula_text,
        "g_latex": g_latex,
        "M_latex": M_latex,
        "no_bias_value": no_bias_value,
        "range_text": range_text,
        "direction_text": direction_text,
        "defined_everywhere": defined_everywhere,
        "defined_everywhere_scope": "For non-empty groups (n_i > 0 and n_j > 0), unless a secondary denominator can be zero.",
        "interpretation": interpretation,
        "aliases": aliases,
        "properties": {
            "chapter_3_bias_desiderata": {
                "objective_testing": objective_testing,
                "real_valued": real_valued,
                "directed": directed,
                "symmetric": symmetric,
                "bounded": bounded,
                "defined_everywhere": defined_everywhere,
            },
            "chapter_4_intra_group_sct": _UNKNOWN_INTRA_PROPERTIES.copy(),
            "chapter_4_inter_group_sct": sct_properties.copy(),
        },
    }
    if note is not None:
        info["note"] = note
    return info


METRIC_INFO: dict[str, dict[str, object]] = {
    TRUE_POSITIVE_COUNT: _single_metric_info(
        display_name="True Positive Count",
        abbreviation="TP",
        family="Group Count (GC)",
        formula_text="TP",
        latex=r"\mathrm{TP}",
        direction_text="Higher values indicate more correctly identified positives.",
        interpretation="Count of positive examples predicted as positive.",
        range_text="[0, n]",
        defined_everywhere=True,
        intra_group_metric_class="Monotonic Count (MC)",
        intra_group_properties={
            "pi_involution_symmetry": "Not applicable",
            "participation": False,
            "universal_domain": True,
            "non_dictatorship": False,
        },
    ),
    FALSE_NEGATIVE_COUNT: _single_metric_info(
        display_name="False Negative Count",
        abbreviation="FN",
        family="Group Count (GC)",
        formula_text="FN",
        latex=r"\mathrm{FN}",
        direction_text="Higher values indicate more missed positives.",
        interpretation="Count of positive examples predicted as negative.",
        range_text="[0, n]",
        defined_everywhere=True,
        intra_group_metric_class="Monotonic Count (MC)",
        intra_group_properties={
            "pi_involution_symmetry": "Not applicable",
            "participation": False,
            "universal_domain": True,
            "non_dictatorship": False,
        },
    ),
    FALSE_POSITIVE_COUNT: _single_metric_info(
        display_name="False Positive Count",
        abbreviation="FP",
        family="Group Count (GC)",
        formula_text="FP",
        latex=r"\mathrm{FP}",
        direction_text="Higher values indicate more false alarms.",
        interpretation="Count of negative examples predicted as positive.",
        range_text="[0, n]",
        defined_everywhere=True,
        intra_group_metric_class="Monotonic Count (MC)",
        intra_group_properties={
            "pi_involution_symmetry": "Not applicable",
            "participation": False,
            "universal_domain": True,
            "non_dictatorship": False,
        },
    ),
    TRUE_NEGATIVE_COUNT: _single_metric_info(
        display_name="True Negative Count",
        abbreviation="TN",
        family="Group Count (GC)",
        formula_text="TN",
        latex=r"\mathrm{TN}",
        direction_text="Higher values indicate more correctly identified negatives.",
        interpretation="Count of negative examples predicted as negative.",
        range_text="[0, n]",
        defined_everywhere=True,
        intra_group_metric_class="Monotonic Count (MC)",
        intra_group_properties={
            "pi_involution_symmetry": "Not applicable",
            "participation": False,
            "universal_domain": True,
            "non_dictatorship": False,
        },
    ),
    ACTUAL_POSITIVE_COUNT: _single_metric_info(
        display_name="Actual Positive Count",
        abbreviation="P",
        family="Group Count (GC)",
        formula_text="TP + FN",
        latex=r"\mathrm{TP} + \mathrm{FN}",
        direction_text="Higher values indicate more actual positives.",
        interpretation="Count of instances whose true label is positive.",
        range_text="[0, n]",
        defined_everywhere=True,
        intra_group_metric_class="Monotonic Count (MC)",
        intra_group_properties={
            "pi_involution_symmetry": "Not applicable",
            "participation": False,
            "universal_domain": True,
            "non_dictatorship": False,
        },
        aliases=("actual_positive_total",),
    ),
    ACTUAL_NEGATIVE_COUNT: _single_metric_info(
        display_name="Actual Negative Count",
        abbreviation="N",
        family="Group Count (GC)",
        formula_text="FP + TN",
        latex=r"\mathrm{FP} + \mathrm{TN}",
        direction_text="Higher values indicate more actual negatives.",
        interpretation="Count of instances whose true label is negative.",
        range_text="[0, n]",
        defined_everywhere=True,
        intra_group_metric_class="Monotonic Count (MC)",
        intra_group_properties={
            "pi_involution_symmetry": "Not applicable",
            "participation": False,
            "universal_domain": True,
            "non_dictatorship": False,
        },
        aliases=("actual_negative_total",),
    ),
    PREDICTED_POSITIVE_COUNT: _single_metric_info(
        display_name="Predicted Positive Count",
        abbreviation="PP",
        family="Group Count (GC)",
        formula_text="TP + FP",
        latex=r"\mathrm{TP} + \mathrm{FP}",
        direction_text="Higher values indicate more positive predictions.",
        interpretation="Count of instances assigned to the positive prediction.",
        range_text="[0, n]",
        defined_everywhere=True,
        intra_group_metric_class="Monotonic Count (MC)",
        intra_group_properties={
            "pi_involution_symmetry": "Not applicable",
            "participation": False,
            "universal_domain": True,
            "non_dictatorship": False,
        },
        aliases=("P-hat", "predicted_positive_total"),
    ),
    PREDICTED_NEGATIVE_COUNT: _single_metric_info(
        display_name="Predicted Negative Count",
        abbreviation="PN",
        family="Group Count (GC)",
        formula_text="FN + TN",
        latex=r"\mathrm{FN} + \mathrm{TN}",
        direction_text="Higher values indicate more negative predictions.",
        interpretation="Count of instances assigned to the negative prediction.",
        range_text="[0, n]",
        defined_everywhere=True,
        intra_group_metric_class="Monotonic Count (MC)",
        intra_group_properties={
            "pi_involution_symmetry": "Not applicable",
            "participation": False,
            "universal_domain": True,
            "non_dictatorship": False,
        },
        aliases=("N-hat", "predicted_negative_total"),
    ),
    INCORRECT_COUNT: _single_metric_info(
        display_name="Incorrect Count",
        abbreviation="I",
        family="Group Count (GC)",
        formula_text="FP + FN",
        latex=r"\mathrm{FP} + \mathrm{FN}",
        direction_text="Higher values indicate more errors.",
        interpretation="Count of misclassified instances.",
        range_text="[0, n]",
        defined_everywhere=True,
        intra_group_metric_class="Monotonic Count (MC)",
        intra_group_properties={
            "pi_involution_symmetry": "Not applicable",
            "participation": False,
            "universal_domain": True,
            "non_dictatorship": False,
        },
        aliases=("error_count",),
    ),
    CORRECT_COUNT: _single_metric_info(
        display_name="Correct Count",
        abbreviation="C",
        family="Group Count (GC)",
        formula_text="TP + TN",
        latex=r"\mathrm{TP} + \mathrm{TN}",
        direction_text="Higher values indicate more correct classifications.",
        interpretation="Count of correctly classified instances.",
        range_text="[0, n]",
        defined_everywhere=True,
        intra_group_metric_class="Monotonic Count (MC)",
        intra_group_properties={
            "pi_involution_symmetry": "Not applicable",
            "participation": False,
            "universal_domain": True,
            "non_dictatorship": False,
        },
    ),
    TOTAL_COUNT: _single_metric_info(
        display_name="Total Count",
        abbreviation="n",
        family="Group Count (GC)",
        formula_text="TP + FN + FP + TN",
        latex=r"n = \mathrm{TP} + \mathrm{FN} + \mathrm{FP} + \mathrm{TN}",
        direction_text="Higher values indicate a larger group or sample.",
        interpretation="Total number of instances represented by the confusion matrix.",
        range_text="[0, infinity)",
        defined_everywhere=True,
        intra_group_metric_class="Monotonic Count (MC)",
        intra_group_properties={
            "pi_involution_symmetry": "Not applicable",
            "participation": False,
            "universal_domain": True,
            "non_dictatorship": False,
        },
    ),
    ACCURACY: _single_metric_info(
        display_name="Accuracy",
        abbreviation="ACC",
        family="Cell-Pair Density (CPD)",
        formula_text="(TP + TN) / n",
        latex=r"\frac{\mathrm{TP} + \mathrm{TN}}{n}",
        direction_text="Higher values indicate more correct classifications.",
        interpretation="Overall share of correct classifications.",
        range_text="[0, 1]",
        defined_everywhere=True,
        intra_group_metric_class="CPD",
        intra_group_properties=_INTRA_GROUP_PROPORTION_PROPERTIES,
    ),
    PREVALENCE: _single_metric_info(
        display_name="Prevalence",
        abbreviation="PREV",
        family="Cell-Pair Density (CPD)",
        formula_text="(TP + FN) / n",
        latex=r"\frac{\mathrm{TP} + \mathrm{FN}}{n}",
        direction_text="Higher values indicate more actual positives in the group.",
        interpretation="Share of the group whose true label is positive.",
        range_text="[0, 1]",
        defined_everywhere=True,
        intra_group_metric_class="CPD",
        intra_group_properties=_INTRA_GROUP_PROPORTION_PROPERTIES,
        aliases=("actual_positive_rate",),
    ),
    PREDICTED_POSITIVE_RATE: _single_metric_info(
        display_name="Predicted Positive Rate",
        abbreviation="PPR",
        family="Cell-Pair Density (CPD)",
        formula_text="(TP + FP) / n",
        latex=r"\frac{\mathrm{TP} + \mathrm{FP}}{n}",
        direction_text="Higher values indicate more positive predictions.",
        interpretation="Share of cases assigned to the positive prediction.",
        range_text="[0, 1]",
        defined_everywhere=True,
        intra_group_metric_class="CPD",
        intra_group_properties=_INTRA_GROUP_PROPORTION_PROPERTIES,
        aliases=("selection_rate",),
    ),
    INACCURACY: _single_metric_info(
        display_name="Inaccuracy",
        abbreviation="INACC",
        family="Cell-Pair Density (CPD)",
        formula_text="(FP + FN) / n",
        latex=r"\frac{\mathrm{FP} + \mathrm{FN}}{n}",
        direction_text="Higher values indicate more incorrect classifications.",
        interpretation="Overall share of misclassified instances.",
        range_text="[0, 1]",
        defined_everywhere=True,
        intra_group_metric_class="CPD",
        intra_group_properties=_INTRA_GROUP_PROPORTION_PROPERTIES,
        aliases=("error_rate", "misclassification_rate"),
    ),
    NEGATIVE_PREVALENCE: _single_metric_info(
        display_name="Negative Prevalence",
        abbreviation="NPREV",
        family="Cell-Pair Density (CPD)",
        formula_text="(FP + TN) / n",
        latex=r"\frac{\mathrm{FP} + \mathrm{TN}}{n}",
        direction_text="Higher values indicate more actual negatives in the group.",
        interpretation="Share of the group whose true label is negative.",
        range_text="[0, 1]",
        defined_everywhere=True,
        intra_group_metric_class="CPD",
        intra_group_properties=_INTRA_GROUP_PROPORTION_PROPERTIES,
        aliases=("actual_negative_rate",),
    ),
    PREDICTED_NEGATIVE_RATE: _single_metric_info(
        display_name="Predicted Negative Rate",
        abbreviation="PNR",
        family="Cell-Pair Density (CPD)",
        formula_text="(FN + TN) / n",
        latex=r"\frac{\mathrm{FN} + \mathrm{TN}}{n}",
        direction_text="Higher values indicate more negative predictions.",
        interpretation="Share of cases assigned to the negative prediction.",
        range_text="[0, 1]",
        defined_everywhere=True,
        intra_group_metric_class="CPD",
        intra_group_properties=_INTRA_GROUP_PROPORTION_PROPERTIES,
    ),
    TRUE_POSITIVE_RATE: _single_metric_info(
        display_name="True Positive Rate",
        abbreviation="TPR",
        family="Joint Ratio Metric (JRM)",
        formula_text="TP / (TP + FN)",
        latex=r"\frac{\mathrm{TP}}{\mathrm{TP} + \mathrm{FN}}",
        direction_text="Higher values indicate stronger sensitivity to actual positives.",
        interpretation="Share of actual positives correctly identified.",
        range_text="[0, 1] where defined",
        defined_everywhere=False,
        intra_group_metric_class="JRM",
        intra_group_properties=_JRM_PROPERTIES,
        aliases=("recall", "sensitivity", "hit_rate"),
    ),
    FALSE_NEGATIVE_RATE: _single_metric_info(
        display_name="False Negative Rate",
        abbreviation="FNR",
        family="Joint Ratio Metric (JRM)",
        formula_text="FN / (FN + TP)",
        latex=r"\frac{\mathrm{FN}}{\mathrm{FN} + \mathrm{TP}}",
        direction_text="Higher values indicate more missed positives among actual positives.",
        interpretation="Share of actual positives incorrectly classified as negative.",
        range_text="[0, 1] where defined",
        defined_everywhere=False,
        intra_group_metric_class="JRM",
        intra_group_properties=_JRM_PROPERTIES,
        aliases=("miss_rate",),
    ),
    FALSE_POSITIVE_RATE: _single_metric_info(
        display_name="False Positive Rate",
        abbreviation="FPR",
        family="Joint Ratio Metric (JRM)",
        formula_text="FP / (FP + TN)",
        latex=r"\frac{\mathrm{FP}}{\mathrm{FP} + \mathrm{TN}}",
        direction_text="Higher values indicate more false alarms among actual negatives.",
        interpretation="Share of actual negatives incorrectly labeled positive.",
        range_text="[0, 1] where defined",
        defined_everywhere=False,
        intra_group_metric_class="JRM",
        intra_group_properties=_JRM_PROPERTIES,
        aliases=("fall_out",),
    ),
    TRUE_NEGATIVE_RATE: _single_metric_info(
        display_name="True Negative Rate",
        abbreviation="TNR",
        family="Joint Ratio Metric (JRM)",
        formula_text="TN / (TN + FP)",
        latex=r"\frac{\mathrm{TN}}{\mathrm{TN} + \mathrm{FP}}",
        direction_text="Higher values indicate stronger specificity to actual negatives.",
        interpretation="Share of actual negatives correctly identified.",
        range_text="[0, 1] where defined",
        defined_everywhere=False,
        intra_group_metric_class="JRM",
        intra_group_properties=_JRM_PROPERTIES,
        aliases=("specificity",),
    ),
    POSITIVE_PREDICTIVE_VALUE: _single_metric_info(
        display_name="Positive Predictive Value",
        abbreviation="PPV",
        family="Joint Ratio Metric (JRM)",
        formula_text="TP / (TP + FP)",
        latex=r"\frac{\mathrm{TP}}{\mathrm{TP} + \mathrm{FP}}",
        direction_text="Higher values indicate cleaner positive predictions.",
        interpretation="Share of positive predictions that are correct.",
        range_text="[0, 1] where defined",
        defined_everywhere=False,
        intra_group_metric_class="JRM",
        intra_group_properties=_JRM_PROPERTIES,
        aliases=("precision",),
    ),
    FALSE_DISCOVERY_RATE: _single_metric_info(
        display_name="False Discovery Rate",
        abbreviation="FDR",
        family="Joint Ratio Metric (JRM)",
        formula_text="FP / (FP + TP)",
        latex=r"\frac{\mathrm{FP}}{\mathrm{FP} + \mathrm{TP}}",
        direction_text="Higher values indicate noisier positive predictions.",
        interpretation="Share of positive predictions that are false positives.",
        range_text="[0, 1] where defined",
        defined_everywhere=False,
        intra_group_metric_class="JRM",
        intra_group_properties=_JRM_PROPERTIES,
    ),
    NEGATIVE_PREDICTIVE_VALUE: _single_metric_info(
        display_name="Negative Predictive Value",
        abbreviation="NPV",
        family="Joint Ratio Metric (JRM)",
        formula_text="TN / (TN + FN)",
        latex=r"\frac{\mathrm{TN}}{\mathrm{TN} + \mathrm{FN}}",
        direction_text="Higher values indicate cleaner negative predictions.",
        interpretation="Share of negative predictions that are correct.",
        range_text="[0, 1] where defined",
        defined_everywhere=False,
        intra_group_metric_class="JRM",
        intra_group_properties=_JRM_PROPERTIES,
    ),
    FALSE_OMISSION_RATE: _single_metric_info(
        display_name="False Omission Rate",
        abbreviation="FOR",
        family="Joint Ratio Metric (JRM)",
        formula_text="FN / (FN + TN)",
        latex=r"\frac{\mathrm{FN}}{\mathrm{FN} + \mathrm{TN}}",
        direction_text="Higher values indicate noisier negative predictions.",
        interpretation="Share of negative predictions that are false negatives.",
        range_text="[0, 1] where defined",
        defined_everywhere=False,
        intra_group_metric_class="JRM",
        intra_group_properties=_JRM_PROPERTIES,
        aliases=("false_reassurance_rate",),
    ),
    TRUE_POSITIVE_SHARE_OF_CORRECT: _single_metric_info(
        display_name="True Positive Share of Correct",
        abbreviation="TPSC",
        family="Joint Ratio Metric (JRM)",
        formula_text="TP / (TP + TN)",
        latex=r"\frac{\mathrm{TP}}{\mathrm{TP} + \mathrm{TN}}",
        direction_text="Higher values indicate correct predictions are more concentrated in true positives.",
        interpretation="Share of correct predictions that are true positives.",
        range_text="[0, 1] where defined",
        defined_everywhere=False,
        intra_group_metric_class="JRM",
        intra_group_properties=_JRM_PROPERTIES,
    ),
    TRUE_NEGATIVE_SHARE_OF_CORRECT: _single_metric_info(
        display_name="True Negative Share of Correct",
        abbreviation="TNSC",
        family="Joint Ratio Metric (JRM)",
        formula_text="TN / (TN + TP)",
        latex=r"\frac{\mathrm{TN}}{\mathrm{TN} + \mathrm{TP}}",
        direction_text="Higher values indicate correct predictions are more concentrated in true negatives.",
        interpretation="Share of correct predictions that are true negatives.",
        range_text="[0, 1] where defined",
        defined_everywhere=False,
        intra_group_metric_class="JRM",
        intra_group_properties=_JRM_PROPERTIES,
    ),
    FALSE_POSITIVE_SHARE_OF_ERRORS: _single_metric_info(
        display_name="False Positive Share of Errors",
        abbreviation="FPSE",
        family="Joint Ratio Metric (JRM)",
        formula_text="FP / (FP + FN)",
        latex=r"\frac{\mathrm{FP}}{\mathrm{FP} + \mathrm{FN}}",
        direction_text="Higher values indicate errors are more concentrated in false positives.",
        interpretation="Share of misclassifications that are false positives.",
        range_text="[0, 1] where defined",
        defined_everywhere=False,
        intra_group_metric_class="JRM",
        intra_group_properties=_JRM_PROPERTIES,
    ),
    FALSE_NEGATIVE_SHARE_OF_ERRORS: _single_metric_info(
        display_name="False Negative Share of Errors",
        abbreviation="FNSE",
        family="Joint Ratio Metric (JRM)",
        formula_text="FN / (FN + FP)",
        latex=r"\frac{\mathrm{FN}}{\mathrm{FN} + \mathrm{FP}}",
        direction_text="Higher values indicate errors are more concentrated in false negatives.",
        interpretation="Share of misclassifications that are false negatives.",
        range_text="[0, 1] where defined",
        defined_everywhere=False,
        intra_group_metric_class="JRM",
        intra_group_properties=_JRM_PROPERTIES,
    ),
    F1_SCORE: _single_metric_info(
        display_name="F1 Score",
        abbreviation="F1",
        family="Other / Composite",
        formula_text="2 TP / (2 TP + FP + FN)",
        latex=r"\frac{2\mathrm{TP}}{2\mathrm{TP} + \mathrm{FP} + \mathrm{FN}}",
        direction_text="Higher values indicate a better balance of precision and recall.",
        interpretation="Harmonic-mean-style summary of positive-class precision and recall.",
        range_text="[0, 1] where defined",
        defined_everywhere=False,
        intra_group_metric_class="Other / Composite",
        intra_group_properties=_UNKNOWN_INTRA_PROPERTIES,
        aliases=("f1_measure", "dice_sorensen_coefficient"),
    ),
    MATTHEWS_CORRELATION_COEFFICIENT: _single_metric_info(
        display_name="Matthews Correlation Coefficient",
        abbreviation="MCC",
        family="Other / Composite",
        formula_text="(TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))",
        latex=r"\frac{\mathrm{TP}\cdot \mathrm{TN} - \mathrm{FP}\cdot \mathrm{FN}}{\sqrt{(\mathrm{TP}+\mathrm{FP})(\mathrm{TP}+\mathrm{FN})(\mathrm{TN}+\mathrm{FP})(\mathrm{TN}+\mathrm{FN})}}",
        direction_text="Higher values indicate stronger agreement between predictions and labels.",
        interpretation="Correlation-style summary that uses all four confusion-matrix cells.",
        range_text="[-1, 1] where defined",
        defined_everywhere=False,
        intra_group_metric_class="Other / Composite",
        intra_group_properties=_UNKNOWN_INTRA_PROPERTIES,
        aliases=("mcc", "phi_coefficient"),
    ),
    MARGINAL_BENEFIT: _single_metric_info(
        display_name="Marginal Benefit",
        abbreviation="B",
        family="Cell-Pair Trade-Off (CPT)",
        formula_text="(FP - FN) / n",
        latex=r"B = \frac{\mathrm{FP} - \mathrm{FN}}{n}",
        direction_text="Positive values indicate more false positives than false negatives; negative values indicate more false negatives than false positives.",
        interpretation="Signed difference between a group's observed benefit and expected benefit under objective testing.",
        range_text="[-1, 1]",
        defined_everywhere=True,
        intra_group_metric_class="CPT",
        intra_group_properties={
            **_INTRA_GROUP_PROPORTION_PROPERTIES,
            "intra_objective_testing": True,
        },
    ),
    OBJECTIVE_FAIRNESS_INDEX: _fairness_metric_info(
        display_name="Objective Fairness Index",
        abbreviation="OFI",
        family="Two-CM fairness difference",
        formula_text="marginal_benefit(Group i) - marginal_benefit(Group j)",
        g_latex=r"g = B = \frac{\mathrm{FP} - \mathrm{FN}}{n}",
        M_latex=r"M(i,j) = B_i - B_j",
        no_bias_value=0.0,
        direction_text="Positive values indicate greater marginal benefit for Group i; negative values indicate greater marginal benefit for Group j.",
        interpretation="Compares groups by the difference between what happened and what should have happened under objective testing.",
        range_text="[-2, 2]",
        defined_everywhere=True,
        objective_testing=True,
        real_valued=True,
        directed=True,
        symmetric=True,
        bounded=True,
        fairness_metric_class="SEP-D",
        intra_group_metric_class="CPT",
        sct_properties=_SEP_D_GP_SCT,
        note="OFI is the only implemented metric marked as satisfying my dissertation's objective-testing property.",
    ),
    DISPARATE_IMPACT: _fairness_metric_info(
        display_name="Disparate Impact",
        abbreviation="DI",
        family="Two-CM fairness ratio",
        formula_text="predicted_positive_rate(Group i) / predicted_positive_rate(Group j)",
        g_latex=r"g = \frac{\mathrm{TP} + \mathrm{FP}}{n}",
        M_latex=r"M(i,j) = \frac{g_i}{g_j}",
        no_bias_value=1.0,
        direction_text="Values above 1 indicate Group i has a higher predicted-positive rate; values below 1 indicate Group j has a higher predicted-positive rate.",
        interpretation="Ratio of predicted positive rates across two groups; useful as a reparative or screening measure, but not an objective-testing metric.",
        range_text="[0, infinity) where defined",
        defined_everywhere=False,
        objective_testing=False,
        real_valued=True,
        directed=True,
        symmetric=False,
        bounded=False,
        fairness_metric_class="SEP-R",
        intra_group_metric_class="CPD",
        sct_properties=_SEP_R_GP_SCT,
        note="Undefined when the reference group has zero predicted positives; asymmetric because M(i,j) is not a signed difference.",
    ),
    ACCURACY_DIFFERENCE: _fairness_metric_info(
        display_name="Accuracy Difference",
        abbreviation="ACCD",
        family="Two-CM fairness difference",
        formula_text="accuracy(Group i) - accuracy(Group j)",
        g_latex=r"g = \frac{\mathrm{TP} + \mathrm{TN}}{n}",
        M_latex=r"M(i,j) = g_i - g_j",
        no_bias_value=0.0,
        direction_text="Positive values indicate higher accuracy for Group i; negative values indicate higher accuracy for Group j.",
        interpretation="Difference in overall correctness between groups.",
        range_text="[-1, 1]",
        defined_everywhere=True,
        objective_testing=False,
        real_valued=True,
        directed=True,
        symmetric=True,
        bounded=True,
        fairness_metric_class="SEP-D",
        intra_group_metric_class="CPD",
        sct_properties=_SEP_D_GP_SCT,
    ),
    MCC_DIFFERENCE: _fairness_metric_info(
        display_name="MCC Difference",
        abbreviation="MCCD",
        family="Two-CM fairness difference",
        formula_text="MCC(Group i) - MCC(Group j)",
        g_latex=r"g = \frac{\mathrm{TP}\cdot \mathrm{TN} - \mathrm{FP}\cdot \mathrm{FN}}{\sqrt{(\mathrm{TP}+\mathrm{FP})(\mathrm{TP}+\mathrm{FN})(\mathrm{TN}+\mathrm{FP})(\mathrm{TN}+\mathrm{FN})}}",
        M_latex=r"M(i,j) = g_i - g_j",
        no_bias_value=0.0,
        direction_text="Positive values indicate higher MCC for Group i; negative values indicate higher MCC for Group j.",
        interpretation="Difference in Matthews correlation coefficient between groups.",
        range_text="[-2, 2] where defined",
        defined_everywhere=False,
        objective_testing=False,
        real_valued=True,
        directed=True,
        symmetric=True,
        bounded=True,
        fairness_metric_class="SEP-D over an Other / Composite metric",
        intra_group_metric_class="Other / Composite",
        sct_properties={
            **_SEP_D_UNKNOWN_G_SCT,
            "consistency": UNKNOWN,
            "participation": UNKNOWN,
        },
    ),
    PREDICTIVE_PARITY_DIFFERENCE: _fairness_metric_info(
        display_name="Predictive Parity Difference",
        abbreviation="PP",
        family="Two-CM fairness difference",
        formula_text="PPV(Group i) - PPV(Group j)",
        g_latex=r"g = \frac{\mathrm{TP}}{\mathrm{TP} + \mathrm{FP}}",
        M_latex=r"M(i,j) = g_i - g_j",
        no_bias_value=0.0,
        direction_text="Positive values indicate higher positive predictive value for Group i; negative values indicate higher positive predictive value for Group j.",
        interpretation="Difference in the share of positive predictions that are correct.",
        range_text="[-1, 1] where defined",
        defined_everywhere=False,
        objective_testing=False,
        real_valued=True,
        directed=True,
        symmetric=True,
        bounded=True,
        fairness_metric_class="SEP-D",
        intra_group_metric_class="JRM",
        sct_properties=_SEP_D_GP_SCT,
        aliases=("predictive_parity",),
    ),
    TREATMENT_EQUALITY: _fairness_metric_info(
        display_name="Treatment Equality",
        abbreviation="TE",
        family="Two-CM fairness difference",
        formula_text="FN_i / FP_i - FN_j / FP_j",
        g_latex=r"g = \frac{\mathrm{FN}}{\mathrm{FP}}",
        M_latex=r"M(i,j) = g_i - g_j",
        no_bias_value=0.0,
        direction_text="Positive values indicate a larger FN-to-FP tradeoff for Group i under this implementation.",
        interpretation="Difference in the false-negative to false-positive tradeoff across groups.",
        range_text="Unbounded where defined",
        defined_everywhere=False,
        objective_testing=False,
        real_valued=True,
        directed=True,
        symmetric=True,
        bounded=False,
        fairness_metric_class="SEP-D over a non-taxonomy ratio",
        intra_group_metric_class="Unknown",
        sct_properties=_SEP_D_UNKNOWN_G_SCT,
    ),
    DIFFERENCE_IN_CONDITIONAL_ACCEPTANCE: _fairness_metric_info(
        display_name="Difference in Conditional Acceptance",
        abbreviation="DCA",
        family="Two-CM fairness difference",
        formula_text="actual_positive_i / predicted_positive_i - actual_positive_j / predicted_positive_j",
        g_latex=r"g = \frac{\mathrm{TP} + \mathrm{FN}}{\mathrm{TP} + \mathrm{FP}}",
        M_latex=r"M(i,j) = g_i - g_j",
        no_bias_value=0.0,
        direction_text="Positive values indicate a larger actual-positive to predicted-positive ratio for Group i.",
        interpretation="Difference in actual-positive concentration within predicted-positive groups.",
        range_text="Unbounded where defined",
        defined_everywhere=False,
        objective_testing=False,
        real_valued=True,
        directed=True,
        symmetric=True,
        bounded=False,
        fairness_metric_class="SEP-D over a non-taxonomy ratio",
        intra_group_metric_class="Unknown",
        sct_properties=_SEP_D_UNKNOWN_G_SCT,
    ),
    DIFFERENCE_IN_CONDITIONAL_REJECTION: _fairness_metric_info(
        display_name="Difference in Conditional Rejection",
        abbreviation="DCR",
        family="Two-CM fairness difference",
        formula_text="actual_negative_j / predicted_negative_j - actual_negative_i / predicted_negative_i",
        g_latex=r"g = \frac{\mathrm{FP} + \mathrm{TN}}{\mathrm{FN} + \mathrm{TN}}",
        M_latex=r"M(i,j) = g_j - g_i",
        no_bias_value=0.0,
        direction_text="Positive values follow the dissertation's DCR orientation: reference-group conditional rejection minus protected-group conditional rejection.",
        interpretation="Difference in actual-negative concentration within predicted-negative groups, using the DCR orientation from the dissertation.",
        range_text="Unbounded where defined",
        defined_everywhere=False,
        objective_testing=False,
        real_valued=True,
        directed=True,
        symmetric=True,
        bounded=False,
        fairness_metric_class="SEP-D over a non-taxonomy ratio with reversed orientation",
        intra_group_metric_class="Unknown",
        sct_properties=_SEP_D_UNKNOWN_G_SCT,
    ),
    DIFFERENCE_IN_POSITIVE_PROPORTION_AND_LABELS: _fairness_metric_info(
        display_name="Difference in Positive Proportion and Labels",
        abbreviation="DPPL",
        family="Two-CM fairness difference",
        formula_text="predicted_positive_rate(Group i) - predicted_positive_rate(Group j)",
        g_latex=r"g = \frac{\mathrm{TP} + \mathrm{FP}}{n}",
        M_latex=r"M(i,j) = g_i - g_j",
        no_bias_value=0.0,
        direction_text="Positive values indicate a higher predicted-positive rate for Group i; negative values indicate a higher predicted-positive rate for Group j.",
        interpretation="Difference in predicted positive rates across groups.",
        range_text="[-1, 1]",
        defined_everywhere=True,
        objective_testing=False,
        real_valued=True,
        directed=True,
        symmetric=True,
        bounded=True,
        fairness_metric_class="SEP-D",
        intra_group_metric_class="CPD",
        sct_properties=_SEP_D_GP_SCT,
        aliases=("statistical_parity_difference", "demographic_parity_difference", "selection_rate_difference"),
    ),
}

if __name__ == "__main__":
    cm = create_cm(8, 2, 1, 9)
    cms = create_cm(tp=[8, 4], fn=[2, 6], fp=[1, 3], tn=[9, 7])
    print("single cm:\n", cm)
    print("accuracy(single):", accuracy(cm))
    print("stacked cms:\n", cms)
    print("accuracy(stacked):", accuracy(cms))
    print("TPR(stacked):", true_positive_rate(cms))

