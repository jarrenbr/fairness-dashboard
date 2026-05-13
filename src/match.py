"""MATCH tests for confusion-matrix metrics.

Implements the Metric Alignment Trial for Checking Homogeneity (MATCH) tests
from the dissertation chapter on sample-size-induced bias.

CM format, matching metrics.py:
    [[tp, fn],
     [fp, tn]]

Core idea
---------
Given an observed subgroup confusion matrix and a reference confusion matrix,
MATCH asks where the observed metric score falls under the metric distribution
induced by sampling n_obs observations from the reference cell probabilities.

Implemented families
--------------------
1. Binomial / cell-pair density metrics: (c_i + c_j) / n
   - exact binomial CDF
   - normal approximation with continuity correction
   - Peizer-Pratt binomial z approximation
2. Marginal benefit: (FP - FN) / n
   - exact trinomial-difference CDF
   - normal approximation for the signed sum S = FP - FN
   - optional Peizer-Pratt signed heuristic, mainly for cross-checking only
3. Joint-ratio metrics: c_i / (c_i + c_j)
   - exact mixture over the random denominator
   - beta approximation

This file uses NumPy plus SciPy's distribution and special-function routines
for the core probability calculations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil, exp, floor, isfinite, log, sqrt
from typing import Callable, Iterable, Literal, Mapping

import numpy as np
from scipy import special, stats

try:
    from .metrics import TP, FN, FP, TN, create_cm  # type: ignore
except Exception:
    try:  # pragma: no cover - fallback for direct script execution.
        from metrics import TP, FN, FP, TN, create_cm  # type: ignore
    except Exception:  # pragma: no cover - final standalone fallback.
        TP = (0, 0)
        FN = (0, 1)
        FP = (1, 0)
        TN = (1, 1)

        def create_cm(tp, fn, fp, tn) -> np.ndarray:
            return np.array([[tp, fn], [fp, tn]], dtype=np.float64)

Cell = tuple[int, int]
Method = Literal["exact", "normal", "peizer_pratt", "beta"]
Alternative = Literal["two-sided", "less", "greater"]
MatchMode = Literal["exact", "approximate"]


@dataclass(frozen=True)
class MetricSpec:
    """Structural description of a MATCH-supported metric."""

    family: Literal["binomial", "jrm", "marginal_benefit"]
    numerator_cells: tuple[Cell, ...] = ()
    denominator_cells: tuple[Cell, ...] = ()
    positive_cell: Cell | None = None
    negative_cell: Cell | None = None


@dataclass(frozen=True)
class MatchResult:
    """Result of a MATCH test.

    cdf is P(S <= S_obs). p_lower and p_upper are one-sided tail probabilities.
    p_value is selected according to the requested alternative.

    reference_probability is the scalar null parameter carried through the
    analytic family: the success probability p for binomial metrics, the pair
    probability p_i + p_j for JRMs, and the signed mean contribution p_+ - p_-
    for marginal benefit.
    """

    metric: str
    method: str
    alternative: str
    observed_score: float
    n_obs: int
    cdf: float
    p_lower: float
    p_upper: float
    p_value: float
    reference_probability: float
    details: dict[str, float | int | str | bool]

    def asdict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Metric registry
# ---------------------------------------------------------------------------

BINOMIAL_SPECS: dict[str, MetricSpec] = {
    "accuracy": MetricSpec("binomial", numerator_cells=(TP, TN)),
    "prevalence": MetricSpec("binomial", numerator_cells=(TP, FN)),
    "predicted_positive_rate": MetricSpec("binomial", numerator_cells=(TP, FP)),
    "inaccuracy": MetricSpec("binomial", numerator_cells=(FP, FN)),
    "negative_prevalence": MetricSpec("binomial", numerator_cells=(TN, FP)),
    "predicted_negative_rate": MetricSpec("binomial", numerator_cells=(TN, FN)),
}

JRM_SPECS: dict[str, MetricSpec] = {
    "true_positive_rate": MetricSpec("jrm", numerator_cells=(TP,), denominator_cells=(TP, FN)),
    "false_negative_rate": MetricSpec("jrm", numerator_cells=(FN,), denominator_cells=(FN, TP)),
    "false_positive_rate": MetricSpec("jrm", numerator_cells=(FP,), denominator_cells=(FP, TN)),
    "true_negative_rate": MetricSpec("jrm", numerator_cells=(TN,), denominator_cells=(TN, FP)),
    "positive_predictive_value": MetricSpec("jrm", numerator_cells=(TP,), denominator_cells=(TP, FP)),
    "false_discovery_rate": MetricSpec("jrm", numerator_cells=(FP,), denominator_cells=(FP, TP)),
    "negative_predictive_value": MetricSpec("jrm", numerator_cells=(TN,), denominator_cells=(TN, FN)),
    "false_omission_rate": MetricSpec("jrm", numerator_cells=(FN,), denominator_cells=(FN, TN)),
    # Less-common JRMs included in the attached metrics.py taxonomy.
    "true_positive_share_of_correct": MetricSpec("jrm", numerator_cells=(TP,), denominator_cells=(TP, TN)),
    "true_negative_share_of_correct": MetricSpec("jrm", numerator_cells=(TN,), denominator_cells=(TN, TP)),
    "false_positive_share_of_errors": MetricSpec("jrm", numerator_cells=(FP,), denominator_cells=(FP, FN)),
    "false_negative_share_of_errors": MetricSpec("jrm", numerator_cells=(FN,), denominator_cells=(FN, FP)),
}

ALIASES: dict[str, str] = {
    "error_rate": "inaccuracy",
    "misclassification_rate": "inaccuracy",
    "actual_positive_rate": "prevalence",
    "actual_negative_rate": "negative_prevalence",
    "recall": "true_positive_rate",
    "sensitivity": "true_positive_rate",
    "hit_rate": "true_positive_rate",
    "miss_rate": "false_negative_rate",
    "fall_out": "false_positive_rate",
    "specificity": "true_negative_rate",
    "precision": "positive_predictive_value",
    "false_reassurance_rate": "false_omission_rate",
    "marginal_benefit": "marginal_benefit",
    "B": "marginal_benefit",
    "b": "marginal_benefit",
}

METRIC_SPECS: dict[str, MetricSpec] = {
    **BINOMIAL_SPECS,
    **JRM_SPECS,
    "marginal_benefit": MetricSpec(
        "marginal_benefit", positive_cell=FP, negative_cell=FN
    ),
}

MATCH_METRICS: tuple[str, ...] = tuple(METRIC_SPECS)


# ---------------------------------------------------------------------------
# Validation and basic CM utilities
# ---------------------------------------------------------------------------


def _supported_methods_for_family(
    family: Literal["binomial", "jrm", "marginal_benefit"]
) -> tuple[Method, ...]:
    if family == "binomial":
        return ("exact", "normal", "peizer_pratt")
    if family == "marginal_benefit":
        return ("exact", "normal")
    return ("exact", "beta")


def _validate_method_for_metric(metric: str, method: Method) -> None:
    family = METRIC_SPECS[metric].family
    supported = _supported_methods_for_family(family)
    if method not in supported:
        supported_text = ", ".join(supported)
        raise ValueError(f"{family} metrics support methods: {supported_text}")


def canonical_metric_name(metric: str) -> str:
    key = metric.strip()
    return ALIASES.get(key, key)


def _require_single_cm(cm: np.ndarray, name: str = "cm") -> np.ndarray:
    arr = np.asarray(cm, dtype=np.float64)
    if arr.shape != (2, 2):
        raise ValueError(f"{name} must have shape (2, 2); got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(arr < 0):
        raise ValueError(f"{name} must contain non-negative counts")
    total = arr.sum()
    if total <= 0:
        raise ValueError(f"{name} must have positive total count")
    # MATCH is a sampling-count procedure. Tiny roundoff is okay, but fractional
    # CMs are rejected because exact tests require integer counts.
    if not np.allclose(arr, np.round(arr), atol=1e-9):
        raise ValueError(f"{name} must contain integer counts for MATCH exact tests")
    return np.round(arr).astype(np.int64)


def _n(cm: np.ndarray) -> int:
    return int(np.asarray(cm).sum())


def _cell_sum(cm: np.ndarray, cells: Iterable[Cell]) -> int:
    return int(sum(int(cm[cell]) for cell in cells))


def _cell_prob(ref_probs: np.ndarray, cells: Iterable[Cell]) -> float:
    return float(sum(float(ref_probs[cell]) for cell in cells))


def reference_probabilities(reference_cm: np.ndarray) -> np.ndarray:
    ref = _require_single_cm(reference_cm, "reference_cm")
    return ref / ref.sum()


def metric_score(cm: np.ndarray, metric: str) -> float:
    """Compute a MATCH-supported metric from one CM."""

    cm_i = _require_single_cm(cm, "cm")
    name = canonical_metric_name(metric)
    if name not in METRIC_SPECS:
        raise KeyError(f"Unsupported metric {metric!r}. Known: {sorted(METRIC_SPECS)}")
    spec = METRIC_SPECS[name]
    n = _n(cm_i)

    if spec.family == "binomial":
        return _cell_sum(cm_i, spec.numerator_cells) / n

    if spec.family == "jrm":
        num = _cell_sum(cm_i, spec.numerator_cells)
        den = _cell_sum(cm_i, spec.denominator_cells)
        return float("nan") if den == 0 else num / den

    if spec.family == "marginal_benefit":
        assert spec.positive_cell is not None and spec.negative_cell is not None
        return (int(cm_i[spec.positive_cell]) - int(cm_i[spec.negative_cell])) / n

    raise AssertionError(f"Unhandled metric family {spec.family}")


# ---------------------------------------------------------------------------
# Distribution helpers
# ---------------------------------------------------------------------------


def _normal_cdf(z: float) -> float:
    if z == float("inf"):
        return 1.0
    if z == float("-inf"):
        return 0.0
    return _clip01(stats.norm.cdf(z))


def _clip01(x: float) -> float:
    if np.isnan(x):
        return float("nan")
    return min(1.0, max(0.0, float(x)))


def _tail_p_value(lower: float, upper: float, alternative: Alternative) -> float:
    lower = _clip01(lower)
    upper = _clip01(upper)
    if alternative == "less":
        return lower
    if alternative == "greater":
        return upper
    if alternative == "two-sided":
        return _clip01(2.0 * min(lower, upper))
    raise ValueError("alternative must be 'two-sided', 'less', or 'greater'")


def _logsumexp(log_values: Iterable[float]) -> float:
    vals = [v for v in log_values if v != float("-inf")]
    if not vals:
        return float("-inf")
    return float(special.logsumexp(vals))


def binom_pmf(k: int, n: int, p: float) -> float:
    if k < 0 or k > n:
        return 0.0
    if p < 0.0 or p > 1.0:
        raise ValueError(f"binomial probability must be in [0,1], got {p}")
    return _clip01(stats.binom.pmf(k, n, p))


def binom_cdf(k: int, n: int, p: float) -> float:
    """Exact Binomial(n, p) CDF P(X <= k)."""

    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if p == 0.0:
        return 1.0
    if p == 1.0:
        return 0.0 if k < n else 1.0
    return _clip01(stats.binom.cdf(k, n, p))


def binom_sf(k: int, n: int, p: float) -> float:
    """Exact Binomial(n, p) survival P(X >= k)."""
    return _clip01(stats.binom.sf(k - 1, n, p))


# ---------------------------------------------------------------------------
# Peizer-Pratt and beta approximation helpers
# ---------------------------------------------------------------------------


def peizer_pratt_binom_cdf(k: int, n: int, p: float, *, improved: bool = True) -> float:
    """Peizer-Pratt normal approximation to Binomial(n, p) CDF.

    Returns an approximation to P(X <= k). It is designed for integer k in [0, n].
    For boundary and ill-conditioned values, it falls back to the normal
    approximation with continuity correction.
    """

    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if n <= 0:
        raise ValueError("n must be positive")
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0 if k < n else 1.0

    q = 1.0 - p
    # Peizer-Pratt is not stable near the boundaries. Use normal there.
    normal = binomial_normal_cdf(k, n, p)
    if min(k + 1, n - k) < 2:
        return normal

    x = float(k)
    denom_mid = x + 0.5 - n * p
    if abs(denom_mid) < 1e-12:
        return normal

    a = x + 2.0 / 3.0 - (n + 1.0 / 3.0) * p
    if improved:
        a += (1.0 / 50.0) * (
            q / (x + 1.0)
            - p / (n - x)
            + (q - 0.5) / (n + 1.0)
        )

    xh = x + 0.5
    nxh = n - x - 0.5
    # Both xh and nxh are positive after boundary handling.
    dev = 2.0 * (xh * log(xh / (n * p)) + nxh * log(nxh / (n * q)))
    if dev < 0.0 and dev > -1e-12:  # roundoff
        dev = 0.0
    if dev < 0.0 or not isfinite(dev):
        return normal

    try:
        z = (a / sqrt((n + 1.0 / 6.0) * p * q))
        z *= sqrt(n * p * q) / denom_mid
        z *= sqrt(dev)
        # Preserve the tail direction relative to the continuity-corrected mean.
        z = float(np.sign(denom_mid) * abs(z))
    except (ValueError, ZeroDivisionError, OverflowError):
        return normal
    return _clip01(_normal_cdf(z))


def binomial_normal_cdf(k: int, n: int, p: float) -> float:
    """Normal approximation to Binomial(n,p) CDF P(X <= k)."""

    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    q = 1.0 - p
    var = n * p * q
    if var <= 0.0:
        return 1.0 if (p == 0.0 or k >= n) else 0.0
    z = (k + 0.5 - n * p) / sqrt(var)
    return _clip01(_normal_cdf(z))


def regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    """Regularized incomplete beta I_x(a,b)."""

    if a <= 0.0 or b <= 0.0:
        raise ValueError("a and b must be positive")
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    return _clip01(special.betainc(a, b, x))


# ---------------------------------------------------------------------------
# Binomial MATCH
# ---------------------------------------------------------------------------


def _binomial_match_from_probs(
    observed_cm: np.ndarray,
    ref_probs: np.ndarray,
    metric: str,
    method: Method,
    alternative: Alternative,
) -> MatchResult:
    obs = _require_single_cm(observed_cm, "observed_cm")
    spec = METRIC_SPECS[metric]
    n = _n(obs)
    k = _cell_sum(obs, spec.numerator_cells)
    p = _cell_prob(ref_probs, spec.numerator_cells)
    score = k / n

    if method == "exact":
        lower = binom_cdf(k, n, p)
        upper = binom_sf(k, n, p)
    elif method == "normal":
        lower = binomial_normal_cdf(k, n, p)
        # P(X >= k) approximated as 1 - P(X <= k-1), with continuity correction.
        upper = _clip01(1.0 - binomial_normal_cdf(k - 1, n, p))
    elif method == "peizer_pratt":
        lower = peizer_pratt_binom_cdf(k, n, p)
        upper = _clip01(1.0 - peizer_pratt_binom_cdf(k - 1, n, p))
    else:
        raise ValueError("Binomial metrics support methods: exact, normal, peizer_pratt")

    return MatchResult(
        metric=metric,
        method=method,
        alternative=alternative,
        observed_score=score,
        n_obs=n,
        cdf=lower,
        p_lower=lower,
        p_upper=upper,
        p_value=_tail_p_value(lower, upper, alternative),
        reference_probability=p,
        details={"k_success": k, "p_success": p},
    )


# ---------------------------------------------------------------------------
# Marginal benefit MATCH
# ---------------------------------------------------------------------------


def marginal_benefit(cm: np.ndarray) -> float:
    return metric_score(cm, "marginal_benefit")


def marginal_benefit_pmf_k(k: int, n: int, p_plus: float, p_minus: float, p_zero: float) -> float:
    """Exact PMF P(FP - FN = k) for one multinomial sample of size n."""

    if k < -n or k > n:
        return 0.0
    if any(p < 0 for p in (p_plus, p_minus, p_zero)):
        raise ValueError("Probabilities must be non-negative")
    if not np.isclose(p_plus + p_minus + p_zero, 1.0, atol=1e-9):
        raise ValueError("p_plus + p_minus + p_zero must equal 1")

    fn_min = max(0, -k)
    fn_max = (n - k) // 2
    if fn_min > fn_max:
        return 0.0

    log_terms: list[float] = []
    for fn in range(fn_min, fn_max + 1):
        fp = k + fn
        zero = n - fp - fn
        if fp < 0 or zero < 0:
            continue
        log_terms.append(
            float(
                stats.multinomial.logpmf(
                    [fp, fn, zero],
                    n=n,
                    p=[p_plus, p_minus, p_zero],
                )
            )
        )
    return _clip01(exp(_logsumexp(log_terms)))


def marginal_benefit_cdf_k(k: int, n: int, p_plus: float, p_minus: float, p_zero: float) -> float:
    """Exact CDF P(FP - FN <= k)."""

    if k < -n:
        return 0.0
    if k >= n:
        return 1.0
    return _clip01(
        sum(marginal_benefit_pmf_k(s, n, p_plus, p_minus, p_zero) for s in range(-n, k + 1))
    )


def marginal_benefit_normal_cdf_k(k: int, n: int, p_plus: float, p_minus: float, p_zero: float) -> float:
    """Normal approximation to P(FP - FN <= k)."""

    mu_one = p_plus - p_minus
    var_one = p_plus + p_minus - mu_one * mu_one
    if var_one <= 0.0:
        deterministic = n * mu_one
        return 1.0 if k >= deterministic else 0.0
    z = (k + 0.5 - n * mu_one) / sqrt(n * var_one)
    return _clip01(_normal_cdf(z))


def _marginal_benefit_match(
    observed_cm: np.ndarray,
    reference_cm: np.ndarray,
    method: Method,
    alternative: Alternative,
) -> MatchResult:
    return _marginal_benefit_match_from_probs(
        observed_cm,
        reference_probabilities(reference_cm),
        method,
        alternative,
    )


def _marginal_benefit_match_from_probs(
    observed_cm: np.ndarray,
    ref_probs: np.ndarray,
    method: Method,
    alternative: Alternative,
) -> MatchResult:
    obs = _require_single_cm(observed_cm, "observed_cm")
    n = _n(obs)
    k = int(obs[FP]) - int(obs[FN])
    score = k / n
    p_plus = float(ref_probs[FP])
    p_minus = float(ref_probs[FN])
    p_zero = float(ref_probs[TP] + ref_probs[TN])

    if method == "exact":
        lower = marginal_benefit_cdf_k(k, n, p_plus, p_minus, p_zero)
        upper = _clip01(1.0 - marginal_benefit_cdf_k(k - 1, n, p_plus, p_minus, p_zero))
    elif method == "normal":
        lower = marginal_benefit_normal_cdf_k(k, n, p_plus, p_minus, p_zero)
        upper = _clip01(1.0 - marginal_benefit_normal_cdf_k(k - 1, n, p_plus, p_minus, p_zero))
    else:
        raise ValueError("marginal_benefit metrics support methods: exact, normal")

    return MatchResult(
        metric="marginal_benefit",
        method=method,
        alternative=alternative,
        observed_score=score,
        n_obs=n,
        cdf=lower,
        p_lower=lower,
        p_upper=upper,
        p_value=_tail_p_value(lower, upper, alternative),
        reference_probability=p_plus - p_minus,
        details={
            "k_fp_minus_fn": k,
            "p_plus_fp": p_plus,
            "p_minus_fn": p_minus,
            "p_zero_tp_tn": p_zero,
        },
    )


# ---------------------------------------------------------------------------
# JRM MATCH
# ---------------------------------------------------------------------------


def _jrm_exact_cdf(
    score: float,
    n: int,
    p_pair: float,
    theta: float,
    *,
    condition_on_defined: bool,
    strict_less: bool = False,
) -> float:
    """Exact mixture CDF for c_i/(c_i+c_j).

    If strict_less is True, computes P(S < score); otherwise P(S <= score).
    """

    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    if p_pair <= 0.0:
        return float("nan")

    total_prob = 0.0
    for k_pair in range(1, n + 1):
        pair_prob = binom_pmf(k_pair, n, p_pair)
        if pair_prob == 0.0:
            continue
        if strict_less:
            max_num = ceil(k_pair * score - 1e-12) - 1
        else:
            max_num = floor(k_pair * score + 1e-12)
        cond = binom_cdf(max_num, k_pair, theta)
        total_prob += pair_prob * cond

    if condition_on_defined:
        defined_prob = 1.0 - (1.0 - p_pair) ** n
        if defined_prob <= 0.0:
            return float("nan")
        total_prob /= defined_prob
    return _clip01(total_prob)


def _jrm_exact_upper(
    score: float,
    n: int,
    p_pair: float,
    theta: float,
    *,
    condition_on_defined: bool,
) -> float:
    # P(S >= score) with the same conditioning convention used by _jrm_exact_cdf.
    less = _jrm_exact_cdf(score, n, p_pair, theta, condition_on_defined=condition_on_defined, strict_less=True)
    if condition_on_defined:
        return _clip01(1.0 - less)
    defined_mass = 1.0 - (1.0 - p_pair) ** n
    if defined_mass <= 0.0:
        return float("nan")
    return _clip01(defined_mass - less)


def _jrm_beta_cdf(
    score: float,
    n: int,
    p_i: float,
    p_j: float,
    observed_pair_count: int,
    *,
    lam: float,
    beta_count: Literal["expected", "observed"],
) -> float:
    """Beta approximation to the JRM distribution.

    beta_count='expected' uses n*p_i and n*p_j as approximate counts under the
    reference distribution. beta_count='observed' uses the observed denominator
    and the reference conditional theta.
    """

    p_pair = p_i + p_j
    if p_pair <= 0.0:
        return float("nan")
    theta = p_i / p_pair
    if beta_count == "expected":
        alpha = n * p_i + lam
        beta = n * p_j + lam
    elif beta_count == "observed":
        alpha = observed_pair_count * theta + lam
        beta = observed_pair_count * (1.0 - theta) + lam
    else:
        raise ValueError("beta_count must be 'expected' or 'observed'")
    return _clip01(stats.beta.cdf(score, alpha, beta))


def _jrm_match(
    observed_cm: np.ndarray,
    reference_cm: np.ndarray,
    metric: str,
    method: Method,
    alternative: Alternative,
    *,
    condition_on_defined: bool,
    beta_lambda: float,
    beta_count: Literal["expected", "observed"],
) -> MatchResult:
    return _jrm_match_from_probs(
        observed_cm,
        reference_probabilities(reference_cm),
        metric,
        method,
        alternative,
        condition_on_defined=condition_on_defined,
        beta_lambda=beta_lambda,
        beta_count=beta_count,
    )


def _jrm_match_from_probs(
    observed_cm: np.ndarray,
    ref_probs: np.ndarray,
    metric: str,
    method: Method,
    alternative: Alternative,
    *,
    condition_on_defined: bool,
    beta_lambda: float,
    beta_count: Literal["expected", "observed"],
) -> MatchResult:
    obs = _require_single_cm(observed_cm, "observed_cm")
    spec = METRIC_SPECS[metric]
    n = _n(obs)

    numerator_cell = spec.numerator_cells[0]
    denom_cells = spec.denominator_cells
    if len(denom_cells) != 2:
        raise AssertionError("JRM specs should use exactly two denominator cells")
    other_cell = denom_cells[1]

    obs_num = _cell_sum(obs, (numerator_cell,))
    obs_den = _cell_sum(obs, denom_cells)
    if obs_den == 0:
        raise ValueError(f"Observed {metric} is undefined because its denominator is zero")
    score = obs_num / obs_den

    p_i = _cell_prob(ref_probs, (numerator_cell,))
    p_j = _cell_prob(ref_probs, (other_cell,))
    p_pair = p_i + p_j
    theta = float("nan") if p_pair == 0.0 else p_i / p_pair

    if method == "exact":
        lower = _jrm_exact_cdf(score, n, p_pair, theta, condition_on_defined=condition_on_defined)
        upper = _jrm_exact_upper(score, n, p_pair, theta, condition_on_defined=condition_on_defined)
    elif method == "beta":
        lower = _jrm_beta_cdf(
            score, n, p_i, p_j, obs_den, lam=beta_lambda, beta_count=beta_count
        )
        alpha, beta = _jrm_beta_params(n, p_i, p_j, obs_den, beta_lambda, beta_count)
        upper = _clip01(stats.beta.sf(score, alpha, beta))
    else:
        raise ValueError("JRM metrics support methods: exact, beta")

    return MatchResult(
        metric=metric,
        method=method,
        alternative=alternative,
        observed_score=score,
        n_obs=n,
        cdf=lower,
        p_lower=lower,
        p_upper=upper,
        p_value=_tail_p_value(lower, upper, alternative),
        reference_probability=p_pair,
        details={
            "observed_numerator": obs_num,
            "observed_denominator": obs_den,
            "p_i": p_i,
            "p_j": p_j,
            "p_pair": p_pair,
            "theta_i_given_pair": theta,
            "condition_on_defined": condition_on_defined,
            "beta_lambda": beta_lambda,
            "beta_count": beta_count,
        },
    )


def _jrm_beta_params(
    n: int,
    p_i: float,
    p_j: float,
    observed_pair_count: int,
    lam: float,
    beta_count: Literal["expected", "observed"],
) -> tuple[float, float]:
    p_pair = p_i + p_j
    if p_pair <= 0.0:
        return float("nan"), float("nan")
    theta = p_i / p_pair
    if beta_count == "expected":
        return n * p_i + lam, n * p_j + lam
    if beta_count == "observed":
        return observed_pair_count * theta + lam, observed_pair_count * (1.0 - theta) + lam
    raise ValueError("beta_count must be 'expected' or 'observed'")


def _binomial_match_score_from_probs(
    observed_score: float,
    n_obs: int,
    ref_probs: np.ndarray,
    metric: str,
    method: Method,
    alternative: Alternative,
) -> MatchResult:
    if not np.isfinite(observed_score):
        raise ValueError("observed_score must be finite")
    spec = METRIC_SPECS[metric]
    p = _cell_prob(ref_probs, spec.numerator_cells)
    k_lower = floor(n_obs * observed_score + 1e-12)
    k_upper = ceil(n_obs * observed_score - 1e-12)

    if method == "exact":
        lower = binom_cdf(k_lower, n_obs, p)
        upper = binom_sf(k_upper, n_obs, p)
    elif method == "normal":
        lower = binomial_normal_cdf(k_lower, n_obs, p)
        upper = _clip01(1.0 - binomial_normal_cdf(k_upper - 1, n_obs, p))
    elif method == "peizer_pratt":
        lower = peizer_pratt_binom_cdf(k_lower, n_obs, p)
        upper = _clip01(1.0 - peizer_pratt_binom_cdf(k_upper - 1, n_obs, p))
    else:
        raise ValueError("Binomial metrics support methods: exact, normal, peizer_pratt")

    return MatchResult(
        metric=metric,
        method=method,
        alternative=alternative,
        observed_score=observed_score,
        n_obs=n_obs,
        cdf=lower,
        p_lower=lower,
        p_upper=upper,
        p_value=_tail_p_value(lower, upper, alternative),
        reference_probability=p,
        details={
            "score_lower_count": k_lower,
            "score_upper_count": k_upper,
            "p_success": p,
        },
    )


def _marginal_benefit_match_score_from_probs(
    observed_score: float,
    n_obs: int,
    ref_probs: np.ndarray,
    method: Method,
    alternative: Alternative,
) -> MatchResult:
    if not np.isfinite(observed_score):
        raise ValueError("observed_score must be finite")
    p_plus = float(ref_probs[FP])
    p_minus = float(ref_probs[FN])
    p_zero = float(ref_probs[TP] + ref_probs[TN])
    k_lower = floor(n_obs * observed_score + 1e-12)
    k_upper = ceil(n_obs * observed_score - 1e-12)

    if method == "exact":
        lower = marginal_benefit_cdf_k(k_lower, n_obs, p_plus, p_minus, p_zero)
        upper = _clip01(1.0 - marginal_benefit_cdf_k(k_upper - 1, n_obs, p_plus, p_minus, p_zero))
    elif method == "normal":
        lower = marginal_benefit_normal_cdf_k(k_lower, n_obs, p_plus, p_minus, p_zero)
        upper = _clip01(1.0 - marginal_benefit_normal_cdf_k(k_upper - 1, n_obs, p_plus, p_minus, p_zero))
    else:
        raise ValueError("marginal_benefit metrics support methods: exact, normal")

    return MatchResult(
        metric="marginal_benefit",
        method=method,
        alternative=alternative,
        observed_score=observed_score,
        n_obs=n_obs,
        cdf=lower,
        p_lower=lower,
        p_upper=upper,
        p_value=_tail_p_value(lower, upper, alternative),
        reference_probability=p_plus - p_minus,
        details={
            "score_lower_count": k_lower,
            "score_upper_count": k_upper,
            "p_plus_fp": p_plus,
            "p_minus_fn": p_minus,
            "p_zero_tp_tn": p_zero,
        },
    )


def _jrm_match_score_from_probs(
    observed_score: float,
    n_obs: int,
    ref_probs: np.ndarray,
    metric: str,
    method: Method,
    alternative: Alternative,
    *,
    observed_pair_count: int | None,
    condition_on_defined: bool,
    beta_lambda: float,
    beta_count: Literal["expected", "observed"],
) -> MatchResult:
    if not np.isfinite(observed_score):
        raise ValueError("observed_score must be finite")
    spec = METRIC_SPECS[metric]
    numerator_cell = spec.numerator_cells[0]
    denom_cells = spec.denominator_cells
    if len(denom_cells) != 2:
        raise AssertionError("JRM specs should use exactly two denominator cells")
    other_cell = denom_cells[1]

    p_i = _cell_prob(ref_probs, (numerator_cell,))
    p_j = _cell_prob(ref_probs, (other_cell,))
    p_pair = p_i + p_j
    theta = float("nan") if p_pair == 0.0 else p_i / p_pair

    if method == "exact":
        lower = _jrm_exact_cdf(observed_score, n_obs, p_pair, theta, condition_on_defined=condition_on_defined)
        upper = _jrm_exact_upper(observed_score, n_obs, p_pair, theta, condition_on_defined=condition_on_defined)
        beta_count_used = beta_count
    elif method == "beta":
        beta_count_used = beta_count if observed_pair_count is not None else "expected"
        pair_count = observed_pair_count if observed_pair_count is not None else max(1, int(round(n_obs * p_pair)))
        lower = _jrm_beta_cdf(
            observed_score,
            n_obs,
            p_i,
            p_j,
            pair_count,
            lam=beta_lambda,
            beta_count=beta_count_used,
        )
        alpha, beta = _jrm_beta_params(n_obs, p_i, p_j, pair_count, beta_lambda, beta_count_used)
        upper = _clip01(stats.beta.sf(observed_score, alpha, beta))
    else:
        raise ValueError("JRM metrics support methods: exact, beta")

    return MatchResult(
        metric=metric,
        method=method,
        alternative=alternative,
        observed_score=observed_score,
        n_obs=n_obs,
        cdf=lower,
        p_lower=lower,
        p_upper=upper,
        p_value=_tail_p_value(lower, upper, alternative),
        reference_probability=p_pair,
        details={
            "observed_denominator": -1 if observed_pair_count is None else observed_pair_count,
            "p_i": p_i,
            "p_j": p_j,
            "p_pair": p_pair,
            "theta_i_given_pair": theta,
            "condition_on_defined": condition_on_defined,
            "beta_lambda": beta_lambda,
            "beta_count": beta_count_used,
        },
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def match_test(
    observed_cm: np.ndarray,
    reference_cm: np.ndarray,
    metric: str,
    *,
    method: Method = "exact",
    alternative: Alternative = "two-sided",
    condition_on_defined: bool = True,
    beta_lambda: float = 1.0,
    beta_count: Literal["expected", "observed"] = "observed",
) -> MatchResult:
    """Run a MATCH test for one observed group against a reference CM.

    Parameters
    ----------
    observed_cm:
        Group/subgroup CM with integer counts.
    reference_cm:
        Reference CM whose normalized cells define the null distribution.
    metric:
        Any key in METRIC_SPECS, or an alias in ALIASES.
    method:
        - binomial metrics: 'exact', 'normal', 'peizer_pratt'
        - marginal_benefit: 'exact', 'normal'
        - JRM metrics: 'exact', 'beta'
    alternative:
        'two-sided', 'less', or 'greater'. 'less' tests unusually small scores;
        'greater' tests unusually large scores.
    condition_on_defined:
        For exact JRMs, condition on c_i+c_j > 0. If False, uses the dissertation's
        unnormalized mixture over k=1..n, so P(S<=1) can be < 1 when undefined
        cases have positive probability, and the upper tail follows the same
        unnormalized convention over defined outcomes.
    beta_lambda:
        Pseudocount for beta approximation.
    beta_count:
        For JRM beta approximation, use expected reference counts or observed
        denominator-scaled counts.
    """

    name = canonical_metric_name(metric)
    if name not in METRIC_SPECS:
        raise KeyError(f"Unsupported metric {metric!r}. Known: {sorted(METRIC_SPECS)}")
    _validate_method_for_metric(name, method)

    return match_test_from_reference_probs(
        observed_cm,
        reference_probabilities(reference_cm),
        name,
        method=method,
        alternative=alternative,
        condition_on_defined=condition_on_defined,
        beta_lambda=beta_lambda,
        beta_count=beta_count,
    )


def match_test_from_reference_probs(
    observed_cm: np.ndarray,
    reference_probs: np.ndarray,
    metric: str,
    *,
    method: Method = "exact",
    alternative: Alternative = "two-sided",
    condition_on_defined: bool = True,
    beta_lambda: float = 1.0,
    beta_count: Literal["expected", "observed"] = "observed",
) -> MatchResult:
    """Run a MATCH test using validated reference probabilities directly."""

    name = canonical_metric_name(metric)
    if name not in METRIC_SPECS:
        raise KeyError(f"Unsupported metric {metric!r}. Known: {sorted(METRIC_SPECS)}")
    spec = METRIC_SPECS[name]
    probs = validate_reference_probs(reference_probs)
    _validate_method_for_metric(name, method)

    if spec.family == "binomial":
        return _binomial_match_from_probs(observed_cm, probs, name, method, alternative)
    if spec.family == "marginal_benefit":
        return _marginal_benefit_match_from_probs(observed_cm, probs, method, alternative)
    if spec.family == "jrm":
        return _jrm_match_from_probs(
            observed_cm,
            probs,
            name,
            method,
            alternative,
            condition_on_defined=condition_on_defined,
            beta_lambda=beta_lambda,
            beta_count=beta_count,
        )
    raise AssertionError(f"Unhandled metric family {spec.family}")


def match_test_score_from_reference_probs(
    observed_score: float,
    n_obs: int,
    reference_probs: np.ndarray,
    metric: str,
    *,
    method: Method = "exact",
    alternative: Alternative = "two-sided",
    observed_denominator: int | None = None,
    condition_on_defined: bool = True,
    beta_lambda: float = 1.0,
    beta_count: Literal["expected", "observed"] = "observed",
) -> MatchResult:
    """Run a MATCH test from an observed metric score and target sample size."""

    if n_obs <= 0:
        raise ValueError("n_obs must be positive")

    name = canonical_metric_name(metric)
    if name not in METRIC_SPECS:
        raise KeyError(f"Unsupported metric {metric!r}. Known: {sorted(METRIC_SPECS)}")
    spec = METRIC_SPECS[name]
    probs = validate_reference_probs(reference_probs)
    _validate_method_for_metric(name, method)

    if spec.family == "binomial":
        return _binomial_match_score_from_probs(observed_score, n_obs, probs, name, method, alternative)
    if spec.family == "marginal_benefit":
        return _marginal_benefit_match_score_from_probs(observed_score, n_obs, probs, method, alternative)
    if spec.family == "jrm":
        return _jrm_match_score_from_probs(
            observed_score,
            n_obs,
            probs,
            name,
            method,
            alternative,
            observed_pair_count=observed_denominator,
            condition_on_defined=condition_on_defined,
            beta_lambda=beta_lambda,
            beta_count=beta_count,
        )
    raise AssertionError(f"Unhandled metric family {spec.family}")


def preferred_match_method(metric: str, mode: MatchMode = "exact") -> Method:
    """Return the MATCH method that best matches the dissertation workflow."""

    if mode == "exact":
        return "exact"

    name = canonical_metric_name(metric)
    if name not in METRIC_SPECS:
        raise KeyError(f"Unsupported metric {metric!r}. Known: {sorted(METRIC_SPECS)}")

    family = METRIC_SPECS[name].family
    if family == "binomial":
        return "peizer_pratt"
    if family == "marginal_benefit":
        return "normal"
    return "beta"


def compare_match_methods(
    observed_cm: np.ndarray,
    reference_cm: np.ndarray,
    metric: str,
    *,
    alternative: Alternative = "two-sided",
    condition_on_defined: bool = True,
    beta_lambda: float = 1.0,
    beta_count: Literal["expected", "observed"] = "observed",
) -> list[MatchResult]:
    """Run exact and approximation methods for cross-checking."""

    name = canonical_metric_name(metric)
    methods = _supported_methods_for_family(METRIC_SPECS[name].family)
    return [
        match_test(
            observed_cm,
            reference_cm,
            name,
            method=m,
            alternative=alternative,
            condition_on_defined=condition_on_defined,
            beta_lambda=beta_lambda,
            beta_count=beta_count,
        )
        for m in methods
    ]


def match_test_many(
    observed_cms: np.ndarray,
    reference_cm: np.ndarray,
    metric: str,
    **kwargs,
) -> list[MatchResult]:
    """Apply match_test over a stack of observed CMs with shape (..., 2, 2)."""

    arr = np.asarray(observed_cms)
    if arr.shape[-2:] != (2, 2):
        raise ValueError("observed_cms must have shape (..., 2, 2)")
    flat = arr.reshape((-1, 2, 2))
    return [match_test(cm, reference_cm, metric, **kwargs) for cm in flat]


def results_to_records(results: Iterable[MatchResult]) -> list[dict]:
    """Convert MatchResult objects into dictionaries for DataFrame/dashboard use."""

    return [r.asdict() for r in results]


# ---------------------------------------------------------------------------
# Simulation-based MATCH for the dashboard UI
# ---------------------------------------------------------------------------


def validate_reference_probs(reference_probs: np.ndarray) -> np.ndarray:
    """Validate multinomial reference probabilities with shape ``(2, 2)``."""

    probs = np.asarray(reference_probs, dtype=np.float64)
    if probs.shape == (4,):
        probs = probs.reshape(2, 2)
    if probs.shape != (2, 2):
        raise ValueError(f"reference_probs must have shape (2, 2); got {probs.shape}")
    if not np.all(np.isfinite(probs)):
        raise ValueError("reference_probs must contain only finite values")
    if np.any(probs < 0):
        raise ValueError("reference_probs must be non-negative")
    total = float(probs.sum())
    if total <= 0.0:
        raise ValueError("reference_probs must have positive mass")
    if not np.isclose(total, 1.0, atol=1e-9):
        raise ValueError("reference_probs must sum to 1")
    return probs


def multinomial_reference_samples(
    probs: np.ndarray,
    n: int,
    num_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw reference confusion matrices from a multinomial distribution."""

    ref_probs = validate_reference_probs(probs).reshape(-1)
    if n < 0:
        raise ValueError("n must be non-negative")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    draws = rng.multinomial(n=n, pvals=ref_probs, size=num_samples)
    return draws.reshape(num_samples, 2, 2).astype(np.float64)


def metric_distribution(
    cms: np.ndarray,
    metric_func,
) -> np.ndarray:
    """Apply a scalar metric to a stack of confusion matrices."""

    arr = np.asarray(cms, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[-2:] != (2, 2):
        raise ValueError("cms must have shape (num_samples, 2, 2)")
    values = [float(np.asarray(metric_func(cm), dtype=np.float64)) for cm in arr]
    return np.asarray(values, dtype=np.float64)


def simulation_match_test(
    observed_cm: np.ndarray,
    reference_probs: np.ndarray,
    metric_func,
    n: int | None = None,
    num_samples: int = 10000,
    alternative: Alternative = "two-sided",
    seed: int | None = 12345,
) -> dict[str, object]:
    """Run a simulation-based MATCH test for one observed confusion matrix."""

    return simulation_match_test_batched(
        observed_cm=observed_cm,
        reference_probs=reference_probs,
        metric_func=metric_func,
        n=n,
        num_samples=num_samples,
        alternative=alternative,
        seed=seed,
        batch_size=num_samples,
    )


def _summarize_simulation_match_result(
    observed_value: float,
    reference_values: np.ndarray,
    *,
    alternative: Alternative,
    sample_size: int,
    seed: int | None,
) -> dict[str, object]:
    finite_mask = np.isfinite(reference_values)
    finite_values = reference_values[finite_mask]
    undefined_rate = float(np.mean(~finite_mask)) if reference_values.size else float("nan")

    result: dict[str, object] = {
        "observed_value": observed_value,
        "reference_mean": float("nan"),
        "reference_std": float("nan"),
        "reference_median": float("nan"),
        "q025": float("nan"),
        "q05": float("nan"),
        "q95": float("nan"),
        "q975": float("nan"),
        "p_value": float("nan"),
        "undefined_rate": undefined_rate,
        "finite_sample_count": int(finite_values.size),
        "total_sample_count": int(reference_values.size),
        "alternative": alternative,
        "n": sample_size,
        "seed": seed,
        "status": "ok",
        "reference_values": reference_values,
        "finite_reference_values": finite_values,
    }

    if not np.isfinite(observed_value):
        result["status"] = "observed_undefined"
        return result

    if finite_values.size == 0:
        result["status"] = "reference_all_undefined"
        return result

    reference_mean = float(np.mean(finite_values))
    reference_std = float(np.std(finite_values))
    reference_median = float(np.median(finite_values))
    q025, q05, q95, q975 = np.quantile(finite_values, [0.025, 0.05, 0.95, 0.975])

    if alternative == "greater":
        p_value = float(np.mean(finite_values >= observed_value))
    elif alternative == "less":
        p_value = float(np.mean(finite_values <= observed_value))
    else:
        center = reference_mean
        p_value = float(np.mean(np.abs(finite_values - center) >= np.abs(observed_value - center)))

    result.update(
        {
            "reference_mean": reference_mean,
            "reference_std": reference_std,
            "reference_median": reference_median,
            "q025": float(q025),
            "q05": float(q05),
            "q95": float(q95),
            "q975": float(q975),
            "p_value": p_value,
        }
    )
    return result


def simulation_match_test_batched(
    observed_cm: np.ndarray,
    reference_probs: np.ndarray,
    metric_func,
    n: int | None = None,
    num_samples: int = 10000,
    alternative: Alternative = "two-sided",
    seed: int | None = 12345,
    *,
    batch_size: int = 2000,
    should_cancel: Callable[[], bool] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, object]:
    """Run a simulation-based MATCH test in batches with optional cancellation."""

    observed = _require_single_cm(observed_cm, "observed_cm").astype(np.float64)
    probs = validate_reference_probs(reference_probs)
    sample_size = _n(observed) if n is None else int(n)
    if sample_size < 0:
        raise ValueError("n must be non-negative")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if alternative not in {"two-sided", "less", "greater"}:
        raise ValueError("alternative must be 'two-sided', 'less', or 'greater'")

    observed_value = float(np.asarray(metric_func(observed), dtype=np.float64))
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    rng = np.random.default_rng(seed)
    value_batches: list[np.ndarray] = []
    produced = 0

    while produced < num_samples:
        if should_cancel is not None and should_cancel():
            return {
                "observed_value": observed_value,
                "reference_mean": float("nan"),
                "reference_std": float("nan"),
                "reference_median": float("nan"),
                "q025": float("nan"),
                "q05": float("nan"),
                "q95": float("nan"),
                "q975": float("nan"),
                "p_value": float("nan"),
                "undefined_rate": float("nan"),
                "finite_sample_count": 0,
                "total_sample_count": produced,
                "alternative": alternative,
                "n": sample_size,
                "seed": seed,
                "status": "cancelled",
                "reference_values": np.array([], dtype=np.float64),
                "finite_reference_values": np.array([], dtype=np.float64),
            }

        current_batch = min(batch_size, num_samples - produced)
        samples = multinomial_reference_samples(probs, sample_size, current_batch, rng)
        value_batches.append(metric_distribution(samples, metric_func))
        produced += current_batch

        if progress_callback is not None:
            progress_callback(produced, num_samples)

    reference_values = np.concatenate(value_batches) if value_batches else np.array([], dtype=np.float64)
    return _summarize_simulation_match_result(
        observed_value,
        reference_values,
        alternative=alternative,
        sample_size=sample_size,
        seed=seed,
    )


if __name__ == "__main__":
    # Small smoke tests and approximation cross-checks.
    ref = create_cm(tp=45, fn=10, fp=15, tn=30)  # accuracy probability = 0.75
    obs = create_cm(tp=48, fn=8, fp=12, tn=32)   # accuracy = 0.80

    for result in compare_match_methods(obs, ref, "accuracy"):
        print(result.asdict())

    print("\nMarginal benefit")
    for result in compare_match_methods(obs, ref, "marginal_benefit"):
        print(result.asdict())

    print("\nPPV")
    for result in compare_match_methods(obs, ref, "positive_predictive_value"):
        print(result.asdict())
