"""Krippendorff's α with bootstrap confidence intervals.

Task 1 + Task 8 dependency.  Implements Krippendorff (2004, 2011,
2018) reliability coefficient via the *coincidence-matrix* formulation.
Supports nominal, ordinal, interval, and ratio difference metrics.
The ordinal path is the primary one for ClaimGuard-CXR — our
5-label grader scale (SUPPORTED, CONTRADICTED, NOVEL_PLAUSIBLE,
NOVEL_HALLUCINATED, UNCERTAIN) is treated as ordered
{0, 1, 2, 3, 4} after encoding.

Why Krippendorff's α and not Cohen's κ?
---------------------------------------
* Handles any number of coders (κ is 2-coder only; Fleiss's κ extends
  to k but assumes all coders rate every unit).
* Handles missing values natively.
* Supports ordinal / interval metrics so near-miss disagreements
  (e.g., SUPPORTED vs NOVEL_PLAUSIBLE) count less than far ones
  (e.g., SUPPORTED vs CONTRADICTED).
* Is the ASA/ACL consensus standard for multi-coder reliability.

Public API
----------

    alpha(data, level="ordinal") -> float
    alpha_with_bootstrap_ci(
        data, level="ordinal", n_bootstrap=1000, seed=42
    ) -> tuple[float, float, float]

``data`` is a 2-D array-like of shape ``(n_coders, n_units)``.
Missing values can be encoded as ``np.nan`` (for float arrays) or
``None`` (for object arrays).  The implementation rejects units with
fewer than 2 valid codings (pair-counting requires pairs).

Bootstrap CI uses percentile bootstrap over *units* (resampling
columns with replacement), which is the standard protocol (Hayes &
Krippendorff 2007) and respects the unit-level exchangeability
assumption.
"""

from __future__ import annotations

import logging
from typing import Iterable, Literal, Optional, Sequence, Union

import numpy as np

logger = logging.getLogger(__name__)

Level = Literal["nominal", "ordinal", "interval", "ratio"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_float_matrix(
    data: Union[np.ndarray, Sequence[Sequence]],
) -> np.ndarray:
    """Cast a ragged reliability matrix to float64 with ``np.nan`` for missing.

    Accepts float arrays directly, object arrays (with ``None``),
    Python lists of lists, and 2-D numpy arrays.  Returns
    ``shape = (n_coders, n_units)``.
    """
    if isinstance(data, np.ndarray) and data.dtype.kind in "fiu":
        return data.astype(np.float64)
    # Fall back to element-wise conversion handling None → nan.
    rows: list[list[float]] = []
    for row in data:
        converted: list[float] = []
        for v in row:
            if v is None:
                converted.append(np.nan)
            else:
                try:
                    converted.append(float(v))
                except (TypeError, ValueError):
                    converted.append(np.nan)
        rows.append(converted)
    return np.asarray(rows, dtype=np.float64)


def _value_index(
    matrix: np.ndarray,
) -> tuple[np.ndarray, dict[float, int]]:
    """Enumerate the sorted unique observed values.

    Returns the sorted-unique values array and a dict mapping each
    value to its row/column index in the coincidence matrix.
    """
    valid = matrix[~np.isnan(matrix)]
    if valid.size == 0:
        return np.array([], dtype=np.float64), {}
    values = np.sort(np.unique(valid))
    value_to_idx = {float(v): i for i, v in enumerate(values)}
    return values, value_to_idx


def _coincidence_matrix(
    matrix: np.ndarray,
    value_to_idx: dict[float, int],
) -> np.ndarray:
    """Build the Krippendorff coincidence matrix ``o[c, k]``.

    For each unit (column) with ``m_u ≥ 2`` valid codings, every
    ordered pair ``(v_i, v_j)`` (``i ≠ j``) contributes
    ``1 / (m_u - 1)`` to ``o[v_i, v_k]``.  The result is a symmetric
    non-negative matrix whose total sum is ``n`` (the total number
    of pairable observations).
    """
    k = len(value_to_idx)
    coincidence = np.zeros((k, k), dtype=np.float64)
    n_coders, n_units = matrix.shape
    for u in range(n_units):
        col = matrix[:, u]
        valid_mask = ~np.isnan(col)
        m_u = int(valid_mask.sum())
        if m_u < 2:
            continue
        codings = col[valid_mask]
        idx = np.array(
            [value_to_idx[float(c)] for c in codings], dtype=np.int64
        )
        scale = 1.0 / (m_u - 1)
        # Add the outer sum without the diagonal self-contributions.
        for i in range(m_u):
            for j in range(m_u):
                if i == j:
                    continue
                coincidence[idx[i], idx[j]] += scale
    return coincidence


def _delta_squared(
    level: Level,
    values: np.ndarray,
    n_c: np.ndarray,
) -> np.ndarray:
    """Build the squared-difference metric matrix for a given level.

    * ``nominal``: δ² = 1 iff c ≠ k, else 0.
    * ``ordinal``: Krippendorff (2004) rank-based metric
        δ²_ck = (Σ_{g ∈ [min,max]} n_g  −  (n_c + n_k) / 2)²
      where the sum is over all ordered values from ``min(c, k)``
      to ``max(c, k)`` inclusive.
    * ``interval``: δ² = (v_c − v_k)².
    * ``ratio``: δ² = ((v_c − v_k) / (v_c + v_k))².
    """
    k = len(values)
    delta_sq = np.zeros((k, k), dtype=np.float64)
    if level == "nominal":
        for i in range(k):
            for j in range(k):
                if i != j:
                    delta_sq[i, j] = 1.0
    elif level == "ordinal":
        for i in range(k):
            for j in range(k):
                if i == j:
                    continue
                lo, hi = (i, j) if i < j else (j, i)
                s = float(n_c[lo:hi + 1].sum())
                s -= (float(n_c[lo]) + float(n_c[hi])) / 2.0
                delta_sq[i, j] = s * s
    elif level == "interval":
        for i in range(k):
            for j in range(k):
                d = float(values[i]) - float(values[j])
                delta_sq[i, j] = d * d
    elif level == "ratio":
        for i in range(k):
            for j in range(k):
                denom = float(values[i]) + float(values[j])
                if denom == 0:
                    delta_sq[i, j] = 0.0
                else:
                    d = (float(values[i]) - float(values[j])) / denom
                    delta_sq[i, j] = d * d
    else:
        raise ValueError(
            f"Unknown level {level!r}; expected one of "
            f"nominal/ordinal/interval/ratio"
        )
    return delta_sq


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def alpha(
    data: Union[np.ndarray, Sequence[Sequence]],
    level: Level = "ordinal",
) -> float:
    """Compute Krippendorff's α on a reliability data matrix.

    Args:
        data: Array-like of shape ``(n_coders, n_units)``.  Missing
            values encoded as ``np.nan`` or ``None``.
        level: Difference metric — one of ``nominal`` / ``ordinal`` /
            ``interval`` / ``ratio`` (default ``"ordinal"``, which is
            the right metric for our 5-label grader scale).

    Returns:
        Krippendorff's α as a ``float``.  The value lives in
        ``(-∞, 1]`` in theory; 1.0 means perfect agreement, 0 means
        agreement is at chance, negatives are systematic disagreement.
        Returns ``1.0`` in degenerate cases where the expected
        disagreement is zero (e.g., all coders gave the same value
        across all valid units).

    Raises:
        ValueError: if the data has fewer than 2 valid units or if
            ``level`` is not recognised.
    """
    matrix = _to_float_matrix(data)
    if matrix.ndim != 2:
        raise ValueError(
            f"data must be 2-D; got shape {matrix.shape}"
        )
    n_coders, n_units = matrix.shape
    if n_coders < 2:
        raise ValueError(
            f"need ≥ 2 coders for reliability; got {n_coders}"
        )

    values, value_to_idx = _value_index(matrix)
    if len(values) == 0:
        raise ValueError("no valid codings in data")

    coincidence = _coincidence_matrix(matrix, value_to_idx)
    n_c = coincidence.sum(axis=1)
    n_total = float(n_c.sum())
    if n_total < 2:
        raise ValueError(
            "need at least 2 pairable observations; got "
            f"{n_total} (did every unit have <2 valid coders?)"
        )

    delta_sq = _delta_squared(level, values, n_c)

    observed = float((coincidence * delta_sq).sum())
    # Expected disagreement: sum_{c,k} (n_c * n_k - [c==k] * n_c) * δ²
    #   In Krippendorff's formulation the "−[c==k] n_c" correction
    #   comes out of the (n × (n − 1)) normalisation; since δ² is
    #   zero on the diagonal it drops anyway.
    expected_num = 0.0
    for i in range(len(values)):
        for j in range(len(values)):
            expected_num += float(n_c[i]) * float(n_c[j]) * delta_sq[i, j]
    expected = expected_num / (n_total - 1.0)

    if expected == 0.0:
        # No variability at all ⇒ perfect agreement by definition.
        return 1.0
    return 1.0 - observed / expected


def alpha_with_bootstrap_ci(
    data: Union[np.ndarray, Sequence[Sequence]],
    level: Level = "ordinal",
    n_bootstrap: int = 1000,
    seed: int = 42,
    ci: float = 0.95,
) -> tuple[float, float, float]:
    """Compute α with a percentile bootstrap CI by resampling units.

    Hayes & Krippendorff (2007) recommend resampling at the *unit*
    level (columns) with replacement.  This preserves the per-unit
    coder structure (which coders rated which unit) and respects the
    unit-level exchangeability of the data.

    Args:
        data: Reliability matrix ``(n_coders, n_units)``.
        level: Metric (default ``"ordinal"``).
        n_bootstrap: Number of bootstrap replicates (default 1000).
        seed: RNG seed for reproducibility (default 42).
        ci: Desired two-sided confidence (default 0.95 ⇒ 2.5/97.5).

    Returns:
        Tuple ``(alpha, ci_low, ci_high)``.  The point estimate is
        the α on the full data, not the bootstrap mean.
    """
    matrix = _to_float_matrix(data)
    n_coders, n_units = matrix.shape
    point = alpha(matrix, level=level)
    if n_units < 2:
        return point, float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    replicates: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n_units, size=n_units)
        resampled = matrix[:, idx]
        try:
            replicates.append(alpha(resampled, level=level))
        except ValueError:
            continue
    if not replicates:
        return point, float("nan"), float("nan")
    arr = np.asarray(replicates, dtype=np.float64)
    lo = float(np.percentile(arr, (1 - ci) / 2 * 100))
    hi = float(np.percentile(arr, (1 + ci) / 2 * 100))
    return point, lo, hi


__all__ = [
    "alpha",
    "alpha_with_bootstrap_ci",
]
