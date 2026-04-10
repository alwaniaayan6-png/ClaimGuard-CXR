"""Temporal shift experiment for ClaimGuard-CXR.

Tests whether conformal FDR guarantees hold when the model is calibrated on
early admission dates and evaluated on later ones (prospective deployment
simulation).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from verifact.inference.conformal_triage import (
    ConformalClaimTriage,
    compute_fdr,
)

logger = logging.getLogger(__name__)


def temporal_split_experiment(
    df: pd.DataFrame,
    scores: np.ndarray,
    labels: np.ndarray,
    pathology_groups: np.ndarray,
    report_ids: np.ndarray,
    date_column: str,
    early_cutoff: str,
    late_cutoff: str,
    alpha: float = 0.05,
    min_group_size: int = 50,
) -> dict:
    """Calibrate on early admissions, evaluate on late admissions.

    Splits claims into two temporal cohorts:

    - **Early cohort** (admission date < ``early_cutoff``): used for
      conformal calibration.
    - **Late cohort** (admission date >= ``late_cutoff``): held-out test set.

    Reports FDR and green fraction at the nominal ``alpha`` level for both
    cohorts, and flags coverage degradation if the late-cohort FDR exceeds
    ``alpha``.

    Args:
        df: DataFrame with one row per claim.  Must contain ``date_column``.
        scores: Verifier faithfulness scores aligned with ``df`` rows.
        labels: Ground-truth binary labels (0=Faithful, 1=Unfaithful),
            aligned with ``df`` rows.
        pathology_groups: Pathology category per claim, aligned with ``df``.
        report_ids: Report ID per claim, aligned with ``df``.
        date_column: Name of the datetime column in ``df``.
        early_cutoff: ISO date string (e.g. ``"2020-01-01"``).  Claims with
            date strictly before this form the calibration set.
        late_cutoff: ISO date string.  Claims with date >= this form the
            test set.  Claims between the two cutoffs are discarded.
        alpha: Target FDR level.  Default 0.05.
        min_group_size: Minimum calibration claims per pathology group before
            pooling into Rare/Other.  Default 50 (relaxed from production
            value to tolerate smaller temporal subsets).

    Returns:
        dict with keys:

        - ``"n_early"`` / ``"n_late"`` — split sizes
        - ``"early_fdr"`` / ``"late_fdr"`` — observed FDR
        - ``"early_green_fraction"`` / ``"late_green_fraction"``
        - ``"nominal_alpha"`` — the requested alpha
        - ``"coverage_degraded"`` — True if late FDR > alpha
        - ``"fdr_delta"`` — late_fdr - early_fdr (positive = degradation)
        - ``"calibration_results"`` — raw :class:`CalibrationResult` dict
    """
    dates = pd.to_datetime(df[date_column])
    early_cut = pd.Timestamp(early_cutoff)
    late_cut = pd.Timestamp(late_cutoff)

    early_mask = (dates < early_cut).to_numpy()
    late_mask = (dates >= late_cut).to_numpy()

    n_early = int(early_mask.sum())
    n_late = int(late_mask.sum())
    n_discarded = len(df) - n_early - n_late

    logger.info(
        f"Temporal split: {n_early} early claims (< {early_cutoff}), "
        f"{n_late} late claims (>= {late_cutoff}), "
        f"{n_discarded} discarded (between cutoffs)"
    )

    if n_early == 0:
        raise ValueError(f"No claims before early_cutoff={early_cutoff!r}")
    if n_late == 0:
        raise ValueError(f"No claims from late_cutoff={late_cutoff!r} onward")

    # Calibrate on early cohort
    triage = ConformalClaimTriage(
        alpha=alpha,
        min_group_size=min_group_size,
    )
    cal_results = triage.calibrate(
        scores=scores[early_mask],
        labels=labels[early_mask],
        pathology_groups=pathology_groups[early_mask],
        report_ids=report_ids[early_mask],
    )

    # Evaluate on early cohort (in-distribution)
    early_results = triage.triage(
        scores=scores[early_mask],
        pathology_groups=pathology_groups[early_mask],
    )
    early_fdr_info = compute_fdr(early_results, labels[early_mask])
    n_early_green = sum(1 for r in early_results if r.is_accepted)
    early_green_frac = n_early_green / n_early if n_early > 0 else 0.0

    # Evaluate on late cohort (temporal shift)
    late_results = triage.triage(
        scores=scores[late_mask],
        pathology_groups=pathology_groups[late_mask],
    )
    late_fdr_info = compute_fdr(late_results, labels[late_mask])
    n_late_green = sum(1 for r in late_results if r.is_accepted)
    late_green_frac = n_late_green / n_late if n_late > 0 else 0.0

    early_fdr = early_fdr_info["fdr"]
    late_fdr = late_fdr_info["fdr"]
    fdr_delta = late_fdr - early_fdr
    coverage_degraded = late_fdr > alpha

    logger.info(
        f"Early FDR={early_fdr:.4f}, Late FDR={late_fdr:.4f}, "
        f"delta={fdr_delta:+.4f}, degraded={coverage_degraded}"
    )

    return {
        "n_early": n_early,
        "n_late": n_late,
        "early_fdr": early_fdr,
        "late_fdr": late_fdr,
        "early_green_fraction": early_green_frac,
        "late_green_fraction": late_green_frac,
        "nominal_alpha": alpha,
        "coverage_degraded": coverage_degraded,
        "fdr_delta": fdr_delta,
        "calibration_results": cal_results,
    }
