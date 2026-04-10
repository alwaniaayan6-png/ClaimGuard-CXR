"""Conformal coverage diagnostic curves for ClaimGuard-CXR.

Plots the empirical FDR and green-claim fraction as a function of the
conformal target level alpha, together with the theoretical guarantee
(FDR <= alpha diagonal).
"""

from __future__ import annotations

from typing import List

import matplotlib.pyplot as plt
import numpy as np

from verifact.inference.conformal_triage import (
    ConformalClaimTriage,
    TriageResult,
    compute_fdr,
)


def plot_coverage_curves(
    triage_results: list[TriageResult],
    gt_labels: np.ndarray,
    alpha_values: List[float] = None,
) -> dict:
    """Plot conformal coverage curves across a range of alpha levels.

    For each alpha in ``alpha_values``, re-applies the BH procedure at that
    level (using the conformal p-values already embedded in ``triage_results``)
    and records:

    - **green fraction** — fraction of claims accepted at that alpha.
    - **observed FDR**   — empirical FDR among accepted claims.

    Also draws the guarantee diagonal (FDR = alpha) to make it visually obvious
    whether the procedure is conservative or liberal.

    Args:
        triage_results: List of :class:`TriageResult` from
            :meth:`ConformalClaimTriage.triage`.  Each result must carry a
            valid ``conformal_pvalue``.
        gt_labels: Ground-truth binary array (0=Faithful, 1=Unfaithful),
            same length as ``triage_results``.
        alpha_values: Target FDR levels to sweep over.  Defaults to
            ``[0.01, 0.025, 0.05, 0.10, 0.15, 0.20]``.

    Returns:
        dict with keys:

        - ``"fig"``           — :class:`matplotlib.figure.Figure`
        - ``"alpha_values"``  — list of alpha values used
        - ``"green_fractions"`` — observed green fraction at each alpha
        - ``"observed_fdrs"`` — observed FDR at each alpha
        - ``"guarantee_line"`` — same as alpha_values (the diagonal)
    """
    if alpha_values is None:
        alpha_values = [0.01, 0.025, 0.05, 0.10, 0.15, 0.20]

    gt_arr = np.asarray(gt_labels, dtype=int)
    pvalues = np.array([r.conformal_pvalue for r in triage_results])
    n = len(triage_results)

    green_fractions: list[float] = []
    observed_fdrs: list[float] = []

    for alpha in alpha_values:
        # Re-run BH at this alpha using stored p-values.
        # BH: sort p-values, accept all p_(k) <= (k/n)*alpha for the
        # largest k satisfying the threshold.
        sorted_idx = np.argsort(pvalues)
        sorted_pv = pvalues[sorted_idx]
        ranks = np.arange(1, n + 1)
        bh_thresh = (ranks / n) * alpha
        below = sorted_pv <= bh_thresh

        if below.any():
            k_star = int(np.max(np.where(below)[0]))
            accepted = np.zeros(n, dtype=bool)
            accepted[sorted_idx[: k_star + 1]] = True
        else:
            accepted = np.zeros(n, dtype=bool)

        n_green = int(accepted.sum())
        green_fractions.append(n_green / n)

        if n_green > 0:
            n_false = int((gt_arr[accepted] == 1).sum())
            fdr = n_false / n_green
        else:
            fdr = 0.0
        observed_fdrs.append(fdr)

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("Conformal Coverage Diagnostics", fontsize=13, fontweight="bold")

    # Panel 1: Green fraction vs alpha
    ax0 = axes[0]
    ax0.plot(alpha_values, green_fractions, marker="o", color="#2196F3",
             linewidth=2, markersize=6, label="Green fraction")
    ax0.set_xlabel("Target FDR level (α)", fontsize=11)
    ax0.set_ylabel("Fraction of green claims", fontsize=11)
    ax0.set_title("Green Fraction vs α", fontsize=11)
    ax0.set_xlim(0, max(alpha_values) * 1.05)
    ax0.set_ylim(0, 1)
    ax0.grid(True, alpha=0.3)
    ax0.legend(fontsize=10)

    # Panel 2: Observed FDR vs alpha with guarantee diagonal
    ax1 = axes[1]
    ax1.plot(alpha_values, alpha_values, linestyle="--", color="#9E9E9E",
             linewidth=1.5, label="Guarantee (FDR = α)")
    ax1.plot(alpha_values, observed_fdrs, marker="s", color="#F44336",
             linewidth=2, markersize=6, label="Observed FDR")
    ax1.fill_between(alpha_values, 0, alpha_values, alpha=0.08, color="#9E9E9E",
                     label="Valid region")
    ax1.set_xlabel("Target FDR level (α)", fontsize=11)
    ax1.set_ylabel("Observed FDR among green claims", fontsize=11)
    ax1.set_title("Observed FDR vs Guarantee", fontsize=11)
    ax1.set_xlim(0, max(alpha_values) * 1.05)
    ax1.set_ylim(0, max(max(observed_fdrs), max(alpha_values)) * 1.15)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    plt.tight_layout()

    return {
        "fig": fig,
        "alpha_values": alpha_values,
        "green_fractions": green_fractions,
        "observed_fdrs": observed_fdrs,
        "guarantee_line": alpha_values,
    }
