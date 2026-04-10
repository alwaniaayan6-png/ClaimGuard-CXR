"""CoFact-inspired adaptive conformal FDR control for ClaimGuard-CXR v2.

Replaces the v1 inverted cfBH procedure with a density-ratio-reweighted
variant that handles distribution shift without exchangeability.

Key design decision: density ratios are estimated on PENULTIMATE HIDDEN
REPRESENTATIONS (256-dim from DeBERTa verdict head), NOT on 1D softmax
scores. Neural softmax outputs are spiky and clustered — density estimation
on them will explode to infinity for OOD scores, breaking FDR guarantees.
The 256-dim representation has far richer distributional structure.

Pipeline:
  1. Extract 256-dim hidden representations from DeBERTa for all cal + test claims
  2. PCA to 32-dim for stable density ratio estimation
  3. Train logistic regression to distinguish cal vs test in PCA space
  4. Convert classifier probs to density ratios via Bayes rule
  5. Reweight calibration p-values by density ratios
  6. Apply global BH on reweighted p-values

Falls back to v1 inverted cfBH when distributions are similar (KS test on
PCA projections).

Reference: CoFact (ICLR 2026) — online density ratio estimation for
conformal factuality guarantees under distribution shift.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class ConformalResult:
    """Result of conformal FDR control procedure."""
    green_mask: np.ndarray       # (n_test,) bool — claims labeled GREEN (safe)
    yellow_mask: np.ndarray      # (n_test,) bool — claims labeled YELLOW (review)
    red_mask: np.ndarray         # (n_test,) bool — claims labeled RED (reject)
    p_values: np.ndarray         # (n_test,) conformal p-values
    density_ratios: np.ndarray   # (n_cal,) estimated density ratios (1.0 if no shift)
    n_green: int
    n_yellow: int
    n_red: int
    fdr_estimate: float          # observed FDR among green claims
    power_estimate: float        # fraction of true not-contra in green
    method: str                  # "cofact" or "inverted_cfbh" (fallback)
    ks_pvalue: float             # KS test p-value for distribution similarity


def estimate_density_ratio_hidden(
    cal_hidden: np.ndarray,
    test_hidden: np.ndarray,
    pca_dim: int = 32,
    clip_max: float = 10.0,
) -> np.ndarray:
    """Estimate density ratio w(h) = p_test(h) / p_cal(h) on hidden representations.

    Uses PCA dimensionality reduction + logistic regression, which is far more
    stable than doing density estimation on 1D softmax scores.

    Args:
        cal_hidden: (n_cal, hidden_dim) penultimate hidden representations for cal claims.
        test_hidden: (n_test, hidden_dim) penultimate hidden representations for test claims.
        pca_dim: PCA dimension for stable estimation (default 32).
        clip_max: Maximum allowed density ratio for stability.

    Returns:
        (n_cal,) density ratios for each calibration claim.
    """
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    n_cal = len(cal_hidden)
    n_test = len(test_hidden)

    # Stack and PCA reduce
    all_hidden = np.vstack([cal_hidden, test_hidden])

    # Standardize before PCA
    scaler = StandardScaler()
    all_hidden_scaled = scaler.fit_transform(all_hidden)

    # PCA to pca_dim
    effective_dim = min(pca_dim, all_hidden_scaled.shape[1], all_hidden_scaled.shape[0] - 1)
    pca = PCA(n_components=effective_dim, random_state=42)
    all_pca = pca.fit_transform(all_hidden_scaled)

    X_cal_pca = all_pca[:n_cal]
    X_test_pca = all_pca[n_cal:]

    # Binary classification: 0 = calibration, 1 = test
    X = np.vstack([X_cal_pca, X_test_pca])
    y = np.concatenate([np.zeros(n_cal), np.ones(n_test)])

    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    clf.fit(X, y)

    # Density ratio via Bayes: w(h) = [P(test|h) / P(cal|h)] * [n_cal / n_test]
    cal_probs = clf.predict_proba(X_cal_pca)  # (n_cal, 2)
    p_test_given_h = np.clip(cal_probs[:, 1], 1e-8, None)
    p_cal_given_h = np.clip(cal_probs[:, 0], 1e-8, None)

    ratio = (p_test_given_h / p_cal_given_h) * (n_cal / n_test)
    ratio = np.clip(ratio, 1e-4, clip_max)

    logger.info(
        f"Density ratio stats: mean={ratio.mean():.3f}, "
        f"std={ratio.std():.3f}, min={ratio.min():.3f}, max={ratio.max():.3f}, "
        f"PCA explained variance={pca.explained_variance_ratio_.sum():.3f}"
    )

    return ratio


def _ks_test_hidden(
    cal_hidden: np.ndarray,
    test_hidden: np.ndarray,
) -> float:
    """KS-like test for multivariate distribution shift on hidden representations.

    Projects to first PCA component and runs 1D KS test. Not a formal
    multivariate test, but a fast heuristic for shift detection.

    Returns:
        p-value (low = distributions differ).
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    all_hidden = np.vstack([cal_hidden, test_hidden])
    scaler = StandardScaler()
    all_scaled = scaler.fit_transform(all_hidden)

    pca = PCA(n_components=1, random_state=42)
    all_pca = pca.fit_transform(all_scaled).ravel()

    cal_proj = all_pca[:len(cal_hidden)]
    test_proj = all_pca[len(cal_hidden):]

    _, p_value = stats.ks_2samp(cal_proj, test_proj)
    return float(p_value)


def inverted_cfbh_standard(
    cal_contra_scores: np.ndarray,
    test_scores: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, int]:
    """V1 inverted cfBH (no density ratio reweighting). Fallback method.

    Null H0: "test claim j is contradicted."
    Calibration: contradicted calibration claims.
    p-value: (|{cal_contra >= s_j}| + 1) / (n_cal + 1)
    Global BH at level alpha.
    """
    n_cal = len(cal_contra_scores)
    p_values = np.zeros(len(test_scores))

    for j, s_j in enumerate(test_scores):
        p_values[j] = (np.sum(cal_contra_scores >= s_j) + 1) / (n_cal + 1)

    # BH procedure
    n_test = len(test_scores)
    sorted_pvals = np.sort(p_values)

    k_star = 0
    for k in range(1, n_test + 1):
        if sorted_pvals[k - 1] <= k * alpha / n_test:
            k_star = k

    return p_values, k_star


def cofact_cfbh_hidden(
    cal_contra_scores: np.ndarray,
    cal_contra_hidden: np.ndarray,
    test_scores: np.ndarray,
    test_hidden: np.ndarray,
    alpha: float,
    pca_dim: int = 32,
    clip_max: float = 10.0,
) -> tuple[np.ndarray, int, np.ndarray]:
    """CoFact density-ratio-reweighted cfBH using hidden representations.

    Density ratios estimated on penultimate hidden layer (not softmax scores).
    p-values still computed using softmax scores but weighted by hidden ratios.

    Args:
        cal_contra_scores: (n_cal,) softmax scores for contradicted cal claims.
        cal_contra_hidden: (n_cal, hidden_dim) penultimate hidden for cal claims.
        test_scores: (n_test,) softmax scores for all test claims.
        test_hidden: (n_test, hidden_dim) penultimate hidden for test claims.
        alpha: Target FDR level.
        pca_dim: PCA dimension for density estimation.
        clip_max: Maximum density ratio.

    Returns:
        Tuple of (p_values, k_star, density_ratios).
    """
    # Estimate density ratios on hidden representations
    density_ratios = estimate_density_ratio_hidden(
        cal_contra_hidden, test_hidden,
        pca_dim=pca_dim, clip_max=clip_max,
    )

    total_weight = np.sum(density_ratios)

    # Compute reweighted p-values (using softmax scores, weighted by hidden ratios)
    p_values = np.zeros(len(test_scores))
    for j, s_j in enumerate(test_scores):
        weighted_count = np.sum(density_ratios[cal_contra_scores >= s_j])
        p_values[j] = (weighted_count + 1) / (total_weight + 1)

    # BH procedure
    n_test = len(test_scores)
    sorted_pvals = np.sort(p_values)

    k_star = 0
    for k in range(1, n_test + 1):
        if sorted_pvals[k - 1] <= k * alpha / n_test:
            k_star = k

    return p_values, k_star, density_ratios


def adaptive_conformal_triage(
    cal_contra_scores: np.ndarray,
    test_scores: np.ndarray,
    cal_contra_hidden: Optional[np.ndarray] = None,
    test_hidden: Optional[np.ndarray] = None,
    test_labels: Optional[np.ndarray] = None,
    alpha: float = 0.05,
    tau_low: float = 0.3,
    ks_threshold: float = 0.05,
    clip_max: float = 10.0,
) -> ConformalResult:
    """Run adaptive conformal FDR control with automatic method selection.

    1. If hidden representations provided: KS test on hidden PCA projections
    2. If shift detected: CoFact density-ratio reweighting on hidden space
    3. If no shift or no hidden: fallback to v1 inverted cfBH

    Args:
        cal_contra_scores: (n_cal,) softmax scores for contradicted cal claims.
        test_scores: (n_test,) softmax scores for all test claims.
        cal_contra_hidden: (n_cal, hidden_dim) optional penultimate hidden for cal.
        test_hidden: (n_test, hidden_dim) optional penultimate hidden for test.
        test_labels: (n_test,) optional ground truth (0=not-contra, 1=contra).
        alpha: Target FDR level.
        tau_low: YELLOW/RED boundary threshold.
        ks_threshold: KS test p-value threshold for shift detection.
        clip_max: Maximum density ratio.

    Returns:
        ConformalResult with green/yellow/red masks and diagnostics.
    """
    n_test = len(test_scores)
    use_cofact = False
    ks_pvalue = 1.0

    # Determine method based on hidden representation shift
    if cal_contra_hidden is not None and test_hidden is not None:
        ks_pvalue = _ks_test_hidden(cal_contra_hidden, test_hidden)
        logger.info(f"Hidden-space KS test: p={ks_pvalue:.4f}")

        if ks_pvalue <= ks_threshold:
            use_cofact = True
            logger.info(
                f"Distribution shift detected (KS p={ks_pvalue:.4f} <= {ks_threshold}). "
                f"Using CoFact density-ratio reweighting on hidden representations."
            )
        else:
            logger.info(
                f"No significant shift (KS p={ks_pvalue:.4f} > {ks_threshold}). "
                f"Using standard inverted cfBH."
            )
    else:
        logger.info(
            "No hidden representations provided. Using standard inverted cfBH. "
            "For CoFact, pass cal_contra_hidden and test_hidden."
        )

    # Run selected method
    if use_cofact:
        method = "cofact_hidden"
        p_values, k_star, density_ratios = cofact_cfbh_hidden(
            cal_contra_scores, cal_contra_hidden,
            test_scores, test_hidden,
            alpha, clip_max=clip_max,
        )
    else:
        method = "inverted_cfbh"
        p_values, k_star = inverted_cfbh_standard(
            cal_contra_scores, test_scores, alpha
        )
        density_ratios = np.ones(len(cal_contra_scores))

    # Determine BH threshold
    if k_star > 0:
        sorted_pvals = np.sort(p_values)
        bh_threshold = sorted_pvals[k_star - 1]
    else:
        bh_threshold = 0.0
        logger.warning(f"BH k*=0 at alpha={alpha}. No claims labeled GREEN.")

    # Assign triage labels
    green_mask = p_values <= bh_threshold if k_star > 0 else np.zeros(n_test, dtype=bool)
    red_mask = (~green_mask) & (test_scores <= tau_low)
    yellow_mask = (~green_mask) & (~red_mask)

    n_green = int(green_mask.sum())
    n_yellow = int(yellow_mask.sum())
    n_red = int(red_mask.sum())

    # Compute FDR and power if labels available
    fdr_estimate = 0.0
    power_estimate = 0.0
    if test_labels is not None and n_green > 0:
        green_labels = test_labels[green_mask]
        fdr_estimate = float(np.mean(green_labels == 1))
        true_not_contra = test_labels == 0
        if true_not_contra.sum() > 0:
            power_estimate = float(green_mask[true_not_contra].mean())

    logger.info(
        f"alpha={alpha}: n_green={n_green}, n_yellow={n_yellow}, n_red={n_red}, "
        f"FDR={fdr_estimate:.4f}, power={power_estimate:.4f}, method={method}"
    )

    return ConformalResult(
        green_mask=green_mask,
        yellow_mask=yellow_mask,
        red_mask=red_mask,
        p_values=p_values,
        density_ratios=density_ratios,
        n_green=n_green,
        n_yellow=n_yellow,
        n_red=n_red,
        fdr_estimate=fdr_estimate,
        power_estimate=power_estimate,
        method=method,
        ks_pvalue=float(ks_pvalue),
    )


def run_conformal_sweep(
    cal_contra_scores: np.ndarray,
    test_scores: np.ndarray,
    cal_contra_hidden: Optional[np.ndarray] = None,
    test_hidden: Optional[np.ndarray] = None,
    test_labels: Optional[np.ndarray] = None,
    alpha_levels: Optional[list[float]] = None,
    tau_low: float = 0.3,
    ks_threshold: float = 0.05,
) -> dict[float, ConformalResult]:
    """Run conformal triage at multiple alpha levels.

    Args:
        cal_contra_scores: Calibration contradicted softmax scores.
        test_scores: Test softmax scores.
        cal_contra_hidden: Optional calibration hidden representations.
        test_hidden: Optional test hidden representations.
        test_labels: Optional test labels.
        alpha_levels: List of FDR targets.
        tau_low: YELLOW/RED threshold.
        ks_threshold: KS test threshold.

    Returns:
        Dict mapping alpha -> ConformalResult.
    """
    if alpha_levels is None:
        alpha_levels = [0.01, 0.05, 0.10, 0.15, 0.20]

    results = {}
    for alpha in alpha_levels:
        results[alpha] = adaptive_conformal_triage(
            cal_contra_scores=cal_contra_scores,
            test_scores=test_scores,
            cal_contra_hidden=cal_contra_hidden,
            test_hidden=test_hidden,
            test_labels=test_labels,
            alpha=alpha,
            tau_low=tau_low,
            ks_threshold=ks_threshold,
        )

    return results
