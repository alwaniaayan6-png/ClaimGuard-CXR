"""Conformal Claim Triage for ClaimGuard-CXR (Contribution 3).

Implements the cfBH procedure (Jin & Candes, JMLR 2023) for controlling
false discovery rate among accepted (green-labeled) claims, with
pathology-group-stratified thresholds.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class TriageResult:
    """Result of conformal triage for a single claim."""
    claim_text: str
    pathology_group: str
    faithfulness_score: float
    conformal_pvalue: float
    label: Literal["green", "yellow", "red"]
    is_accepted: bool  # True if green


@dataclass
class CalibrationResult:
    """Result of conformal calibration on a pathology group."""
    group_name: str
    n_calibration_claims: int
    alpha: float
    is_pooled: bool  # True if group was too small and pooled into Rare/Other
    calibration_scores: np.ndarray = field(repr=False)


def subsample_one_per_report(
    scores: np.ndarray,
    labels: np.ndarray,
    report_ids: np.ndarray,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Subsample one claim per report to ensure exchangeability.

    Within-report claims share the same image/patient/radiologist and violate
    exchangeability. Subsampling one per report breaks this dependence.

    Args:
        scores: Verifier faithfulness scores (n_claims,).
        labels: Ground truth labels, 0=Faithful, 1=Unfaithful (n_claims,).
        report_ids: Report ID for each claim (n_claims,).
        seed: Random seed.

    Returns:
        Tuple of (subsampled_scores, subsampled_labels, subsampled_report_ids,
        selected_indices) where selected_indices maps back to original arrays.
    """
    rng = np.random.RandomState(seed)
    unique_reports = np.unique(report_ids)

    selected_indices = []
    for rid in unique_reports:
        mask = report_ids == rid
        indices = np.where(mask)[0]
        selected = rng.choice(indices)
        selected_indices.append(selected)

    selected_indices = np.array(selected_indices)
    logger.info(
        f"Subsampled {len(selected_indices)} claims (one per report) "
        f"from {len(scores)} total claims across {len(unique_reports)} reports"
    )

    return (
        scores[selected_indices],
        labels[selected_indices],
        report_ids[selected_indices],
        selected_indices,
    )


def compute_conformal_pvalues(
    test_scores: np.ndarray,
    calibration_scores: np.ndarray,
) -> np.ndarray:
    """Compute conformal p-values for test claims using calibration scores.

    The calibration set contains FAITHFUL claims. We test the null hypothesis
    "this claim is unfaithful" via an upper-tail test. A HIGH faithfulness
    score yields a SMALL p-value (unlikely under unfaithful null), so BH
    rejects the null (accepts the claim as faithful/green).

    p_j = (|{i in cal : s_i >= s_j}| + 1) / (n_cal + 1)

    Small p-value = high score relative to faithful calibration = accept as green.
    Large p-value = low score = do not accept.

    Args:
        test_scores: Verifier scores for test claims (n_test,).
        calibration_scores: Verifier scores from faithful calibration claims (n_cal,).

    Returns:
        Conformal p-values for each test claim (n_test,).
    """
    n_cal = len(calibration_scores)
    cal_sorted = np.sort(calibration_scores)

    # Count how many calibration scores are >= test score (upper tail)
    # searchsorted with side="left" gives the index of the first cal score >= test score
    # So n_cal - searchsorted("left") = count of cal scores >= test score
    ranks_geq = n_cal - np.searchsorted(cal_sorted, test_scores, side="left")
    pvalues = (ranks_geq + 1) / (n_cal + 1)

    return pvalues


def benjamini_hochberg(
    pvalues: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Apply the Benjamini-Hochberg procedure for FDR control.

    Args:
        pvalues: Conformal p-values (n,).
        alpha: Target FDR level.

    Returns:
        Boolean array indicating which claims are accepted (green) (n,).
    """
    n = len(pvalues)
    if n == 0:
        return np.array([], dtype=bool)

    # Sort p-values and track original indices
    sorted_indices = np.argsort(pvalues)
    sorted_pvalues = pvalues[sorted_indices]

    # BH thresholds: (rank / n) * alpha
    ranks = np.arange(1, n + 1)
    bh_thresholds = (ranks / n) * alpha

    # Find largest k such that p_(k) <= threshold_(k)
    below_threshold = sorted_pvalues <= bh_thresholds
    if not below_threshold.any():
        return np.zeros(n, dtype=bool)

    k_star = np.max(np.where(below_threshold)[0])

    # Accept all claims with p-value <= p_(k_star)
    # (This is equivalent to rejecting the null that the claim is unfaithful)
    accepted = np.zeros(n, dtype=bool)
    accepted[sorted_indices[: k_star + 1]] = True

    return accepted


def assign_triage_labels(
    scores: np.ndarray,
    accepted: np.ndarray,
    tau_low: float,
) -> np.ndarray:
    """Assign green/yellow/red labels based on acceptance and scores.

    Args:
        scores: Verifier faithfulness scores (n,).
        accepted: Boolean array from BH procedure (n,).
        tau_low: Threshold for yellow/red boundary among non-accepted claims.

    Returns:
        Array of string labels: 'green', 'yellow', or 'red' (n,).
    """
    labels = np.full(len(scores), "red", dtype=object)
    labels[accepted] = "green"

    # Among non-accepted: yellow if score > tau_low, red otherwise
    non_accepted = ~accepted
    labels[non_accepted & (scores > tau_low)] = "yellow"

    return labels


class ConformalClaimTriage:
    """Pathology-stratified conformal FDR control for claim triage.

    Implements cfBH (Jin & Candes, JMLR 2023) with per-pathology-group
    calibration. Groups with fewer than min_group_size calibration claims
    are merged into a pooled 'Rare/Other' group.

    Args:
        alpha: Target FDR level (default 0.05).
        alpha_low: Threshold level for yellow/red boundary (default 0.25).
        min_group_size: Minimum calibration claims per group (default 200).
        seed: Random seed for one-per-report subsampling.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        alpha_low: float = 0.25,
        min_group_size: int = 200,
        seed: int = 42,
    ):
        self.alpha = alpha
        self.alpha_low = alpha_low
        self.min_group_size = min_group_size
        self.seed = seed

        # Populated during calibration
        self.calibration_results: dict[str, CalibrationResult] = {}
        self.tau_low_per_group: dict[str, float] = {}
        self.is_calibrated = False

    def calibrate(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        pathology_groups: np.ndarray,
        report_ids: np.ndarray,
    ) -> dict[str, CalibrationResult]:
        """Calibrate conformal thresholds on the calibration split.

        Args:
            scores: Verifier faithfulness scores for all calibration claims.
            labels: Ground truth: 0=Faithful, 1=Unfaithful.
            pathology_groups: Pathology category for each claim.
            report_ids: Report ID for each claim (for one-per-report subsampling).

        Returns:
            Dict of group_name -> CalibrationResult.
        """
        # Step 1: Subsample one claim per report for exchangeability
        scores_sub, labels_sub, report_ids_sub, selected_indices = subsample_one_per_report(
            scores, labels, report_ids, seed=self.seed
        )

        # Use the returned indices to get pathology groups for subsampled claims
        groups_sub = pathology_groups[selected_indices]

        # Step 2: Group claims by pathology
        unique_groups = np.unique(groups_sub)
        group_counts = {g: np.sum(groups_sub == g) for g in unique_groups}

        # Identify underpowered groups
        pooled_groups = [g for g, c in group_counts.items() if c < self.min_group_size]
        powered_groups = [g for g, c in group_counts.items() if c >= self.min_group_size]

        if pooled_groups:
            logger.warning(
                f"Groups with <{self.min_group_size} calibration claims "
                f"(merged into Rare/Other): {pooled_groups}"
            )

        # Step 3: Calibrate per group
        self.calibration_results = {}

        for group in powered_groups:
            mask = groups_sub == group
            group_scores = scores_sub[mask]
            group_labels = labels_sub[mask]

            # Store calibration scores (faithful claims only for conformal p-values)
            faithful_mask = group_labels == 0
            cal_scores = group_scores[faithful_mask]

            self.calibration_results[group] = CalibrationResult(
                group_name=group,
                n_calibration_claims=len(group_scores),
                alpha=self.alpha,
                is_pooled=False,
                calibration_scores=cal_scores,
            )

            # Compute tau_low for this group
            if len(group_scores) > 0:
                self.tau_low_per_group[group] = np.quantile(group_scores, self.alpha_low)
            else:
                self.tau_low_per_group[group] = 0.5

        # Handle pooled group
        if pooled_groups:
            pooled_mask = np.isin(groups_sub, pooled_groups)
            pooled_scores = scores_sub[pooled_mask]
            pooled_labels = labels_sub[pooled_mask]

            faithful_mask = pooled_labels == 0
            cal_scores = pooled_scores[faithful_mask]

            self.calibration_results["Rare/Other"] = CalibrationResult(
                group_name="Rare/Other",
                n_calibration_claims=len(pooled_scores),
                alpha=self.alpha,
                is_pooled=True,
                calibration_scores=cal_scores,
            )
            if len(pooled_scores) > 0:
                self.tau_low_per_group["Rare/Other"] = np.quantile(pooled_scores, self.alpha_low)
            else:
                self.tau_low_per_group["Rare/Other"] = 0.5

            # Map pooled groups
            for g in pooled_groups:
                self.calibration_results[g] = self.calibration_results["Rare/Other"]
                self.tau_low_per_group[g] = self.tau_low_per_group["Rare/Other"]

        self.is_calibrated = True
        logger.info(
            f"Calibrated {len(powered_groups)} powered groups + "
            f"{len(pooled_groups)} pooled groups at alpha={self.alpha}"
        )
        return self.calibration_results

    def triage(
        self,
        scores: np.ndarray,
        pathology_groups: np.ndarray,
        claim_texts: Optional[list[str]] = None,
    ) -> list[TriageResult]:
        """Apply conformal triage to test claims.

        Args:
            scores: Verifier faithfulness scores for test claims (n,).
            pathology_groups: Pathology category for each test claim (n,).
            claim_texts: Optional claim text strings for result objects.

        Returns:
            List of TriageResult objects, one per claim.
        """
        if not self.is_calibrated:
            raise RuntimeError("Must call calibrate() before triage()")

        n = len(scores)
        if claim_texts is None:
            claim_texts = [f"claim_{i}" for i in range(n)]

        results = []

        # Process each pathology group separately
        unique_groups = np.unique(pathology_groups)
        all_labels = np.full(n, "red", dtype=object)
        all_pvalues = np.zeros(n)
        all_accepted = np.zeros(n, dtype=bool)

        for group in unique_groups:
            group_mask = pathology_groups == group

            # Get calibration data for this group
            if group in self.calibration_results:
                cal_result = self.calibration_results[group]
            elif "Rare/Other" in self.calibration_results:
                cal_result = self.calibration_results["Rare/Other"]
                logger.warning(f"Group '{group}' not seen during calibration, using Rare/Other")
            else:
                logger.error(f"No calibration data for group '{group}', marking all as red")
                continue

            group_scores = scores[group_mask]
            group_indices = np.where(group_mask)[0]

            # Compute conformal p-values
            pvalues = compute_conformal_pvalues(group_scores, cal_result.calibration_scores)
            all_pvalues[group_indices] = pvalues

            # Apply BH procedure within this group
            accepted = benjamini_hochberg(pvalues, self.alpha)
            all_accepted[group_indices] = accepted

            # Assign triage labels
            tau_low = self.tau_low_per_group.get(group, self.tau_low_per_group.get("Rare/Other", 0.5))
            group_labels = assign_triage_labels(group_scores, accepted, tau_low)
            all_labels[group_indices] = group_labels

        # Build result objects
        for i in range(n):
            results.append(TriageResult(
                claim_text=claim_texts[i],
                pathology_group=str(pathology_groups[i]),
                faithfulness_score=float(scores[i]),
                conformal_pvalue=float(all_pvalues[i]),
                label=str(all_labels[i]),
                is_accepted=bool(all_accepted[i]),
            ))

        # Log summary
        n_green = sum(1 for r in results if r.label == "green")
        n_yellow = sum(1 for r in results if r.label == "yellow")
        n_red = sum(1 for r in results if r.label == "red")
        logger.info(
            f"Triage results: {n_green} green ({n_green/n:.1%}), "
            f"{n_yellow} yellow ({n_yellow/n:.1%}), "
            f"{n_red} red ({n_red/n:.1%}) at alpha={self.alpha}"
        )

        return results


def compute_fdr(
    triage_results: list[TriageResult],
    ground_truth_labels: np.ndarray,
) -> dict[str, float]:
    """Compute observed false discovery rate among green claims.

    Args:
        triage_results: List of TriageResult from triage().
        ground_truth_labels: 0=Faithful, 1=Unfaithful for each claim.

    Returns:
        Dict with 'fdr', 'n_green', 'n_false_discoveries', 'n_true_discoveries'.
    """
    green_mask = np.array([r.is_accepted for r in triage_results])
    n_green = green_mask.sum()

    if n_green == 0:
        return {"fdr": 0.0, "n_green": 0, "n_false_discoveries": 0, "n_true_discoveries": 0}

    green_gt = ground_truth_labels[green_mask]
    n_false = (green_gt == 1).sum()  # unfaithful claims labeled green
    n_true = (green_gt == 0).sum()   # faithful claims labeled green

    fdr = n_false / n_green

    return {
        "fdr": float(fdr),
        "n_green": int(n_green),
        "n_false_discoveries": int(n_false),
        "n_true_discoveries": int(n_true),
    }


def compute_intra_report_icc(
    scores: np.ndarray,
    report_ids: np.ndarray,
) -> float:
    """Compute intra-report intraclass correlation coefficient (ICC) of verifier scores.

    ICC measures how much verifier scores vary within a report vs between reports.
    High ICC (>0.3) means claims from the same report get similar scores,
    violating the independence assumption for conformal prediction.

    Uses ICC(1,1) — one-way random effects model.

    Args:
        scores: Verifier scores for all claims.
        report_ids: Report ID for each claim.

    Returns:
        ICC value. 0 = no within-report correlation, 1 = perfect correlation.
    """
    unique_reports = np.unique(report_ids)

    # Filter to reports with >1 claim
    multi_claim_reports = [
        rid for rid in unique_reports if np.sum(report_ids == rid) > 1
    ]

    if len(multi_claim_reports) < 2:
        logger.warning("Too few multi-claim reports to compute ICC")
        return 0.0

    # Compute ICC(1,1) using ANOVA
    # Between-group variance (between reports) and within-group variance
    grand_mean = scores.mean()
    k = len(multi_claim_reports)

    ss_between = 0.0
    ss_within = 0.0
    n_total = 0
    group_sizes = []

    for rid in multi_claim_reports:
        mask = report_ids == rid
        group_scores = scores[mask]
        n_i = len(group_scores)
        group_mean = group_scores.mean()

        ss_between += n_i * (group_mean - grand_mean) ** 2
        ss_within += np.sum((group_scores - group_mean) ** 2)
        n_total += n_i
        group_sizes.append(n_i)

    if ss_within == 0:
        return 1.0

    ms_between = ss_between / (k - 1)
    ms_within = ss_within / (n_total - k)

    # Correct n_0 for unbalanced one-way ANOVA ICC(1,1)
    # n_0 = (1/(k-1)) * (N - sum(n_i^2)/N)  per Shrout & Fleiss (1979)
    group_sizes_arr = np.array(group_sizes, dtype=float)
    n_0 = (n_total - np.sum(group_sizes_arr ** 2) / n_total) / (k - 1)

    icc = (ms_between - ms_within) / (ms_between + (n_0 - 1) * ms_within)
    icc = max(0.0, min(1.0, icc))  # clamp to [0, 1]

    logger.info(
        f"Intra-report ICC = {icc:.4f} "
        f"(computed from {len(multi_claim_reports)} reports with >1 claim, "
        f"{'MINOR' if icc < 0.1 else 'MODERATE' if icc < 0.3 else 'SERIOUS'} "
        f"exchangeability concern)"
    )

    return icc
