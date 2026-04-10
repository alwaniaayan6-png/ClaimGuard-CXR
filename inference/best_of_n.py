"""Verifier-Guided Best-of-N Selection for ClaimGuard-CXR (Contribution 2).

Implements constrained best-of-N: maximize coverage subject to a faithfulness
threshold. Also supports MBR-BoN (NAACL 2025) with proximity regularization.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class ScoredReport:
    """A candidate report with its decomposed claims and scores."""
    report_text: str
    claims: list[dict]  # [{text, pathology, score, verdict}, ...]
    avg_faithfulness: float
    coverage: float
    final_score: float
    candidate_index: int


@dataclass
class SelectionResult:
    """Result of best-of-N selection for a single image."""
    selected_report: ScoredReport
    all_candidates: list[ScoredReport]
    flagged_for_review: bool  # True if no candidate met faithfulness threshold
    selection_method: str


def compute_coverage(
    report_claims: list[dict],
    detected_findings: set[str],
) -> float:
    """Compute fraction of CheXbert-detected findings mentioned in the report.

    Args:
        report_claims: List of claim dicts with 'pathology' key.
        detected_findings: Set of CheXpert finding labels detected in the image.

    Returns:
        Coverage fraction in [0, 1]. Returns 1.0 for "No Finding" images
        if the report correctly indicates no significant findings.
    """
    if not detected_findings or detected_findings == {"No Finding"}:
        # "No Finding" image: coverage = 1.0 if report says no findings
        report_findings = {c["pathology"] for c in report_claims}
        if "No Finding" in report_findings or not report_findings - {"No Finding", "Support Devices"}:
            return 1.0
        return 0.0

    # Normal case: fraction of detected findings mentioned
    mentioned = {c["pathology"] for c in report_claims}
    # Remove meta-categories from comparison
    relevant_detected = detected_findings - {"No Finding"}
    if not relevant_detected:
        return 1.0

    overlap = mentioned & relevant_detected
    return len(overlap) / len(relevant_detected)


def constrained_selection(
    candidates: list[ScoredReport],
    tau_faith: float = 0.85,
) -> tuple[ScoredReport, bool]:
    """Select report maximizing coverage subject to faithfulness constraint.

    maximize: coverage(report)
    subject to: avg_faithfulness(report) >= tau_faith

    If no candidate meets the threshold, select argmax(faithfulness) and
    flag for mandatory human review.

    Args:
        candidates: List of scored candidate reports.
        tau_faith: Minimum faithfulness threshold.

    Returns:
        Tuple of (selected_report, flagged_for_review).
    """
    # Filter candidates meeting faithfulness threshold
    feasible = [c for c in candidates if c.avg_faithfulness >= tau_faith]

    if feasible:
        # Among feasible, pick highest coverage
        best = max(feasible, key=lambda c: c.coverage)
        return best, False
    else:
        # No candidate meets threshold — pick most faithful, flag for review
        best = max(candidates, key=lambda c: c.avg_faithfulness)
        logger.warning(
            f"No candidate met faithfulness threshold {tau_faith}. "
            f"Best faithfulness: {best.avg_faithfulness:.3f}. Flagged for review."
        )
        return best, True


def mbr_bon_selection(
    candidates: list[ScoredReport],
    tau_faith: float = 0.85,
    lambda_proximity: float = 0.1,
    distance_fn: str = "rouge",
) -> tuple[ScoredReport, bool]:
    """MBR-BoN selection with proximity regularization (NAACL 2025).

    Adds a penalty for reports that are outliers compared to other candidates,
    which helps prevent reward hacking.

    score(r_i) = coverage(r_i) - lambda * avg_distance(r_i, {r_j : j != i})

    Subject to: avg_faithfulness(r_i) >= tau_faith

    Args:
        candidates: List of scored candidate reports.
        tau_faith: Minimum faithfulness threshold.
        lambda_proximity: Weight for proximity regularization.
        distance_fn: Distance function ('rouge' or 'jaccard').

    Returns:
        Tuple of (selected_report, flagged_for_review).
    """
    if not candidates:
        raise ValueError("Cannot select from empty candidate list")
    if len(candidates) == 1:
        return candidates[0], candidates[0].avg_faithfulness < tau_faith

    # Compute pairwise distances
    n = len(candidates)
    distances = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            if distance_fn == "jaccard":
                # Jaccard distance on claim sets
                claims_i = {c["text"] for c in candidates[i].claims}
                claims_j = {c["text"] for c in candidates[j].claims}
                if claims_i | claims_j:
                    dist = 1.0 - len(claims_i & claims_j) / len(claims_i | claims_j)
                else:
                    dist = 0.0
            else:
                # Simple word overlap distance as ROUGE-1 proxy
                words_i = set(candidates[i].report_text.lower().split())
                words_j = set(candidates[j].report_text.lower().split())
                if words_i | words_j:
                    dist = 1.0 - len(words_i & words_j) / len(words_i | words_j)
                else:
                    dist = 0.0

            distances[i, j] = dist
            distances[j, i] = dist

    # Compute average distance for each candidate
    avg_distances = distances.sum(axis=1) / max(n - 1, 1)

    # Apply constrained selection with proximity penalty
    feasible = [
        (i, c) for i, c in enumerate(candidates)
        if c.avg_faithfulness >= tau_faith
    ]

    if feasible:
        best_idx, best = max(
            feasible,
            key=lambda ic: ic[1].coverage - lambda_proximity * avg_distances[ic[0]],
        )
        return best, False
    else:
        best = max(candidates, key=lambda c: c.avg_faithfulness)
        return best, True


def select_best_report(
    candidate_reports: list[str],
    candidate_claims: list[list[dict]],
    detected_findings: set[str],
    tau_faith: float = 0.85,
    method: str = "constrained",
    lambda_proximity: float = 0.1,
) -> SelectionResult:
    """Full best-of-N selection pipeline.

    Args:
        candidate_reports: List of N candidate report texts.
        candidate_claims: List of N claim lists (each claim is a dict with
            'text', 'pathology', 'score', 'verdict' keys).
        detected_findings: Set of CheXpert labels detected in the image.
        tau_faith: Faithfulness threshold for constrained selection.
        method: Selection method ('constrained' or 'mbr_bon').
        lambda_proximity: MBR-BoN proximity weight (ignored for 'constrained').

    Returns:
        SelectionResult with the selected report and metadata.
    """
    assert len(candidate_reports) == len(candidate_claims), \
        f"Mismatched candidates: {len(candidate_reports)} reports vs {len(candidate_claims)} claim lists"

    # Score each candidate
    scored = []
    for i, (report, claims) in enumerate(zip(candidate_reports, candidate_claims)):
        if claims:
            avg_faith = np.mean([c["score"] for c in claims])
        else:
            avg_faith = 0.0

        cov = compute_coverage(claims, detected_findings)

        scored.append(ScoredReport(
            report_text=report,
            claims=claims,
            avg_faithfulness=avg_faith,
            coverage=cov,
            final_score=0.0,  # set by selection method
            candidate_index=i,
        ))

    # Select
    if method == "mbr_bon":
        selected, flagged = mbr_bon_selection(
            scored, tau_faith=tau_faith, lambda_proximity=lambda_proximity
        )
    else:
        selected, flagged = constrained_selection(scored, tau_faith=tau_faith)

    return SelectionResult(
        selected_report=selected,
        all_candidates=scored,
        flagged_for_review=flagged,
        selection_method=method,
    )
