"""Silver-standard fallback evaluation (when image-grounded GT is NO_GT).

Uses the LLM ensemble label (data.labeler_ensemble.EnsembleLabel) as the
reference. Reported separately with explicit caveats in the paper.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SilverEvalReport:
    n: int
    accuracy: float
    contra_recall: float
    supp_recall: float
    fraction_uncertain: float


def silver_metrics(
    scores: np.ndarray,
    silver_labels: list[str],  # {SUPPORTED, CONTRADICTED, UNCERTAIN, NOVEL_*}
) -> SilverEvalReport:
    # Map silver labels to binary: SUPPORTED / NOVEL_PLAUSIBLE → 0 (not contradicted).
    # CONTRADICTED / NOVEL_HALLUCINATED → 1. UNCERTAIN → excluded.
    kept_scores: list[float] = []
    kept_labels: list[int] = []
    n_uncertain = 0
    for s, lab in zip(scores.tolist(), silver_labels, strict=False):
        if lab in {"UNCERTAIN", ""}:
            n_uncertain += 1
            continue
        kept_scores.append(s)
        if lab in {"SUPPORTED", "NOVEL_PLAUSIBLE"}:
            kept_labels.append(0)
        elif lab in {"CONTRADICTED", "NOVEL_HALLUCINATED"}:
            kept_labels.append(1)
        else:
            n_uncertain += 1
    ks = np.asarray(kept_scores)
    kl = np.asarray(kept_labels)
    if len(kl) == 0:
        return SilverEvalReport(n=0, accuracy=0.0, contra_recall=0.0, supp_recall=0.0, fraction_uncertain=1.0)
    preds = (ks < 0.5).astype(int)
    acc = float((preds == kl).mean())
    pos = kl == 1
    neg = kl == 0
    return SilverEvalReport(
        n=len(kl),
        accuracy=acc,
        contra_recall=float((preds[pos] == 1).mean()) if pos.any() else float("nan"),
        supp_recall=float((preds[neg] == 0).mean()) if neg.any() else float("nan"),
        fraction_uncertain=n_uncertain / max(1, len(scores)),
    )
