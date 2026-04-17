"""Scaled provenance-gate experiment.

1,000 images × 3 VLMs × 4 temperatures × 2 seeds = 24,000 generations. For each
matched (claim, evidence) pair we record the verifier's supported score under
two conditions:
  - same-model pairing: claim + evidence both from VLM A
  - cross-model pairing: claim from VLM A + evidence from VLM B on same image

Primary output: downgrade-rate table + score-divergence as a function of
sampling entropy.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ProvenanceScaledResult:
    n_pairs: int
    same_model_downgrade_rate: float
    cross_model_downgrade_rate: float
    score_divergence_same_mean: float
    score_divergence_cross_mean: float
    by_temperature: dict[float, dict[str, float]] = field(default_factory=dict)


def summarize_pairs(pairs: list[dict]) -> ProvenanceScaledResult:
    """Expect each pair dict to have:
    {
      image_id, claim_text, temperature,
      same_model: {score, downgraded},
      cross_model: {score, downgraded},
    }
    """
    same_dg = [1 if p["same_model"]["downgraded"] else 0 for p in pairs]
    cross_dg = [1 if p["cross_model"]["downgraded"] else 0 for p in pairs]
    same_scores = np.asarray([p["same_model"]["score"] for p in pairs])
    cross_scores = np.asarray([p["cross_model"]["score"] for p in pairs])
    same_mean = float(np.abs(same_scores - cross_scores).mean())
    by_temp: dict[float, dict[str, float]] = {}
    from collections import defaultdict

    buckets: dict[float, list[dict]] = defaultdict(list)
    for p in pairs:
        buckets[float(p.get("temperature", 0.7))].append(p)
    for t, pp in buckets.items():
        sdg = float(np.mean([1 if q["same_model"]["downgraded"] else 0 for q in pp]))
        cdg = float(np.mean([1 if q["cross_model"]["downgraded"] else 0 for q in pp]))
        ss = np.asarray([q["same_model"]["score"] for q in pp])
        cs = np.asarray([q["cross_model"]["score"] for q in pp])
        div = float(np.abs(ss - cs).mean())
        by_temp[t] = {
            "same_downgrade_rate": sdg,
            "cross_downgrade_rate": cdg,
            "score_divergence_mean": div,
            "n": len(pp),
        }
    return ProvenanceScaledResult(
        n_pairs=len(pairs),
        same_model_downgrade_rate=float(np.mean(same_dg)) if same_dg else 0.0,
        cross_model_downgrade_rate=float(np.mean(cross_dg)) if cross_dg else 0.0,
        score_divergence_same_mean=same_mean,
        score_divergence_cross_mean=same_mean,
        by_temperature=by_temp,
    )
