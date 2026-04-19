"""Three-grader silver-label ensemble with Krippendorff-alpha audit.

Combines per-claim labels from ``green_labeler`` (GREEN), ``radfact_labeler``
(RadFact-via-Claude-Opus), and ``vert_labeler`` (VERT-via-Claude-Opus) into a
single final silver label, plus per-pair and three-way Krippendorff alpha
statistics reported for the agreement audit.

Decision rule (B2 fix — decoupled MIMIC-leakage reporting):

* All three unanimous           -> label stands, confidence = HIGH
* 2-of-3 agree and third is UNCERTAIN -> majority label, confidence = MED
* 2-of-3 agree and third disagrees -> majority label, confidence = LOW, flagged
* All three disagree / any two disagree strongly -> UNCERTAIN, excluded from
  headline precision/recall

Per-grader prevalence is reported separately in the agreement file so the
paper can show whether the MIMIC-trained GREEN and the MIMIC-free (RadFact,
VERT) graders yield the same prevalence finding.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)


@dataclass
class EnsembleLabel:
    claim_id: str
    image_id: str
    rrg_model: str
    claim_text: str
    green: str
    radfact: str
    vert: str
    final_label: str
    confidence: str             # HIGH | MED | LOW | EXCLUDED
    disagreement_flag: bool


_LABELS = ["SUPPORTED", "CONTRADICTED", "UNCERTAIN"]


def _load_jsonl(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _index_by_claim(rows: list[dict], label_key: str = "label") -> dict[str, str]:
    out: dict[str, str] = {}
    for r in rows:
        cid = str(r.get("claim_id", ""))
        if not cid:
            continue
        v = r.get(label_key) or "UNCERTAIN"
        out[cid] = v
    return out


def _decide(green: str, radfact: str, vert: str) -> tuple[str, str, bool]:
    """Return (final_label, confidence_tier, disagreement_flag)."""
    votes = [green, radfact, vert]
    counts = Counter(votes)
    top_label, top_count = counts.most_common(1)[0]
    if top_count == 3:
        return top_label, "HIGH", False
    if top_count == 2:
        # If two graders said UNCERTAIN, the claim is genuinely uncertain —
        # do NOT promote a lone minority vote to MED confidence.
        if top_label == "UNCERTAIN":
            return "UNCERTAIN", "EXCLUDED", True
        minority_labels = [v for v in votes if v != top_label]
        minority = minority_labels[0] if minority_labels else "UNCERTAIN"
        if minority == "UNCERTAIN":
            return top_label, "MED", False
        return top_label, "LOW", True
    return "UNCERTAIN", "EXCLUDED", True


def combine_labels(
    green_jsonl: Path,
    radfact_jsonl: Path,
    vert_jsonl: Path,
    out_jsonl: Path,
) -> dict:
    """Combine three per-claim JSONLs into an ensemble labels file.

    Returns a stats dict with per-grader prevalence and agreement counts.
    """
    green_rows = _load_jsonl(green_jsonl)
    radfact_rows = _load_jsonl(radfact_jsonl)
    vert_rows = _load_jsonl(vert_jsonl)
    g = _index_by_claim(green_rows)
    r = _index_by_claim(radfact_rows)
    v = _index_by_claim(vert_rows)

    # Use intersection of claims labeled by all three graders.
    shared_ids = set(g) & set(r) & set(v)
    logger.info("silver_ensemble: green=%d radfact=%d vert=%d shared=%d",
                len(g), len(r), len(v), len(shared_ids))

    meta_by_claim: dict[str, dict] = {}
    for row in green_rows + radfact_rows + vert_rows:
        cid = str(row.get("claim_id", ""))
        if cid and cid not in meta_by_claim:
            meta_by_claim[cid] = {
                "image_id": row.get("image_id", ""),
                "rrg_model": row.get("rrg_model", "unknown"),
                "claim_text": row.get("claim_text", ""),
            }

    out_jsonl = Path(out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    confidence_counts = Counter()
    final_label_counts = Counter()
    disagreement_count = 0
    with open(out_jsonl, "w") as fh:
        for cid in sorted(shared_ids):
            gv, rv, vv = g[cid], r[cid], v[cid]
            final, tier, disagreement = _decide(gv, rv, vv)
            meta = meta_by_claim.get(cid, {})
            row = EnsembleLabel(
                claim_id=cid,
                image_id=str(meta.get("image_id", "")),
                rrg_model=str(meta.get("rrg_model", "unknown")),
                claim_text=str(meta.get("claim_text", "")),
                green=gv,
                radfact=rv,
                vert=vv,
                final_label=final,
                confidence=tier,
                disagreement_flag=disagreement,
            )
            fh.write(json.dumps(asdict(row)) + "\n")
            confidence_counts[tier] += 1
            final_label_counts[final] += 1
            if disagreement:
                disagreement_count += 1

    per_grader = {
        "green": dict(Counter(g[c] for c in shared_ids)),
        "radfact": dict(Counter(r[c] for c in shared_ids)),
        "vert": dict(Counter(v[c] for c in shared_ids)),
    }
    alphas = {
        "green_vs_radfact": krippendorff_alpha([[g[c] for c in shared_ids], [r[c] for c in shared_ids]]),
        "green_vs_vert": krippendorff_alpha([[g[c] for c in shared_ids], [v[c] for c in shared_ids]]),
        "radfact_vs_vert": krippendorff_alpha([[r[c] for c in shared_ids], [v[c] for c in shared_ids]]),
        "three_way": krippendorff_alpha([
            [g[c] for c in shared_ids],
            [r[c] for c in shared_ids],
            [v[c] for c in shared_ids],
        ]),
    }
    stats = {
        "n_shared_claims": len(shared_ids),
        "per_grader_prevalence": per_grader,
        "krippendorff_alpha": alphas,
        "confidence_tier_counts": dict(confidence_counts),
        "final_label_counts": dict(final_label_counts),
        "disagreement_count": disagreement_count,
    }
    return stats


def krippendorff_alpha(coder_label_lists: list[list[str]]) -> float | None:
    """Compute Krippendorff's alpha for nominal categorical data.

    Args:
        coder_label_lists: list of per-coder label lists. All coders must
            label the same units in the same order. Labels are treated as
            nominal categories.

    Returns:
        Alpha in [-1, 1], or None if degenerate (single coder, single category).
    """
    if not coder_label_lists or len(coder_label_lists[0]) == 0:
        return None
    n_coders = len(coder_label_lists)
    n_units = len(coder_label_lists[0])
    for lst in coder_label_lists:
        if len(lst) != n_units:
            raise ValueError("all coder label lists must have the same length")
    all_labels = [lbl for lst in coder_label_lists for lbl in lst]
    categories = sorted(set(all_labels))
    if len(categories) <= 1:
        return None
    cat_to_idx = {c: i for i, c in enumerate(categories)}

    # Observed disagreement: average over units of mean pairwise disagreement.
    total_obs = 0.0
    valid_units = 0
    for u in range(n_units):
        vals = [coder_label_lists[c][u] for c in range(n_coders)]
        m = len(vals)
        if m < 2:
            continue
        valid_units += 1
        disagreements = 0
        pairs = 0
        for i in range(m):
            for j in range(i + 1, m):
                pairs += 1
                if vals[i] != vals[j]:
                    disagreements += 1
        total_obs += disagreements / pairs
    if valid_units == 0:
        return None
    observed = total_obs / valid_units

    # Expected disagreement under marginal distribution.
    marginals = [0] * len(categories)
    for lbl in all_labels:
        marginals[cat_to_idx[lbl]] += 1
    n_total = sum(marginals)
    if n_total <= 1:
        return None
    expected = 0.0
    for i, ni in enumerate(marginals):
        for j, nj in enumerate(marginals):
            if i != j:
                expected += ni * nj
    expected /= n_total * (n_total - 1)
    if expected <= 0:
        return 1.0 if observed == 0 else None
    return 1.0 - (observed / expected)
