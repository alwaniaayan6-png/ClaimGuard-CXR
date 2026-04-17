"""Fairness / subgroup analysis.

Stratify on: sex, age_quartile, race, scanner_manufacturer, country. Compute
per-subgroup accuracy + 95% bootstrap CI; parity gap = max - min accuracy.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _bootstrap_acc(scores: np.ndarray, labels: np.ndarray, n: int = 10_000) -> tuple[float, float, float]:
    if len(labels) == 0:
        return float("nan"), float("nan"), float("nan")
    preds = (scores < 0.5).astype(int)
    acc = float((preds == labels).mean())
    rng = np.random.default_rng(17)
    vals = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, len(labels), size=len(labels))
        vals[i] = float(((scores[idx] < 0.5).astype(int) == labels[idx]).mean())
    return acc, float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


@dataclass
class FairnessReport:
    site: str
    per_sex: dict
    per_age_quartile: dict
    per_race: dict
    per_scanner: dict
    per_country: dict
    parity_gaps: dict


def compute_fairness(rows: list[dict], scores: np.ndarray, labels: np.ndarray, site: str) -> FairnessReport:
    def group_by(key: str) -> dict[str, dict[str, float]]:
        groups: dict[str, tuple[list[int], list[int]]] = defaultdict(lambda: ([], []))
        for i, r in enumerate(rows):
            g = r.get(key)
            if g is None:
                continue
            groups[str(g)][0].append(scores[i])
            groups[str(g)][1].append(labels[i])
        out: dict[str, dict[str, float]] = {}
        for g, (sc, la) in groups.items():
            acc, lo, hi = _bootstrap_acc(np.asarray(sc), np.asarray(la))
            out[g] = {"n": len(la), "acc": acc, "ci_lo": lo, "ci_hi": hi}
        return out

    # Age quartile requires binning
    ages = [r.get("age") for r in rows]
    valid_ages = [a for a in ages if a is not None]
    if valid_ages:
        q1, q2, q3 = np.quantile(valid_ages, [0.25, 0.5, 0.75])
        quart_rows = []
        for r in rows:
            a = r.get("age")
            if a is None:
                quart_rows.append({**r, "age_quartile": "unknown"})
            elif a <= q1:
                quart_rows.append({**r, "age_quartile": "q1"})
            elif a <= q2:
                quart_rows.append({**r, "age_quartile": "q2"})
            elif a <= q3:
                quart_rows.append({**r, "age_quartile": "q3"})
            else:
                quart_rows.append({**r, "age_quartile": "q4"})
    else:
        quart_rows = rows

    per_sex = group_by("sex")
    per_age = {}
    if valid_ages:
        age_groups: dict[str, tuple[list[int], list[int]]] = defaultdict(lambda: ([], []))
        for i, r in enumerate(quart_rows):
            age_groups[r["age_quartile"]][0].append(scores[i])
            age_groups[r["age_quartile"]][1].append(labels[i])
        for g, (sc, la) in age_groups.items():
            acc, lo, hi = _bootstrap_acc(np.asarray(sc), np.asarray(la))
            per_age[g] = {"n": len(la), "acc": acc, "ci_lo": lo, "ci_hi": hi}
    per_race = group_by("race")
    per_scanner = group_by("scanner_manufacturer")
    per_country = group_by("country")

    def parity_gap(d: dict[str, dict[str, float]]) -> float:
        accs = [v["acc"] for v in d.values() if not np.isnan(v["acc"])]
        return float(max(accs) - min(accs)) if len(accs) >= 2 else 0.0

    return FairnessReport(
        site=site,
        per_sex=per_sex,
        per_age_quartile=per_age,
        per_race=per_race,
        per_scanner=per_scanner,
        per_country=per_country,
        parity_gaps={
            "sex": parity_gap(per_sex),
            "age_quartile": parity_gap(per_age),
            "race": parity_gap(per_race),
            "scanner": parity_gap(per_scanner),
            "country": parity_gap(per_country),
        },
    )
