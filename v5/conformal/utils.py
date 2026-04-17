"""Conformal diagnostics shared across v5 variants."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ConformalDiagnostics:
    alpha: float
    n_test: int
    n_green: int
    empirical_fdr: float
    fdr_ci_low: float
    fdr_ci_high: float
    power: float
    coverage: float


def compute_empirical_fdr(
    green_mask: np.ndarray,
    true_labels: np.ndarray,  # 1 = contradicted, 0 = supported (not contradicted)
    *,
    alpha: float,
    n_bootstrap: int = 10_000,
    seed: int = 17,
) -> ConformalDiagnostics:
    green_mask = green_mask.astype(bool)
    true_labels = true_labels.astype(int)
    n_test = len(true_labels)
    n_green = int(green_mask.sum())
    if n_green == 0:
        return ConformalDiagnostics(
            alpha=alpha,
            n_test=n_test,
            n_green=0,
            empirical_fdr=0.0,
            fdr_ci_low=0.0,
            fdr_ci_high=0.0,
            power=0.0,
            coverage=0.0,
        )
    fdr = float(true_labels[green_mask].sum()) / n_green  # contradicted among green
    # Power: fraction of truly supported that are in the green set
    supported_mask = true_labels == 0
    power = float(green_mask[supported_mask].sum()) / max(1, int(supported_mask.sum()))
    coverage = float(n_green) / max(1, n_test)

    rng = np.random.default_rng(seed)
    fdrs = np.empty(n_bootstrap)
    idx = np.arange(n_test)
    for b in range(n_bootstrap):
        sample = rng.choice(idx, size=n_test, replace=True)
        sm = green_mask[sample]
        tl = true_labels[sample]
        ng = int(sm.sum())
        if ng == 0:
            fdrs[b] = 0.0
        else:
            fdrs[b] = float(tl[sm].sum()) / ng
    ci_low, ci_high = np.quantile(fdrs, [0.025, 0.975])
    return ConformalDiagnostics(
        alpha=alpha,
        n_test=n_test,
        n_green=n_green,
        empirical_fdr=fdr,
        fdr_ci_low=float(ci_low),
        fdr_ci_high=float(ci_high),
        power=power,
        coverage=coverage,
    )
