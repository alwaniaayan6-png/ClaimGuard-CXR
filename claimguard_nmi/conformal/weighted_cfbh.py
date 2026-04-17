"""Weighted conformal BH under covariate shift (Tibshirani et al. 2019 extension).

When the test distribution differs from the calibration distribution, the
standard conformal exchangeability assumption fails. Tibshirani 2019
restores coverage by re-weighting calibration samples by the density
ratio w(x) = p_test(x) / p_cal(x).

For claim-level verification, "x" is a vector of per-claim features
(finding, laterality, certainty, report length, ...). We estimate the
density ratio with a calibrated gradient-boosted classifier discriminating
calibration vs test covariates; the classifier's predicted probability is
transformed into a ratio via p / (1 - p) * (n_cal / n_test).

ESS diagnostic: if effective sample size falls below a threshold, the
weighted estimator is unstable and we fall back to un-weighted inverted
cfBH with a honest 'weighting-degenerate' flag attached to the result.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from .inverted_cfbh import BHResult


@dataclass
class WeightedBHResult(BHResult):
    effective_sample_size: float = float("nan")
    weighting_degenerate: bool = False
    weights: Optional[np.ndarray] = None


class DensityRatioEstimator:
    """Gradient-boosted classifier-based density ratio estimator.

    Estimates w(x) = p_test(x) / p_cal(x) by training a classifier to
    predict P(from test | x) and mapping that probability to a ratio.

    Uses sklearn.ensemble.GradientBoostingClassifier; falls back to a
    logistic regression if sklearn is unavailable.
    """

    def __init__(self, random_state: int = 13):
        self.random_state = random_state
        self._clf = None
        self._n_cal = 0
        self._n_test = 0
        self._is_calibrated = False
        self._calibrator = None  # isotonic calibrator fitted on classifier probs

    def fit(self, X_cal: np.ndarray, X_test: np.ndarray) -> "DensityRatioEstimator":
        """Fit on stacked (X_cal labeled 0, X_test labeled 1) with class balance."""
        X = np.vstack([X_cal, X_test])
        y = np.concatenate([np.zeros(len(X_cal)), np.ones(len(X_test))])
        self._n_cal = len(X_cal)
        self._n_test = len(X_test)

        try:
            from sklearn.ensemble import GradientBoostingClassifier
            from sklearn.isotonic import IsotonicRegression
            from sklearn.model_selection import train_test_split

            X_tr, X_val, y_tr, y_val = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=self.random_state
            )
            clf = GradientBoostingClassifier(random_state=self.random_state)
            clf.fit(X_tr, y_tr)
            self._clf = clf
            # Isotonic calibration of out-of-fold probabilities.
            p_val = clf.predict_proba(X_val)[:, 1]
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(p_val, y_val)
            self._calibrator = iso
            self._is_calibrated = True
        except ImportError:
            # Minimal fallback — logistic via numpy; not production quality.
            self._clf = _NumpyLogistic().fit(X, y)
            self._calibrator = None
            self._is_calibrated = False
        return self

    def ratio(self, X: np.ndarray) -> np.ndarray:
        """Return w(x) for each row of X."""
        if self._clf is None:
            raise RuntimeError("Call fit() first")
        if hasattr(self._clf, "predict_proba"):
            p = self._clf.predict_proba(X)[:, 1]
        else:
            p = self._clf.predict(X)
        if self._calibrator is not None:
            p = self._calibrator.transform(np.clip(p, 1e-6, 1 - 1e-6))
        p = np.clip(p, 1e-6, 1 - 1e-6)
        # Ratio formula: p / (1 - p) * (n_cal / n_test).
        n_cal = max(self._n_cal, 1)
        n_test = max(self._n_test, 1)
        return (p / (1 - p)) * (n_cal / n_test)


class WeightedCfBH:
    """Inverted cfBH with Tibshirani-2019 weighted p-values.

    Parameters
    ----------
    ess_floor : float
        If effective sample size after weighting < ess_floor * n_cal, mark
        the result as weighting-degenerate and fall back to un-weighted.
    """

    def __init__(self, ess_floor: float = 0.30):
        self._cal_scores: Optional[np.ndarray] = None
        self._cal_features: Optional[np.ndarray] = None
        self._density = DensityRatioEstimator()
        self._ess_floor = ess_floor
        self._is_fit = False

    def fit(
        self,
        calibration_contradicted_scores: Sequence[float],
        calibration_features: np.ndarray,
        test_features: np.ndarray,
    ) -> "WeightedCfBH":
        cal = np.asarray(list(calibration_contradicted_scores), dtype=float)
        self._cal_scores = cal
        self._cal_features = np.asarray(calibration_features)
        if self._cal_features.shape[0] != cal.size:
            raise ValueError("calibration scores and features differ in length")
        self._density.fit(self._cal_features, np.asarray(test_features))
        self._is_fit = True
        return self

    def predict(
        self,
        test_scores: Sequence[float],
        test_features: np.ndarray,
        alpha: float,
    ) -> WeightedBHResult:
        if not (0 < alpha < 1):
            raise ValueError(f"alpha must be in (0, 1); got {alpha}")
        if not self._is_fit:
            raise RuntimeError("Call fit() before predict()")
        ts = np.asarray(list(test_scores), dtype=float)
        test_features = np.asarray(test_features)
        if test_features.shape[0] != ts.size:
            raise ValueError(
                f"test_features len {test_features.shape[0]} != test_scores len {ts.size}"
            )

        # Precompute calibration weights once. ESS is a property of the
        # weight vector, not of any individual test point.
        cal_weights = self._density.ratio(self._cal_features)
        w_sum = float(cal_weights.sum())
        denom = w_sum + 1.0
        ess = (w_sum ** 2) / max(float((cal_weights ** 2).sum()), 1e-12)

        # Vectorized p-value: for each test score s_j, count weighted mass of
        # calibration scores >= s_j.
        cal_scores = self._cal_scores
        ps = np.empty(len(ts), dtype=float)
        for i, s in enumerate(ts):
            numerator = float((cal_weights * (cal_scores >= s)).sum()) + 1.0
            ps[i] = numerator / denom
        n = ps.size

        order = np.argsort(ps)
        sorted_ps = ps[order]
        ranks = np.arange(1, n + 1)
        bh_crit = ranks * alpha / n
        passing = sorted_ps <= bh_crit
        if passing.any():
            k = int(np.max(np.where(passing)[0])) + 1
            threshold = sorted_ps[k - 1]
        else:
            k = 0
            threshold = 0.0
        green_mask = np.zeros(n, dtype=bool)
        if k > 0:
            green_mask[order[:k]] = True

        degenerate = bool(ess < self._ess_floor * (self._cal_scores.size or 1))

        return WeightedBHResult(
            alpha=alpha,
            n_green=int(green_mask.sum()),
            n_total=n,
            green_mask=green_mask,
            p_values=ps,
            bh_threshold=float(threshold),
            effective_sample_size=float(ess),
            weighting_degenerate=degenerate,
            weights=cal_weights,
        )


# ---------------------------------------------------------------------------
# Fallback — tiny logistic regression when sklearn is unavailable.
# ---------------------------------------------------------------------------
class _NumpyLogistic:
    def __init__(self, lr: float = 0.05, n_iter: int = 200, reg: float = 1e-3):
        self.lr = lr
        self.n_iter = n_iter
        self.reg = reg
        self.w = None
        self.b = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_NumpyLogistic":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, d = X.shape
        self.w = np.zeros(d)
        for _ in range(self.n_iter):
            z = X @ self.w + self.b
            p = 1.0 / (1.0 + np.exp(-z))
            grad_w = (X.T @ (p - y)) / n + self.reg * self.w
            grad_b = float((p - y).mean())
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        z = X @ self.w + self.b
        return 1.0 / (1.0 + np.exp(-z))
