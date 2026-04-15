"""StratCP — Stratified Conformal Prediction baseline.

Clean-room implementation of the stratified split-conformal predictor
described in the Zitnik-Lab medRxiv preprint (Feb 2026).  StratCP is
our comparison baseline for the inverted conformal BH (cfBH) procedure
used by ClaimGuard-CXR.

Contract (one-sided upper-tail rejection-style test)
----------------------------------------------------
The predictor is distribution-agnostic: it treats the calibration
scores as the *null* reference distribution and rejects any test
score that exceeds the stratified upper-tail quantile.  For each
stratum ``s`` with calibration scores ``{s_1, ..., s_{n_s}}``, we
compute the marginal quantile

    Q_s = quantile( cal_scores_s ,  ( n_s + 1 ) * ( 1 - α ) / n_s ).

At test time, we reject the null iff ``score_s ≥ Q_s``.  Under
exchangeability of a test score with the stratum's calibration pool,
the finite-sample per-stratum miscoverage is bounded by ``α``
(standard split-conformal Theorem 1) — i.e. the *false rejection*
rate on exchangeable null-class points is ≤ ``α`` per stratum.

How to use this with ClaimGuard's faithful-score verifier
---------------------------------------------------------
The ClaimGuard RoBERTa verifier outputs a *faithful probability* in
``[0, 1]`` (higher = more faithful).  To plug it into this upper-tail
predictor as a contradicted-class detector, define a *suspicion*
score::

    suspicion = 1.0 - faithful_probability

and calibrate StratCP on **faithful** (null-class, label ``== 0``)
calibration examples' suspicion values.  Then the per-stratum
``Q_s`` is an upper tail of the *faithful* suspicion distribution
and rejecting ``test_suspicion ≥ Q_s`` means "this claim looks
contradicted with per-stratum false-positive rate bounded by α".

This is the correct polarity.  Calibrating on contradicted scores
with the upper tail is a common pitfall — it gives you the upper
tail *of the contradicted distribution*, which is above most
contradicted claims, so almost nothing gets flagged (power ≈ α).

StratCP does NOT attempt global BH-style FDR control — that is the
cfBH path, and this class is here as a baseline to show that pure
stratified marginal conformal is *not* sufficient for formal FDR in
this setting (the per-stratum α does not translate into a global FDR
bound on a mixture test set).

Implementation notes
--------------------
* ``calibrate`` accepts 1D arrays of scores and string strata labels.
  Strata are arbitrary hashable labels (e.g., pathology names); we
  cast them to ``str`` internally for stable dict keys.
* ``predict`` returns a boolean array: ``True`` if the test point is
  rejected.  Unknown strata at test time fall back to the *pooled*
  calibration quantile, which matches the standard split-conformal
  default.
* The quantile uses numpy ``"higher"`` interpolation to match the
  conservative rounding in Theorem 1 (``⌈(n+1)(1-α)⌉``).
* We intentionally do not expose a continuous p-value — the paper's
  framing is acceptance-region style.  Add ``score(...)`` later if
  needed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np


@dataclass
class StratCPConfig:
    """Configuration bundle for a StratCP predictor."""

    alpha: float
    min_stratum_size: int = 20
    """If a stratum has fewer calibration points than this, fall back
    to the pooled quantile rather than risk a degenerate estimator."""


class StratCPPredictor:
    """Stratified split-conformal predictor.

    Args:
        alpha: Target miscoverage (e.g. 0.05 for 95% coverage).
        min_stratum_size: Strata smaller than this use the pooled
            quantile (default 20 — standard split-CP rule of thumb).

    Example:
        >>> import numpy as np
        >>> predictor = StratCPPredictor(alpha=0.05)
        >>> # 200 cal scores from a Gaussian, 3 strata
        >>> rng = np.random.default_rng(0)
        >>> scores = rng.normal(size=600)
        >>> strata = np.array(["A", "B", "C"] * 200)
        >>> predictor.calibrate(scores, strata=strata)
        >>> test_scores = rng.normal(size=100)
        >>> test_strata = np.array(["A"] * 100)
        >>> rejections = predictor.predict(test_scores, strata=test_strata)
        >>> rejections.shape
        (100,)
    """

    def __init__(
        self,
        alpha: float,
        min_stratum_size: int = 20,
    ) -> None:
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = float(alpha)
        self.min_stratum_size = int(min_stratum_size)
        self._per_stratum_q: dict[str, float] = {}
        self._pooled_q: Optional[float] = None
        self._stratum_sizes: dict[str, int] = {}
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    @staticmethod
    def _quantile_level(n: int, alpha: float) -> float:
        """The ``(n+1)(1-α)/n`` quantile level from split-CP Thm 1.

        Clipped to ``[0, 1]`` so numpy does not raise on tiny strata.
        Note that for very small ``n``, this can exceed 1, in which
        case numpy with ``"higher"`` interpolation returns the maximum
        calibration score — a vacuous (never-reject) threshold.  That
        is the correct conservative behaviour.
        """
        if n <= 0:
            raise ValueError("need at least one calibration point")
        level = (n + 1) * (1 - alpha) / n
        return min(max(level, 0.0), 1.0)

    def calibrate(
        self,
        cal_scores: np.ndarray,
        strata: np.ndarray,
    ) -> "StratCPPredictor":
        """Fit the per-stratum and pooled calibration quantiles.

        Args:
            cal_scores: 1-D array of calibration scores.  Higher =
                more suspicious (contradicted-like), matching the
                inverted-calibration convention used by cfBH.
            strata: 1-D array of stratum labels, same length as
                ``cal_scores``.  Labels are used as dict keys, so any
                hashable type works (we cast to ``str`` internally for
                stability).

        Returns:
            self (for fluent chaining).

        Raises:
            ValueError: on shape mismatch or empty calibration.
        """
        cal_scores = np.asarray(cal_scores, dtype=np.float64)
        strata = np.asarray(strata)
        if cal_scores.ndim != 1:
            raise ValueError("cal_scores must be 1-D")
        if strata.ndim != 1:
            raise ValueError("strata must be 1-D")
        if cal_scores.shape[0] != strata.shape[0]:
            raise ValueError(
                f"cal_scores ({cal_scores.shape[0]}) and strata "
                f"({strata.shape[0]}) have different lengths"
            )
        if cal_scores.shape[0] == 0:
            raise ValueError("cal_scores is empty")

        # Pooled quantile first (used as fallback for small strata).
        n_pool = cal_scores.shape[0]
        q_level_pool = self._quantile_level(n_pool, self.alpha)
        self._pooled_q = float(
            np.quantile(cal_scores, q_level_pool, method="higher")
        )

        # Per-stratum quantiles.
        self._per_stratum_q = {}
        self._stratum_sizes = {}
        # Coerce stratum labels to strings for stable dict keys.
        str_strata = strata.astype(str)
        for label in np.unique(str_strata):
            mask = str_strata == label
            n_s = int(mask.sum())
            self._stratum_sizes[label] = n_s
            if n_s < self.min_stratum_size:
                # Too small: fall back to pooled.  We still store the
                # pooled value under the stratum key so predict() can
                # do a single dict lookup.
                self._per_stratum_q[label] = self._pooled_q
                continue
            q_level = self._quantile_level(n_s, self.alpha)
            self._per_stratum_q[label] = float(
                np.quantile(cal_scores[mask], q_level, method="higher")
            )

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        test_scores: np.ndarray,
        strata: np.ndarray,
    ) -> np.ndarray:
        """Return a boolean rejection mask.

        A ``True`` entry means "reject the null", i.e. call the claim
        contradicted.  ``False`` means "fail to reject", i.e. keep the
        claim as-is (faithful by default).

        Strata unseen during calibration fall back to the pooled
        quantile.

        Args:
            test_scores: 1-D array of scores on the test set.
            strata: 1-D array of stratum labels (same shape).

        Returns:
            Boolean numpy array with ``test_scores.shape``.

        Raises:
            RuntimeError: if :meth:`calibrate` was not called first.
            ValueError: on shape mismatch.
        """
        if not self._fitted:
            raise RuntimeError(
                "StratCPPredictor.predict called before calibrate()"
            )
        test_scores = np.asarray(test_scores, dtype=np.float64)
        strata = np.asarray(strata)
        if test_scores.shape != strata.shape:
            raise ValueError(
                f"test_scores {test_scores.shape} and strata {strata.shape} "
                f"must share the same shape"
            )
        if test_scores.ndim != 1:
            raise ValueError("test_scores must be 1-D")

        str_strata = strata.astype(str)
        # Vectorised threshold lookup
        thresholds = np.empty_like(test_scores)
        for i, label in enumerate(str_strata):
            thresholds[i] = self._per_stratum_q.get(
                label, self._pooled_q  # type: ignore[arg-type]
            )
        return test_scores >= thresholds

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def per_stratum_thresholds(self) -> dict[str, float]:
        """Return a copy of the fitted per-stratum quantile table."""
        return dict(self._per_stratum_q)

    def stratum_sizes(self) -> dict[str, int]:
        """Return the per-stratum calibration counts."""
        return dict(self._stratum_sizes)

    def pooled_threshold(self) -> Optional[float]:
        return self._pooled_q

    def empirical_coverage(
        self,
        test_scores: np.ndarray,
        test_labels: np.ndarray,
        strata: np.ndarray,
        positive_label: int = 1,
    ) -> dict[str, float]:
        """Empirical per-stratum rejection rate on a labelled test set.

        This is a diagnostic helper, NOT a real coverage measurement
        (those require a held-out split where labels are unknown).  We
        return the *rejection rate on the positive class* and the
        *overall false-discovery rate* for the threshold we fitted —
        the same two numbers the paper uses in its baseline table.

        Args:
            test_scores: 1-D test scores.
            test_labels: 1-D integer labels; ``positive_label`` marks
                contradicted claims, everything else is faithful.
            strata: 1-D stratum labels for the test set.
            positive_label: Which label encodes "contradicted".

        Returns:
            Dict with keys
            ``{"power", "fdr", "rejection_rate", "n_test"}``.
        """
        rejections = self.predict(test_scores, strata)
        test_labels = np.asarray(test_labels)
        is_pos = test_labels == positive_label
        n_rej = int(rejections.sum())
        n_pos = int(is_pos.sum())
        # Power = true-positive rate on the positive class
        power = (
            float((rejections & is_pos).sum()) / max(n_pos, 1)
            if n_pos > 0
            else math.nan
        )
        # FDR = false discoveries / total rejections
        fdr = (
            float((rejections & ~is_pos).sum()) / max(n_rej, 1)
            if n_rej > 0
            else 0.0
        )
        return {
            "power": power,
            "fdr": fdr,
            "rejection_rate": n_rej / max(test_scores.shape[0], 1),
            "n_test": int(test_scores.shape[0]),
        }
