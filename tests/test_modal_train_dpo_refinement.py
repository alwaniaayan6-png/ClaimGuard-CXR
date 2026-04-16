"""Tests for pure helpers in ``scripts.modal_train_dpo_refinement``.

The heavy training loop (``_run_training``) needs torch / transformers
/ CUDA and a real v3 checkpoint — it is only exercised by the actual
Modal H100 run.  Here we lock down the pure helpers that run on a
CPU-only laptop without torch:

* ``DPOTrainingConfig`` — JSON round-trip + defaults
* ``load_preference_pairs`` — schema validation, dedupe, missing-field handling
* ``DPOEarlyStopTracker`` — streak counting + reset behavior
* ``format_reward_histogram`` — printable line for log_every step reports

The Modal app / volume / remote function may or may not exist at
import time depending on whether the ``modal`` package is installed;
that code path is guarded in the module itself and not tested here.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.modal_train_dpo_refinement import (  # noqa: E402
    APP_NAME,
    DPOEarlyStopTracker,
    DPOTrainingConfig,
    VOLUME_NAME,
    format_reward_histogram,
    load_preference_pairs,
)


# ---------------------------------------------------------------------------
# Constants / module-level sanity
# ---------------------------------------------------------------------------


class TestModuleConstants(unittest.TestCase):
    def test_app_name_is_locked(self) -> None:
        self.assertEqual(APP_NAME, "claimguard-dpo-refinement")

    def test_volume_name_is_locked(self) -> None:
        self.assertEqual(VOLUME_NAME, "claimguard-data")


# ---------------------------------------------------------------------------
# DPOTrainingConfig
# ---------------------------------------------------------------------------


class TestDPOTrainingConfig(unittest.TestCase):
    def test_defaults_match_plan(self) -> None:
        c = DPOTrainingConfig()
        self.assertEqual(c.hf_backbone, "roberta-large")
        self.assertEqual(c.beta, 0.1)
        self.assertEqual(c.lr, 5e-6)
        self.assertEqual(c.num_epochs, 1)
        self.assertEqual(c.gradient_clip, 1.0)
        self.assertEqual(c.kl_max, 5.0)
        self.assertEqual(c.margin_min, 0.0)
        self.assertEqual(c.patience, 50)
        self.assertEqual(c.log_every, 100)
        self.assertEqual(c.freeze_first_n_layers, 8)
        self.assertEqual(c.target_label, 1)

    def test_consistency_mode_defaults(self) -> None:
        """Reviewer-requested fix: the default loss mode is the
        R-Drop consistency regularizer with MIXED data (post-2026-04-15
        v4 v1 collapse fix), NOT legacy DPO. Lock the default +
        hyperparameter values so a future edit can't silently revert
        to the broken DPO-only path or the broken single-class
        consistency path."""
        c = DPOTrainingConfig()
        self.assertEqual(c.loss_mode, "consistency_mixed")
        self.assertEqual(c.ce_weight, 1.0)
        self.assertEqual(c.consistency_weight, 0.5)
        self.assertEqual(c.ce_blowup_threshold, 2.5)
        self.assertEqual(c.faithful_per_cf, 1.0)
        self.assertEqual(
            c.full_training_data,
            "/data/verifier_training_data_v3.json",
        )

    def test_json_round_trip(self) -> None:
        c = DPOTrainingConfig(lr=1e-5, beta=0.2)
        s = c.to_json()
        c2 = DPOTrainingConfig.from_json(s)
        self.assertEqual(c, c2)

    def test_json_round_trip_with_loss_mode_override(self) -> None:
        c = DPOTrainingConfig(
            loss_mode="dpo",
            ce_weight=0.8,
            consistency_weight=1.2,
        )
        c2 = DPOTrainingConfig.from_json(c.to_json())
        self.assertEqual(c, c2)
        self.assertEqual(c2.loss_mode, "dpo")

    def test_json_is_valid_json(self) -> None:
        s = DPOTrainingConfig().to_json()
        parsed = json.loads(s)
        self.assertIn("base_checkpoint", parsed)
        self.assertIn("beta", parsed)
        self.assertIn("loss_mode", parsed)
        self.assertIn("consistency_weight", parsed)


# ---------------------------------------------------------------------------
# load_preference_pairs
# ---------------------------------------------------------------------------


class TestLoadPreferencePairs(unittest.TestCase):
    def _write(self, rows: list) -> str:
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        )
        json.dump(rows, f)
        f.close()
        self.addCleanup(lambda: os.unlink(f.name))
        return f.name

    def test_single_counterfactual_key(self) -> None:
        path = self._write([
            {
                "claim": "heart is enlarged",
                "evidence": "cardiomegaly noted",
                "counterfactual": "the heart appears enlarged",
            }
        ])
        pairs = load_preference_pairs(path)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["claim"], "heart is enlarged")
        self.assertEqual(pairs[0]["counterfactual"], "the heart appears enlarged")

    def test_counterfactuals_list_key(self) -> None:
        path = self._write([
            {
                "claim": "heart is enlarged",
                "evidence": "cardiomegaly noted",
                "counterfactuals": [
                    "the heart appears enlarged",
                    "enlarged heart observed",
                    "heart has grown in size",
                ],
            }
        ])
        pairs = load_preference_pairs(path)
        self.assertEqual(len(pairs), 3)
        for p in pairs:
            self.assertEqual(p["claim"], "heart is enlarged")

    def test_both_keys_combined(self) -> None:
        path = self._write([
            {
                "claim": "heart is enlarged",
                "evidence": "cardiomegaly noted",
                "counterfactual": "cfA",
                "counterfactuals": ["cfB", "cfC"],
            }
        ])
        pairs = load_preference_pairs(path)
        self.assertEqual(len(pairs), 3)
        texts = {p["counterfactual"] for p in pairs}
        self.assertEqual(texts, {"cfA", "cfB", "cfC"})

    def test_missing_claim_row_skipped(self) -> None:
        path = self._write([
            {"evidence": "x", "counterfactual": "y"},  # no claim
            {
                "claim": "z",
                "evidence": "w",
                "counterfactual": "v",
            },
        ])
        pairs = load_preference_pairs(path)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["claim"], "z")

    def test_empty_counterfactual_skipped(self) -> None:
        path = self._write([
            {
                "claim": "heart is enlarged",
                "evidence": "x",
                "counterfactual": "",
            }
        ])
        self.assertEqual(load_preference_pairs(path), [])

    def test_counterfactual_equal_to_claim_skipped(self) -> None:
        path = self._write([
            {
                "claim": "heart is enlarged",
                "evidence": "x",
                "counterfactual": "heart is enlarged",
            }
        ])
        self.assertEqual(load_preference_pairs(path), [])

    def test_missing_evidence_ok(self) -> None:
        path = self._write([
            {
                "claim": "heart is enlarged",
                "counterfactual": "enlarged heart seen",
            }
        ])
        pairs = load_preference_pairs(path)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["evidence"], "")

    def test_non_list_top_level_raises(self) -> None:
        path = self._write({"not": "a list"})  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            load_preference_pairs(path)

    def test_non_dict_rows_skipped(self) -> None:
        path = self._write([
            "just a string",
            42,
            None,
            {
                "claim": "heart is enlarged",
                "counterfactual": "enlarged heart seen",
            },
        ])
        pairs = load_preference_pairs(path)
        self.assertEqual(len(pairs), 1)

    def test_file_not_found_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_preference_pairs("/nonexistent/path/pairs.json")

    def test_whitespace_fields_stripped(self) -> None:
        path = self._write([
            {
                "claim": "  heart is enlarged  ",
                "evidence": "\tcardiomegaly\n",
                "counterfactual": "  enlarged heart  ",
            }
        ])
        pairs = load_preference_pairs(path)
        self.assertEqual(pairs[0]["claim"], "heart is enlarged")
        self.assertEqual(pairs[0]["evidence"], "cardiomegaly")
        self.assertEqual(pairs[0]["counterfactual"], "enlarged heart")


# ---------------------------------------------------------------------------
# DPOEarlyStopTracker
# ---------------------------------------------------------------------------


class TestDPOEarlyStopTracker(unittest.TestCase):
    def test_no_violations_no_stop(self) -> None:
        tracker = DPOEarlyStopTracker(
            kl_max=5.0, margin_min=0.0, patience=3,
        )
        for _ in range(10):
            self.assertFalse(tracker.update(kl=1.0, margin=0.5))
        self.assertIsNone(tracker.reason)
        self.assertEqual(tracker.kl_streak, 0)
        self.assertEqual(tracker.margin_streak, 0)

    def test_kl_violation_streak_triggers_stop(self) -> None:
        tracker = DPOEarlyStopTracker(
            kl_max=5.0, margin_min=0.0, patience=3,
        )
        # 2 violations — not enough.
        self.assertFalse(tracker.update(kl=6.0, margin=0.5))
        self.assertFalse(tracker.update(kl=6.0, margin=0.5))
        self.assertEqual(tracker.kl_streak, 2)
        # 3rd violation triggers stop.
        self.assertTrue(tracker.update(kl=6.0, margin=0.5))
        self.assertIn("KL", tracker.reason or "")

    def test_kl_streak_resets_on_good_step(self) -> None:
        tracker = DPOEarlyStopTracker(
            kl_max=5.0, margin_min=0.0, patience=3,
        )
        tracker.update(kl=6.0, margin=0.5)
        tracker.update(kl=6.0, margin=0.5)
        self.assertEqual(tracker.kl_streak, 2)
        # Good step resets the streak.
        tracker.update(kl=1.0, margin=0.5)
        self.assertEqual(tracker.kl_streak, 0)
        # Now we need 3 new violations in a row.
        self.assertFalse(tracker.update(kl=6.0, margin=0.5))
        self.assertFalse(tracker.update(kl=6.0, margin=0.5))
        self.assertTrue(tracker.update(kl=6.0, margin=0.5))

    def test_margin_violation_streak_triggers_stop(self) -> None:
        tracker = DPOEarlyStopTracker(
            kl_max=5.0, margin_min=0.0, patience=3,
        )
        self.assertFalse(tracker.update(kl=1.0, margin=-0.5))
        self.assertFalse(tracker.update(kl=1.0, margin=-0.5))
        self.assertTrue(tracker.update(kl=1.0, margin=-0.5))
        self.assertIn("margin", tracker.reason or "")

    def test_margin_streak_resets_on_good_step(self) -> None:
        tracker = DPOEarlyStopTracker(
            kl_max=5.0, margin_min=0.0, patience=3,
        )
        tracker.update(kl=1.0, margin=-0.5)
        tracker.update(kl=1.0, margin=-0.5)
        self.assertEqual(tracker.margin_streak, 2)
        tracker.update(kl=1.0, margin=0.5)
        self.assertEqual(tracker.margin_streak, 0)

    def test_both_violations_kl_wins(self) -> None:
        # When both conditions trip on the same step, KL is checked
        # first and is reported as the reason.
        tracker = DPOEarlyStopTracker(
            kl_max=5.0, margin_min=0.0, patience=2,
        )
        tracker.update(kl=6.0, margin=-0.5)
        stopped = tracker.update(kl=6.0, margin=-0.5)
        self.assertTrue(stopped)
        self.assertIn("KL", tracker.reason or "")

    def test_margin_exactly_at_threshold_is_not_violation(self) -> None:
        tracker = DPOEarlyStopTracker(
            kl_max=5.0, margin_min=0.0, patience=2,
        )
        # margin == margin_min is NOT a violation (strict less-than).
        self.assertFalse(tracker.update(kl=1.0, margin=0.0))
        self.assertFalse(tracker.update(kl=1.0, margin=0.0))
        self.assertEqual(tracker.margin_streak, 0)

    def test_kl_exactly_at_threshold_is_not_violation(self) -> None:
        tracker = DPOEarlyStopTracker(
            kl_max=5.0, margin_min=0.0, patience=2,
        )
        # kl == kl_max is NOT a violation (strict greater-than).
        self.assertFalse(tracker.update(kl=5.0, margin=0.5))
        self.assertFalse(tracker.update(kl=5.0, margin=0.5))
        self.assertEqual(tracker.kl_streak, 0)


# ---------------------------------------------------------------------------
# format_reward_histogram
# ---------------------------------------------------------------------------


class TestFormatRewardHistogram(unittest.TestCase):
    def test_empty_input(self) -> None:
        self.assertIn("<empty>", format_reward_histogram([]))

    def test_single_value(self) -> None:
        line = format_reward_histogram([0.5])
        self.assertIn("min=0.500", line)
        self.assertIn("max=0.500", line)
        self.assertIn("mean=0.500", line)

    def test_known_bucket_counts(self) -> None:
        # Values evenly spread from 0 to 9 into 10 buckets of width 1.
        values = [float(v) for v in range(10)]
        line = format_reward_histogram(values, n_buckets=10)
        # Each bucket should get exactly 1 count.
        self.assertIn("1 1 1 1 1 1 1 1 1 1", line)

    def test_all_identical_values(self) -> None:
        # Degenerate case — hi == lo — no divide-by-zero.
        line = format_reward_histogram([0.5, 0.5, 0.5], n_buckets=5)
        self.assertIn("min=0.500", line)
        self.assertIn("max=0.500", line)
        # First bucket gets everything.
        self.assertIn("3 0 0 0 0", line)

    def test_total_count_preserved(self) -> None:
        values = [0.1, 0.2, 0.3, 0.4, 0.9]
        line = format_reward_histogram(values, n_buckets=5)
        # Pull out the bucket counts after the closing bracket ']:'.
        after = line.split("]:")[-1].strip()
        counts = [int(x) for x in after.split()]
        self.assertEqual(sum(counts), len(values))


# ---------------------------------------------------------------------------
# _consistency_classifier_loss (torch-gated)
# ---------------------------------------------------------------------------
# These tests exercise the R-Drop / UDA style consistency loss added
# for Task 3 BUG A.  The loss function imports torch lazily, so we
# skip the whole class if torch isn't available on the current
# interpreter (CPU-only laptop).  When torch IS available, these
# tests validate that (a) the loss is non-negative, (b) identical
# inputs yield zero consistency KL, (c) the loss decreases as the
# two input distributions converge, and (d) the CE term picks up
# the true-label supervision.


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


@unittest.skipUnless(
    _torch_available(),
    "torch not installed — consistency loss tests require torch",
)
class TestConsistencyClassifierLoss(unittest.TestCase):
    """Reviewer-requested: lock the Task 3 BUG A fix with tests.

    The consistency loss must:
    (1) return a non-negative scalar
    (2) produce zero consistency KL when the two logit tensors are
        identical (the invariance objective is already satisfied)
    (3) be strictly larger when the two logit tensors disagree,
        and decrease monotonically as they converge
    (4) preserve label supervision via the CE term
    """

    def _make_logits(self, values):
        import torch
        return torch.tensor(values, dtype=torch.float32)

    def test_identical_logits_zero_consistency(self) -> None:
        from scripts.modal_train_dpo_refinement import (
            _consistency_classifier_loss,
        )
        # Both sides equal → consistency KL must be zero.
        logits = self._make_logits([[-1.0, 1.0], [-0.5, 0.5]])
        loss, ce_mean, cons_kl, agreement = _consistency_classifier_loss(
            logits_original=logits,
            logits_counterfactual=logits.clone(),
            target_label=1,
        )
        self.assertAlmostEqual(cons_kl, 0.0, places=5)
        self.assertAlmostEqual(agreement, 0.0, places=5)
        # CE is still positive (depends on how confident the logits are)
        self.assertGreater(ce_mean, 0.0)

    def test_loss_is_nonnegative(self) -> None:
        from scripts.modal_train_dpo_refinement import (
            _consistency_classifier_loss,
        )
        logits_a = self._make_logits([[-1.0, 1.0], [2.0, -2.0]])
        logits_b = self._make_logits([[-0.5, 0.5], [1.0, -1.0]])
        loss, _, _, _ = _consistency_classifier_loss(
            logits_original=logits_a,
            logits_counterfactual=logits_b,
            target_label=1,
        )
        self.assertGreaterEqual(float(loss.item()), 0.0)

    def test_disagreeing_logits_have_positive_consistency(self) -> None:
        from scripts.modal_train_dpo_refinement import (
            _consistency_classifier_loss,
        )
        # One side strongly contradicted, other side strongly supported.
        logits_a = self._make_logits([[-5.0, 5.0]])  # P(contra)≈1
        logits_b = self._make_logits([[5.0, -5.0]])  # P(contra)≈0
        _, _, cons_kl, _ = _consistency_classifier_loss(
            logits_original=logits_a,
            logits_counterfactual=logits_b,
            target_label=1,
        )
        self.assertGreater(cons_kl, 1.0)

    def test_consistency_kl_decreases_as_logits_converge(self) -> None:
        """The consistency KL must be monotone in the distance
        between the two output distributions.  Sweep a pair of
        logits from far-apart to identical and check monotone
        decrease."""
        from scripts.modal_train_dpo_refinement import (
            _consistency_classifier_loss,
        )
        kls = []
        for gap in (4.0, 3.0, 2.0, 1.0, 0.5, 0.0):
            logits_a = self._make_logits([[-gap, gap]])
            logits_b = self._make_logits([[gap, -gap]])
            _, _, cons_kl, _ = _consistency_classifier_loss(
                logits_original=logits_a,
                logits_counterfactual=logits_b,
                target_label=1,
            )
            kls.append(cons_kl)
        # Each successive gap reduction should give a strictly lower
        # consistency KL.
        for i in range(len(kls) - 1):
            self.assertGreaterEqual(
                kls[i], kls[i + 1],
                f"consistency KL not monotone: kls[{i}]={kls[i]} "
                f"< kls[{i+1}]={kls[i+1]}",
            )
        self.assertAlmostEqual(kls[-1], 0.0, places=5)

    def test_ce_picks_up_label_supervision(self) -> None:
        """CE term should be LOW when both sides confidently predict
        the target label, and HIGH when they confidently predict
        the wrong label."""
        from scripts.modal_train_dpo_refinement import (
            _consistency_classifier_loss,
        )
        # Correct direction: both predict contradicted (class 1).
        correct = self._make_logits([[-5.0, 5.0]])
        _, ce_correct, _, _ = _consistency_classifier_loss(
            logits_original=correct,
            logits_counterfactual=correct.clone(),
            target_label=1,
        )
        # Wrong direction: both predict not-contradicted (class 0).
        wrong = self._make_logits([[5.0, -5.0]])
        _, ce_wrong, _, _ = _consistency_classifier_loss(
            logits_original=wrong,
            logits_counterfactual=wrong.clone(),
            target_label=1,
        )
        self.assertLess(ce_correct, 0.1)
        self.assertGreater(ce_wrong, 5.0)

    def test_agreement_signal_is_signed(self) -> None:
        """The agreement return (`P(y|cf) - P(y|orig)`) should be
        positive when the counterfactual has a higher contradicted
        probability and negative when the original does."""
        from scripts.modal_train_dpo_refinement import (
            _consistency_classifier_loss,
        )
        orig_low = self._make_logits([[2.0, -2.0]])   # P(contra)≈0.02
        cf_high = self._make_logits([[-2.0, 2.0]])    # P(contra)≈0.98
        _, _, _, agreement = _consistency_classifier_loss(
            logits_original=orig_low,
            logits_counterfactual=cf_high,
            target_label=1,
        )
        self.assertGreater(agreement, 0.5)
        # Flip: original high, cf low
        _, _, _, agreement2 = _consistency_classifier_loss(
            logits_original=cf_high,
            logits_counterfactual=orig_low,
            target_label=1,
        )
        self.assertLess(agreement2, -0.5)

    def test_custom_weights_affect_total_loss(self) -> None:
        """The ce_weight and consistency_weight parameters should
        both change the total loss value linearly."""
        from scripts.modal_train_dpo_refinement import (
            _consistency_classifier_loss,
        )
        logits_a = self._make_logits([[-1.0, 1.0]])
        logits_b = self._make_logits([[1.0, -1.0]])
        loss_default, _, _, _ = _consistency_classifier_loss(
            logits_original=logits_a,
            logits_counterfactual=logits_b,
            target_label=1,
            ce_weight=1.0,
            consistency_weight=0.5,
        )
        loss_double_cons, _, _, _ = _consistency_classifier_loss(
            logits_original=logits_a,
            logits_counterfactual=logits_b,
            target_label=1,
            ce_weight=1.0,
            consistency_weight=1.0,
        )
        self.assertGreater(
            float(loss_double_cons.item()),
            float(loss_default.item()),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
