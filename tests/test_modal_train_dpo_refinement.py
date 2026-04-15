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

    def test_json_round_trip(self) -> None:
        c = DPOTrainingConfig(lr=1e-5, beta=0.2)
        s = c.to_json()
        c2 = DPOTrainingConfig.from_json(s)
        self.assertEqual(c, c2)

    def test_json_is_valid_json(self) -> None:
        s = DPOTrainingConfig().to_json()
        parsed = json.loads(s)
        self.assertIn("base_checkpoint", parsed)
        self.assertIn("beta", parsed)


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
