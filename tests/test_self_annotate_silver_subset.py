"""Tests for pure helpers in the Task 8 self-annotation CLI.

The interactive loop cannot be unit-tested directly (it reads from a
TTY and shows a PIL window), but its dependency-injection design lets
us drive it with canned responses and no-op hooks — which is exactly
what ``TestRunInteractive`` below does.

What this file covers:
    * ``parse_label_key`` / ``parse_confidence_key``
    * ``coarsen_to_binary``
    * ``sample_stratified``
    * ``build_self_annotation_row``
    * ``atomic_save_json``
    * ``load_existing_annotations`` / ``filter_unlabeled``
    * ``format_prompt``
    * ``run_interactive`` end-to-end via injected callables
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from typing import Any

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.self_annotate_silver_subset import (  # noqa: E402
    CONFIDENCE_KEYS,
    DEFAULT_N_PER_CLASS,
    DEFAULT_SEED,
    LABEL_KEYS,
    QUIT_KEY,
    atomic_save_json,
    build_self_annotation_row,
    coarsen_to_binary,
    filter_unlabeled,
    format_prompt,
    load_existing_annotations,
    load_silver_workbook,
    parse_confidence_key,
    parse_label_key,
    run_interactive,
    sample_stratified,
)


# ---------------------------------------------------------------------------
# Constants / module sanity
# ---------------------------------------------------------------------------


class TestModuleConstants(unittest.TestCase):
    def test_label_keys_cover_all_five_classes(self) -> None:
        self.assertEqual(
            set(LABEL_KEYS.values()),
            {
                "SUPPORTED",
                "CONTRADICTED",
                "NOVEL_PLAUSIBLE",
                "NOVEL_HALLUCINATED",
                "UNCERTAIN",
            },
        )

    def test_label_keys_are_single_letters(self) -> None:
        for key in LABEL_KEYS:
            self.assertEqual(len(key), 1)
            self.assertEqual(key.lower(), key)

    def test_quit_key_distinct_from_label_keys(self) -> None:
        self.assertNotIn(QUIT_KEY, LABEL_KEYS)

    def test_default_n_per_class_matches_plan(self) -> None:
        # Plan specifies 20 per class × 5 classes = 100 total.
        self.assertEqual(DEFAULT_N_PER_CLASS, 20)

    def test_default_seed_is_locked(self) -> None:
        self.assertEqual(DEFAULT_SEED, 42)

    def test_confidence_keys_cover_three_tiers(self) -> None:
        self.assertEqual(
            set(CONFIDENCE_KEYS.values()),
            {"low", "medium", "high"},
        )


# ---------------------------------------------------------------------------
# parse_label_key
# ---------------------------------------------------------------------------


class TestParseLabelKey(unittest.TestCase):
    def test_supported_lowercase(self) -> None:
        self.assertEqual(parse_label_key("s"), "SUPPORTED")

    def test_supported_uppercase(self) -> None:
        self.assertEqual(parse_label_key("S"), "SUPPORTED")

    def test_with_whitespace(self) -> None:
        self.assertEqual(parse_label_key("  c  "), "CONTRADICTED")

    def test_all_five_classes(self) -> None:
        self.assertEqual(parse_label_key("s"), "SUPPORTED")
        self.assertEqual(parse_label_key("c"), "CONTRADICTED")
        self.assertEqual(parse_label_key("p"), "NOVEL_PLAUSIBLE")
        self.assertEqual(parse_label_key("h"), "NOVEL_HALLUCINATED")
        self.assertEqual(parse_label_key("u"), "UNCERTAIN")

    def test_unknown_letter_returns_none(self) -> None:
        self.assertIsNone(parse_label_key("x"))
        self.assertIsNone(parse_label_key(""))
        self.assertIsNone(parse_label_key("su"))

    def test_quit_letter_returns_none(self) -> None:
        # q is NOT a label — the interactive loop handles it separately.
        self.assertIsNone(parse_label_key("q"))

    def test_non_string_returns_none(self) -> None:
        self.assertIsNone(parse_label_key(None))  # type: ignore[arg-type]
        self.assertIsNone(parse_label_key(42))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# parse_confidence_key
# ---------------------------------------------------------------------------


class TestParseConfidenceKey(unittest.TestCase):
    def test_low(self) -> None:
        self.assertEqual(parse_confidence_key("l"), "low")

    def test_medium(self) -> None:
        self.assertEqual(parse_confidence_key("m"), "medium")

    def test_high(self) -> None:
        self.assertEqual(parse_confidence_key("h"), "high")

    def test_empty_defaults_to_medium(self) -> None:
        self.assertEqual(parse_confidence_key(""), "medium")

    def test_whitespace_defaults_to_medium(self) -> None:
        self.assertEqual(parse_confidence_key("   "), "medium")

    def test_unknown_defaults_to_medium(self) -> None:
        self.assertEqual(parse_confidence_key("z"), "medium")

    def test_non_string_defaults_to_medium(self) -> None:
        self.assertEqual(
            parse_confidence_key(None),  # type: ignore[arg-type]
            "medium",
        )


# ---------------------------------------------------------------------------
# coarsen_to_binary
# ---------------------------------------------------------------------------


class TestCoarsenToBinary(unittest.TestCase):
    def test_supported_maps_to_zero(self) -> None:
        self.assertEqual(coarsen_to_binary("SUPPORTED"), 0)

    def test_contradicted_maps_to_one(self) -> None:
        self.assertEqual(coarsen_to_binary("CONTRADICTED"), 1)

    def test_novel_plausible_maps_to_one(self) -> None:
        self.assertEqual(coarsen_to_binary("NOVEL_PLAUSIBLE"), 1)

    def test_novel_hallucinated_maps_to_one(self) -> None:
        self.assertEqual(coarsen_to_binary("NOVEL_HALLUCINATED"), 1)

    def test_uncertain_maps_to_one(self) -> None:
        self.assertEqual(coarsen_to_binary("UNCERTAIN"), 1)

    def test_unknown_label_returns_none(self) -> None:
        self.assertIsNone(coarsen_to_binary("BOGUS"))
        self.assertIsNone(coarsen_to_binary(""))

    def test_lowercase_not_coerced(self) -> None:
        # The canonical labels are uppercase; we do NOT silently
        # accept lowercase because the grader pipeline would never
        # produce them — treat as unknown.
        self.assertIsNone(coarsen_to_binary("supported"))


# ---------------------------------------------------------------------------
# sample_stratified
# ---------------------------------------------------------------------------


def _silver_row(claim_id: str, majority: str) -> dict[str, Any]:
    return {
        "claim_id": claim_id,
        "image_file": f"{claim_id}.png",
        "image_path": f"/tmp/{claim_id}.png",
        "extracted_claim": f"claim for {claim_id}",
        "ground_truth_report": "gt report text",
        "majority_label": majority,
        "grader_chexbert_label": majority,
        "grader_claude_label": majority,
        "grader_medgemma_label": majority,
    }


class TestSampleStratified(unittest.TestCase):
    def _balanced_workbook(self, per_class: int = 40) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for lbl in (
            "SUPPORTED", "CONTRADICTED", "NOVEL_PLAUSIBLE",
            "NOVEL_HALLUCINATED", "UNCERTAIN",
        ):
            for i in range(per_class):
                rows.append(_silver_row(f"{lbl.lower()}_{i:03d}", lbl))
        return rows

    def test_empty_workbook_returns_empty(self) -> None:
        self.assertEqual(sample_stratified([]), [])

    def test_n_per_class_zero_returns_empty(self) -> None:
        wb = self._balanced_workbook()
        self.assertEqual(sample_stratified(wb, n_per_class=0), [])

    def test_balanced_workbook_returns_exact_count(self) -> None:
        wb = self._balanced_workbook()
        sampled = sample_stratified(wb, n_per_class=20)
        self.assertEqual(len(sampled), 100)

    def test_sampled_rows_preserve_class_balance(self) -> None:
        wb = self._balanced_workbook()
        sampled = sample_stratified(wb, n_per_class=20)
        from collections import Counter
        counts = Counter(r["majority_label"] for r in sampled)
        for lbl in (
            "SUPPORTED", "CONTRADICTED", "NOVEL_PLAUSIBLE",
            "NOVEL_HALLUCINATED", "UNCERTAIN",
        ):
            self.assertEqual(counts[lbl], 20)

    def test_deterministic_under_seed(self) -> None:
        wb = self._balanced_workbook()
        a = sample_stratified(wb, n_per_class=20, seed=123)
        b = sample_stratified(wb, n_per_class=20, seed=123)
        self.assertEqual(
            [r["claim_id"] for r in a],
            [r["claim_id"] for r in b],
        )

    def test_different_seeds_give_different_picks(self) -> None:
        wb = self._balanced_workbook()
        a = sample_stratified(wb, n_per_class=20, seed=1)
        b = sample_stratified(wb, n_per_class=20, seed=2)
        self.assertNotEqual(
            [r["claim_id"] for r in a],
            [r["claim_id"] for r in b],
        )

    def test_underfull_class_still_samples_available(self) -> None:
        # Only 5 SUPPORTED rows, asking for 20 → take all 5.
        wb = [_silver_row(f"sup_{i}", "SUPPORTED") for i in range(5)]
        wb += [
            _silver_row(f"con_{i}", "CONTRADICTED") for i in range(20)
        ]
        sampled = sample_stratified(wb, n_per_class=20)
        sup_count = sum(
            1 for r in sampled if r["majority_label"] == "SUPPORTED"
        )
        con_count = sum(
            1 for r in sampled if r["majority_label"] == "CONTRADICTED"
        )
        self.assertEqual(sup_count, 5)
        self.assertEqual(con_count, 20)

    def test_rows_with_missing_class_dropped(self) -> None:
        wb = [
            {"claim_id": "c1", "majority_label": ""},  # missing
            {"claim_id": "c2"},  # no key
            _silver_row("sup_001", "SUPPORTED"),
        ]
        sampled = sample_stratified(wb, n_per_class=20)
        self.assertEqual(len(sampled), 1)
        self.assertEqual(sampled[0]["claim_id"], "sup_001")

    def test_rows_with_unknown_class_dropped(self) -> None:
        wb = [_silver_row("x1", "NOT_A_REAL_LABEL")]
        self.assertEqual(sample_stratified(wb, n_per_class=20), [])

    def test_class_key_override(self) -> None:
        # Sample using a custom class field — tests that class_key is
        # actually threaded through.
        wb = [
            {"claim_id": "c1", "my_class": "SUPPORTED"},
            {"claim_id": "c2", "my_class": "CONTRADICTED"},
            {"claim_id": "c3", "my_class": "SUPPORTED"},
        ]
        sampled = sample_stratified(
            wb, n_per_class=10, class_key="my_class",
        )
        ids = {r["claim_id"] for r in sampled}
        self.assertEqual(ids, {"c1", "c2", "c3"})


# ---------------------------------------------------------------------------
# build_self_annotation_row
# ---------------------------------------------------------------------------


class TestBuildSelfAnnotationRow(unittest.TestCase):
    def _silver(self) -> dict[str, Any]:
        return {
            "claim_id": "c001",
            "image_file": "img_001.png",
            "image_path": "/tmp/img_001.png",
            "extracted_claim": "heart is enlarged",
            "ground_truth_report": "cardiomegaly noted",
            "majority_label": "CONTRADICTED",
            "grader_chexbert_label": "CONTRADICTED",
            "grader_claude_label": "CONTRADICTED",
            "grader_medgemma_label": "UNCERTAIN",
            "verifier_score": 0.42,
        }

    def test_basic_row_build(self) -> None:
        row = build_self_annotation_row(
            self._silver(),
            user_label="CONTRADICTED",
        )
        self.assertEqual(row["claim_id"], "c001")
        self.assertEqual(row["user_label"], "CONTRADICTED")
        self.assertEqual(row["user_confidence"], "medium")
        self.assertIn("annotated_at", row)

    def test_invalid_user_label_raises(self) -> None:
        with self.assertRaises(ValueError):
            build_self_annotation_row(
                self._silver(),
                user_label="BOGUS",
            )

    def test_preserves_all_grader_columns(self) -> None:
        row = build_self_annotation_row(
            self._silver(),
            user_label="SUPPORTED",
        )
        self.assertEqual(row["grader_chexbert_label"], "CONTRADICTED")
        self.assertEqual(row["grader_claude_label"], "CONTRADICTED")
        self.assertEqual(row["grader_medgemma_label"], "UNCERTAIN")

    def test_preserves_verifier_score(self) -> None:
        row = build_self_annotation_row(
            self._silver(),
            user_label="SUPPORTED",
        )
        self.assertEqual(row["verifier_score"], 0.42)

    def test_explicit_timestamp_used(self) -> None:
        row = build_self_annotation_row(
            self._silver(),
            user_label="SUPPORTED",
            annotated_at="2026-04-14T12:00:00Z",
        )
        self.assertEqual(row["annotated_at"], "2026-04-14T12:00:00Z")

    def test_confidence_passed_through(self) -> None:
        row = build_self_annotation_row(
            self._silver(),
            user_label="SUPPORTED",
            user_confidence="high",
        )
        self.assertEqual(row["user_confidence"], "high")

    def test_missing_fields_coerce_to_empty_string(self) -> None:
        row = build_self_annotation_row(
            {"claim_id": "x"},
            user_label="UNCERTAIN",
        )
        self.assertEqual(row["image_file"], "")
        self.assertEqual(row["extracted_claim"], "")


# ---------------------------------------------------------------------------
# atomic_save_json + load_existing_annotations
# ---------------------------------------------------------------------------


class TestAtomicSaveLoad(unittest.TestCase):
    def test_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.json")
            data = [{"claim_id": "c1", "user_label": "SUPPORTED"}]
            atomic_save_json(data, path)
            loaded = load_existing_annotations(path)
            self.assertEqual(loaded, data)

    def test_overwrite_is_atomic_no_tmp_leftover(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.json")
            atomic_save_json([{"a": 1}], path)
            atomic_save_json([{"b": 2}], path)
            loaded = load_existing_annotations(path)
            self.assertEqual(loaded, [{"b": 2}])
            # No lingering tmp files.
            leftovers = [
                f for f in os.listdir(tmp) if f.startswith("out.json.tmp.")
            ]
            self.assertEqual(leftovers, [])

    def test_load_missing_returns_empty_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "does_not_exist.json")
            self.assertEqual(load_existing_annotations(path), [])

    def test_load_corrupt_json_returns_empty_list(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write("{not valid json")
            path = f.name
        try:
            self.assertEqual(load_existing_annotations(path), [])
        finally:
            os.unlink(path)

    def test_load_non_list_json_returns_empty_list(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write('{"not": "a list"}')
            path = f.name
        try:
            self.assertEqual(load_existing_annotations(path), [])
        finally:
            os.unlink(path)

    def test_save_creates_parent_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            nested = os.path.join(tmp, "a", "b", "c", "out.json")
            atomic_save_json([{"x": 1}], nested)
            self.assertTrue(os.path.exists(nested))


# ---------------------------------------------------------------------------
# filter_unlabeled
# ---------------------------------------------------------------------------


class TestFilterUnlabeled(unittest.TestCase):
    def test_no_existing_returns_all(self) -> None:
        sampled = [{"claim_id": "c1"}, {"claim_id": "c2"}]
        self.assertEqual(filter_unlabeled(sampled, []), sampled)

    def test_partial_skip(self) -> None:
        sampled = [
            {"claim_id": "c1"},
            {"claim_id": "c2"},
            {"claim_id": "c3"},
        ]
        existing = [{"claim_id": "c2"}]
        result = filter_unlabeled(sampled, existing)
        ids = [r["claim_id"] for r in result]
        self.assertEqual(ids, ["c1", "c3"])

    def test_all_already_done(self) -> None:
        sampled = [{"claim_id": "c1"}, {"claim_id": "c2"}]
        existing = [{"claim_id": "c1"}, {"claim_id": "c2"}]
        self.assertEqual(filter_unlabeled(sampled, existing), [])

    def test_ordering_preserved(self) -> None:
        sampled = [
            {"claim_id": "z"},
            {"claim_id": "a"},
            {"claim_id": "m"},
        ]
        result = filter_unlabeled(sampled, [])
        self.assertEqual(
            [r["claim_id"] for r in result],
            ["z", "a", "m"],
        )

    def test_empty_claim_id_rows_kept(self) -> None:
        # Defensive: a row with missing claim_id is never in existing,
        # so it should survive the filter.
        sampled = [{"claim_id": ""}]
        existing = [{"claim_id": ""}]
        # But empty strings match empty strings, so it WILL be filtered.
        # The helper explicitly drops empties from labeled_ids, so:
        result = filter_unlabeled(sampled, existing)
        self.assertEqual(len(result), 1)  # labeled_ids had no "" entry


# ---------------------------------------------------------------------------
# format_prompt
# ---------------------------------------------------------------------------


class TestFormatPrompt(unittest.TestCase):
    def test_contains_progress(self) -> None:
        row = {
            "claim_id": "c1",
            "extracted_claim": "heart enlarged",
            "ground_truth_report": "gt",
            "majority_label": "CONTRADICTED",
            "image_file": "img.png",
            "grader_chexbert_label": "CONTRADICTED",
            "grader_claude_label": "SUPPORTED",
            "grader_medgemma_label": "NOVEL_PLAUSIBLE",
        }
        prompt = format_prompt(row, (3, 100))
        self.assertIn("[3/100]", prompt)
        self.assertIn("c1", prompt)
        self.assertIn("heart enlarged", prompt)
        self.assertIn("gt", prompt)
        self.assertIn("img.png", prompt)

    def test_prompt_mentions_all_label_keys(self) -> None:
        prompt = format_prompt(
            {"claim_id": "c"}, (1, 1),
        )
        for key in ("[s]", "[c]", "[p]", "[h]", "[u]", "[q]"):
            self.assertIn(key, prompt)

    def test_prompt_does_not_leak_silver_majority(self) -> None:
        """Reviewer-flagged methodology critical: the prompt must not
        prime the user with the silver graders' answer.  This test
        locks the fix so a future regression can't silently leak the
        ensemble labels back into the UI."""
        row = {
            "claim_id": "c1",
            "extracted_claim": "heart is enlarged",
            "ground_truth_report": "cardiomegaly noted",
            "majority_label": "CONTRADICTED",
            "grader_chexbert_label": "CONTRADICTED",
            "grader_claude_label": "NOVEL_HALLUCINATED",
            "grader_medgemma_label": "UNCERTAIN",
            "image_file": "img.png",
        }
        prompt = format_prompt(row, (1, 100))
        # None of the 5 class labels may appear verbatim — the user
        # must not be primed.  (Note: the LABEL_KEYS help line uses
        # lowercase "[s]upported" etc., which we allow.)
        for leak in (
            "CONTRADICTED",
            "NOVEL_HALLUCINATED",
            "UNCERTAIN",
            "silver majority",
            "majority=",
            "grader_",
        ):
            self.assertNotIn(leak, prompt)

    def test_prompt_does_not_leak_grader_labels_even_when_present(self) -> None:
        row = {
            "claim_id": "c1",
            "extracted_claim": "claim text",
            "ground_truth_report": "report text",
            "grader_chexbert_label": "SUPPORTED",
            "grader_claude_label": "CONTRADICTED",
            "grader_medgemma_label": "NOVEL_PLAUSIBLE",
            "majority_label": "SUPPORTED",
            "image_file": "img.png",
        }
        prompt = format_prompt(row, (1, 100))
        # Individual grader labels must not appear.  We check for the
        # unique-to-label tokens (SUPPORTED/CONTRADICTED/etc. in their
        # canonical uppercase form).
        self.assertNotIn("SUPPORTED", prompt)
        self.assertNotIn("CONTRADICTED", prompt)
        self.assertNotIn("NOVEL_PLAUSIBLE", prompt)


# ---------------------------------------------------------------------------
# load_silver_workbook
# ---------------------------------------------------------------------------


class TestLoadSilverWorkbook(unittest.TestCase):
    def test_loads_valid_list(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump([{"claim_id": "c1"}], f)
            path = f.name
        try:
            rows = load_silver_workbook(path)
            self.assertEqual(rows, [{"claim_id": "c1"}])
        finally:
            os.unlink(path)

    def test_missing_path_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_silver_workbook("/nonexistent/path.json")

    def test_non_list_raises(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump({"not": "a list"}, f)
            path = f.name
        try:
            with self.assertRaises(ValueError):
                load_silver_workbook(path)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# run_interactive — injected dependencies
# ---------------------------------------------------------------------------


class FakePromptSession:
    """Mutable prompt driver for the injected ``prompt_fn``."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[str] = []

    def __call__(self, prompt_text: str) -> str:
        self.calls.append(prompt_text)
        if not self._responses:
            return QUIT_KEY
        return self._responses.pop(0)


class TestRunInteractive(unittest.TestCase):
    def _make_workbook(self, n: int = 3) -> list[dict[str, Any]]:
        return [
            _silver_row(f"claim_{i:03d}", "SUPPORTED")
            for i in range(n)
        ]

    def test_labels_every_row_and_saves(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "out.json")
            wb = self._make_workbook(3)
            prompt = FakePromptSession(["s", "c", "u"])
            displayed: list[str] = []

            all_rows = run_interactive(
                workbook=wb,
                output_path=out_path,
                display_image_fn=lambda row: displayed.append(
                    row["claim_id"]
                ),
                prompt_fn=prompt,
            )
            self.assertEqual(len(all_rows), 3)
            self.assertEqual(
                [r["user_label"] for r in all_rows],
                ["SUPPORTED", "CONTRADICTED", "UNCERTAIN"],
            )
            self.assertEqual(displayed, [
                "claim_000", "claim_001", "claim_002",
            ])
            # File persisted.
            with open(out_path, "r") as f:
                persisted = json.load(f)
            self.assertEqual(len(persisted), 3)

    def test_quit_key_saves_partial(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "out.json")
            wb = self._make_workbook(3)
            prompt = FakePromptSession(["s", "q"])

            all_rows = run_interactive(
                workbook=wb,
                output_path=out_path,
                display_image_fn=lambda _row: None,
                prompt_fn=prompt,
            )
            self.assertEqual(len(all_rows), 1)
            with open(out_path, "r") as f:
                persisted = json.load(f)
            self.assertEqual(len(persisted), 1)

    def test_invalid_input_reprompts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "out.json")
            wb = self._make_workbook(1)
            prompt = FakePromptSession(["x", "zzz", "  ", "s"])

            all_rows = run_interactive(
                workbook=wb,
                output_path=out_path,
                display_image_fn=lambda _row: None,
                prompt_fn=prompt,
            )
            self.assertEqual(len(all_rows), 1)
            self.assertEqual(all_rows[0]["user_label"], "SUPPORTED")
            # The prompt session burned through all 4 inputs.
            self.assertEqual(len(prompt.calls), 4)

    def test_resume_skips_already_labeled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "out.json")
            wb = self._make_workbook(3)
            # Pre-seed the output file with claim_000 already done.
            existing = [
                build_self_annotation_row(
                    wb[0],
                    user_label="CONTRADICTED",
                )
            ]
            atomic_save_json(existing, out_path)

            prompt = FakePromptSession(["s", "u"])
            all_rows = run_interactive(
                workbook=wb,
                output_path=out_path,
                display_image_fn=lambda _row: None,
                prompt_fn=prompt,
            )
            self.assertEqual(len(all_rows), 3)
            labels = [r["user_label"] for r in all_rows]
            # First row retained its existing label; new ones appended.
            self.assertEqual(
                labels,
                ["CONTRADICTED", "SUPPORTED", "UNCERTAIN"],
            )

    def test_empty_workbook_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "out.json")
            prompt = FakePromptSession([])
            all_rows = run_interactive(
                workbook=[],
                output_path=out_path,
                display_image_fn=lambda _row: None,
                prompt_fn=prompt,
            )
            self.assertEqual(all_rows, [])
            # Prompt was never called.
            self.assertEqual(prompt.calls, [])

    def test_display_error_does_not_stop_session(self) -> None:
        def bad_display(_row: dict[str, Any]) -> None:
            raise RuntimeError("no PIL here")

        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "out.json")
            wb = self._make_workbook(1)
            prompt = FakePromptSession(["s"])
            all_rows = run_interactive(
                workbook=wb,
                output_path=out_path,
                display_image_fn=bad_display,
                prompt_fn=prompt,
            )
            self.assertEqual(len(all_rows), 1)
            self.assertEqual(all_rows[0]["user_label"], "SUPPORTED")

    def test_quit_hook_early_exit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "out.json")
            wb = self._make_workbook(5)
            prompt = FakePromptSession(["s", "s", "s", "s", "s"])
            # Quit after 2 rows.
            call_count = {"n": 0}

            def quit_after_two() -> bool:
                call_count["n"] += 1
                return call_count["n"] > 2

            all_rows = run_interactive(
                workbook=wb,
                output_path=out_path,
                display_image_fn=lambda _row: None,
                prompt_fn=prompt,
                quit_requested=quit_after_two,
            )
            self.assertEqual(len(all_rows), 2)

    def test_confidence_fn_threaded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "out.json")
            wb = self._make_workbook(1)
            prompt = FakePromptSession(["s"])
            conf_calls: list[str] = []

            def confidence_fn(prompt_text: str) -> str:
                conf_calls.append(prompt_text)
                return "h"

            all_rows = run_interactive(
                workbook=wb,
                output_path=out_path,
                display_image_fn=lambda _row: None,
                prompt_fn=prompt,
                confidence_fn=confidence_fn,
            )
            self.assertEqual(len(all_rows), 1)
            self.assertEqual(all_rows[0]["user_confidence"], "high")
            self.assertEqual(len(conf_calls), 1)

    def test_eof_on_input_saves_partial(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "out.json")
            wb = self._make_workbook(3)

            def prompt_fn(_prompt_text: str) -> str:
                return None  # type: ignore[return-value]

            all_rows = run_interactive(
                workbook=wb,
                output_path=out_path,
                display_image_fn=lambda _row: None,
                prompt_fn=prompt_fn,
            )
            self.assertEqual(all_rows, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
