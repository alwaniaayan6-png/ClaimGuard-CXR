"""Tests for ``data.augmentation.counterfactual_generator``.

Focuses on the pure helpers and the transport-injected code path:

* ``normalize_causal_tokens`` — whitespace / dedupe
* ``build_prompt`` — shape + required substrings
* ``parse_variants_json`` — preamble / markdown / malformed
* ``validate_preservation`` — substring match, case-insensitive
* ``levenshtein_distance`` — known reference values
* ``CounterfactualGenerator`` with a fake transport — happy path,
  dedupe, retry-on-parse-error, max_retries exhaustion, CausalSpan
  input, empty input
* ``CounterfactualGenerator`` ImportError path when ``anthropic``
  is unavailable (sys.modules patch)

No network / no Anthropic SDK required.
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from unittest import mock

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from data.augmentation.causal_term_identifier import CausalSpan  # noqa: E402
from data.augmentation.counterfactual_generator import (  # noqa: E402
    CounterfactualGenerator,
    CounterfactualVariant,
    build_prompt,
    levenshtein_distance,
    normalize_causal_tokens,
    parse_variants_json,
    validate_preservation,
)


# ---------------------------------------------------------------------------
# normalize_causal_tokens
# ---------------------------------------------------------------------------


class TestNormalizeCausalTokens(unittest.TestCase):
    def test_strips_leading_trailing_whitespace(self) -> None:
        self.assertEqual(
            normalize_causal_tokens([" enlarged", "heart ", " beat "]),
            ["enlarged", "heart", "beat"],
        )

    def test_dedupes_case_sensitive(self) -> None:
        # Case-sensitive dedupe: "heart" and "Heart" are distinct.
        out = normalize_causal_tokens(["heart", "Heart", "heart"])
        self.assertEqual(out, ["heart", "Heart"])

    def test_drops_empty_strings(self) -> None:
        self.assertEqual(
            normalize_causal_tokens(["heart", "", "  ", "beat"]),
            ["heart", "beat"],
        )

    def test_drops_non_strings(self) -> None:
        out = normalize_causal_tokens(["heart", 42, None, "beat"])  # type: ignore[list-item]
        self.assertEqual(out, ["heart", "beat"])

    def test_preserves_internal_whitespace(self) -> None:
        # Multi-word merged spans must stay intact.
        out = normalize_causal_tokens([" heart beat "])
        self.assertEqual(out, ["heart beat"])

    def test_empty_input(self) -> None:
        self.assertEqual(normalize_causal_tokens([]), [])


# ---------------------------------------------------------------------------
# build_prompt
# ---------------------------------------------------------------------------


class TestBuildPrompt(unittest.TestCase):
    def test_contains_claim_and_tokens(self) -> None:
        prompt = build_prompt(
            claim="heart is enlarged",
            causal_tokens=["heart", "enlarged"],
            n_variants=3,
        )
        self.assertIn("heart is enlarged", prompt)
        self.assertIn("heart", prompt)
        self.assertIn("enlarged", prompt)

    def test_requests_exact_n_variants(self) -> None:
        prompt = build_prompt(
            claim="x", causal_tokens=["y"], n_variants=5,
        )
        self.assertIn("5", prompt)

    def test_mentions_verbatim_preservation(self) -> None:
        prompt = build_prompt(
            claim="x", causal_tokens=["y"], n_variants=3,
        )
        # The prompt must make it crystal-clear the tokens are pinned.
        self.assertIn("verbatim", prompt)

    def test_tokens_serialized_as_json(self) -> None:
        prompt = build_prompt(
            claim="x", causal_tokens=["heart", "enlarged"], n_variants=3,
        )
        # JSON-encoded form must appear so the LLM sees a parseable list.
        self.assertIn('["heart", "enlarged"]', prompt)


# ---------------------------------------------------------------------------
# parse_variants_json
# ---------------------------------------------------------------------------


class TestParseVariantsJson(unittest.TestCase):
    def test_clean_json_array(self) -> None:
        raw = '["a", "b", "c"]'
        self.assertEqual(parse_variants_json(raw), ["a", "b", "c"])

    def test_with_preamble(self) -> None:
        raw = 'Sure, here are 3 variants:\n["a", "b", "c"]'
        self.assertEqual(parse_variants_json(raw), ["a", "b", "c"])

    def test_with_markdown_fence_json_tag(self) -> None:
        raw = '```json\n["a", "b", "c"]\n```'
        self.assertEqual(parse_variants_json(raw), ["a", "b", "c"])

    def test_with_markdown_fence_bare(self) -> None:
        raw = '```\n["a", "b", "c"]\n```'
        self.assertEqual(parse_variants_json(raw), ["a", "b", "c"])

    def test_object_instead_of_array_raises(self) -> None:
        with self.assertRaises(ValueError):
            parse_variants_json('{"variants": ["a"]}')

    def test_empty_string_raises(self) -> None:
        with self.assertRaises(ValueError):
            parse_variants_json("")

    def test_none_raises(self) -> None:
        with self.assertRaises(ValueError):
            parse_variants_json(None)

    def test_whitespace_only_raises(self) -> None:
        with self.assertRaises(ValueError):
            parse_variants_json("   \n  \t")

    def test_garbage_raises(self) -> None:
        with self.assertRaises(ValueError):
            parse_variants_json("who knows")

    def test_empty_array_raises(self) -> None:
        with self.assertRaises(ValueError):
            parse_variants_json("[]")

    def test_mixed_types_filters_non_strings(self) -> None:
        # Integers / nulls are dropped; strings are kept.
        raw = '["a", 42, null, "b", ""]'
        self.assertEqual(parse_variants_json(raw), ["a", "b"])

    def test_strips_per_item_whitespace(self) -> None:
        raw = '[" a ", "b  "]'
        self.assertEqual(parse_variants_json(raw), ["a", "b"])

    def test_trailing_prose_after_array(self) -> None:
        # The trailing prose must be ignored — we fall back to regex
        # extraction of the first JSON array.
        raw = '["a", "b"] -- I hope that helps!'
        self.assertEqual(parse_variants_json(raw), ["a", "b"])


# ---------------------------------------------------------------------------
# validate_preservation
# ---------------------------------------------------------------------------


class TestValidatePreservation(unittest.TestCase):
    def test_all_tokens_present(self) -> None:
        missing = validate_preservation(
            "the heart is enlarged",
            ["heart", "enlarged"],
        )
        self.assertEqual(missing, [])

    def test_one_token_missing(self) -> None:
        missing = validate_preservation(
            "cardiac enlargement is seen",
            ["heart", "enlarged"],
        )
        self.assertEqual(missing, ["heart", "enlarged"])

    def test_case_insensitive_match(self) -> None:
        # "Heart" in the variant still counts as "heart" preserved.
        missing = validate_preservation(
            "The Heart Is Enlarged",
            ["heart", "enlarged"],
        )
        self.assertEqual(missing, [])

    def test_empty_required_tokens(self) -> None:
        self.assertEqual(validate_preservation("anything", []), [])

    def test_empty_token_in_required_is_skipped(self) -> None:
        missing = validate_preservation(
            "heart is big",
            ["heart", "", "big"],
        )
        self.assertEqual(missing, [])

    def test_multiword_token_preserved(self) -> None:
        missing = validate_preservation(
            "the left lower lobe shows opacity",
            ["left lower lobe"],
        )
        self.assertEqual(missing, [])

    def test_multiword_token_broken_up(self) -> None:
        missing = validate_preservation(
            "the lower left lobe shows opacity",  # order swapped
            ["left lower lobe"],
        )
        self.assertEqual(missing, ["left lower lobe"])


# ---------------------------------------------------------------------------
# levenshtein_distance
# ---------------------------------------------------------------------------


class TestLevenshteinDistance(unittest.TestCase):
    def test_identical(self) -> None:
        self.assertEqual(levenshtein_distance("heart", "heart"), 0)

    def test_empty_a(self) -> None:
        self.assertEqual(levenshtein_distance("", "heart"), 5)

    def test_empty_b(self) -> None:
        self.assertEqual(levenshtein_distance("heart", ""), 5)

    def test_both_empty(self) -> None:
        self.assertEqual(levenshtein_distance("", ""), 0)

    def test_canonical_kitten_sitting(self) -> None:
        self.assertEqual(levenshtein_distance("kitten", "sitting"), 3)

    def test_single_substitution(self) -> None:
        self.assertEqual(levenshtein_distance("cat", "bat"), 1)

    def test_single_deletion(self) -> None:
        self.assertEqual(levenshtein_distance("heart", "hear"), 1)

    def test_single_insertion(self) -> None:
        self.assertEqual(levenshtein_distance("hear", "heart"), 1)

    def test_symmetric(self) -> None:
        self.assertEqual(
            levenshtein_distance("radiology", "radiologist"),
            levenshtein_distance("radiologist", "radiology"),
        )


# ---------------------------------------------------------------------------
# CounterfactualGenerator with injected transport
# ---------------------------------------------------------------------------


class TestCounterfactualGeneratorHappyPath(unittest.TestCase):
    def test_happy_path_returns_n_variants(self) -> None:
        def fake_transport(prompt: str) -> str:
            return json.dumps([
                "the heart appears enlarged",
                "an enlarged heart is visible",
                "enlarged heart size seen",
            ])

        gen = CounterfactualGenerator(transport=fake_transport)
        variants = gen.generate(
            claim="heart is enlarged",
            causal_spans=["heart", "enlarged"],
            n_variants=3,
        )
        self.assertEqual(len(variants), 3)
        for v in variants:
            self.assertIsInstance(v, CounterfactualVariant)
            self.assertIn("heart", v.text.lower())
            self.assertIn("enlarged", v.text.lower())
            self.assertGreater(v.edit_distance, 0)

    def test_sorted_by_edit_distance_ascending(self) -> None:
        def fake_transport(prompt: str) -> str:
            return json.dumps([
                "heart is very much enlarged to a substantial degree",
                "heart enlarged",
                "the heart is enlarged today",
            ])

        gen = CounterfactualGenerator(transport=fake_transport)
        variants = gen.generate(
            claim="heart is enlarged",
            causal_spans=["heart", "enlarged"],
            n_variants=3,
        )
        # Smallest edit distance first.
        distances = [v.edit_distance for v in variants]
        self.assertEqual(distances, sorted(distances))

    def test_accepts_causal_span_input(self) -> None:
        def fake_transport(prompt: str) -> str:
            return json.dumps([
                "heart appears enlarged today",
                "enlarged heart is noted here",
                "heart size enlarged on image",
            ])

        gen = CounterfactualGenerator(transport=fake_transport)
        spans = [
            CausalSpan(
                text="heart", source="claim", score=0.9,
                start_char=0, end_char=5,
            ),
            CausalSpan(
                text=" enlarged", source="claim", score=0.8,
                start_char=8, end_char=17,
            ),
        ]
        variants = gen.generate(
            claim="heart is enlarged",
            causal_spans=spans,
            n_variants=3,
        )
        # Leading-space causal token is normalized to "enlarged".
        self.assertEqual(len(variants), 3)
        for v in variants:
            self.assertIn("heart", v.text.lower())
            self.assertIn("enlarged", v.text.lower())


class TestCounterfactualGeneratorValidationLoop(unittest.TestCase):
    def test_drops_variants_missing_causal_tokens(self) -> None:
        def fake_transport(prompt: str) -> str:
            return json.dumps([
                "heart is clearly enlarged today",        # valid
                "cardiac size is increased",              # INVALID - no "heart"
                "an enlarged heart is present always",    # valid
            ])

        gen = CounterfactualGenerator(transport=fake_transport)
        variants = gen.generate(
            claim="heart is enlarged",
            causal_spans=["heart", "enlarged"],
            n_variants=3,
        )
        self.assertEqual(len(variants), 2)

    def test_drops_variant_identical_to_claim(self) -> None:
        def fake_transport(prompt: str) -> str:
            return json.dumps([
                "heart is enlarged",                 # identical → dropped
                "enlarged heart is evident today",   # valid
                "the heart is enlarged clearly",     # valid
            ])

        gen = CounterfactualGenerator(
            transport=fake_transport, max_retries=0,
        )
        variants = gen.generate(
            claim="heart is enlarged",
            causal_spans=["heart", "enlarged"],
            n_variants=3,
        )
        texts = {v.text for v in variants}
        self.assertNotIn("heart is enlarged", texts)
        self.assertEqual(len(variants), 2)

    def test_dedupes_repeated_variants_across_attempts(self) -> None:
        # Same 1 valid variant on both attempts → 1 result, not 2.
        def fake_transport(prompt: str) -> str:
            return json.dumps([
                "heart is markedly enlarged",
                "cardiac shape differs",   # missing heart
                "size changes observed",   # missing both
            ])

        gen = CounterfactualGenerator(
            transport=fake_transport, max_retries=2,
        )
        variants = gen.generate(
            claim="heart is enlarged",
            causal_spans=["heart", "enlarged"],
            n_variants=3,
        )
        self.assertEqual(len(variants), 1)
        self.assertEqual(variants[0].text, "heart is markedly enlarged")

    def test_retry_on_parse_failure(self) -> None:
        call_counter = {"n": 0}

        def fake_transport(prompt: str) -> str:
            call_counter["n"] += 1
            if call_counter["n"] == 1:
                return "totally unparseable garbage"
            return json.dumps([
                "enlarged heart noted clearly",
                "heart appears enlarged on exam",
                "the heart seems enlarged today",
            ])

        gen = CounterfactualGenerator(
            transport=fake_transport, max_retries=1,
        )
        variants = gen.generate(
            claim="heart is enlarged",
            causal_spans=["heart", "enlarged"],
            n_variants=3,
        )
        self.assertEqual(len(variants), 3)
        self.assertEqual(call_counter["n"], 2)

    def test_max_retries_exhausted_returns_partial(self) -> None:
        # Always returns 1 valid unique variant — dedupe caps us there.
        def fake_transport(prompt: str) -> str:
            return json.dumps([
                "heart is somewhat enlarged overall",
                "cardiac mass is increased",   # missing heart
                "shape changes",               # missing both
            ])

        gen = CounterfactualGenerator(
            transport=fake_transport, max_retries=3,
        )
        variants = gen.generate(
            claim="heart is enlarged",
            causal_spans=["heart", "enlarged"],
            n_variants=3,
        )
        self.assertEqual(len(variants), 1)


class TestCounterfactualGeneratorEdgeCases(unittest.TestCase):
    def test_empty_claim_returns_empty(self) -> None:
        gen = CounterfactualGenerator(
            transport=lambda _: '["x", "y", "z"]',
        )
        self.assertEqual(gen.generate("", ["heart"]), [])
        self.assertEqual(gen.generate("  ", ["heart"]), [])

    def test_empty_causal_spans_returns_empty(self) -> None:
        gen = CounterfactualGenerator(
            transport=lambda _: '["x", "y", "z"]',
        )
        self.assertEqual(
            gen.generate("heart is enlarged", []),
            [],
        )

    def test_n_variants_zero_returns_empty(self) -> None:
        gen = CounterfactualGenerator(
            transport=lambda _: '["x", "y", "z"]',
        )
        self.assertEqual(
            gen.generate(
                "heart is enlarged",
                ["heart"],
                n_variants=0,
            ),
            [],
        )

    def test_transport_exception_is_swallowed(self) -> None:
        # A transport error on the first attempt should not crash the
        # generator — it logs and retries (or returns partial).
        def flaky_transport(prompt: str) -> str:
            raise RuntimeError("transport boom")

        gen = CounterfactualGenerator(
            transport=flaky_transport, max_retries=1,
        )
        result = gen.generate(
            "heart is enlarged", ["heart"], n_variants=3,
        )
        self.assertEqual(result, [])


# ---------------------------------------------------------------------------
# Import-path / ImportError surface
# ---------------------------------------------------------------------------


class TestCounterfactualGeneratorImportPath(unittest.TestCase):
    def test_class_importable_without_anthropic(self) -> None:
        self.assertTrue(callable(CounterfactualGenerator))

    def test_missing_anthropic_raises_on_default_transport(self) -> None:
        with mock.patch.dict(sys.modules, {"anthropic": None}):
            with self.assertRaises(ImportError) as cm:
                CounterfactualGenerator()  # no transport injected
        self.assertIn("anthropic", str(cm.exception))
        self.assertIn("pip install", str(cm.exception))

    def test_missing_anthropic_ok_when_transport_injected(self) -> None:
        # If the caller provides a transport, anthropic is never imported.
        with mock.patch.dict(sys.modules, {"anthropic": None}):
            gen = CounterfactualGenerator(transport=lambda _: "[]")
            self.assertIsNotNone(gen)


if __name__ == "__main__":
    unittest.main(verbosity=2)
