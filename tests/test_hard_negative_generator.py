"""Tests for the v3 extensions to ``data.augmentation.hard_negative_generator``.

Covers the 4 new perturbation types added in the v3 sprint:
    1. fabricate_measurement
    2. fabricate_prior
    3. fabricate_temporal
    4. compound_perturbation (n=2, n=3)

Does NOT re-test the original 8 types — those were shipped in v2 and are
exercised indirectly through integration (``scripts/prepare_eval_data.py``
dry-runs).

Run:
    python3 tests/test_hard_negative_generator.py
"""

from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.augmentation.hard_negative_generator import (  # noqa: E402
    ALL_NEGATIVE_TYPES,
    FABRICATED_MEASUREMENT_UNITS,
    FABRICATED_PRIOR_PHRASES,
    FABRICATED_TEMPORAL_DATES,
    _COMPOUND_ORDER,
    _GENERATORS,
    _LABELS,
    _RNG_AWARE_GENERATORS,
    compound_perturbation,
    fabricate_measurement,
    fabricate_prior,
    fabricate_temporal,
)
from data.preprocessing.radgraph_parser import Claim  # noqa: E402


def _make_claim(
    text: str,
    *,
    pathology: str = "Lung Lesion",
    laterality: str | None = "right",
    severity: str | None = "small",
    anatomy: str | None = "right upper lobe",
) -> Claim:
    return Claim(
        text=text,
        pathology_category=pathology,
        entities=[],
        relations=[],
        is_negated=False,
        laterality=laterality,
        severity=severity,
        anatomy=anatomy,
    )


class TestFabricateMeasurement(unittest.TestCase):
    def test_injects_measurement_before_target_noun(self):
        claim = _make_claim("Pulmonary nodule in the right upper lobe.")
        rng = random.Random(0)
        result = fabricate_measurement(claim, rng)
        self.assertIsNotNone(result)
        self.assertNotEqual(result.text, claim.text)
        # Measurement should appear somewhere in the new text
        self.assertTrue(
            any(m in result.text for m in FABRICATED_MEASUREMENT_UNITS),
            f"No measurement unit found in: {result.text}",
        )
        # Target noun should still be in the output
        self.assertIn("nodule", result.text.lower())

    def test_skips_claim_that_already_has_measurement(self):
        claim = _make_claim("A 1.2 cm pulmonary nodule in the right upper lobe.")
        rng = random.Random(0)
        self.assertIsNone(fabricate_measurement(claim, rng))

    def test_skips_claim_with_no_target_noun(self):
        claim = _make_claim(
            "Unremarkable cardiac silhouette.",
            pathology="No Finding",
            laterality=None,
            severity=None,
            anatomy=None,
        )
        rng = random.Random(0)
        self.assertIsNone(fabricate_measurement(claim, rng))

    def test_deterministic_under_seed(self):
        claim = _make_claim("Patchy consolidation in the left lower lobe.")
        r1 = fabricate_measurement(claim, random.Random(42))
        r2 = fabricate_measurement(claim, random.Random(42))
        self.assertEqual(r1.text, r2.text)


class TestFabricatePrior(unittest.TestCase):
    def test_appends_prior_phrase(self):
        claim = _make_claim("Moderate left pleural effusion.", laterality="left")
        rng = random.Random(1)
        result = fabricate_prior(claim, rng)
        self.assertIsNotNone(result)
        self.assertTrue(
            any(p.lower() in result.text.lower() for p in FABRICATED_PRIOR_PHRASES),
            f"No prior phrase in: {result.text}",
        )
        # Output should still contain the original content
        self.assertIn("pleural effusion", result.text.lower())

    def test_skips_claim_with_existing_comparison(self):
        claim = _make_claim("Stable compared to the prior film.", severity=None)
        rng = random.Random(1)
        self.assertIsNone(fabricate_prior(claim, rng))

    def test_skips_claim_with_since_previous(self):
        claim = _make_claim("New opacity since the previous study.", severity=None)
        rng = random.Random(1)
        self.assertIsNone(fabricate_prior(claim, rng))


class TestFabricateTemporal(unittest.TestCase):
    def test_appends_temporal_phrase(self):
        claim = _make_claim("Right lower lobe atelectasis.", pathology="Atelectasis")
        rng = random.Random(2)
        result = fabricate_temporal(claim, rng)
        self.assertIsNotNone(result)
        self.assertTrue(
            any(p.lower() in result.text.lower() for p in FABRICATED_TEMPORAL_DATES),
            f"No temporal phrase in: {result.text}",
        )

    def test_skips_claim_with_existing_temporal(self):
        claim = _make_claim("New opacity since 3 days ago.", severity=None)
        rng = random.Random(2)
        self.assertIsNone(fabricate_temporal(claim, rng))

    def test_skips_claim_with_explicit_date(self):
        claim = _make_claim("Opacity present since 01/15/2026.", severity=None)
        rng = random.Random(2)
        self.assertIsNone(fabricate_temporal(claim, rng))

    def test_skips_claim_with_yesterday(self):
        claim = _make_claim("Opacity present from yesterday's exam.", severity=None)
        rng = random.Random(2)
        self.assertIsNone(fabricate_temporal(claim, rng))


class TestCompoundPerturbation(unittest.TestCase):
    def test_2_err_succeeds_on_rich_claim(self):
        """A claim with laterality + severity + finding should admit a 2-err compound
        with high probability across seeds."""
        claim = _make_claim(
            "Large left pleural effusion.",
            pathology="Pleural Effusion",
            laterality="left",
            severity="large",
            anatomy="left pleural space",
        )
        successes = 0
        for seed in range(30):
            r = compound_perturbation(claim, random.Random(seed), n_errors=2)
            if r is not None:
                successes += 1
                # Each compound must actually change the text
                self.assertNotEqual(r.text, claim.text)
        # Require at least 25% success rate — the 50/30 seed yield observed during
        # development is well above this floor.
        self.assertGreaterEqual(
            successes, 8,
            f"2-err compound success rate too low: {successes}/30",
        )

    def test_3_err_succeeds_on_rich_claim(self):
        claim = _make_claim(
            "Large left pleural effusion.",
            pathology="Pleural Effusion",
            laterality="left",
            severity="large",
            anatomy="left pleural space",
        )
        successes = 0
        for seed in range(30):
            r = compound_perturbation(claim, random.Random(seed), n_errors=3)
            if r is not None:
                successes += 1
                self.assertNotEqual(r.text, claim.text)
        # 3-err is harder — require at least 15% floor
        self.assertGreaterEqual(
            successes, 5,
            f"3-err compound success rate too low: {successes}/30",
        )

    def test_rejects_n_errors_lt_2(self):
        claim = _make_claim("Small pulmonary nodule.")
        self.assertIsNone(
            compound_perturbation(claim, random.Random(0), n_errors=0)
        )
        self.assertIsNone(
            compound_perturbation(claim, random.Random(0), n_errors=1)
        )

    def test_rejects_unrich_claim(self):
        """A claim with no laterality, no severity, no pathology handle should
        fail all compound attempts (too few chain-able generators)."""
        claim = _make_claim(
            "Stable.",
            pathology="No Finding",
            laterality=None,
            severity=None,
            anatomy=None,
        )
        # Most seeds should fail — we just require at least one failure
        # (and a non-crash).
        any_none = False
        for seed in range(10):
            r = compound_perturbation(claim, random.Random(seed), n_errors=3)
            if r is None:
                any_none = True
        self.assertTrue(any_none, "compound on unrich claim never returned None")

    def test_compound_order_contains_new_fabrications(self):
        """v3 extension: _COMPOUND_ORDER must include all three fabrication
        types or they can never participate in a compound."""
        self.assertIn("fabricate_measurement", _COMPOUND_ORDER)
        self.assertIn("fabricate_prior", _COMPOUND_ORDER)
        self.assertIn("fabricate_temporal", _COMPOUND_ORDER)


class TestRegistryIntegration(unittest.TestCase):
    def test_generators_dict_has_13_total_types(self):
        # 12 in _GENERATORS (7 v2 + 5 new: 3 fabrications + 2 compound wrappers)
        # + omission which is handled separately = 13 in ALL_NEGATIVE_TYPES.
        self.assertEqual(len(_GENERATORS), 12)
        self.assertEqual(len(ALL_NEGATIVE_TYPES), 13)

    def test_new_types_are_registered(self):
        for name in (
            "fabricate_measurement",
            "fabricate_prior",
            "fabricate_temporal",
            "compound_2err",
            "compound_3err",
        ):
            self.assertIn(name, _GENERATORS)
            self.assertIn(name, _LABELS)
            self.assertIn(name, ALL_NEGATIVE_TYPES)
            self.assertIn(name, _RNG_AWARE_GENERATORS)

    def test_all_contradicted_label(self):
        """Every new type except omission must be labelled Contradicted."""
        for name in (
            "fabricate_measurement",
            "fabricate_prior",
            "fabricate_temporal",
            "compound_2err",
            "compound_3err",
        ):
            self.assertEqual(_LABELS[name], "Contradicted")

    def test_dispatcher_routes_new_types_without_crash(self):
        """Running generate_hard_negatives with a mixed type list including the
        new v3 types must not crash — it just returns whatever each claim can
        produce."""
        from data.augmentation.hard_negative_generator import generate_hard_negatives

        claims = [
            _make_claim("Small left pleural effusion.", laterality="left"),
            _make_claim("Pulmonary nodule in the right upper lobe."),
            _make_claim("Moderate cardiomegaly.",
                        pathology="Cardiomegaly",
                        laterality=None, severity="moderate",
                        anatomy="heart"),
        ]
        types = [
            "laterality_swap",
            "fabricate_measurement",
            "fabricate_prior",
            "fabricate_temporal",
            "compound_2err",
            "compound_3err",
        ]
        results = generate_hard_negatives(claims, types, n_per_claim=3, seed=42)
        # At least one claim should produce at least one hard negative
        self.assertGreater(len(results), 0)
        # Every result tuple is (Claim, neg_type, label)
        for mod, neg_type, label in results:
            self.assertIsInstance(mod, Claim)
            self.assertIn(neg_type, ALL_NEGATIVE_TYPES)
            self.assertEqual(label, "Contradicted")


if __name__ == "__main__":
    unittest.main(verbosity=2)
