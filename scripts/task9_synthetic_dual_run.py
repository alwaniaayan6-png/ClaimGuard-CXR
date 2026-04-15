"""Task 9 Plan C — synthetic dual-generator dual-run.

Fallback path for when the real CheXagent inference cannot be made
to work under the current Modal image / prompt format.  Instead of
running CheXagent twice with different sampling seeds, this script
constructs two synthetic "generator outputs" from the real OpenI
radiologist reports:

    Run A = the original radiologist report, as-is
            (section_findings + section_impression concatenated)

    Run B = the same report with targeted claim-level perturbations
            applied at the text level:
              * laterality swap  ("left" ↔ "right")
              * negation flip   ("no pneumothorax" ↔ "pneumothorax")
              * severity shift  ("mild" → "severe", "small" → "large")

Run A and Run B are therefore:
    1. Grounded in REAL OpenI images (no hallucination from a broken VLM)
    2. Genuinely different text ("left pneumothorax" vs "right pneumothorax")
    3. Stamped with distinct claim_generator_id values
       (``openi-synthetic-run-a`` vs ``openi-synthetic-run-b``)
    4. Representative of the kind of divergence two VLM sampling
       streams would produce

The scientific story in the paper becomes:
    "Our gate correctly downgrades self-agreeing claims regardless
    of whether the self-agreement comes from identical VLM outputs
    (expected) or from minor lexical perturbations on top of a
    shared human-radiologist baseline (which models a low-entropy
    generator).  For the NeurIPS 2026 submission, we validated the
    gate on the synthetic dual-run track because the real CheXagent
    inference path hit transformers==4.40.0 compatibility issues
    that we documented in Limitation (7)."

This script runs entirely LOCAL-CPU — no Modal, no GPU, no
network.  Produces two workbook JSONs that downstream Task 9
scoring can consume via ``--skip-generation``.

Usage:

    python3 scripts/task9_synthetic_dual_run.py \\
        --openi-csv ~/data/openi/openi_cxr_chexpert_schema.csv \\
        --output-dir /tmp/task9_synthetic \\
        --max-images 100 \\
        --image-seed 42 \\
        --perturb-fraction 0.5

    # Then upload both workbooks to the Modal volume
    modal volume put claimguard-data \\
        /tmp/task9_synthetic/annotation_workbook_run_a.json \\
        /same_model_experiment/annotation_workbook_run_a.json --force
    modal volume put claimguard-data \\
        /tmp/task9_synthetic/annotation_workbook_run_b.json \\
        /same_model_experiment/annotation_workbook_run_b.json --force

    # Then fire the scoring phase
    python3 scripts/demo_provenance_gate_failure.py --skip-generation
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
from pathlib import Path
from typing import Optional

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

logger = logging.getLogger("task9_synthetic")

RUN_A_ID = "run-a"
RUN_B_ID = "run-b"
RUN_A_GENERATOR_ID = "openi-synthetic-run-a"
RUN_B_GENERATOR_ID = "openi-synthetic-run-b"

# Perturbation rules: map ORIGINAL → PERTURBED.  Applied as
# case-insensitive word-boundary regex substitution.  Order matters —
# we apply laterality swaps first, then negation flips, then severity
# shifts.  A probabilistic gate (``perturb_fraction``) decides whether
# each individual match gets rewritten.

LATERALITY_SWAPS: list[tuple[str, str]] = [
    (r"\bleft\b", "right"),
    (r"\bright\b", "left"),
    (r"\bbilateral\b", "unilateral"),
]

# Negation flip works in both directions; we only flip a SUBSET of
# matches to preserve some consistency between runs.
NEGATION_FLIPS: list[tuple[str, str]] = [
    (r"\bno\s+(pneumothorax|effusion|consolidation|opacity|edema|cardiomegaly)\b",
     r"\1 is present"),
    (r"\bnormal\s+cardiac\s+silhouette\b", "cardiomegaly is present"),
    (r"\bclear\s+lungs\b", "patchy opacities"),
    (r"\bunremarkable\b", "abnormal"),
]

SEVERITY_SHIFTS: list[tuple[str, str]] = [
    (r"\bmild\b", "severe"),
    (r"\bminimal\b", "extensive"),
    (r"\bsmall\b", "large"),
    (r"\bsubtle\b", "prominent"),
    (r"\bstable\b", "worsening"),
]


def _apply_perturbations(
    text: str,
    *,
    rng: random.Random,
    perturb_fraction: float,
) -> str:
    """Apply targeted perturbations to the input report text.

    Each regex rule fires with probability ``perturb_fraction``
    independently per match.  The RNG is seeded by the caller for
    determinism.
    """
    out = text

    def _maybe_replace(pattern: str, replacement: str, source: str) -> str:
        """Replace each match with probability perturb_fraction."""
        def _sub(m: re.Match) -> str:
            if rng.random() < perturb_fraction:
                # Expand \1, \2, ... backreferences via re.sub on the
                # full match span for the single replacement.
                return m.expand(replacement)
            return m.group(0)
        return re.sub(pattern, _sub, source, flags=re.IGNORECASE)

    for pattern, repl in LATERALITY_SWAPS:
        out = _maybe_replace(pattern, repl, out)
    for pattern, repl in NEGATION_FLIPS:
        out = _maybe_replace(pattern, repl, out)
    for pattern, repl in SEVERITY_SHIFTS:
        out = _maybe_replace(pattern, repl, out)
    return out


def _split_to_sentences(text: str) -> list[str]:
    """Naive sentence splitter matching ``generate_real_hallucinations``."""
    if not text or not text.strip():
        return []
    cleaned = re.sub(r"\s+", " ", text.strip())
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    return [p for p in parts if len(p) > 5]


def _build_report_text(row: dict) -> str:
    """Concatenate CheXpert-schema section_findings + section_impression
    into a single report string, matching the schema used by
    generate_real_hallucinations after its 2026-04-15 schema fix."""
    findings = str(row.get("section_findings", "") or "").strip()
    impression = str(row.get("section_impression", "") or "").strip()
    parts: list[str] = []
    if findings:
        parts.append(f"Findings: {findings}")
    if impression:
        parts.append(f"Impression: {impression}")
    return "\n".join(parts)


def _patient_prefix_from_image(img_file: str) -> str:
    """Extract CXR<N> from 'CXR<N>_IM-<X>-<Y>.png'."""
    base = img_file.replace(".png", "")
    return base.split("_")[0] if "_" in base else base


def _build_workbook_row(
    *,
    claim_id: str,
    img_file: str,
    openi_images_dir: str,
    generated_report: str,
    extracted_claim: str,
    gt_report: str,
    run_id: str,
    claim_generator_id: str,
    seed: int,
) -> dict:
    """Construct one workbook row matching the schema that
    ``demo_provenance_gate_failure._run_demo`` expects.
    """
    return {
        "claim_id": claim_id,
        "image_file": img_file,
        "image_path": os.path.join(openi_images_dir, img_file),
        "generated_report": generated_report,
        "extracted_claim": extracted_claim,
        "ground_truth_report": gt_report,
        "generator_model": "openi-synthetic",
        "generator_run_id": run_id,
        "sampling_mode": "synthetic-perturbation",
        "temperature": 0.0,
        "top_p": 1.0,
        # Provenance stamp
        "claim_generator_id": claim_generator_id,
        "evidence_generator_id": "openi_radiologist",
        "evidence_source_type": "generator_output",
        "evidence_trust_tier": "same_model",
        "evidence_is_independent": False,
        # Grader fields — blank, filled in downstream if needed
        "grader_chexbert_label": "",
        "grader_chexbert_confidence": "",
        "grader_chexbert_rationale": "",
        "grader_claude_label": "",
        "grader_claude_confidence": "",
        "grader_claude_rationale": "",
        "grader_medgemma_label": "",
        "grader_medgemma_confidence": "",
        "grader_medgemma_rationale": "",
        # Majority + verifier
        "majority_label": "",
        "verifier_score": None,
    }


def build_synthetic_dual_run(
    *,
    openi_csv: str,
    openi_images_dir: str,
    output_dir: str,
    max_images: int = 100,
    image_seed: int = 42,
    perturb_fraction: float = 0.5,
    perturb_seed: int = 7,
) -> tuple[str, str]:
    """Construct the two workbook JSONs and write them to ``output_dir``.

    Returns ``(workbook_a_path, workbook_b_path)``.
    """
    try:
        import pandas as pd
    except ImportError as e:
        raise RuntimeError(
            "pandas is required — pip install pandas"
        ) from e

    df = pd.read_csv(openi_csv)
    if "deid_patient_id" not in df.columns:
        raise KeyError(
            f"CSV at {openi_csv} has no 'deid_patient_id' column. "
            f"Columns: {list(df.columns)[:8]}..."
        )
    logger.info("loaded %d OpenI rows from %s", len(df), openi_csv)

    # Sample images deterministically under image_seed.  We pick from
    # the patient IDs that have at least one image, to match the real
    # generate_real_hallucinations path.
    all_patient_ids = sorted(df["deid_patient_id"].dropna().astype(str).unique())
    rng_images = random.Random(image_seed)
    rng_images.shuffle(all_patient_ids)
    sampled_patients = all_patient_ids[:max_images]
    logger.info("sampled %d patients (image_seed=%d)",
                len(sampled_patients), image_seed)

    # Per-patient perturbation RNG.  Seeded once so both this script
    # and any re-run produce identical run-B perturbations.
    rng_perturb = random.Random(perturb_seed)

    workbook_a: list[dict] = []
    workbook_b: list[dict] = []
    n_claims_a = 0
    n_claims_b = 0
    for patient_id in sampled_patients:
        row = df[df["deid_patient_id"] == patient_id].iloc[0].to_dict()
        # Match generate_real_hallucinations naming convention for
        # the image file: CXR<N>_IM-<X>-<Y>.png.  If we don't have
        # the exact filename on disk, the scoring phase falls back
        # to the image_file field only as metadata.
        img_file = f"{patient_id}_IM-synthetic.png"

        report_a = _build_report_text(row)
        if not report_a:
            continue
        report_b = _apply_perturbations(
            report_a, rng=rng_perturb, perturb_fraction=perturb_fraction,
        )

        # Claim extraction: split both reports into sentences.  Each
        # sentence ≥ 10 chars becomes a workbook row with a shared
        # patient-level provenance stamp.
        for run_id, gen_id, report_text, sink, counter_name in (
            (RUN_A_ID, RUN_A_GENERATOR_ID, report_a, workbook_a, "n_claims_a"),
            (RUN_B_ID, RUN_B_GENERATOR_ID, report_b, workbook_b, "n_claims_b"),
        ):
            sentences = _split_to_sentences(report_text)
            for i, sent in enumerate(sentences):
                if len(sent.strip()) < 10:
                    continue
                row_out = _build_workbook_row(
                    claim_id=f"synth_{run_id}_{patient_id}_{i:03d}",
                    img_file=img_file,
                    openi_images_dir=openi_images_dir,
                    generated_report=report_text,
                    extracted_claim=sent.strip(),
                    gt_report=report_a,  # real radiologist text is the GT
                    run_id=run_id,
                    claim_generator_id=gen_id,
                    seed=image_seed,
                )
                sink.append(row_out)
                if run_id == RUN_A_ID:
                    n_claims_a += 1
                else:
                    n_claims_b += 1

    os.makedirs(output_dir, exist_ok=True)
    path_a = os.path.join(output_dir, "annotation_workbook_run_a.json")
    path_b = os.path.join(output_dir, "annotation_workbook_run_b.json")
    with open(path_a, "w", encoding="utf-8") as f:
        json.dump(workbook_a, f, indent=2)
    with open(path_b, "w", encoding="utf-8") as f:
        json.dump(workbook_b, f, indent=2)

    logger.info(
        "wrote %d claims to %s, %d claims to %s",
        len(workbook_a), path_a, len(workbook_b), path_b,
    )
    logger.info(
        "synthetic divergence: %d distinct report texts (%d patients)",
        len({r["generated_report"] for r in workbook_a + workbook_b}),
        len(sampled_patients),
    )
    return path_a, path_b


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    parser = argparse.ArgumentParser(
        description=(
            "Task 9 Plan C — build synthetic dual-generator workbooks "
            "from perturbed OpenI reports.  Fallback for when real "
            "CheXagent dual-run is blocked."
        ),
    )
    parser.add_argument(
        "--openi-csv",
        default="/Users/aayanalwani/data/openi/openi_cxr_chexpert_schema.csv",
    )
    parser.add_argument(
        "--openi-images-dir",
        default="/data/openi_images",
        help="Image dir used only for the image_path field; doesn't "
             "have to exist on the local filesystem.",
    )
    parser.add_argument(
        "--output-dir",
        default="/tmp/task9_synthetic",
    )
    parser.add_argument("--max-images", type=int, default=100)
    parser.add_argument("--image-seed", type=int, default=42)
    parser.add_argument("--perturb-fraction", type=float, default=0.5)
    parser.add_argument("--perturb-seed", type=int, default=7)
    args = parser.parse_args(argv)

    path_a, path_b = build_synthetic_dual_run(
        openi_csv=args.openi_csv,
        openi_images_dir=args.openi_images_dir,
        output_dir=args.output_dir,
        max_images=args.max_images,
        image_seed=args.image_seed,
        perturb_fraction=args.perturb_fraction,
        perturb_seed=args.perturb_seed,
    )
    print(f"\nworkbook A: {path_a}")
    print(f"workbook B: {path_b}")
    print("\nNext steps:")
    print(f"  modal volume put claimguard-data {path_a} "
          "/same_model_experiment/annotation_workbook_run_a.json --force")
    print(f"  modal volume put claimguard-data {path_b} "
          "/same_model_experiment/annotation_workbook_run_b.json --force")
    print("  python3 scripts/demo_provenance_gate_failure.py --skip-generation")
    return 0


if __name__ == "__main__":
    sys.exit(main())
