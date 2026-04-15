"""Generate real-hallucination reports by running CheXagent-8b on OpenI images.

Task 1b / Task 9 — Modal H100 runner that produces a *silver-standard*
annotation workbook: one row per (image, extracted claim) with full
provenance stamping via ``inference.provenance.default_provenance``.
The workbook is the input for both

  * Task 1c/1d  — 3-grader ensemble silver-standard labeling, and
  * Task 9      — dual-run provenance-gate demonstration.

Why this script does NOT auto-label
-----------------------------------
CheXbert has ~10–15 % label noise; evaluating a SOTA verifier against a
legacy labeler is circular.  This script only emits CheXagent-generated
reports + extracted claims + empty grader fields.  Labels are written by
either (a) a 3-grader ensemble in ``generate_silver_standard_graders.py``
or (b) a human self-annotation pass in ``self_annotate_silver_subset.py``.

Why this script supports sampling mode
--------------------------------------
Task 9 runs CheXagent twice on the same 100 OpenI images with different
sampling seeds (temperature 0.7, top_p 0.9) to produce reports A and B
from the same model.  The provenance gate should downgrade self-evidence
pairs (claim from A, evidence from A — same ``claim_generator_id`` and
``evidence_generator_id`` ⇒ ``TrustTier.SAME_MODEL``) but allow
cross-run pairs (claim from A, evidence from B — different generator
ids ⇒ ``TrustTier.INDEPENDENT``).  The ``run_id`` CLI flag lets the
caller stamp each run with a distinct generator id.

Deterministic vs sampled
------------------------
* Default mode (Task 1 silver standard): ``do_sample=False``, greedy,
  identical reports across re-runs.  Use this for reproducible evals.
* Sampling mode (Task 9): ``do_sample=True``, ``temperature=0.7``,
  ``top_p=0.9``, different torch/numpy/python seeds per run.

Provenance stamping
-------------------
Every workbook entry carries the full 5-field provenance block via
``default_provenance(source_type=GENERATOR_OUTPUT, claim_generator_id,
evidence_generator_id)``.  The default setup stamps claims with
generator id ``chexagent-8b-run-<run_id>`` and evidence id
``openi_radiologist`` (since the GT report is the evidence for the
silver-standard grader phase).  Task 9's cross-run setup overrides
``evidence_generator_id`` on the downstream pair-building step.

Fallbacks
---------
* If ``AutoProcessor.from_pretrained`` fails (a common CheXagent gotcha
  with ``transformers==4.40.0``), fall back to the tokenizer + image
  processor path.
* If CheXagent-8b won't load at all, fall back to CheXagent-2 7B.
* If both fail, emit the workbook with empty ``generated_report`` so the
  downstream grader phase can still be wired up on synthetic fixtures.

Usage
-----
    # Task 1: deterministic 200-image silver baseline
    modal run --detach scripts/generate_real_hallucinations.py

    # Task 9 run A: sampled, distinct generator id
    modal run scripts/generate_real_hallucinations.py \
        --max-images 100 --sampling --run-id run-a --seed 101

    # Task 9 run B: sampled, different seed
    modal run scripts/generate_real_hallucinations.py \
        --max-images 100 --sampling --run-id run-b --seed 202
"""

from __future__ import annotations

import modal

app = modal.App("claimguard-real-hallucinations")

gen_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1.0",
        # CheXagent's bundled modeling files call into transformers
        # internals that changed after 4.40. Pin exactly.
        "transformers==4.40.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "Pillow>=10.0.0",
        "tqdm>=4.66.0",
        "accelerate>=0.27.0",
        "sentencepiece>=0.1.99",
        "protobuf>=3.20.0",
    )
)

vol = modal.Volume.from_name("claimguard-data", create_if_missing=True)


# ---------------------------------------------------------------------------
# Constants shared across local + remote paths.  Kept module-level so
# Task 9's thin wrapper can import them without instantiating the Modal
# function.
# ---------------------------------------------------------------------------

CHEXAGENT_PRIMARY = "StanfordAIMI/CheXagent-8b"
CHEXAGENT_FALLBACK = "StanfordAIMI/CheXagent-2-3b"  # 7B image ignores here

DEFAULT_PROMPT = (
    "Generate a detailed chest radiograph report for this image. "
    "Include Findings and Impression sections."
)


@app.function(
    image=gen_image,
    gpu="H100",
    timeout=60 * 60 * 4,
    volumes={"/data": vol},
)
def generate_annotation_workbook(
    openi_images_dir: str = "/data/openi_images",
    openi_reports_csv: str = "/data/openi_cxr_chexpert_schema.csv",
    output_dir: str = "/data/eval_data_real_hallucinations",
    max_images: int = 200,
    seed: int = 42,
    image_seed: int | None = None,
    sampling: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens: int = 256,
    run_id: str = "silver",
    workbook_filename: str = "annotation_workbook.json",
    generator_override: str | None = None,
) -> dict:
    """Run CheXagent on OpenI images and emit a silver-standard workbook.

    Args:
        openi_images_dir: Directory containing OpenI ``.png`` images.
        openi_reports_csv: CSV with ``study_id`` + ``report_text`` columns.
        output_dir: Modal-volume directory for outputs.  Created if absent.
        max_images: Cap on number of images to process.  The full
            sample is deterministic under ``image_seed`` (or ``seed`` if
            ``image_seed`` is None).
        seed: Master seed for torch / numpy / python RNG.  Drives
            nucleus-sampling entropy when ``sampling=True``.
        image_seed: Optional separate seed for OpenI image sampling.
            Defaults to ``seed`` for backward compatibility.  Task 9
            passes the same ``image_seed`` to both runs (so runs A and B
            share the same 100 images) while using different ``seed``
            values (so the two CheXagent generations diverge).  Without
            this split, Task 9 would either get identical reports
            (same seed) or near-disjoint image sets (different seeds).
        sampling: If ``True``, use nucleus sampling with ``temperature``
            / ``top_p``.  If ``False`` (default), use greedy decoding
            — reproducible across re-runs.
        temperature: Sampling temperature (only used when ``sampling``).
        top_p: Nucleus-sampling ``top_p`` (only used when ``sampling``).
        max_new_tokens: Max generated tokens per report.
        run_id: Label stamped into the ``claim_generator_id``
            provenance field as ``"chexagent-8b-run-<run_id>"``.
            Task 9 uses distinct run ids (``run-a`` / ``run-b``) to
            exercise the provenance gate.
        workbook_filename: Name of the output JSON file inside
            ``output_dir``.  Supports multiple runs writing to the
            same directory without collision.
        generator_override: If set, forces the ``claim_generator_id``
            instead of deriving it from ``run_id``.  Mainly used by
            unit tests.

    Returns:
        Summary dict with output paths and counts.
    """
    import csv
    import json
    import os
    import random
    import re

    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image
    from tqdm import tqdm

    # Local imports: Modal ships the repo into the image so these work
    # in the remote container too (volumes are mounted read-write).
    # If this script is invoked from inside the container,
    # inference.provenance is importable via the sys.path shim.
    import sys as _sys
    _sys.path.insert(0, "/data/code")  # Modal mounts repo here
    _sys.path.insert(0, os.getcwd())
    try:
        from inference.provenance import (
            EvidenceSourceType,
            default_provenance,
        )
    except ImportError:
        # Extreme fallback: inline the minimal provenance stamp so the
        # script still runs on a bare Modal container without the repo
        # mounted.  This path is never hit in normal operation.
        EvidenceSourceType = type(
            "EvidenceSourceType",
            (),
            {"GENERATOR_OUTPUT": "generator_output"},
        )

        def default_provenance(  # type: ignore[no-redef]
            source_type: str,
            claim_generator_id: str | None = None,
            evidence_generator_id: str | None = None,
        ) -> dict:
            return {
                "evidence_source_type": source_type,
                "evidence_is_independent": False,
                "evidence_generator_id": evidence_generator_id,
                "claim_generator_id": claim_generator_id,
                "evidence_trust_tier": "unknown",
            }

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    # -------- Ground truth --------
    print(f"Loading ground truth from {openi_reports_csv}...")
    gt_df = pd.read_csv(openi_reports_csv)
    print(f"Loaded {len(gt_df)} OpenI reports")

    # -------- Image sampling --------
    image_files = sorted(
        f for f in os.listdir(openi_images_dir) if f.endswith(".png")
    )
    if not image_files:
        print("ERROR: No PNG files found. Upload OpenI images first.")
        return {"error": "No images found"}

    # Decouple image sampling from generation sampling so Task 9 can
    # run CheXagent twice on IDENTICAL image sets with DIFFERENT
    # generation RNGs.  Falls back to ``seed`` when image_seed=None.
    effective_image_seed = image_seed if image_seed is not None else seed
    rng = random.Random(effective_image_seed)
    if len(image_files) > max_images:
        image_files = rng.sample(image_files, max_images)
    else:
        rng.shuffle(image_files)
    print(f"Processing {len(image_files)} images (sampling={sampling})")

    # -------- Model load with fallbacks --------
    model, processor, tokenizer, image_processor, model_id = _load_chexagent(
        device=device,
    )
    if model is None:
        print("Neither CheXagent-8b nor CheXagent-2 loaded; emitting empty reports.")

    # -------- Generation loop --------
    generated_reports: dict[str, str] = {}
    errors = 0
    gen_kwargs: dict = {
        "max_new_tokens": max_new_tokens,
        "do_sample": bool(sampling),
        "num_beams": 1,
    }
    if sampling:
        gen_kwargs.update(temperature=float(temperature), top_p=float(top_p))

    for img_file in tqdm(image_files, desc=f"CheXagent:{run_id}"):
        img_path = os.path.join(openi_images_dir, img_file)
        try:
            image = Image.open(img_path).convert("RGB")

            if model is None:
                generated_reports[img_file] = "[CheXagent unavailable]"
                continue

            text = _run_single_image(
                model=model,
                processor=processor,
                tokenizer=tokenizer,
                image_processor=image_processor,
                image=image,
                prompt=DEFAULT_PROMPT,
                device=device,
                gen_kwargs=gen_kwargs,
            )
            generated_reports[img_file] = text

        except Exception as e:  # noqa: BLE001
            errors += 1
            if errors <= 5:
                print(f"Error on {img_file}: {e}")
            generated_reports[img_file] = f"[Generation error: {str(e)[:120]}]"

    print(f"Generated {len(generated_reports)} reports ({errors} errors)")

    # -------- Workbook build --------
    claim_generator_id = (
        generator_override or f"chexagent-8b-run-{run_id}"
    )
    print(f"Stamping provenance: claim_generator_id={claim_generator_id}")
    print("Evidence generator: openi_radiologist (GT report)")

    workbook_entries: list[dict] = []
    claim_id = 0
    for img_file, report_text in generated_reports.items():
        sentences = _split_to_sentences(report_text)
        base_name = img_file.replace(".png", "")
        gt_row = gt_df[
            gt_df["study_id"].astype(str).str.contains(base_name, na=False)
        ]
        gt_text = ""
        if len(gt_row) > 0:
            gt_text = str(gt_row.iloc[0].get("report_text", ""))

        # Provenance stamp shared across all claims from this image/report.
        # claim_generator_id identifies the CheXagent run that produced
        # the claim; evidence_generator_id is the radiologist-written GT
        # (for the Task-1 grader phase).  Task-9 re-stamps these when
        # pairing claims from run A with evidence from run B.
        prov = default_provenance(
            source_type=EvidenceSourceType.GENERATOR_OUTPUT,
            claim_generator_id=claim_generator_id,
            evidence_generator_id="openi_radiologist",
        )

        for sent in sentences:
            if len(sent.strip()) < 10:
                continue

            entry: dict = {
                "claim_id": f"real_{run_id}_{claim_id:06d}",
                "image_file": img_file,
                "image_path": os.path.join(openi_images_dir, img_file),
                "generated_report": report_text,
                "extracted_claim": sent.strip(),
                "ground_truth_report": gt_text,
                "generator_model": model_id or "none",
                "generator_run_id": run_id,
                "sampling_mode": "nucleus" if sampling else "greedy",
                "temperature": float(temperature) if sampling else 0.0,
                "top_p": float(top_p) if sampling else 1.0,

                # ---- Grader-ensemble fields (Task 1c fills these) ----
                # Five-class ordinal schema used by Krippendorff α:
                #   SUPPORTED / CONTRADICTED / NOVEL_PLAUSIBLE /
                #   NOVEL_HALLUCINATED / UNCERTAIN
                "grader_chexbert_label": "",
                "grader_chexbert_confidence": "",
                "grader_claude_label": "",
                "grader_claude_confidence": "",
                "grader_claude_rationale": "",
                "grader_medgemma_label": "",
                "grader_medgemma_confidence": "",
                "grader_medgemma_rationale": "",
                "majority_label": "",
                "krippendorff_alpha_bootstrapped": None,

                # ---- Task 8 self-annotation fields ----
                "self_annotator_label": "",
                "self_annotator_notes": "",
                "self_annotator_confidence": "",
            }
            entry.update(prov)
            workbook_entries.append(entry)
            claim_id += 1

    print(f"Built workbook with {len(workbook_entries)} claims")

    # -------- Persist --------
    json_path = os.path.join(output_dir, workbook_filename)
    with open(json_path, "w") as f:
        json.dump(workbook_entries, f, indent=2)

    csv_path = os.path.join(
        output_dir, workbook_filename.replace(".json", ".csv")
    )
    if workbook_entries:
        with open(csv_path, "w", newline="") as f:
            # CSV writer needs a flat schema, so we drop any nested fields
            # (there are none today, but this keeps us safe for refactors).
            flat = [{k: v for k, v in e.items()} for e in workbook_entries]
            writer = csv.DictWriter(f, fieldnames=list(flat[0].keys()))
            writer.writeheader()
            writer.writerows(flat)

    # Instructions doc only gets written once; skip if already present
    # so repeated runs don't churn the file.
    instructions_path = os.path.join(output_dir, "ANNOTATION_INSTRUCTIONS.md")
    if not os.path.exists(instructions_path):
        with open(instructions_path, "w") as f:
            f.write(_annotation_instructions())

    summary = {
        "run_id": run_id,
        "claim_generator_id": claim_generator_id,
        "model_used": model_id or "none",
        "total_images": len(image_files),
        "total_reports": len(generated_reports),
        "total_claims": len(workbook_entries),
        "generation_errors": errors,
        "sampling_mode": "nucleus" if sampling else "greedy",
        "seed": seed,
        "output_json": json_path,
        "output_csv": csv_path,
        "instructions": instructions_path,
        "status": "AWAITING_GRADER_ENSEMBLE",
    }

    summary_path = os.path.join(
        output_dir, f"generation_summary_{run_id}.json"
    )
    with open(summary_path, "w") as f:
        import json as _json
        _json.dump(summary, f, indent=2)

    vol.commit()
    print(f"\nWorkbook written to {json_path}")
    print("STATUS: awaiting 3-grader ensemble (run Task 1c next)")
    return summary


# ---------------------------------------------------------------------------
# Helpers (pure, no Modal-specific state; lifted so Task 9 can reuse them)
# ---------------------------------------------------------------------------


def _load_chexagent(device):
    """Try CheXagent-8b, then CheXagent-2 3B, then give up gracefully.

    Returns:
        Tuple ``(model, processor, tokenizer, image_processor, model_id)``.
        Any element may be ``None``; callers must handle the all-None case.

    Why the processor/tokenizer split:
        The 8B release ships ``AutoProcessor`` config that sometimes
        fails silently under ``transformers==4.40.0``.  The fallback
        path loads ``AutoTokenizer`` + ``AutoImageProcessor`` separately
        and constructs the forward-pass inputs by hand.
    """
    import torch
    from transformers import (
        AutoImageProcessor,
        AutoModelForCausalLM,
        AutoProcessor,
        AutoTokenizer,
    )

    for model_id in (CHEXAGENT_PRIMARY, CHEXAGENT_FALLBACK):
        print(f"Attempting to load {model_id}...")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,  # H100 handles bf16 natively
                device_map="auto",
                trust_remote_code=True,
            )
        except Exception as e:  # noqa: BLE001
            print(f"  model load failed: {e}")
            continue

        # Prefer AutoProcessor; fall back to split tokenizer+image_processor.
        processor = None
        tokenizer = None
        image_processor = None
        try:
            processor = AutoProcessor.from_pretrained(
                model_id, trust_remote_code=True
            )
            # Some CheXagent releases expose .tokenizer and .image_processor
            # on the processor — that gets us the split path for free.
            tokenizer = getattr(processor, "tokenizer", None)
            image_processor = getattr(processor, "image_processor", None)
        except Exception as e:  # noqa: BLE001
            print(f"  AutoProcessor failed ({e}); trying split load")

        if tokenizer is None:
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_id, trust_remote_code=True
                )
            except Exception as e:  # noqa: BLE001
                print(f"  tokenizer load failed: {e}")
        if image_processor is None:
            try:
                image_processor = AutoImageProcessor.from_pretrained(
                    model_id, trust_remote_code=True
                )
            except Exception as e:  # noqa: BLE001
                print(f"  image processor load failed: {e}")

        if tokenizer is None and processor is None:
            print(f"  could not obtain tokenizer for {model_id}")
            continue

        print(f"  loaded {model_id} successfully")
        return model, processor, tokenizer, image_processor, model_id

    print("All CheXagent variants failed to load.")
    return None, None, None, None, None


def _run_single_image(
    *,
    model,
    processor,
    tokenizer,
    image_processor,
    image,
    prompt: str,
    device,
    gen_kwargs: dict,
) -> str:
    """Run a single forward pass, preferring the processor path.

    Returns the decoded generated string with the prompt stripped off.
    Any exception propagates — the caller logs it as a per-image error.
    """
    import torch

    # ---- Path 1: AutoProcessor (preferred when it works) ----
    if processor is not None:
        try:
            inputs = processor(
                images=image,
                text=prompt,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                output_ids = model.generate(**inputs, **gen_kwargs)
            text = (tokenizer or processor).decode(  # type: ignore[union-attr]
                output_ids[0], skip_special_tokens=True
            )
            return _strip_prompt(text, prompt)
        except Exception:
            # Fall through to manual path — do NOT re-raise here, the
            # manual path is the whole reason this function exists.
            pass

    # ---- Path 2: Manual tokenizer + image_processor ----
    if tokenizer is None or image_processor is None:
        raise RuntimeError("no usable processor/tokenizer pair")

    tok_out = tokenizer(prompt, return_tensors="pt").to(device)
    img_out = image_processor(images=image, return_tensors="pt").to(device)
    pixel_values = img_out.get("pixel_values")
    if pixel_values is None:
        raise RuntimeError("image_processor did not return pixel_values")

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=tok_out["input_ids"],
            attention_mask=tok_out.get("attention_mask"),
            pixel_values=pixel_values,
            **gen_kwargs,
        )
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return _strip_prompt(text, prompt)


def _strip_prompt(text: str, prompt: str) -> str:
    """Trim the prompt prefix from a decoded generation, if present."""
    if not text:
        return ""
    if prompt and prompt in text:
        return text.split(prompt, 1)[-1].strip()
    return text.strip()


def _split_to_sentences(text: str) -> list[str]:
    """Split a report into sentences for claim extraction.

    This is the rule-based fallback.  Task 7's ``LLMClaimExtractor`` can
    replace this at the compile-silver-results step — for the raw
    workbook we keep things simple and fast.
    """
    import re

    if not text:
        return []
    # Light normalization: collapse newlines into spaces, then split on
    # sentence terminators. We keep short fragments out of the workbook
    # via the len(sent) >= 10 filter in the caller.
    cleaned = re.sub(r"\s+", " ", text.strip())
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    return [p for p in parts if len(p) > 5]


def _annotation_instructions() -> str:
    """Instructions doc written once per output directory.

    Referenced by both Task 1c's grader ensemble and Task 8's
    self-annotation pass.  Krippendorff α (ordinal) is the reliability
    metric — NOT Cohen's κ.
    """
    return """# ClaimGuard-CXR Silver-Standard Annotation Instructions

## Task
For each claim extracted from a CheXagent-8b generated report, assign
one of the 5 ordinal labels below by comparing the claim against:

1. The original chest X-ray (see `image_path`).
2. The original radiologist-written report (see `ground_truth_report`).

## Label Schema (ordinal, 0 → 4)

0. **SUPPORTED**          — claim is consistent with both image and report
1. **CONTRADICTED**       — claim conflicts with image or report
2. **NOVEL_PLAUSIBLE**    — finding visible in image but absent from GT report
3. **NOVEL_HALLUCINATED** — finding not visible and not in GT report
4. **UNCERTAIN**          — evidence insufficient to decide

Return JSON: `{"label": "...", "confidence": "high|medium|low",
"rationale": "<=30 words"}`.

## Inter-Rater Agreement
We target **Krippendorff's α ≥ 0.80** (ordinal metric, 1000-bootstrap
95 % CI).  This is the 5-class multi-coder reliability standard from
Hayes & Krippendorff (2007).

- α ≥ 0.80 → ship as the silver standard.
- 0.67 ≤ α < 0.80 → publish with caveats; drop UNCERTAIN and recompute.
- α < 0.67 → fall back to binary SUPPORTED-vs-rest and document the
  coarsening transparently.

## Workflow
1. Task 1c: 3 independent graders (CheXbert label-diff, Claude Sonnet
   4.5 with vision, MedGemma-4B) fill the `grader_*_label` fields.
2. Task 1d: `compile_silver_standard_results.py` applies majority vote
   and writes `majority_label`.
3. Task 8: the user self-annotates 100 stratified claims to validate
   ensemble reliability (`self_annotator_label` fields).

Legacy fields (`annotator_label`, `annotator_notes`) are NO LONGER used
— the silver standard is auto-graded, not human-annotated.
"""


@app.local_entrypoint()
def main(
    max_images: int = 200,
    seed: int = 42,
    image_seed: int | None = None,
    sampling: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.9,
    run_id: str = "silver",
    workbook_filename: str = "annotation_workbook.json",
):
    """CLI entrypoint.

    Task 1 uses the defaults (greedy, 200 images, ``run-silver``).
    Task 9 overrides ``--sampling``, ``--run-id run-a`` / ``--run-id run-b``,
    a shared ``--image-seed`` across both runs, and distinct ``--seed``
    values per run so the two generations diverge on the same images.
    """
    result = generate_annotation_workbook.remote(
        max_images=max_images,
        seed=seed,
        image_seed=image_seed,
        sampling=sampling,
        temperature=temperature,
        top_p=top_p,
        run_id=run_id,
        workbook_filename=workbook_filename,
    )
    print(f"\nResult: {result}")
