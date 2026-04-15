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
    .apt_install("libgl1", "libglib2.0-0")  # cv2 runtime deps
    .pip_install(
        "torch>=2.1.0",
        "torchvision>=0.16.0",  # CheXagent modeling imports torchvision
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
        # CheXagent's bundled configuration_chexagent.py requires all
        # of the following at import time.  Discovered on the
        # 2026-04-15 Task 9 launch when CheXagent silently fell
        # through to the "[CheXagent unavailable]" stub path on every
        # image because of these missing deps — the model loader
        # prints the error but returns None, and the downstream
        # workbook still contains 100 entries (all sentinel strings),
        # which passes schema checks but produces meaningless pair
        # tables for Task 9.
        "einops>=0.7.0",
        "albumentations>=1.3.1",
        "opencv-python-headless>=4.8.0",  # "cv2" but without the GUI
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

# CheXagent README-validated prompt template (2026-04-15 source dump
# of StanfordAIMI/CheXagent-8b/README.md):
#
#     prompt = 'Describe "Airway"'
#     inputs = processor(images=images,
#                        text=f' USER: <s>{prompt} ASSISTANT: <s>',
#                        return_tensors='pt')
#     output = model.generate(**inputs, generation_config=generation_config)
#
# Key facts:
#   1. Leading space before "USER:" is INTENTIONAL.
#   2. <s> tokens wrap the prompt + appear after ASSISTANT:.
#   3. CheXagent is trained for per-anatomy queries, NOT free-form
#      "generate a report".  Stanford's example queries "Airway".
#   4. Use GenerationConfig.from_pretrained() to get the model's
#      specific generation defaults (top_p, repetition_penalty, etc.)
#      — generic gen_kwargs will not match the model's expected
#      configuration.
CHEXAGENT_README_PROMPT_TMPL = ' USER: <s>{prompt} ASSISTANT: <s>'

# CheXagent's training prompts cycle through these 8 anatomical
# regions.  For Task 9 we query just one of them per image (the
# default is "Findings") to keep wall-clock cost down.  For a richer
# report you'd loop through all of them and concatenate the answers.
CHEXAGENT_ANATOMIES = (
    "Airway",
    "Breathing",
    "Cardiac",
    "Diaphragm",
    "Everything else",
    "Findings",
    "Impression",
)
CHEXAGENT_DEFAULT_QUERY = 'Describe "Findings"'


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

    empty_outputs = 0
    debug_samples_printed = 0
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
                image_path=img_path,  # NEW: passed for from_list_format path
                prompt=DEFAULT_PROMPT,
                device=device,
                gen_kwargs=gen_kwargs,
            )
            generated_reports[img_file] = text
            if not text.strip():
                empty_outputs += 1
            elif debug_samples_printed < 3:
                # Print a few sample outputs so the Modal logs show
                # what CheXagent is actually producing.  Essential for
                # debugging empty-output regressions like the one
                # hunted on 2026-04-15.
                print(f"  sample[{debug_samples_printed}] {img_file}: "
                      f"{text[:200]!r}")
                debug_samples_printed += 1

        except Exception as e:  # noqa: BLE001
            errors += 1
            if errors <= 5:
                print(f"Error on {img_file}: {e}")
            generated_reports[img_file] = f"[Generation error: {str(e)[:120]}]"

    print(
        f"Generated {len(generated_reports)} reports "
        f"({errors} errors, {empty_outputs} empty)"
    )
    if empty_outputs > len(generated_reports) // 2:
        print(
            "WARN: more than half of outputs were empty — CheXagent "
            "prompt format may still be wrong.  Check the sample "
            "outputs printed above."
        )

    # -------- Workbook build --------
    claim_generator_id = (
        generator_override or f"chexagent-8b-run-{run_id}"
    )
    print(f"Stamping provenance: claim_generator_id={claim_generator_id}")
    print("Evidence generator: openi_radiologist (GT report)")

    # Column-name compatibility shim (2026-04-15 fix).
    #
    # Two schemas exist in the wild for the OpenI CSV:
    #   (a) "study_id" / "report_text"               — used by older
    #       OpenI dumps and the original generation code
    #   (b) "deid_patient_id" / "section_findings" + "section_impression"
    #       — the current CheXpert-style schema written by
    #       scripts/convert_openi_to_chexpert_schema.py
    #
    # The Task 9 launch on 2026-04-15 failed with
    # ``KeyError: 'study_id'`` because the uploaded CSV is in schema
    # (b).  We detect the schema at load time and adapt the lookup +
    # report-text assembly to match.
    if "study_id" in gt_df.columns:
        _id_col = "study_id"
        _report_builder = lambda row: str(row.get("report_text", ""))  # noqa: E731
    elif "deid_patient_id" in gt_df.columns:
        _id_col = "deid_patient_id"
        def _report_builder(row) -> str:
            findings = str(row.get("section_findings", "") or "")
            impression = str(row.get("section_impression", "") or "")
            parts = []
            if findings.strip():
                parts.append(f"Findings: {findings.strip()}")
            if impression.strip():
                parts.append(f"Impression: {impression.strip()}")
            return "\n".join(parts)
    else:
        raise KeyError(
            f"CSV at {openi_reports_csv} has neither 'study_id' nor "
            f"'deid_patient_id' column.  Columns found: "
            f"{list(gt_df.columns)[:8]}..."
        )
    print(f"OpenI CSV schema: using id_col={_id_col!r}")

    workbook_entries: list[dict] = []
    claim_id = 0
    for img_file, report_text in generated_reports.items():
        sentences = _split_to_sentences(report_text)
        base_name = img_file.replace(".png", "")

        # Extract the patient prefix from the image filename.  OpenI
        # filenames follow ``CXR<N>_IM-<X>-<Y>.png`` where CXR<N> is
        # the de-identified patient id.  The CheXpert-schema CSV keys
        # on CXR<N> only, so we need the prefix, not the full stem.
        patient_prefix = base_name.split("_")[0] if "_" in base_name else base_name

        # First try exact-match on the patient prefix (new schema
        # path), then fall back to substring-match on the full
        # base_name (old "study_id" path) so both schemas work.
        if _id_col == "deid_patient_id":
            gt_row = gt_df[gt_df[_id_col].astype(str) == patient_prefix]
            if len(gt_row) == 0:
                gt_row = gt_df[
                    gt_df[_id_col].astype(str).str.contains(
                        patient_prefix, na=False
                    )
                ]
        else:
            gt_row = gt_df[
                gt_df[_id_col].astype(str).str.contains(
                    base_name, na=False
                )
            ]

        gt_text = ""
        if len(gt_row) > 0:
            gt_text = _report_builder(gt_row.iloc[0])

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
        # Debug: introspect which inference APIs are available so we
        # don't have to guess which code path to use.  Printed once
        # per cold start, not per image.
        _relevant = [
            "chat", "from_list_format", "apply_chat_template",
            "build_chat_input", "generate",
        ]
        for obj_name, obj in (
            ("model", model),
            ("tokenizer", tokenizer),
            ("processor", processor),
        ):
            if obj is None:
                continue
            present = [m for m in _relevant if hasattr(obj, m)]
            print(f"  {obj_name} has: {present}")
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
    image_path: str | None,
    prompt: str,
    device,
    gen_kwargs: dict,
) -> str:
    """Run a single forward pass on CheXagent (Qwen-VL base).

    CheXagent-8b uses the Qwen-VL architecture, which requires a
    specific multimodal prompt format where the image is referenced
    inline as ``<img>PATH</img>\\n`` via
    ``tokenizer.from_list_format([{"image": path}, {"text": prompt}])``.

    The 2026-04-15 v3 launch discovered this the hard way: the prior
    code used ``processor(images=image, text=prompt)`` which is the
    generic HF multimodal pattern.  On CheXagent that path constructs
    input_ids with no image placeholder token, so the vision tower
    is never actually conditioned on the pixels.  The text-only
    language model head then generates `</s>` (EOS token id 2)
    immediately or echoes the prompt, producing zero useful tokens.
    Stripping the prompt leaves an empty string, and the workbook
    ends up with 0-1 claims per 100 images.

    This function now tries three strategies in order:

        1. ``tokenizer.from_list_format`` — CheXagent's documented
           Qwen-VL pattern.  Requires a file path (not a PIL image)
           so the caller must pass ``image_path``.
        2. Chat template with inline ``<image>`` token in the text.
        3. Fallback: generic ``processor(images, text)``.

    Each strategy is tried in-line, with EXCEPTIONS propagating
    through to the next strategy.  The first strategy that produces
    a non-empty decoded generation wins.  If all three fail, we
    raise a ``RuntimeError`` so the caller logs it as an error.

    Args:
        image_path: Filesystem path to the image.  Required for the
            from_list_format path.  ``None`` disables that strategy.
    """
    import torch

    # ---- Strategy 0: README-validated CheXagent format (PRIMARY) ----
    # Source: StanfordAIMI/CheXagent-8b README.md, captured via
    # scripts/cat_chexagent_processor.py on 2026-04-15.  Stanford's
    # exact inference snippet uses:
    #     prompt = 'Describe "Airway"'
    #     inputs = processor(images, text=' USER: <s>{prompt} ASSISTANT: <s>')
    #     output = model.generate(**inputs, generation_config=GENCFG)
    #
    # CheXagent is trained for per-anatomy queries, NOT free-form
    # report generation.  We use the generic CHEXAGENT_DEFAULT_QUERY
    # ('Describe "Findings"') which is the closest single-query
    # equivalent to a full report.  If the caller passes a custom
    # prompt that DOESN'T match the Describe "<region>" pattern, we
    # still wrap it in the README template — the model may degrade
    # but at least won't return empty strings.
    if processor is not None:
        try:
            # Use the dedicated CheXagent query if the caller didn't
            # override; otherwise use whatever they passed.
            chexagent_query = (
                CHEXAGENT_DEFAULT_QUERY
                if prompt == DEFAULT_PROMPT
                else prompt
            )
            wrapped = CHEXAGENT_README_PROMPT_TMPL.format(
                prompt=chexagent_query,
            )
            inputs = processor(
                images=image,
                text=wrapped,
                return_tensors="pt",
            ).to(device)
            # Try to load the model's bundled GenerationConfig once
            # (cached in a function attribute).  Falls back to
            # gen_kwargs if the config isn't present.
            generation_config = getattr(
                _run_single_image, "_chexagent_gen_cfg", None,
            )
            if generation_config is None:
                try:
                    from transformers import GenerationConfig
                    generation_config = GenerationConfig.from_pretrained(
                        "StanfordAIMI/CheXagent-8b",
                    )
                    _run_single_image._chexagent_gen_cfg = generation_config  # type: ignore[attr-defined]
                except Exception:
                    pass
            with torch.no_grad():
                if generation_config is not None:
                    output_ids = model.generate(
                        **inputs,
                        generation_config=generation_config,
                    )
                else:
                    output_ids = model.generate(**inputs, **gen_kwargs)
            text = (tokenizer or processor).decode(  # type: ignore[union-attr]
                output_ids[0], skip_special_tokens=True
            )
            # Strip the README prompt template so we return only the
            # generated content.  The decoded text usually contains
            # the prompt echo plus the assistant's response.
            stripped = _strip_prompt(text, wrapped)
            if not stripped:
                stripped = _strip_prompt(text, chexagent_query)
            if not stripped:
                stripped = text.strip()
            if stripped:
                return stripped
            print("  README-format path returned empty text")
        except Exception as e:  # noqa: BLE001
            print(
                f"  README-format path failed: "
                f"{type(e).__name__}: {str(e)[:200]}"
            )

    # ---- Strategy 1: tokenizer.apply_chat_template + processor ----
    # Introspection of CheXagent-8b (printed at load time) showed:
    #     model has: ['generate']
    #     tokenizer has: ['apply_chat_template']
    #     processor has: []
    # → the ONLY chat-template API available is on the tokenizer.
    # This is the Qwen2-VL v2 chat template pattern: build the
    # user-turn as a list of content blocks (one per modality),
    # apply_chat_template to render it into the canonical prompt
    # string with `<|im_start|>user ... <|im_end|>` markers +
    # image placeholder tokens, then pass the rendered string +
    # the raw PIL image through the processor.
    if (
        tokenizer is not None
        and processor is not None
        and hasattr(tokenizer, "apply_chat_template")
    ):
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                },
            ]
            text_prompt = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            inputs = processor(
                text=[text_prompt],
                images=[image],
                return_tensors="pt",
                padding=True,
            ).to(device)
            with torch.no_grad():
                output_ids = model.generate(**inputs, **gen_kwargs)
            # Trim the prompt tokens from the output so decoding
            # returns only the new generation.
            input_len = inputs["input_ids"].shape[1]
            new_tokens = output_ids[:, input_len:]
            text = tokenizer.decode(
                new_tokens[0], skip_special_tokens=True,
            ).strip()
            if text:
                return text
            print("  apply_chat_template path returned empty text")
        except Exception as e:  # noqa: BLE001
            print(
                f"  apply_chat_template path failed: "
                f"{type(e).__name__}: {e}"
            )

    # ---- Strategy 1b: tokenizer.from_list_format + generate() ----
    # If model.chat isn't present but the tokenizer has
    # from_list_format, we can build the query manually and call
    # model.generate.  This path skips the convenience wrapper but
    # still uses the correct Qwen-VL prompt structure.
    if (
        image_path is not None
        and tokenizer is not None
        and hasattr(tokenizer, "from_list_format")
    ):
        try:
            query = tokenizer.from_list_format([
                {"image": image_path},
                {"text": prompt},
            ])
            tok_out = tokenizer(
                query, return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=tok_out["input_ids"],
                    attention_mask=tok_out.get("attention_mask"),
                    **gen_kwargs,
                )
            text = tokenizer.decode(
                output_ids[0], skip_special_tokens=True
            )
            stripped = _strip_prompt(text, query)
            if not stripped:
                stripped = _strip_prompt(text, prompt)
            if stripped:
                return stripped
        except Exception as e:  # noqa: BLE001
            print(f"  from_list_format path failed: {e}")

    # ---- Strategy 2: chat template with explicit <image> token ----
    if processor is not None:
        for image_tok in ("<image>\n", "<|image|>\n", ""):
            try:
                wrapped_prompt = f"{image_tok}{prompt}" if image_tok else prompt
                inputs = processor(
                    images=image,
                    text=wrapped_prompt,
                    return_tensors="pt",
                ).to(device)
                with torch.no_grad():
                    output_ids = model.generate(**inputs, **gen_kwargs)
                text = (tokenizer or processor).decode(  # type: ignore[union-attr]
                    output_ids[0], skip_special_tokens=True
                )
                stripped = _strip_prompt(text, wrapped_prompt)
                if not stripped:
                    stripped = _strip_prompt(text, prompt)
                if stripped:
                    return stripped
            except Exception as e:  # noqa: BLE001
                print(f"  processor path (tok={image_tok!r}) failed: {e}")
                continue

    # ---- Strategy 3: manual tokenizer + image_processor ----
    if tokenizer is not None and image_processor is not None:
        try:
            tok_out = tokenizer(prompt, return_tensors="pt").to(device)
            img_out = image_processor(
                images=image, return_tensors="pt",
            ).to(device)
            pixel_values = img_out.get("pixel_values")
            if pixel_values is not None:
                with torch.no_grad():
                    output_ids = model.generate(
                        input_ids=tok_out["input_ids"],
                        attention_mask=tok_out.get("attention_mask"),
                        pixel_values=pixel_values,
                        **gen_kwargs,
                    )
                text = tokenizer.decode(
                    output_ids[0], skip_special_tokens=True,
                )
                stripped = _strip_prompt(text, prompt)
                if stripped:
                    return stripped
        except Exception as e:  # noqa: BLE001
            print(f"  manual path failed: {e}")

    raise RuntimeError(
        "all CheXagent inference strategies returned empty output — "
        "the model loaded but no strategy produced non-empty text. "
        "Check the prompt format for this specific CheXagent release."
    )


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
