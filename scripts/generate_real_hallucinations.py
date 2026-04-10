"""Generate real hallucination test set by running CheXagent-8b on OpenI images.

This produces the RAW generated claims for subsequent MANUAL human annotation.
The output is NOT auto-labeled — it produces an annotation workbook (JSON + CSV)
for a human annotator to label each claim as Supported/Contradicted/Novel.

Why NOT auto-label with CheXbert:
  CheXbert has ~10-15% label noise. Evaluating a SOTA verifier against a legacy
  labeler is circular. Manual annotation is required for a credible NeurIPS eval.

Pipeline:
  Phase 1 (this script, Modal GPU):
    1. Load 200 OpenI test images
    2. Run CheXagent-8b to generate reports
    3. Extract claims from generated reports (LLM-based or rule-based fallback)
    4. Pair each claim with its ground-truth report + image path
    5. Output annotation workbook for human review

  Phase 2 (manual, ~8-12 hours annotator time):
    - Annotator labels each claim against image + report
    - Label schema: Supported / Contradicted / Novel-Plausible / Novel-Hallucinated / Uncertain
    - 25% double-annotated for inter-rater kappa

  Phase 3 (scripts/compile_real_hallucination_eval.py):
    - Ingest annotated labels, compute verifier accuracy on real hallucinations

Usage:
    modal run --detach scripts/generate_real_hallucinations.py
"""

from __future__ import annotations

import modal

app = modal.App("claimguard-real-hallucinations")

gen_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1.0",
        "transformers==4.40.0",  # CheXagent requires exactly 4.40.0
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "Pillow>=10.0.0",
        "tqdm>=4.66.0",
        "accelerate>=0.27.0",
    )
)

vol = modal.Volume.from_name("claimguard-data", create_if_missing=True)


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
) -> dict:
    """Run CheXagent on OpenI images and produce annotation workbook.

    Does NOT auto-label. Produces a JSON/CSV workbook for human annotation.
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

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    # Load OpenI ground truth reports
    print(f"Loading ground truth from {openi_reports_csv}...")
    gt_df = pd.read_csv(openi_reports_csv)
    print(f"Loaded {len(gt_df)} OpenI reports")

    # Get available images
    image_files = sorted([f for f in os.listdir(openi_images_dir) if f.endswith(".png")])
    if len(image_files) == 0:
        print("ERROR: No PNG files found. Upload OpenI images to Modal volume first.")
        return {"error": "No images found"}

    # Sample subset
    rng = random.Random(seed)
    if len(image_files) > max_images:
        image_files = rng.sample(image_files, max_images)
    else:
        rng.shuffle(image_files)
    print(f"Processing {len(image_files)} images")

    # Load CheXagent-8b
    print("Loading CheXagent-8b...")
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

    model = None
    processor = None
    tokenizer = None
    try:
        model = AutoModelForCausalLM.from_pretrained(
            "StanfordAIMI/CheXagent-8b",
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(
            "StanfordAIMI/CheXagent-8b",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "StanfordAIMI/CheXagent-8b",
            trust_remote_code=True,
        )
        print("CheXagent-8b loaded successfully")
    except Exception as e:
        print(f"CheXagent loading failed: {e}")
        print("Will generate empty reports — annotator will need to fill in generated text")

    # Generate reports
    generated_reports = {}
    errors = 0

    for img_file in tqdm(image_files, desc="Generating reports"):
        img_path = os.path.join(openi_images_dir, img_file)
        try:
            image = Image.open(img_path).convert("RGB")

            if model is not None:
                prompt = "Generate a detailed radiology report for this chest X-ray."
                inputs = processor(
                    images=image,
                    text=prompt,
                    return_tensors="pt",
                ).to(device)

                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=256,
                        do_sample=False,
                        num_beams=1,
                    )
                generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                if prompt in generated_text:
                    generated_text = generated_text.split(prompt)[-1].strip()
            else:
                generated_text = "[CheXagent failed to load — manual entry required]"

            generated_reports[img_file] = generated_text

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"Error on {img_file}: {e}")
            generated_reports[img_file] = f"[Generation error: {str(e)[:100]}]"

    print(f"Generated {len(generated_reports)} reports ({errors} errors)")

    # Extract claims and build annotation workbook
    print("Building annotation workbook...")
    workbook_entries = []
    claim_id = 0

    for img_file, report_text in generated_reports.items():
        # Extract claims (rule-based fallback — LLM extractor runs separately)
        sentences = _split_to_sentences(report_text)

        # Find matching ground truth
        base_name = img_file.replace(".png", "")
        gt_row = gt_df[gt_df["study_id"].astype(str).str.contains(base_name, na=False)]

        gt_text = ""
        if len(gt_row) > 0:
            gt_text = str(gt_row.iloc[0].get("report_text", ""))

        for sent in sentences:
            if len(sent.strip()) < 10:
                continue

            entry = {
                "claim_id": f"real_{claim_id:06d}",
                "image_file": img_file,
                "image_path": os.path.join(openi_images_dir, img_file),
                "generated_report": report_text,
                "extracted_claim": sent.strip(),
                "ground_truth_report": gt_text,
                "generator": "CheXagent-8b",

                # ANNOTATION FIELDS — to be filled by human annotator
                "annotator_label": "",  # Supported / Contradicted / Novel-Plausible / Novel-Hallucinated / Uncertain
                "annotator_notes": "",
                "annotator_confidence": "",  # High / Medium / Low
                "annotator_id": "",

                # For double annotation (25% subset)
                "needs_double_annotation": claim_id % 4 == 0,  # every 4th claim
                "annotator2_label": "",
                "annotator2_notes": "",
            }
            workbook_entries.append(entry)
            claim_id += 1

    print(f"Built workbook with {len(workbook_entries)} claims for annotation")
    print(f"  - {sum(1 for e in workbook_entries if e['needs_double_annotation'])} marked for double annotation")

    # Save JSON workbook
    json_path = os.path.join(output_dir, "annotation_workbook.json")
    with open(json_path, "w") as f:
        json.dump(workbook_entries, f, indent=2)

    # Save CSV workbook (easier for annotators to work with in spreadsheet)
    csv_path = os.path.join(output_dir, "annotation_workbook.csv")
    if workbook_entries:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=workbook_entries[0].keys())
            writer.writeheader()
            writer.writerows(workbook_entries)

    # Save annotation instructions
    instructions_path = os.path.join(output_dir, "ANNOTATION_INSTRUCTIONS.md")
    with open(instructions_path, "w") as f:
        f.write(_annotation_instructions())

    summary = {
        "total_images": len(image_files),
        "total_reports": len(generated_reports),
        "total_claims": len(workbook_entries),
        "double_annotation_claims": sum(1 for e in workbook_entries if e["needs_double_annotation"]),
        "generation_errors": errors,
        "output_json": json_path,
        "output_csv": csv_path,
        "instructions": instructions_path,
        "status": "AWAITING_MANUAL_ANNOTATION",
    }

    summary_path = os.path.join(output_dir, "generation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    vol.commit()
    print(f"\nAnnotation workbook saved to {output_dir}")
    print(f"STATUS: AWAITING MANUAL ANNOTATION by human annotator")
    print(f"Estimated annotation time: {len(workbook_entries) * 0.5:.0f} - {len(workbook_entries) * 0.75:.0f} minutes")
    return summary


def _split_to_sentences(text: str) -> list[str]:
    """Split report text into sentences (rule-based fallback)."""
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if len(s) > 5]


def _annotation_instructions() -> str:
    """Generate annotation instruction document."""
    return """# ClaimGuard-CXR Real Hallucination Annotation Instructions

## Task
For each claim extracted from a CheXagent-8b generated report, determine
whether the claim is accurate by comparing it against:
1. The original chest X-ray image (provided as image_path)
2. The original radiologist-written report (provided as ground_truth_report)

## Label Schema

- **Supported**: The claim accurately describes what is shown in the image
  AND is consistent with the ground truth report.
- **Contradicted**: The claim directly contradicts what is visible in the
  image OR contradicts a specific finding in the ground truth report.
  Examples: wrong laterality, wrong severity, fabricated finding, negation error.
- **Novel-Plausible**: The claim describes something that IS visible in the
  image but was NOT mentioned in the ground truth report. This is a valid
  observation the original radiologist omitted.
- **Novel-Hallucinated**: The claim describes something that is NOT visible
  in the image and NOT in the ground truth report. This is a fabrication.
- **Uncertain**: The claim is ambiguous, the image quality makes it
  impossible to judge, or you are not confident in your assessment.

## Guidelines

1. View the image FIRST, form your own assessment, THEN read the GT report.
2. If the claim is about a finding that is genuinely subtle or borderline,
   prefer "Uncertain" over forced labeling.
3. For each label, note your confidence: High / Medium / Low.
4. Claims marked needs_double_annotation=True will be reviewed by a second
   annotator. Please still label these with your best judgment.
5. If the generated_report or extracted_claim contains [error], skip.

## Inter-Annotator Agreement
25% of claims are double-annotated. We target Cohen's kappa >= 0.75.
Disagreements will be resolved by a third reviewer.

## Estimated Time
~30-45 seconds per claim. Total: 8-12 hours for ~1000 claims.
"""


@app.local_entrypoint()
def main():
    result = generate_annotation_workbook.remote()
    print(f"\nResult: {result}")
