"""Task 1c — 3-grader ensemble for the silver-standard annotation pipeline.

This Modal H100 job consumes the annotation workbook produced by
``scripts/generate_real_hallucinations.py`` (Task 1b) and fills in the
per-grader label fields used by Task 1d's compiler.  Three graders run
per (image, claim) tuple, each producing a 5-class ordinal label from
the schema

    0 SUPPORTED          — claim consistent with image AND GT report
    1 CONTRADICTED       — claim conflicts with image or GT report
    2 NOVEL_PLAUSIBLE    — claim visible in image but absent from GT report
    3 NOVEL_HALLUCINATED — claim not visible AND not in GT report
    4 UNCERTAIN          — evidence insufficient to decide

Graders
-------
1. **CheXbert labeler diff** (text-only, fast)
   - Runs ``stanfordmlgroup/CheXbert`` on the GT report and on the claim
     text.  Compares the 14-dim pathology label vectors and maps the
     diff to a 5-class label.
   - Cannot see the image, so NOVEL findings are defaulted to
     NOVEL_HALLUCINATED (CheXbert cannot distinguish plausible-from-image
     novel findings from truly fabricated ones).
   - Graceful fallback: if CheXbert won't load, this grader abstains
     with an empty label and we rely on the other two.

2. **Claude Sonnet 4.5 (vision)**
   - Direct Anthropic API call with a base64-encoded JPEG of the OpenI
     image + the GT report text + the claim.  Strong image grounding.
   - Uses the shared grader prompt (see ``GRADER_PROMPT`` below).

3. **MedGemma-4B-IT** (primary) / **LLaVA-Med** (fallback)
   - A second VLM independent of Claude.  Gives us a 3rd vote so
     Krippendorff α is not degenerate on a 2-coder matrix.
   - Graceful fallback through a small model-id chain.

Output
------
The script updates each workbook row in place with:

    grader_chexbert_label / _confidence
    grader_claude_label / _confidence / _rationale
    grader_medgemma_label / _confidence / _rationale
    raw_grader_responses  (for audit)

and writes the updated JSON back next to the input (or to a new path
via ``--output-filename``).

Per-claim inference costs
-------------------------
* CheXbert: ~0.05 s on H100 (text-only). 200 images × 5 claims ≈ 50 s.
* Claude Sonnet 4.5 with vision: ~2 s / claim via the API. 1000 claims
  ≈ 35 min wall clock @ $4 total (Anthropic pricing as of 2026-04).
* MedGemma-4B: ~0.8 s / claim on H100 bf16. 1000 claims ≈ 15 min.

Modal orchestration
-------------------
A single ``@app.function`` runs all three graders in one container so
we pay the GPU cold-start once.  ``transformers==4.40.0`` is pinned to
match the CheXagent container in ``generate_real_hallucinations.py`` —
we want the two jobs interchangeable.

Usage
-----
    modal run --detach scripts/generate_silver_standard_graders.py \\
        --workbook-path /data/eval_data_real_hallucinations/annotation_workbook.json

    # Subset mode for pilot-α run (50 claims)
    modal run scripts/generate_silver_standard_graders.py \\
        --workbook-path ... --max-claims 50 --output-filename workbook_pilot.json
"""

from __future__ import annotations

import modal

app = modal.App("claimguard-silver-graders")

grader_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1.0",
        # Pin to match the CheXagent container so the same Modal image
        # layer can host both jobs.
        "transformers==4.40.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "Pillow>=10.0.0",
        "tqdm>=4.66.0",
        "accelerate>=0.27.0",
        "sentencepiece>=0.1.99",
        "protobuf>=3.20.0",
        "anthropic>=0.25.0",
        "requests>=2.31.0",
    )
    # Anthropic API key is read from a Modal secret named "anthropic"
)

vol = modal.Volume.from_name("claimguard-data", create_if_missing=True)


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

# The 14 pathology classes that CheXbert emits, in CheXbert's own order.
# This must match the order of the output logits so we can map
# per-pathology matches.
CHEXBERT_PATHOLOGIES: tuple[str, ...] = (
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
)

# Keyword → pathology mapping for mapping a free-text claim to one of
# the 14 CheXbert classes.  This is a deliberately conservative list:
# if a claim matches no keyword we mark it UNCERTAIN, which is the
# safest default (Krippendorff ordinal treats UNCERTAIN as the top
# label and penalises far-from-UNCERTAIN disagreements more).
PATHOLOGY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "Cardiomegaly": ("cardiomegaly", "enlarged heart", "cardiac silhouette"),
    "Enlarged Cardiomediastinum": ("mediastin", "cardiomediastinal"),
    "Lung Opacity": ("opacity", "opacification"),
    "Lung Lesion": ("lesion", "nodule", "mass"),
    "Edema": ("edema", "pulmonary edema", "interstitial edema"),
    "Consolidation": ("consolidation", "consolidated"),
    "Pneumonia": ("pneumonia", "infect"),
    "Atelectasis": ("atelectas",),
    "Pneumothorax": ("pneumothorax", "ptx"),
    "Pleural Effusion": ("effusion", "pleural fluid"),
    "Pleural Other": ("pleural thickening", "pleural plaque"),
    "Fracture": ("fracture", "broken rib"),
    "Support Devices": (
        "line",
        "catheter",
        "tube",
        "pacemaker",
        "pacer",
        "icd",
    ),
    "No Finding": (
        "no acute",
        "unremarkable",
        "normal chest",
        "clear lungs",
    ),
}

GRADER_PROMPT = """You are a radiology claim auditor. Given a chest X-ray, the original radiologist report, and a single claim extracted from a model-generated report, pick exactly one label:
SUPPORTED          — claim is consistent with both image and report
CONTRADICTED       — claim conflicts with image or report
NOVEL_PLAUSIBLE    — finding visible in image but absent from the original report
NOVEL_HALLUCINATED — finding not visible and not in the original report
UNCERTAIN          — evidence insufficient to decide
Return JSON: {"label": "...", "confidence": "high|medium|low", "rationale": "<=30 words"}."""

VALID_LABELS: frozenset[str] = frozenset(
    {
        "SUPPORTED",
        "CONTRADICTED",
        "NOVEL_PLAUSIBLE",
        "NOVEL_HALLUCINATED",
        "UNCERTAIN",
    }
)

# Ordinal encoding used by the Krippendorff α computation downstream.
# The order matches the 5-class schema in the instructions doc.
LABEL_TO_ORDINAL: dict[str, int] = {
    "SUPPORTED": 0,
    "CONTRADICTED": 1,
    "NOVEL_PLAUSIBLE": 2,
    "NOVEL_HALLUCINATED": 3,
    "UNCERTAIN": 4,
}


# ---------------------------------------------------------------------------
# Grader 1 — CheXbert labeler diff (text-only, fastest)
# ---------------------------------------------------------------------------


def claim_to_pathology(claim: str) -> str | None:
    """Map a free-text claim to a CheXbert pathology via keyword hit.

    Returns ``None`` if no keyword matches, which the caller turns into
    an UNCERTAIN label (conservative default).
    """
    text = claim.lower()
    for pathology, keywords in PATHOLOGY_KEYWORDS.items():
        if any(k in text for k in keywords):
            return pathology
    return None


def label_from_chexbert_diff(
    claim: str,
    claim_vec: "list[int] | None",
    gt_vec: "list[int] | None",
) -> tuple[str, str]:
    """Translate a CheXbert 14-dim diff into the 5-class label schema.

    Args:
        claim: Free-text claim.  Used only for keyword → pathology mapping.
        claim_vec: 14-dim CheXbert output for the claim text.  Each
            entry is 0 (blank / not mentioned), 1 (positive), 2
            (negative), or 3 (uncertain).  ``None`` means CheXbert
            abstained.
        gt_vec: Same for the GT report.

    Returns:
        Tuple ``(label, confidence)`` where ``label`` is one of
        ``VALID_LABELS``.
    """
    if claim_vec is None or gt_vec is None:
        return "UNCERTAIN", "low"

    pathology = claim_to_pathology(claim)
    if pathology is None:
        return "UNCERTAIN", "low"
    try:
        idx = CHEXBERT_PATHOLOGIES.index(pathology)
    except ValueError:
        return "UNCERTAIN", "low"

    c = claim_vec[idx]
    g = gt_vec[idx]

    # CheXbert label codes: 0 = blank, 1 = positive, 2 = negative,
    # 3 = uncertain.  Map to our schema.
    POS, NEG, BLANK, UNC = 1, 2, 0, 3

    if c == UNC or g == UNC:
        return "UNCERTAIN", "medium"
    if c == POS and g == POS:
        return "SUPPORTED", "high"
    if c == NEG and g == NEG:
        return "SUPPORTED", "high"
    if c == POS and g == NEG:
        return "CONTRADICTED", "high"
    if c == NEG and g == POS:
        return "CONTRADICTED", "high"
    if c == POS and g == BLANK:
        # Claim asserts a positive finding the GT report did not
        # mention.  CheXbert cannot see the image, so we cannot
        # distinguish NOVEL_PLAUSIBLE (radiologist missed it) from
        # NOVEL_HALLUCINATED (model fabricated it).  Default to
        # HALLUCINATED, which is the more conservative call for a
        # false-alarm-oriented safety system.
        return "NOVEL_HALLUCINATED", "low"
    if c == NEG and g == BLANK:
        # Claim asserts absence of a finding not mentioned either
        # way — cannot say it is contradicted, nor novel; safest =
        # SUPPORTED with low confidence.
        return "SUPPORTED", "low"
    if c == BLANK and g != BLANK:
        return "UNCERTAIN", "low"
    return "UNCERTAIN", "low"


def run_chexbert_grader(
    workbook: list[dict],
    openi_reports_csv: str,
) -> None:
    """Populate ``grader_chexbert_*`` fields for every workbook row.

    This function mutates ``workbook`` in place.  It tries to load the
    ``stanfordmlgroup/CheXbert`` checkpoint via HuggingFace; on failure
    it stamps every row with UNCERTAIN / low confidence and continues.
    """
    try:
        import torch  # noqa: F401
        from transformers import AutoModel, AutoTokenizer  # noqa: F401
    except Exception as e:  # noqa: BLE001
        print(f"chexbert grader: torch/transformers import failed ({e})")
        _stamp_abstain_chexbert(workbook)
        return

    # The canonical CheXbert weights are not packaged on HF Hub by
    # Stanford.  Users typically re-upload them at
    # "StanfordAIMI/chexbert" or similar.  We probe a couple of common
    # IDs and fall back to the abstention path if none work.
    candidates = (
        "StanfordAIMI/chexbert",
        "StanfordAIMI/CheXbert",
        "stanfordmlgroup/CheXbert",
    )
    model, tokenizer = None, None
    for model_id in candidates:
        try:
            import torch

            from transformers import AutoModel, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                model_id, trust_remote_code=True
            )
            model = AutoModel.from_pretrained(
                model_id, trust_remote_code=True
            )
            model.eval()
            if torch.cuda.is_available():
                model = model.to("cuda")
            print(f"chexbert grader: loaded {model_id}")
            break
        except Exception as e:  # noqa: BLE001
            print(f"chexbert grader: {model_id} failed ({e})")
            continue

    if model is None or tokenizer is None:
        print("chexbert grader: no checkpoint loaded; abstaining")
        _stamp_abstain_chexbert(workbook)
        return

    # Per-image CheXbert label vectors are cached by study so we don't
    # re-label the same GT report multiple times.
    import pandas as pd

    gt_df = pd.read_csv(openi_reports_csv)
    gt_label_cache: dict[str, list[int]] = {}

    def _cached_labels(report_text: str) -> list[int] | None:
        key = report_text.strip()
        if not key:
            return None
        if key in gt_label_cache:
            return gt_label_cache[key]
        vec = _chexbert_label_vector(model, tokenizer, key)
        gt_label_cache[key] = vec
        return vec

    from tqdm import tqdm

    for row in tqdm(workbook, desc="CheXbert"):
        claim_text = row.get("extracted_claim", "")
        gt_text = row.get("ground_truth_report", "")
        claim_vec = _chexbert_label_vector(model, tokenizer, claim_text)
        gt_vec = _cached_labels(gt_text)
        label, conf = label_from_chexbert_diff(claim_text, claim_vec, gt_vec)
        row["grader_chexbert_label"] = label
        row["grader_chexbert_confidence"] = conf


def _chexbert_label_vector(model, tokenizer, text: str) -> list[int] | None:
    """Compute a 14-dim CheXbert label vector for ``text``.

    This is a best-effort wrapper: because CheXbert's downstream head
    is sometimes missing from HF uploads, we run the encoder and
    apply a heuristic fallback when the full classifier head isn't
    available.  The heuristic is deliberately conservative — it
    returns all-BLANK (0) for empty text and all-UNCERTAIN (3) for
    any failure, both of which map to UNCERTAIN in
    ``label_from_chexbert_diff``.
    """
    if not text.strip():
        return [0] * 14
    try:
        import torch

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        # CheXbert's full implementation has a 14-head MLP on top of
        # BERT.  HF uploads often omit this head — we fall back to a
        # conservative all-UNCERTAIN vector in that case so the diff
        # step does not emit spurious labels.
        if hasattr(outputs, "logits"):
            logits = outputs.logits
            preds = logits.argmax(dim=-1).flatten().tolist()
            if len(preds) == 14:
                return [int(p) for p in preds]
        return [3] * 14  # all-UNCERTAIN ⇒ all rows go to UNCERTAIN
    except Exception:  # noqa: BLE001
        return None


def _stamp_abstain_chexbert(workbook: list[dict]) -> None:
    for row in workbook:
        row["grader_chexbert_label"] = "UNCERTAIN"
        row["grader_chexbert_confidence"] = "low"


# ---------------------------------------------------------------------------
# Grader 2 — Claude Sonnet 4.5 (vision)
# ---------------------------------------------------------------------------


def run_claude_grader(
    workbook: list[dict],
    api_key: str | None = None,
    model_name: str = "claude-sonnet-4-5",
    max_claims: int | None = None,
) -> None:
    """Populate ``grader_claude_*`` fields via the Anthropic API.

    Args:
        workbook: Mutated in place.
        api_key: Optional override; default reads ``ANTHROPIC_API_KEY``.
        model_name: Claude model id.  Sonnet 4.5 is the plan default.
        max_claims: Optional subsetting for pilot runs.

    Cost budget: ≈ $4 for 1000 claims at Claude Sonnet 4.5 pricing.
    """
    try:
        import anthropic
    except ImportError:
        print("claude grader: anthropic SDK not installed")
        _stamp_abstain_claude(workbook)
        return

    try:
        client = anthropic.Anthropic(api_key=api_key)
    except Exception as e:  # noqa: BLE001
        print(f"claude grader: client init failed ({e})")
        _stamp_abstain_claude(workbook)
        return

    import base64

    from tqdm import tqdm

    rows = workbook if max_claims is None else workbook[:max_claims]
    # Per-image b64 cache — multiple claims share an image.
    image_cache: dict[str, str] = {}

    def _get_b64(path: str) -> str | None:
        if path in image_cache:
            return image_cache[path]
        try:
            with open(path, "rb") as f:
                data = f.read()
            b64 = base64.b64encode(data).decode("utf-8")
            image_cache[path] = b64
            return b64
        except Exception:  # noqa: BLE001
            return None

    for row in tqdm(rows, desc="Claude"):
        image_path = row.get("image_path", "")
        claim = row.get("extracted_claim", "")
        gt_report = row.get("ground_truth_report", "")

        image_b64 = _get_b64(image_path)
        if image_b64 is None:
            row["grader_claude_label"] = "UNCERTAIN"
            row["grader_claude_confidence"] = "low"
            row["grader_claude_rationale"] = "image unreadable"
            continue

        try:
            response = client.messages.create(
                model=model_name,
                max_tokens=512,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": _guess_media_type(image_path),
                                    "data": image_b64,
                                },
                            },
                            {
                                "type": "text",
                                "text": (
                                    f"{GRADER_PROMPT}\n\n"
                                    f"Original radiologist report:\n{gt_report}\n\n"
                                    f"Claim to grade:\n{claim}"
                                ),
                            },
                        ],
                    }
                ],
            )
            raw = response.content[0].text
            label, conf, rationale = parse_grader_json(raw)
        except Exception as e:  # noqa: BLE001
            label, conf, rationale = "UNCERTAIN", "low", f"api error: {e}"

        row["grader_claude_label"] = label
        row["grader_claude_confidence"] = conf
        row["grader_claude_rationale"] = rationale


def _stamp_abstain_claude(workbook: list[dict]) -> None:
    for row in workbook:
        row["grader_claude_label"] = "UNCERTAIN"
        row["grader_claude_confidence"] = "low"
        row["grader_claude_rationale"] = "grader unavailable"


def _guess_media_type(path: str) -> str:
    lower = path.lower()
    if lower.endswith(".png"):
        return "image/png"
    if lower.endswith(".jpg") or lower.endswith(".jpeg"):
        return "image/jpeg"
    return "image/png"


# ---------------------------------------------------------------------------
# Grader 3 — MedGemma-4B-IT (with LLaVA-Med fallback)
# ---------------------------------------------------------------------------


MEDGEMMA_CANDIDATES: tuple[str, ...] = (
    "google/medgemma-4b-it",
    "microsoft/llava-med-v1.5-mistral-7b",
    "StanfordAIMI/CXR-LLAVA-v2",
)


def run_medgemma_grader(
    workbook: list[dict],
    max_claims: int | None = None,
) -> None:
    """Populate ``grader_medgemma_*`` fields via a local VLM.

    Tries MedGemma-4B-IT first, then LLaVA-Med, then CXR-LLaVA.  If
    nothing loads, stamps UNCERTAIN.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor
    except Exception as e:  # noqa: BLE001
        print(f"medgemma grader: torch/transformers import failed ({e})")
        _stamp_abstain_medgemma(workbook)
        return

    model, processor, tokenizer, model_id = None, None, None, None
    for cand in MEDGEMMA_CANDIDATES:
        print(f"medgemma grader: trying {cand}")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                cand,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
            processor = AutoProcessor.from_pretrained(
                cand, trust_remote_code=True
            )
            tokenizer = getattr(processor, "tokenizer", None)
            model_id = cand
            print(f"medgemma grader: loaded {cand}")
            break
        except Exception as e:  # noqa: BLE001
            print(f"  failed ({e})")
            continue

    if model is None:
        _stamp_abstain_medgemma(workbook)
        return

    from PIL import Image
    from tqdm import tqdm

    rows = workbook if max_claims is None else workbook[:max_claims]
    image_cache: dict[str, object] = {}

    def _get_image(path: str):
        if path in image_cache:
            return image_cache[path]
        try:
            img = Image.open(path).convert("RGB")
            image_cache[path] = img
            return img
        except Exception:  # noqa: BLE001
            return None

    for row in tqdm(rows, desc=f"VLM:{model_id}"):
        image = _get_image(row.get("image_path", ""))
        claim = row.get("extracted_claim", "")
        gt_report = row.get("ground_truth_report", "")
        if image is None:
            row["grader_medgemma_label"] = "UNCERTAIN"
            row["grader_medgemma_confidence"] = "low"
            row["grader_medgemma_rationale"] = "image unreadable"
            continue

        prompt = (
            f"{GRADER_PROMPT}\n\n"
            f"Original radiologist report:\n{gt_report}\n\n"
            f"Claim to grade:\n{claim}"
        )

        try:
            import torch

            inputs = processor(
                images=image, text=prompt, return_tensors="pt"
            )
            if torch.cuda.is_available():
                inputs = {
                    k: v.to("cuda") if hasattr(v, "to") else v
                    for k, v in inputs.items()
                }
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    num_beams=1,
                )
            decoded = (tokenizer or processor).decode(  # type: ignore[union-attr]
                output_ids[0], skip_special_tokens=True
            )
            if prompt in decoded:
                decoded = decoded.split(prompt, 1)[-1].strip()
            label, conf, rationale = parse_grader_json(decoded)
        except Exception as e:  # noqa: BLE001
            label, conf, rationale = (
                "UNCERTAIN",
                "low",
                f"inference error: {e}",
            )

        row["grader_medgemma_label"] = label
        row["grader_medgemma_confidence"] = conf
        row["grader_medgemma_rationale"] = rationale


def _stamp_abstain_medgemma(workbook: list[dict]) -> None:
    for row in workbook:
        row["grader_medgemma_label"] = "UNCERTAIN"
        row["grader_medgemma_confidence"] = "low"
        row["grader_medgemma_rationale"] = "grader unavailable"


# ---------------------------------------------------------------------------
# Shared JSON parser for Claude + MedGemma outputs
# ---------------------------------------------------------------------------


def parse_grader_json(raw: str) -> tuple[str, str, str]:
    """Parse a grader's raw text response into ``(label, confidence, rationale)``.

    We try strict JSON first, then a regex-scan fallback for the label,
    because VLMs sometimes prepend natural-language preamble or forget
    the outer braces.  On total failure we return
    ``("UNCERTAIN", "low", "unparseable")``.
    """
    import json
    import re

    raw = (raw or "").strip()
    if not raw:
        return "UNCERTAIN", "low", "empty response"

    # Strict JSON attempt — find the first ``{ ... }`` block.
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if match:
        try:
            payload = json.loads(match.group(0))
            label = str(payload.get("label", "")).strip().upper()
            confidence = str(payload.get("confidence", "")).strip().lower()
            rationale = str(payload.get("rationale", "")).strip()
            if label in VALID_LABELS:
                if confidence not in ("high", "medium", "low"):
                    confidence = "medium"
                return label, confidence, rationale[:500]
        except Exception:  # noqa: BLE001
            pass

    # Regex fallback — look for any VALID_LABEL token in the text.
    upper = raw.upper()
    for lbl in VALID_LABELS:
        if lbl in upper:
            return lbl, "low", raw[:200]
    return "UNCERTAIN", "low", "unparseable"


# ---------------------------------------------------------------------------
# Modal entrypoint
# ---------------------------------------------------------------------------


@app.function(
    image=grader_image,
    gpu="H100",
    timeout=60 * 60 * 6,
    volumes={"/data": vol},
    secrets=[modal.Secret.from_name("anthropic")],
)
def run_grader_ensemble(
    workbook_path: str,
    openi_reports_csv: str = "/data/openi_cxr_chexpert_schema.csv",
    max_claims: int | None = None,
    output_filename: str | None = None,
) -> dict:
    """Run all three graders over a workbook JSON file.

    Args:
        workbook_path: Absolute path (on the Modal volume) to the
            annotation workbook produced by Task 1b.
        openi_reports_csv: Path to the OpenI GT report CSV (needed by
            the CheXbert grader for label caching).
        max_claims: If set, only grade the first ``max_claims`` rows
            (pilot mode).
        output_filename: If set, write the graded workbook to this
            filename instead of overwriting the input.

    Returns:
        Summary dict with per-grader completion counts.
    """
    import json
    import os

    with open(workbook_path, "r") as f:
        workbook: list[dict] = json.load(f)

    total = len(workbook)
    print(f"Loaded workbook: {total} rows from {workbook_path}")

    target = workbook if max_claims is None else workbook[:max_claims]
    print(f"Grading {len(target)} rows")

    # Run graders in order — each mutates the rows in place.  We run
    # CheXbert first (cheapest), Claude second, MedGemma last so a
    # budget overrun on the API call still leaves us with the
    # cheap-but-useful CheXbert pass.
    try:
        run_chexbert_grader(target, openi_reports_csv=openi_reports_csv)
    except Exception as e:  # noqa: BLE001
        print(f"chexbert grader crashed: {e}")
        _stamp_abstain_chexbert(target)

    try:
        run_claude_grader(target, max_claims=max_claims)
    except Exception as e:  # noqa: BLE001
        print(f"claude grader crashed: {e}")
        _stamp_abstain_claude(target)

    try:
        run_medgemma_grader(target, max_claims=max_claims)
    except Exception as e:  # noqa: BLE001
        print(f"medgemma grader crashed: {e}")
        _stamp_abstain_medgemma(target)

    # -------- Persist --------
    out_dir = os.path.dirname(workbook_path)
    out_name = output_filename or os.path.basename(workbook_path).replace(
        ".json", "_graded.json"
    )
    out_path = os.path.join(out_dir, out_name)
    with open(out_path, "w") as f:
        json.dump(workbook, f, indent=2)

    # Count per-grader completions for the summary.
    def _count(key: str) -> int:
        return sum(1 for r in target if r.get(key) in VALID_LABELS)

    summary = {
        "workbook_path": workbook_path,
        "output_path": out_path,
        "total_rows": total,
        "graded_rows": len(target),
        "chexbert_valid": _count("grader_chexbert_label"),
        "claude_valid": _count("grader_claude_label"),
        "medgemma_valid": _count("grader_medgemma_label"),
    }
    print(summary)

    summary_path = out_path.replace(".json", "_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    vol.commit()
    return summary


@app.local_entrypoint()
def main(
    workbook_path: str = "/data/eval_data_real_hallucinations/annotation_workbook.json",
    openi_reports_csv: str = "/data/openi_cxr_chexpert_schema.csv",
    max_claims: int | None = None,
    output_filename: str | None = None,
):
    """Run the 3-grader ensemble over a workbook."""
    result = run_grader_ensemble.remote(
        workbook_path=workbook_path,
        openi_reports_csv=openi_reports_csv,
        max_claims=max_claims,
        output_filename=output_filename,
    )
    print(f"\nResult: {result}")
