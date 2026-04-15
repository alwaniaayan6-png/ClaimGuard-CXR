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
        "torch==2.3.0",
        # **Critical (2026-04-14 pre-flight fix):** the grader image
        # is INTENTIONALLY on transformers 4.50.0, NOT 4.40.0.  The
        # CheXagent generation step (in ``generate_real_hallucinations.py``)
        # uses a separate Modal image pinned to 4.40.0 — the two
        # containers do not share a layer, so the pins can diverge.
        # MedGemma-4B-IT is ``Gemma3ForConditionalGeneration`` which
        # first appeared in transformers 4.50 — on 4.40 the load
        # silently falls through to text-only causal LM and the
        # grader stamps all-UNCERTAIN.  4.50 is the floor that
        # unblocks Grader 3.
        "transformers==4.50.0",
        "numpy==1.26.4",
        "pandas==2.2.2",
        "Pillow==10.3.0",
        "tqdm==4.66.4",
        "accelerate==0.30.1",
        "sentencepiece==0.2.0",
        "protobuf==5.27.1",
        # Anthropic SDK pinned to 0.40.0 — known-good for the
        # ``{"type": "image", "source": {"type": "base64", ...}}``
        # vision block shape used in the Claude grader path.
        "anthropic==0.40.0",
        "requests==2.32.3",
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
    "Cardiomegaly": (
        "cardiomegaly",
        "enlarged heart",
        "heart is enlarged",
        "heart enlarged",
        "enlarged cardiac",
        "cardiac enlargement",
        "cardiac silhouette is enlarged",
        "cardiac silhouette enlarged",
        "enlarged cardiac silhouette",
    ),
    "Enlarged Cardiomediastinum": (
        "mediastin",
        "cardiomediastinal",
        "widened mediastinum",
        "mediastinal widening",
    ),
    "Lung Opacity": (
        "opacity",
        "opacities",
        "opacification",
    ),
    "Lung Lesion": (
        "lesion",
        "nodule",
        "mass",
        "pulmonary nodule",
        "lung mass",
    ),
    "Edema": (
        "edema",
        "pulmonary edema",
        "interstitial edema",
        "vascular congestion",
    ),
    "Consolidation": (
        "consolidation",
        "consolidated",
        "airspace disease",
    ),
    "Pneumonia": (
        "pneumonia",
        "infect",
        "infection",
        "infectious",
    ),
    "Atelectasis": (
        "atelectas",
        "lung collapse",
        "collapsed lung",
    ),
    "Pneumothorax": (
        "pneumothorax",
        "ptx",
    ),
    "Pleural Effusion": (
        "effusion",
        "pleural fluid",
    ),
    "Pleural Other": (
        "pleural thickening",
        "pleural plaque",
        "pleural scarring",
    ),
    "Fracture": (
        "fracture",
        "broken rib",
        "rib fracture",
    ),
    "Support Devices": (
        "line",
        "catheter",
        "tube",
        "pacemaker",
        "pacer",
        "icd",
        "endotracheal",
        "ett",
        "picc",
    ),
    "No Finding": (
        "no acute",
        "unremarkable",
        "normal chest",
        "clear lungs",
        "lungs are clear",
        "lungs clear",
        "no abnormality",
        "within normal limits",
        "no focal consolidation",
        "no acute cardiopulmonary",
        "no findings",
    ),
}

# Negation cues that flip a positive mention to "absent" (CheXpert
# label 2).  These are matched within a local window BEFORE the
# pathology keyword so "no pneumothorax" → (Pneumothorax, absent).
#
# Reviewer note (2026-04-14): the original CheXpert labeler (Irvin
# et al., AAAI 2019) uses a 50+ item negation lexicon with parse-
# tree scope.  This conservative subset covers the common radiology
# cases without bringing a parser dependency.  Anything not matched
# here stays POSITIVE by default, which is the CheXpert labeler's
# behavior as well (positive-default + targeted negation).
NEGATION_CUES: tuple[str, ...] = (
    "no ",
    "no\t",
    "no\n",
    "without",
    "absent",
    "absence of",
    "negative for",
    "free of",
    "rule out",
    "rules out",
    "ruled out",
    "no evidence of",
    "no evidence for",
    "no sign of",
    "no signs of",
    "not seen",
    "not identified",
    "not visualized",
    "not visualised",
    "not appreciated",
    "not present",
    "no acute",
    "no focal",
    "clear of",
    "resolved",
    "denies",
    "unremarkable for",
)

# Uncertainty cues that flip a positive mention to "uncertain"
# (CheXpert label 3).  Same local-window rule as negation cues.
UNCERTAINTY_CUES: tuple[str, ...] = (
    "possible",
    "possibly",
    "probable",
    "probably",
    "may represent",
    "may be",
    "may reflect",
    "could be",
    "could represent",
    "suspected",
    "suspicious for",
    "suggestive of",
    "concerning for",
    "cannot exclude",
    "cannot be excluded",
    "difficult to exclude",
    "versus",
    "vs.",
    "consider",
    "questionable",
    "query",
    "indeterminate",
    "equivocal",
    "borderline",
)

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

    Uses a **rule-based CheXpert-style labeler** (not the neural
    CheXbert model) as the primary path.  A 2026-04-14 pre-flight
    reviewer flagged that Stanford's ``stanfordmlgroup/CheXbert`` HF
    upload does not include the 14-head classifier MLP — calling
    ``AutoModel.from_pretrained`` on it silently returns the BERT
    encoder only, and the downstream logits have the wrong shape.
    The old code then stamped every row with all-UNCERTAIN, which
    collapsed the Krippendorff α matrix to 2 effective coders.

    The rule-based labeler mirrors the CheXpert labeler (Irvin et al.,
    AAAI 2019) approach: for each of 14 pathology keywords, check if
    any alias appears in the text, check for negation cues within a
    local window before the mention, check for uncertainty cues in
    the same window.  Emit 0 (blank / not mentioned), 1 (positive),
    2 (negative / absent), or 3 (uncertain).  This is ~5 pp noisier
    than the real neural CheXbert but is deterministic, cheap, and
    produces REAL label vectors instead of a ghost column.

    This function mutates ``workbook`` in place.  It does NOT depend
    on torch or transformers — the rule-based path runs in ~1 ms per
    row, so the grader is effectively free.
    """
    try:
        import pandas as pd
        from tqdm import tqdm
    except Exception as e:  # noqa: BLE001
        print(f"chexbert grader: pandas/tqdm import failed ({e})")
        _stamp_abstain_chexbert(workbook)
        return

    # The GT-report CSV is used only for the cache key; the actual
    # GT report text lives inside each workbook row and is passed
    # directly into the labeler.
    try:
        pd.read_csv(openi_reports_csv)  # sanity check; not used downstream
    except Exception as e:  # noqa: BLE001
        print(
            f"chexbert grader: openi_reports_csv unreadable ({e}); "
            "continuing with per-row GT report text only"
        )

    gt_label_cache: dict[str, list[int]] = {}

    def _cached_labels(report_text: str) -> list[int] | None:
        key = report_text.strip()
        if not key:
            return None
        if key not in gt_label_cache:
            gt_label_cache[key] = _rule_based_chexpert_label_vector(key)
        return gt_label_cache[key]

    for row in tqdm(workbook, desc="CheXbert (rule-based)"):
        claim_text = row.get("extracted_claim", "")
        gt_text = row.get("ground_truth_report", "")
        claim_vec = _rule_based_chexpert_label_vector(claim_text)
        gt_vec = _cached_labels(gt_text)
        label, conf = label_from_chexbert_diff(claim_text, claim_vec, gt_vec)
        row["grader_chexbert_label"] = label
        row["grader_chexbert_confidence"] = conf


# Conjunctions / punctuation that TERMINATE a negation or uncertainty
# scope.  When scanning the window BEFORE a pathology keyword, any
# negation cue to the LEFT of the nearest scope-terminator is ignored.
# This matches the CheXpert labeler's sentence-level scope rule
# without requiring a full dependency parser.
#
# Example: "There is no cardiomegaly, but a large pneumothorax is
# seen" — the "no" applies to cardiomegaly (before the "but"), but
# NOT to pneumothorax (after the "but").
SCOPE_TERMINATORS: tuple[str, ...] = (
    "but",
    "however",
    "although",
    "though",
    "whereas",
    "while",
    ";",
    ".",
    "?",
    "!",
)


def _find_in_window_with_scope(
    cues: tuple[str, ...],
    window: str,
) -> bool:
    """Return True if any cue appears in the window AND is not cut
    off from the right edge by a scope-terminator.

    Algorithm: scan from the right edge of the window leftward.  Stop
    as soon as we see a scope-terminator.  If any cue appears in the
    unterminated suffix, return True.
    """
    if not window:
        return False
    # Find the rightmost scope-terminator position.  Anything to its
    # LEFT is out of scope for the current keyword.
    cut = 0
    for terminator in SCOPE_TERMINATORS:
        pos = window.rfind(terminator)
        if pos >= 0 and pos + len(terminator) > cut:
            cut = pos + len(terminator)
    effective = window[cut:]
    return any(cue in effective for cue in cues)


def _rule_based_chexpert_label_vector(text: str) -> list[int]:
    """CheXpert-style 14-class label vector from a free-text report.

    Returns a list of 14 integers, one per pathology in
    ``CHEXBERT_PATHOLOGIES`` order, with values:

        * 0 — not mentioned (blank / missing)
        * 1 — mentioned and positive
        * 2 — mentioned and negative (absent)
        * 3 — mentioned and uncertain / hedged

    Algorithm:
        1. Lowercase the input.
        2. For each pathology, find all keyword hits (substring match,
           from ``PATHOLOGY_KEYWORDS``).
        3. For each hit, look at the 60-character window BEFORE the
           hit position.  Trim the window at the rightmost
           scope-terminator (conjunction, period, semicolon, etc.)
           so only the SAME-CLAUSE context counts.
        4. If any ``NEGATION_CUES`` appears in the effective window → 2.
        5. Else if any ``UNCERTAINTY_CUES`` appears in the effective
           window → 3.
        6. Else → label 1 (positive).
        7. If a pathology has multiple hits with different labels,
           pick the LAST one (latest sentence wins — matches CheXpert
           labeler behaviour on report-level documents).

    "No Finding" is inferred:
        * If any other pathology is positive → 0 (blank for No Finding)
        * If the text matches ``PATHOLOGY_KEYWORDS["No Finding"]``
          directly → 1 (positive for No Finding)
        * Otherwise → 0 (blank)

    Empty input returns all-zeros.
    """
    if not text or not text.strip():
        return [0] * len(CHEXBERT_PATHOLOGIES)

    lower = text.lower()
    vec: list[int] = [0] * len(CHEXBERT_PATHOLOGIES)

    for idx, pathology in enumerate(CHEXBERT_PATHOLOGIES):
        if pathology == "No Finding":
            continue  # handled after the main loop
        keywords = PATHOLOGY_KEYWORDS.get(pathology, ())
        if not keywords:
            continue
        # Find all hits for this pathology.
        hits: list[tuple[int, int]] = []  # (position, label)
        for kw in keywords:
            start = 0
            while True:
                i = lower.find(kw.lower(), start)
                if i < 0:
                    break
                # Local window: 60 chars before the hit, then
                # scope-terminated so only the same-clause context
                # counts toward negation / uncertainty.
                window_start = max(0, i - 60)
                window = lower[window_start:i]
                if _find_in_window_with_scope(NEGATION_CUES, window):
                    hits.append((i, 2))  # absent
                elif _find_in_window_with_scope(UNCERTAINTY_CUES, window):
                    hits.append((i, 3))  # uncertain
                else:
                    hits.append((i, 1))  # positive
                start = i + max(1, len(kw))
        if hits:
            # Last hit wins — matches CheXpert labeler behaviour on
            # multi-sentence reports where later sentences override
            # earlier tentative mentions.
            hits.sort(key=lambda h: h[0])
            vec[idx] = hits[-1][1]

    # "No Finding" inference.
    nf_idx = CHEXBERT_PATHOLOGIES.index("No Finding")
    any_positive = any(
        vec[i] == 1 for i in range(len(vec)) if i != nf_idx
    )
    nf_keywords = PATHOLOGY_KEYWORDS.get("No Finding", ())
    nf_matches = any(kw.lower() in lower for kw in nf_keywords)
    if any_positive:
        vec[nf_idx] = 0  # some pathology is positive → no "no finding"
    elif nf_matches:
        vec[nf_idx] = 1
    else:
        vec[nf_idx] = 0

    return vec


def _chexbert_label_vector(model, tokenizer, text: str) -> list[int] | None:
    """LEGACY wrapper — uses the rule-based labeler regardless of
    ``model`` / ``tokenizer`` arguments.

    The old neural CheXbert path has been retired (see
    ``run_chexbert_grader`` docstring).  This stub is preserved so
    any external caller importing ``_chexbert_label_vector`` still
    gets a real 14-dim vector instead of an AttributeError.
    """
    if not text or not text.strip():
        return [0] * len(CHEXBERT_PATHOLOGIES)
    try:
        return _rule_based_chexpert_label_vector(text)
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
        # max_retries at the SDK level gives us exponential backoff on
        # 429 / 529 / connection errors for free.  Reviewer flag: the
        # prior code had no retry and would silently stamp UNCERTAIN on
        # every rate-limited call, collapsing the grader matrix.
        client = anthropic.Anthropic(api_key=api_key, max_retries=5)
    except Exception as e:  # noqa: BLE001
        print(f"claude grader: client init failed ({e})")
        _stamp_abstain_claude(workbook)
        return

    import base64
    import io
    import time

    from tqdm import tqdm

    rows = workbook if max_claims is None else workbook[:max_claims]
    # Per-image b64 cache — multiple claims share an image.  Caches the
    # POST-RESIZE bytes so re-encoding is amortized across claims.
    image_cache: dict[str, tuple[str, str]] = {}  # path -> (b64, media_type)

    def _load_resized_b64(
        path: str,
        *,
        max_side: int = 1568,
        jpeg_quality: int = 85,
        max_bytes: int = 4_500_000,  # soft cap, Claude hard limit is 5 MB
    ) -> tuple[str, str] | None:
        """Load, resize, re-encode, and base64-encode an OpenI image.

        Reviewer flag: the prior code read the raw file and base64-
        encoded it directly.  OpenI PNGs can exceed 3.75 MB uncompressed,
        and base64 inflates by ~33%, so the resulting payload could
        cross Claude's 5 MB vision cap and trigger API errors.  The fix:

        * Lazy-import PIL (only when the grader actually runs).
        * Resize so the long side is ≤ ``max_side`` (1568 px — the
          native resolution Claude's vision preprocessor targets).
        * Re-encode as JPEG quality 85 (good enough for radiology
          intensity patterns, massively smaller than PNG).
        * Fall back to PNG if PIL can't convert (RGBA, etc.).
        * Guard against the 5 MB cap — if the resized payload is
          still too big, halve the long side and retry.

        Returns ``(b64_string, media_type)`` or ``None`` on any
        unrecoverable error.  The cached value is the resized bytes,
        not the raw file, so memory pressure stays bounded.
        """
        if path in image_cache:
            return image_cache[path]
        try:
            from PIL import Image  # noqa: WPS433 lazy
        except ImportError:
            # Fall back to raw encoding if PIL isn't in the image.
            try:
                with open(path, "rb") as f:
                    data = f.read()
            except Exception:  # noqa: BLE001
                return None
            b64 = base64.b64encode(data).decode("utf-8")
            entry = (b64, _guess_media_type(path))
            image_cache[path] = entry
            return entry

        try:
            img = Image.open(path)
        except Exception:  # noqa: BLE001
            return None

        # Convert grayscale / RGBA / P-mode to RGB for JPEG.
        if img.mode != "RGB":
            img = img.convert("RGB")

        side = max_side
        for _attempt in range(3):
            w, h = img.size
            long_side = max(w, h)
            if long_side > side:
                scale = side / long_side
                new_size = (int(w * scale), int(h * scale))
                resized = img.resize(new_size, Image.LANCZOS)
            else:
                resized = img
            buf = io.BytesIO()
            resized.save(buf, format="JPEG", quality=jpeg_quality)
            blob = buf.getvalue()
            if len(blob) <= max_bytes:
                b64 = base64.b64encode(blob).decode("utf-8")
                entry = (b64, "image/jpeg")
                image_cache[path] = entry
                return entry
            side = side // 2  # halve and retry
        # Still too big after 3 halvings (unlikely for X-rays) — drop.
        return None

    # Backwards-compatible wrapper used by the loop below.  Returns
    # just the b64 string (media type is tracked separately).
    def _get_b64(path: str) -> str | None:
        entry = _load_resized_b64(path)
        return entry[0] if entry else None

    for row in tqdm(rows, desc="Claude"):
        image_path = row.get("image_path", "")
        claim = row.get("extracted_claim", "")
        gt_report = row.get("ground_truth_report", "")

        entry = _load_resized_b64(image_path)
        if entry is None:
            row["grader_claude_label"] = "UNCERTAIN"
            row["grader_claude_confidence"] = "low"
            row["grader_claude_rationale"] = "image unreadable"
            continue
        image_b64, media_type = entry

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
                                    "media_type": media_type,
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
        except anthropic.RateLimitError as e:
            # SDK's 5 retries already covered the transient-burst case;
            # at this point we're genuinely rate-limited.  Sleep 30 s
            # and retry once more before stamping UNCERTAIN.
            time.sleep(30.0)
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
                                        "media_type": media_type,
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
            except Exception as e2:  # noqa: BLE001
                label, conf, rationale = (
                    "UNCERTAIN", "low", f"rate limit after retry: {e2}",
                )
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

    **Loader class rotation (2026-04-14 pre-flight fix).**  MedGemma-4B-IT
    is ``Gemma3ForConditionalGeneration`` — a vision-language model
    whose correct HF loader is ``AutoModelForImageTextToText`` (added
    in transformers 4.50).  The prior code used
    ``AutoModelForCausalLM``, which on 4.50+ silently falls back to
    the text-only causal head and discards the vision tower, so the
    grader would see no image at all and stamp every row UNCERTAIN.

    This loop now tries the image-text-to-text class FIRST (correct
    for MedGemma-4B-IT and most modern radiology VLMs), then falls
    back to ``AutoModelForVision2Seq`` (older LLaVA variants), then
    ``AutoModelForCausalLM`` (legacy text-only path — last resort).
    """
    try:
        import torch
        from transformers import AutoProcessor
    except Exception as e:  # noqa: BLE001
        print(f"medgemma grader: torch/transformers import failed ({e})")
        _stamp_abstain_medgemma(workbook)
        return

    # Loader classes tried in order of correctness.  Each one is
    # imported lazily because older transformers versions may not
    # export all three names.
    def _load_loader_class(name: str):
        try:
            import transformers as _t
            return getattr(_t, name, None)
        except ImportError:
            return None

    loader_classes: list[tuple[str, object]] = []
    for cls_name in (
        "AutoModelForImageTextToText",
        "AutoModelForVision2Seq",
        "AutoModelForCausalLM",
    ):
        cls = _load_loader_class(cls_name)
        if cls is not None:
            loader_classes.append((cls_name, cls))
    if not loader_classes:
        print("medgemma grader: no usable loader class in transformers")
        _stamp_abstain_medgemma(workbook)
        return

    model, processor, tokenizer, model_id = None, None, None, None
    for cand in MEDGEMMA_CANDIDATES:
        print(f"medgemma grader: trying {cand}")
        for cls_name, cls in loader_classes:
            try:
                model = cls.from_pretrained(
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
                print(f"medgemma grader: loaded {cand} via {cls_name}")
                break
            except Exception as e:  # noqa: BLE001
                print(f"  {cls_name} failed ({e})")
                continue
        if model is not None:
            break

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
