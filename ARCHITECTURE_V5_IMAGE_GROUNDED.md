# ClaimGuard-CXR v5 — Image-Grounded Claim Verification

**Status:** design, pre-implementation
**Author:** Aayan Alwani (Laughney Lab, Weill Cornell Medicine)
**Target venue:** Nature Machine Intelligence
**Supersedes:** `ARCHITECTURE.md` (v1–v4, text-only verifier with synthetic perturbations)

---

## 0. Executive summary

ClaimGuard-CXR v5 ("Path B") reframes claim-level hallucination detection in radiology reports as an **image-grounded verification** problem. Instead of validating claims against another radiologist's text (which is inaccessible without PhysioNet credentialing), v5 validates claims against **radiologist-drawn pixel-level annotations** from public datasets — bounding boxes, segmentation masks, and anatomically-localized labels. This converts the "ground truth" layer from text-to-text NLI to text-to-image grounding, which is (a) radiologically more rigorous, (b) legally accessible without institutional credentialing, and (c) architecturally more defensible for an AI-safety contribution.

**Three core contributions:**

1. **ClaimGuard-GroundBench.** The first public benchmark for radiology claim-level hallucination detection where every test claim has pixel-level radiologist ground truth. Assembled from MS-CXR, CheXmask, PadChest-localized, RSNA Pneumonia, SIIM-ACR Pneumothorax, Object-CXR, ChestX-Det10.
2. **Image-grounded claim verifier.** A BiomedCLIP-image + RoBERTa-text cross-modal verifier trained with a multi-objective loss (classification + grounding supervision + evidence-masking consistency + contrastive evidence-substitution) that demonstrably *uses* the image (image-masked ablation drops ≥ 15 pp).
3. **Conformal FDR control under institutional distribution shift,** evaluated with weighted + doubly-robust variants across 6+ sites and 5+ VLM generators. Provenance-aware gating handles self-consistency failure modes.

**Non-goals:**
- Text-only verification (v4 is the last text-only checkpoint; v5 is strictly multimodal).
- Synthetic-perturbation-dominant training (v5 uses real VLM-generated hallucinations as primary training signal).
- Any dependency on PhysioNet-credentialed datasets (MIMIC-CXR, ReXVal, ReXErr, VinDr-CXR are explicitly out of scope for the primary story; they may be added later as bonus validation if credentialing completes).

**What is novel vs v4:**

| Axis | v4 (text-only) | v5 (image-grounded) |
|---|---|---|
| Input | claim + retrieved text evidence | claim + retrieved text evidence + CXR image |
| Ground truth | LLM-judge ensemble | radiologist-drawn pixel annotations (primary) + LLM ensemble (secondary) |
| Training data | synthetic perturbations on CheXpert Plus reports | real VLM hallucinations on public images + synthetic perturbations + MS-CXR grounding pairs |
| HO-baseline gap | negative (-2.61 pp, evidence-blind ≥ evidence-using) | target ≥ 10 pp closed via contrastive evidence loss |
| Image-masked degradation | N/A | target ≥ 15 pp |
| FDR framework | inverted cfBH only | inverted + weighted + doubly-robust cfBH, multi-site |
| Benchmark release | none | ClaimGuard-GroundBench, Zenodo DOI, HF weights |

---

## 1. Scope, constraints, and assumptions

### 1.1 Hard constraints

- **No credentialed data.** No MIMIC-CXR, MIMIC-IV, VinDr-CXR, ReXVal, ReXErr, RadGraph-XL (MIMIC-derived). CheXpert Plus (Stanford registration) and PadChest (BIMCV registration) are allowed because they are registration-only, not DUA-credentialed.
- **No recruited radiologist panel.** Public radiologist-drawn annotations (MS-CXR, CheXmask, PadChest, RSNA, SIIM, Object-CXR, ChestX-Det10) are the ground-truth layer. A lightweight Weill Cornell radiology consultant (review co-author) is desirable but not blocking.
- **Single GPU class (H100)** on Modal; no multi-GPU distributed training beyond DDP on 2–8 H100s within one Modal container.
- **Budget cap ~$2,500** (pending approval; current spend $93 of $900).

### 1.2 Soft constraints

- Reproducibility: every artifact ships with a Docker image, pinned model revisions, Zenodo DOI for data, HF for weights, GitHub release with semver for code.
- Privacy: no patient-identifiable data leaves Modal or Weill Cornell infrastructure unless it has been through the PII scrubber (Presidio + CXR regex).
- Compute hygiene: every Modal launch is preceded by a pre-flight Opus logic-review agent per user CLAUDE.md rules; unit tests must pass before `modal run`.

### 1.3 Assumptions

- BiomedCLIP's vision tower (ViT-B/16, trained on PMC-OA image-text) is strong enough on CXR when light-fine-tuned on CheXpert Plus images. If not, fall back to BioMedCLIP-PubMedBERT-CXR-domain-adapted (train a domain-adapter on CheXpert) before considering RadDINO (which has MIMIC-CXR contamination).
- RoBERTa-large text encoder remains the text backbone for continuity with v4 and because its NLI behaviour is well-characterized.
- Image annotations in MS-CXR / CheXmask / PadChest-localized are sufficient coverage to ground ~70% of test claims. Uncovered claims (no annotation of that finding type available) are labeled `NO_GT` and excluded from the headline metrics, with a coverage statistic reported.

---

## 2. System overview

```
                                 ┌──────────────────────────────────────┐
                                 │   ClaimGuard-GroundBench (public)    │
                                 │                                       │
                                 │  Images: CheXpert+ / OpenI / PadChest │
                                 │          / BRAX / RSNA / SIIM /       │
                                 │          Object-CXR / ChestX-Det10    │
                                 │                                       │
                                 │  Radiologist pixel annotations:       │
                                 │    MS-CXR boxes, CheXmask masks,      │
                                 │    PadChest-localized, RSNA BBs,      │
                                 │    SIIM masks, Object-CXR, ChestX-Det │
                                 │                                       │
                                 │  VLM-generated reports (5 models ×    │
                                 │  all images) → extracted claims       │
                                 └──────────────────────────────────────┘
                                                   │
                          ┌────────────────────────┴──────────────────────┐
                          ▼                                               ▼
        ┌────────────────────────────────┐         ┌─────────────────────────────┐
        │      Training pipeline         │         │     Evaluation pipeline     │
        │                                │         │                             │
        │  Per-claim triples:            │         │  For each (image, claim):   │
        │    (image, claim, evidence)    │         │   1. Parse claim → struct   │
        │  Labels:                       │         │   2. Query annotations       │
        │    synthetic perturbation      │         │   3. Compute pixel-grounded  │
        │    + real-VLM-hallucination    │         │      GT label               │
        │    + MS-CXR grounding pairs    │         │   4. Run v5 verifier        │
        │  Loss = CE + λ_g L_ground      │         │   5. Conformal → green set  │
        │       + λ_c L_consist          │         │   6. Provenance gate        │
        │       + λ_e L_contrast         │         │   7. Aggregate metrics      │
        │                                │         │      (acc, FDR, power,      │
        │                                │         │       calibration, IoU,     │
        │                                │         │       parity)               │
        └────────────────────────────────┘         └─────────────────────────────┘
                          │                                               │
                          ▼                                               ▼
        ┌────────────────────────────────────────────────────────────────────┐
        │                     v5 Image-Grounded Verifier                     │
        │                                                                    │
        │  ┌──────────────────┐     ┌──────────────────┐                    │
        │  │ Image encoder    │     │ Text encoder     │                    │
        │  │ BiomedCLIP ViT   │     │ RoBERTa-large    │                    │
        │  │ 14×14 + CLS      │     │ [CLS] claim [SEP]│                    │
        │  │   d=768          │     │  evidence [SEP]  │                    │
        │  └────────┬─────────┘     └────────┬─────────┘                    │
        │           │                        │                              │
        │           └──────────┬─────────────┘                              │
        │                      │                                            │
        │              ┌───────▼────────┐                                   │
        │              │ Linear proj    │   (both to d=768)                 │
        │              └───────┬────────┘                                   │
        │                      │                                            │
        │            ┌─────────▼──────────┐                                 │
        │            │  4-layer cross-    │                                 │
        │            │  modal transformer │                                 │
        │            │  (bidirectional    │                                 │
        │            │   img↔txt attn)    │                                 │
        │            └─────────┬──────────┘                                 │
        │                      │                                            │
        │      ┌───────────────┼───────────────┬─────────────────┐         │
        │      ▼               ▼               ▼                 ▼         │
        │  ┌────────┐    ┌──────────┐    ┌──────────┐    ┌─────────────┐  │
        │  │verdict │    │grounding │    │uncertainty│    │score        │  │
        │  │head (2)│    │head      │    │head       │    │head (sigmoid│  │
        │  │        │    │(14×14    │    │(MC-dropout│    │over support)│  │
        │  │        │    │attn mask)│    │ ensemble) │    │             │  │
        │  └────────┘    └──────────┘    └──────────┘    └─────────────┘  │
        └────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
           ┌────────────────────────────────────────────────────────────┐
           │        Conformal layer + Provenance gate                   │
           │                                                             │
           │  Scores → inverted cfBH (+ weighted, + doubly-robust)       │
           │  → test-claim p-values                                      │
           │  → global BH at level α                                     │
           │  → green-set G_α                                            │
           │  → apply provenance gate: (same-model / unknown) → downgrade│
           │  → final certified-safe set                                 │
           └────────────────────────────────────────────────────────────┘
```

---

## 3. Data layer

### 3.1 Dataset inventory (public, no credentialing)

| Dataset | n CXRs | Annotation type | Annotation source | Access | License | Primary role |
|---|---|---|---|---|---|---|
| **CheXpert Plus** | 223,462 / 16k test | reports + CheXpert-labeler 14-class | Stanford labeler + 8 radiologists (test) | registration | Stanford AIMI | Training images + reports |
| **OpenI** | 3,996 | reports | Indiana U. radiologists (written) | free | NLM | Silver eval + provenance |
| **PadChest** | 160,868 | reports + 174 structured labels (27% radiologist, rest NLP) | BIMCV radiologists (subset) | registration | CC BY-SA | Image-grounded eval (localized subset) |
| **BRAX** | 40,967 | reports + CheXpert-labeler labels | Brazilian radiologists (reports) | IEEE DataPort registration | CC BY-NC | Silver eval (multi-lingual/geographic) |
| **MS-CXR** | 1,162 image-phrase pairs | bounding boxes + anchor phrases | MSR radiologists | HuggingFace | MSR research | **Grounding supervision + eval** |
| **CheXmask** | ~676k anatomy masks | lung / heart / clavicle segmentation | DeepLabV3 + radiologist-verified subset | Physionet **(wait: this IS on PhysioNet; substitute with open-source version)** | — | **anatomical location resolver** |
| **RSNA Pneumonia** | 30,227 | pneumonia bounding boxes | RSNA radiologist panel | Kaggle | RSNA research | Pneumonia grounding |
| **SIIM-ACR Pneumothorax** | 12,047 | pneumothorax segmentation masks | SIIM radiologists | Kaggle | SIIM research | Pneumothorax grounding |
| **Object-CXR** | 9,000 | foreign object bounding boxes | MICCAI 2020 radiologists | Github (JF Healthcare) | CC BY | Foreign-object grounding |
| **ChestX-Det10** | 3,543 | 10-class bounding boxes | radiologist panel (Chen et al. 2020) | Github | CC BY-NC | 10-finding grounding |
| **VinDr-PCXR (pediatric)** | 9,125 | bounding boxes | Vietnamese radiologists | **PhysioNet → out of scope** | — | excluded |
| **NIH ChestX-ray14** | 112,120 | labels only (no reports) | NLP on reports | free | CC0 | Image-only training augmentation |

**Note on CheXmask:** the canonical release is PhysioNet-credentialed. Substitute with **CheXmask-Open** (independent open-source re-derivation using HybridGNet on public CheXpert PA images) or compute our own anatomy segmentation using a lightweight U-Net trained on public anatomy-labeled CXR (e.g., `torchxrayvision`'s anatomy head). Anatomy resolution is needed for the claim-matcher and must not depend on credentialed data.

### 3.2 Directory layout (Modal volume `claimguard-data`)

```
/data/
  chexpert_plus/
    images/                   # jpg, 512x512
    reports/                  # text files, section-normalized
    splits_v5/
      train_claims.jsonl      # ~250k claims × 5 VLM-generators + synthetic
      cal_claims.jsonl        # 20k claims, held-out patients
      test_claims.jsonl       # 20k claims, held-out patients
  padchest/
    images/
    reports_en/               # auto-translated to English, dual-LLM verified
    localized_v5/             # subset with radiologist bounding boxes
  brax/
    images/
    reports/
    splits_v5/
  openi/
    images/
    reports/
    splits_v5/
  ms_cxr/
    images/                   # from MIMIC — substitute with MS-CXR HF release on CheXpert images
    phrases.jsonl             # phrase, box, image_id
  rsna_pneumonia/
    images/
    bboxes.jsonl
  siim_acr_pneumothorax/
    images/
    masks/                    # png rle-decoded
  object_cxr/
    images/
    bboxes.jsonl
  chestx_det10/
    images/
    bboxes.jsonl
  anatomy_masks_v5/           # our open-source CheXmask substitute
    <image_id>.png            # 5-class segmentation (R/L upper/mid/lower + heart)
  groundbench_v5/             # assembled ClaimGuard-GroundBench
    manifest.parquet
    per_claim_gt.jsonl
    unified_metadata.parquet
```

### 3.3 Claim corpus construction

Five VLM generators are run over every training/eval image:

- **MAIRA-2** (Microsoft, public weights via HF)
- **CheXagent-8b** (StanfordAIMI/CheXagent-8b)
- **MedGemma-4B** (google/medgemma-4b-it; gated — fallback to non-gated MedGemma-27B if access denied)
- **LLaVA-Rad** (microsoft/llava-med-v1.5-mistral-7b)
- **Llama-3.2-11B-Vision** (general-domain control)

Each generator's output is run through the claim extractor (§4.1), producing sentence-level claims tagged with `(image_id, generator_id, generator_temperature, generator_seed, claim_text, report_context)`.

**Training mix:**
- 50% real VLM hallucinations (labeled by LLM ensemble, validated against radiologist annotations where available)
- 30% synthetic perturbations from v4's 12-type taxonomy (kept for coverage of rare types)
- 20% grounding-supervision pairs from MS-CXR (claim text = MS-CXR phrase; label from bounding-box presence)

### 3.4 LLM-labeler ensemble

Used only when no radiologist annotation is available (silver fallback).

Three labelers, majority vote with confidence-weighted tie-break:
- GPT-4o with vision (Azure OpenAI or direct API)
- Claude Sonnet 4.5 with vision (Anthropic API)
- Llama-3.1-405B-Instruct (no vision; text-only fallback when the other two disagree on a non-visual claim)

Per-label output schema:
```json
{
  "label": "SUPPORTED | CONTRADICTED | NOVEL_PLAUSIBLE | NOVEL_HALLUCINATED | UNCERTAIN",
  "confidence": 0.0-1.0,
  "rationale": "≤30 words",
  "grader_id": "...",
  "grader_version": "...",
  "prompt_version": "..."
}
```

Ensemble rule: 2-of-3 majority; ties → UNCERTAIN; confidence = min of agreeing labelers.

**Validation of the labeler ensemble:** before using ensemble labels as training signal, compute κ against MS-CXR + RSNA + SIIM + ChestX-Det10 radiologist annotations on a held-out 500-claim subset. Ship only claim categories where κ ≥ 0.70. Categories below threshold are dropped from silver training.

### 3.5 PII scrubber

Every released text artifact passes through:
1. **Presidio** (Microsoft) with medical recognizers.
2. **Custom CXR regex**: dates, accession numbers, "Dr./Mr./Mrs.", institution names (CheXpert / Indiana / BIMCV / Fleury), room numbers.
3. **Manual audit** on a 500-sample random slice before release; document residual false-negative rate.

---

## 4. Claim parser and image-grounded matcher

### 4.1 Claim extractor

Input: free-text radiology report (sections: Impression, Findings, Technique).
Output: list of sentence-level claims with `(claim_text, finding_category, location, laterality, severity, temporality, comparison)`.

**Implementation:**
- Primary: GPT-4o-mini with a locked prompt version (`prompt_v5_claim_extract.jinja`). Validated on RadGraph-style reference claims (use `torchxrayvision`'s claim-extraction head as validation oracle because RadGraph-XL is MIMIC-credentialed).
- Fallback: a fine-tuned `T5-base-claim-extractor` trained on CheXpert Plus reports + LLM-labeled claim spans.
- Required precision ≥ 0.95 on validation set; locked and versioned before any downstream use.

### 4.2 Claim-to-annotation matcher

The central novel component of v5. Pseudocode:

```python
def compute_image_grounded_gt(image_id, claim_struct, annotation_db) -> GTLabel:
    """
    Given a structured claim and all available radiologist annotations on image_id,
    return one of {SUPPORTED, CONTRADICTED, NO_GT, NO_FINDING}.

    Semantics:
      SUPPORTED:    a radiologist annotation exists that matches the claim's
                    (finding, location, laterality, severity) within tolerance.
      CONTRADICTED: a radiologist annotation of the same finding family exists
                    but location or laterality is mismatched; OR the claim asserts
                    a finding that is explicitly absent per structured labels (e.g.,
                    CheXpert "no consolidation" = 0).
      NO_GT:        no radiologist annotation of that finding category exists on
                    this image; GT cannot be established.
      NO_FINDING:   image is labeled "no finding" / normal across all applicable
                    annotation sets; claims asserting any finding → CONTRADICTED.
    """
    finding = claim_struct.finding
    loc = claim_struct.location              # CheXmask anatomical region id
    lat = claim_struct.laterality            # {L, R, bilateral, none}

    # 1. Collect all annotations of the claim's finding family on this image
    cands = annotation_db.query(image_id=image_id, finding_family=finding_family(finding))

    # 2. If image has no annotations of that family AND no structured negative label:
    if not cands and not annotation_db.has_structured_negative(image_id, finding):
        return GTLabel.NO_GT

    # 3. Structured negative present → claim of that finding is CONTRADICTED
    if annotation_db.has_structured_negative(image_id, finding) and not cands:
        return GTLabel.CONTRADICTED

    # 4. Cand exists → check spatial + laterality match
    for ann in cands:
        if laterality_match(ann.laterality, lat) and \
           anatomical_overlap(ann.bbox, loc, anatomy_mask=annotation_db.anatomy(image_id)) >= TAU_IOU:
            return GTLabel.SUPPORTED

    # 5. Finding present but wrong location/laterality → CONTRADICTED
    return GTLabel.CONTRADICTED
```

**Key subroutines:**

- `finding_family(finding)`: maps claim findings to a common vocabulary. Uses a curated ontology mapping:
  - CheXpert 14 findings → RSNA / SIIM / Object-CXR / ChestX-Det10 / MS-CXR / PadChest 174 → unified 23-class finding family.
  - Mapping stored in `data/annotations/finding_ontology_v5.yaml`, reviewed by a radiology consultant if available, otherwise derived from SNOMED-CT + RadLex cross-references.

- `anatomical_overlap(bbox, claim_location, anatomy_mask)`: compute IoU between the annotation's bounding box and the claim's referenced anatomical region per the anatomy segmentation mask. `TAU_IOU = 0.3` (loose, since annotations and claim-referenced regions are coarse).

- `laterality_match(ann_lat, claim_lat)`: claim "left upper lobe" must match annotation on left side (determined by centroid x coordinate relative to image midline, corrected for radiograph orientation per DICOM `PatientOrientation` when available).

**Coverage analysis:** report, per test set, the fraction of claims that receive `NO_GT` vs resolved ground truth. Headline metrics compute over the resolved subset; `NO_GT` statistics are reported separately with a discussion of coverage.

### 4.3 Validation of the matcher

Two independent validations before production use:

1. **Matcher vs MS-CXR native consensus.** MS-CXR ships with phrase-to-box pairs; the matcher's SUPPORTED/CONTRADICTED on MS-CXR claims must reach ≥ 95% accuracy against MS-CXR's own consensus.
2. **Matcher vs LLM-ensemble on a 500-claim subset.** If matcher and LLM ensemble agree ≥ 85% of the time, cross-use for silver labels is defensible. Disagreements flagged for a human (radiology consultant or careful author inspection).

---

## 5. Model architecture

### 5.1 Component specifications

**Image encoder (primary):** `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`
- Input: 224×224 RGB (CXR is grayscale; replicate to 3 channels).
- Tokens: 196 patch tokens (14×14) + 1 CLS → 197 × 768.
- Unfrozen layers: last 4 transformer blocks + final LayerNorm + CLS. Earlier layers frozen.
- Domain adapter: 2-layer MLP head (768 → 768 → 768) trained from scratch on CheXpert Plus images with MAE (masked image reconstruction) pretraining for 1 epoch before downstream training. Closes the PMC-OA → CXR domain gap without touching MIMIC.

**Text encoder:** `roberta-large`
- Input: `[CLS] claim [SEP] evidence [SEP]`, max 256 tokens.
- Tokens: 256 × 1024.
- Unfrozen layers: last 8 transformer blocks + final LayerNorm + CLS. Earlier 16 frozen.

**Projection layers:** two independent `Linear(d_in → 768)` projecting image and text to a shared 768-dim space, followed by `LayerNorm`.

**Cross-modal fusion:** 4-layer standard transformer encoder, 12 heads, d_model=768, d_ff=3072, dropout=0.1. Bidirectional attention over the concatenated [image_tokens; text_tokens] sequence (total 453 tokens). A new learnable `[VERDICT]` token is prepended.

**Output heads:**
- `verdict_head`: Linear(768, 2) on `[VERDICT]` token → {not-contradicted, contradicted}.
- `score_head`: Linear(768, 1) + sigmoid → continuous support probability (used by cfBH).
- `grounding_head`: Linear(768, 1) on each image patch token → 14×14 attention map; trained to match MS-CXR bounding-box masks via BCE loss on the binarized mask.
- `uncertainty_head`: 5-member MC-dropout ensemble over `verdict_head` with dropout p=0.2; predictive entropy as the uncertainty.

Total parameters: ~480M (roughly: BiomedCLIP 86M + RoBERTa-large 355M + fusion+heads ~40M). Fits one H100 (80 GB) with batch size 32 at 224×224 / 256-token.

### 5.2 Checkpoint versioning

`v5.0-base`: BiomedCLIP + RoBERTa + fusion, no grounding head, trained on v4 synthetic data. Sanity check.
`v5.1-ground`: add grounding head, train with MS-CXR supervision.
`v5.2-real`: add real VLM hallucinations to training mix.
`v5.3-contrast`: add evidence-contrastive loss.
`v5.4-final`: all of the above + MC-dropout uncertainty head, 5-seed mean.

Each checkpoint saved to Modal volume `/data/checkpoints/claimguard_v5/<version>/best.pt` with a `manifest.json` recording: git SHA, data hash, random seed, wandb run id, hyperparameters, evaluation summary.

---

## 6. Training pipeline

### 6.1 Losses

```
L_total = L_cls + λ_g · L_ground + λ_c · L_consist + λ_e · L_contrast + λ_u · L_uncert
```

| Term | Form | λ (default) | Purpose |
|---|---|---|---|
| `L_cls` | binary cross-entropy on verdict | 1.0 | primary task |
| `L_ground` | BCE over 14×14 grounding map against MS-CXR box mask (only on MS-CXR subset) | 0.5 | force image attention to attach to radiologist-drawn regions |
| `L_consist` | KL(p(y\|x, image, evidence), p(y\|x, image_masked, evidence)) for "not contradicted" claims; and symmetric for "contradicted" | 0.3 | image-evidence consistency (the HO-gap and image-masked-degradation fixes) |
| `L_contrast` | max(0, margin − (s(claim, correct_evidence) − s(claim, wrong_evidence))) | 0.3 | force the model to prefer matching evidence over mismatched evidence |
| `L_uncert` | MC-dropout ensemble disagreement term (inverse ECE) | 0.1 | calibrate uncertainty |

λ values are starting points; tune on a 10% subset with Hydra + Optuna.

### 6.2 Adversarial filtering for HO-gap

Per CLAUDE.md decision memory: v4's HO-baseline collapsed the gap. Fix:

1. Train an **evidence-blind baseline** (hypothesis-only RoBERTa-large, no image) on the current training set.
2. Any training example the HO-baseline solves with ≥ 0.9 confidence across ≥ 3 seeds is **dropped** from v5 training.
3. Re-train v5 on the residual ("genuinely evidence-dependent") examples.
4. Validation criterion: HO-baseline val-accuracy on the residual drops to ≤ 65% (near majority-class); v5 val-accuracy stays ≥ 90%.

### 6.3 Image-masked degradation check

After every training epoch, run a sanity eval:
- Full v5 accuracy on val set
- v5 with image input replaced by zero-tensor (text-only path) accuracy

Target gap ≥ 15 pp by epoch 3. If < 10 pp at epoch 3, abort the run and increase `λ_c` or re-sample training distribution toward image-dependent claims.

### 6.4 Training configuration

| Hyperparameter | Value |
|---|---|
| GPU | H100 80GB × 1 (primary); × 4 for final 5-seed sweep |
| Batch size | 32 (effective 64 via grad accumulation × 2) |
| Optimizer | AdamW, β=(0.9, 0.999), weight decay 0.01 |
| LR | 1e-5 (encoders), 5e-5 (heads + fusion) |
| LR schedule | cosine, 500-step linear warmup |
| Epochs | 4 |
| Mixed precision | bf16 |
| Gradient clipping | 1.0 |
| Seeds | {17, 29, 41, 53, 67} for final run |
| Tracking | wandb project `claimguard-v5` |
| Determinism | `torch.use_deterministic_algorithms(True)`; `PYTHONHASHSEED=0` |

### 6.5 Pre-flight review

Before any Modal launch spawns GPU work, follow the CLAUDE.md Pre-flight Review Rule:

1. Spawn an Opus logic-review agent with:
   - File paths: `verifact/v5/model.py`, `verifact/v5/train.py`, `verifact/v5/losses.py`, `verifact/v5/data/*.py`.
   - Numbered worries: (1) λ-terms mis-weighted, (2) loss reduction over batch wrong, (3) grounding mask alignment to image patches off-by-one, (4) evidence-contrastive pairs leaking correct-evidence, (5) MC-dropout not active at eval, (6) data-loader label leakage across splits, (7) image normalization mismatch with BiomedCLIP expected stats, (8) tokenizer max-length truncating claim before evidence.
   - Expected cost: best $5 / expected $15 / worst $40 per training run.
   - Ask for a verdict: ready-to-launch / launch-with-fixes / do-not-launch.
2. Never launch against an unfixed DO-NOT-LAUNCH verdict.

---

## 7. Evaluation framework

### 7.1 Primary: image-grounded evaluation

For every (image, claim) in the test split:
1. Run the claim extractor + parser to get structured form.
2. Run the claim-to-annotation matcher (§4.2) to assign GT in {SUPPORTED, CONTRADICTED, NO_GT, NO_FINDING}.
3. Exclude NO_GT from the primary table; include in a separate "coverage and uncovered claims" table.
4. Compute verifier output (binary verdict + continuous score + uncertainty + grounding map).
5. Compute metrics over the resolved subset.

**Primary metrics (with 95% bootstrap CI, 10,000 replicates):**
- Accuracy, Macro-F1, Contradiction-Recall, AUROC.
- FDR at α ∈ {0.01, 0.05, 0.10, 0.20}, measured empirically over the green set.
- Power: fraction of truly SUPPORTED claims in the green set.
- Expected Calibration Error (ECE), Maximum Calibration Error (MCE).
- **Grounding IoU**: for claims with MS-CXR-style boxes available, IoU between model grounding head output and GT box.

### 7.2 Secondary: silver evaluation

For claims with `NO_GT` (no radiologist annotation), fall back to LLM-ensemble labels. Report separately with caveat about reliability (validated κ from §3.4).

### 7.3 Robustness axes

- **Cross-site:** train on CheXpert Plus, evaluate on {CheXpert-test, OpenI, PadChest, BRAX, public-MS-CXR subset, RSNA, SIIM, Object-CXR, ChestX-Det10}. Per-site tables with CIs.
- **Temporal:** CheXpert Plus pre-2016 train, post-2018 test where dates are available.
- **Per-VLM:** disaggregate by generator (MAIRA-2 / CheXagent / MedGemma / LLaVA-Rad / Llama-3.2-V) to check for generator-specific failure modes.
- **Image corruption:** Gaussian noise σ ∈ {0, 5, 15} (on 0–255 pixel range); JPEG compression quality 75, 50, 25; contrast jitter ±20%. Report accuracy at each level.
- **Paraphrase robustness:** rewrite claims via 3 independent LLM paraphrasers; check verdict stability.

### 7.4 Fairness / subgroup

Stratify on: sex, age quartile, race (CheXpert Plus has race), scanner manufacturer (PadChest), country (PadChest/BRAX/OpenI), and report length. Report parity gap (max − min accuracy across subgroups) with 95% CI. Flag disparities > 5 pp.

### 7.5 Ablation matrix

Minimum ablations (primary site; each with 3 seeds):

| Ablation | Varies |
|---|---|
| Image on/off | zero-image vs real-image input |
| Grounding loss on/off | λ_g ∈ {0, 0.5} |
| Consistency loss on/off | λ_c ∈ {0, 0.3} |
| Contrastive loss on/off | λ_e ∈ {0, 0.3} |
| Training mix | {synth only, real only, synth+real, synth+real+MSCXR} |
| Text backbone | {RoBERTa-large, BioClinicalBERT, PubMedBERT} |
| Image backbone | {BiomedCLIP, CLIP-ViT-L/14, RadDINO (w/ contamination caveat)} |
| Fusion depth | {0 (late fusion), 2, 4, 6 cross-modal layers} |
| Conformal variant | {inverted cfBH, weighted cfBH, doubly-robust cfBH, StratCP, forward cfBH} |
| Provenance gate on/off | — |

Full matrix = 2×2×2×2×4×3×3×4×5×2 = ~11,520 cells, but most are one-dim sweeps. Target ~50 cells reported; rest in supplementary.

### 7.6 Simulated-reader-study substitute

Since we lack recruited radiologists and are blocked from ReXVal, run a **synthetic reader study** on MS-CXR:
- For each of the 1,162 MS-CXR images: simulate a "radiologist" by treating MS-CXR ground truth as the reference.
- Compare ClaimGuard-flagged errors against MS-CXR-anchored errors.
- Metrics: sensitivity, specificity, decision-curve analysis.
- Caveat in paper: this is *not* a real reader study; it is an agreement analysis against radiologist-drawn annotations.

### 7.7 Provenance-gate scaled experiment

1,000 images from CheXpert Plus test × 3 VLMs × 4 temperatures (0.3, 0.7, 1.0, 1.2) × 2 seeds = 24,000 generations. For each:
- Same-model pairing: claim from run A + evidence from run A → provenance tier SAME_MODEL.
- Cross-model pairing: claim from VLM A + evidence from VLM B → INDEPENDENT.
- Cross-seed same-model: claim from seed 1 + evidence from seed 2 same model → SAME_MODEL.

Report: downgrade rate, verifier score divergence as a function of temperature, and the cross-model score divergence as the threat-free condition.

---

## 8. Conformal + provenance layer

### 8.1 cfBH variants to compare

- **Inverted cfBH** (v4 carry-over): calibrate p-values using contradicted class.
- **Weighted cfBH** (Tibshirani et al. 2019): density-ratio reweighting on calibration scores via a classifier-based estimator (gradient-boosted on calibration features: report length, patient age, scanner, pathology prevalence).
- **Doubly-robust cfBH** (Fannjiang et al. 2024 / Yang & Kuchibhotla 2024): robust under misspecified density ratio.
- **Prediction-Powered Inference** (Angelopoulos et al. 2023): baseline under shift.
- **StratCP** (Zitnik lab, medRxiv Feb 2026): stratified conformal baseline.
- **Forward cfBH** (standard Jin & Candes 2023): expected to fail due to score concentration; document.

### 8.2 Per-site coverage audit

For each external site, report:
- Actual FDR vs nominal α (bootstrap CI).
- Effective Sample Size (ESS) for weighted variants.
- Coverage violation rate.
- Fall back to un-weighted with per-site recalibration if ESS < 200.

### 8.3 Provenance gate

Unchanged from v4's design (see `inference/provenance.py`). Key metadata fields:
- `claim_generator_id`
- `evidence_generator_id`
- `evidence_source_type` ∈ {oracle_human, retrieved_human, same_model_generated, cross_model_generated, unknown}
- `evidence_trust_tier` ∈ {trusted, independent, same_model, unknown}

Gate rule: only `trusted` and `independent` tiers are eligible for conformal certification. Everything else downgrades to `supported_uncertified` regardless of verifier score.

### 8.4 Threat model (new in v5)

Explicit threat model section in the paper:
- Gate trusts `evidence_generator_id` as honestly reported metadata.
- Not a cryptographic attestation — a malicious generator could spoof metadata.
- Recommended production deployment: signed generation records or trusted-execution-environment attestation. Out of scope for the paper; mentioned as future work.

---

## 9. Infrastructure

### 9.1 Repo layout (new under `verifact/v5/`)

```
verifact/v5/
  __init__.py
  model.py                  # ImageGroundedVerifier
  losses.py                 # multi-objective loss
  train.py                  # train loop (for modal_train_v5.py to import)
  eval.py                   # eval loop (for modal_eval_v5.py to import)
  data/
    __init__.py
    chexpert_plus.py        # loader
    padchest.py             # loader + english translation
    openi.py                # loader
    brax.py                 # loader
    ms_cxr.py               # loader + box supervision
    rsna_pneumonia.py       # loader + bbox
    siim_pneumothorax.py    # loader + mask
    object_cxr.py           # loader + bbox
    chestx_det10.py         # loader + bbox
    anatomy_masks.py        # anatomy segmentation loader (our CheXmask-open)
    vlm_generators.py       # orchestrates the 5 VLMs over images
    claim_extractor.py      # LLM-based extractor
    claim_parser.py         # structured parser
    claim_matcher.py        # image-grounded GT matcher
    labeler_ensemble.py     # LLM-ensemble silver labeler
    pii_scrubber.py         # Presidio + regex
    groundbench.py          # assembles ClaimGuard-GroundBench
  conformal/
    __init__.py
    inverted_cfbh.py        # (wraps existing)
    weighted_cfbh.py        # new
    doubly_robust_cfbh.py   # new
    ppi.py                  # new
    utils.py                # shared
  provenance/
    __init__.py             # (wraps existing inference/provenance.py)
  eval/
    __init__.py
    image_grounded.py       # primary metric pipeline
    silver.py               # fallback metric pipeline
    calibration.py
    fairness.py
    robustness.py
    provenance_scaled.py    # dual-run experiment at scale
  configs/
    base.yaml
    v5_0_base.yaml
    v5_1_ground.yaml
    v5_2_real.yaml
    v5_3_contrast.yaml
    v5_4_final.yaml
    eval_site_chexpert.yaml
    eval_site_openi.yaml
    eval_site_padchest.yaml
    eval_site_brax.yaml
    eval_ablation.yaml
  tests/
    test_matcher.py
    test_model_shapes.py
    test_losses.py
    test_conformal.py
    test_dataloaders.py
    test_parser.py
    test_pii.py
  modal/
    train_v5.py             # Modal entrypoint
    eval_v5.py              # Modal entrypoint
    sweep_v5.py             # Modal entrypoint
    build_anatomy_masks.py  # Modal entrypoint (one-shot)
    build_groundbench.py    # Modal entrypoint (one-shot)
    run_vlms.py             # Modal entrypoint (generate claims across VLMs)
```

### 9.2 Environment pins

```
python = 3.10
torch = 2.4.0
torchvision = 0.19.0
transformers = 4.44.2          # pinned for BiomedCLIP compatibility
open_clip_torch = 2.26.1       # BiomedCLIP load path
sentencepiece = 0.2.0
accelerate = 0.33.0
datasets = 2.20.0
peft = 0.12.0                  # not used for v5; kept for DPO ablations
trl = 0.9.6                    # for DPO ablation (optional)
faiss-cpu = 1.8.0
rank-bm25 = 0.2.2
wandb = 0.17.7
hydra-core = 1.3.2
optuna = 3.6.1
scikit-learn = 1.5.1
numpy = 1.26.4
pandas = 2.2.2
pyarrow = 17.0.0
pillow = 10.4.0
nibabel = 5.2.1
pydicom = 2.4.4
presidio-analyzer = 2.2.356
presidio-anonymizer = 2.2.356
anthropic = 0.34.2
openai = 1.45.0
google-genai = 0.3.0
```

All HF model loads pinned to `revision=<sha>` (see `configs/base.yaml` for the SHA list).

### 9.3 Modal images

Two containers:

1. `claimguard-v5-train`: pytorch 2.4 / cu12.1 / transformers 4.44 / BiomedCLIP / all data loaders.
2. `claimguard-v5-vlm`: heavier, includes MAIRA-2 + CheXagent + MedGemma + LLaVA-Rad + Llama-3.2-V. Run only on VLM-generation passes.

### 9.4 Experiment tracking

- wandb project `claimguard-v5`, entity `laughney-lab`.
- Every Modal run dumps to wandb AND writes a local JSON manifest.
- Hydra config snapshot in every artifact.

### 9.5 Reproducibility

- Every released checkpoint paired with: training config, data hash, eval harness config, expected metric values.
- Dockerfile + `make reproduce` target clones the repo, pulls weights from HF, downloads a 100-claim demo subset from Zenodo, runs evaluation, asserts numbers match within tolerance.

---

## 10. Milestones and phased rollout

Explicit gates from the master execution plan, mapped to code milestones.

| Phase | Milestone | Code artifact | Gate |
|---|---|---|---|
| P0 | Governance + infra | `verifact/v5/` skeleton, CI green, configs | Repo + CI + wandb + compute reauthorized |
| P1a | Data loaders | `data/*.py`, unit tests | All 8 dataset loaders return `(image, annotations, metadata)` tuples |
| P1b | Anatomy masks open | `data/anatomy_masks.py`, `modal/build_anatomy_masks.py` | Coverage on CheXpert Plus ≥ 95% |
| P1c | Claim extractor + parser | `data/claim_extractor.py`, `data/claim_parser.py` | ≥ 0.95 claim precision on validation oracle |
| P1d | Claim matcher | `data/claim_matcher.py` | ≥ 95% agreement with MS-CXR native consensus |
| P2a | VLM generators | `data/vlm_generators.py`, `modal/run_vlms.py` | 5 VLMs run on CheXpert Plus test, outputs valid |
| P2b | Labeler ensemble | `data/labeler_ensemble.py` | κ ≥ 0.70 vs radiologist annotations on 500-claim probe |
| P2c | GroundBench assembly | `modal/build_groundbench.py` | ClaimGuard-GroundBench v0.9 in Parquet, DOI reserved |
| P3 | Model + losses | `model.py`, `losses.py` | Unit tests pass; shapes correct; forward on dummy batch |
| P4a | Training loop v5.0-base | `train.py`, `modal/train_v5.py` | 4-epoch training completes; v4 parity on val |
| P4b | v5.1-ground | grounding head enabled | Grounding IoU on MS-CXR val ≥ 0.30 |
| P4c | v5.2-real | real-VLM training mix on | HO gap ≥ 10 pp, image-masked degradation ≥ 15 pp |
| P4d | v5.3-contrast | contrastive evidence loss on | Both gaps improve ≥ 3 pp |
| P4e | v5.4-final | 5-seed final | Mean val-acc reported with ±CI |
| P5 | Conformal variants | `conformal/*.py`, tests | All 6 variants produce empirical FDR ≤ α on CheXpert val |
| P6a | Per-site eval | `eval/*.py`, per-site configs | All sites report, CIs computed |
| P6b | Fairness + robustness | `eval/fairness.py`, `eval/robustness.py` | Parity gaps + corruption accuracy tables |
| P6c | Provenance scaled | `eval/provenance_scaled.py` | 24k-generation dual-run result |
| P7 | Release | HF weights, Zenodo DOI, GitHub semver tag | Dockerized `make reproduce` passes |
| P8 | Manuscript | paper draft, figures, supplementary | Submitted |

### 10.1 Decision gates (abort / pivot)

- **G1d fails** (matcher < 95% MS-CXR agreement): manual ontology review; expand `finding_family` map; reduce TAU_IOU to 0.2; if still < 90%, manual radiologist consultation required.
- **G2b fails** (labeler κ < 0.70): drop bad categories; use only the categories that clear the bar; reduce training-mix real-hallucination share; narrow paper scope accordingly.
- **G4c fails** (HO gap < 10 pp after real training mix): adversarial filtering round 2; if still fails, reframe paper around honest negative result + image-grounded benchmark (benchmark alone is still NMI-worthy).
- **G4b fails** (grounding IoU < 0.30): increase `λ_g`; train grounding head longer before enabling other losses; check box alignment.

---

## 11. Success criteria

**Hard criteria (paper is NMI-submittable):**
1. Image-grounded accuracy ≥ 82% on the assembled GroundBench test split.
2. HO-baseline gap ≥ 10 pp (evidence-blind model clearly worse).
3. Image-masked degradation ≥ 15 pp (image clearly used).
4. FDR ≤ α on ≥ 5 of 6 external sites at α ∈ {0.05, 0.10, 0.20} with 95% CIs.
5. Grounding IoU ≥ 0.35 on MS-CXR test.
6. Provenance gate downgrade rate ≥ 0.95 on same-model pairs, ≤ 0.05 on cross-model pairs.
7. Labeler κ ≥ 0.70 on the categories kept in silver subset.
8. Parity gaps ≤ 5 pp (or clearly discussed if exceeded).
9. 5-seed variance < 2 pp on all headline numbers.
10. Independent reproducer reproduces headline numbers within ±1 pp.

**Soft criteria (stronger submission):**
- ClaimGuard-GroundBench adopted by at least one other group within 6 months of release.
- Image-grounded accuracy beats MAIRA-2 / CheXagent / MedGemma (tuned) on zero-shot verification by ≥ 10 pp.
- Weighted cfBH reduces FDR variance across sites by ≥ 30% vs inverted cfBH.

**Null-result fallback:** if the image-grounded verifier does not clear hard criterion 2 or 3 even after all three HO-gap-fix plans, the paper pivots to **"a public image-grounded benchmark + a rigorous negative-result methodology note"**. Still NMI-submittable as a benchmark paper with honest methodology contribution.

---

## 12. Risks and mitigations

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| R1 | BiomedCLIP too weak on CXR | High | Domain-adapter MAE pretrain on CheXpert Plus; fallback to CLIP + CheXpert-domain-adaptation |
| R2 | PadChest translation noise | Medium | Dual-LLM translation + filter to high-agreement; report PadChest separately |
| R3 | CheXmask-open coverage insufficient | Medium | Train our own lightweight U-Net on `torchxrayvision` anatomy head outputs |
| R4 | Claim parser hallucinates structure | Medium | Require 0.95 precision; lock version; fallback T5 extractor |
| R5 | Finding-ontology gaps | Medium | SNOMED+RadLex cross-reference; radiology consultant review if available |
| R6 | LLM labeler κ too low | Medium | Drop low-κ categories; narrow paper claims |
| R7 | Image-grounded GT coverage low | Medium | Report coverage; supplement with silver for uncovered |
| R8 | Modal budget overrun | High | Per-phase cap; auto-halt at 110% |
| R9 | MAIRA-2 / MedGemma access restrictions | Medium | Substitute with open-access VLMs; document |
| R10 | Provenance gate overclaim | Low | Explicit threat model; framed as engineering complement |
| R11 | Weighted conformal instability | Medium | ESS diagnostic; fallback un-weighted |
| R12 | Reviewer finds inverted cfBH redundant with Bates 2023 | Medium | Upfront citation + explicit contribution statement; position as application |
| R13 | Real VLM hallucinations not diverse enough | Medium | 5 generators × 4 temperatures × 2 seeds; per-generator disaggregation in evaluation |
| R14 | HF model-revision drift | Low | `revision=<sha>` pinning everywhere |
| R15 | IRB exempt determination delayed | Low | Public data only; no human subjects; determination should be exempt/NHSR |

---

## 13. Open research questions

These are not gating but feed into the paper's discussion section:

1. Is image-anchored ground truth strictly better than text-anchored (radiologist report) for claim-level verification, or are they complementary?
2. Does the grounding head generalize across anatomical regions it was not supervised on in MS-CXR?
3. How does image-grounded FDR control behave when the claim refers to a finding outside the annotation ontology entirely (true `NO_GT`)?
4. Can the image-grounded verifier's uncertainty head detect `NO_GT` claims autonomously, i.e., predict "annotation-coverage-missing" as its own failure mode?
5. Does the provenance gate's downgrade rate scale with sampling temperature the way a decision-theoretic model would predict?

---

## 14. Change log

- 2026-04-17 — initial draft, image-grounded pivot (supersedes v4 text-only architecture). Author: Aayan Alwani.

---

## Appendix A — Notation

- `s_j` = verifier's continuous support probability for claim j.
- `p_j` = conformal p-value for claim j under inverted/weighted/DR cfBH.
- `G_α` = green set at level α.
- `TAU_IOU` = IoU threshold for anatomical-region overlap in matcher (default 0.3).
- `finding_family` = coarse grouping of findings across annotation vocabularies.
- `NO_GT` = the matcher cannot establish ground truth due to annotation-coverage gap.

## Appendix B — Finding ontology (seed)

Unified 23-class finding family (finalized in `data/annotations/finding_ontology_v5.yaml`):

1. atelectasis
2. cardiomegaly
3. consolidation
4. edema
5. enlarged_cardiomediastinum
6. fracture
7. lung_lesion / nodule
8. lung_opacity (generic)
9. pleural_effusion
10. pleural_other (thickening, calcification)
11. pneumonia
12. pneumothorax
13. support_device / line / tube
14. foreign_object
15. hernia
16. infiltration
17. mass
18. emphysema
19. fibrosis
20. calcification
21. cavitation
22. bronchiectasis
23. no_finding (explicit negation)

Mapping tables from CheXpert 14 → unified 23, PadChest 174 → unified 23, RSNA/SIIM/Object-CXR/ChestX-Det10/MS-CXR → unified 23 live in `data/annotations/finding_ontology_v5.yaml` with explicit many-to-one mappings and "unmapped" fallback.

## Appendix C — Claim structure schema

```json
{
  "claim_id": "string",
  "raw_text": "string",
  "report_id": "string",
  "finding": "string (unified 23-class vocab)",
  "finding_family": "string",
  "location": "string (anatomical region id)",
  "laterality": "L | R | bilateral | none | unknown",
  "severity": "mild | moderate | severe | unknown",
  "temporality": "new | chronic | improving | worsening | stable | unknown",
  "comparison": "present | absent | unknown",
  "modifier_tags": ["size_small", "size_large", ...],
  "evidence_source_type": "oracle_human | retrieved_human | same_model_generated | cross_model_generated | unknown",
  "generator_id": "chexagent | maira-2 | medgemma | llava-rad | llama-3.2-v | oracle",
  "generator_temperature": "float | null",
  "generator_seed": "int | null"
}
```
