# ClaimGuard-CXR v5.0 — Evidence-Blindness Diagnosis & Mitigation

**Status:** design, pre-implementation
**Author:** Aayan Alwani (Laughney Lab, Weill Cornell Medicine)
**Date:** 2026-04-17
**Target venue:** NeurIPS 2027 main conference (May 2027 deadline)
**Runway:** 9 months (Apr 2026 → Jan 2027)
**Budget:** $900 USD hard cap (no slack for credentialed-data fees)
**Clinical collaborators:** none (explicit — no radiologist involvement at any phase)

---

## 0. Version disambiguation (READ FIRST)

This project has undergone multiple major pivots. There are now **six** architecture documents in the repo and vault. Only **this document** is authoritative for the current work. The others describe prior versions retained for historical record.

| Version | Doc | Status | Input modality | GT source | Thesis |
|---|---|---|---|---|---|
| v1 | `ARCHITECTURE.md` | deprecated | text only | LLM-as-judge | evidence-conditioned verifier + cfBH |
| v2 | `ARCHITECTURE_PATH_B.md` | deprecated | image + text | pixel annotations + LLM | image-grounded pivot (Path B) |
| v2 (rename) | `ARCHITECTURE_V2_IMAGE_GROUNDED.md` | deprecated | same as Path B | same | renamed duplicate of Path B |
| v3 | (inside `ARCHITECTURE.md` as sprint notes) | deprecated | text only | LLM-as-judge | multi-dataset OpenI transfer |
| v4 | (inside `ARCHITECTURE.md` as sprint notes) | deprecated | text only | LLM-as-judge | HO-gap discovery led to pivot |
| v5 "Image-Grounded" | `ARCHITECTURE_V5_IMAGE_GROUNDED.md` | deprecated | image + text | pixel annotations (credentialed OK) | benchmark + verifier, Nature MI target |
| **v5.0 "Evidence-Blindness"** | **this doc** | **authoritative** | image + text | pixel annotations (public only) | **diagnose + mitigate evidence-blindness across VLMs** |

### How v5.0 differs from v5 Image-Grounded (the immediately prior doc)

| Axis | v5 Image-Grounded | v5.0 Evidence-Blindness (this doc) |
|---|---|---|
| Primary scientific contribution | benchmark + verifier | diagnostic framework + mitigation, evaluated across 8 VLMs |
| Target venue | Nature Machine Intelligence | NeurIPS 2027 main |
| Budget | "$2,500 (pending approval)" | **$900 hard cap, no pending approval** |
| Radiologist involvement | "desirable but not blocking" (Weill Cornell co-author) | **none at any phase — explicit** |
| Credentialed data (MS-CXR, CheXmask, VinDr, BRAX) | "out of scope primary; add as bonus if credentialing lands" | **strictly forbidden — no credentialed data, no future extension in scope** |
| GT for grounding | pixel annotations from ≥7 sources incl. credentialed | pixel annotations from **5 non-credentialed sources only** |
| Multilingual coverage | EN + ES + PT (BRAX) | EN + ES only (PadChest only; BRAX excluded as credentialed) |
| Data scale | ~600k images | ~555k images (BRAX and VinDr removed) |
| Evidence-blindness treatment | bug to fix via adversarial HO filter | **first-class scientific problem, central to the paper's story** |
| Evaluated baselines | unspecified | **8 VLMs evaluated on the diagnostic** (GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Pro, CheXagent, LLaVA-Med-v1.5, MedVLM, BiomedCLIP zero-shot, ours) |
| Provenance gate | present | present (same as v5 Image-Grounded) |

### How v5.0 differs from v4

| Axis | v4 (text-only) | v5.0 (this doc) |
|---|---|---|
| Input | claim + retrieved text evidence | claim + retrieved text evidence + CXR image |
| GT | LLM-judge ensemble | pixel annotations (public) + existing radiologist report labels |
| HO-baseline gap at v4 | **-2.61 pp (evidence-blind beat full model — fatal)** | target ≥ 6 pp after HO-filter training |
| Image-masked degradation | N/A | target ≥ 10 pp |
| Evidence-blindness | undiagnosed artifact | **formalized, measured, mitigated — central contribution** |

### Vault disambiguation

- `~/Vault/wiki/projects/VeriFact.md` — project landing page; contains v1 headline numbers, should be updated to point here as the current authoritative spec.
- `~/Vault/wiki/synthesis/verifact-v5-pivot-2026-04-17.md` — captures the v4→v5 pivot reasoning (pre-9-month-reframe).
- `~/Vault/wiki/synthesis/claimguard-v5-0-evidence-blindness-2026-04-17.md` — **this plan.** New synthesis page to be created alongside this doc.
- `~/Vault/wiki/sources/verifact-architecture.md` — snapshot of earlier architecture docs. Retained as history.

---

## 1. Executive summary

ClaimGuard-CXR v5.0 reframes radiology claim verification as a problem of **evidence-blindness** — the failure mode in which multimodal classifiers produce predictions that depend primarily on textual shortcuts in the claim and retrieved evidence rather than on the image content. v4 accidentally demonstrated this failure mode on its own held-out set (a hypothesis-only baseline with the image zeroed out outperformed the full model by 2.61 pp). v5.0 treats this observation not as a bug to patch but as a scientific finding about the field.

**The paper's three sub-claims:**

1. **Diagnosis.** We formalize evidence-blindness via three counterfactual metrics (image-masking gap, evidence-shuffling gap, image-perturbation gap) that can be applied to any multimodal verifier — including closed-weight ones accessed via API.
2. **Prevalence.** We show evidence-blindness is present in eight existing multimodal systems (three API-based generalists, three medical VLMs, one zero-shot retriever, and our own baseline trained without the mitigation). This establishes it as a systemic property of the current state of the art, not an artifact of any single architecture.
3. **Mitigation.** We introduce an **adversarial hypothesis-only (HO) filtering** procedure that downweights training examples predictable from text alone, and show it measurably reduces all three diagnostic metrics without sacrificing accuracy.

**The benchmark (ClaimGuard-GroundBench)** is assembled from eight non-credentialed public datasets (~555k chest X-ray images, 8 sites, 2 languages) and released publicly with code, weights, and an evaluation harness. It is *infrastructure* for the paper, not the headline contribution.

**What v5.0 is NOT:**

- Not a clinical safety claim. Does not argue for deployment.
- Not an attempt to replace radiologist review.
- Not a credentialed-data study. No MIMIC-CXR, CheXmask, MS-CXR, VinDr, BRAX, or ReXVal appear anywhere in this work. A future extension may incorporate them, but any such extension is **explicitly out of scope for this paper and must not influence any design decision documented here.**

---

## 2. Scope, constraints, assumptions

### 2.1 Hard constraints

| Constraint | Value |
|---|---|
| Total budget | $900 USD, no overflow |
| Calendar | 9 months (2026-04-17 → 2027-01-17) |
| GPU class | H100 only (per user global CLAUDE.md policy) |
| Credentialed datasets | **forbidden** (MIMIC-CXR, CheXmask, MS-CXR, VinDr-CXR, BRAX, RadGraph-XL, ReXVal, ReXErr) |
| Clinician collaborators | none at any phase |
| LLM vendors | Anthropic (Claude family), OpenAI (GPT-4o), Google (Gemini) for baseline eval only; Anthropic for claim extraction |

### 2.2 Soft constraints

- Reproducibility: every artifact ships with a pinned Modal image, HF model revision IDs, Zenodo DOI for benchmark data, GitHub release with semver.
- Privacy: public datasets only contain pre-de-identified images and reports; we additionally run Presidio + CXR-specific regex PII scrubber on all report text before training.
- Compute hygiene: every Modal launch is preceded by a pre-flight Opus logic-review agent per user CLAUDE.md rules.
- Doc sync: every code change lands in the same commit as the relevant architecture/proposal update.

### 2.3 Assumptions

- **BiomedCLIP ViT-B/16** (microsoft/BiomedCLIP-PubMedBERT\_256-vit\_base\_patch16\_224) is strong enough on CXR when light-adapted on CheXpert Plus. Fallback: domain-adapter fine-tune; further fallback: CheXNet DenseNet-121.
- **RoBERTa-large** remains the text backbone for continuity with v4 and because its NLI behavior is well-characterized.
- Public pixel-annotated datasets (RSNA, SIIM, Object-CXR, ChestX-Det10) ground approximately 40-60% of test claims drawn from the text-bearing datasets. Unmatched claims are labeled `NO_GT` and excluded from headline metrics; a coverage statistic is reported separately.
- CheXpert Plus access via Stanford AIMI registration does not require DUA signing and is not considered "credentialed" in the PhysioNet sense. If this assumption breaks, fall back to OpenI + NIH ChestX-ray14 + annotated-only datasets (~170k images; story weakens but remains feasible).

---

## 3. Scientific thesis: evidence-blindness

### 3.1 Definition

Let `f: (x_img, x_claim, x_evidence) → y ∈ {SUPPORTED, CONTRADICTED}` be a multimodal verifier, where `x_img` is an image, `x_claim` is a natural-language claim about the image, and `x_evidence` is a retrieved text passage. We say `f` is **evidence-blind** if its output distribution is approximately invariant under interventions that preserve only the textual inputs:

- **Image-masking gap (IMG):** `IMG(f) = acc(f, D) - acc(f, D^{img=∅})` where `D^{img=∅}` zeros every image. If `IMG(f) < 5 pp`, `f` is image-blind.
- **Evidence-shuffling gap (ESG):** `ESG(f) = acc(f, D) - acc(f, D^{evid=π})` where `π` is a random permutation of evidence across claims within a batch. If `ESG(f) < 5 pp`, `f` is evidence-blind.
- **Image-perturbation gap (IPG):** `IPG(f) = acc(f, D) - acc(f, D^{img=τ})` where `τ` is a horizontal flip (tests laterality use). If `IPG(f) < 3 pp` on laterality-sensitive claims, `f` does not use spatial structure.

The 5 pp / 5 pp / 3 pp thresholds are calibrated empirically using control experiments on adversarial datasets (Section 7.4).

### 3.2 Why this matters

- **Clinical safety:** a verifier that passes "the aortic knob is enlarged" based on text alone, without looking at the image, is not a safety tool — it's a text classifier masquerading as one.
- **Scientific:** the evidence-blind baseline outperforming the full model in v4 (-2.61 pp gap) is a concrete instance of a broader pattern documented in NLI (Gururangan 2018, Poliak 2018) but not yet systematically studied in multimodal medical verification.
- **Actionable:** if evidence-blindness can be measured, it can be optimized against.

### 3.3 Why this is NeurIPS-main-shaped

- **Sharp single contribution** (diagnose a failure + fix it).
- **Broad implications**: the diagnostic framework is domain-agnostic in principle; the paper argues it should be a default check for any multimodal verifier.
- **Partial-input baselines exist in NLI** (Gururangan 2018, Poliak 2018), but their systematic application to medical multimodal verification — and coupling with a training-time mitigation — is new.
- **Honest positioning**: this is an ML methodology contribution, not a clinical deployment claim.

---

## 4. Data: ClaimGuard-GroundBench v5.0

### 4.1 Dataset roster (public only)

| Dataset | Images | Country | Language | Text | Spatial GT | License | Role |
|---|---|---|---|---|---|---|---|
| CheXpert Plus | 223,000 | US | EN | reports + labels | none | Stanford AIMI (registration) | Primary training — reports |
| PadChest | 160,000 | Spain | ES→EN | reports + labels | none | BIMCV (registration) | Multilingual eval |
| NIH ChestX-ray14 | 112,000 | US | EN | weak labels only | none | public domain | Scale + pre-training |
| RSNA Pneumonia | 30,000 | US | — | none | bboxes (pneumonia) | Kaggle CC BY-NC-SA 4.0 | Synthesized claims + grounding |
| SIIM-ACR Pneumothorax | 12,000 | US | — | none | masks (pneumothorax) | Kaggle | Synthesized claims + grounding |
| Object-CXR | 10,000 | China | — | none | bboxes (foreign objects) | research-use | Synthesized claims + grounding |
| OpenI | 3,996 | US | EN | reports + labels | some bboxes | CC BY 4.0 | Cross-site eval |
| ChestX-Det10 | 3,543 | US | — | none | pixel masks (10 classes) | CC BY-NC 4.0 | Synthesized claims + grounding |
| **Total** | **~554,500** | **3 countries** | **EN + ES** | | | | |

Excluded (credentialed, out of scope):

| Dataset | Why excluded |
|---|---|
| MIMIC-CXR | PhysioNet credentialed |
| CheXmask | PhysioNet credentialed |
| MS-CXR | PhysioNet credentialed |
| VinDr-CXR | PhysioNet credentialed |
| BRAX | PhysioNet credentialed |
| RadGraph-XL | Derived from MIMIC |
| ReXVal, ReXErr | Derived from MIMIC |

### 4.2 Data pipeline

```
  [8 public datasets]
        │
        ▼
  [Per-site loaders: v5/data/{chexpert_plus,padchest,nih,rsna,siim,objcxr,openi,chestx_det10}.py]
        │
        ▼
  [Per-site record dataclass: image_path, report_en, labels, boxes, masks, metadata]
        │
  ┌─────┴─────────────────────────┐
  ▼                               ▼
[Report-bearing sites]         [Annotation-only sites]
(CheXpert+, PadChest,           (RSNA, SIIM, Object-CXR,
 OpenI, NIH-weak-labels)         ChestX-Det10)
  │                               │
  ▼                               ▼
[LLM claim extractor             [Deterministic claim synthesizer:
 (Claude Haiku) → list of         box/mask label + anatomical region
 atomic claims]                   → template claim]
  │                               │
  └──────────────┬────────────────┘
                 ▼
         [Claim parser (rule-based primary, Claude fallback)
          → structured: (finding, location, laterality, severity, certainty, polarity)]
                 │
                 ▼
         [Anatomy mask computation: torchxrayvision per image,
          caches to /data/anatomy_masks_v5/]
                 │
                 ▼
         [Claim-to-annotation matcher:
          ontology lookup + IoU threshold 0.3 + laterality check
          → GT label in {SUPPORTED, CONTRADICTED, NO_GT}]
                 │
                 ▼
         [PII scrub: Presidio + CXR regex on all text]
                 │
                 ▼
         [Assembly: one JSONL row per (claim, image) pair
          with full provenance trace]
                 │
                 ▼
         [Train / Val / Cal / Test split
          70 / 10 / 10 / 10, patient-stratified, per-site balanced]
                 │
                 ▼
         [Aggregation: all sites → /data/groundbench_v5/all/{train,val,cal,test}.jsonl]
```

### 4.3 Claim synthesis from annotations (for detection-only datasets)

Detection-only datasets contribute training data by deterministically generating claims from their annotations. Template:

```
TEMPLATE_POSITIVE = "There is {finding} in the {anatomical_region}."
TEMPLATE_NEGATIVE = "There is no {finding} in the {anatomical_region}."
TEMPLATE_LATERALITY = "{finding} is present in the {laterality} {anatomical_region}."
```

Where:
- `{finding}` comes from the dataset's label taxonomy mapped to the ClaimGuard ontology (14 CheXpert findings + Support Devices + Foreign Object + Pleural Effusion + ...)
- `{anatomical_region}` comes from `torchxrayvision` anatomy mask intersected with the bounding box centroid
- `{laterality}` is derived from centroid x-coordinate relative to image midline

For each annotated finding, we emit one positive claim (GT = SUPPORTED). For balance, we sample one negative claim per image from the empty regions (GT = CONTRADICTED), using a finding absent from the current image but present elsewhere in the dataset.

This produces high-quality, low-noise training claims — at the cost of lexical simplicity (templated language). To prevent the model from memorizing template patterns, we mix report-derived claims (natural language, noisier) and synthesized claims (templated, cleaner) in a 60/40 ratio during training.

### 4.4 Claim-to-annotation matcher

A claim parses to a structured tuple `(finding, location, laterality, severity, certainty, polarity)`. Matching proceeds:

```
def match(claim_struct, annotations):
    # 1. Filter annotations to same finding type (via ontology)
    candidates = [a for a in annotations if ontology.match(a.finding, claim_struct.finding)]
    if not candidates:
        return "NO_GT"
    # 2. Filter by laterality if specified
    if claim_struct.laterality:
        candidates = [a for a in candidates if a.laterality == claim_struct.laterality]
    # 3. Spatial overlap if claim specifies location
    if claim_struct.location:
        loc_mask = anatomy_masks[claim_struct.location]
        candidates = [a for a in candidates if iou(a.mask, loc_mask) > 0.3]
    # 4. Final call
    if candidates:
        return "SUPPORTED" if claim_struct.polarity == "positive" else "CONTRADICTED"
    else:
        return "CONTRADICTED" if claim_struct.polarity == "positive" else "SUPPORTED"
```

**Validation protocol (no radiologist version):** author manually reviews 500 claim-annotation matches and publishes the agreement rate between the automated matcher and self-review. Any disagreement is documented. This is presented as *matcher reliability*, not clinical validation.

### 4.5 Splits

Patient-stratified, per-site balanced:
- **train** 70% — full supervision
- **val** 10% — early stopping, hyperparameter selection
- **cal** 10% — conformal calibration (held out from val to preserve exchangeability)
- **test** 10% — headline metrics, evidence-blindness diagnostics

Each site contributes proportionally. Leave-one-site-out experiments hold out one site's entire share for testing while training on the remaining seven.

**Known v4 failure to avoid**: never use the calibration set as the validation set. These must be disjoint. The code path `V5TrainConfig.val_jsonl` must read from `val.jsonl`, not `cal.jsonl` (Bug 2 in the pre-flight — blocker fix).

---

## 5. Model architecture

### 5.1 High-level diagram

```
  x_img ∈ ℝ^(3×224×224)                  x_claim, x_evidence (tokenized via RoBERTa)
         │                                            │
         ▼                                            ▼
  [BiomedCLIP ViT-B/16]                       [RoBERTa-large]
  86M params, frozen top 8 blocks            355M params, frozen bottom 16 layers
         │                                            │
         ▼                                            ▼
  patch tokens: 196 × 768                   token embeddings: L × 1024
         │                                            │
         ▼                                            ▼
  [Domain adapter: 2-layer MLP                [Text projection:
   768→768→768, residual]                      Linear(1024→768)]
         │                                            │
         └──────────────────┬─────────────────────────┘
                            │
                            ▼
              [Prepend [VERDICT] learnable token (1 × 768)]
                            │
                            ▼
              [Concat: (1 + 196 + L) × 768]
                            │
                            ▼
              [4-layer bidirectional cross-modal transformer
               d_model=768, heads=12, d_ff=3072, dropout=0.1, mc_dropout_p=0.1]
                            │
              ┌─────────────┼─────────────────────┐
              ▼             ▼                     ▼
       [VERDICT head]  [Support score head]  [Grounding head]
       Linear(768→2)   Linear(768→1)+sigmoid  Linear(768→1) over patch tokens
              │             │                     │
              ▼             ▼                     ▼
         y ∈ {S, C}    s ∈ [0,1]           g ∈ ℝ^(14×14) (sigmoid)
```

Parameter budget (trainable):
- Domain adapter: ~1.2M
- Text projection: ~0.8M
- VERDICT token: 0.0008M
- 4-layer fusion transformer: ~14.2M
- Three heads: ~2.3M
- BiomedCLIP top 4 ViT blocks (unfrozen): ~28.3M
- RoBERTa top 8 layers (unfrozen): ~75.5M
- **Total trainable: ~122M** of **~480M** total (fits H100 at batch 32 with gradient accumulation 2)

### 5.2 Image encoder

- Model: `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`
- Pinned revision: record exact SHA on first load
- Input: 224×224 RGB (CXR loaded as grayscale, replicated to 3 channels)
- Output: 196 patch tokens + 1 CLS, d=768
- Freeze policy: freeze bottom 8 of 12 ViT blocks. Top 4 blocks trainable.
- **Known v5.0 bug risk**: `_freeze_image()` in `v5/model.py:152` tries three attribute paths (`encoder.layer`, `vision_model.encoder.layers`, `visual.trunk.blocks`). BiomedCLIP via open\_clip\_torch may match none, falling back to freezing the entire encoder. Must verify actual attribute path during M1 week 1 smoke test (Bug 7 in pre-flight list).

### 5.3 Text encoder

- Model: `roberta-large`
- Pinned revision: record exact SHA on first load
- Tokenizer: RoBERTa BPE. Max length 256 tokens.
- Input format: `tokenizer(claim, evidence, max_length=256, truncation='only_second', padding='max_length', return_tensors='pt')` — uses the proper `text_pair` API.
- **Known v5.0 bug risk**: current code at `v5/train.py:119` uses `f"{claim}</s></s>{evidence}"` as a raw string. RoBERTa's `</s>` tokens must be injected by the tokenizer, not as text — otherwise they tokenize as three separate tokens (`<`, `/s`, `>`) and the pair boundary is lost. Must fix to use `tokenizer(claim, evidence, ...)` (Bug 8 in pre-flight list).
- Output: sequence hidden states, d=1024, projected to d=768.
- Freeze policy: freeze bottom 16 of 24 layers. Top 8 layers trainable.

### 5.4 Fusion transformer

- 4 bidirectional transformer encoder blocks.
- d\_model=768, heads=12, d\_ff=3072, pre-norm, GELU activation.
- Dropout=0.1 (standard). MC-dropout p=0.1 (separate dropout modules that stay active during inference for uncertainty estimation).
- Input: concat of `[VERDICT, img_patch_tokens (196), text_tokens (padded to 256)]` → sequence length 453.
- Attention mask: correctly marks padded text positions. `nn.MultiheadAttention` expects `True = pad`; the v5 code builds this correctly (verified) but the inversion logic has historically caused bugs.

### 5.5 Heads

- **Verdict head**: `Linear(768→2)` on the `[VERDICT]` token output. Softmax → `p(SUPPORTED)`, `p(CONTRADICTED)`.
- **Support score head**: `Linear(768→1) + sigmoid` on the `[VERDICT]` token output. Interpreted as calibrated support strength ∈ [0,1], used downstream by the conformal procedure.
- **Grounding head**: `Linear(768→1)` applied independently to each of the 196 patch tokens, reshaped to 14×14, sigmoid. Target is a binary mask projected from GT bounding boxes onto the patch grid (union of boxes if multiple findings align with claim).

### 5.6 MC-dropout for uncertainty

At inference, `predict_with_uncertainty(x, n_samples=20)` runs 20 forward passes with dropout active and returns:
- `mean_p` — averaged softmax probabilities
- `predictive_entropy` — `H(mean_p)`
- `epistemic_uncertainty` — `H(mean_p) - E[H(p_sample)]` (BALD-style decomposition)

MC-dropout is applied throughout the fusion transformer (4 blocks × 2 dropout layers each = 8 sample points) and on the verdict head input.

**Known v5.0 bug risk**: `losses.uncertainty_regularizer()` at `v5/losses.py:97` applies `nn.Dropout` to pre-computed logit tensors, which is NOT MC-dropout (it's just noise on fixed outputs). Must either (a) disable this loss for v5.0-v5.3 and only activate in v5.4 via true MC forward passes, or (b) replace with temperature-scaling calibration loss (Bug 5 in pre-flight list).

---

## 6. Training

### 6.1 Five-objective loss

For each batch `B`:

```
L_total = w_cls · L_cls
        + w_grd · L_grd  (only on rows with spatial GT)
        + w_cns · L_cns  (consistency: full vs image-masked)
        + w_ctr · L_ctr  (contrastive: matched vs shuffled evidence)
        + w_unc · L_unc  (MC-ECE, activated in v5.4 only)
```

**1. Classification loss** (standard):
```
L_cls = CE(verdict_logits, y) where y ∈ {0=SUPPORTED, 1=CONTRADICTED}
```

**2. Grounding loss** (masked BCE):
```
L_grd = BCE(grounding_logits, target_mask) · grounding_mask / sum(grounding_mask)
```
where `target_mask ∈ {0,1}^(14×14)` is derived from bounding boxes projected onto the 14×14 patch grid, and `grounding_mask ∈ {0,1}` indicates whether this row has pixel-level GT available.

**3. Consistency loss** (margin between full and image-masked prediction):
```
L_cns = ReLU(margin - (p_full[y] - p_masked[y])).mean()
```
where `p_full` is the softmax prediction on `(x_img, x_claim, x_evid)` and `p_masked` is the prediction on `(0, x_claim, x_evid)` (zero image). `margin = 0.2`. This explicitly penalizes the model when masking the image doesn't hurt accuracy — directly optimizing for the image-masking gap (IMG) to be large.

**4. Contrastive evidence loss** (margin between correct and shuffled evidence):
```
# Only for SUPPORTED anchors (filtering fixes v5.0 Bug 1 in pre-flight list)
supported_mask = (y == 0)
s_pos = support_score[supported_mask]
perm = random_permutation(supported_mask.sum())
# Identity-rejection: ensure perm[i] ≠ i
perm = fix_identity(perm)
s_neg = run_forward(x_img[supported_mask], x_claim[supported_mask], x_evid[supported_mask][perm]).support_score
L_ctr = ReLU(margin - (s_pos - s_neg)).mean()
```
Margin = 0.2. This explicitly optimizes for the evidence-shuffling gap (ESG) to be large.

**5. Uncertainty loss** (MC-ECE; activated v5.4 only):
```
# Requires n_samples > 1 MC forward passes
mean_p, _, epistemic = predict_with_uncertainty(batch, n_samples=10)
L_unc = |epistemic - |mean_p[y] - 1||  # epistemic should match error
```
This is a calibration-adjacent regularizer. If this formulation underperforms in pilot, replace with temperature-scaling loss on a held-out calibration split.

### 6.2 Loss weights per config

| Config | w_cls | w_grd | w_cns | w_ctr | w_unc | HO filter | Epochs |
|---|---|---|---|---|---|---|---|
| v5.0-base | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | off | 2 |
| v5.1-ground | 1.0 | 0.5 | 0.0 | 0.0 | 0.0 | off | 3 |
| v5.2-real | 1.0 | 0.5 | 0.3 | 0.0 | 0.0 | **on** | 4 |
| v5.3-contrast | 1.0 | 0.5 | 0.3 | 0.3 | 0.0 | on | 4 |
| v5.4-final | 1.0 | 0.5 | 0.3 | 0.3 | 0.1 | on | 4 |

Final config uses 3 seeds (budget doesn't support 5 seeds under $900 cap).

### 6.3 Adversarial hypothesis-only (HO) filtering — the mitigation

**The core mitigation of the paper. Must actually work, not be a no-op.**

```
# Step 1: Train HO model
ho_model = RobertaForSequenceClassification.from_pretrained('roberta-large')
ho_model.train_on(train_set, input='claim [SEP] evidence', target='y', epochs=1)

# Step 2: Score all training examples
ho_scores = ho_model.predict_proba(train_set)[..., true_label_idx]

# Step 3: Compute per-example training weight
threshold = 0.7
# Shortcut-solvable examples are downweighted
weights = torch.where(ho_scores > threshold, torch.tensor(0.2), torch.tensor(1.0))

# Step 4: Train full model on reweighted distribution
for batch, batch_weights in zip(train_loader, weights_loader):
    logits = model(batch)
    loss = (CE(logits, batch.y, reduction='none') * batch_weights).mean()
```

Threshold 0.7 is the default; we ablate over {0.5, 0.6, 0.7, 0.8, 0.9}.

**Known v5.0 bug**: `train.adversarial_ho_filter()` at `v5/train.py:182` currently raises `NotImplementedError`, and configs v5.2/v5.3/v5.4 set `adversarial_ho_filter: true` but `train_v5()` never actually calls this function. This makes the central contribution silently inactive in the current code (Bug 9 in pre-flight list). Fixing this is the highest-priority code fix.

### 6.4 Optimizer and schedule

- AdamW, betas=(0.9, 0.999), weight\_decay=0.01
- Two parameter groups:
  - Encoder params (unfrozen ViT blocks + RoBERTa layers): lr=1e-5
  - Head + fusion + adapter params: lr=5e-5
- Linear warmup for 500 steps → cosine decay to 10% of peak
- Batch size 32, gradient accumulation 2 (effective batch 64)
- Mixed precision: bf16
- Max 4 epochs per run

### 6.5 Data augmentation

Image:
- Random resized crop (scale 0.85–1.0)
- Random horizontal flip — **disabled** (breaks laterality — important for our diagnostics)
- Random rotation ±5°
- ColorJitter (brightness 0.1, contrast 0.1)

Text:
- No augmentation (preserves claim semantics)

---

## 7. Evidence-blindness diagnostic framework (central contribution)

### 7.1 Three counterfactual metrics

Formal definitions in Section 3.1. Implementation:

```
def diagnose_evidence_blindness(model, test_loader, laterality_subset):
    # IMG: zero the image
    acc_full = evaluate(model, test_loader, transform=identity)
    acc_img_zero = evaluate(model, test_loader, transform=lambda b: replace_img(b, zeros))
    img_gap = acc_full - acc_img_zero

    # ESG: shuffle evidence across batch
    acc_evid_shuf = evaluate(model, test_loader, transform=lambda b: shuffle_evidence(b))
    esg_gap = acc_full - acc_evid_shuf

    # IPG: horizontal flip on laterality-sensitive subset
    acc_lat_full = evaluate(model, laterality_subset, transform=identity)
    acc_lat_flip = evaluate(model, laterality_subset, transform=hflip)
    ipg_gap = acc_lat_full - acc_lat_flip

    return {"IMG": img_gap, "ESG": esg_gap, "IPG": ipg_gap,
            "evidence_blind": img_gap < 0.05 or esg_gap < 0.05}
```

### 7.2 Protocol for closed-weight API models

API models (GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Pro) are evaluated via:
1. Format each test example as a chat message: `{image: base64, claim: "...", evidence: "..."}`
2. System prompt: "You are a radiology claim verifier. Given an image, a claim, and supporting evidence, output SUPPORTED or CONTRADICTED. Do not explain."
3. For each intervention (full / image-zero / evidence-shuffle / image-flip), re-run with modified inputs.
4. Image-zero for API models: replace image with a 224×224 solid gray image.

Cost estimate: 8 models × 4 conditions × 1000 test claims × ~$0.005/call = ~$160. Budget allocates $100 for this (we evaluate 3 API models at 1000 claims × 4 conditions = 12k calls ≈ $60).

### 7.3 Protocol for local medical VLMs

CheXagent, LLaVA-Med-v1.5, MedVLM-v0.1, BiomedCLIP zero-shot: all run on Modal H100, one pass per condition. Approximately 1 GPU-hour per model per condition.

### 7.4 Threshold calibration

To justify the 5 pp / 5 pp / 3 pp thresholds, we construct two control datasets:

- **Fully image-blind control**: a RoBERTa-only model with no image input. Trained on same claims. Its IMG must be ≤ 1 pp (sanity check).
- **Fully image-using control**: a ViT-only model with no text input, trained to classify `(image, claim)` pairs by whether the image contains the claimed finding (derived from annotations). Its IMG must be ≥ 30 pp.

The 5 pp threshold is chosen to lie well above the fully-blind control's noise floor. Sensitivity analysis over thresholds ∈ {3, 5, 7, 10} is reported.

---

## 8. Evaluation protocol

### 8.1 Baselines (eight methods)

| Method | Access | Compute | Cost |
|---|---|---|---|
| GPT-4o | API (OpenAI) | — | $25 |
| Claude 3.5 Sonnet | API (Anthropic) | — | $20 |
| Gemini 1.5 Pro | API (Google) | — | $15 |
| CheXagent-8b | Local (HF) | Modal H100, ~2 hr | $10 |
| LLaVA-Med-v1.5-7b | Local (HF) | Modal H100, ~1.5 hr | $8 |
| MedVLM-v0.1 | Local (HF) | Modal H100, ~1 hr | $5 |
| BiomedCLIP zero-shot | Local (HF) | Modal H100, ~0.5 hr | $3 |
| ClaimGuard-CXR v5.4 (ours) | Trained | Modal H100 (in train budget) | — |

Each baseline is evaluated in two modes:
- **Standard accuracy** on test set.
- **Evidence-blindness diagnostic** (IMG, ESG, IPG) with the four counterfactual conditions.

### 8.2 Primary metrics

Per method:
- Accuracy, F1, AUROC on binary verdict
- IMG, ESG, IPG (with confidence intervals via bootstrap, 1000 resamples)
- Classification of evidence-blind (yes/no)
- Grounding IoU @ 0.3 threshold (only for our method and CheXagent, which support localization)
- ECE (expected calibration error) on verdict probabilities

Per method × site:
- Leave-one-site-out accuracy
- Evidence-blindness stability across sites

### 8.3 Conformal FDR framework

Three variants compared:
- **Inverted cfBH** (v4 carry-over, calibration on CONTRADICTED claims)
- **Weighted cfBH** (Tibshirani 2019) with per-site weighting
- **Doubly-robust cfBH** (Fannjiang 2024) with density-ratio reweighting

Target FDR levels: α ∈ {0.05, 0.10, 0.20}.

Reported:
- Achieved FDR vs target
- Power (recall on SUPPORTED claims)
- Group-wise FDR (per-finding, per-site)

### 8.4 Provenance gate

When a claim was generated by VLM X and the evidence for verification is also from VLM X, the verdict is downgraded:
- `SUPPORTED` → `supported_uncertified`
- `CONTRADICTED` → unchanged

This prevents a VLM's hallucination from being self-validated. The gate applies at inference only; training uses full labels.

### 8.5 Ablations

| Ablation | Variants | Budget |
|---|---|---|
| Loss components | drop-one, keep-one for each of 5 losses | $40 |
| HO-filter strength | threshold ∈ {0.5, 0.6, 0.7, 0.8, 0.9} | $30 |
| Scale curve | 7.5k / 50k / 500k training images | $30 |
| Leave-one-site-out | 8 holdouts × 1 seed | $20 |
| Synthetic vs report mix | 0/100, 30/70, 40/60, 50/50, 70/30, 100/0 | covered by training iteration budget |

Total ablation budget: $120.

### 8.6 Handling the no-radiologist constraint

Explicit in-paper framing:
- Ground truth traces to radiologist annotations created by the source dataset curators.
- Claim-to-annotation matching is automated; matcher reliability is self-reviewed (500 claims) with protocol published.
- The paper's scope is methodological (measure and reduce evidence-blindness), not clinical (help radiologists in deployment).
- Future work section acknowledges clinical validation as a necessary next step that is out of scope for this contribution.

No paid or recruited radiologist review at any phase. Budget never allocates funds for clinician hours.

---

## 9. Compute and budget

### 9.1 Budget allocation ($900 hard cap)

| Line item | Budget |
|---|---|
| LLM claim extraction (CheXpert Plus, PadChest, OpenI reports via Claude Haiku) | $120 |
| Translation (PadChest Spanish → English via Claude Haiku) | $30 |
| Main training ladder (v5.0 → v5.4, 3 seeds on final) on Modal H100 | $350 |
| Iteration / debug training | $80 |
| Ablations (loss components, HO strength, scale curve, leave-one-site-out) | $120 |
| API baseline evals (GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Pro) | $60 |
| Local baseline evals (CheXagent, LLaVA-Med, MedVLM, BiomedCLIP) on Modal H100 | $40 |
| Evidence-blindness diagnostic runs across 8 models × 4 conditions | $50 |
| Buffer (reruns, bug fixes) | $50 |
| **Total** | **$900** |

### 9.2 GPU hours

At Modal H100 ~$4/hr:
- Main training ladder: ~80 GPU-hours
- Iteration: ~20 GPU-hours
- Ablations: ~30 GPU-hours
- Local baselines: ~10 GPU-hours
- Total: ~140 GPU-hours ≈ $560

CPU-only work (data assembly, aggregation, evaluation aggregation): Modal CPU containers at $0.15/hr, negligible.

### 9.3 Stop-on-exhaust policy

When buffer depletes, we stop adding runs. The paper ships with whatever is complete. Training iteration budget overruns come out of buffer first, then ablation budget, never from main training budget.

---

## 10. Release artifacts

All artifacts public on paper acceptance:
- **Code:** GitHub `alwaniaayan6-png/ClaimGuard-CXR`, MIT license, semver-tagged release
- **Benchmark data:** Zenodo DOI, CC BY-NC 4.0 (inherits from source datasets' most restrictive license)
- **Model weights:** HuggingFace Hub, same license as code
- **Evaluation harness:** included in GitHub release, reproduces Table 1 end-to-end from a single `modal run` invocation
- **Docker image:** pinned Python 3.10 + CUDA 12.1, Modal-compatible
- **Documentation:** README + datasheet (required for NeurIPS D&B; we produce it even for main track submission) + reproducibility checklist

---

## 11. Known bugs (pre-flight list)

All 11 pre-flight bugs fixed 2026-04-17. Status is reproduced here for the record. Pre-flight review (second pass) still owed before any Modal launch.

| # | File | Issue | Status | Fix |
|---|---|---|---|---|
| 1 | `v5/losses.py` + `v5/train.py` | Contrastive loss applied to CONTRADICTED anchors (wrong direction) | ✅ Fixed | `train.py` now filters to `y==0` subset with identity-rejecting permutation before calling `contrastive_evidence_loss`. |
| 2 | `v5/modal/train_v5.py` + `v5/data/_common.py` | `val_jsonl = cal_jsonl` voids conformal guarantee | ✅ Fixed | `ensure_split` now produces 4-way split (train/val/cal/test). `train_v5.py` reads `groundbench_v5_val.jsonl`. |
| 3 | `v5/modal/build_groundbench.py` + `v5/data/groundbench.py` | No aggregation function; `/data/groundbench_v5/all/` never created | ✅ Fixed | Added `aggregate_groundbench()` in `groundbench.py`. New Modal function `aggregate_groundbench_fn` + `aggregate_entrypoint` in `build_groundbench.py`. |
| 4 | `v5/modal/train_v5.py` | 12h timeout risks budget overrun | ✅ Fixed | Reduced to 4h (`60*60*4`). |
| 5 | `v5/losses.py` + `v5/train.py` | MC-ECE applies Dropout to pre-computed logits (not MC-dropout) | ✅ Fixed | `uncertainty_regularizer` now accepts `(S,B,C)` MC sample tensor. `train.py` runs `uncertainty_n_samples` forward passes when `uncert>0` with probability `uncertainty_prob`. |
| 6 | `v5/modal/build_groundbench.py` + `v5/data/claim_synthesizer.py` (new) | ChestX-Det10 + RSNA + SIIM + Object-CXR produce zero training rows | ✅ Fixed | New module `claim_synthesizer.py` deterministically synthesizes claims from bounding-box/mask labels. Wired into `build_groundbench.py` elif branch for detection-only sites. Added RSNA/SIIM/Object-CXR to site dispatcher. |
| 7 | `v5/model.py` | BiomedCLIP block-freeze likely falls through (wrong attribute paths) | ✅ Fixed | `_freeze_image` now tries 9 candidate paths and RAISES `RuntimeError` on no-match (instead of silently freezing the entire encoder). |
| 8 | `v5/train.py` | Text separator `</s></s>` as raw string breaks tokenization | ✅ Fixed | `_encode_text` uses `tokenizer(claim, evidence, truncation='only_second')` so `</s></s>` is injected as real special tokens. |
| 9 | `v5/train.py` + `v5/ho_filter.py` (new) | Adversarial HO filter raises `NotImplementedError`; never called | ✅ Fixed | Removed placeholder. New module `ho_filter.py` trains a RoBERTa-large baseline on `(claim,evidence)->label`, scores all training rows, and writes per-row weights JSONL. `train_v5()` auto-invokes it when `cfg.adversarial_ho_filter=True`. Per-example weights applied to classification loss. |
| 10 | `v5/configs/base.yaml` | `hard_cap_usd: 2500` (should be 900) | ✅ Fixed | `hard_cap_usd: 900`, `phase_cap_usd: 200`. |
| 11 | `v5/data/groundbench.py` | `assemble_row` uses wrong `image_id` fallback when anatomy is None | ✅ Fixed | Simplified to `anatomy.image_id if anatomy is not None else structured.report_id`. |

Second pre-flight review owed before Modal launch to confirm the fixes are integration-correct (tests only verify unit-level behaviour).

---

## 12. Timeline (9 months)

| Month | Focus | Deliverables |
|---|---|---|
| M1 (Apr 17 – May 17) | Foundation | All 11 bugs fixed; CheXpert Plus access confirmed; all 8 datasets loaded; smoke training runs complete |
| M2 (May 17 – Jun 17) | Method v1 | GroundBench assembled at full scale; evidence-blindness diagnostic implemented; first full v5.4 training run (1 seed) |
| M3 (Jun 17 – Jul 17) | Baselines | All 7 non-ours baselines evaluated on test set; evidence-blindness diagnostic run across all 8 models; first complete result table |
| M4 (Jul 17 – Aug 17) | Method v2 | Diagnostic-guided retraining (3 seeds final); leave-one-site-out experiments |
| M5 (Aug 17 – Sep 17) | Full experiments | All ablations; scale curve; stress tests; multilingual analysis; provenance gate eval |
| M6 (Sep 17 – Oct 17) | Writing pass 1 | NeurIPS 2026 workshop submission (free feedback round); full paper draft |
| M7 (Oct 17 – Nov 17) | Regeneron STS | STS submission polished and filed (Nov 2026 deadline) |
| M8 (Nov 17 – Dec 17) | Revision | Address workshop reviewer feedback; close experimental gaps |
| M9 (Dec 17 – Jan 17) | ISEF + venue | ISEF regional materials; final venue selection (NeurIPS 2027 main May 2027 = 4-month buffer post-M9) |

---

## 13. Success criteria

**NeurIPS 2027 main acceptance** requires all three:
- Our method: IMG ≥ 6 pp, ESG ≥ 6 pp, IPG ≥ 4 pp on laterality subset
- At least 6 of 8 baselines: IMG < 5 pp (establishing prevalence of evidence-blindness)
- Our accuracy within 2 pp of the strongest baseline

**Minimum publishable outcome**: Workshop paper at NeurIPS 2026 GenAI4Health or ML4H + Regeneron STS submission + ISEF materials. Covered by M6-M9 regardless of main-track outcome.

**Unacceptable outcome**: No experiments complete by M5 due to bugs / budget / data access issues. M1 and M2 pre-flight reviews prevent this.

---

## 14. Risks and mitigations

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| CheXpert Plus requires credentialing | Medium | High (−223k images) | M1 week 1 verification; fallback to NIH (112k) + detection-only datasets |
| HO-filter mitigation fails to reduce IMG | Medium | High | Publishable as diagnostic-only negative result; pivot to "evidence-blindness is hard to fix" |
| Full model doesn't beat HO baseline | Low | Very high | M1 smoke training + pre-flight review catches this |
| Budget overrun | Medium | Medium | Hard stop-on-exhaust; buffer consumed first |
| Reviewer demands radiologist validation | Medium | Medium | Explicit in-paper scope statement; cite self-review protocol |
| Synthesized claims too easy | Low | Medium | Report metrics separately for synth vs report claims; ablate |
| BiomedCLIP freeze bug silently disables grounding | High (current state) | High | Fix (Bug 7) is Month 1 priority |

---

## 15. Pointers to code

| Component | File |
|---|---|
| Model definition | `v5/model.py` (`V5Config`, `ImageGroundedVerifier`) |
| Training loop | `v5/train.py` (`V5TrainConfig`, `train_v5`) |
| Losses | `v5/losses.py` (`LossWeights`, `total_loss`) |
| Data loaders | `v5/data/{chexpert_plus,padchest,openi,chestx_det10,rsna_pneumonia,siim_pneumothorax,object_cxr,nih}.py` |
| Claim extractor | `v5/data/claim_extractor.py` |
| Claim parser | `v5/data/claim_parser.py` |
| Claim matcher | `v5/data/claim_matcher.py` |
| Anatomy masks | `v5/data/anatomy_masks.py` |
| GroundBench assembly | `v5/data/groundbench.py` + `v5/modal/build_groundbench.py` |
| Conformal FDR | `v5/conformal.py` |
| Evidence-blindness diagnostic | *to be created:* `v5/eval/evidence_blindness.py` |
| Baseline eval harness | *to be created:* `v5/eval/baselines.py` |
| Training entrypoint | `v5/modal/train_v5.py` |
| Configs | `v5/configs/{base,v5_0_base,v5_1_ground,v5_2_real,v5_3_contrast,v5_4_final}.yaml` |
| Tests | `v5/tests/` (23 passing, 2 skipped as of 2026-04-17) |

---

## 16. Change log

| Date | Version | Author | Change |
|---|---|---|---|
| 2026-04-17 | v5.0 (initial) | Aayan Alwani | Initial spec. Replaces v5 Image-Grounded after 9-month / $900 / no-radiologist re-scoping and evidence-blindness reframe. |
