# ClaimGuard-CXR Path B: Image-Grounded Claim Verification

**Status:** design v2 (post-feasibility-review)
**Target venue:** npj Digital Medicine / Medical Image Analysis (primary); Nature Communications (stretch); Nature Machine Intelligence (only if Laughney + radiologist co-author + PhysioNet credentialing all land)
**Written:** 2026-04-17 (v1), revised 2026-04-17 (v2 post-feasibility-review)
**Authors (planned):** Aayan Alwani, Ashley Laughney (senior, pending confirmation), Weill Cornell Biostatistics Core co-author (TBD), Weill Cornell Radiology consultant (TBD)

---

## 0. Why Path B

ClaimGuard's prior v1–v4 pipeline uses *text-vs-text* ground truth: an extracted claim is compared against radiologist-written reference report text. That pipeline has two structural issues that a Nature-family reviewer will flag immediately:

1. **Hypothesis-only (HO) artifacts.** The v1/v3/v4 checkpoints show that evidence-masked baselines perform nearly as well as the evidence-using verifier (HO gap −2.61 pp on v4 v2). A verifier that doesn't *use* evidence cannot claim to *ground* claims in evidence.
2. **Circularity of text-vs-text ground truth.** A claim labeled "supported" by CheXbert-diff on the reference report is only as correct as CheXbert on that report. Real radiologist disagreement on reports runs ~15–20% on individual findings.

Path B solves both by **anchoring ground truth in pixels that radiologists drew on**, not in text that a labeler parsed.

**Credentialing status (CORRECTED after v2 review):** MS-CXR and CheXmask (and the images CheXmask indexes for MIMIC-CXR / CheXpert / VinDr-CXR) all require **PhysioNet credentialed access**. An earlier draft claimed otherwise — this was wrong. The v2 plan **drops MS-CXR from the headline** and rebuilds around datasets that are genuinely PhysioNet-free:

- **PadChest-GR** (BIMCV + Microsoft, Nov 2024): 4,555 CXR studies with 7,037 positive-finding sentences each tied to a radiologist-drawn bounding box. Distributed by BIMCV under a data-use request (institutional email) — **not PhysioNet**. Primary grounded test set.
- **RSNA Pneumonia Detection Challenge** (Kaggle): ~30k CXRs, ~6k radiologist-drawn bounding boxes for lung opacity. Pathology-specific grounded eval.
- **SIIM-ACR Pneumothorax** (Kaggle): ~12k CXRs with radiologist pneumothorax segmentation. Pathology-specific grounded eval.
- **NIH ChestX-ray14 (with bounding boxes)**: ~1k radiologist-drawn boxes across 8 pathologies on NIH CXR14 images. Fully open.
- **ChestX-Det10 / ChestX-Det-Dataset**: ~3.5k images with 10 disease bounding boxes, radiologist-drawn, fully open on GitHub.
- **Object-CXR** (MIDL 2020): ~9k images with bounding boxes for foreign objects. Fully open. Useful for fabricated-device-claim evaluation.
- **CheXpert Plus** (Stanford registration, already in hand): training source. Test subset used for in-distribution ablation only.
- **OpenI**: text-only cross-dataset eval; no grounded labels, used for robustness check.

**Optional bonus if PhysioNet credentialing is filed in parallel and clears in time:** MS-CXR (phrase-grounding), CheXmask-on-PadChest and CheXmask-on-NIH subsets (anatomical masks using CC-BY masks on non-credentialed images), VinDr-CXR, ReXVal, ReXErr. None of these are required for submission. All are treated as supplementary.

### Key reframing

We stop asking "does this claim agree with a radiologist's report text?" and instead ask

> "Does this claim's *referent* (the pathology it asserts, its location, its laterality, its size) overlap the radiologist's pixel-level annotation of that pathology?"

This is the core methodological contribution of the paper.

---

## 1. Scope & contributions

Three tight contributions, each separately publishable but bundled here:

### C1. ClaimGuard-Bench-Grounded
**First unified open-data, multi-site, IoU-thresholded, conformal-FDR-calibrated claim-level radiology hallucination benchmark.** Every evaluated claim is scored against radiologist-drawn pixel annotations. Covers 4 primary datasets (PadChest-GR, RSNA, SIIM, ChestX-Det10) plus 2 supplementary (NIH CXR14 boxes, Object-CXR), 3 production VLMs (CheXagent-8b, MedGemma-4B, Llama-3.2-Vision 11B), and 10 pathology classes with dense annotations.

**Prior-art differentiation (required — C1's novelty claim was wrong in v1):** MAIRA-2 + RadFact (Jun 2024) does grounded report *generation* with IoU entailment on MIMIC-CXR; PadChest-GR (Nov 2024) is a *dataset* release, not a claim-verification benchmark; "Anatomically-Grounded Fact Checking of Automated Chest X-ray Reports" (arXiv 2412.02177, Dec 2024) is the closest prior art — a multi-label grounded fact-checker trained on a single dataset. C1's genuine novelty is **(i) open-data multi-site evaluation that unifies 4 independently-collected grounded datasets under one canonical ontology, (ii) IoU-sensitivity analysis across thresholds, and (iii) conformal FDR calibration per site**. Not "first image-grounded claim verification" — that ship sailed in Dec 2024.

### C2. Image-grounded claim verifier
First claim verifier that reads the CXR pixels *and* the claim text *and* retrieved textual evidence, and that has been trained so the HO baseline fails (gap ≥ 10 pp). Architecture: BiomedCLIP (frozen → partially unfrozen) image encoder + RoBERTa-large text encoder + cross-modal attention + per-claim ROI extraction from image features + verdict head + epistemic-uncertainty head. Trained with an evidence-contrastive objective that mathematically requires the model to degrade under evidence masking.

### C3. Grounded conformal FDR with distribution-shift handling
Four conformal variants implemented and compared (inverted cfBH, weighted cfBH with density-ratio estimation, doubly-robust cfBH, prediction-powered inference). Per-site empirical FDR and power across 5 public datasets with formal exchangeability treatment and weighted-variant fallback logic when effective sample size degrades.

---

## 2. Non-goals for Path B

- **No claim-level radiologist-labeled error benchmark.** Those live on PhysioNet (ReXVal, ReXErr) and are blocked by credentialing. If credentialing succeeds in parallel, we will add them as secondary validation — but the paper does not depend on them.
- **No prospective deployment / reader study.** Those require recruited radiologists and IRB-live protocols. Out of scope for Path B. Reader-study substitute uses radiologist image annotations as the oracle.
- **No clinical utility endpoints** (time-to-read, diagnostic error rate in deployment). Reserved for a follow-up Nature Medicine paper.
- **No MIMIC-CXR, no VinDr-CXR (PhysioNet release), no ReXVal, no ReXErr, no RadGraph-XL as primary data.** All PhysioNet-bound.

---

## 3. Datasets (all non-credentialed)

| Dataset | Size | Annotation type | Access | Role |
|---|---|---|---|---|
| **PadChest-GR** | 4,555 studies / 7,037 grounded sentences | Radiologist bounding boxes tied to positive-finding sentences | BIMCV data-use request (institutional email) | **Primary grounded test** |
| **RSNA Pneumonia** | ~30k (6k with boxes) | Radiologist-drawn lung-opacity boxes (panel of 3) | Kaggle, free | Pneumonia-specific grounded test |
| **SIIM-ACR Pneumothorax** | ~12k | Radiologist pneumothorax segmentation | Kaggle, free | Pneumothorax-specific grounded test |
| **ChestX-Det10** | ~3.5k | Radiologist boxes, 10 disease classes | GitHub open | Grounded test — 10 pathologies |
| **NIH CXR14 + BBox subset** | 112k (1k with boxes) | Radiologist boxes on 8 pathologies | Direct download | Supplementary grounded test |
| **Object-CXR** | ~9k | Foreign-object boxes | Open | Device/fabricated-object grounded test |
| **CheXpert Plus** | ~200k with reports | 14 labels, CheXbert labeler + radiologist-adjudicated test | Stanford registration (have) | Training source |
| **OpenI** | ~4k | Reports only | Direct download | Text-only OOD robustness |

**Dropped from v1 draft because of credentialing:** MS-CXR (PhysioNet-credentialed), CheXmask source images for MIMIC-CXR / CheXpert / VinDr (PhysioNet).

**Optional supplementary if PhysioNet credentialing lands in parallel:** MS-CXR phrase-grounding, CheXmask-on-PadChest anatomical masks, VinDr-CXR, ReXVal, ReXErr, RadGraph-XL. None are required for submission.

### Data splits

All datasets are split at the **patient** level where patient ID is available (CheXpert Plus, PadChest, OpenI, RSNA, SIIM); at the **study** level where not (MS-CXR, NIH, CheXmask).

Train / calibration / test sizes targeted (CheXpert Plus is the training source):

| Split | Source | n claims (target) |
|---|---|---|
| Train | CheXpert Plus training patients | ~25,000 |
| Calibration | CheXpert Plus calibration patients | ~8,000 |
| Test (primary) | CheXpert Plus held-out | ~12,000 |
| Test (OOD-1, site) | OpenI | ~3,000 |
| Test (OOD-2, pixel-grounded) | MS-CXR | 1,162 |
| Test (OOD-3, localized) | PadChest-localized | ~3,000 |
| Test (OOD-4, pathology-specific) | RSNA Pneumonia | ~6,000 claims |
| Test (OOD-5, pathology-specific) | SIIM Pneumothorax | ~2,000 claims |

### Radiologist ground truth chain

For each test pair `(image, claim)`:

1. **Claim referent extraction.** LLM-ensemble (GPT-4o + Claude Sonnet 4.5 + Llama-3.1-405B if available) parses the claim into a structured `{finding, laterality, region, severity, certainty, temporality}` tuple.
2. **Image annotation lookup.** For the finding the claim asserts, look up radiologist-drawn boxes/masks on the image. If the dataset has this annotation type, proceed. If not, mark pair as `ungrounded` and exclude from the pixel-grounded evaluation table (include in text-vs-text supplementary).
3. **Grounding rule.** The claim is `GROUNDED_SUPPORTED` iff the radiologist annotation exists and overlaps the claim's asserted region (IoU ≥ threshold) and the laterality/severity/size agree within tolerance. `GROUNDED_CONTRADICTED` iff annotation exists and disagrees on location / laterality / severity / size. `GROUNDED_ABSENT` iff the finding is absent in the annotations but the claim asserts presence. `UNGROUNDED` iff the finding is not in the dataset's annotation ontology.

Only `GROUNDED_*` pairs are used for the headline tables. This is the key design principle: **no LLM-in-the-loop in the ground truth**.

---

## 4. Model architecture

### 4.1 Image-grounded verifier

```
                claim text        retrieved evidence text         CXR image
                    │                      │                         │
              tokenizer (RoBERTa)    tokenizer (RoBERTa)      BiomedCLIP / RadDINO
                    │                      │                         │
                RoBERTa-large         RoBERTa-large              image encoder
                (shared weights;      (shared weights;          (last 4 layers
                mean-pooled)          mean-pooled)              unfrozen)
                    │                      │                         │
                    └──────────┬───────────┘                         │
                               │                                     │
                      text fusion MLP                         ROI extraction
                               │                             (per-claim spatial
                               │                              attention pooling)
                               │                                     │
                               └─────────────┬───────────────────────┘
                                             │
                                    cross-modal attention
                                    (text→image & image→text)
                                             │
                                             ▼
                          ┌──────────────────────────────────┐
                          │       fused representation        │
                          │      (1792-dim, frozen after     │
                          │       contrastive pretraining)   │
                          └──────────┬───────────────────────┘
                                     │
                 ┌───────────────────┼─────────────────────┐
                 ▼                   ▼                     ▼
          verdict head          score head            uncertainty head
          (2-class softmax)     (sigmoid 0..1)     (MC-dropout ensemble)
```

Components:

- **Text encoder:** RoBERTa-large, pinned revision. Two forward passes with shared weights for claim and evidence; outputs mean-pooled over tokens.
- **Image encoder:** BiomedCLIP vision tower (preferred, not MIMIC-trained) or RadDINO-v2 as alternative. Last 4 transformer layers unfrozen during fine-tune. 224×224 input, normalized per BiomedCLIP preprocessor.
- **ROI extraction:** per-claim spatial attention pooling. The claim's structured `{region, laterality}` tuple selects a spatial attention mask over image patches (hard priors: left/right, upper/mid/lower, apical/basal). Soft attention on top. This is the key image-grounding mechanism — the model looks at the region the claim claims to describe.
- **Cross-modal attention:** 4-layer Transformer with bidirectional cross-attention between text (concatenated claim+evidence) and image patches.
- **Verdict head:** 2-class softmax — `Not Contradicted` (0) vs `Contradicted` (1).
- **Score head:** sigmoid — calibrated supported-probability for conformal p-values.
- **Uncertainty head:** MC-dropout (drop rate 0.1, 10 samples at inference) yielding epistemic variance estimate.

Parameter counts (target): RoBERTa-large 355M + BiomedCLIP ViT-B/16 86M + cross-attn ~40M ≈ **~480M total**, fits comfortably on a single H100 with fp16.

### 4.2 Four-way contrastive training objective (REVISED after v2 review)

v1 of this section used only evidence-swap + image-mask negatives, which are gameable — the model can learn lexical signatures of `evidence_supp` vs `evidence_contra` and ignore the image entirely. The v2 design forces both pathways to carry signal with **four variants per claim `c`**:

| Variant | Claim | Evidence | Image | Label |
|---|---|---|---|---|
| V1 (positive) | `c` | `evidence_supp` | `image_correct` | 0 (Not Contradicted) |
| V2 (evidence-swap) | `c` | `evidence_contra` | `image_correct` | 1 (Contradicted) |
| V3 (image-swap) | `c` | `evidence_supp` | `image_random_patient` | 1 (Contradicted) |
| V4 (image-mask) | `c` | `evidence_supp` | all-zeros image | 1 (Contradicted) |

Total loss:

```
L = λ_ce · CE(verdict | {V1..V4}) + λ_evi · hinge(m_e, [s(V1) − s(V2)]) + λ_img · hinge(m_i, [s(V1) − s(V3)])
```

- `λ_ce = 1.0`, `λ_evi = 0.3`, `λ_img = 0.3`, `m_e = m_i = 0.2`.
- V3 (image-swap with a random-patient CXR) **mathematically forces the image pathway to contribute** — a text-only model cannot distinguish V1 from V3 because text is identical.
- V4 (image-mask) is a redundant sanity check; kept as diagnostic, weight `λ_mask = 0.05`.

**HO-baseline fails this loss by construction** — it cannot distinguish V1 from V3.

Plus:
- Adversarial HO filtering — any training example the HO baseline (3-seed ensemble) solves at probability ≥ 0.9 on V1/V2 is dropped. Iterate 2×. Expected training-set shrinkage 2–3×; document actual yield before G2 evaluation.

Success criterion (Gate G2):
- Text-only HO gap ≥ 10 pp.
- Image-swap gap ≥ 8 pp (V1 supported-prob minus V3 supported-prob, averaged).
- PadChest-GR grounded agreement ≥ 75% (weighted-κ vs radiologist boxes at IoU 0.5).

### 4.3 Retrieval

Evidence retrieval is **per-site institution-disjoint**:

- Test site S → retrieval corpus built from *all other sites combined*.
- Patient-disjoint within each corpus (no patient can appear in both the retrieval corpus and the test set).
- Report-disjoint (trivially enforced by patient-disjoint).

Retriever: MedCPT dual encoder + BM25 via `rank_bm25`, fused with RRF (k=60), cross-encoder rerank (MedCPT cross-encoder). Top-3 passages concatenated as evidence.

Adversarial-leakage probe: for a held-out set, assert retrieval never returns a passage from the source report. Must pass (0% failure) before any eval.

---

## 5. Grounding logic (the core novel module)

For each `(image, claim)`:

```python
IOU_THRESHOLD = 0.5  # headline; sensitivity curve at {0.1, 0.3, 0.5, 0.7}

def ground_claim(image, claim, annotations, dataset_label_schema):
    """
    Inputs:
      image                 — the CXR under test
      claim                 — structured {finding, laterality, region, severity, certainty}
      annotations           — radiologist-drawn {boxes, masks} keyed by canonical_finding
      dataset_label_schema  — set of findings the annotators were asked to draw

    Output: one of GROUNDED_SUPPORTED, GROUNDED_CONTRADICTED,
            GROUNDED_ABSENT_OK, GROUNDED_ABSENT_FAIL, UNGROUNDED
    """
    canonical_finding = canonicalize(claim.finding)

    # CRITICAL v2 fix — annotation absence != finding absence.
    # A dataset that only labels 8 pathologies cannot tell you about the 9th.
    if canonical_finding not in dataset_label_schema:
        return UNGROUNDED  # we cannot evaluate this claim on this dataset

    # Prior-comparison, measurement-fabrication, and implicit claims
    if claim.claim_type in {"prior_comparison", "measurement", "implicit_negation_global"}:
        return UNGROUNDED

    finding_annotations = annotations.get(canonical_finding, [])

    # Claim asserts finding is absent
    if claim.certainty == 'absent':
        if len(finding_annotations) == 0:
            return GROUNDED_ABSENT_OK       # claim says absent, no box drawn — agree
        return GROUNDED_CONTRADICTED        # claim says absent, box exists — disagree

    # Claim asserts finding is present
    if len(finding_annotations) == 0:
        return GROUNDED_ABSENT_FAIL          # claim says present, no box drawn — contradicts

    # Both claim and annotation assert presence; check spatial agreement.
    # For diffuse/global findings (cardiomegaly, pulmonary edema), overlap is
    # degenerate — use size-ratio match instead of IoU.
    if canonical_finding in DIFFUSE_FINDINGS:
        return size_ratio_match(claim, finding_annotations)

    # For localizable findings, compare the claim's asserted region to the
    # radiologist-drawn annotation.
    soft_prior = soft_region_prior(claim.region, claim.laterality, image.shape)
    ann_mask = union_of_masks(finding_annotations, image.shape)
    iou = weighted_iou(soft_prior, ann_mask)

    if iou >= IOU_THRESHOLD and lateralities_match(claim, finding_annotations):
        return GROUNDED_SUPPORTED
    return GROUNDED_CONTRADICTED
```

Key changes from v1 draft:

- **IoU threshold raised from 0.3 → 0.5** for the headline. Sensitivity reported at {0.1, 0.3, 0.5, 0.7}.
- **Region prior is a soft attention bias**, not a hard mask. `soft_region_prior` returns a 2D weight map peaking in the asserted region and decaying smoothly; `weighted_iou` is the IoU between weighted masks.
- **`dataset_label_schema` gates UNGROUNDED.** A claim about "pneumothorax" evaluated on a dataset that only labels "pneumonia" is `UNGROUNDED`, not `GROUNDED_ABSENT_OK`. This prevents the category error flagged by review.
- **`GROUNDED_ABSENT_OK` vs `GROUNDED_ABSENT_FAIL`** distinguishes valid negation from missing-annotation fail. Both are still "the claim is correctly grounded" in one direction.
- **Diffuse findings** (cardiomegaly, pulmonary edema, pleural effusion when not localized) bypass IoU and use size-ratio match.
- **Ungroundable claim types** (prior-comparison, measurement-fabrication, implicit-global-negation) return `UNGROUNDED` explicitly — these claims cannot be verified against any still-image annotation and are excluded from headline tables. They are reported as a separate category ("fraction of real VLM claims that current grounded benchmarks cannot evaluate") — and that number (estimated 40–55%) is itself a contribution.

The IoU threshold (0.5) is the standard used by PadChest-GR and MAIRA-2/RadFact; no LLM-in-the-loop calibration is used, removing the v1 dependency on LLM-ensemble labels for threshold setting.

### Ontology harmonization

Each dataset's finding ontology is mapped to a common canonical ontology (28 findings). Mapping table stored in `configs/ontology.yaml` and versioned. Example: RSNA `opacity` → canonical `lung opacity`; MS-CXR `lung opacity` → canonical `lung opacity`; PadChest `patron_alveolar_pulmonar` → canonical `alveolar pattern / lung opacity`.

Claims whose finding does not map to the canonical ontology are marked `UNGROUNDED`.

---

## 6. Conformal FDR

Down-scoped in v2 from 4 variants to 2, per review's staffing-realism critique. Doubly-robust and PPI variants dropped unless biostatistics co-author joins the project with capacity for them.

| Variant | Exchangeability | Shift-handling | When it wins |
|---|---|---|---|
| Inverted cfBH (current) | Strong (calibration in contradicted class) | None | In-distribution; clean calibration set |
| Weighted cfBH (Tibshirani 2019) | Weakened; weights via density ratio | Yes, via importance weights | Moderate covariate shift |

For each variant, report per-site empirical FDR at α ∈ {0.01, 0.05, 0.10, 0.20} with bootstrap CIs (1000 replicates), plus power and effective-sample-size (ESS) diagnostics. Weighted cfBH density ratio estimated via calibrated gradient-boosted classifier on claim-level features (finding, laterality, certainty, report-length).

**Guarantee framing (v2 — removes p-hacky v1 language):** For each variant, characterize the operating regime — which per-site shift-magnitude and calibration-size does it tolerate? Report per-site which variant applies and which does not. Do not report a pooled "at least one variant controls" claim.

Doubly-robust cfBH and PPI are acknowledged in the Discussion as natural extensions for future work.

---

## 7. Baselines

All prompt-tuned with ≥3 prompt variants per baseline:

1. **Rule-based negation detector** — proximity heuristic over claim and evidence.
2. **Untrained RoBERTa-large** — majority class predictor.
3. **Zero-shot LLM judge** — GPT-4o, Claude Sonnet 4.5, Llama-3.1-405B, each prompted as a medical fact-checker.
4. **CheXagent-8b** — given CXR + claim, asked whether supported.
5. **MedGemma-4B** — same prompt protocol.
6. **Llama-3.2-Vision 11B** — same.
7. **MAIRA-2** (if weights accessible; else documented as gated-access limitation).
8. **RadFlag** — consistency-based flagging per Chen et al. 2025.
9. **Prior ClaimGuard v4 (text-only)** — the pre-Path-B system, explicitly positioned as the ablation baseline for "add image grounding".

Contamination flags on each (baseline, test site) pair — results on contaminated combinations go to supplementary, not headlines.

---

## 8. Evaluation protocol

### 8.1 Primary tables

- **Table 1** — per-site accuracy, F1, contradiction recall, AUROC, with 95% bootstrap CIs (1000 replicates). Rows: all 9 baselines + ClaimGuard-text-only + ClaimGuard-image-grounded (3 seeds each).
- **Table 2** — per-site conformal FDR at 4 α levels for 4 conformal variants.
- **Table 3** — ablation: image-on/off × retrieval-on/off × contrastive-loss-on/off × adversarial-filter-on/off. ≥24 cells on CheXpert Plus test.
- **Table 4** — HO gap and image-masked gap on CheXpert synthetic + MS-CXR grounded + CheXpert Plus reports.
- **Table 5** — fairness parity: sex, age-quartile, race (where available), scanner manufacturer, report length.
- **Table 6** — grounding rate per dataset (what fraction of claims are `GROUNDED_*` vs `UNGROUNDED`).

All tables emit confidence intervals; all cross-model comparisons apply DeLong (AUROC) or McNemar (accuracy) with Bonferroni correction.

### 8.2 Key figures

- **Fig 1** — system + grounding pipeline.
- **Fig 2** — per-site accuracy/FDR bars.
- **Fig 3** — ablation heatmap.
- **Fig 4** — MS-CXR qualitative case panels (bounding-box overlay with verifier agreement).
- **Fig 5** — calibration curves + fairness parity.
- **Fig 6** — provenance gate at scale (temperature × downgrade rate).

### 8.3 Reader-study substitute

Since we have no recruited radiologists, the reader-study substitute is:

1. Aggregate all MS-CXR phrase-box pairs (1,162) + RSNA (6k) + SIIM (2k) + PadChest-localized (3k).
2. For each, produce the ClaimGuard decision and the grounded ground truth.
3. Report sensitivity / specificity / NPV / PPV per pathology with CIs.
4. **Decision-curve analysis** at clinically plausible operating points.
5. Frame as "agreement with radiologist-drawn image annotations" — an image-anchored reader-equivalent metric.

---

## 9. Reproducibility & release

- **Code:** GitHub repo `claimguard-nmi`, Apache-2.0 license, semver tags.
- **Docker image:** `claimguard/nmi:vX.Y.Z`, pinned to Hydra configs.
- **Weights:** Hugging Face org `claimguard-cxr`, pinned revisions.
- **Benchmark artifact:** Zenodo DOI (ClaimGuard-Bench-Grounded v1.0) with VLM-generated reports + extracted claims + grounded labels + data card + model card.
- **Eval harness:** `python -m claimguard.eval --site <X> --model <Y> --variant <Z>` reproduces every table cell.
- **Experiment tracking:** MLflow; all runs logged with code SHA + data hash + seed.

All released text passes through a Presidio + medical-regex PII scrubber, manually audited on 500 samples.

---

## 10. Risk register (Path B-specific)

| # | Risk | Likelihood | Mitigation |
|---|---|---|---|
| B1 | BiomedCLIP pretrained on PMC-OA / PubMed — possible contamination on CheXpert-adjacent figures | Low | Use RadDINO-v2 as alternative; compare; contamination audit |
| B2 | MS-CXR is small (1,162 pairs) for a headline | High | Bundle with RSNA (6k) + SIIM (2k) + PadChest-localized (3k) → ~12k image-grounded claims total |
| B3 | Ontology harmonization introduces label noise | Medium | Canonical ontology review by Laughney + radiology consultant if available; release mapping publicly |
| B4 | IoU threshold (0.3) is hyperparameter the reader may push back on | Medium | Report results at IoU ∈ {0.1, 0.3, 0.5, 0.7} as a sensitivity curve |
| B5 | RadDINO-v2 and BiomedCLIP are pretrained on MIMIC or CheXpert — verifier could have pretraining leakage | Medium | Use BiomedCLIP (PMC-OA only, no MIMIC); document backbone choice clearly |
| B6 | CheXmask hosted on PhysioNet | Medium | Fallback to Shenzhen/Montgomery/JSRT lung segmentation + trained anatomy model |
| B7 | PadChest BIMCV registration gates lab-email requirement | Low | Laughney's Weill Cornell email; alternative: drop PadChest-localized from headline |
| B8 | MAIRA-2 weights gated | High | Skip MAIRA-2 baseline or document; replace with CheXagent-8b + MedGemma-4B as 2 strong-VLM baselines |
| B9 | LLM claim extractor hallucinates claims | Medium | RadGraph-XL-free validation on CheXpert Plus test; reject extractor below 0.95 precision |
| B10 | Compute budget overrun | Medium | Per-phase auto-halt; dedicated Optuna budget of $150 for HP search |
| B11 | Anthropic credits exhausted (already true) | Certain | Use GPT-4o and Llama-3.1-405B (via OpenRouter/Together) as primary LLM labelers; Claude re-added when credits topped up |
| B12 | HO-gap fix fails (Plans A/B/C all fail) | Low–Medium | §2.3 negative-result pivot path retained; paper still publishable as honest-failure benchmark |
| B13 | Image-grounded verifier doesn't beat text-only ClaimGuard v4 | Medium | Report honestly; frame as "image grounding helps on out-of-distribution + novel pathologies but synthetic CheXpert is saturated" |

---

## 11. Execution phases (Path B specific)

Condensed from master plan; see `docs/path_b_execution_plan.md` (to be authored) for Gantt-level detail.

| Phase | Work | Dependencies | Claude-executable? |
|---|---|---|---|
| P0 | Governance, repo scaffold, MLflow, Docker, CI | None | Yes (except Laughney sign-off) |
| P1 | Data download, ontology harmonization, grounding logic impl + unit tests | P0 | Yes (download depends on user's network) |
| P2 | Claim extractor validation; if fails, replace | P1 | Yes |
| P3 | Image-grounded verifier implementation + contrastive training | P0 | Yes (code); GPU launch requires pre-flight review |
| P4 | Baseline runs (all 9 baselines × 5 sites) | P1, P2 | Mixed — some local, some Modal |
| P5 | Conformal module (4 variants) | P1 | Yes |
| P6 | Full evaluation sweep | P3, P4, P5 | Mixed — GPU heavy |
| P7 | Ablations + fairness + stats | P6 | Yes |
| P8 | Benchmark release packaging | P6 | Yes |
| P9 | Manuscript draft + figures | P7, P8 | Yes |
| P10 | Internal review (Laughney, biostats, radiology) | P9 | Blocked on humans |
| P11 | Submission | P10 | Blocked on humans |

---

## 12. What this doc replaces / supersedes

- **`ARCHITECTURE.md`** — superseded for v5+ (Path B). The v1–v4 doc remains as historical reference.
- **`CLAIMGUARD_PROPOSAL.md`** — appendix Path B is added. V3.x proposals reframed.
- **`MANUSCRIPT_MINI.md`** — v5 draft will be a new document; v4 draft archived.

All changes to training / evaluation code in the Path B package MUST update this doc in the same commit (doc-sync policy).

---

## 13. Hard blockers requiring human action

These cannot be resolved by Claude autonomously:

1. **Laughney sign-off** on (a) Path B scope, (b) PI authorship, (c) IRB exempt determination submission, (d) Modal budget lift to $2,500, (e) PhysioNet credentialing initiation (optional but recommended).
2. **Biostatistician co-author** — Weill Cornell Biostatistics Core consult → co-authorship.
3. **Anthropic credit top-up** — required to use Claude as one of the three LLM labelers; Path B can proceed without Claude if blocked (GPT-4o + Llama-3.1-405B suffice).
4. **PadChest BIMCV registration** — institutional-email web form.
5. **Kaggle API token** — user-level, needed to download RSNA and SIIM datasets programmatically.
6. **Hugging Face account / org creation** — for final weight release.

All other steps are executable by Claude end-to-end once the above are resolved.

---

## 14. Success / failure criteria per gate

- **G0 (end of setup):** repo exists, configs pinned, CI green on smoke test.
- **G1 (end of data):** ≥3 of 4 primary grounded datasets downloaded (PadChest-GR, RSNA, SIIM, ChestX-Det10), ontology harmonization covers ≥85% of findings in each, grounding logic passes unit tests.
- **G2 (end of training):** text-only HO gap ≥ 10 pp AND image-swap gap ≥ 8 pp AND PadChest-GR weighted-κ ≥ 0.5 at IoU 0.5.
- **G3 (end of conformal):** per-site FDR control characterized; variant-applicability table populated for ≥3 of 4 primary sites.
- **G4 (end of release):** ClaimGuard-Bench-Grounded v0.9 internal-release-ready.
- **G5 (end of eval):** all tables populated with CIs and significance tests.
- **G6 (end of manuscript):** submission out the door to npj Digital Medicine or Medical Image Analysis.

Failing G2 triggers the three-tier HO-gap fallback tree. Failing G3 triggers per-site honest calibration documentation. Failing G5 triggers patch runs.

---

## 15. What happens in parallel with this plan

- Laughney approach (email + meeting) should proceed in parallel with P0–P2.
- PhysioNet credentialing (optional, for bonus ReXVal/ReXErr validation) should be initiated in parallel; if it comes through, add a supplementary section.
- Radiology consultant introduction (optional) should be requested of Laughney in the initial meeting.

None of these block the engineering work in P0–P9.

---

## Appendix A — Why not just fine-tune MAIRA-2

MAIRA-2 is a generator, not a verifier. Its RadFact score is computed via an LLM-as-judge, which means MAIRA-2's own "hallucination evaluation" has the same text-vs-text circularity we are trying to break. MAIRA-2 is used in Path B as a baseline claim generator, not as ground truth.

## Appendix B — Why BiomedCLIP over RadDINO

RadDINO-v2 is pretrained on MIMIC-CXR — using it as the image encoder creates train-on-test contamination when we evaluate on MIMIC-CXR subsets (MS-CXR). BiomedCLIP is pretrained on PMC-OA (figures from open-access biomedical papers), which is much less contaminated with respect to our test sites. We accept the possible quality hit for cleaner pretraining provenance.

## Appendix C — Failure-mode analysis of the image-grounded approach

The image-grounded pipeline has known failure modes:

1. **Global findings (e.g., "cardiomegaly")** — not localizable. Handled by whole-image ROI and size-ratio ground truth.
2. **Negated claims (e.g., "no pleural effusion")** — scored as `GROUNDED_SUPPORTED` iff annotations for pleural effusion are absent.
3. **Prior-comparison claims (e.g., "increased compared to prior")** — no image ground truth available without paired prior image. Marked `UNGROUNDED`, excluded.
4. **Implicit-negation claims (e.g., "lungs are clear")** — treated as negation of all thoracic pathologies; scored `GROUNDED_SUPPORTED` iff all annotations absent.
5. **Fabricated-detail claims (e.g., "3 mm nodule")** — size ground truth from segmentation masks when available; otherwise `UNGROUNDED`.

Each failure mode is documented in the grounding module and reflected in the supplementary benchmark metadata.

---

*End of ARCHITECTURE_PATH_B.md.*
