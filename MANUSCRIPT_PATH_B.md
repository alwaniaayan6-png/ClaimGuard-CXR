# ClaimGuard-CXR (Path B): Image-Grounded Claim-Level Hallucination Detection for Radiology Reports with Conformal False Discovery Rate Control

**Status:** skeleton draft, numbers deferred until Gate G5 populated.
**Target venue:** npj Digital Medicine (primary) or Medical Image Analysis; Nature Communications if results support it.

**Authors (planned):** Aayan Alwani, Weill Cornell Biostatistics Core TBD, Weill Cornell Radiology consultant TBD, Ashley Laughney (senior, corresponding).

---

## Abstract  (~250 words — final pass once numbers land)

Generative radiology systems hallucinate: they produce claims that contradict the underlying imaging study. Prior work has approached this with report-level similarity scores that cannot isolate the offending claim, and with claim-level LLM-as-judge pipelines that offer no formal error control. We present **ClaimGuard-CXR**, a claim-level hallucination detection system for chest radiography that (i) grounds every test claim against radiologist-drawn pixel annotations rather than another radiologist's text, (ii) trains an image-grounded cross-modal verifier with a four-way contrastive objective that mathematically defeats the hypothesis-only lexical shortcut, and (iii) controls false discovery rate on the set of claims certified as "safe" via a site-aware conformal procedure. We build **ClaimGuard-Bench-Grounded**, a unified open-data evaluation across four public radiologist-annotated datasets (PadChest-GR, RSNA Pneumonia, SIIM-ACR Pneumothorax, ChestX-Det10). Across `N` claims and three production vision-language models, ClaimGuard-CXR achieves `TBD` accuracy with `TBD` AUROC, controlling FDR at every tested α ∈ {0.01, 0.05, 0.10, 0.20} under weighted conformal calibration. A per-site ablation demonstrates that the image-swap contrastive term is the necessary-and-sufficient fix for the hypothesis-only shortcut: without it, HO gap is −2.6 pp (v4 historical); with it, HO gap is +`TBD` pp and image-swap gap is +`TBD` pp. Finally, we show that a provenance-tier gate using only three metadata fields downgrades 100% of same-model self-agreement pairs without changing the verifier's decisions on genuinely independent evidence. All code, model weights, and `ClaimGuard-Bench-Grounded` benchmark artifacts are released under permissive licenses.

---

## 1. Introduction

*Problem framing.* Free-text chest radiology report generators now produce clinically-plausible reports but continue to hallucinate findings — laterality swaps, severity inversions, fabricated measurements, invented prior comparisons. A deployed system that flags unreliable claims must answer three questions the prior literature does not address jointly:

1. **What is ground truth?** Comparing generated text to reference text inherits the reference's errors. We argue ground truth must be anchored in the pixels a radiologist actually drew on.
2. **Does the verifier use evidence?** A binary classifier that achieves 98% accuracy on synthetic perturbations can score 97.7% with the evidence masked out — demonstrating shortcut learning. Without an evidence-use guarantee, the "verifier" is a pattern-matcher.
3. **What statistical guarantee is offered?** Accuracy + AUROC on a single test set does not bound the rate of hallucinations among "safe" claims in a clinical deployment. Formal FDR control is necessary.

*Three contributions.*

- **C1 — ClaimGuard-Bench-Grounded.** An open-data, multi-site, radiologist-image-annotation-anchored benchmark unifying PadChest-GR + RSNA + SIIM + ChestX-Det10 under one canonical ontology. Grounded decisions at IoU ≥ 0.5 (sensitivity reported at {0.1, 0.3, 0.5, 0.7}).
- **C2 — Image-grounded verifier with four-way contrastive training.** An architecture (BiomedCLIP + RoBERTa + cross-modal attention + soft region prior) and training objective that make the hypothesis-only baseline fail by construction through image-swap negatives.
- **C3 — Site-aware conformal FDR.** Inverted cfBH + Tibshirani-2019 weighted cfBH with effective-sample-size fallback. Per-site characterization of when each variant is tenable.

*Relation to prior work.* MAIRA-2 + RadFact (2024) grounds report generation in IoU but uses LLM-judged entailment for evaluation. PadChest-GR releases grounded sentences but not a verification benchmark. Anatomically-Grounded Fact Checking (arXiv 2412.02177, Dec 2024) is the closest prior art — single-dataset grounded fact-checking — and we explicitly frame our contribution as *open-data multi-site* extension with conformal FDR layered on top.

---

## 2. Methods

### 2.1 ClaimGuard-Bench-Grounded construction

**Datasets (public, no PhysioNet credentialing required).** PadChest-GR (4,555 studies, 7,037 grounded sentences), RSNA Pneumonia (≈6k radiologist lung-opacity boxes), SIIM-ACR Pneumothorax (radiologist segmentation), ChestX-Det10 (≈3.5k radiologist boxes across 10 classes). NIH CXR14 BBox subset (≈1k boxes, 8 classes) and Object-CXR (≈9k device boxes) as secondary.

**Ontology harmonization.** 13 canonical findings (`configs/ontology.yaml`) map English + Spanish surface forms to a common id. Claims whose findings do not map are marked `UNGROUNDED` and excluded from headline tables — this fraction (~40–55% estimated) is itself a reported result.

**Claim extraction.** Pluggable LLM-backed extractor. Validated on RadGraph-XL reference claims; extractor must pass a claim-precision ≥ 0.95 gate before use on test data.

**Grounding logic.** `ground_claim()` returns `GROUNDED_SUPPORTED`, `GROUNDED_CONTRADICTED`, `GROUNDED_ABSENT_OK`, `GROUNDED_ABSENT_FAIL`, or `UNGROUNDED`. Absent claims (e.g., "no pneumothorax") are `GROUNDED_ABSENT_OK` only when the dataset's annotators drew boxes for that pathology and drew none on this image. A dataset that does not label a given finding cannot evaluate absence claims about it.

### 2.2 Image-grounded verifier

**Architecture.** Three encoders: RoBERTa-large (shared for claim + evidence), BiomedCLIP ViT-B/16 (last 4 layers unfrozen), cross-modal attention (4-layer bidirectional). Three heads: verdict (2-class softmax), score (sigmoid, for conformal p-values), epistemic uncertainty (MC-dropout). Soft region prior (Gaussian-smoothed laterality+region mask) is an additive attention bias, not a hard gate.

**Four-way contrastive objective.** For each claim `c`:

| Variant | Evidence | Image | Label |
|---|---|---|---|
| V1 | supp | correct | 0 |
| V2 | contra | correct | 1 |
| V3 | supp | random-patient | 1 |
| V4 | supp | zero | 1 |

Loss: `L = λ_ce · CE({V1,V2,V3}) + λ_evi · hinge(m, s(V1)-s(V2)) + λ_img · hinge(m, s(V1)-s(V3)) + λ_mask · CE(V4)`.

V3 is the HO-shortcut fix: text is identical between V1 and V3; the only signal distinguishing them is the image. A text-only model cannot satisfy the image margin.

**Adversarial HO filtering.** Any training example solved by the 3-seed HO ensemble at probability ≥ 0.9 is dropped. 2 iterations. Document yield loss.

### 2.3 Conformal FDR

Two variants. **Inverted cfBH:** calibrate on contradicted claims; p-value is one-sided tail probability of the test score exceeding contradicted-distribution scores. **Weighted cfBH (Tibshirani 2019):** density-ratio-weighted calibration using a calibrated gradient-boosted classifier discriminating calibration from test covariates; ESS diagnostic; falls back to un-weighted when ESS degrades.

### 2.4 Provenance gate

Three metadata fields (`claim_generator_id`, `evidence_generator_id`, `evidence_source_type`) assign an evidence trust tier (`trusted` / `independent` / `same_model` / `unknown`). Only `trusted` and `independent` tiers pass the gate. FDR guarantees in the paper are reported on the post-gate subset.

---

## 3. Results

### 3.1 Baselines & main result  (Table 1)

| Method | Contaminated on | Accuracy | Macro F1 | Contra Recall | AUROC |
|---|---|---:|---:|---:|---:|
| Majority class | all | TBD | TBD | TBD | TBD |
| Rule-based negation | all | TBD | TBD | TBD | TBD |
| Zero-shot LLM judge (GPT-4o) | MIMIC? | TBD | TBD | TBD | TBD |
| Zero-shot LLM judge (Llama-3.1-405B) | MIMIC? | TBD | TBD | TBD | TBD |
| CheXagent-8b | MIMIC, CheXpert | TBD | TBD | TBD | TBD |
| MedGemma-4B | MIMIC, CheXpert | TBD | TBD | TBD | TBD |
| Llama-3.2-Vision 11B | unknown | TBD | TBD | TBD | TBD |
| RadFlag | — | TBD | TBD | TBD | TBD |
| ClaimGuard v4 (text-only) | CheXpert | TBD | TBD | TBD | TBD |
| **ClaimGuard-CXR (image-grounded)** | BiomedCLIP=PMC-OA | **TBD** | **TBD** | **TBD** | **TBD** |

Primary test: PadChest-GR held-out. All CIs are 95% bootstrap over 1000 replicates; all pairwise comparisons use McNemar (accuracy) or DeLong (AUROC) with Bonferroni correction.

### 3.2 Conformal FDR control  (Table 2)

4 conformal variants × 4 α × 4 primary sites. Per-variant ESS reported. Weighted cfBH falls back to un-weighted on sites where ESS < 30% of calibration set.

### 3.3 HO-gap closure  (Table 4)

| Training objective | Synthetic HO gap | Image-swap gap | PadChest-GR weighted-κ |
|---|---:|---:|---:|
| v4 text-only (prior) | −2.6 pp | n/a | n/a |
| v4 + evidence-contrast only | TBD | TBD | TBD |
| v4 + 4-way contrast (ours) | TBD | TBD | TBD |

Gate G2 success criterion: HO gap ≥ 10 pp AND image-swap gap ≥ 8 pp AND κ ≥ 0.5 at IoU 0.5.

### 3.4 Ablation  (Table 3)

image-on/off × retrieval-on/off × contrastive-loss-on/off × adversarial-filter-on/off × training-mix — ≥ 24 cells on PadChest-GR.

### 3.5 Fairness

Per-subgroup metrics by sex, age quartile, scanner manufacturer, report length (PadChest-GR has the fields; other sites fall back to what's available). Parity gap reported with CI.

### 3.6 Provenance-gate scaling experiment

1,000 images × 3 VLMs × 4 temperatures. Downgrade rate as a function of sampling entropy.

### 3.7 Clinical utility via decision-curve analysis

Vickers–Elkin net benefit at threshold probabilities {0.05, 0.10, 0.20}. Compared against treat-all and treat-none reference curves.

---

## 4. Discussion

Three honest limitations:

1. **Synthetic perturbations in training data.** Real VLM hallucinations have a different distribution. Mitigated by the real-VLM training-data pipeline, but residual distribution shift remains.
2. **IoU threshold and region prior granularity.** Hard hyperparameters. Sensitivity curves are supplied; 0.5 chosen to match MAIRA-2/RadFact and PadChest-GR conventions.
3. **No recruited radiologist panel.** Ground truth comes from radiologist annotations embedded in public datasets. A prospective reader study remains future work.

---

## 5. Data & Code Availability

- Code: GitHub `claimguard-nmi` (Apache-2.0).
- Weights: HuggingFace `claimguard-cxr/*` with pinned revisions.
- Benchmark: Zenodo DOI (ClaimGuard-Bench-Grounded v1.0).
- Docker image: `claimguard-nmi:0.1.0a0`.

---

## References (placeholder list; expand at submission)

1. Jin & Candès (2023). Selection by prediction with conformal p-values. JMLR.
2. Tibshirani, Barber, Candès, Ramdas (2019). Conformal prediction under covariate shift. NeurIPS.
3. Bates, Candès, Lei, Romano, Sesia (2023). Testing for outliers with conformal p-values. Annals of Statistics.
4. Benjamini & Hochberg (1995). Controlling the false discovery rate. JRSS-B.
5. Chen et al. (2025). RadFlag: hallucination flagging for CXR reports.
6. Bannur et al. (2024). MAIRA-2 + RadFact. Microsoft.
7. Feyjoo et al. (2024). PadChest-GR: grounded chest radiograph sentences. BIMCV.
8. Vickers & Elkin (2006). Decision curve analysis. Medical Decision Making.
9. DeLong, DeLong, Clarke-Pearson (1988). Comparing AUROC.
10. McNemar (1947). Note on the sampling error of the difference between correlated proportions.
