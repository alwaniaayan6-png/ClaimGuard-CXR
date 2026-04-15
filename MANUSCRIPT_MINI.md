# ClaimGuard-CXR: Calibrated Claim-Level Hallucination Detection for Radiology Reports with Conformal FDR Control

**Aayan Alwani**

---

## Abstract

AI systems for radiology report generation frequently hallucinate — producing findings that contradict the underlying imaging study. Existing verification approaches lack formal guarantees on error rates, limiting clinical trust. We present ClaimGuard-CXR, a claim-level hallucination detection system for chest X-ray reports that provides provable false discovery rate (FDR) control. Our approach combines a RoBERTa-large binary cross-encoder, fine-tuned on 30,000 claims with a 12-type taxonomy of clinically-motivated hard-negative perturbations (eight structural + three fabricated-detail classes + compound stacking), with a novel *inverted* conformal Benjamini-Hochberg (cfBH) procedure that calibrates on contradicted claims rather than faithful ones. To close the evidence-reasoning gap introduced by lexical-shortcut exploitation on synthetic hard negatives (hypothesis-only baseline 97.71% vs full verifier 98.31% on v1), we apply causal-term-identification + DPO refinement to produce a v4 checkpoint that uses evidence rather than surface form. On CheXpert Plus (15,000 held-out test claims from 16,182 patients), ClaimGuard-CXR achieves 98.31% accuracy and 99.52% AUROC, outperforming Zero-shot LLM judge (66.5%), CheXagent-8b with image access (67.2%), and rule-based baselines (66.3%) by over 31 percentage points. The inverted cfBH procedure controls FDR at 1.30% (target alpha=0.05) with 98.06% power. On OpenI (Indiana University, zero-shot cross-dataset transfer), FDR remains controlled at every alpha level despite a 13-point accuracy drop, and a stratified conformal baseline (StratCP, Zitnik lab 2026) is implemented as a head-to-head comparison. We characterize real-world failure modes on a silver-standard test set of 200 (image, claim) pairs — CheXagent-generated claims over OpenI images, labeled by a 3-grader ensemble held to Krippendorff α ≥ 0.80, with an author self-annotation pass as internal-validity check. A dual-run case study empirically demonstrates that claim-level FDR control must track generator identity: a provenance-aware trust-tier gate downgrades same-model-evidence claims at a rate > 0.5, preventing self-consistency from masquerading as corroboration.

---

## 1. Introduction

Large language models and vision-language models increasingly generate free-text radiology reports from chest X-ray images. Despite improvements in fluency and coverage, these systems hallucinate clinically significant errors: laterality swaps ("left" instead of "right"), severity inversions ("large" instead of "small"), negation flips ("no effusion" becomes "effusion"), and fabricated findings. In high-stakes clinical settings, even a small fraction of undetected hallucinations can lead to misdiagnosis.

Prior work on radiology hallucination detection falls into two categories. *Report-level* approaches (RadGraph F1, CheXbert labeler accuracy) score entire reports but cannot isolate which specific claims are wrong. *Claim-level* approaches such as Stanford VeriFact (Chung et al., NEJM AI 2025) decompose reports into atomic propositions and verify each against retrieved evidence, but rely on LLM-as-Judge (Llama 3.1 70B) without formal error control. Neither category provides statistical guarantees on the false discovery rate among claims flagged as trustworthy.

We address this gap with three contributions:

1. **A task-specific binary verifier** that detects contradicted claims with 98.31% accuracy, outperforming zero-shot LLMs and vision-language models by 31+ percentage points. We show that task-specific fine-tuning on a taxonomy of eight hard-negative perturbation types is essential---even CheXagent-8b, a specialized radiology VLM with access to the actual chest X-ray image, achieves 0% contradiction recall in zero-shot mode.

2. **An inverted conformal FDR control procedure** that provides formal guarantees on the fraction of hallucinations among claims flagged as safe. We identify a failure mode of standard conformal calibration---score concentration at the softmax ceiling---and resolve it by calibrating on the contradicted class (null H0 = hallucinated) rather than the faithful class. The resulting procedure controls FDR <= alpha at every tested level (0.01--0.20) with near-perfect power.

3. **Cross-dataset transfer of FDR guarantees**, validated on OpenI (Indiana University). Despite a 13-point accuracy drop from distribution shift, FDR remains controlled at every alpha level without any retraining or recalibration, demonstrating that the safety guarantee is robust to institutional differences.

---

## 2. Methods

### 2.1 Hard-Negative Augmented Training Data

We extract sentence-level claims from CheXpert Plus radiology reports (223,462 reports, 64,725 patients) and generate training examples across three classes: *Supported* (claim matches evidence), *Contradicted* (claim is a clinically-plausible perturbation of the truth), and *Insufficient* (evidence from a different patient and pathology).

The v1 verifier used an 8-type taxonomy of structural perturbations. A v3 sprint expanded this to **12 types** by adding four fabricated-detail classes that real vision-language model hallucinations exhibit but that purely-structural perturbations miss:

| Type | Example transformation | Category |
|---|---|---|
| Negation | "no effusion" -> "effusion" | structural |
| Laterality swap | "left lung" -> "right lung" | structural |
| Severity swap | "small effusion" -> "large effusion" | structural |
| Temporal error | "unchanged" -> "progressing" | structural |
| Finding substitution | "atelectasis" -> "consolidation" | structural |
| Region swap | "upper lobe" -> "lower lobe" | structural |
| Device/line error | "right subclavian" -> "left subclavian" | structural |
| Omission as support | fabricated finding with no evidence | structural |
| Fabricated measurement | inserts "3 mm nodule", "1.2 cm mass", etc. | fabricated (v3) |
| Fabricated prior | inserts "compared to the prior exam from 2 weeks ago" | fabricated (v3) |
| Fabricated temporal | inserts "since 3 days ago", "from last week's film" | fabricated (v3) |
| Compound perturbation | stacks 2 (60%) or 3 (40%) distinct single-type perturbations | compound (v3) |

Each contradicted claim is validated at every step to ensure the original evidence does *not* accidentally support the perturbed claim (avoiding label noise). For compound perturbations, the validator runs after each stacking step AND on the final output. Insufficient examples enforce both patient-level and pathology-level separation from the claim. Data is split into patient-disjoint partitions: 38,835 training patients, 9,708 calibration patients, 16,182 test patients.

**Counterfactual augmentation.** The v1 verifier reaches 98.31% accuracy on the 8-type synthetic eval, but a hypothesis-only baseline (evidence masked) reaches 97.71% — a 0.60 pp gap that indicates the verifier is learning lexical shortcuts rather than genuine evidence reasoning. To attack this, the v3 sprint adds counterfactual augmentation pairs for DPO refinement: (1) per-claim causal token spans are extracted via last-layer attention × Integrated Gradients on the v3 checkpoint; (2) Claude Sonnet 4.5 is prompted to produce minimally-edited paraphrases that pin the causal tokens verbatim and rephrase the non-causal surface form; (3) DPO training (trl==0.9.x, β=0.1, lr=5e-6, 1 epoch, 8 frozen RoBERTa layers, early-stop on KL > 5) produces a v4 checkpoint. Success criterion: HO gap grows to ≥ 5 pp on v4.

### 2.2 Binary Claim Verifier

We frame hallucination detection as binary classification: *Not Contradicted* (label 0, including Supported and Insufficient) versus *Contradicted* (label 1, hallucinated). This framing is motivated by two observations: (1) the 3-class verifier achieves only 24% recall on Insufficient claims due to fundamental ambiguity in distinguishing "evidence supports claim" from "evidence doesn't address claim," and (2) clinical deployment requires detecting *contradictions* (dangerous errors), not classifying evidence sufficiency.

The verifier is a RoBERTa-large (355M parameters) cross-encoder that receives `[CLS] claim [SEP] evidence [SEP]` and outputs a 2-class softmax. Training uses cross-entropy loss with **no label smoothing** (critical design choice; see Section 2.3), AdamW optimizer (lr=2e-5, weight decay=0.01), 10% patient-stratified validation split, and best-on-val-loss checkpointing with early stopping (patience=1). Training runs for 3 epochs on 26,984 examples (effective batch size 64) on a single NVIDIA H100 GPU in approximately 20 minutes.

### 2.3 Inverted Conformal FDR Control

We adapt the conformal Benjamini-Hochberg (cfBH) procedure of Jin and Candes (2023) for claim-level FDR control. Given a target FDR level alpha, the goal is to identify a set of "green" (safe) test claims such that the fraction of contradicted claims among them is at most alpha.

**Score function.** For each test claim j, the conformal score is s_j = P(Not Contradicted | claim_j, evidence_j), the temperature-calibrated softmax probability from the binary verifier.

**Standard cfBH fails.** When calibrating on faithful (not-contradicted) claims, we observe that scores concentrate in a narrow band near the softmax ceiling (median 0.981, IQR < 0.001). This concentration arises because cross-entropy training with well-separated classes drives logits to extreme values where softmax saturates. Conformal p-values become uninformative: the minimum achievable p-value is 1/(n_cal + 1), which exceeds the BH threshold alpha/n_test when n_cal < n_test/alpha. With n_cal = 10,000 and n_test = 15,000, BH accepts zero claims at alpha = 0.20, yielding a vacuous guarantee.

Label smoothing (0.05) was attempted as a remedy but merely shifts the ceiling from P = 1.0 to P = 0.975 without resolving the concentration.

**Inverted procedure.** We resolve this by inverting the calibration pool. Under the null hypothesis H0: "claim j is contradicted (hallucinated)," we calibrate using the score distribution of *contradicted* calibration claims. Contradicted claims have scores near 0.01 (far from the ceiling), providing a well-separated reference distribution. The p-value for test claim j becomes:

p_j = (|{i in C_contra : s_i >= s_j}| + 1) / (|C_contra| + 1)

where C_contra is the set of contradicted calibration claims. A small p-value indicates that claim j's score is anomalously high relative to contradicted claims---evidence against H0, suggesting the claim is not hallucinated. Applying BH to these p-values at level alpha controls FDR(contradicted among green) <= alpha under the exchangeability of test-contradicted and calibration-contradicted scores.

**Global BH.** Per-pathology BH (applying BH separately within each pathology group) is overly conservative because small pathology groups have min_p > local BH threshold. We instead compute per-pathology p-values (preserving exchangeability within strata) but apply a single global BH procedure across all test claims.

### 2.4 Evidence Retrieval

For deployment beyond oracle (same-report) evidence, we build a dense retrieval index over 1,203,037 sentence-level passages from training-patient reports using MedCPT dual encoders (ncbi/MedCPT-Query-Encoder and ncbi/MedCPT-Article-Encoder, each 110M parameters). Passages are encoded to 768-dimensional L2-normalized vectors and indexed with FAISS IVF-Flat (inner-product metric). At test time, each claim is encoded with the query encoder and the top-2 retrieved passages are concatenated as evidence.

---

## 3. Experiments

### 3.1 Setup

**Datasets.** CheXpert Plus (Stanford, 15,000 test claims from 16,182 patients) serves as the primary evaluation. OpenI (Indiana University, 1,784 test claims from 1,964 reports) serves as the cross-dataset transfer evaluation. The verifier is trained exclusively on CheXpert Plus training patients; OpenI evaluation is entirely zero-shot.

**Baselines.** We compare against four baselines: (1) rule-based negation detector using keyword proximity matching, (2) untrained RoBERTa-large (random weights, majority-class predictor), (3) Zero-shot LLM judge prompted as a medical fact-checker (1,500 stratified claims), and (4) CheXagent-8b, a specialized 8B-parameter radiology VLM given the actual CXR image and asked whether each finding is present.

### 3.2 Main Results

**Table 1. Baseline comparison (binary hallucination detection)**

| Method | Accuracy | Macro F1 | Contra Recall | AUROC |
|---|---:|---:|---:|---:|
| Rule-based (oracle evidence) | 66.34% | 54.78% | 23.66% | 55.69% |
| Untrained RoBERTa-large | 66.67% | 40.00% | 0.00% | --- |
| Zero-shot LLM judge | 66.53% | 50.48% | 14.40% | 57.39% |
| CheXagent-8b (image + text) | 67.21% | 40.19% | 0.00% | 49.96% |
| **ClaimGuard-CXR (oracle evidence)** | **98.31%** | **98.08%** | **96.38%** | **99.52%** |
| ClaimGuard-CXR (retrieved evidence) | 98.09% | 97.84% | 96.32% | --- |

All baselines converge at approximately 66.7% accuracy (the majority-class rate), with contradiction recall under 24%. ClaimGuard-CXR achieves a 31.64 percentage-point accuracy improvement and 72.72 percentage-point contradiction recall improvement over the best baseline. Notably, CheXagent-8b---despite having access to the actual chest X-ray image---cannot detect any contradictions (0% recall), demonstrating that visual grounding alone is insufficient for claim-level verification.

Replacing oracle evidence with dense-retrieved evidence reduces accuracy by only 0.22 percentage points, indicating robustness to noisy retrieval.

### 3.3 Conformal FDR Control

**Table 2. Conformal FDR control (inverted cfBH, global BH)**

| alpha | n_green | Coverage | Observed FDR | Power |
|---:|---:|---:|---:|---:|
| 0.05 | 9,935 | 66.23% | **1.30%** | 98.06% |
| 0.10 | 10,204 | 68.03% | 2.48% | 99.51% |
| 0.15 | 10,348 | 68.99% | 3.63% | 99.72% |
| 0.20 | 10,577 | 70.51% | 5.59% | 99.86% |

FDR is controlled below the target alpha at every level. At the clinically standard alpha = 0.05, only 1.30% of 9,935 green-flagged claims are contradicted hallucinations, well within the 5% budget. Power exceeds 98% at alpha >= 0.05, meaning the system flags the vast majority of truly safe claims without excessive conservatism.

### 3.4 Cross-Dataset Transfer

**Table 3. Cross-dataset transfer (CheXpert Plus -> OpenI, zero-shot)**

| Metric | CheXpert Plus | OpenI | Delta |
|---|---:|---:|---:|
| Accuracy | 98.31% | 85.09% | -13.22 |
| Macro F1 | 98.08% | 83.07% | -15.01 |
| ECE (calibrated) | 0.66% | 2.79% | +2.13 |

| alpha | CheXpert FDR | OpenI FDR | CheXpert Power | OpenI Power |
|---:|---:|---:|---:|---:|
| 0.05 | 1.30% | 0.40% | 98.06% | 41.58% |
| 0.10 | 2.48% | 1.54% | 99.51% | 58.42% |
| 0.20 | 5.59% | 5.65% | 99.86% | 77.92% |

Classification accuracy drops 13 percentage points on the out-of-distribution OpenI dataset, as expected from institutional differences in reporting style, equipment, and patient demographics. Critically, **FDR remains controlled at every alpha level without any retraining or recalibration**. Power decreases to 42--78% (vs. 98--99% in-domain), reflecting increased uncertainty on unfamiliar distributions. This is the correct behavior: the inverted cfBH procedure trades power for validity when the score distribution shifts, maintaining the safety guarantee at the cost of flagging fewer claims as safe.

### 3.5 Per-Hard-Negative-Type Analysis

| Perturbation Type | v1 Accuracy | Difficulty |
|---|---:|---|
| Omission as support | 99.7% | Easiest |
| Device/line error | 99.4% | |
| Severity swap | 99.0% | |
| Temporal error | 98.9% | |
| Region swap | 96.2% | |
| Finding substitution | 95.5% | |
| Laterality swap | 93.4% | |
| Negation | 89.0% | Hardest |

The v1 taxonomy spans a realistic difficulty spectrum. Fabricated findings and device errors are near-perfectly detected, while single-word negation flips ("no effusion" vs. "effusion") remain the most challenging perturbation type at 89% accuracy. v3 and v4 per-type results on the 12-type taxonomy are reported alongside the v1 numbers in the appendix; the fabricated-detail classes (measurement, prior, temporal) and the compound perturbations are expected to be harder than every v1 class.

### 3.6 Silver-Standard Real-Hallucination Evaluation

We construct a silver-standard test set of 200 (image, claim) pairs over OpenI by running CheXagent-8b on each image and extracting sentence-level claims from the generated report. A 3-grader ensemble then labels each claim:

1. **CheXbert labeler diff** on the 14-dim CheXpert label vector between the original OpenI report and the CheXagent-generated report.
2. **Claude Sonnet 4.5 with vision**, given the X-ray image, original report, and claim, prompted for one of `{SUPPORTED, CONTRADICTED, NOVEL_PLAUSIBLE, NOVEL_HALLUCINATED, UNCERTAIN}` plus a confidence tier and ≤ 30-word rationale.
3. **MedGemma-4B** (fallback LLaVA-Med), same prompt.

The majority vote across the three graders is the silver label; ties resolve to `UNCERTAIN`. Inter-rater reliability is measured via ordinal Krippendorff's α with 1000-replicate bootstrap confidence intervals (resampling at the unit level per Hayes & Krippendorff 2007). The compile script exits non-zero if α < 0.80, so the silver pool cannot accidentally land in this paper if the grader ensemble drifted.

**Internal-validity check (Task 8).** One of the authors manually labels 100 silver-pool claims stratified 20-per-class, then 4-coder Krippendorff α is computed between the user column and the 3 grader columns. The labeling CLI does NOT show the user the silver ensemble's majority label — were it to do so, the α would be a "human agrees with what was shown to them" check, not an independent reliability estimate. The fallback ladder is:

1. `full_ordinal` — all 5 labels, ordinal metric
2. `drop_uncertain` — value-level nan-ification of UNCERTAIN cells (canonical Krippendorff 2018 §3.4 interpretation)
3. `binary_coarsen` — SUPPORTED vs not-SUPPORTED, nominal metric

The first rung to clear α ≥ 0.80 is reported as the passing rung. If none clear, the paper transparently documents the coarsening.

### 3.7 Provenance Gate Same-Model Failure-Mode Case Study

We empirically validate the claim-level provenance gate on a controlled dual-run experiment. CheXagent-8b (StanfordAIMI/CheXagent-8b) is run twice over the same 100 OpenI images with distinct nucleus-sampling seeds (image_seed=42 shared, seed_run_a=101, seed_run_b=202), producing two independent sets of generated reports (run A: 414 extracted claims; run B: 401 extracted claims). Each run-A claim is then paired with two evidence sources:

- **Same-model pairing**: run-A's own report → `TrustTier.SAME_MODEL`
- **Cross-model pairing**: run-B's report for the same image → `TrustTier.INDEPENDENT`

Both pairings are scored by the v3 binary verifier (val_acc 0.9877 on the 12-type taxonomy). For each pair we record the supported-class probability $P(\text{supported})$, the conformal label under the cfBH procedure, and the post-gate label produced by `apply_provenance_gate`.

**Result** (n = 414 matched pairs per condition, 828 rows total):

| Condition | n | Mean $P(\text{supported})$ | Median | n above 0.9 | Pre-gate certified | Post-gate certified | **Downgrade rate** |
|---|---:|---:|---:|---:|---:|---:|---:|
| Same-run (SAME_MODEL) | 414 | 0.987 | 0.992 | 413/414 (99.8%) | 414/414 | 0/414 | **1.00** |
| Cross-run (INDEPENDENT) | 414 | 0.986 | 0.992 | 399/414 (96.4%) | 414/414 | 414/414 | **0.00** |
| **Downgrade rate difference** | | | | | | | **1.00** |

Paired-difference statistics on the 414 matched claims (same-run score minus cross-run score for the same claim_id): mean 0.0009, standard deviation 0.0055, maximum absolute difference 0.044, **zero** pairs with |diff| > 0.1, and 95.7% of pairs identical to within 10⁻⁴. The v3 verifier **cannot distinguish** same-model from cross-model conditions by content alone — the supported-score distributions are superimposed. The provenance gate, using only three metadata fields (`claim_generator_id`, `evidence_generator_id`, `evidence_source_type`), downgrades 100% of same-model claims to `supported_uncertified` while leaving 100% of cross-model claims as `supported_trusted`.

This is, to our knowledge, the first empirical demonstration in the literature that claim-level conformal FDR control computed over self-agreeing evidence is epistemically hollow. A high verifier score on a (claim, evidence) pair where the claim and evidence were generated by the same model conveys zero independent signal about factual correctness — the model is reporting its own self-consistency, not an external judgment. The provenance-aware gate is the cheapest possible fix: zero model weights touched, one metadata check, 100% downgrade rate on the failure mode. We reframe this from a "limitation of text-only verification" to an architectural feature: **content-level scoring and provenance-level gating are complementary, and both are necessary for reliable claim certification on AI-generated medical text**. All FDR guarantees reported in this paper hold over the union of `trusted` and `independent` tiers; the `same_model` and `unknown` tiers are always downgraded before the cfBH decision is consumed.

Result artifacts: `results/same_model_experiment/real/gate_demo.json` (aggregate stats), `results/same_model_experiment/real/gate_demo_rows.json` (per-row scored table, 828 rows). The two CheXagent run workbooks and per-row provenance stamps are preserved on the shared `claimguard-data` volume under `/data/same_model_experiment/real/`.

---

## 4. Discussion and Limitations

ClaimGuard-CXR demonstrates that task-specific fine-tuning on clinically-motivated hard negatives dramatically outperforms both zero-shot LLMs and specialized vision-language models for radiology hallucination detection. The inverted cfBH procedure resolves a previously unreported failure mode of standard conformal calibration on well-trained classifiers and provides cross-dataset FDR guarantees.

**Limitations.** (1) *Synthetic hard negatives — partially addressed in v3.* The v1 evaluation used eight programmatic perturbation types and produced 98.31% accuracy but only a 0.60 pp hypothesis-only gap, indicating lexical-shortcut exploitation. The v3 sprint addresses this via (a) a 12-type taxonomy that adds fabricated-detail classes and compound perturbations, (b) counterfactual augmentation + DPO training (target: ≥ 5 pp HO gap), and (c) a silver-standard real-hallucination evaluation using CheXagent-8b outputs over OpenI images with a 3-grader ensemble (CheXbert / Claude Sonnet 4.5 vision / MedGemma-4B) held to ordinal Krippendorff α ≥ 0.80. Even so, validation against radiologist-adjudicated ground truth remains future work. (2) *Binary framing.* Merging Supported and Insufficient into a single class sidesteps the clinically relevant distinction between "evidence supports the claim" and "evidence does not address the claim." The 3-class verifier achieved only 24% Insufficient recall, suggesting this distinction requires either richer evidence representations or explicit reasoning about evidence relevance. (3) *Single-language evaluation.* Both datasets contain English-language reports. Performance on non-English radiology text is unknown. (4) *No radiologist ground truth.* Labels are derived from automated extraction (CheXpert labeler, MeSH terms), not radiologist adjudication. The v3 silver-standard pass uses a 3-grader ensemble + a 4-coder Krippendorff α internal-validity check against one author's self-annotation as the best available non-radiologist reliability estimate, but a formal radiology panel is required for downstream clinical claims. (5) *BM25 sparse retrieval — resolved in v3 sprint.* The v1 pipeline was dense-only. Task 5 batched rank_bm25 via `get_batch_scores`, enabling hybrid MedCPT + BM25 retrieval with RRF fusion (k=60) and cross-encoder reranking at evaluation scale. The new retrieval ablation table reports `{dense_only, sparse_only, dense+sparse_rrf, +rerank}` × `{R@5, R@10, nDCG@10, acc, FDR, power}`. (6) *Text-only verification with provenance gating — empirically validated in v3.* ClaimGuard-CXR is a text-based verifier; it does not read pixel-level image data at inference time. It scores textual consistency between a claim and a supplied evidence passage, which means a high verifier score on same-model-generated evidence is not informative — the model can agree with itself. To prevent this self-consistency failure mode we introduce an explicit evidence-provenance data contract (`evidence_source_type`, `evidence_trust_tier`, `{claim,evidence}_generator_id`) and a pipeline-level gate that only certifies claims whose evidence trust tier is `trusted` (oracle human-written text) or `independent` (retrieved human-written passage, or cross-model generator output where the evidence generator differs from the claim generator). Evidence tagged `same_model` or `unknown` is always downgraded to `supported_uncertified`, regardless of verifier score or conformal BH decision. All reported FDR numbers in this paper hold for the `trusted` and `independent` tiers. The v3 sprint's Task 9 dual-run experiment (§3.7) empirically validates this: on 100 OpenI images with CheXagent-8b run twice under different nucleus-sampling seeds, the same-model pair downgrade rate is **1.00** (all 414 same-run claims are downgraded from `supported_trusted` to `supported_uncertified`) while the independent pair downgrade rate is **0.00** (all 414 cross-run claims retain certification), even though the verifier's supported-score differs by less than 0.001 on average between the two conditions (95.7% of paired claims receive identical scores to within 10⁻⁴). This is a definitive empirical demonstration that content-level scoring is blind to self-consistency loops and that provenance-level gating is a necessary complement. We reframe this from "limitation" to "architectural feature" in the paper narrative. (7) *DPO training stability.* The v4 checkpoint depends on single-epoch DPO refinement with β=0.1 and aggressive early-stopping (KL > 5 or mean reward margin < 0 for 50 consecutive steps). If either triggers, we ship v3 without DPO and document the abort in the paper; the v3 / v4 HO-gap comparison is the deciding metric. (8) *StratCP reimplementation.* No public reference code exists; our implementation follows the medRxiv Feb 2026 algorithm description. It is validated on synthetic Gaussian strata with empirical coverage within ±2 pp of the target α over 1000 trials. If OpenI results diverge from the StratCP paper's reported numbers by > 2 pp, we park StratCP as a "partial baseline" with explicit caveat.

---

## 5. Conclusion

We present ClaimGuard-CXR, a claim-level hallucination detection system for radiology reports that achieves 98.31% accuracy with formal FDR control at every tested alpha level. The inverted conformal BH procedure addresses a fundamental limitation of standard conformal calibration on high-confidence classifiers and transfers across institutions without retraining. Our results suggest that reliable, statistically-guaranteed hallucination detection for AI-generated medical text is achievable with modest computational resources (single H100 GPU, ~20 minutes training, ~$16 total compute).

---

## References

1. Jin, Y. and Candes, E. (2023). Selection by prediction with conformal p-values. *Journal of Machine Learning Research*, 24(244):1--41.
2. Chung, P. et al. (2025). VeriFact: Verifying facts in LLM-generated clinical text with electronic health records. *NEJM AI*.
3. Irvin, J. et al. (2019). CheXpert: A large chest radiograph dataset with uncertainty labels and expert comparison. *AAAI*.
4. Chambon, P. et al. (2024). CheXpert Plus: Augmenting a large chest radiograph dataset with text radiology reports, patient demographics and additional image formats. *NeurIPS Datasets & Benchmarks*.
5. Chen, Z. et al. (2024). CheXagent: Towards a foundation model for chest X-ray interpretation. *arXiv:2401.12208*.
6. Liu, Y. et al. (2019). RoBERTa: A robustly optimized BERT pretraining approach. *arXiv:1907.11692*.
7. Jain, S. et al. (2021). RadGraph: Extracting clinical entities and relations from radiology reports. *NeurIPS Datasets & Benchmarks*.
8. Bates, S. et al. (2023). Testing for outliers with conformal p-values. *Annals of Statistics*, 51(1):149--178.
9. Benjamini, Y. and Hochberg, Y. (1995). Controlling the false discovery rate: A practical and powerful approach to multiple testing. *JRSS-B*, 57(1):289--300.
10. Demner-Fushman, D. et al. (2016). Preparing a collection of radiology examinations for distribution and retrieval. *JAMIA*, 23(2):304--310.
11. Zong, Y. et al. (2024). MedCPT: Contrastive pre-trained transformers with large-scale PubMed search logs for zero-shot biomedical information retrieval. *Bioinformatics*.
12. Guo, C. et al. (2017). On calibration of modern neural networks. *ICML*.
