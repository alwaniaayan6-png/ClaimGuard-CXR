# ClaimGuard-CXR: Calibrated Claim-Level Hallucination Detection for Radiology Reports with Conformal FDR Control

**Aayan Alwani**

---

## Abstract

AI systems for radiology report generation frequently hallucinate---producing findings that contradict the underlying imaging study. Existing verification approaches lack formal guarantees on error rates, limiting clinical trust. We present ClaimGuard-CXR, a claim-level hallucination detection system for chest X-ray reports that provides provable false discovery rate (FDR) control. Our approach combines a RoBERTa-large binary cross-encoder, fine-tuned on 30,000 claims with eight clinically-motivated hard-negative perturbation types, with a novel *inverted* conformal Benjamini-Hochberg (cfBH) procedure that calibrates on contradicted claims rather than faithful ones. On CheXpert Plus (15,000 held-out test claims from 16,182 patients), ClaimGuard-CXR achieves 98.31% accuracy and 99.52% AUROC, outperforming Zero-shot LLM judge (66.5%), CheXagent-8b with image access (67.2%), and rule-based baselines (66.3%) by over 31 percentage points. The inverted cfBH procedure controls FDR at 1.30% (target alpha=0.05) with 98.06% power. On OpenI (Indiana University, zero-shot cross-dataset transfer), FDR remains controlled at every alpha level despite a 13-point accuracy drop, demonstrating that the safety guarantee generalizes across institutions without retraining.

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

Contradicted claims are generated via eight perturbation types designed to cover the taxonomy of real radiology hallucinations:

| Type | Example transformation | Frequency |
|---|---|---|
| Negation | "no effusion" -> "effusion" | 12.5% |
| Laterality swap | "left lung" -> "right lung" | 12.5% |
| Severity swap | "small effusion" -> "large effusion" | 12.5% |
| Temporal error | "unchanged" -> "progressing" | 12.5% |
| Finding substitution | "atelectasis" -> "consolidation" | 12.5% |
| Region swap | "upper lobe" -> "lower lobe" | 12.5% |
| Device/line error | "right subclavian" -> "left subclavian" | 12.5% |
| Omission as support | fabricated finding with no evidence | 12.5% |

Each contradicted claim is validated to ensure the original evidence does *not* accidentally support the perturbed claim (avoiding label noise). Insufficient examples enforce both patient-level and pathology-level separation from the claim. Data is split into patient-disjoint partitions: 38,835 training patients, 9,708 calibration patients, 16,182 test patients.

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

| Perturbation Type | Accuracy | Difficulty |
|---|---:|---|
| Omission as support | 99.7% | Easiest |
| Device/line error | 99.4% | |
| Severity swap | 99.0% | |
| Temporal error | 98.9% | |
| Region swap | 96.2% | |
| Finding substitution | 95.5% | |
| Laterality swap | 93.4% | |
| Negation | 89.0% | Hardest |

The taxonomy spans a realistic difficulty spectrum. Fabricated findings and device errors are near-perfectly detected, while single-word negation flips ("no effusion" vs. "effusion") remain the most challenging perturbation type at 89% accuracy.

---

## 4. Discussion and Limitations

ClaimGuard-CXR demonstrates that task-specific fine-tuning on clinically-motivated hard negatives dramatically outperforms both zero-shot LLMs and specialized vision-language models for radiology hallucination detection. The inverted cfBH procedure resolves a previously unreported failure mode of standard conformal calibration on well-trained classifiers and provides cross-dataset FDR guarantees.

**Limitations.** (1) *Synthetic hard negatives.* Our evaluation uses programmatically generated contradictions. While the eight perturbation types are designed to cover the taxonomy of real radiology errors, validation on actual generator hallucinations (e.g., from CheXagent or RadFM) would strengthen the generalizability claim. (2) *Binary framing.* Merging Supported and Insufficient into a single class sidesteps the clinically relevant distinction between "evidence supports the claim" and "evidence does not address the claim." The 3-class verifier achieved only 24% Insufficient recall, suggesting this distinction requires either richer evidence representations or explicit reasoning about evidence relevance. (3) *Single-language evaluation.* Both datasets contain English-language reports. Performance on non-English radiology text is unknown. (4) *No radiologist ground truth.* Labels are derived from automated extraction (CheXpert labeler, MeSH terms), not radiologist adjudication. (5) *BM25 sparse retrieval deferred.* The per-query cost of rank_bm25 on 1.2M passages (~2s) was infeasible at evaluation scale; only dense MedCPT retrieval was used.

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
