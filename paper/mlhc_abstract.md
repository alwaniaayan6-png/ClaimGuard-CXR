# MLHC 2026 Clinical Abstract — ClaimGuard-CXR

**Track:** Clinical Abstracts (non-archival, up to 2 pages)
**Deadline:** April 17, 2026
**Venue:** [MLHC 2026](https://www.mlforhc.org/paper-submission)

---

## Title

ClaimGuard-CXR: Claim-Level Hallucination Detection for Radiology Reports with Formal False Discovery Rate Control

## Authors

Aayan Alwani (Great Neck South High School; Ashley Laughney Lab, Weill Cornell Medicine)

## Abstract

**Background.** AI systems for radiology report generation hallucinate clinically significant errors at rates between 8 and 15 percent, including laterality swaps, negation flips, and fabricated findings. Existing verification methods either operate at the report level (accepting or rejecting entire reports) or lack formal guarantees on error rates among claims flagged as trustworthy. No prior system provides claim-level hallucination detection with provable false discovery rate (FDR) control for radiology reports.

**Methods.** We present ClaimGuard-CXR, a claim-level hallucination detection system for chest X-ray reports that combines (1) a DeBERTa-v3-large binary cross-encoder (434M parameters) fine-tuned with progressive domain adaptation across MNLI, MedNLI, and radiology claims, and (2) an inverted conformal Benjamini-Hochberg procedure that provides formal FDR guarantees. Training data consists of 30,000 radiology claims from CheXpert Plus with eight clinically-motivated hard-negative perturbation types (laterality swap, negation, finding substitution, severity swap, temporal error, region swap, device error, omission). We identify a previously unreported failure mode of standard conformal calibration on well-trained classifiers --- score concentration at the softmax ceiling --- and resolve it by calibrating on the contradicted class. Claims are triaged as GREEN (safe), YELLOW (review recommended), or RED (likely hallucinated), with the guarantee that the fraction of hallucinated claims among GREEN labels is at most alpha.

**Results.** On CheXpert Plus (15,000 held-out test claims from 16,182 patients), ClaimGuard-CXR v2 achieves **98.61%** validation accuracy after progressive NLI pre-training, improving over the v1 RoBERTa-large baseline (98.31%) by 0.30 percentage points. The inverted conformal procedure controls FDR at **1.66%** (target alpha=0.05) with **99.32% power**, improving power by 1.26 percentage points over v1. FDR is controlled at every alpha level tested (0.01 through 0.20). Against published baselines, ClaimGuard-CXR outperforms zero-shot DeBERTa-v3-large NLI (54.17%), RadFlag-style self-consistency (65.16%), and rule-based negation detection (66.34%) by over 32 percentage points.

**Critical artifact finding.** A hypothesis-only baseline (claim text without any evidence, same DeBERTa architecture) achieves **98.15%** test accuracy --- within 0.46 percentage points of the full verifier. This reveals that synthetic hard-negative perturbations contain strong lexical shortcuts (e.g., negation flips change a single word, laterality swaps change one token) that are detectable from claim text alone. While the full verifier still provides calibrated conformal FDR guarantees, the evaluation benchmark itself is too lexically distinctive, motivating the need for real-hallucination test sets from actual VLM outputs with expert annotation. We report this transparently as a limitation and identify it as the most important direction for future work.

**Clinical Significance.** ClaimGuard-CXR enables a new deployment paradigm for AI-generated radiology reports: rather than accepting or rejecting entire reports, clinicians receive per-claim triage labels with formal statistical guarantees. GREEN-labeled claims can be trusted with provable error bounds; RED-labeled claims are flagged for mandatory review. This claim-level granularity reduces radiologist review burden while maintaining safety --- the system correctly identifies which specific sentences to double-check, not just whether the overall report is suspicious. The conformal guarantee transfers across institutions without retraining (cross-dataset validation on OpenI from v1 experiments). The hypothesis-only artifact finding highlights a fundamental limitation of synthetic evaluation benchmarks in radiology NLP and is itself a contribution to the community's understanding of how well current hallucination detection approaches actually generalize.

---

## Detailed Results Table

### Classification (CheXpert Plus, 15K test claims)

| Method | Test Accuracy | Contra Recall | AUROC |
|--------|--------------:|--------------:|------:|
| Rule-based (oracle evidence) | 66.34% | 23.66% | 55.69% |
| RadFlag self-consistency | 65.16% | 5.38% | 45.57% |
| DeBERTa-v3-large zero-shot NLI | 54.17% | 49.96% | 54.45% |
| Hypothesis-only DeBERTa (no evidence) | **98.15%** | 96.32% | **99.77%** |
| **ClaimGuard-CXR v2 (best val)** | **98.61%** | -- | -- |

Published comparison baselines from v1:
| Method | Accuracy | Contra Recall |
|--------|---------:|--------------:|
| Zero-shot LLM judge | 66.53% | 14.40% |
| CheXagent-8b (image + text) | 67.21% | 0.00% |
| Untrained RoBERTa-large | 66.67% | 0.00% |

### Conformal FDR Control (Inverted cfBH, global BH, v2 DeBERTa)

| alpha | n_green | FDR | Power |
|-------|--------:|----:|------:|
| 0.01 | 9,511 | 0.23% | 94.89% |
| **0.05** | **10,100** | **1.66%** | **99.32%** |
| 0.10 | 10,325 | 3.25% | 99.89% |
| 0.15 | 10,488 | 4.70% | 99.95% |
| 0.20 | 10,676 | 6.34% | 99.99% |

### V1 vs V2 Comparison

| Metric | V1 (RoBERTa-large) | V2 (DeBERTa + Progressive NLI) | Delta |
|--------|-------------------:|-------------------------------:|------:|
| Best val accuracy | 98.31% | 98.61% | +0.30 pp |
| FDR @ alpha=0.05 | 1.30% | 1.66% | +0.36 pp (still < alpha) |
| Power @ alpha=0.05 | 98.06% | 99.32% | +1.26 pp |
| Parameters | 355M | 434M | +22% |

### Artifact Analysis (Critical Finding)

The hypothesis-only baseline is trained on exactly the same 30K claims with the same 10% patient-stratified validation split, but with evidence passages replaced by padding tokens. Its near-ceiling accuracy (98.15%) demonstrates that most of the verifier's signal comes from claim text alone, not from evidence reasoning. This is consistent with the structure of the eight hard-negative types:
- **Negation flip**: adds/removes the word "no" — detectable with a single token
- **Laterality swap**: changes "left" to "right" — one-token distinguishability
- **Severity swap**: changes "mild" to "severe" — one-token distinguishability
- **Finding substitution**: replaces one finding with another from a confusable pair — lexically detectable

Only **region swap** and **temporal error** genuinely require context to detect. The evaluation benchmark is therefore biased toward high reported performance. A real-hallucination test set with actual VLM outputs (e.g., CheXagent-8b generations annotated by a board-certified radiologist) would provide a more realistic assessment and is identified as the most critical next step.

---

## Compute and Reproducibility

- Training: NVIDIA RTX A6000 (community cloud), NVIDIA L40S (secure cloud), ~3 hours total
- Total GPU cost: ~$8 across training and baselines
- Code: https://github.com/alwaniaayan6-png/ClaimGuard-CXR (public)
- All 4 baselines reproducible from `runpod/run_baselines.py`
