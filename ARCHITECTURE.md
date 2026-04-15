# ClaimGuard-CXR: Technical Architecture Document

**Version:** 2.1 (post-audit bug fixes, scope reduction for NeurIPS 2026)
**Target:** NeurIPS 2026 main conference
**Submission date:** May 6, 2026

---

## 🔖 SCOPE NOTE FOR NEURIPS 2026 SUBMISSION (Added v2.1, 2026-04-04)

After a deep code + data audit, the following scope reduction applies to the
NeurIPS 2026 submission:

**IN SCOPE (primary contributions):**
- **Component 5** — Evidence-Conditioned Claim Verifier (**Contribution 1**)
  - Text-only RoBERTa-large cross-encoder on (claim, evidence) pairs
  - Trained with 8-type hard-negative taxonomy (v2 data, post-C5/C6 fixes)
  - Patient-stratified 10% val split + label-smoothed CE + best-on-val-loss
  - Temperature-scaled for calibration
- **Component 4** — Evidence Retriever (hybrid MedCPT + BM25 + reranker) feeds
  evidence into the verifier.
- **Component 7** — Conformal Claim Triage (**Contribution 3**)
  - Pathology-stratified cfBH with one-per-patient subsampling
  - ICC exchangeability diagnostic, temporal-shift sanity check
  - Label-conditional (calibration on faithful scores only, per Jin & Candes)

**DEFERRED to future work (not in May 6 submission):**
- **Component 1** (Generator): no CXR images available locally; no Modal
  training script; cannot be trained in the 32-day window.
- **Component 3** (Grounding): depends on a trained generator's cross-attention.
  During v1 + v2 verifier training, the CNN heatmap encoder receives
  all-zero inputs — the vision branch learns nothing. Paper must NOT claim
  multimodal verification.
- **Component 6** (Best-of-N Selection, formerly Contribution 2): requires a
  trained generator to produce N candidate reports. Moved to future work.
- **Component 2** (Claim Decomposer) is partially scoped: we use a rule-based
  sentence-splitter fallback in v2. Phi-3 LoRA decomposer is a stretch goal.

**HONEST LIMITATIONS** (will be prominent in paper):
- Verifier is text-only; the vision branch is unused in evaluation
- Ground-truth labels come from CheXbert (~13% label noise)
- Conformal guarantees hold per-pathology-group, not globally
- Exchangeability assumes in-distribution test data

The rest of this document preserves the full multimodal design for future work.
Sections 2, 4 (grounding), and 7 (BoN) are marked FUTURE WORK.

---

## Table of Contents
1. [System Overview](#1-system-overview)
2. [Component 1: Generator](#2-component-1-generator)
3. [Component 2: Claim Decomposer](#3-component-2-claim-decomposer)
4. [Component 3: Visual Evidence Grounding](#4-component-3-visual-evidence-grounding)
5. [Component 4: Evidence Retriever](#5-component-4-evidence-retriever)
6. [Component 5: Evidence-Conditioned Claim Verifier](#6-component-5-claim-verifier)
7. [Component 6: Best-of-N Selection](#7-component-6-best-of-n-selection)
8. [Component 7: Conformal Claim Triage](#8-component-7-conformal-claim-triage)
9. [Inference Pipeline](#9-inference-pipeline)
10. [Training Dependency Graph](#10-training-dependency-graph)
11. [Data Splits and Isolation](#11-data-splits)
12. [Conformal Calibration Mathematics](#12-conformal-math)

---

## 1. System Overview

```
                          INPUT: Chest X-ray (384x384 RGB)
                                      |
                                      v
                    ┌─────────────────────────────────┐
                    │    COMPONENT 1: GENERATOR        │
                    │  RadJEPA ViT-B/14 (frozen+adapt) │
                    │  + Phi-3-mini (LoRA, RadPhi-3)   │
                    │  Nucleus sampling, N=4-8 reports  │
                    └────────────┬────────────────────┘
                                 │ N candidate reports
                                 v
                    ┌─────────────────────────────────┐
                    │  COMPONENT 2: CLAIM DECOMPOSER   │
                    │  Phi-3-mini (LoRA on RadGraph-XL) │
                    │  -> atomic claims + pathology cat  │
                    └────────────┬────────────────────┘
                                 │ claims per report
                          ┌──────┴──────┐
                          v              v
            ┌──────────────────┐  ┌─────���────────────┐
            │ COMP 3: GROUNDING│  │ COMP 4: RETRIEVER│
            │ Cross-attn on    │  │ MedCPT + BM25    │
            │ 27x27 ViT feats  │  │ + DeBERTa rerank │
            │ -> heatmap       │  │ -> top-2 passages│
            └────────┬─────────┘  └────────┬���────────┘
                     │ region descriptions  │ evidence text
                     └──────────┬───────────┘
                                v
                    ┌───��─────────────────────────────┐
                    │  COMPONENT 5: CLAIM VERIFIER     │
                    │  RoBERTa-large cross-encoder  │
                    │  [claim; SEP; evid1; SEP; evid2] │
                    │  -> verdict (3-class) + score s   │
                    └────────────┬─���──────────────────┘
                                 │ scores per claim per report
                                 v
                    ┌─────────────────────────────────┐
                    │  COMPONENT 6: BEST-OF-N SELECT   │
                    │  max coverage s.t. faith >= tau   │
                    │  Naive BoN or MBR-BoN            │
                    └��───────────┬────────────────────┘
                                 │ best report with scored claims
                                 v
                    ┌─────────────────────────────────┐
                    │  COMPONENT 7: CONFORMAL TRIAGE   │
                    │  cfBH procedure (Jin & Candes)   │
                    │  Pathology-stratified thresholds  │
                    │  -> GREEN / YELLOW / RED labels   │
                    └─���───────────────────────────────┘
                                 │
                                 v
                    OUTPUT: Best report with per-claim
                    triage labels + confidence scores
```

---

## 2. Component 1: Generator   **[FUTURE WORK — NOT IN NEURIPS 2026 SUBMISSION]**

### Vision Encoder: RadJEPA ViT-B/14 (Primary)

| Attribute | Value |
|-----------|-------|
| Architecture | Vision Transformer (ViT-B/14) |
| Parameters | 86M (ALL FROZEN) |
| Hidden dimension | 768 |
| Transformer layers | 12 |
| Attention heads | 12 |
| Patch size | 14x14 pixels |
| Input resolution | **384x384** RGB |
| Spatial feature map | 27x27x768 (384/14 ≈ 27) |
| [CLS] token output | 1x768 |
| Pretraining | I-JEPA self-supervised on 839K unlabeled CXR images |
| Weights | HuggingFace: AIDElab-IITBombay/RadJEPA |
| Fine-tuning | NONE — frozen encoder only |

**Adapter layers (trainable):**
| Attribute | Value |
|-----------|-------|
| Type | Bottleneck adapter after each ViT block |
| Down projection | Linear(768 → 32) |
| Activation | ReLU |
| Up projection | Linear(32 → 768) |
| Residual connection | Yes (adapter output added to block output) |
| Params per adapter | 768*32 + 32*768 = 49,152 |
| Number of adapters | 12 (one per ViT block) |
| Total adapter params | ~590K |

**Alternative vision encoders (robustness check):**

| Encoder | Params | Hidden | Spatial Features | Notes |
|---------|--------|--------|------------------|-------|
| BiomedCLIP ViT-B/16 | 86M | 768 | 24x24 (at 384px) | CLIP-trained on PubMed, frozen+adapters |
| CheXNet DenseNet-121 | 7M | 1024 | 12x12 (at 384px) | ImageNet+CheXpert pretrained |

### Decoder: Phi-3-mini-4k-instruct (from RadPhi-3)

| Attribute | Value |
|-----------|-------|
| Base model | Phi-3-mini-4k-instruct (3.8B params) |
| Initialization | **RadPhi-3 weights** (radiology SOTA) |
| Quantization | **None (16-bit)** — NOT QLoRA |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| LoRA trainable params | ~14M |
| Decoder layers | 32 |
| Decoder hidden dim | 3072 |
| Decoder attention heads | 32 |
| Context length | 4096 tokens |

**Cross-attention layers:**

| Attribute | Value |
|-----------|-------|
| Injection frequency | Every 4 decoder blocks → 8 cross-attn layers |
| Query dimension | 3072 (from decoder hidden states) |
| Key/Value dimension | 768 (from ViT spatial features) |
| Projection to shared dim | Q: Linear(3072 → 768), K: passthrough, V: passthrough |
| Attention heads | 8 |
| Head dimension | 96 (768/8) |
| Output projection | Linear(768 → 3072) |
| Params per cross-attn layer | 3072*768 + 768*3072 = ~4.7M |
| Total cross-attn params | 8 * 4.7M = ~38M |

**Total trainable generator parameters:**
- LoRA: ~14M
- Cross-attention: ~38M
- Adapters: ~0.6M
- **Total: ~53M trainable, ~3.9B total**

### Generator Training

| Attribute | Value |
|-----------|-------|
| Objective | Next-token prediction (teacher forcing) |
| Loss | Cross-entropy on output tokens |
| Optimizer | AdamW (beta1=0.9, beta2=0.999, weight_decay=0.01) |
| Learning rate | 2e-4 (peak) |
| Schedule | Cosine with linear warmup (500 steps) |
| Batch size | 8 (effective 32 with grad accumulation 4) |
| Epochs | ~3 over MIMIC-CXR training split |
| Data | MIMIC-CXR training patients only (~136K reports) |
| Input format | Image tokens (flattened 27x27 ViT features) + text tokens |
| Max sequence length | 2048 tokens |
| GPU | A100 80GB |
| Estimated time | 50-60 hours |

---

## 3. Component 2: Claim Decomposer

| Attribute | Value |
|-----------|-------|
| Base model | Phi-3-mini-4k-instruct (3.8B) |
| Fine-tuning | LoRA (rank 8, alpha 16) |
| Trainable params | ~7M |
| Input | Report text (max 512 tokens) |
| Output | JSON: [{claim_text, pathology_category, confidence}] |
| Training data | RadGraph-XL (2,300) + LLM-augmented (2,700) = 5,000 reports |
| Ontology | 17 categories: CheXpert 14 + No Finding + Support Devices + Rare/Other |
| Merge threshold | P(boundary) < 0.5 → merge adjacent claims |

### Training

| Attribute | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Batch size | 16 |
| Epochs | 5 |
| GPU | T4 16GB |
| Estimated time | 6-8 hours |

### Standalone Validation Metrics
- Entity extraction F1 on held-out RadGraph-XL subset
- Reconstruction consistency: ROUGE-L between reassembled claims and original report
- Pathology category assignment accuracy

---

## 4. Component 3: Visual Evidence Grounding   **[FUTURE WORK — NOT IN NEURIPS 2026 SUBMISSION]**

| Attribute | Value |
|-----------|-------|
| Claim encoder | PubMedBERT (110M), output 768-dim |
| Spatial features | 27x27x768 from RadJEPA (384px input) |
| Projection | Linear(768 → 768) for claim embedding |
| Attention | Multi-head cross-attention, 4 heads, head_dim=192 |
| Query | Claim embedding (1x768) |
| Key/Value | Spatial features (729x768) |
| Output | 27x27 heatmap (sigmoid activation) |
| Upsampling | Bilinear interpolation to 384x384 for visualization |
| Trainable params | ~8M |

### Training

| Attribute | Value |
|-----------|-------|
| Primary source | Generator cross-attention maps (no extra data needed) |
| Optional supervision | PadChest-GR (4,555, open access); ChestImagenome (242K, PhysioNet) if available |
| Loss | BCE(heatmap, target_mask) + 0.1 * IoU_loss |
| Optimizer | AdamW, lr=5e-5 |
| Batch size | 32 |
| Epochs | 10 |
| GPU | T4 16GB |
| Estimated time | 10-12 hours |

### Evaluation Metrics
- IoU on ChestImagenome gold bounding boxes
- Pointing game accuracy on MS-CXR phrase grounding test set

### Usage in Pipeline
The grounding module produces heatmaps used for:
1. **Region-based evidence descriptions** — attended regions are described in text for the verifier input
2. **Qualitative visualization** — showing which image regions support each claim
3. **NOT directly injected into verifier** — the verifier is a text-only cross-encoder

---

## 5. Component 4: Evidence Retriever

> **Status (2026-04-14, updated for Task 5 sprint):** The eval pipeline
> now uses a **batched hybrid retriever** — MedCPT dense + BM25 sparse
> fused via reciprocal-rank fusion (k=60), then a cross-encoder reranker
> selects the top-2 passages. BM25 is now feasible at eval time because
> `models/retriever/bm25_index.py` exposes a batched
> `search_batch(queries, top_k)` that amortizes `get_batch_scores` over
> the full eval set in a single vectorized pass (replaces the per-query
> O(corpus) path that blocked this in 2026-04-05). The reranker
> (`models/retriever/reranker.py`, `cross-encoder/ms-marco-MiniLM-L-12-v2`)
> exposes `rerank_batch(query_passage_pairs, batch_size=64)` that flattens
> candidate pairs across queries into a single GPU pass then regroups.
> Task 5 ablation table compares `{dense_only, sparse_only, dense+sparse_rrf,
> +rerank}` × `{R@5, R@10, nDCG@10, acc, FDR, power}`. See
> `scripts/modal_build_retrieval_eval.py`.

### Dense Retriever: MedCPT

| Attribute | Value |
|-----------|-------|
| Query encoder | ncbi/MedCPT-Query-Encoder (110M) |
| Passage encoder | ncbi/MedCPT-Article-Encoder (110M) |
| Output dim | 768 |
| Index type | FAISS IVF-Flat (inner product on L2-normalized vectors) |
| Index size | 1,203,037 passages (sentences from 38,825 training patients) |
| nlist (IVF clusters) | min(2048, n/50) |
| nprobe (search clusters) | 32 (build) / 64 (eval) |

### Sparse Retriever: BM25

| Attribute | Value |
|-----------|-------|
| Implementation | rank_bm25 or Elasticsearch |
| Tokenization | Medical tokenizer with stop word removal |
| k1 | 1.5 |
| b | 0.75 |

### Hybrid Fusion: RRF

| Attribute | Value |
|-----------|-------|
| Method | Reciprocal Rank Fusion |
| k parameter | 60 (standard) |
| Formula | score(d) = sum over rankers r of 1/(k + rank_r(d)) |

### Reranker (ACTIVE — Task 5)

`models/retriever/reranker.py` uses the
`cross-encoder/ms-marco-MiniLM-L-12-v2` MS-MARCO cross-encoder (kept, NOT
swapped to DeBERTa per the sprint plan — the MiniLM model retains a
favourable latency/accuracy trade-off for radiology passages and the
BEIR numbers hold up on medical text). The `rerank_batch` path flattens
`(query, passage)` pairs across the full eval set into one GPU forward
pass, regroups the scores back to per-query lists, and the top-2
passages feed the verifier. This replaces the prior "top-2 from RRF
fusion directly" path.

### Top-K selection
- **Top-K = 2 passages per claim** (fed directly to verifier as concatenated evidence)
- Passages are individual sentences (mean ~40 chars), so two-passage concat fits within 512-token verifier budget

### Data Isolation
- Index populated EXCLUSIVELY from training split patients (38,825 of 38,835)
- Verified zero leakage: 0 passages overlap with calibration or test patient sets
- All splits at patient level

---

## 6. Component 5: Evidence-Conditioned Claim Verifier (Contribution 1)

> **Status (2026-04-09):** Binary (not-contradicted vs contradicted) text-only verifier.
> The 3-class verifier (Supported/Contradicted/Insufficient) was abandoned because
> it catastrophically confused Insufficient with Supported (3786/5000 misclassified,
> AUC=0.66). Binary framing achieves 98.31% accuracy. The CNN heatmap encoder exists
> in code but receives all-zeros (no trained generator → no real heatmaps). Vision
> branch is INACTIVE and deferred to future work.

### Architecture: Binary Text-Only Verifier (RoBERTa-large cross-encoder)

| Attribute | Value |
|-----------|-------|
| Base model | RoBERTa-large |
| Parameters | ~355M |
| Hidden dimension | 1024 |
| Layers | 24 |
| Attention heads | 16 |
| Max input length | 512 tokens |
| Output classes | **2** (Not-Contradicted=0, Contradicted=1) |
| Label smoothing | **0.0** (CRITICAL — 0.05 creates softmax ceiling that breaks conformal) |
| Vision branch | INACTIVE (receives zeros, deferred to future work) |

### Input Format

```
[CLS] claim_text [SEP] evidence_passage_1 [SEP] evidence_passage_2 [SEP]

Token budget:
- claim_text: ~30 tokens
- evidence_passage_1: ~200 tokens (truncated)
- evidence_passage_2: ~200 tokens (truncated)
- special tokens: ~10
- Total: ~440 tokens (within 512 limit)
```

### Output Head

**Verdict head (binary):**
```
Fused [text_cls; zero_heatmap_feat] (1792)
    → Linear(1792, 256)
    → ReLU
    → Dropout(0.1)
    → Linear(256, 2)
    → Softmax
    → {Not-Contradicted, Contradicted}
```

Note: `zero_heatmap_feat` is a 768-dim zero vector (no grounding heatmaps available).
The score head and contrastive projection head exist in code but are unused.

### Hard Negative Types (12) — Clinically Motivated

Canonical generator lives in `data/augmentation/hard_negative_generator.py`
and is reused by `scripts/prepare_eval_data.py::generate_hard_negative()`
(used for BOTH training and eval to guarantee distribution match).

The taxonomy was expanded from 8 → 12 types in the Task 2 sprint (v3
checkpoint) to broaden real-world coverage. The four additions target
fabricated-detail classes that the original taxonomy missed, and that a
literature review (Chen et al. 2025 RadFlag, HalluGuard) identified as
distinct from the 8-class structural perturbations.

| # | Implementation name | Construction | Clinical Motivation | Constraint |
|---|---|---|---|---|
| 1 | `negation` | "present" ↔ "absent" or prepend "no" | One word changes management | All findings |
| 2 | `laterality_swap` | "left" ↔ "right" | Wrong-side procedures kill | ONLY laterality-sensitive findings (gated by `should_swap_laterality`). NOT cardiomegaly/edema/mediastinum. |
| 3 | `severity_swap` | e.g., "mild" ↔ "severe", "small" ↔ "large" | Observation vs emergency intervention | All findings with severity qualifier |
| 4 | `temporal_error` | "new" ↔ "unchanged", "worsening" ↔ "improved", etc. | Triggers/prevents workup | All findings with temporal qualifier |
| 5 | `finding_substitution` | Swap with clinically confusable pair | Real diagnostic errors | Uses `CONFUSABLE_PAIRS` (consolidation↔atelectasis, pneumonia↔edema, mass↔round pneumonia) |
| 6 | `region_swap` | Anatomical region swap (upper↔lower, apical↔basal) | Wrong-anatomy errors | Non-laterality anatomy only |
| 7 | `device_line_error` | Support device mispositioned (ETT in mainstem, etc.) | Ventilation adequacy | Only claims mentioning device keywords |
| 8 | `omission_as_support` | Fabricated claim about finding NOT in report | Unnecessary procedures | Sampled from `FABRICATION_FINDINGS` |
| 9 | `fabricated_measurement` | Insert a plausible measurement ("3 mm nodule", "1.2 cm mass") not present in the original claim | VLM hallucination class; inflates urgency | Drawn from `FABRICATED_MEASUREMENT_UNITS` |
| 10 | `fabricated_prior` | Insert "compared to the prior exam from …" / "since the previous study …" phrasing | Fabricates a prior study that doesn't exist | Drawn from `FABRICATED_PRIOR_PHRASES` |
| 11 | `fabricated_temporal` | Insert an explicit relative timestamp ("since 3 days ago", "from last week's film") | Fabricates a temporal comparison anchor | Drawn from `FABRICATED_TEMPORAL_DATES` |
| 12 | `compound_perturbation` | 2-error (60%) / 3-error (40%) stacking of distinct single-type perturbations | Real hallucinations often combine structural + fabricated errors | Rejected if any step returns None or if `_evidence_supports_claim_text` still holds after stacking |

**Label-correctness validators (v3, carried over from v2):**
- C5 fix: "Insufficient Evidence" pairs enforce different patient AND different pathology
- C6 fix: `_evidence_supports_claim_text` rejects contradicted pairings where the
  perturbed claim is still supported by its original evidence
- Compound path applies validators at EACH step, then one final check on the stacked output

**Hypothesis-only baseline gap (Task 3 motivation):** the v1 verifier on
the 8-type eval set achieves 98.31% accuracy, but a hypothesis-only
baseline (evidence masked) reaches **97.71%** — a 0.60 pp gap that
indicates the verifier is exploiting one-token lexical shortcuts in the
synthetic hard negatives rather than genuine evidence reasoning.
**Task 3 (counterfactual augmentation + DPO)** is designed to push this
gap to ≥ 5 pp on the v4 checkpoint; see §13.2 below.

### Difficulty Curriculum

| Epoch Range | Easy | Medium | Hard |
|-------------|------|--------|------|
| 1 (warmup) | 60% | 30% | 10% |
| 2 | 40% | 35% | 25% |
| 3 (final) | 20% | 40% | 40% |

### Training (v2, as implemented in `scripts/modal_train_verifier.py`)

| Attribute | Value |
|-----------|-------|
| Loss | Cross-entropy with label smoothing (ε=0.05). InfoNCE dropped (NaN with small batches). |
| Batch composition | 1/3 Supported + 1/3 Contradicted + 1/3 Insufficient (enforced at data-gen) |
| Optimizer | AdamW, weight decay 0.01 |
| Learning rate | 2e-5, linear warmup 10% of total effective-batch steps |
| Batch size | 32 per-device × grad_accum 2 → effective 64 |
| Epochs | 3 (with early stopping on val loss, patience=1) |
| Precision | FP32 (bf16/fp16 cause NaN on this model) |
| Validation | 10% patient-stratified held-out split (seed=42); best checkpoint on val loss |
| Post-hoc calibration | Temperature scaling, fit on calibration split |
| GPU | H100 80GB on Modal (~40 min end-to-end) |

### Calibration Quality Assessment
- Reliability diagrams (10-bin) before and after temperature scaling
- Expected Calibration Error (ECE)
- Brier score

---

## 7. Component 6: Best-of-N Selection (Contribution 2)   **[FUTURE WORK — DEFERRED FROM NEURIPS 2026]**

### Generation

| Attribute | Value |
|-----------|-------|
| Candidates per image | N=4 (default); N=8 on 10K subset for ablation |
| Sampling method | Nucleus sampling |
| top_p | 0.9 |
| Temperatures | Sampled from {0.7, 0.8, 0.9, 1.0} |
| Max generation length | 512 tokens |

### Selection Objective (Constrained Optimization)

```
GIVEN: N candidate reports {r_1, ..., r_N} for image x
FOR each r_i:
    claims_i = decompose(r_i)
    scores_i = [verifier(c) for c in claims_i]
    faithfulness_i = mean(scores_i)
    coverage_i = |CheXbert_findings(x) ∩ mentioned_findings(r_i)| / max(1, |CheXbert_findings(x)|)

SELECT: r* = argmax_{i: faithfulness_i >= tau_faith} coverage_i

IF no candidate meets faithfulness threshold:
    r* = argmax_i faithfulness_i
    FLAG r* for mandatory human review

tau_faith ∈ {0.80, 0.85, 0.90} — report results at all three
```

### MBR-BoN Variant (NAACL 2025)

```
score_MBR(r_i) = faithfulness_constrained_score(r_i) - lambda * avg_dist(r_i, {r_j : j != i})
where avg_dist uses ROUGE-L or BERTScore distance
lambda = 0.1 (proximity regularization weight)
```

### CheXbert Label Noise Sensitivity
Report metrics with:
1. CheXbert labels only
2. Ensemble: CheXbert + NegBio + GPT-4-based labeler (majority vote)
3. Quantify metric change between (1) and (2)

---

## 8. Component 7: Conformal Claim Triage (Contribution 3)

### INVERTED cfBH Procedure (adapted from Jin and Candes, JMLR 2023)

> **Status (2026-04-09):** The standard cfBH procedure (calibrate on faithful claims)
> FAILED — score concentration at the softmax ceiling (~0.98) made p-values
> uninformative, yielding n_green=0 at all alpha levels. This happened with both
> the 3-class and binary verifiers, with and without label smoothing.
>
> The INVERTED procedure (calibrate on CONTRADICTED claims) resolves this.
> See `scripts/modal_run_evaluation.py` for the implementation.

**Why standard cfBH fails on well-trained binary classifiers:**

The binary verifier assigns P(not-contra) ≈ 0.98 to most not-contradicted claims
and P(not-contra) ≈ 0.01 to most contradicted claims. When calibrating on faithful
(not-contra) claims, all calibration scores cluster in [0.97, 0.98]. The minimum
achievable p-value is 1/(n_cal+1), which exceeds the BH threshold alpha/n_test
when scores are concentrated. Result: BH accepts zero claims.

Label smoothing (0.05) shifts the ceiling from 1.0 to 0.975 without spreading
the distribution. Per-group BH makes it worse (smaller n_cal per group).

**Inverted procedure:**

Null hypothesis H0: "test claim j is contradicted (hallucinated)."
Calibration pool: CONTRADICTED calibration claims (label=1), which have scores
near 0.01 — a well-separated reference distribution far from the ceiling.

For each test claim j with score s_j = P(not-contradicted):
```
p_j = (|{i in C_contra : s_i >= s_j}| + 1) / (|C_contra| + 1)
```

A small p_j means claim j's score is anomalously high compared to contradicted
claims — evidence against H0, suggesting claim j is NOT hallucinated.

**Global BH (not per-group):**

Per-pathology p-values are computed (preserving within-strata exchangeability)
but BH is applied GLOBALLY across all test claims. Per-group BH was too
conservative (small groups had min_p > local BH threshold).

**Guarantee:**

Under exchangeability of test-contradicted and cal-contradicted scores:
```
E[FDR] = E[|contra in green| / max(1, |green|)] <= alpha
```

**Validated results:**
- alpha=0.05: FDR=1.30%, power=98.06% (CheXpert Plus)
- alpha=0.10: FDR=2.48%, power=99.51% (CheXpert Plus)
- Cross-dataset (OpenI): FDR controlled at every alpha without retraining

### Exchangeability Handling

1. **One-per-patient subsampling:** For each calibration patient, uniformly sample one claim. This breaks within-patient dependence.
2. **ICC diagnostic:** Compute intra-patient ICC of verifier scores.
3. **Temperature scaling:** LBFGS-fit temperature on calibration NLL before computing scores.

---

## 9. Inference Pipeline (Single X-ray)

```
INPUT: chest X-ray image (any resolution)

1. PREPROCESS
   Resize to 384x384, normalize (ImageNet or RadJEPA stats)
   → tensor: (1, 3, 384, 384)

2. ENCODE IMAGE
   RadJEPA ViT-B/14 (frozen) + adapters
   → spatial features: (1, 729, 768)  [27*27 = 729 patches]
   → [CLS] token: (1, 768)

3. GENERATE N CANDIDATE REPORTS
   Phi-3-mini decoder with cross-attention to spatial features
   Nucleus sampling (p=0.9, temp ∈ {0.7-1.0}), N=4 candidates (N=8 on 10K subset for ablation)
   → reports: list of N strings

4. DECOMPOSE EACH REPORT
   Claim decomposer (Phi-3-mini LoRA)
   → per report: list of {claim_text, pathology_cat, confidence}
   Total: N * avg_claims_per_report claims (typically 5-10 per report)

5. GROUND EACH CLAIM
   Cross-attention(claim_embedding, spatial_features) → 27x27 heatmap
   Extract text description of attended region
   → per claim: region_description string

6. RETRIEVE EVIDENCE FOR EACH CLAIM
   MedCPT encode claim → dense search FAISS index
   BM25 search over training reports
   RRF fusion → DeBERTa reranker → top-2 passages
   → per claim: [evidence_1, evidence_2] strings

7. VERIFY EACH CLAIM
   RoBERTa-large: [CLS] claim [SEP] evidence_1 [SEP] evidence_2 [SEP]
   → per claim: verdict ∈ {Supported, Contradicted, Insufficient}, score s ∈ [0,1]

8. SELECT BEST REPORT
   For each candidate: compute avg_faithfulness and coverage
   Select: argmax(coverage) s.t. avg_faithfulness >= tau_faith
   → best_report with all claim scores

9. TRIAGE CLAIMS
   Apply cfBH with pathology-group thresholds
   → per claim: GREEN (accepted) / YELLOW (review) / RED (reject)

OUTPUT: {
    report_text: str,
    claims: [{
        text: str,
        pathology: str,
        verifier_score: float,
        verdict: str,
        triage_label: "green" | "yellow" | "red",
        evidence_passages: [str, str],
        heatmap: ndarray(27, 27)
    }],
    report_faithfulness: float,
    report_coverage: float,
    flagged_for_review: bool
}
```

---

## 10. Training Dependency Graph

```
                    [MIMIC-CXR Data + Patient Splits]
                    [CheXpert Plus (immediate access)]
                         /          |           \
                        /           |            \
                       v            v             v
          [1. Generator]   [2. Decomposer]   [4. Retriever Index]
          (RadJEPA+Phi-3)  (RadGraph-XL,     (MedCPT+BM25+FAISS,
          (RTX 4090, 40-50h)    5K reports)        training reports)
                |           (T4, 6-8h)        (T4, 12-15h)
                |                |                   |
                v                v                   |
         [3. Grounding]   [claims available]         |
        (ChestImagenome,       |                     |
         MS-CXR, PadChest)     |                     |
        (T4, 10-12h)           |                     |
                |               v                    v
                |        [5. Verifier (DeBERTa)]<---/
                |        (needs: claims, evidence passages)
                |        (pre: fine-tune on MedNLI/RadNLI)
                |        (T4, 25-30h)
                |              |
                |              v
                |     [5b. Temperature Scaling]
                |     (on calibration split)
                |              |
                 \             v
                  \    [6. Best-of-N Selection]
                   \   (needs: generator, decomposer,
                    \   retriever, verifier)
                     \  (RTX 4090, 50-60h gen N=4
                      \  + 30-40h verification)
                       \       |
                        \      v
                         -> [7. Conformal Calibration]
                            (cfBH on calibration split)
                            (CPU, minutes)
                                |
                                v
                         [8. Evaluation]
                         (all metrics on test split)
```

**Parallelizable:** Components 1, 2, and 4 can train in parallel (all only need MIMIC-CXR data + splits). Component 3 needs the generator's frozen vision encoder (just RadJEPA loaded, no training needed). Component 5 needs outputs from 2 and 4. Components 6 and 7 are sequential after 5.

---

## 11. Data Splits and Isolation

### Patient-Level Split Procedure

```python
# Pseudocode
patients = get_unique_patients(mimic_cxr_metadata)  # ~65,000 patients
stratify_by = [sex, num_studies_bucket]

train_patients, rest = stratified_split(patients, train_size=0.60, seed=42, stratify=stratify_by)
cal_patients, test_patients = stratified_split(rest, train_size=0.375, seed=42, stratify=stratify_by)
# 0.375 of 0.40 = 0.15 of total → cal
# 0.625 of 0.40 = 0.25 of total → test

assert len(set(train_patients) & set(cal_patients)) == 0
assert len(set(train_patients) & set(test_patients)) == 0
assert len(set(cal_patients) & set(test_patients)) == 0
```

### Expected Counts (approximate)

| Split | Patients | Studies | Reports | Images |
|-------|----------|---------|---------|--------|
| Train | ~39,000 | ~130,000 | ~136,000 | ~226,000 |
| Cal | ~9,750 | ~33,000 | ~34,000 | ~56,000 |
| Test | ~16,250 | ~55,000 | ~57,000 | ~95,000 |

### Data Isolation Rules
1. Generator training: train split images + reports ONLY
2. Retrieval index: train split reports ONLY
3. Verifier training: train split claims + hard negatives ONLY
4. Conformal calibration: cal split claims ONLY (one per report)
5. All reported metrics: test split ONLY
6. Temperature scaling: cal split (same as conformal, different purpose)

---

## 12. Conformal Calibration Mathematics

### Notation

| Symbol | Meaning |
|--------|---------|
| s_i | Temperature-scaled verifier score for claim i, s_i ∈ [0,1] |
| Y_i | Ground truth: Faithful (Supported) or Unfaithful (Contradicted or Insufficient) |
| C_g | Set of calibration claims in pathology group g (one per report) |
| n_g | \|C_g\| |
| alpha | Target FDR level (0.05 or 0.10) |
| p_j | Conformal p-value for test claim j |
| FDP_g | False discovery proportion among green claims in group g |

### cfBH Algorithm

**Input:** Calibration set C_g, test claims T_g, target level alpha.

**Calibration phase (one-time):**
1. From each calibration report in group g, sample one claim uniformly at random → C_g
2. Record (s_i) for each i ∈ C_g

**Test phase (per test claim j ∈ T_g):**
1. Compute conformal p-value:
```
   p_j = (|{i ∈ C_g : s_i ≥ s_j}| + 1) / (n_g + 1)
```
2. Collect all p-values for test claims in group g: {p_j : j ∈ T_g}
3. Sort: p_(1) ≤ p_(2) ≤ ... ≤ p_(|T_g|)
4. Find k* = max{k : p_(k) ≤ k * alpha / |T_g|}
5. Label green: all claims j with p_j ≤ p_(k*)
6. Among non-green: yellow if s_j > tau_low, red if s_j ≤ tau_low

### Guarantee (Theorem, Jin and Candes 2023)

If the calibration claims (one per report, subsampled) and test claims are exchangeable within group g, then:

```
E[FDP_g] ≤ alpha * n_g / (n_g + 1) ≤ alpha
```

This is a finite-sample, distribution-free guarantee. No distributional assumptions beyond exchangeability.

### When the Guarantee Breaks
1. **Distribution shift** between calibration and test (different time period, institution, scanner, patient population)
2. **Decomposer errors** that change the meaning of claims (guarantee is conditional on correct decomposition)
3. **ICC > 0.3** within reports, even with one-per-report subsampling at calibration, if test claims retain dependence structure

### Temporal Shift Diagnostic
Split MIMIC-CXR by admission year. Calibrate on years [2011-2014], test on years [2015-2016]. Report:
- Observed FDR among green claims vs nominal alpha
- Coverage (fraction of test claims labeled green)
- Degradation magnitude

---

## 13. Sprint Additions (Tasks 1, 3, 8, 9)

This section documents the pieces added in the 2026-04-14 sprint
(see `/Users/aayanalwani/.claude/plans/rosy-discovering-bubble.md`).
Each subsection is a pointer to the implementation file(s) + a
summary of what the code does and why. The pure-helper unit tests
for every subsection run on CPU without torch/Modal.

### 13.1 Provenance-aware trust-tier gate (existing code, newly documented)

**Purpose.** A conformal "supported" verdict is only epistemically
meaningful when the evidence and the claim come from **independent
sources**. If the evidence was produced by the same generator that
produced the claim, "supported" degenerates into "the model agrees
with itself" — useless for certification. The provenance gate refuses
to emit `SUPPORTED_TRUSTED` in that case and downgrades to
`SUPPORTED_UNCERTIFIED` instead.

**Files.**
- `inference/provenance.py` (389 lines, 36/36 tests): `EvidenceSourceType`,
  `TrustTier`, `ProvenanceTriageLabel`, `classify_trust_tier`,
  `apply_provenance_gate`, `apply_provenance_gate_batch`,
  `summarize_by_trust_tier`, `default_provenance`, `ensure_provenance_fields`,
  `is_certifiable`.
- `inference/conformal_triage.py`: `TriageResult` carries `trust_tier` and
  `final_label`; `gate_triage_with_provenance()` glues cfBH + provenance.
- `scripts/modal_run_evaluation.py` (lines 212–654): the gate is wired
  INLINE in the eval pipeline (not via `gate_triage_with_provenance`);
  `full_results.json` emits `provenance_gate` + `per_trust_tier` dicts.

**Trust tier rules for `EvidenceSourceType.GENERATOR_OUTPUT`:**
- `claim_gen == evidence_gen` (or either is None) → `SAME_MODEL`
- distinct non-empty ids → `INDEPENDENT`
- A human-curated evidence source → `TRUSTED` (overrides anything else)
- `CERTIFIABLE = {TRUSTED, INDEPENDENT}`

**Gate decision table:**

| conformal label | trust tier | final label | `was_overridden` |
|---|---|---|---|
| green | TRUSTED / INDEPENDENT | `SUPPORTED_TRUSTED` | False |
| green | SAME_MODEL / UNKNOWN | `SUPPORTED_UNCERTIFIED` | True |
| yellow | any | `REVIEW_REQUIRED` | False |
| red | any | `CONTRADICTED` | False |
| any | `PROVENANCE_BLOCKED` | `PROVENANCE_BLOCKED` | False |

### 13.2 Counterfactual augmentation + DPO refinement (Task 3)

**Purpose.** Attack the 0.60 pp HO-baseline gap by forcing the verifier
to learn from **causal** evidence reasoning rather than lexical
shortcuts. Implements the ACL 2025 "Dually Self-Improved" recipe:
(a) extract per-claim causal token spans via attention ×
Integrated-Gradients on the v3 verifier; (b) prompt Claude Sonnet 4.5
for minimally-edited counterfactual paraphrases that pin the causal
tokens verbatim and rephrase everything else; (c) train a v4
checkpoint via DPO (β=0.1, lr=5e-6, 1 epoch, KL early-stop,
freeze first 8 RoBERTa layers).

**Files.**
- `data/augmentation/causal_term_identifier.py` (32/32 tests): loads v3,
  runs `captum.attr.LayerIntegratedGradients` on the last-layer attention
  over the CLS token, returns ranked `CausalSpan` objects.
- `data/augmentation/counterfactual_generator.py` (54/54 tests):
  `CounterfactualGenerator` drives Claude Sonnet 4.5 through an injected
  transport (default = Anthropic SDK, test stub = callable), parses
  JSON variants tolerantly, validates causal-token preservation via
  case-insensitive substring, ranks by Levenshtein distance.
- `scripts/modal_train_dpo_refinement.py` (29/29 tests, H100 training
  entry point): `DPOTrainingConfig`, `load_preference_pairs`,
  `DPOEarlyStopTracker`, `format_reward_histogram`. Uses
  `trl.DPOTrainer` pinned to `trl==0.9.x`. Runs on the
  `claimguard-dpo-refinement` app, volume `claimguard-data`.

**Success criterion.** The HO-baseline gap grows from 0.60 pp (v1) to
≥ 5 pp (v4 DPO) — meaning the v4 verifier now uses evidence rather
than surface form. See §13.5 for the pass/fail metric.

### 13.3 Silver-standard real-hallucination evaluation (Task 1)

**Purpose.** First real-world measurement of the verifier on
CheXagent-generated claims grounded in real OpenI images. Replaces
the synthetic evaluation story — the paper's new headline is
"how well does ClaimGuard triage REAL VLM hallucinations," not
"how well does ClaimGuard classify synthetic hard negatives."

**Files.**
- `scripts/generate_real_hallucinations.py` (Modal H100,
  `claimguard-real-hallucinations` app): runs CheXagent-8b on N OpenI
  images, extracts claims via the rule-based decomposer, stamps
  provenance via `default_provenance(source_type=GENERATOR_OUTPUT,
  claim_generator_id="chexagent-8b-run-<run_id>",
  evidence_generator_id="openi_radiologist")`. Supports an optional
  `image_seed` distinct from `seed` so the same image set can be
  regenerated with different sampling streams (Task 9 consumes this).
- `scripts/generate_silver_standard_graders.py` (Modal H100): runs
  the 3-grader ensemble per (image, claim) — CheXbert labeler diff,
  Claude Sonnet 4.5 w/ vision, MedGemma-4B (fallback LLaVA-Med).
- `scripts/compile_silver_standard_results.py` (26/26 tests):
  computes majority vote, ordinal Krippendorff α with 1000-bootstrap
  CI, per-grader accuracy vs majority, regex-error flags, optional
  verifier inference, comparison to RadFlag 73% precision baseline.
  Exit code 2 if α < `--expect-alpha-min`.
- `evaluation/krippendorff_alpha.py` (28/28 tests): ordinal Krippendorff
  with coincidence matrix, ordinal / nominal / interval / ratio metrics,
  percentile bootstrap CI resampling at the unit level.

**Grader prompt (shared):** see `scripts/generate_silver_standard_graders.py`
line range for the 5-label scale (`SUPPORTED / CONTRADICTED /
NOVEL_PLAUSIBLE / NOVEL_HALLUCINATED / UNCERTAIN`) with JSON output
`{"label": ..., "confidence": "high|medium|low", "rationale": "<=30 words"}`.

### 13.4 Self-annotation pass (Task 8)

**Purpose.** Internal-validity sanity check on the silver pool. The
user manually labels 100 claims stratified 20-per-class, and
`compute_user_vs_ensemble_alpha.py` computes 4-coder Krippendorff α
(3 silver graders + user) with a three-rung fallback ladder:

1. `full_ordinal` — all 5 labels, ordinal metric
2. `drop_uncertain` — value-level nan-ification of UNCERTAIN codings
   (canonical Krippendorff 2018 §3.4 interpretation; softer than
   unit-level drop)
3. `binary_coarsen` — SUPPORTED vs not-SUPPORTED, nominal metric

**Files.**
- `scripts/self_annotate_silver_subset.py` (71/71 tests): interactive
  CLI with dependency-injected `display_image_fn` / `prompt_fn` so the
  tests drive the loop without a TTY or PIL. Resume-safe via
  `load_existing_annotations` + `filter_unlabeled`. Atomic writes via
  tmp + rename after every labeled row.
- `scripts/compute_user_vs_ensemble_alpha.py` (51/51 tests): pure
  helpers `align_rows_with_drops`, `build_4coder_matrix`,
  `drop_uncertain_values`, `drop_uncertain_units`, `build_binary_matrix`,
  `compute_fallback_ladder`, `format_summary`. Reports drop counts
  (missing_claim_id / no_silver_match / invalid_user_label /
  silver_no_valid_graders) so bias sources are visible in the JSON.

**Methodology critical (reviewer-flagged).** The interactive prompt
MUST NOT leak the silver graders' majority label. If the user sees
the ensemble's answer before they label, the resulting α is a "human
agrees with what was shown to them" check, not an independent
reliability estimate. Two regression tests lock this:
`test_prompt_does_not_leak_silver_majority` and
`test_prompt_does_not_leak_grader_labels_even_when_present`.

### 13.5 Same-model failure-mode case study (Task 9)

**Purpose.** Empirically validate the provenance gate (§13.1) by
running CheXagent twice over the same 100 OpenI images with distinct
sampling seeds (runs A and B), then pairing each run-A claim with
(a) run-A's own report as evidence (same-model, tier=SAME_MODEL)
and (b) run-B's report as evidence (cross-model, tier=INDEPENDENT).

**Expected result.** Both conditions see high verifier "supported"
scores because the claim and evidence come from the same
distribution (self-agreement). Under the gate, the same-model pairs
get downgraded from `SUPPORTED_TRUSTED` to `SUPPORTED_UNCERTIFIED`
at a high rate; the cross-model pairs retain certification. The
plan assertion is `downgrade_rate > 0.5` on the same-model condition.
Synthetic smoke test produces `downgrade_rate = 1.0` (same-run),
`0.0` (cross-run), `diff = 1.0`.

**Files.**
- `scripts/demo_provenance_gate_failure.py` (68/68 tests, ~990 lines):
  three-layer architecture — pure helpers (unit-testable on CPU) →
  torch helpers (verifier inference on H100) → Modal wrapper. Key
  helpers: `_conformal_label_from_score` (simplified two-threshold
  cut, defensible for the demo because downgrade rate is driven by
  tier classification not threshold value), `_extract_generator_id`
  (strict — raises on conflicting ids rather than silently keeping
  the last row's value), `_is_error_sentinel` (drops
  `[Generation error: ...]` and `[CheXagent unavailable]` rows),
  `build_gate_demo_row`, `compute_gate_demo_stats`,
  `pair_claims_with_evidence` (enforces image intersection so
  same_run and cross_run share a common denominator),
  `_sanity_check_verifier` (post-load probe that catches silent
  head-weight load failures — 2 probe pairs with opposite-polarity
  evidence, raises RuntimeError if the margin is < 0.1).

**Architecture note (reviewer fix).** Generation is launched from
the local entrypoint in `main()` (two sequential `with
gen_app.run()` blocks for runs A and B). `_run_demo` on H100 does
ONLY workbook loading + pairing + verifier scoring + persistence.
The earlier "nested `.remote()` from inside the container" pattern
was flagged by the reviewer as both fragile (requires a parent
app.run context the outer container does not own) and wasteful
(~2× GPU time because the outer container sat idle during the
inner CheXagent runs).

---

## File Structure for Implementation

```
verifact/
├── __init__.py
├── ARCHITECTURE.md                    ← this file
├── CLAIMGUARD_PROPOSAL.md             ← full proposal
├── configs/
│   ├── generator.yaml
│   ├── verifier.yaml
│   ├── retriever.yaml
│   ├── conformal.yaml
│   └── experiment.yaml
├── data/
│   ├��─ __init__.py
│   ├── preprocessing/
��   │   ├── __init__.py
│   │   ├── mimic_cxr_loader.py
│   │   ├── radgraph_parser.py         ← now uses RadGraph-XL
│   │   ├── chestimagenome_loader.py
│   │   ├── mscxr_loader.py
│   │   └── patient_splits.py
│   └── augmentation/
│       ├── __init__.py
│       ├── hard_negative_generator.py  ← 8 types (was 5)
│       └── claim_augmentation.py
├── models/
│   ├── __init__.py
│   ├── generator/
│   │   ├── __init__.py
│   │   ├── vision_encoder.py          ← RadJEPA + adapters (was BiomedCLIP ViT-L)
│   │   ├── report_decoder.py          ← Phi-3 LoRA from RadPhi-3 (was QLoRA)
│   │   └── train_generator.py
���   ├── decomposer/
│   │   ├��─ __init__.py
│   │   ├── claim_decomposer.py
│   │   └── train_decomposer.py
│   ├── grounding/
│   │   ├── __init__.py
│   │   ├── visual_grounding.py        ← 27x27 features (was 14x14)
│   │   └── train_grounding.py
│   ├── retriever/
│   │   ├── __init__.py
│   │   ├── medcpt_encoder.py
│   │   ├── bm25_index.py
│   │   ├── reranker.py
│   │   └── build_index.py
│   └── verifier/
│       ├── __init__.py
│       ├── claim_verifier.py          ← DeBERTa cross-encoder (was dual-encoder)
│       ├── hard_negatives.py          ← 8 types (was 5)
│       ├── temperature_scaling.py     ← NEW
│       └── train_verifier.py
├── inference/
│   ├── __init__.py
│   ├── best_of_n.py                   ← constrained optimization (was multiplicative)
│   ├── mbr_bon.py                     ← NEW: MBR-BoN variant
│   ├── coverage_penalty.py
│   ├── # dpo_training.py          ← removed from scope
│   └── conformal_triage.py            ← cfBH procedure (was quantile thresholding)
��── evaluation/
│   ├���─ __init__.py
│   ├── metrics.py
│   ├── conformal_coverage.py
│   ├── fairness_audit.py
│   ├── grounding_eval.py
│   ├── leakage_check.py
│   ├── temporal_shift.py
│   ├── label_sensitivity.py
│   ├── reward_hacking_check.py
│   ├── icc_diagnostic.py             ← NEW: intra-report ICC
│   ├── pipeline_error_audit.py       ← NEW: end-to-end error audit
│   ├── cluster_bootstrap.py          ← NEW: patient-level bootstrap
│   ├── ablations.py
│   └── baselines.py
├── paper/
│   ├── figures/
│   └── claimguard_cxr.tex
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_generator_training.ipynb
│   ├── 03_verifier_training.ipynb
│   ├── 04_conformal_calibration.ipynb
│   └── 05_full_evaluation.ipynb
├── requirements.txt
└── README.md
```
