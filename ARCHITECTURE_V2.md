# ClaimGuard-CXR: Technical Architecture Document v2

**Version:** 3.0 (v2 iteration — multimodal fusion, DeBERTa backbone, CoFact conformal, artifact validation)
**Target:** NeurIPS 2026 main conference + hackathon demo
**Submission date:** May 6, 2026

---

## Changelog: v1 -> v2

| Area | v1 (completed) | v2 (this iteration) |
|------|----------------|---------------------|
| Verifier backbone | RoBERTa-large (355M) | DeBERTa-v3-large (434M) — disentangled attention beats RoBERTa on NLI by 2-5% |
| Modality | Text-only | **Multimodal**: CheXzero image-claim scorer (trained on MIMIC-CXR, not PMC figures) fused with text verifier via calibrated 2-layer MLP gate |
| Claim extraction | Rule-based sentence splitter | **LLM-based contextual extraction** (Phi-3-mini) preserving negation/temporal scope across sentence boundaries |
| Conformal method | Inverted cfBH (exchangeability required) | **CoFact-inspired adaptive conformal** with density-ratio estimation on **penultimate hidden representations** (not softmax scores) |
| Artifact validation | None | **Hypothesis-only baseline** to prove verifier uses evidence, not surface cues |
| Real hallucination eval | None (synthetic only) | **Manually annotated** real-hallucination test set from CheXagent-8b on 200 OpenI images |
| Baselines | 4 weak baselines (~66%) | Add RadFlag-style self-consistency, CheXzero zero-shot, DeBERTa zero-shot NLI |
| Demo | None | **Gradio + HF ZeroGPU** web demo with per-claim color coding |
| Grounding | Deferred | **CheXzero attention maps** (radiologically meaningful, trained on CXRs not PMC figures) |

---

## Part A: V1 Architecture (Completed, All Results Final)

### A.1 System Overview

ClaimGuard-CXR v1 is a claim-level hallucination detection system for chest X-ray radiology reports with formal FDR control. It operates on text only.

```
INPUT: Radiology report text + evidence passages

1. CLAIM EXTRACTION
   Rule-based sentence splitter -> atomic claims
   Each claim tagged with pathology category (CheXpert 14 + No Finding + Support Devices)

2. EVIDENCE RETRIEVAL
   MedCPT dual-encoder dense retrieval over 1.2M training-patient passages
   FAISS IVF-Flat index, top-2 passages per claim

3. BINARY VERIFICATION
   RoBERTa-large cross-encoder: [CLS] claim [SEP] evidence1 [SEP] evidence2 [SEP]
   Output: P(Not-Contradicted), P(Contradicted)

4. CONFORMAL TRIAGE
   Inverted cfBH: calibrate on contradicted claims, global BH
   Output: GREEN (safe) / YELLOW (review) / RED (reject) per claim
```

### A.2 V1 Final Results

**Classification (CheXpert Plus, 15K test claims):**

| Metric | Oracle Evidence | Retrieved Evidence | OpenI (cross-dataset) |
|--------|----------------|--------------------|-----------------------|
| Accuracy | 98.31% | 98.09% | 85.09% |
| Macro F1 | 98.08% | 97.84% | 83.07% |
| AUROC | 99.52% | -- | -- |
| ECE | 0.0066 | 0.0060 | 0.0279 |

**Conformal FDR (inverted cfBH):**

| alpha | CheXpert FDR | CheXpert Power | OpenI FDR | OpenI Power |
|-------|-------------|----------------|-----------|-------------|
| 0.05 | 1.30% | 98.06% | 0.40% | 41.58% |
| 0.10 | 2.48% | 99.51% | 1.54% | 58.42% |
| 0.20 | 5.59% | 99.86% | 5.65% | 77.92% |

**V1 Baselines (all ~66% = majority class):**

| Method | Accuracy | Contra Recall |
|--------|----------|---------------|
| Rule-based | 66.34% | 23.66% |
| Untrained RoBERTa | 66.67% | 0.00% |
| Zero-shot LLM judge | 66.53% | 14.40% |
| CheXagent-8b (image+text) | 67.21% | 0.00% |
| **ClaimGuard v1 (oracle)** | **98.31%** | **96.38%** |

### A.3 V1 Key Design Decisions

1. **Binary (not 3-class)**: 3-class verifier confused Supported/Insufficient catastrophically (AUC=0.66). Binary captures the critical question: "is this hallucinated?"
2. **No label smoothing**: 0.05 smoothing creates softmax ceiling at P=0.975 that collapses conformal score distribution
3. **Inverted conformal**: Standard cfBH (calibrate on faithful) gives n_green=0 because faithful scores cluster at ceiling. Inverting to calibrate on contradicted resolves this.
4. **Global BH**: Per-pathology BH too conservative for small groups. Global BH with per-pathology p-values.
5. **Dense-only retrieval**: BM25 per-query cost O(corpus) infeasible at 30K claims. Dense MedCPT loses only 0.22pp.

### A.4 V1 Known Issues (Motivating V2)

1. **Synthetic-only evaluation**: No validation on real LLM hallucinations
2. **Surface shortcut vulnerability**: Synthetic perturbations may contain exploitable lexical cues (negation = add/remove one word)
3. **Text-only**: Cannot detect hallucinations where text is plausible but contradicts the image
4. **Exchangeability assumption**: cfBH assumes cal/test exchangeability, violated under distribution shift
5. **Weak baselines**: All baselines at ~66% makes the 98% look inflated
6. **Missing published comparisons**: RadFlag, ReXTrust, ConfLVLM, PRMs, StratCP not compared
7. **Label noise ceiling**: CheXpert labeler has ~13% noise on some pathologies

### A.5 V1 Component Details

#### Binary Claim Verifier (v1)

| Attribute | Value |
|-----------|-------|
| Base model | RoBERTa-large |
| Parameters | ~355M |
| Hidden dimension | 1024 |
| Layers | 24 |
| Attention heads | 16 |
| Max input length | 512 tokens |
| Output classes | 2 (Not-Contradicted=0, Contradicted=1) |
| Label smoothing | 0.0 |
| Training loss | Cross-entropy, no label smoothing |
| Optimizer | AdamW (lr=2e-5, weight_decay=0.01) |
| Batch size | 32 x 2 grad_accum = effective 64 |
| Epochs | 3 (early stop patience=1) |
| Validation | 10% patient-stratified |
| Calibration | Temperature scaling via LBFGS on cal NLL |
| GPU | H100 80GB, ~20 min |

#### Hard Negative Types (8)

| # | Type | Example | Clinical Motivation |
|---|------|---------|---------------------|
| 1 | Negation | "no effusion" -> "effusion" | One word changes management |
| 2 | Laterality swap | "left lung" -> "right lung" | Wrong-side procedures |
| 3 | Severity swap | "mild" -> "severe" | Observation vs emergency |
| 4 | Temporal error | "unchanged" -> "progressing" | Triggers/prevents workup |
| 5 | Finding substitution | "atelectasis" -> "consolidation" | Real diagnostic errors |
| 6 | Region swap | "upper lobe" -> "lower lobe" | Wrong-anatomy errors |
| 7 | Device/line error | "right subclavian" -> "left subclavian" | Ventilation adequacy |
| 8 | Omission as support | Fabricated finding | Unnecessary procedures |

#### Evidence Retriever

| Attribute | Value |
|-----------|-------|
| Dense encoder | MedCPT (ncbi/MedCPT-Query-Encoder + Article-Encoder, 110M each) |
| Index | FAISS IVF-Flat, 1,203,037 passages, 768-dim L2-normalized |
| Sparse | BM25 (built but unused at eval — O(corpus) per query) |
| Top-K | 2 passages per claim |
| Data isolation | Index from training patients only (38,825 patients) |

#### Inverted cfBH Conformal Procedure (v1)

Null H0: "test claim j is contradicted."
Calibration pool: contradicted calibration claims (label=1).
Score: s_j = P(Not-Contradicted) from temperature-calibrated softmax.
p-value: (|{cal_contra scores >= s_j}| + 1) / (n_cal_contra + 1).
Global BH at level alpha.

---

## Part B: V2 Architecture (New in This Iteration)

### B.1 V2 System Overview

```
INPUT: Chest X-ray image (optional) + Radiology report text + evidence passages

1. CLAIM EXTRACTION (UPGRADED from v1)
   LLM-based contextual extraction (Llama-3-8B-Instruct or local Phi-3-mini)
   Produces contextually complete claims, not naive sentence splits
   Preserves temporal/negation context across sentence boundaries

2. EVIDENCE RETRIEVAL (unchanged from v1)
   MedCPT dense retrieval -> top-2 passages per claim

3. TEXT VERIFICATION (upgraded backbone)
   DeBERTa-v3-large cross-encoder: [CLS] claim [SEP] evidence1 [SEP] evidence2 [SEP]
   Disentangled attention with enhanced mask decoder
   Output: text_logit (scalar), P(Not-Contradicted)

4. MULTIMODAL CONSISTENCY (NEW)
   CheXzero frozen encoder: encode (image, claim) pair
   Projection MLP + learned tau_clip temperature scaling
   2-layer MLP gate: final_score = gate * text_score + (1 - gate) * image_score

5. ADAPTIVE CONFORMAL TRIAGE (upgraded method)
   CoFact-inspired density-ratio-reweighted cfBH
   Handles distribution shift without exchangeability assumption
   Output: GREEN / YELLOW / RED per claim + visual grounding heatmap
```

### B.2 Component Upgrades

#### B.2.0 LLM-Based Contextual Claim Extractor (replaces rule-based splitter)

**Rationale (why rule-based splitting is clinically dangerous):**
Radiology reports are heavily context-dependent. Consider: *"The previously noted
moderate pleural effusion has resolved."* A naive sentence splitter might extract
"moderate pleural effusion" without the "resolved" context, triggering a false
positive contradiction. Worse, temporal references ("compared to prior", "interval
decrease in") and negation scope ("no evidence of pneumothorax or effusion") cross
sentence boundaries regularly.

**Method:**
Use a local LLM (Phi-3-mini-4k-instruct, 3.8B params, LoRA optional) prompted to
extract *contextually complete* atomic claims. Each claim must be self-contained —
it must include any negation, temporal qualifier, laterality, and anatomical context
needed to verify it independently.

**Prompt template:**
```
You are a radiology claim extractor. Given a radiology report, extract a list
of atomic, self-contained claims. Each claim must:
1. Include all negation context ("no effusion", not just "effusion")
2. Include temporal qualifiers ("new", "unchanged", "resolved")
3. Include laterality ("left", "right", "bilateral")
4. Include anatomical location ("left lower lobe", not just "lobe")
5. Be independently verifiable without reading the full report

Report: {report_text}

Output a JSON list: [{"claim": "...", "pathology": "..."}]
```

| Attribute | Value |
|-----------|-------|
| Model | Phi-3-mini-4k-instruct (3.8B, runs on CPU in ~2s/report) |
| Fallback | Rule-based sentence splitter when LLM unavailable (demo/offline mode) |
| Validation | Reconstruction ROUGE-L >= 0.85 (reassemble claims vs original) |
| Pathology tagging | LLM assigns CheXpert ontology category per claim |
| Context window | Full report (~100-300 tokens, well within 4K limit) |
| GPU | None required (CPU inference sufficient for 3.8B at short inputs) |

**Fallback behavior:** When running on ZeroGPU demo or in CPU-only mode, the
rule-based splitter is used with a post-processing step that merges claims
sharing negation or temporal scope (heuristic: if the previous sentence
contains "no", "without", "resolved", or "unchanged", merge).

#### B.2.1 DeBERTa-v3-large Text Verifier (replaces RoBERTa-large)

**Rationale:** DeBERTa-v3-large's disentangled attention mechanism separates content and position information, consistently outperforming RoBERTa-large on NLI benchmarks by 2-5%. At SemEval 2024 Task 2, DeBERTa-v3-large with contrastive learning showed strong biomedical NLI performance.

| Attribute | Value |
|-----------|-------|
| Base model | microsoft/deberta-v3-large |
| Parameters | ~434M |
| Hidden dimension | 1024 |
| Layers | 24 |
| Attention heads | 16 |
| Attention type | Disentangled (content-to-content, content-to-position, position-to-content) |
| Enhanced mask decoder | Yes (uses absolute position in final layer) |
| Max input length | 512 tokens |
| Output classes | 2 (Not-Contradicted=0, Contradicted=1) |
| Label smoothing | 0.0 (same constraint as v1) |
| Tokenizer | SentencePiece (vs RoBERTa's BPE) — different tokenization |

**Training (v2):**

| Attribute | Value |
|-----------|-------|
| Pre-fine-tuning | MedNLI (11K pairs) -> RadNLI (960 pairs) -> ClaimGuard hard negatives |
| Training data | Same 30K v2 claims (8 hard-neg types) |
| Loss | Cross-entropy (no label smoothing) |
| Optimizer | AdamW (lr=1e-5 for DeBERTa — lower than RoBERTa due to larger model) |
| Batch size | 16 x 4 grad_accum = effective 64 |
| Epochs | 5 (DeBERTa benefits from more epochs with lower LR) |
| Warmup | 10% of total steps |
| Weight decay | 0.01 |
| Validation | 10% patient-stratified (same split as v1) |
| GPU | H100 80GB, ~35 min estimated |

**Progressive domain adaptation chain:**
1. General NLI (MNLI, 433K pairs) -> DeBERTa-v3-large
2. Medical NLI (MedNLI, 11K pairs)
3. Radiology NLI (RadNLI, 960 pairs)
4. ClaimGuard hard negatives (30K pairs)

This chain is consistently best in the literature for domain-specific NLI.

**Output head (same structure as v1):**
```
Fused [deberta_cls_hidden] (1024)
    -> Linear(1024, 256)
    -> ReLU
    -> Dropout(0.1)
    -> Linear(256, 2)
    -> Softmax
    -> {Not-Contradicted, Contradicted}
```

Note: The old fused `[text_cls; zero_heatmap_feat]` (1792-dim) is replaced by a clean 1024-dim input since the heatmap path was always zeros.

#### B.2.2 CheXzero Multimodal Fusion (NEW COMPONENT)

**Rationale:** The most dangerous radiology hallucinations are where text is
internally consistent but contradicts the image. A text-only verifier cannot
catch these.

**Why CheXzero, not BiomedCLIP:** BiomedCLIP was trained on PMC-15M — academic
figures (charts, graphs, cropped scans), NOT raw clinical chest X-rays. Its
zero-shot CXR performance is brittle. CheXzero (Tiu et al., Nature Biomedical
Engineering 2022) was trained directly on 377K MIMIC-CXR image-report pairs
via contrastive learning. It achieves expert-level pathology classification
on CheXpert without any labels. Its embeddings are inherently aligned to CXR
semantics — the exact domain we need.

**Architecture:**

```
                    CXR Image (224x224)
                         |
                         v
              CheXzero Vision Encoder (ViT-B/32, FROZEN)
                         |
                         v
                  image_embedding (512-dim)
                         |
                         v
                  Projection MLP (512 -> 256 -> 256)
                         |
                         v
                  proj_image (256-dim, L2-normalized)
                         |
                         +-----> Temperature-scaled cosine similarity
                         |              |
                         v              v
              CheXzero Text Encoder (FROZEN)
                         |
                  claim_embedding (512-dim)
                         |
                         v
                  Projection MLP (512 -> 256 -> 256)
                         |
                         v
                  proj_claim (256-dim, L2-normalized)
                         |
                         v
              Cosine Sim / tau_clip -> image_claim_logit
                         |
                         v
              Sigmoid -> image_claim_prob (in [0, 1])
```

| Attribute | Value |
|-----------|-------|
| Base model | CheXzero (rajpurkarlab/CheXzero, Nature BME 2022) |
| Vision encoder | ViT-B/32 (frozen, from CLIP pre-trained on MIMIC-CXR) |
| Text encoder | GPT-2 text encoder (frozen, from same CLIP training) |
| Image input | 224x224 RGB |
| Embedding dim | 512 |
| Projection MLP | Linear(512, 256) -> GELU -> Linear(256, 256) |
| Trainable params | ~260K (projection MLPs only) |
| **CLIP temperature** | **Learned tau_clip (init=0.07), applied BEFORE sigmoid** |
| Training data | CheXpert Plus image-claim pairs from training split |
| Positive pairs | (image, claim supported by that image) |
| Negative pairs | (image, claim contradicted by that image — same 8 hard-neg types) |
| Loss | Contrastive loss (InfoNCE) on projected embeddings |
| GPU | H100 80GB, ~15 min estimated |

**Critical: CLIP temperature scaling.** Raw CLIP cosine similarities cluster
tightly (e.g., [0.2, 0.35]), while DeBERTa post-softmax scores are heavily
polarized ([0.01, 0.99]). Without temperature scaling on the CLIP side, a
linear gate would be dominated by the text logit, rendering the multimodal
branch effectively dead. We learn tau_clip jointly with the gate to
normalize CLIP scores into a comparable dynamic range.

**Learned Gating Fusion (2-layer MLP, not linear):**

```
# Temperature-scale CLIP score to match DeBERTa dynamic range
image_logit = cosine_sim(proj_image, proj_claim) / tau_clip   # spread out
image_score = sigmoid(image_logit)                              # [0, 1]

text_score = DeBERTa_verifier(claim, evidence)                  # [0, 1]

# 2-layer MLP gate (not linear — must learn non-linear calibration mapping)
gate_input = concat([text_score, image_score, |text_score - image_score|, text_score * image_score])  # 4-dim
h = ReLU(Linear(4, 16)(gate_input))         # hidden layer
gate = sigmoid(Linear(16, 1)(h))            # [0, 1]

final_score = gate * text_score + (1 - gate) * image_score
```

| Attribute | Value |
|-----------|-------|
| Gate architecture | Linear(4, 16) -> ReLU -> Linear(16, 1) -> Sigmoid |
| Gate trainable params | 81 (4*16 + 16 + 16*1 + 1) |
| Gate input features | [text_score, image_score, abs_diff, product] (4-dim) |
| tau_clip | Learned scalar, initialized 0.07, clamped [0.01, 1.0] |
| Training | Joint: projection MLPs + gate MLP + tau_clip |
| Fallback | When no image available, gate = 1.0 (text-only, backward compatible) |

**Why 2-layer gate, not linear:** DeBERTa and CLIP have fundamentally different
calibration profiles. A 4-parameter logistic regression cannot learn the non-linear
mapping between these distributions. A 2-layer MLP with 81 params is still tiny
(no overfitting risk) but can model the calibration mismatch.

**When image is unavailable:**
Gate forced to 1.0, final_score = text_score. V2 is a strict superset of v1.

#### B.2.3 Visual Grounding via CheXzero Attention (NEW)

**Rationale:** Providing visual explanations for verification decisions
dramatically improves clinical trust.

**Method:**
1. Extract attention maps from the last CheXzero ViT layer (7x7 = 49 patches at 224px/32)
2. For each claim, compute claim-conditioned attention using cross-attention between
   claim text embedding and spatial patch embeddings
3. Upscale to original image resolution via bilinear interpolation
4. Overlay as semi-transparent heatmap on the CXR for visualization

**Advantage over BiomedCLIP grounding:** CheXzero was trained on actual CXR-report
pairs, so its attention maps are inherently radiologically meaningful — it attends
to lung fields, cardiac silhouette, mediastinum, etc. BiomedCLIP's attention is
tuned to PMC figures (often irrelevant for raw CXRs).

#### B.2.4 CoFact-Inspired Adaptive Conformal Procedure (replaces inverted cfBH)

**Rationale:** The inverted cfBH procedure (v1) assumes exchangeability between calibration and test claims. This is violated under distribution shift (CheXpert -> OpenI showed 13pp accuracy drop). CoFact (ICLR 2026) uses density ratio estimation to reweight calibration data, providing valid FDR control even under shift.

**Procedure:**

```
GIVEN:
  - Calibration contradicted claims C_cal with hidden representations {h_i} and scores {s_i}
  - Test claims T_test with hidden representations {h_j} and scores {s_j}
  - Target FDR level alpha

STEP 1: Density Ratio Estimation (on HIDDEN REPRESENTATIONS, not softmax scores)
  Extract 256-dim penultimate hidden layer from DeBERTa verdict head for all
  cal + test claims. Train a lightweight binary classifier to distinguish
  calibration vs test distributions in this representation space:
      w(h) = p_test(h) / p_cal(h)  (estimated via classifier probability ratio)

  WHY NOT 1D SOFTMAX SCORES: Neural softmax outputs are famously spiky and
  clustered (cal scores near 0.01, test faithful near 0.99). Density ratio
  estimation on 1D softmax scores will explode to infinity for OOD test
  scores, breaking the FDR guarantee. The 256-dim penultimate representation
  has far richer distributional structure for stable density estimation.

STEP 2: Reweighted p-values (using softmax scores, weighted by hidden-space ratios)
  For each test claim j:
      p_j = (sum_{i in C_cal} w(h_i) * I(s_i >= s_j) + 1) / (sum_{i in C_cal} w(h_i) + 1)

STEP 3: Global BH on reweighted p-values
  Sort p_(1) <= ... <= p_(|T_test|)
  Find k* = max{k : p_(k) <= k * alpha / |T_test|}
  GREEN: claims with p_j <= p_(k*)
```

| Attribute | Value |
|-----------|-------|
| Density ratio estimator | Logistic regression on **256-dim penultimate hidden representations** |
| Hidden representation | Output of Linear(1024, 256) before ReLU in verdict head |
| Regularization | L2 (C=1.0) with PCA pre-reduction to 32-dim for stability |
| Ratio clipping | w(h) clipped to [0.01, 10.0] to prevent explosion |
| Fallback | When cal and test distributions are similar (KS test on hidden PCA p > 0.05), use standard inverted cfBH (v1) |
| Guarantee | FDR <= alpha under density ratio boundedness (w(h) <= M for known M) |

**Key advantage over v1:** When evaluating on OpenI (distribution shift), CoFact
maintains power while controlling FDR. V1's inverted cfBH was overly conservative
under shift (power dropped to 42% at alpha=0.05). Using hidden representations
(not softmax) for density estimation avoids the degeneracy of 1D score clustering.

### B.3 New Validation Experiments

#### B.3.1 Hypothesis-Only Baseline (Artifact Check)

**Rationale:** If the verifier achieves 98% accuracy, a natural concern is that synthetic perturbations contain surface shortcuts. A hypothesis-only baseline (verifier sees only the claim, not the evidence) measures how much accuracy comes from lexical cues rather than evidence reasoning.

**Implementation:**
```
Input: [CLS] claim_text [SEP] [PAD] ... [SEP]
       (evidence replaced with padding tokens)
```

Train a separate DeBERTa-v3-large on the same 30K training set but with evidence masked. If hypothesis-only accuracy is significantly above 66% (majority class), it indicates surface shortcuts in the perturbations.

**Expected outcome:** Hypothesis-only should achieve ~70-80% (some perturbations like negation ARE detectable from claim alone — "no effusion" vs "effusion" is a valid signal). But if it reaches >90%, the task is too easy.

**Mitigation if artifacts found:** Add adversarial perturbations that preserve surface features while changing meaning (e.g., paraphrase contradictions without keyword swaps).

#### B.3.2 Real-Hallucination Test Set

**Rationale:** The biggest reviewer concern is that evaluation on synthetic perturbations doesn't prove the verifier works on actual LLM-generated hallucinations.

**Construction (two-phase: generate then manually annotate):**

Phase 1 — Generation:
1. Run CheXagent-8b on 200 OpenI test images (CC BY-NC-ND 4.0, public)
2. CheXagent generates a report for each image
3. Extract claims from generated reports using LLM claim extractor
4. Produces ~800-1200 generated claims

Phase 2 — Manual Annotation (CRITICAL — cannot use CheXbert as ground truth):
5. Each generated claim is labeled by a human annotator (medical student or
   radiologist) against the original OpenI report + image as reference
6. Label schema: Supported (claim matches image+report), Contradicted (claim
   contradicts image or report), Novel-Plausible (claim is valid but not in
   original report), Novel-Hallucinated (claim is fabricated)
7. Double annotation on 25% subset for inter-annotator agreement (Cohen's kappa)

**Why NOT CheXbert auto-labeling:** CheXbert has ~10-15% label noise. If
ClaimGuard disagrees with CheXbert, who is right? Evaluating a SOTA verifier
against a legacy labeler is circular — reviewers at NeurIPS will reject this
immediately. Manual annotation of 200 reports (~1000 claims) is feasible in
~8-12 hours of annotator time. Quality > quantity.

**Annotation protocol:**
- Annotator sees: CXR image + original report + generated claim
- Annotator answers: "Does this claim accurately describe what is shown in the
  image and consistent with the original report?"
- If ambiguous, annotator marks "Uncertain" (excluded from evaluation)
- Claims where original report is itself ambiguous are also excluded

| Attribute | Value |
|-----------|-------|
| VLM | CheXagent-8b (StanfordAIMI/CheXagent-8b) |
| Images | 200 OpenI test images (smaller set, manually annotated) |
| Ground truth | **Manual human annotation** against image + report |
| Annotator | Medical student or physician (Ashley Laughney's lab contacts) |
| Double annotation | 25% subset for inter-rater reliability |
| Expected kappa | >= 0.75 (good agreement) |
| Total annotation time | ~8-12 hours |
| GPU (generation only) | H100 80GB on Modal, ~15 min for 200 images |

#### B.3.3 Additional Baselines

| Method | Type | What it tests |
|--------|------|---------------|
| Hypothesis-only DeBERTa | Artifact check | Surface shortcut vulnerability |
| CheXzero zero-shot | Image-text matching | Can CLIP alone detect contradictions? |
| DeBERTa-v3-large zero-shot NLI | Text NLI | How much does domain fine-tuning help? |
| RadFlag-style self-consistency | Sampling-based | Generate multiple reports, flag inconsistencies |
| Cross-encoder NLI (MedNLI-trained) | Medical NLI | Transfer from general medical NLI |

### B.4 V2 Training Dependency Graph

```
[CheXpert Plus Data + Patient Splits + OpenI Data]
              /            |              \
             /             |               \
            v              v                v
[1. DeBERTa Progressive]  [2. CheXzero]    [3. Real Hallucination
     Fine-tuning               Projection       Test Set]
  MNLI -> MedNLI ->           Training      (CheXagent on OpenI)
  RadNLI -> ClaimGuard     (contrastive,      (H100, ~30 min)
  (H100, ~2 hr total)      H100, ~15 min)         |
            |                    |                  |
            v                    v                  v
[4. Gating Fusion Training]                  [5. Validation on
  (joint DeBERTa + CheXzero gate)             real hallucinations]
  (H100, ~10 min)                                   |
            |                                       |
            v                                       v
[5. CoFact Conformal Calibration]       [6. Artifact Analysis]
  (density ratio + reweighted BH)        (hypothesis-only baseline)
  (CPU, minutes)                         (H100, ~35 min)
            |                                       |
            v                                       v
[7. Full Evaluation]                    [8. Results Comparison]
  (CheXpert + OpenI + real hallucinations)  (v1 vs v2, all baselines)
```

### B.5 V2 Expected Results (Estimates)

| Metric | V1 (actual) | V2 (estimated) | Source of improvement |
|--------|-------------|----------------|----------------------|
| CheXpert accuracy | 98.31% | 98.5-99.0% | DeBERTa + progressive NLI |
| OpenI accuracy | 85.09% | 88-91% | DeBERTa + CoFact adaptation |
| Real hallucination detection | -- | 80-85% | New eval (harder than synthetic) |
| FDR@0.05 (CheXpert) | 1.30% | 1.0-1.5% | CoFact tightens |
| Power@0.05 (OpenI) | 41.58% | 55-65% | CoFact improves power under shift |
| Hypothesis-only accuracy | -- | 70-80% | Artifact baseline (should be low) |

---

## Part C: Hackathon Web Demo Architecture

### C.1 Overview

A Gradio web application on Hugging Face Spaces with ZeroGPU for free GPU inference.

```
Gradio App (HF Spaces + ZeroGPU H200)
  |
  |-- Tab 1: Report Verifier
  |     Input: paste/type a radiology report
  |     Processing:
  |       1. Claim extraction (CPU, rule-based)
  |       2. MedCPT encoding (CPU)
  |       3. FAISS retrieval (CPU, 100K passage subset)
  |       4. DeBERTa verification (@spaces.GPU)
  |       5. Conformal triage (CPU)
  |     Output: gr.HighlightedText with green/yellow/red claims
  |             + evidence panel showing retrieved passages
  |             + confidence scores
  |
  |-- Tab 2: End-to-End Multimodal Demo
  |     Input: upload CXR image (from OpenI public dataset)
  |     Processing:
  |       1. CheXagent-8b generates report (@spaces.GPU)
  |       2. ClaimGuard verifies each claim
  |       3. CheXzero provides visual grounding
  |     Output: generated report with color-coded claims
  |             + heatmap overlay on CXR image
  |             + side-by-side with ground truth (if available)
  |
  |-- Tab 3: Calibration Explorer
  |     Interactive FDR alpha slider (Plotly)
  |     Drag alpha: watch claims change color in real-time
  |     Reliability diagram + score distribution
  |
  |-- Tab 4: Baseline Comparison
  |     Same report verified by rule-based / LLM judge / CheXagent / ClaimGuard
  |     Side-by-side accuracy display
```

### C.2 Tech Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| Frontend | Gradio 4.x | Required for ZeroGPU, purpose-built HighlightedText |
| GPU inference | HF ZeroGPU (H200) | Free tier: 3.5 min/day, enough for ~50 demos |
| Model serving | @spaces.GPU decorator | Zero cold-start within HF Spaces |
| Retrieval | FAISS (100K passage subset, ~300MB) | Loaded on startup, CPU inference |
| Visualization | Plotly (calibration plots), matplotlib (heatmaps) | Interactive + publication-quality |
| Synthetic reports | 30 pre-generated examples | Cover normal + 5 common pathologies + hallucinations |
| CXR images | OpenI public dataset (CC BY-NC-ND 4.0) | Pre-loaded 10 examples + user upload |

### C.3 Key Demo Files to Create

```
verifact/
  demo/
    app.py                    # Main Gradio application
    models.py                 # Model loading + inference wrappers
    retrieval.py              # FAISS index loading + search
    conformal.py              # Conformal triage procedure
    claim_extractor.py        # Rule-based claim extraction
    examples/
      synthetic_reports.json  # 30 synthetic radiology reports
      openi_images/           # 10 pre-selected OpenI CXR images
      ground_truth.json       # Ground truth for examples
    assets/
      logo.png
      custom.css
```

### C.4 Demo Cost

| Resource | Cost |
|----------|------|
| HF Spaces free tier | $0 |
| HF Spaces PRO (optional) | $9/month |
| Modal (for CheXagent if needed) | ~$2 per hackathon |
| **Total** | **$0-11** |

---

## Part D: File Manifest

### V1 Files (existing, unchanged)

| Purpose | Path |
|---------|------|
| V1 architecture | `ARCHITECTURE.md` |
| Research proposal | `CLAIMGUARD_PROPOSAL.md` |
| Reproducibility | `REPRODUCIBILITY.md` |
| Paper writing handoff | `HANDOFF_PAPER_WRITING.md` |
| Mini manuscript | `MANUSCRIPT_MINI.md` |
| Binary training (v1) | `scripts/modal_train_verifier_binary.py` |
| Eval pipeline (v1) | `scripts/modal_run_evaluation.py` |
| Data generation | `scripts/prepare_eval_data.py` |
| Hard-neg generator | `data/augmentation/hard_negative_generator.py` |
| Evidence retriever | `models/retriever/medcpt_encoder.py` |
| All 7 figures | `figures/fig1-7_*.pdf` |

### V2 Files (new, to be created)

| Purpose | Path |
|---------|------|
| V2 architecture (this file) | `ARCHITECTURE_V2.md` |
| LLM claim extractor | `models/decomposer/llm_claim_extractor.py` |
| DeBERTa verifier model | `models/verifier/deberta_verifier.py` |
| DeBERTa training script | `scripts/modal_train_verifier_deberta.py` |
| Progressive NLI fine-tuning | `scripts/modal_progressive_nli.py` |
| CheXzero fusion module | `models/multimodal/biomedclip_fusion.py` |
| CheXzero training script | `scripts/modal_train_chexzero_fusion.py` |
| Gating fusion module | `models/multimodal/gating_fusion.py` |
| CoFact conformal procedure | `inference/cofact_conformal.py` |
| Hypothesis-only baseline | `scripts/baseline_hypothesis_only.py` |
| Real hallucination generator | `scripts/generate_real_hallucinations.py` |
| Additional baselines | `scripts/baseline_biomedclip_zeroshot.py` |
| Gradio demo app | `demo/app.py` |
| Demo model wrappers | `demo/models.py` |
| Demo retrieval | `demo/retrieval.py` |
| Demo conformal | `demo/conformal.py` |
| Demo claim extraction | `demo/claim_extractor.py` |

---

## Part E: Critical Paths and Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| DeBERTa worse than RoBERTa on this task | Low | Medium | Keep v1 RoBERTa as fallback, report both |
| CheXzero adds noise instead of signal | Medium | Medium | Gate defaults to text-only; ablation shows delta |
| CoFact density ratio poorly estimated | Low | High | Fallback to v1 inverted cfBH when KS test says distributions similar |
| Hypothesis-only baseline reveals severe artifacts | Medium | High | Add adversarial perturbations, report transparently |
| Real hallucination test set too small | Low | Medium | 500 images with ~2500 claims is sufficient for significance |
| HF ZeroGPU quota insufficient for hackathon | Low | Low | Fallback to Modal serving ($2-5) |
| CheXagent-8b requires transformers==4.40.0 | Known | Low | Pin version in Modal image |
