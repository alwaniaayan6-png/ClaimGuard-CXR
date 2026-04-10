# ClaimGuard-CXR: Claim-Level Verification and Conformal False Discovery Rate Control for Radiology Report Generation

**Target venue:** NeurIPS 2026 main conference (deadline: May 6, 2026)
**Format:** 9 pages content + unlimited references/appendix

---

## 🔖 SCOPE FOR NEURIPS 2026 SUBMISSION (updated 2026-04-04)

Following a comprehensive audit, the scope of this submission has been reduced to
two focused contributions on text-level claim verification:

**Contribution 1:** Evidence-conditioned claim verifier with an 8-type
clinically-motivated hard-negative taxonomy. RoBERTa-large cross-encoder over
(claim, evidence) pairs, trained on CheXpert Plus reports with patient-stratified
validation, label smoothing, and temperature-scaled output.

**Contribution 2 (was 3):** Pathology-stratified cfBH conformal FDR control over
claim-level triage labels, with exchangeability diagnostics (intra-patient ICC,
one-per-patient subsampling) and label-conditional calibration.

**Deferred to future work** (not present in v2 experiments or the May 6 paper):
- End-to-end report generation from CXR images (no trained generator; images
  not downloaded — 450 GB dataset)
- Best-of-N candidate selection via constrained optimization (requires generator)
- Visual grounding via cross-attention heatmaps (requires generator)
- "Multimodal" verification: the paper is honestly framed as TEXT-ONLY
  verification. Ground-truth labels come from CheXbert (image-derived), so
  evaluation is image-grounded, but the verifier itself does not process pixels.

**Hybrid retrieval (MedCPT + BM25 + RoBERTa reranker) feeds evidence into the
verifier and remains an evaluated component.**

The **HONEST LIMITATIONS** section (end of this doc) discusses these scope
decisions and reviewers' likely concerns in full.

---

## Background

### The Clinical Burden of Radiology Reporting
Radiologists spend roughly a third of their working hours writing reports. With a growing global shortage of radiologists, particularly in low-resource settings, automated report generation from medical images has the potential to free up millions of physician hours annually for direct patient care.

### The Hallucination Problem
Current vision-language models hallucinate at rates between 8 and 15 percent when generating radiology reports. These errors include fabricating findings that are not present in the image, omitting critical observations, misattributing laterality such as confusing left and right, and contradicting visible evidence. A single hallucinated claim of "no pneumothorax" could delay life-saving treatment. Laterality errors and missed critical findings represent the highest-risk failure modes.

### The Trust and Triage Gap
Prior work studies hallucination detection and conformal trust selection for generated medical text:

**Hallucination detection.** RadFlag (ML4H 2025) uses multiple sampled generations at varying temperatures to identify inconsistent claims via self-agreement, flagging 28 percent of hallucinations at 73 percent precision. ReXTrust (AAAI Bridge 2025) leverages LVLM hidden states through a self-attention module to produce finding-level hallucination risk scores, achieving 0.8751 AUROC across findings and 0.8963 on clinically significant findings. Process Reward Models for sentence-level verification (October 2025) train a lightweight 0.5B parameter model that assigns correctness probabilities conditioned on clinical context, outperforming ReXTrust by 7.5 percent MCC. FactCheXcker (CVPR 2025) targets measurement hallucinations specifically, achieving 94 percent reduction via a query-code-update paradigm. Clinical Contrastive Decoding (CCD, September 2025) provides training-free hallucination reduction with 17 percent RadGraph-F1 improvement.

**Retrieval-augmented generation.** Fact-Aware multimodal RAG retrieves similar reports to ground generation. FactMM-RAG (NAACL 2025) uses RadGraph to mine factual pairs and trains a multimodal retriever. However, these systems do not provide post-hoc claim-level triage or formal error control.

**Conformal trust selection.** Conformal Alignment (Angelopoulos et al., NeurIPS 2024) applies conformal trust-selection to radiology report generation with FDR control, but operates at the report level, accepting or rejecting entire reports. ConfLVLM (Li et al., EMNLP 2025) extends conformal prediction to the claim level for vision-language models and evaluates on radiology, reducing error rates from 87.8 to 10.0 percent with 95.3 percent true positive rate, but uses global thresholds without evidence conditioning or generation loop integration. StratCP (Harvard/Zitnik lab, medRxiv February 2026) provides stratified conformal prediction for medical foundation models with pathology-conditional calibration. CoFact (ICLR 2026) extends conformal factuality guarantees to handle distribution shift. The Conformal LM Reasoning paper (ICLR 2025, U. Penn/IBM) uses conformal prediction for LLM factuality via graph-based subgraph selection. Selective Generation (NeurIPS 2024) controls FDR with respect to textual entailment.

**The gap.** No prior work integrates (a) evidence-conditioned claim verification with clinical hard negatives, (b) cfBH-based false discovery rate control with pathology-stratified calibration at the claim level, and (c) verification-guided generation selection into a single pipeline for radiology reports. Each component has strong prior art; our contribution is the principled integration and the theoretical analysis of FDR guarantees under the pipeline's dependency structure.

### Scope and Intent
ClaimGuard-CXR is not an autonomous report generation system. It is a drafting and triage tool designed to reduce hallucinations and provide calibrated per-claim triage labels for human review. The goal is to surface which claims are well-supported by image evidence and prior verified reports, which are uncertain, and which are likely errors. The specific error notion we control is the false discovery rate among claims labeled as high confidence: the expected fraction of unfaithful claims among those the system marks as accepted. An unfaithful claim is one classified by the verifier as Contradicted or as having Insufficient Evidence given the image and retrieved context. This error control holds under the standard exchangeability assumption between calibration and test claims (one claim subsampled per report to ensure independence) and does not imply individual conditional validity or robustness under dataset shift.

---

## Related Work

**Radiology hallucination detection.** RadFlag (ML4H 2025) flags 28 percent of hallucinations at 73 percent precision via sampling-based self-agreement. ReXTrust (AAAI Bridge 2025) achieves AUROC 0.8751 via LVLM hidden states, requiring white-box model access. The PRM approach (October 2025, 0.5B parameters) outperforms ReXTrust by 7.5 percent MCC using text-conditioned sentence scoring. FactCheXcker (CVPR 2025) specifically targets measurement hallucinations with 94 percent reduction. CCD (September 2025) provides training-free hallucination reduction. The Generative LLMs for Error Detection paper (Radiology 2025) achieves F1 of 0.780 with fine-tuned Llama-3-70B. The Phrase-Grounded Fact-Checking paper (MICCAI 2025, IBM) uses multi-label cross-modal contrastive regression trained on 27M synthetic pairs for error detection with image localization. ClaimGuard-CXR differs from all of these by conditioning verification on externally retrieved evidence passages (not just model internals or self-consistency) and integrating verification into the generation loop rather than applying it purely post-hoc.

**Stanford VeriFact** (Chung et al., NEJM AI 2025) decomposes clinical text into propositions, retrieves evidence via hybrid search, and uses Llama 3.1 70B as an LLM-as-Judge, achieving 92.7 percent agreement with clinicians. It targets EHR discharge summaries, uses no conformal prediction, and has no image component. ClaimGuard-CXR differs by targeting radiology with multimodal (image + text) evidence, using a trained verifier rather than LLM-as-Judge, and providing calibrated FDR guarantees.

**Verifier-guided generation.** Process reward models score intermediate steps in reasoning chains and can guide generation via best-of-N selection or reranking. Best-of-N is susceptible to reward hacking when N is large (ICML 2025 proves this is an inevitable property of proxy optimization). MBR-BoN (NAACL 2025) adds a Minimum Bayes Risk proximity regularizer. InferenceTimePessimism (ICML 2025) provides scaling-monotonic guarantees. ClaimGuard-CXR uses constrained best-of-N with a faithfulness threshold and reports both naive BoN and MBR-BoN.

**Conformal trust selection.** Conformal Alignment (NeurIPS 2024) provides report-level accept/reject for radiology. ConfLVLM (EMNLP 2025) provides claim-level conformal filtering for VLMs with global thresholds. StratCP (medRxiv 2026) provides pathology-stratified conformal prediction for medical FMs. CoFact (ICLR 2026) handles distribution shift via density-ratio reweighting. The Conformal LM Reasoning paper (ICLR 2025) uses graph-based subgraph selection. ClaimGuard-CXR contributes: (a) cfBH-based FDR control (not marginal coverage) at the claim level, (b) pathology-group-stratified thresholds, (c) explicit handling of within-report claim dependence via one-claim-per-report subsampling for calibration, and (d) integration into the generation loop. Where Conformal Alignment tells a clinician whether to trust an entire report, ClaimGuard-CXR tells them which specific claims to trust, which to double-check, and which to discard.

**Radiology RAG.** Fact-Aware multimodal RAG and FactMM-RAG (NAACL 2025) reduce hallucination via retrieval but provide no formal error control. ClaimGuard-CXR uses retrieval as one input to an explicit verification step rather than relying on retrieval alone.

---

## Methods

### Overview
ClaimGuard-CXR consists of standard infrastructure components (a vision-language generator, a claim decomposer, a visual grounding module, and an evidence retrieval system) and three novel contributions built on top: (1) an evidence-conditioned claim verifier trained with clinically realistic hard negatives, (2) verifier-guided best-of-N selection with a faithfulness-constrained coverage objective, and (3) cfBH-based conformal false discovery rate control for claim triage with pathology-group-stratified thresholds.

### Contributions
Our contributions are precisely three:

**Contribution 1: Evidence-conditioned claim verifier.** A RoBERTa-large cross-encoder (~304M parameters) trained with eight types of clinically motivated hard negatives (laterality swap, negation, finding substitution, region swap, severity swap, temporal error, device/line error, omission) that operates on claim text concatenated with top-2 retrieved evidence passages and a textual grounding description derived from the spatial attention heatmap (e.g., "strong attention on left lower lung, moderate attention on cardiac silhouette"). This grounding signal provides indirect visual evidence to the text-only verifier. Extends prior hallucination detectors (ReXTrust, PRM, FactCheXcker) with external evidence conditioning and clinical hard negative diversity. Trained with cross-entropy as primary loss and InfoNCE contrastive as auxiliary, with post-hoc temperature scaling for calibration quality.

**Contribution 2: Verifier-guided best-of-N selection with constrained optimization.** Generate N candidate reports, decompose and score all claims, select the report that maximizes coverage of detected findings subject to a faithfulness threshold (avg verifier score >= tau). This replaces the common multiplicative faithfulness-times-coverage heuristic with a principled constrained formulation that prevents the system from accepting reports with high completeness but poor faithfulness. We report results with both naive BoN and MBR-BoN (NAACL 2025), at N=4 and N=8.

**Contribution 3: cfBH-based conformal FDR control with pathology-stratified calibration.** We apply the conformal Benjamini-Hochberg procedure (Jin and Candes, JMLR 2023) to construct conformal p-values for each claim and control the false discovery rate among accepted claims. Calibration thresholds are computed per pathology category (CheXpert 14 + No Finding + Support Devices) when the group contains sufficient calibration claims, with small groups pooled. To handle within-report claim dependence, we subsample one claim per report for calibration and report the intra-report ICC of verifier scores as a diagnostic. This extends ConfLVLM (which uses global thresholds and marginal coverage) and StratCP (which does not integrate with a generation loop or use cfBH-based FDR) to provide per-pathology FDR guarantees.

### Component 1: Generator (Infrastructure)

**Vision encoder:** We evaluate two frozen encoder options. The primary encoder is RadJEPA ViT-B/14 (86M parameters, 12 transformer layers, hidden dimension 768, 12 attention heads; pretrained on 839K unlabeled CXR images via the I-JEPA self-supervised protocol; weights available at HuggingFace AIDElab-IITBombay/RadJEPA). We keep the encoder frozen and train lightweight bottleneck adapter layers (768 to 32 to 768, ReLU, ~600K trainable parameters per adapter, 24 total = ~14.4M). Input resolution is 384x384 to produce 27x27 spatial features (384/14 ~ 27), providing substantially better spatial resolution than the default 224x224 (which yields only 16x16). As a robustness check we also report results with BiomedCLIP ViT-B/16 (86M parameters, hidden dimension 768) and CheXNet-pretrained DenseNet-121 to verify contributions are not dependent on a specific vision backbone.

**Decoder:** Phi-3-mini-4k-instruct (3.8B parameters) initialized from RadPhi-3 weights (pretrained for radiology, SOTA on RaLEs benchmark). We use standard LoRA (not QLoRA) with rank 16, alpha 32, in 16-bit precision. This fits on a single 24GB GPU without the quality loss associated with 4-bit quantization. Cross-attention is injected every 4 transformer blocks (8 cross-attention layers total), with query from decoder hidden states (3072-dim) and key/value from ViT spatial features (768-dim), projected to a shared dimension of 768. Total trainable parameters: ~14M (LoRA) + ~48M (cross-attention) + ~14M (adapters) = ~76M. Total parameters: ~3.9B (mostly frozen). Training uses standard next-token prediction on MIMIC-CXR image-report pairs from training split patients only. Optimizer: AdamW, learning rate 2e-4 with cosine schedule, batch size 8 with gradient accumulation of 4, approximately 3 epochs.

### Component 2: Claim Decomposer (Infrastructure)

A fine-tuned language model splits generated reports into atomic verifiable claims. Base model: Phi-3-mini with LoRA (rank 8, alpha 16). Input: full radiology report text (max 512 tokens). Output: JSON list of atomic claims, each with a claim text, a pathology category from the fixed ontology, and a confidence score.

Training data: RadGraph-XL (2,300 annotated reports with entity-relation graphs, ACL 2024) augmented using LLM-generated decompositions following the RadAnnotate approach. We use a prompted language model to generate candidate claim decompositions for an additional 2,700 MIMIC-CXR reports, filter these against CheXbert-extracted labels for consistency, yielding approximately 5,000 total training reports. This is a substantial improvement over the original 500-report RadGraph.

When the decomposer confidence is below 0.5 for a claim boundary, adjacent claims are merged into a coarser unit rather than split incorrectly. Each extracted claim is mapped to a pathology category from a fixed ontology: the 14 CheXpert observations plus No Finding plus Support Devices. Claims that do not map cleanly are assigned to a Rare/Other category (17 total categories).

We validate the decomposer independently before pipeline integration: standalone entity extraction F1 is reported on a held-out subset of RadGraph-XL annotated reports. We additionally compute a reconstruction consistency metric: reassembling claims back into text and comparing with the original report via ROUGE-L to bound downstream error propagation from decomposition errors.

### Component 3: Visual Evidence Grounding (Infrastructure)

Cross-attention between each claim embedding and the ViT 27x27 spatial feature map (from 384x384 input) produces a heatmap showing which image regions are most relevant to each claim. The claim embedding is obtained from PubMedBERT (768-dim). Architecture: linear projection (768 to 768) plus multi-head cross-attention (4 heads, claim as query, spatial features as key/value) plus sigmoid activation. Output: 27x27 heatmap, upsampled to 384x384 for visualization.

The primary grounding source is the generator's own cross-attention maps: when the generator produces text, its cross-attention layers attend to specific image patches, producing spatial attention weights that serve as grounding heatmaps. This requires no external supervision data and no separate training — the generator learns to attend to relevant regions as part of report generation. Optionally, the grounding can be supervised with PadChest-GR (4,555 grounded reports, open access). If PhysioNet access is available, ChestImagenome (242K anatomical annotations) and MS-CXR (1,153 phrase grounding pairs) can further improve grounding quality.

We quantify grounding quality using pointing game accuracy on PadChest-GR test set, and include laterality-sensitive qualitative examples showing correct and incorrect spatial attention for left versus right findings.

**Note on heatmap usage:** The grounding heatmap is used for (a) region-based evidence retrieval (extracting text descriptions of attended regions) and (b) qualitative visualization for the paper. It is NOT injected directly into the verifier cross-encoder. This avoids the architectural complexity of multimodal adapters and keeps the verifier purely text-based, which is cleaner and more defensible.

### Component 4: Evidence Retriever (Infrastructure)

A hybrid dense and sparse retrieval system searches over 1.2M sentence-level passages from CheXpert Plus training-split reports (38,825 patients). Evaluated as a retrieval-augmented ablation; the main verifier results table uses oracle (same-report) evidence to isolate verifier quality.

**Dense component:** MedCPT dual-encoder (ncbi/MedCPT-Query-Encoder + ncbi/MedCPT-Article-Encoder, each 110M parameters, trained on PubMed search logs). Encodes claims and report passages to 768-dim L2-normalized vectors. FAISS IVF-Flat index with inner-product metric over the 1.2M training passages.

**Sparse component:** BM25 via rank_bm25 (k1=1.5, b=0.75), whitespace tokenization.

**At eval time:** Dense retrieval only. The BM25 sparse index is built but not used at eval time — rank_bm25's per-query cost scales linearly with corpus size (~2s for 1.2M passages), making hybrid retrieval infeasible at 30K eval claims. Batched sparse retrieval (e.g., via Pyserini or a sparse-matrix reformulation) and hybrid RRF fusion are deferred as future work. Top-2 dense-retrieved passages are concatenated as evidence for the verifier.

**Reranker:** A RoBERTa-large cross-encoder reranker is implemented (`models/retriever/reranker.py`) but not applied in the current eval. Reranker ablation is deferred.

We enforce strict data isolation: the retrieval index is populated exclusively from training-set patients, verified empirically with zero passage overlap into calibration or test patient sets. All splits are at the patient level.

### Component 5: Evidence-Conditioned Claim Verifier (Contribution 1)

> **Status (2026-04-09):** Binary text-only verifier. The 3-class framing
> (Supported/Contradicted/Insufficient) was abandoned due to catastrophic
> Insufficient-Supported confusion (AUC=0.66). Vision branch deferred (no
> trained generator = no heatmaps). Binary achieves 98.31% accuracy.

A text-only binary verifier (~355M parameters) using RoBERTa-large as a cross-encoder. Input: [CLS] claim [SEP] evidence [SEP]. Output: 2-class softmax (Not-Contradicted=0, Contradicted=1).

**Architecture:** RoBERTa-large cross-encoder processes [CLS] claim_text [SEP] evidence_passage_1 [SEP] evidence_passage_2 [SEP]. With claim (~30 tokens) and 2 evidence passages (~200 tokens each), total input is approximately 430-480 tokens, fitting within the 512-token limit. The CNN heatmap encoder exists in code but receives all-zero input (no trained generator or grounding module). Vision fusion is deferred to future work.

**Binary framing rationale:** The 3-class verifier achieved val_acc=72.84% but confused Insufficient with Supported catastrophically (3786/5000 Insufficient claims predicted Supported). The distinction between "evidence supports claim" and "evidence doesn't address claim" is fundamentally ambiguous for automated text verification. Binary framing (contradicted vs not) captures the clinically critical question: "is this claim hallucinated?"

**Output:** A two-class verdict head via MLP(1792 to 256 to 2) with softmax. P(Not-Contradicted) serves as the conformal score.

**Training loss:** Binary cross-entropy with **no label smoothing** (0.0). Label smoothing (0.05) was tested but creates a hard ceiling at P=0.975/logit_margin=3.66 that collapses the calibration score distribution and breaks conformal BH (n_green=0 at all alpha). The score_head and contrastive_proj exist in code but are unused.

**Hard negatives (8 types)** constructed from RadGraph-XL annotated reports. Each type corresponds to a documented real-world radiology error mode, not an arbitrary perturbation:

1. **Laterality swap** — replacing "left" with "right" and vice versa. **Clinical motivation:** laterality errors are the single most dangerous error class in radiology, as they can lead to wrong-side procedures. Applied ONLY to laterality-sensitive findings (pneumothorax, pleural effusion, consolidation, etc.) — NOT to midline findings like cardiomegaly or pulmonary edema where laterality swap creates nonsense, not a confounder.
2. **Finding substitution with confusable pairs** — replacing a finding with one commonly confused in clinical practice: consolidation with atelectasis (the most common radiology confusion), pneumonia with pulmonary edema (similar CXR appearance), lung mass with round pneumonia. These pairs are drawn from documented diagnostic error literature, not random substitutions.
3. **Negation** — flipping present to absent. **Clinical motivation:** "no pneumothorax" vs "pneumothorax" is a one-word difference that changes clinical management from observation to chest tube.
4. **Region swap** — attributing a finding to the wrong anatomical region. **Clinical motivation:** "right lower lobe consolidation" vs "left lower lobe consolidation" changes surgical planning.
5. **Severity swap** — replacing "mild" with "severe". **Clinical motivation:** "small pneumothorax" (observe) vs "tension pneumothorax" (emergency decompression) determines whether a patient lives or dies.
6. **Temporal error** — confusing "new" with "unchanged" with "resolved". **Clinical motivation:** "new consolidation" triggers workup; "unchanged consolidation" does not. Misclassifying temporal status causes both over- and under-treatment.
7. **Device/line error** — wrong device identification or wrong position. **Clinical motivation:** "ETT in appropriate position" vs "ETT in right mainstem bronchus" determines whether a patient is adequately ventilated.
8. **Omission-as-support** — claim about a finding NOT present in the image. **Clinical motivation:** hallucinating a finding that doesn't exist triggers unnecessary procedures and patient anxiety.

**Difficulty curriculum:** Hard negatives are organized into three difficulty tiers (easy: negation + obvious swaps; medium: region/severity/temporal; hard: confusable pair substitution + subtle partial changes). Training begins with 60% easy / 30% medium / 10% hard, shifting to 20% easy / 40% medium / 40% hard by the final epoch. This prevents the verifier from overfitting on easy patterns while ensuring convergence.

**CheXpert label noise handling for Insufficient Evidence:** CheXpert labels have documented per-finding noise rates (e.g., fracture ~25% false negative, cardiomegaly ~6%). When constructing Insufficient Evidence examples from label absence, we weight each example by the label's estimated reliability. A "missing fracture label" example gets weight 0.75 (25% chance the label is wrong), while a "missing cardiomegaly label" gets weight 0.94. This prevents the verifier from learning that absent-but-actually-present findings are "insufficient."

Class balance is maintained by sampling equal numbers of Supported, Contradicted, and Insufficient Evidence examples per training batch, with Insufficient Evidence examples down-weighted by label confidence.

**Post-hoc temperature scaling:** After training, we learn a temperature parameter T on the calibration split via LBFGS on calibration NLL. ECE (calibrated): 0.0066 on CheXpert Plus.

### Inverted Conformal FDR Control (Contribution 2)

> **Key methodological finding (2026-04-05):** Standard cfBH (calibrate on faithful
> claims) fails on well-trained binary classifiers because softmax scores concentrate
> at the ceiling (~0.98), making conformal p-values uninformative. The INVERTED
> procedure — calibrating on CONTRADICTED claims as the null — resolves this.

**Procedure:** Null H0: "test claim j is contradicted." Calibration pool: contradicted
calibration claims (label=1). Score: s_j = P(Not-Contradicted) from temperature-
calibrated softmax. p-value: (|{cal_contra scores >= s_j}| + 1) / (n_cal_contra + 1).
Global BH at level alpha. Per-pathology p-values preserve exchangeability but BH
is applied globally (per-group BH too conservative for small pathology groups).

**Results (CheXpert Plus, 15K test claims):**
- alpha=0.05: FDR=1.30%, power=98.06%, n_green=9935
- alpha=0.10: FDR=2.48%, power=99.51%, n_green=10204
- alpha=0.20: FDR=5.59%, power=99.86%, n_green=10577

**Cross-dataset transfer (OpenI, 1784 test claims, zero-shot):**
- FDR controlled at every alpha level without retraining
- alpha=0.05: FDR=0.40%, power=41.58%
- alpha=0.20: FDR=5.65%, power=77.92%

Training details: AdamW optimizer, learning rate 2e-5, batch size 64, 15 epochs, linear warmup for 10 percent of steps.

### Component 6: Verifier-Guided Best-of-N Selection (Contribution 2)

For each input image we generate N candidate reports from the generator using nucleus sampling (p=0.9, temperatures sampled from {0.7, 0.8, 0.9, 1.0}). Each report is decomposed into claims, and each claim is scored by the verifier.

**Constrained selection objective:** Select the report that maximizes coverage subject to a faithfulness constraint:
- maximize: coverage(report)
- subject to: avg_faithfulness(report) >= tau_faith
- where avg_faithfulness = mean of verifier scores for all claims in the report
- where coverage = |CheXbert_detected_findings intersect report_mentioned_findings| / max(1, |CheXbert_detected_findings|)
- tau_faith in {0.80, 0.85, 0.90} — we report results at all three thresholds

For "No Finding" images (CheXbert detects no pathology), coverage is defined as a binary: 1 if the report correctly states no significant findings, 0 otherwise.

If no candidate meets the faithfulness threshold, we select argmax(avg_faithfulness) and flag the entire report for mandatory human review.

This constrained formulation replaces the multiplicative heuristic (faithfulness times coverage) used in earlier versions. The multiplicative form has pathological tradeoffs: it prefers slightly-hallucinating-but-complete reports (80 percent faithfulness, 100 percent coverage = 0.80) over faithful-but-incomplete reports (100 percent faithfulness, 50 percent coverage = 0.50). The constrained form ensures all accepted reports meet a minimum faithfulness bar, then optimizes for completeness.

We report results with both naive BoN and MBR-BoN (NAACL 2025), which adds a Minimum Bayes Risk proximity term to prevent reward hacking. The default is N=4 on the full test+calibration split (25K images, 100K candidates). We additionally run N=8 on a 10K-image subset to check for reward hacking patterns (quality degrading as N increases).

**CheXbert label noise sensitivity:** Because CheXbert labels have approximately 13 percent disagreement with radiologist labels (particularly for atelectasis and consolidation), we repeat the coverage computation using an ensemble of CheXbert, NegBio, and a GPT-4-based labeler, and report how much the selected reports and downstream metrics change.

### Component 7: Conformal Claim Triage (Contribution 3)

We target false discovery rate control among accepted (green-labeled) claims using the conformal Benjamini-Hochberg procedure (cfBH) from Jin and Candes (JMLR 2023, "Selection by Prediction with Conformal p-values").

**Data splits:** Three patient-disjoint pools: training patients (approximately 60 percent; generator training, verifier training, retrieval index population), calibration patients (approximately 15 percent; solely for threshold computation), and test patients (approximately 25 percent; all reported metrics). No information from calibration or test patients is accessible during any training or retrieval step.

**Handling within-report claim dependence:** Claims from the same report share the same image, patient, and radiologist, violating the exchangeability assumption required by conformal prediction. We address this by subsampling one claim uniformly at random per report for the calibration set. This yields approximately 34K independent calibration claims (one per calibration report), which is more than sufficient. We report the intra-report intraclass correlation coefficient (ICC) of verifier scores as a diagnostic: if ICC < 0.1, the violation is minor; if ICC > 0.3, we note the guarantees are materially weakened.

**cfBH procedure:**
1. For each test claim i, compute the conformal p-value using the upper tail: p_i = (|{j in calibration : s_j >= s_i}| + 1) / (n_cal + 1), where s_i is the temperature-scaled verifier score and the calibration set contains faithful claims only. This ensures that high faithfulness scores yield small p-values (strong evidence against the null of "unfaithful"), while low scores yield large p-values.
2. Apply the Benjamini-Hochberg step-up procedure at level alpha. Sort p-values: p_(1) <= p_(2) <= ... <= p_(m). Find the largest k such that p_(k) <= (k * alpha) / m. Accept (label green) all claims with p_i <= p_(k). This controls E[FDP] <= alpha under exchangeability.
3. Among non-accepted claims: those with s > tau_low are labeled yellow (review recommended), those with s <= tau_low are labeled red (likely hallucination).

**Pathology-group stratification:** We calibrate separate cfBH thresholds per pathology category (CheXpert 14 + No Finding + Support Devices ontology), provided that pathology group contains at least 200 calibration claims (after one-per-report subsampling). Groups with fewer than 200 calibration claims are merged into a Rare/Other pool which receives a single pooled threshold. Groups falling back to the pooled threshold are explicitly flagged as underpowered in reported results.

**Yellow/red boundary:** tau_low is set via sensitivity analysis at alpha_low in {0.15, 0.20, 0.25, 0.30, 0.40} and we report results at all five values. We acknowledge this boundary has no formal guarantee attached (unlike the green threshold which has FDR control).

**Validity statement:** Under exchangeability between calibration claims and test claims within each pathology group (ensured by one-per-report subsampling), the expected false discovery rate among green claims within that group is at most alpha. We report results at alpha = 0.05 and alpha = 0.10. This is a finite-sample distribution-free guarantee that does not require any distributional assumptions beyond exchangeability. It does not imply individual conditional validity and would not hold under significant dataset shift. Deployment in a new clinical setting would require recalibration on local data.

**The per-group guarantee is stated explicitly:** controlling FDR within each pathology group separately, not globally across all groups. This is the natural clinical interpretation (a clinician cares whether green cardiomegaly claims are reliable, not about some aggregate rate).

**Temporal shift experiment:** To empirically assess robustness to exchangeability violations, we split MIMIC-CXR by admission year, calibrate on earlier admissions, and test on later admissions. We report whether coverage degrades and by how much. This is a sanity check (if coverage breaks within a 5-year window at one institution, that is alarming) rather than evidence of robustness to arbitrary shift. We additionally discuss weighted conformal prediction (Barber et al., Annals of Statistics 2023) and CoFact (ICLR 2026) as paths toward shift-robust guarantees.

---

## Data Split Strategy

ALL splits are at the PATIENT level, not image level. Every image from the same patient must be in the same split.

| Split | Percentage | Purpose |
|-------|-----------|---------|
| Training | ~60% of patients | Generator training, verifier training, retrieval index |
| Calibration | ~15% of patients | Conformal threshold computation ONLY |
| Test | ~25% of patients | All reported metrics |

No information from calibration or test patients accessible during training or retrieval. Report exact patient counts, random seed, and split procedure for reproducibility. Verify no patient ID overlap across any split.

---

## Datasets (All Public)

| Dataset | Content | Size | Access | Use |
|---------|---------|------|--------|-----|
| CheXpert Plus | CXR images + reports + RadGraph-XL + CheXbert labels | 223K pairs, 64K patients | Stanford AIMI (free) | Training, calibration, test |
| PadChest-GR | Grounded radiology reports | 4,555 studies | Open access (NEJM AI 2024) | Grounding supervision + external validation |
| CheXpert Plus | CXR images + reports + RadGraph | 223K report-image pairs | Stanford (immediate) | External validation (primary) |
| IU X-Ray | CXR + reports (external) | 7,470 images | Open access | External validation (secondary, limitations acknowledged) |

---

## Evaluation

### Baselines (3)
1. Generator only (no verification or selection)
2. ConfLVLM (EMNLP 2025) — claim-level conformal with global thresholds (mandatory baseline)
3. PRM-style sentence verifier + weighted best-of-N (reproducing the PRM approach)

### Main Metrics
- CheXbert F1 and RadGraph F1 (clinical accuracy)
- Claim-level hallucination precision and recall, with laterality error rate reported separately
- Percentage of green claims versus error rate tradeoff curves at varying alpha
- Reliability diagrams for verifier score calibration (before and after temperature scaling)
- Stratified error and coverage by pathology category and by demographics (sex, age, race where available in MIMIC-CXR metadata)

### Statistics
- **Patient-level cluster bootstrap** 95 percent confidence intervals for all key comparisons (resampling patients with replacement, keeping all their reports and claims)
- Permutation test for primary claim-error reduction versus the strongest baseline
- Binomial confidence interval and p-value for observed error rate among green claims exceeding alpha

### Ablations (4)
1. Remove retrieval from verifier input (text-only claim scoring)
2. Remove hard negatives from verifier training (train with random negatives only)
3. N=4 versus N=8 versus N=4+MBR-BoN (on 10K-image subset)
4. Global threshold versus pathology-group thresholds (show underpowered group behavior)

### Additional Evaluations
- Standalone claim decomposer F1 plus reconstruction consistency (ROUGE-L)
- Grounding pointing game accuracy (PadChest-GR)
- Temporal shift experiment (calibrate on early admissions, test on later)
- CheXbert label noise sensitivity analysis (CheXbert vs ensemble labeling)
- External validation on CheXpert Plus and PadChest-GR
- Retrieval leakage diagnostics (BM25 similarity distributions + MinHash)
- Intra-report ICC of verifier scores (exchangeability diagnostic)
- Automated high-risk error audit on approximately 100 test studies stratified across pneumothorax, pleural effusion, cardiomegaly, and no finding
- End-to-end pipeline error audit: manually review 200+ green claims checking for decomposition errors, grounding errors, and retrieval failures that produce "correctly verified but clinically wrong" claims

---

## Known Limitations and Honest Assessment

### 0. Scope reduction from original proposal (v2.1)
The NeurIPS 2026 submission evaluates **text-only claim verification with conformal
FDR control** over radiology reports. The full multimodal pipeline described in the
original proposal (generator + grounding + Best-of-N selection) is future work. The
verifier is a text cross-encoder that processes (claim, evidence) pairs only — no
pixel-level vision is used. Ground-truth labels come from CheXbert (image-derived),
so the evaluation remains image-grounded even though the verifier is text-only.
Hybrid retrieval (MedCPT + BM25 + RoBERTa reranker) supplies evidence to the
verifier. We do not claim generator-level report-generation improvements; we claim
claim-level hallucination detection + calibrated triage.

### 1. Pipeline error cascade
The FDR guarantee holds for pipeline-assessed faithfulness (the verifier's judgment given its inputs), NOT for clinical faithfulness (whether the claim is actually true about the patient). Decomposition errors (e.g., dropping "no" from "no pleural effusion") can produce claims the verifier correctly assesses as supported but that misrepresent the original report. We mitigate via end-to-end error audit and decomposer reconstruction consistency check. The guarantee is explicitly conditional on correct decomposition.

### 2. Visual grounding is via heatmap, not raw pixels
The verifier receives image information through a CNN-encoded grounding heatmap (27x27 spatial attention map) rather than raw pixel features. This captures which anatomical regions the generator attended for each claim, providing genuine spatial signal. However, it does not process the full radiological detail visible in the raw image. A claim could be spatially consistent (attending the right region) but factually wrong about what is in that region. We chose heatmap-level vision over raw-pixel VLM integration because (a) DeBERTa cross-encoders have better probability calibration than VLMs, which is critical for conformal prediction, and (b) the heatmap provides interpretable spatial evidence. A fully pixel-level multimodal verifier remains future work.

### 3. Synthetic hard negatives do not fully represent real hallucinations
The 8 types of hard negatives are template-based perturbations. Real model hallucinations can be subtler (e.g., inventing vague findings, mixing temporal references). We mitigate by including an evaluation on actual model-generated outputs, checking whether the verifier catches real (not synthetic) errors. We report detection rates on genuine hallucinations separately from synthetic-negative performance.

### 4. Claim dependence at test time
Calibration uses one-per-report subsampling for exchangeability. At test time, multiple correlated claims from the same report are triaged. The Benjamini-Hochberg procedure controls FDR under positive regression dependence on subsets (PRDS, Benjamini and Yekutieli 2001). We conjecture that within-report claim scores satisfy PRDS (higher image quality improves all claims), but do not prove this formally. We report the intra-report ICC as a diagnostic and note that if ICC exceeds 0.3, the FDR guarantee is approximate. As a conservative option, we also report results with Benjamini-Yekutieli adjustment (controls FDR under arbitrary dependence at the cost of lower power).

### 5. Novelty is in the integration, not individual components
We are explicit: claim decomposition, retrieval-augmented verification, conformal prediction, and best-of-N selection all have strong prior art. ConfLVLM (EMNLP 2025) does claim-level conformal filtering with global thresholds. Conformal Alignment (NeurIPS 2024) does report-level FDR control. Our contribution is the principled integration of evidence-conditioned verification (external retrieval + spatial grounding descriptions) with pathology-stratified cfBH-based FDR control in a generation loop. We additionally contribute the subsampling-based exchangeability fix for within-report dependence and the analysis of when pathology stratification helps versus hurts.

### 6. CheXbert label noise (approximately 13 percent)
CheXbert labels are noisy ground truth. A missing CheXpert label does NOT mean the finding is absent — it may be a false negative. We mitigate via sensitivity analysis with ensemble labeling (CheXbert + NegBio + GPT-4) and explicitly state: "missing label does not equal confirmed false claim." We report how metrics change under the ensemble.

### 7. CheXpert ontology coverage (14 labels)
Many findings (masses, lymphadenopathy, surgical changes) are lumped into Rare/Other. We report what fraction of claims fall into this category and how pathology-stratified FDR control degrades for underpowered groups. The text-based NLI verifier partially compensates (operates on free text regardless of ontology), but the conformal guarantee for Rare/Other claims is weaker.

### 8. No human validation
We use automated proxies only. This is the most important gap. The triage labels ("green" = safe) are ultimately a claim about clinical safety that requires radiologist verification. Even a small blind review of 100 green vs red claims by a radiologist would substantially strengthen the paper. We note this as essential future work and do not claim clinical readiness.

### 9. Decomposer error propagation
LLM-based decomposition can break semantics (e.g., splitting "no left pleural effusion" into separate fragments). We mitigate with (a) the low-confidence merge heuristic, (b) reconstruction ROUGE-L checking, and (c) explicit statement that FDR guarantees apply to pipeline-extracted claims, not ground-truth report statements. We report decomposer error rates and their downstream impact on verification accuracy.

### 10. Exchangeability and distribution shift
All conformal guarantees are in-distribution only. The temporal shift experiment (calibrate on early admissions, test on later) is a sanity check, not a robustness proof. Deployment in a new institution requires recalibration. We discuss CoFact (ICLR 2026) and weighted conformal (Barber et al. 2023) as future directions for shift-robust guarantees.

---

## Compute Plan

| Platform | Use |
|----------|-----|
| Vast.ai (RTX 4090 rentals, ~$0.35-0.50/hr) | Generator training, best-of-N generation, baselines, ablations |
| Google Colab Pro ($12/mo) | Verifier, retriever, decomposer, grounding, evaluation |
| Local (M4 Mac Air) | Data preprocessing, FAISS index building, conformal calibration, plotting |

**Estimated total cost: $100-150** (RTX 4090 instead of A100; includes contingency for reruns).

| Job | GPU | Hours | Est. Cost |
|-----|-----|-------|-----------|
| Generator training | RTX 4090 | 40-50 | $16-20 |
| Best-of-4 generation (25K images x 4) | RTX 4090 | 50-60 | $20-24 |
| Verification pipeline on 100K candidates | RTX 4090 | 30-40 | $12-16 |
| N=8 ablation (10K images x 8) | RTX 4090 | 25-30 | $10-12 |
| Baselines (3) | RTX 4090 | 15-20 | $6-8 |
| Ablations (4) | RTX 4090 | 15-20 | $6-8 |
| Verifier training (RoBERTa-large) | Colab Pro T4 | 25-30 | included |
| Retriever (MedCPT off-the-shelf) + FAISS index | Local CPU | 2-3 | $0 |
| Grounding module | Colab Pro T4 | 10-12 | included |
| Claim decomposer | Colab Pro T4 | 6-8 | included |
| Evaluation + plotting | Local / Colab | 10-15 | included |
| Colab Pro subscription (2 months) | - | - | $24 |
| **Total** | | **~230-275h** | **$94-136** |

---

## Key References

- Angelopoulos et al., Conformal Alignment, NeurIPS 2024
- Barber et al., Conformal prediction beyond exchangeability, Annals of Statistics 2023
- Chung et al., VeriFact, NEJM AI 2025
- Hardy et al., ReXTrust, AAAI Bridge 2025
- Jin and Candes, Selection by Prediction with Conformal p-values, JMLR 2023
- Li et al., ConfLVLM, EMNLP 2025
- PRM for radiology, arXiv October 2025
- RadFlag, ML4H 2025
- FactCheXcker, CVPR 2025
- CCD, arXiv September 2025
- CoFact, ICLR 2026
- StratCP, medRxiv February 2026
- Conformal LM Reasoning, ICLR 2025
- Selective Generation, NeurIPS 2024
- FactMM-RAG, NAACL 2025
- MBR-BoN, NAACL 2025
- InferenceTimePessimism, ICML 2025
- Best-of-N reward hacking, ICML 2025
- BiomedCLIP negative result, December 2025

---

## V2 ADDENDUM (2026-04-09)

### Overview of V2 Changes

V2 addresses critical reviewer concerns identified through deep literature analysis
and adds multimodal capabilities:

### New Contribution: Multimodal Verification via BiomedCLIP Fusion

The v1 text-only verifier cannot detect hallucinations where text is plausible but
contradicts the image. V2 adds:

1. **BiomedCLIP image-claim scorer**: Frozen BiomedCLIP (15M biomedical image-text
   pairs) encodes (CXR image, claim) pairs. Trainable projection MLPs (~260K params)
   map both to shared 256-dim space. Cosine similarity -> sigmoid -> [0,1] score.

2. **Learned gating fusion**: Gate = sigmoid(Linear([text_score, image_score, |diff|])).
   Fused score = gate * text_score + (1-gate) * image_score. When no image available,
   gate = 1.0 (pure text-only, backward compatible with all v1 results).

3. **BiomedCLIP attention grounding**: Extract attention maps from final ViT layer
   to provide visual explanations for each verdict. No additional training.

### Upgraded Verifier Backbone: DeBERTa-v3-large

DeBERTa-v3-large (434M params) replaces RoBERTa-large (355M) as the text verifier:
- Disentangled attention (content-to-content, content-to-position, position-to-content)
- Enhanced mask decoder with absolute position in final layer
- Consistently +2-5% on NLI benchmarks (SemEval 2024 Task 2)
- Progressive domain adaptation: MNLI -> MedNLI -> RadNLI -> ClaimGuard hard negatives

### Upgraded Conformal Method: CoFact-Inspired Adaptive FDR

V1's inverted cfBH assumes exchangeability, which breaks under distribution shift
(power dropped to 42% on OpenI at alpha=0.05). V2 adds:

1. **KS test for shift detection**: Automatically detect cal/test distribution mismatch
2. **Density ratio estimation**: Logistic regression on [score, score^2, logit(score)]
   distinguishes calibration vs test distributions
3. **Reweighted p-values**: p_j = (sum w(s_i)*I(s_i >= s_j) + 1) / (sum w(s_i) + 1)
4. **Automatic fallback**: When distributions are similar (KS p > 0.05), use v1 method

Guarantee: FDR <= alpha under bounded density ratios (not strict exchangeability).

### New Validation: Artifact Analysis

**Hypothesis-only baseline**: Train DeBERTa on claims with evidence masked. If accuracy
>>66%, perturbations have surface shortcuts. Expected: 70-80% (some perturbations like
negation ARE detectable from claim alone — valid signal, not artifact).

### New Validation: Real Hallucination Test Set

Run CheXagent-8b on 500 OpenI test images. Compare generated reports to ground truth.
Label real hallucinations by type. Test verifier on these real errors (not just synthetic).

### Hackathon Web Demo

Gradio app on HF Spaces with ZeroGPU. Features:
- Per-claim green/yellow/red color coding
- Interactive alpha slider for FDR exploration
- End-to-end: CXR image -> CheXagent report -> ClaimGuard verification
- Baseline comparison panel

### New Files (V2)

| Purpose | Path |
|---------|------|
| V2 architecture | `ARCHITECTURE_V2.md` |
| DeBERTa verifier model | `models/verifier/deberta_verifier.py` |
| DeBERTa training script | `scripts/modal_train_verifier_deberta.py` |
| BiomedCLIP fusion | `models/multimodal/biomedclip_fusion.py` |
| CoFact conformal | `inference/cofact_conformal.py` |
| Hypothesis-only baseline | `scripts/baseline_hypothesis_only.py` |
| Real hallucination generator | `scripts/generate_real_hallucinations.py` |
| Gradio demo | `demo/app.py` |
- RadJEPA, arXiv January 2026
- RadPhi-3, arXiv November 2024
- MedCPT, Bioinformatics 2023
- RadGraph-XL, ACL 2024
- RadAnnotate, 2025
- PadChest-GR, NEJM AI 2024
- CheXpert Plus, Stanford 2024
- ReXErr, PSB 2025
- Phrase-Grounded Fact-Checking, MICCAI 2025

---

## Reproducibility

Code, trained verifier weights, configuration files, and exact data split patient ID lists will be released. All hyperparameters, random seeds, and training details are specified above. The release will include a clear statement that this is a research prototype not intended for clinical use.
