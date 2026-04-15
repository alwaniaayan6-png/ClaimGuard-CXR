# Handoff: ClaimGuard-CXR — 9 Improvements (No-Radiologist Track)

**Date:** 2026-04-14
**Status:** active — design phase complete, implementation pending
**Last session model:** claude-opus-4-6[1m]

## Current State (1-paragraph summary)

ClaimGuard-CXR is a claim-level hallucination detection system for chest X-ray radiology reports with formal FDR control. The system works: v1 RoBERTa-large binary verifier achieves 98.31% accuracy on a synthetic benchmark with FDR=1.30% at alpha=0.05 and 98.06% power via an inverted conformal Benjamini-Hochberg procedure. Cross-dataset transfer to OpenI holds FDR at every alpha level. A provenance-aware evidence-trust-tier gate was just added (completed in the previous session, 36 passing tests) to prevent self-consistency loops where the same model generates both the claim and the "evidence." The work is published on GitHub and the v1 checkpoint is stored locally at `checkpoints/v1_best_verifier.pt` (1.6GB). **The critical problem going forward is that the 98.31% number is partially artifact-driven**: a hypothesis-only baseline (same verifier, evidence masked) reaches 97.71%, meaning most of the signal comes from lexical shortcuts in the synthetic hard negatives, not from evidence reasoning. The eight synthetic perturbation types also miss entire classes of real-world hallucinations (fabricated measurements, fabricated priors, compound errors, missed-finding omissions, style-matched fabrications). This handoff defines 9 specific improvements that can be made **without radiologist annotation**, targeting MLHC Clinical Abstracts (April 17), NeurIPS D&B Pilot Track, MICCAI workshops, and Regeneron STS. The goal is to transform the paper from "we built a verifier and evaluated it on synthetic data" into "we built a verifier, identified its failure modes through silver-standard real-world evaluation, and extended it to cover a larger fraction of real hallucination classes." Budget: $900 total.

## What's Running / In Progress

Nothing is running. All previous RunPod and Modal jobs have completed or been terminated. No background processes. The v1 checkpoint at `/Users/aayanalwani/VeriFact/verifact/checkpoints/v1_best_verifier.pt` is usable immediately for inference.

## Completed Tasks

- [x] **v1 verifier trained and evaluated** (completed 2026-04-04 through 2026-04-09)
  - Binary RoBERTa-large, 98.31% accuracy on CheXpert Plus 15K test claims
  - FDR = 1.30% at alpha = 0.05, power = 98.06%
  - Cross-dataset OpenI: 85.09% accuracy, FDR controlled at every alpha
  - 4 baselines: rule-based, untrained RoBERTa, zero-shot LLM judge, CheXagent-8b (all ~66%)
  - Key files: `scripts/modal_train_verifier_binary.py`, `scripts/modal_run_evaluation.py`, `scripts/prepare_eval_data.py`
  - Decisions: binary > 3-class (3-class confused Supported/Insufficient at AUC=0.66), no label smoothing (0.05 creates softmax ceiling collapsing conformal scores), inverted cfBH (calibrate on contradicted not faithful, otherwise n_green=0)

- [x] **v2 DeBERTa + progressive NLI (partial)** (2026-04-09 through 2026-04-10)
  - Trained on RunPod A100 then L40S. Progressive NLI chain MNLI -> MedNLI -> ClaimGuard.
  - Best val acc: 98.61% (ClaimGuard epoch 3)
  - Eval on test: FDR=1.66% at alpha=0.05, power=99.32% (+1.26pp over v1)
  - Hypothesis-only baseline: **98.15% test accuracy** — critical artifact finding
  - DeBERTa NLI zero-shot baseline: 54.17%
  - RadFlag self-consistency baseline: 65.16%
  - **Checkpoints LOST** — RunPod community cloud reclaimed the pods before download. v2 verifier checkpoint no longer exists anywhere. Only result JSONs survived (see `results/baselines/`, `results/v2_runpod/captured_results.json`).
  - Key files: `scripts/modal_progressive_nli.py`, `scripts/modal_train_verifier_deberta.py`, `runpod/run_progressive_nli.py`

- [x] **Provenance-aware evidence gate** (completed 2026-04-14, this session)
  - New module `inference/provenance.py` (~300 lines) defining `EvidenceSourceType`, `TrustTier`, `ProvenanceTriageLabel`, `classify_trust_tier`, `apply_provenance_gate`, `ensure_provenance_fields`
  - `scripts/prepare_eval_data.py` stamps `evidence_source_type = oracle_report_text` + trust tier on every example via `default_provenance()`
  - `scripts/modal_build_retrieval_eval.py` stamps `retrieved_report_text` + `independent` tier when swapping oracle for MedCPT-retrieved evidence
  - `scripts/modal_run_evaluation.py` carries trust tiers through inference, applies the gate in the triage step, emits `provenance_gate` and `per_trust_tier` breakdowns per alpha level in `full_results.json`
  - `inference/conformal_triage.py` gained `gate_triage_with_provenance()` helper plus extended `TriageResult` with `trust_tier` and `final_label` fields (backward-compat defaults to `unknown`)
  - `demo/app.py` gained a provenance dropdown, 5-tier triage labels (supported_trusted / supported_uncertified / review_required / contradicted / provenance_blocked), updated color map, updated examples
  - **36 tests passing** at `tests/test_provenance.py` covering all (label × tier) combinations, fixture cases, backward compat, literal-string locks for the Modal-inlined version
  - Docs updated: `REPRODUCIBILITY.md` got a new "Scope and trust model" section + caveats 5-6, `MANUSCRIPT_MINI.md` got Limitation (6)
  - **Policy:** same-model or unknown provenance is always downgraded to supported_uncertified regardless of verifier score or BH acceptance

- [x] **GitHub repo published (public, clean)**
  - https://github.com/alwaniaayan6-png/ClaimGuard-CXR
  - Single-commit squashed history, zero Claude/Anthropic references
  - `HACKATHON_PLAN.md` committed with Gradio + HF Spaces + ZeroGPU plan

- [x] **MLHC 2026 Clinical Abstract drafted** with real v1 numbers at `paper/mlhc_abstract.md` (deadline April 17 — 3 days away)

## Pending Tasks (priority order)

The 9 improvements below are the scientific backbone of the next sprint. Each is scoped to be executable without radiologist annotation, stays within the $900 budget, and produces a measurable result. They're ordered by dependency + impact, not by the numbering in the conversation.

### 1. **Silver-standard real-hallucination evaluation** (HIGHEST PRIORITY — the single most valuable experiment)
- **Why:** The paper's biggest gap is that every reported number is on synthetic perturbations. A silver-standard pilot using ensemble grader agreement is the first credible real-world measurement and unlocks most other improvements.
- **Plan:**
  1. Run CheXagent-8b on 200 OpenI test images (scaffold exists at `scripts/generate_real_hallucinations.py`). Extract atomic claims from the generated reports.
  2. For each claim, run three independent graders:
     - CheXbert labeler on original + generated report, diff the 14-dim label vectors
     - GPT-4 or Claude Sonnet with vision on the (image, claim) pair, ask "consistent?"
     - A second VLM (RadFM or CXR-LLaVA) asked the same question
  3. A claim is `silver_contradicted` if 2/3 graders flag it, `silver_supported` if 0/3, `silver_uncertain` if exactly 1.
  4. Filter to claims with unanimous or near-unanimous agreement (~60-70% of the set).
  5. Run v1 verifier on the filtered pilot. Report: accuracy, per-hallucination-type breakdown (categorized via CheXbert label diffs), hypothesis-only baseline on the pilot, comparison to synthetic.
- **Expected result:** Accuracy drops from 98.31% (synthetic) to ~75-85% (real). Hypothesis-only drops from 97.71% to ~60-70%. These are the honest numbers that make the paper credible.
- **Blocked by:** Nothing. Can start immediately.
- **Files to touch:** `scripts/generate_real_hallucinations.py` (finish), `scripts/silver_standard_eval.py` (new), `inference/silver_graders.py` (new), `results/silver_pilot/` (new output dir)
- **Cost:** $30-50 in API (GPT-4V or Claude Sonnet vision), $5-10 in Modal compute
- **Critical framing:** Paper must call this a SILVER-STANDARD PILOT, not a gold-standard benchmark. Reviewers will accept silver standards as pilot-scale evidence when explicitly framed.

### 2. **Extended synthetic hard-negative taxonomy** (fabricated measurements, fabricated priors, compound errors)
- **Why:** The current eight perturbation types don't cover entire classes of real hallucinations. Adding three new types broadens coverage without needing any annotation.
- **Plan:**
  1. **Fabricated measurements:** extend `scripts/prepare_eval_data.py::generate_hard_negative` with a new `fabricated_measurement` type. Take a claim mentioning a device (ETT, central line, NG tube) and inject a plausible but wrong numeric value (`"ETT in appropriate position"` → `"ETT 4.7 cm above carina in appropriate position"`). Evidence doesn't contain any measurement.
  2. **Fabricated priors:** new `fabricated_prior` type. Prepend references to nonexistent prior exams (`"Heart size is normal"` → `"Compared to the exam of 2023-04-17, heart size is unchanged and remains normal"`).
  3. **Compound perturbations:** new `compound_perturbation` type. Apply two of the existing eight types in sequence (laterality + severity, negation + temporal). Validated via `_evidence_supports_claim_text` to avoid label noise.
  4. Regenerate the 30K training set + 15K cal + 15K test with ~10% each of the new types.
  5. Retrain v1 on the extended set via `scripts/modal_train_verifier_binary.py`.
  6. Re-run `modal_run_evaluation.py`. Report old vs new confusion matrix broken out by hallucination type.
- **Expected result:** Overall accuracy drops 3-8pp (honest), but per-type accuracy is reported for the first time. The drop IS the result.
- **Blocked by:** Nothing.
- **Files to touch:** `scripts/prepare_eval_data.py` (add 3 generators), `data/augmentation/hard_negative_generator.py`, `scripts/modal_train_verifier_binary.py` (re-run), `scripts/modal_run_evaluation.py` (re-run)
- **Cost:** $10-20 in Modal compute (one retraining + one eval)

### 3. **Paraphrase-preserving augmentation to attack the lexical shortcut**
- **Why:** Hypothesis-only baseline hits 97.71% because the hard negatives are one-token swaps. Adding paraphrased positives forces the verifier to learn that lexical variation doesn't imply contradiction.
- **Plan:**
  1. For each supported claim in the training set, generate 2-3 paraphrases via Claude Haiku or GPT-4o-mini. Same meaning, different surface form.
  2. For each contradicted claim, paraphrase the perturbation while preserving the wrong meaning.
  3. Add paraphrased positives and negatives to the training set (3-4x expansion).
  4. Retrain v1.
  5. Re-run the hypothesis-only baseline on the augmented eval set. Measure the delta.
- **Expected result:** Hypothesis-only drops from 97.71% toward 80-85%, main accuracy drops from 98.31% toward 92-95%. The HO drop is the artifact fix.
- **Blocked by:** Nothing, but synergizes with task 2 (extended perturbations should be added before augmentation so both get paraphrased).
- **Files to touch:** `scripts/paraphrase_augment.py` (new), `scripts/prepare_eval_data.py` (hook into pipeline)
- **Cost:** $10-30 in API (10K paraphrases at Claude Haiku pricing), $5 in Modal retraining

### 4. **Existence-check deterministic safety overrides**
- **Why:** Some hallucination classes (fabricated measurements, fabricated priors) are structural enough that a regex pass catches them at zero training cost. Trade-off: low precision, high recall on that specific failure mode.
- **Plan:**
  1. New module `inference/existence_check.py`.
  2. Patterns:
     - Numeric fabrication: regex `\b\d+\.?\d*\s?(cm|mm|%)\b`. If claim has, evidence doesn't → flag.
     - Prior-exam fabrication: regex `(compared to (prior|exam|study)|interval (change|decrease|increase))`. Same rule.
     - Date fabrication: regex `\d{4}-\d{2}-\d{2}` or month names. Same rule.
  3. Wire into `inference/conformal_triage.py` as a post-processing override alongside the provenance gate. If existence check fires, final label = `review_required`.
  4. Unit tests in `tests/test_existence_check.py`.
  5. Ablation: with vs without existence check on the extended synthetic benchmark and the silver pilot.
- **Expected result:** Measurable recall improvement on fabricated-measurement and fabricated-prior classes. Some false alarms on legitimate measurements.
- **Blocked by:** Nothing.
- **Files to touch:** `inference/existence_check.py` (new), `inference/conformal_triage.py` (wire override), `tests/test_existence_check.py` (new)
- **Cost:** $0

### 5. **Retrieval quality improvements (batched BM25 + cross-encoder reranker)**
- **Why:** Currently using dense-only MedCPT because BM25 was per-query slow. Hybrid retrieval + reranker is standard for RAG and you're leaving 2-5pp on the table.
- **Plan:**
  1. Rewrite `models/retriever/bm25_index.py` to batch-score all test queries at once via vectorized posting-list scoring instead of per-query `rank_bm25` calls. Target throughput: 1000 queries in ~100ms.
  2. Hybrid fusion via reciprocal rank fusion (RRF) combining MedCPT top-20 and BM25 top-20 into a final top-2.
  3. Wire the cross-encoder reranker at `models/retriever/reranker.py` into the eval pipeline. Fine-tune a DeBERTa-base cross-encoder on (query, passage) pairs from the training set. Run over top-20 candidates, take top-2.
  4. Retrieval ablation table: dense only / dense+BM25 / dense+reranker / dense+BM25+reranker with accuracy, FDR, power for each.
- **Expected result:** ~1-3pp accuracy improvement on retrieval-augmented eval. Retrieval ablation table is table-stakes for any RAG paper.
- **Blocked by:** Nothing.
- **Files to touch:** `models/retriever/bm25_index.py`, `models/retriever/reranker.py`, `scripts/modal_train_reranker.py` (new), `scripts/modal_run_evaluation.py` (add retrieval branches)
- **Cost:** $10-20 in Modal compute

### 6. **Recalibrated + pathology-stratified OpenI analysis**
- **Why:** Current OpenI cross-dataset story is zero-shot and pooled. Recalibrating on OpenI (half cal, half test) and stratifying by pathology gives a much richer transfer narrative.
- **Plan:**
  1. Split OpenI 50/50 into cal/test at patient level.
  2. Run inverted cfBH calibration on the OpenI cal half. Evaluate on OpenI test half. Report FDR/power.
  3. Also keep the zero-shot story (CheXpert cal, OpenI test) for comparison.
  4. Stratify OpenI results by CheXpert-14 pathology group. Report per-group accuracy + FDR + power.
- **Expected result:** Recalibrated bounds tighter than zero-shot. Per-group analysis surfaces whether some pathologies generalize worse than others. Richer cross-dataset story.
- **Blocked by:** Nothing.
- **Files to touch:** `scripts/modal_run_evaluation.py` (add OpenI cal split), `scripts/convert_openi_to_chexpert_schema.py` (add split logic)
- **Cost:** ~$2 in Modal compute

### 7. **LLM-based claim extractor wired in**
- **Why:** The current regex sentence splitter breaks on compound sentences, anaphora, and temporal clauses. A Phi-3-mini prompt-based extractor is scaffolded at `models/decomposer/llm_claim_extractor.py` but never integrated.
- **Plan:**
  1. Wire `models/decomposer/llm_claim_extractor.py` into `scripts/prepare_eval_data.py` (optional flag, default off for backward compat) and `demo/app.py` (runtime claim extraction).
  2. Measure round-trip reconstruction fidelity: extract claims, reassemble, compare to original. Higher = better.
  3. Run CPU inference (Phi-3-mini is small enough). No GPU needed.
  4. Ablation: verifier accuracy with rule-based vs LLM-based extraction on the same eval set.
- **Expected result:** Slightly different (cleaner, more precise) claim boundaries. Headline accuracy may shift by ~1-2pp either direction. Round-trip reconstruction fidelity is the cleaner metric.
- **Blocked by:** Nothing.
- **Files to touch:** `models/decomposer/llm_claim_extractor.py`, `scripts/prepare_eval_data.py`, `demo/app.py`
- **Cost:** $0 (CPU)

### 8. **Self-annotation validation pass (100 claims by you)**
- **Why:** You are not a radiologist, but for the specific task of "does this generated claim match the original radiologist report on the same image?" you can do better than chance. Pairing your labels with the silver standard (task 1) is an internal validity check on silver-standard quality.
- **Plan:**
  1. Sample 100 claims from the silver-standard pilot (task 1 output).
  2. Label each yourself: Supported / Contradicted / Novel-Plausible / Novel-Hallucinated / Uncertain.
  3. Compare to silver-standard labels. Report Cohen's kappa.
  4. If agreement is >80%, silver standard is validated as pilot-quality. If lower, acknowledge silver-standard noise explicitly in the paper.
- **Expected result:** Kappa ~0.65-0.80 with silver standard. Not radiologist-grade, but pilot-valid.
- **Blocked by:** Task 1 must complete first.
- **Files to touch:** `paper/self_annotation_log.md` (new, for the labeling work), `scripts/compile_annotated_results.py` (extend to handle self-labels)
- **Cost:** $0 and 3-5 hours of your time

### 9. **Provenance-gate failure-mode demonstration (self-consistency loop experiment)**
- **Why:** The provenance gate was added in the previous session but there's no empirical demonstration that the failure mode it prevents is real. A small experiment proves the gate catches something.
- **Plan:**
  1. Run CheXagent on 50 OpenI images to generate radiology reports (= claims).
  2. Run CheXagent **again** on the same images with a different prompt asking it to "describe the visual findings that support the claim X" for each claim. These are same-model "evidence" outputs.
  3. Feed the (claim, same-model evidence) pairs through the v1 verifier. Record verifier scores.
  4. Hypothesis: scores will be high (the model agrees with itself). Even when the claims are factually wrong (validated against the original OpenI ground-truth report).
  5. Show that the provenance gate correctly downgrades all these claims to `supported_uncertified` despite the high verifier score.
  6. Quantify: "N pairs with verifier score > 0.9 that would have been labeled supported_trusted without the gate are all correctly downgraded to supported_uncertified."
- **Expected result:** Empirical validation of the provenance gate. Concrete numbers for the paper: "X% of same-model-generated evidence pairs receive verifier scores > 0.9, all of which are correctly downgraded by the gate."
- **Blocked by:** Nothing, but thematically depends on task 1 being in progress.
- **Files to touch:** `scripts/same_model_experiment.py` (new), `results/same_model_experiment/` (new)
- **Cost:** $5-10 in Modal compute + $5 in API if using external grader for ground-truth correctness check

---

## Key Files & Locations

| Purpose | Path | Notes |
|---|---|---|
| Architecture doc (v1) | `ARCHITECTURE.md` | **READ FULLY**. 1700+ lines. The system blueprint. |
| Architecture doc (v2) | `ARCHITECTURE_V2.md` | **READ FULLY**. Planned extensions including multimodal + CoFact. |
| Research proposal | `CLAIMGUARD_PROPOSAL.md` | **READ FULLY**. Every design decision with rationale. |
| Reproducibility guide | `REPRODUCIBILITY.md` | **READ FULLY**. End-to-end pipeline with commands + the new Scope and trust model section. |
| Mini manuscript | `MANUSCRIPT_MINI.md` | **READ FULLY**. 5-page draft, includes Limitation (6) for provenance. |
| MLHC 2026 abstract draft | `paper/mlhc_abstract.md` | **READ FULLY**. Deadline April 17. |
| Hackathon plan | `HACKATHON_PLAN.md` | Gradio + HF Spaces + ZeroGPU deployment plan. |
| v1 verifier checkpoint | `checkpoints/v1_best_verifier.pt` | **1.6GB, LOCAL, do not commit.** RoBERTa-large binary verifier. 98.31% test acc. |
| Binary training (v1) | `scripts/modal_train_verifier_binary.py` | Primary training script. No label smoothing. |
| Eval pipeline | `scripts/modal_run_evaluation.py` | **READ FULLY**. Most complex file. Inverted cfBH + provenance gate + per-tier metrics. |
| Data generation | `scripts/prepare_eval_data.py` | **READ FULLY**. 8 hard-neg types + provenance stamping. |
| Retrieval eval builder | `scripts/modal_build_retrieval_eval.py` | Swaps oracle for MedCPT evidence, stamps `retrieved_report_text` + `independent`. |
| Real hallucination generator | `scripts/generate_real_hallucinations.py` | Scaffold only — annotation workbook generator. Task 1 uses this. |
| Provenance module | `inference/provenance.py` | **READ FULLY**. Trust tiers, gate function, data contract. ~300 lines. |
| Conformal triage | `inference/conformal_triage.py` | **READ FULLY**. Has `gate_triage_with_provenance` helper. |
| Demo app | `demo/app.py` | Gradio app with provenance dropdown + 5-tier labels. |
| Hard-negative generator | `data/augmentation/hard_negative_generator.py` | Task 2 extends this with 3 new types. |
| Retriever | `models/retriever/medcpt_encoder.py` | Dense retrieval. |
| BM25 index | `models/retriever/bm25_index.py` | Sparse retrieval. **Task 5 rewrites this.** |
| Reranker | `models/retriever/reranker.py` | Cross-encoder reranker, scaffolded, not wired. **Task 5 wires this.** |
| LLM claim extractor | `models/decomposer/llm_claim_extractor.py` | Phi-3-mini prompt-based extractor, scaffolded, not wired. **Task 7 wires this.** |
| Provenance tests | `tests/test_provenance.py` | 36 passing tests. Run via `python3 tests/test_provenance.py` |
| V1 captured results | `results/v2_runpod/captured_results.json` | V1 numbers + partial v2 training log. |
| V1 baselines | `results/baselines/*.json` | Hypothesis-only, DeBERTa NLI, RadFlag. |
| Training data (v2) | `/Users/aayanalwani/data/claimguard/verifier_training_data_v2.json` | 30K examples, 9.6MB |
| Cal/test claims | `/Users/aayanalwani/data/claimguard/eval_data/{calibration,test}_claims.json` | 15K each, ~4.8MB each |
| OpenI images | `/Users/aayanalwani/data/openi/*.png` | 7,470 PNGs, 2.6GB |
| OpenI CSV | `/Users/aayanalwani/data/openi/openi_cxr_chexpert_schema.csv` | Converted to CheXpert schema |

## Critical Decisions & Why

- **Binary classification, not 3-class.** 3-class verifier confused Supported and Insufficient catastrophically (AUC=0.66 on the Insufficient vs Supported boundary, 3786/5000 Insufficient predicted as Supported). Binary framing captures the clinically critical question: "is this hallucinated?" 98.31% accuracy. Do not revisit.

- **No label smoothing (0.0).** Label smoothing at 0.05 creates a hard softmax ceiling at P=0.975 that collapses the conformal score distribution and breaks BH (n_green=0 at every alpha). Discovered after two failed training runs. Do not revisit.

- **Inverted cfBH (calibrate on contradicted).** Standard cfBH calibrates on faithful claims, but faithful scores cluster at the softmax ceiling (~0.98) making p-values uninformative. Inverted version calibrates on contradicted scores near 0.01, which are well-separated. This is the paper's methodological contribution.

- **Global BH, not per-pathology.** Per-group BH is too conservative because small groups have min_p > local BH threshold, so it always accepts zero claims. Per-pathology p-values are computed for within-strata exchangeability, but BH is applied globally across all test claims.

- **Dense-only retrieval at eval time.** `rank_bm25` per-query cost is O(corpus) (~2s per query for 1.2M passages), infeasible at 30K eval claims. Dense MedCPT only loses 0.22pp. **Task 5 revisits this with batched BM25 + reranker.**

- **Text-only verifier, provenance-gated.** The v2 multimodal attempt (DeBERTa + CheXzero fusion) was planned but the v2 checkpoints died with RunPod pods. The current system is text-only with a provenance gate that refuses to certify claims under same-model or unknown evidence. A multimodal version is explicitly out of scope for the no-radiologist track — it requires CheXpert Plus image access (450GB) and significant additional compute.

- **Provenance gate downgrades to `supported_uncertified`, not `provenance_blocked`.** The softer downgrade lets downstream consumers see that the verifier scored the claim positively, but explicitly denies certification. Hard blocking is reserved for a future audit-mode label.

- **Silver-standard pilot framed explicitly as a pilot.** Task 1 produces silver-standard labels, not gold standard. The paper must call this out. Silver standards are publishable at pilot venues (MLHC, MICCAI workshops, NeurIPS D&B) but not at Nature Medicine or Lancet Digital Health. That ceiling is accepted and acknowledged.

- **Budget $900, no radiologist.** Attending radiologists cost $200-500/hr. At $900 the only options for human annotation are medical student / resident via lab connections ($15-50/hr) or self-annotation. The 9 improvements deliberately avoid the radiologist bottleneck.

## Known Issues / Gotchas

- **v2 DeBERTa checkpoints are lost.** RunPod community cloud reclaimed the pods before downloading the trained weights. Only result JSONs + log scrapings survived. If you want to retrain v2, use secure cloud (not community) and push checkpoints to HF Hub after every epoch. Do NOT rely on local pod disk.
- **CheXagent requires `transformers==4.40.0`** exactly. Newer versions removed `find_pruneable_heads_and_indices` that CheXagent's custom code imports. Pin the version in any Modal image.
- **Zero-shot LLM judge baseline had 30% API error rate** due to rate limiting at concurrency=10. Errors were treated as pred=0 (majority class). Results are still comparatively valid but absolute numbers are conservative.
- **`configs/verifier.yaml` is STALE.** Code does not read it. All hyperparameters are hardcoded in the training scripts. Ignore the YAML.
- **`/tmp/v2_final/`, `/tmp/v2_retr/`, `/tmp/openi_res/` may be cleared on Mac reboot.** Re-download from Modal volume with `python3 -m modal volume get claimguard-data /eval_results_binary_v2/ /tmp/v2_final/ --force`.
- **Two leaked API keys in the original chat history.** Both should be revoked. Current key is in `~/.claude-secrets/` (legacy path, rename if any active work).
- **DeBERTa-v3 fails on `token_type_ids`** because `type_vocab_size=0`. The provenance patch removed the token_type_ids pass-through from the v2 verifier wrapper. Do not re-add.
- **ClaimGuard stage 3 OOMs on 45GB GPUs** at batch_size=16 with max_length=512. Use batch_size=4 + grad_accum=16 for L40S / A6000 / similar. A100 80GB can use batch_size=16.
- **`prepare_eval_data.py` evidence-supports-claim validator (Bug C6)** rejects ~20% of proposed contradicted pairs because the perturbed claim still matches evidence tokens. This is a feature, not a bug — it prevents label noise. Expect the hard-neg generation loop to take more attempts than `n_contradicted` suggests.
- **Provenance gate test file imports work with `python3 tests/test_provenance.py`** but may not work under `pytest` if the parent path isn't on sys.path. The test file adds the path manually as a safeguard.
- **Modal cost cap may hit billing limit.** Previous sessions spent ~$16 of $200 cap. Budget should be fine for the new work but monitor after each run.

## Environment

- Python: 3.9 (system, macOS) locally; 3.11 on Modal container images
- PyTorch: 2.1+ (local), 2.4 (Modal)
- Transformers: 4.40.0 pinned for CheXagent; 4.42+ fine for RoBERTa/DeBERTa only
- Key packages: torch, transformers, faiss-cpu, rank_bm25, scikit-learn, scipy, numpy, pandas, anthropic (for LLM grader/paraphrase)
- Modal: app `claimguard-evaluation`, `claimguard-verifier-binary`, etc. Volume `claimguard-data`. H100 for GPU jobs per user policy unless job is definitely <15min.
- Hugging Face: HF Hub public repo ready, model hosting for `v1_best_verifier.pt` is a TODO
- OS: macOS Darwin 25.0.0, single Mac workstation
- No conda/venv — system Python + pip install --user
- GitHub: https://github.com/alwaniaayan6-png/ClaimGuard-CXR (public)

## Reference Materials — MUST READ FULLY

A fresh session has **zero project context**. Read every document below **completely**, not just the summary. Do not skim design docs or the proposal.

### User's Global Instructions (READ FIRST, always)
- `~/.claude/CLAUDE.md` — **READ FULLY**. User's global preferences: model selection (NEVER use Sonnet for reasoning/research — Opus only for intellectual work), H100-only GPU policy, doc-sync requirement (update ARCHITECTURE.md + PROPOSAL.md whenever code changes), autonomy preferences (never ask user to debug, self-check all code, run overnight if needed), plain text (no LaTeX in chat), communication style (direct and concise, user is an advanced HS junior). Every rule is mandatory and supersedes defaults.

### Architecture & Design Docs
- `ARCHITECTURE.md` — **READ FULLY**. v2.1 scope reduction for NeurIPS 2026. System blueprint covering components 1-7 (generator, decomposer, grounding, retriever, verifier, best-of-N, conformal triage). The verifier section (Component 5) and conformal triage section (Component 7) are the ones actively in use.
- `ARCHITECTURE_V2.md` — **READ FULLY**. Covers the planned multimodal extensions (CheXzero fusion, CoFact adaptive conformal) that were attempted but shelved due to checkpoint loss. Valuable for understanding the multimodal direction.
- `CLAIMGUARD_PROPOSAL.md` — **READ FULLY**. Research proposal with every design decision annotated. Pay especial attention to the scope reduction notes and the inverted-cfBH derivation.
- `REPRODUCIBILITY.md` — **READ FULLY**. End-to-end pipeline with commands. Includes the new Scope and trust model section explaining the provenance gate policy.
- `MANUSCRIPT_MINI.md` — **READ FULLY**. 5-page draft of the paper in its current form. Includes Limitation (6) for the provenance gate and text-only constraint.
- `HACKATHON_PLAN.md` — Skim. Deployment plan for the Gradio demo.
- `paper/mlhc_abstract.md` — **READ FULLY**. MLHC 2026 Clinical Abstract with April 17 deadline.

### Literature / Prior Work — fetch from arXiv if not local

Every paper below must be **read in full** before touching the relevant subsystem. Do not rely on the summaries in the proposal or the previous session's memory.

- **Jin & Candes (JMLR 2023)** — "Selection by Prediction with Conformal p-values." arXiv:2210.01408. Foundation for the cfBH procedure. The inverted variant is the paper's key methodological contribution.
- **Bates et al. (Annals of Statistics 2023)** — "Testing for Outliers with Conformal p-values." arXiv:2104.08279. Related conformal testing framework.
- **Barber et al. (Annals of Statistics 2023)** — "Conformal Prediction Beyond Exchangeability." Treats non-exchangeable cases relevant to cross-dataset transfer.
- **Benjamini & Hochberg (JRSS-B 1995)** — Original BH procedure for FDR control. Classic reference.
- **Guo et al. (ICML 2017)** — "On Calibration of Modern Neural Networks." Temperature scaling for calibration.
- **Chung et al. (NEJM AI 2025)** — "VeriFact." LLM-as-Judge for EHR text. The main competitor. ClaimGuard differs by using a trained verifier with formal FDR guarantees.
- **Rao et al. (PSB 2025)** — "ReXErr." arXiv:2409.10829. 12-type radiologist-validated error taxonomy. The most important reference for the extended-perturbation work (task 2). **Read this in full before implementing task 2.**
- **Chambon et al. (NeurIPS D&B 2024)** — "CheXpert Plus." Dataset paper for the primary training/eval corpus.
- **Chen et al. (2024)** — "CheXagent." arXiv:2401.12208. The VLM used for silver-standard real-hallucination generation in task 1.
- **Chen et al. (npj Digital Medicine 2025)** — Real hallucination rates in medical AI: 8-15% generally, dominated by omissions (3.45%) over fabrications (1.47%). Context for the hallucination-type coverage discussion.
- **Li et al. (EMNLP 2025)** — "ConfLVLM." Claim-level conformal for LVLMs with global thresholds. Competitor method.
- **Tiu et al. (Nature Biomedical Engineering 2022)** — "CheXzero." Expert-level zero-shot CXR pathology classification. Reference for any future multimodal work.
- **Hardy et al. (AAAI Bridge 2025)** — "ReXTrust." Hidden-state hallucination detection, AUROC 0.8963. Competitor.
- **RadFlag (ML4H 2025)** — Sampling-based hallucination flagging. Competitor baseline (you have a simplified version at `scripts/baseline_radflag_consistency.py`).
- **StratCP (medRxiv Feb 2026)** — Stratified conformal FDR for medical foundation models. Directly relevant to per-pathology stratification (task 6).
- **CoFact (ICLR 2026)** — Online density-ratio estimation for conformal factuality under distribution shift. Relevant to recalibration story (task 6).
- **Herlihy & Rudinger (ACL 2021)** — Hypothesis-only baselines for MedNLI. Direct reference for the artifact analysis (tasks 3 and 8).

**Critical:** After reading the literature, the new session should **evaluate whether the 9 improvements below are the right ones.** Are there techniques from 2026 papers that would be better? Should any of the 9 be dropped or reordered? Are there additional improvements that are obvious from the literature that the previous session missed? Report findings to the user before starting implementation.

### Existing Codebase — Read Before Editing

| Purpose | Path | Read completely? |
|---|---|---|
| Provenance module | `inference/provenance.py` | **YES — newest, most critical** |
| Conformal triage | `inference/conformal_triage.py` | **YES** |
| Eval pipeline | `scripts/modal_run_evaluation.py` | **YES — the most complex file** |
| Data generation | `scripts/prepare_eval_data.py` | **YES** |
| Retrieval eval builder | `scripts/modal_build_retrieval_eval.py` | **YES** |
| Binary trainer | `scripts/modal_train_verifier_binary.py` | **YES** |
| Provenance tests | `tests/test_provenance.py` | **YES — lock file for invariants** |
| Hard-negative generator | `data/augmentation/hard_negative_generator.py` | **YES — task 2 extends this** |
| Demo app | `demo/app.py` | **YES — reference for UI patterns** |
| MedCPT retriever | `models/retriever/medcpt_encoder.py` | YES if touching retrieval |
| BM25 index | `models/retriever/bm25_index.py` | YES — task 5 rewrites |
| Reranker | `models/retriever/reranker.py` | YES — task 5 wires |
| LLM claim extractor | `models/decomposer/llm_claim_extractor.py` | YES — task 7 wires |
| Real hallucination generator | `scripts/generate_real_hallucinations.py` | YES — task 1 extends |
| CheXbert labeler reference | Any existing CheXbert integration | Find and YES |
| Compile annotated results | `scripts/compile_annotated_results.py` | YES |
| Baselines | `scripts/baseline_*.py` | Skim all; read whichever you're extending |

### Data & Artifacts
- **Training data v2:** `/Users/aayanalwani/data/claimguard/verifier_training_data_v2.json` (30K examples, 9.6MB)
- **Eval data:** `/Users/aayanalwani/data/claimguard/eval_data/{calibration,test}_claims.json` (15K each, 4.8MB each)
- **OpenI images:** `/Users/aayanalwani/data/openi/*.png` (7,470 PNGs, 2.6GB)
- **OpenI schema CSV:** `/Users/aayanalwani/data/openi/openi_cxr_chexpert_schema.csv`
- **v1 checkpoint:** `checkpoints/v1_best_verifier.pt` (1.6GB, RoBERTa-large binary, 98.31% test acc) — **LOCAL ONLY, do not commit**
- **Modal volume:** `claimguard-data` — eval results, training data, retrieval indices, v1 checkpoint backup
- **Result files:** `results/baselines/*.json`, `results/v2_runpod/captured_results.json`

## How to Resume

Exact steps a fresh Claude session should take, **in order**:

1. **Read `~/.claude/CLAUDE.md` FULLY.** Model-selection rules, autonomy preferences, coding style. Mandatory.
2. **Read this handoff doc completely.** All sections, all tables.
3. **Read `ARCHITECTURE.md` and `ARCHITECTURE_V2.md` fully.** The system blueprints.
4. **Read `CLAIMGUARD_PROPOSAL.md` fully.** Every design decision annotated.
5. **Read `REPRODUCIBILITY.md` fully.** Pay attention to the Scope and trust model section.
6. **Read `MANUSCRIPT_MINI.md` fully.** Current paper state, including Limitation (6).
7. **Read `paper/mlhc_abstract.md`.** Deadline context.
8. **Fetch and read the full literature list above.** Use WebFetch or arXiv. The ReXErr paper and Jin & Candes paper are the two most critical for understanding the new work.
9. **Evaluate the 9 improvements below against the literature.** Do any of them need to be revised? Are there better techniques? Are there additional improvements obvious from recent papers that the previous session missed? Report the evaluation to the user before starting implementation.
10. **Explore the codebase.** Read every file marked "YES" in the Existing Codebase table. Do not skim. Pay special attention to `inference/provenance.py` (newest), `scripts/modal_run_evaluation.py` (most complex), and `tests/test_provenance.py` (invariant lock file).
11. **Run the test suite to verify current state:** `cd /Users/aayanalwani/VeriFact/verifact && python3 tests/test_provenance.py`. All 36 tests should pass. If any fail, something has regressed and must be fixed before new work starts.
12. **Verify the v1 checkpoint exists** at `checkpoints/v1_best_verifier.pt` and is 1.6GB. If missing, re-download from the Modal volume via `python3 -m modal volume get claimguard-data /checkpoints/verifier_binary_v2/best_verifier.pt ./checkpoints/v1_best_verifier.pt`.
13. **Only after steps 1-12 are complete**, begin implementation. Start with task 1 (silver-standard real-hallucination evaluation) unless the literature review in step 9 suggests reordering.

**Do NOT start coding before steps 1-12 are complete.** The previous session made multiple mistakes from jumping ahead: training runs that lost checkpoints because the training script wrote to ephemeral pod disk, OOMs because batch sizes weren't tuned for L40S, and a provenance gate that had to be retrofitted after the verifier was already trained. Reading the full context first prevents repeating any of these.

## Self-Check Agents — MANDATORY

After writing **any** non-trivial code in the resumed session, spawn self-check agents to verify correctness. This is non-negotiable.

**When to spawn self-check agents:**
- After implementing any new function, class, or module
- After modifying data pipelines, training loops, or evaluation code
- After fixing bugs (to verify the fix is correct AND doesn't break anything else)
- **Before launching any Modal GPU job** (ALWAYS — cost of bugs > cost of check)
- Before declaring any of the 9 tasks "done"

**How to spawn them:**
- Use the `Agent` tool with `subagent_type="general-purpose"` or `subagent_type="Explore"`
- **Use Opus, never Sonnet or Haiku for self-check.** The user's global rule is explicit: never cut corners on review quality.
- Run multiple agents in parallel for independent checks (logic review, data-distribution check, edge-case analysis, comparison to requirements)

**Minimum self-checks after code changes:**
1. **Logic review agent** — reads the new code, checks for off-by-one errors, wrong signs, shape mismatches, incorrect broadcasting, missed edge cases
2. **Requirements agent** — verifies the code matches the proposal/architecture doc and task descriptions in this handoff exactly
3. **Integration agent** — checks whether the change breaks existing callers or invariants. Critical for tasks 4, 5, 7 which touch existing pipelines.
4. **Data-distribution agent** — required for tasks 1, 2, 3 (anything touching training data or real hallucination generation). Verifies training/eval/test distributions match, no leakage between splits, no label imbalance introduced, and that silver-standard labels are not contaminated with verifier outputs.

The new session **must report the self-check agent findings to the user** before proceeding to the next task.

## Open Questions for User

- **Silver-standard grader choice (task 1):** Claude Sonnet with vision vs GPT-4V vs both for the ensemble? Claude is cheaper and has comparable vision capability; GPT-4V has a longer track record on medical vision benchmarks. Recommend: both if budget allows ($30 for the ensemble), Claude alone if budget is tight ($10).
- **Paraphrase model choice (task 3):** Claude Haiku vs GPT-4o-mini? Both are ~$2-10 for the full 10K paraphrase job. Claude Haiku has slightly better medical terminology fidelity in informal testing. Recommend: Claude Haiku.
- **Retraining target after tasks 2+3:** v1 RoBERTa-large (safe, proven) or attempt v2 DeBERTa-v3-large again (better but previous attempts died on RunPod)? Recommend: v1 RoBERTa-large, trained on Modal (not RunPod community cloud, which lost the checkpoints last time).
- **Task 9 (same-model experiment) positioning in the paper:** is this a separate methods section or a subsection of the provenance gate discussion? Affects paper structure.
- **MLHC abstract submission:** submit the current v1-numbers version on April 17, or try to slip in task 1 results before the deadline? Task 1 probably cannot finish in 3 days. Recommend: submit current version, update for camera-ready or follow-up venue.
- **Literature gaps:** after reading the full 2026 conformal prediction and medical NLP literature, are there techniques the previous session missed? Specifically check for: (a) newer hallucination benchmarks (RealCXR-like datasets), (b) improvements to inverted conformal procedures, (c) alternative existence-check methods, (d) any published work specifically on same-model self-consistency failure modes in RAG systems. Report findings before implementation.
