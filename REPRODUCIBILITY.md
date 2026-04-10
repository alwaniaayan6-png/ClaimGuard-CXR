# ClaimGuard-CXR — Reproducibility Guide

**Paper:** "Calibrated Claim-Level Hallucination Detection for Radiology Report Generation with Conformal FDR Control"
**Target venue:** NeurIPS 2026 (May 6 deadline)
**Last updated:** 2026-04-05

This document describes how to reproduce every result in the paper from the
CheXpert Plus dataset. All experiments run on Modal (cloud GPU) except
baselines which run locally.

## Hardware / software environment

- Training / eval: Modal H100 80GB HBM3
- Retrieval indexing: Modal H100 (MedCPT article encoder, 1.2M passages)
- Local analysis: Python 3.9, macOS 14
- Seeds: 42 everywhere (training, splits, bootstrap, conformal subsampling)

## Full pipeline (end-to-end reproduce)

### 1. Patient splits + data generation

```bash
# Create patient-disjoint 60/15/25 train/cal/test splits (seed=42)
python3 scripts/create_patient_splits.py \
    --data-path ${CHEXPERT_PLUS_CSV} \
    --output-dir ${DATA_DIR}/splits_chexpert_plus/

# Generate 30K training claims + 15K calibration + 15K test claims
# with 8 hard-negative types (1250/class-type balanced)
python3 scripts/prepare_eval_data.py \
    --data-path ${CHEXPERT_PLUS_CSV} \
    --splits-dir ${DATA_DIR}/splits_chexpert_plus/ \
    --output-dir ${DATA_DIR}/eval_data/

# Wrap for training split (same 8 hard-neg types, 30K examples)
python3 scripts/prepare_verifier_training_data_v2.py \
    --data-path ${CHEXPERT_PLUS_CSV} \
    --splits-dir ${DATA_DIR}/splits_chexpert_plus/ \
    --output-path ${DATA_DIR}/verifier_training_data.json
```

### 2. Upload to Modal volume

```bash
modal volume put claimguard-data ${DATA_DIR}/verifier_training_data.json /verifier_training_data.json
modal volume put claimguard-data ${DATA_DIR}/eval_data/calibration_claims.json /eval_data/calibration_claims.json
modal volume put claimguard-data ${DATA_DIR}/eval_data/test_claims.json /eval_data/test_claims.json
```

### 3. Train binary verifier (RoBERTa-large, no label smoothing)

```bash
# ~20 min on H100, ~$2
modal run --detach scripts/modal_train_verifier_binary.py
# Outputs: /data/checkpoints/verifier_binary_v2/best_verifier.pt
```

**Expected metrics:** val_acc ~98.0-98.1% (epoch 2 is best), train/val loss gap < 0.01.

### 4. Build retrieval index

```bash
# ~10 min on H100 — encodes 1.2M training passages via MedCPT Article Encoder
modal run --detach scripts/modal_build_index.py
# Outputs: /data/indices/{faiss_index.index, bm25_index.pkl, passage_lookup.json, passage_embeddings.npy}

# Build retrieval-augmented eval set (~2 min)
modal run --detach scripts/modal_build_retrieval_eval.py
# Outputs: /data/eval_data/{calibration,test}_claims_retrieved.json
```

### 5. Run main evaluation (oracle evidence + conformal FDR)

```bash
# ~5 min on H100, ~$0.30
modal run --detach scripts/modal_run_evaluation.py \
    --verifier-path /data/checkpoints/verifier_binary_v2/best_verifier.pt \
    --num-classes 2 \
    --cal-claims-path /data/eval_data/calibration_claims.json \
    --test-claims-path /data/eval_data/test_claims.json \
    --output-dir /data/eval_results_binary_v2
```

**Expected:** accuracy 98.31%, FDR=1.30% @ α=0.05, power=98.06%.

### 6. Run retrieval-augmented evaluation

```bash
modal run --detach scripts/modal_run_evaluation.py \
    --verifier-path /data/checkpoints/verifier_binary_v2/best_verifier.pt \
    --num-classes 2 \
    --cal-claims-path /data/eval_data/calibration_claims_retrieved.json \
    --test-claims-path /data/eval_data/test_claims_retrieved.json \
    --output-dir /data/eval_results_binary_v2_retrieved
```

**Expected:** accuracy 98.09%, FDR=1.19% @ α=0.05, power=96.64%.

### 7. Cross-dataset evaluation (OpenI / Indiana University)

OpenI is publicly accessible — no credentialing needed.

```bash
# Download OpenI reports (1.1 MB, ~10 seconds)
mkdir -p ${DATA_DIR}/openi
cd ${DATA_DIR}/openi
curl -L -O https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz
tar xzf NLMCXR_reports.tgz

# Convert OpenI XML -> CheXpert Plus CSV schema
python3 scripts/convert_openi_to_chexpert_schema.py \
    --xml-dir ${DATA_DIR}/openi/ecgen-radiology \
    --output ${DATA_DIR}/openi/openi_cxr_chexpert_schema.csv

# Create 50/50 cal/test split (seed=42)
# Then generate claims with same pipeline
python3 scripts/prepare_eval_data.py \
    --data-path ${DATA_DIR}/openi/openi_cxr_chexpert_schema.csv \
    --splits-dir ${DATA_DIR}/openi/splits_openi \
    --output-dir ${DATA_DIR}/openi/eval_data \
    --n-claims 600

# Upload to Modal + run cross-dataset eval
modal volume put claimguard-data \
    ${DATA_DIR}/openi/eval_data/calibration_claims.json \
    /eval_data_openi/calibration_claims.json --force
modal volume put claimguard-data \
    ${DATA_DIR}/openi/eval_data/test_claims.json \
    /eval_data_openi/test_claims.json --force

modal run --detach scripts/modal_run_evaluation.py \
    --verifier-path /data/checkpoints/verifier_binary_v2/best_verifier.pt \
    --num-classes 2 \
    --cal-claims-path /data/eval_data_openi/calibration_claims.json \
    --test-claims-path /data/eval_data_openi/test_claims.json \
    --output-dir /data/eval_results_openi
```

**Expected:** accuracy 85%, FDR=0.40% @ α=0.05, power=41.58%. FDR ≤ α at all levels.

### 8. Baselines

```bash
# Rule-based (local, seconds)
python3 scripts/baseline_rule_based.py

# Untrained RoBERTa (Modal H100, ~3 min)
modal run --detach scripts/modal_run_evaluation.py \
    --verifier-path /data/checkpoints/verifier_binary_v2/nonexistent.pt \
    --num-classes 2 \
    --cal-claims-path /data/eval_data/calibration_claims.json \
    --test-claims-path /data/eval_data/test_claims.json \
    --output-dir /data/eval_results_untrained

# Zero-shot LLM judge (local, needs LLM_API_KEY, ~15 min)
export LLM_API_KEY=...
python3 scripts/baseline_zeroshot_llm.py
```

### 8. Figures and tables

```bash
# Download predictions + results locally
modal volume get claimguard-data /eval_results_binary_v2/ /tmp/v2_final/ --force
modal volume get claimguard-data /eval_results_binary_v2_retrieved/ /tmp/v2_retr/ --force

# Generate all paper figures
python3 scripts/generate_paper_figures.py
# -> figures/fig1_score_distributions.pdf (+ png)
# -> figures/fig2_reliability.pdf
# -> figures/fig3_fdr_power.pdf
# -> figures/fig4_confusion.pdf
# -> figures/fig5_roc_pr.pdf
# -> figures/fig6_per_neg_type.pdf

# Extract all tables
python3 scripts/extract_ablation_tables.py
# -> figures/ablation_tables.md / .tex

python3 scripts/compile_baseline_comparison.py
# -> figures/baseline_comparison.md / .tex

# Decision gate check
python3 scripts/compute_final_results_local.py
```

## Key hyperparameters (verifier training)

| Parameter | Value |
|---|---|
| Base model | `roberta-large` |
| Batch size | 32 (× 2 gradient accumulation = effective 64) |
| Epochs | 3 |
| Learning rate | 2e-5 (linear decay, 10% warmup) |
| Weight decay | 0.01 |
| Label smoothing | **0.0** (critical — 0.05 creates conformal score ceiling) |
| Dropout | 0.1 |
| Max seq length | 512 |
| Optimizer | AdamW |
| Val split | 10% patients (stratified) |
| Best ckpt | lowest val_loss |
| Early stopping | patience=1 on val_loss |
| Determinism | cudnn.deterministic=True + use_deterministic_algorithms=True |
| Seed | 42 |

## Key conformal hyperparameters

| Parameter | Value |
|---|---|
| Procedure | Inverted label-conditional cfBH (Jin & Candes 2023, adapted) |
| Calibration pool | Contradicted test claims (label==1), one per patient (C7) |
| Score | P(not-contradicted) from temperature-calibrated softmax |
| p-value | `(|cal_contra_score >= test_score| + 1) / (n_cal + 1)` |
| BH scope | **Global** (not per-pathology) — per-group BH is too strict |
| α levels | {0.01, 0.05, 0.10, 0.15, 0.20} |
| Temperature | LBFGS on calibration NLL |

## Data generation contract

- **Patient separation:** no patient in more than one split
- **Hard-neg types** (8, equal frequency): negation, laterality_swap,
  severity_swap, temporal_error, finding_substitution, region_swap,
  device_line_error, omission_as_support
- **Contradiction validator:** perturbed claim verified to NOT already be
  supported by original evidence (avoids label noise bug C6)
- **Insufficient examples:** evidence from DIFFERENT patient AND DIFFERENT
  pathology (avoids leakage bug C5)
- **Laterality safety (M8):** only swap laterality in claims that aren't
  bilateral in the original report

## Expected results reference

### In-domain (CheXpert Plus)
| Result | Value |
|---|---|
| Binary verifier val_acc | 98.01% (epoch 2 best) |
| Binary verifier test_acc (oracle) | 98.31% |
| Binary verifier test_acc (retrieval) | 98.09% |
| AUROC (oracle) | 99.52% |
| AP (oracle) | 99.63% |
| ECE (calibrated) | 0.0066 |
| FDR @ α=0.05 | 1.30% |
| Power @ α=0.05 | 98.06% |
| FDR @ α=0.10 | 2.48% |
| Power @ α=0.10 | 99.51% |

### Cross-dataset (OpenI, zero-shot)
| Result | Value |
|---|---|
| Accuracy | 85.09% |
| Macro F1 | 83.07% |
| Contradicted F1 | 77.23% |
| ECE (calibrated) | 0.0279 |
| FDR @ α=0.05 | 0.40% |
| Power @ α=0.05 | 41.58% |
| FDR @ α=0.10 | 1.54% |
| Power @ α=0.10 | 58.42% |
| FDR @ α=0.20 | 5.65% |
| Power @ α=0.20 | 77.92% |

FDR remains controlled ≤ α at every level even on out-of-distribution data.

## File manifest

Scripts (all in `scripts/`):
- `create_patient_splits.py` — 60/15/25 patient-disjoint splits
- `prepare_eval_data.py` — 8 hard-neg claim generation
- `prepare_verifier_training_data_v2.py` — training data wrapper
- `modal_train_verifier_binary.py` — binary verifier training (H100)
- `modal_build_index.py` — FAISS + BM25 retrieval indices (H100)
- `modal_build_retrieval_eval.py` — retrieval-augmented eval claims (H100)
- `modal_run_evaluation.py` — main eval pipeline with inverted cfBH
- `baseline_rule_based.py` — rule-based baseline (local)
- `baseline_zeroshot_llm.py` — zero-shot LLM judge (local, API)
- `generate_paper_figures.py` — all figures
- `extract_ablation_tables.py` — formatted tables
- `compile_baseline_comparison.py` — baseline comparison table
- `compute_final_results_local.py` — decision gate check (local)

Docs:
- `ARCHITECTURE.md` — full system architecture
- `CLAIMGUARD_PROPOSAL.md` — research proposal + scope
- `REPRODUCIBILITY.md` — this file

## Known caveats

1. **Label smoothing must be 0** for binary training. Setting to 0.05
   creates a hard ceiling at P=0.975 that collapses the conformal score
   distribution — BH accepts nothing (n_green=0 at all α). The 3-class
   model used label smoothing 0.05 and had the same issue.

2. **Global BH (not per-group)** is required for the conformal procedure.
   Per-group BH fails because each pathology's cal pool has min_p > local
   BH threshold. We compute per-pathology p-values for exchangeability,
   then apply BH globally.

3. **Inverted conformal** — calibrate on CONTRADICTED (null H0=contra),
   not on faithful. Calibrating on faithful fails because both cal and
   test faithful scores sit at the softmax ceiling.

4. **BM25 sparse retrieval is built but not used at eval time** — rank_bm25
   per-query scoring is O(corpus) (~2s for 1.2M passages), infeasible at
   30K eval claims. Dense-only MedCPT retrieval gives 98.09% accuracy
   with only 0.22pp drop from oracle. Batched sparse retrieval is future
   work.
