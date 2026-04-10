# ClaimGuard-CXR v2 — Progress Tracker

**Last updated:** 2026-04-09

## Tier 5: Infrastructure
- [x] Git repo initialized with .gitignore
- [x] Initial commit (91 files)
- [ ] README.md (deferred to after experiments)

## Tier 2: Core Scripts (code written, not yet run)
- [x] `scripts/modal_progressive_nli.py` — MNLI -> MedNLI -> (skip RadNLI) -> ClaimGuard
- [x] `scripts/modal_train_verifier_deberta.py` — Updated to load from progressive NLI checkpoint
- [x] `scripts/modal_train_chexzero_fusion.py` — CheXzero projection + gate training
- [x] `scripts/compile_annotated_results.py` — Ingest human annotations, compute metrics

## Tier 3: Baselines (code written, not yet run)
- [x] `scripts/baseline_hypothesis_only.py` — Claim-only, no evidence (artifact check)
- [x] `scripts/baseline_chexzero_zeroshot.py` — CheXzero CLIP zero-shot
- [x] `scripts/baseline_deberta_zeroshot_nli.py` — Off-the-shelf DeBERTa-v3 MNLI
- [x] `scripts/baseline_radflag_consistency.py` — Self-consistency keyword overlap

## Tier 4: Experiments (BLOCKED — needs GPU execution)

### Execution Order (dependency chain):

```
Step 1: modal run --detach scripts/modal_progressive_nli.py
        [~2-3 hrs on H100, MNLI 100K + MedNLI 11K + ClaimGuard 30K]
        Output: /data/checkpoints/progressive_nli/final/best_verifier.pt

Step 2: (OPTIONAL — only if progressive NLI doesn't include ClaimGuard fine-tuning)
        modal run --detach scripts/modal_train_verifier_deberta.py \
            --pretrained-checkpoint /data/checkpoints/progressive_nli/final/best_verifier.pt
        [~35 min on H100]
        Output: /data/checkpoints/verifier_deberta_v2/best_verifier.pt

Step 3: Upload OpenI images to Modal volume (if not already done):
        modal volume put claimguard-data /Users/aayanalwani/data/openi/*.png /openi_images/ --force

Step 4: (PARALLEL — all baselines can run simultaneously)
    4a: modal run --detach scripts/baseline_hypothesis_only.py
    4b: modal run --detach scripts/baseline_deberta_zeroshot_nli.py
    4c: modal run --detach scripts/baseline_chexzero_zeroshot.py
    4d: python3 scripts/baseline_radflag_consistency.py \
            --test-claims /Users/aayanalwani/data/claimguard/eval_data/test_claims.json \
            --output figures/radflag_baseline.json

Step 5: Run v2 full eval pipeline (uses checkpoint from Step 1 or 2):
        modal run --detach scripts/modal_run_evaluation.py \
            --verifier-path /data/checkpoints/progressive_nli/final/best_verifier.pt \
            --num-classes 2 \
            --cal-claims-path /data/eval_data/calibration_claims.json \
            --test-claims-path /data/eval_data/test_claims.json \
            --output-dir /data/eval_results_deberta_v2

Step 6: Generate real hallucination annotation workbook:
        modal run --detach scripts/generate_real_hallucinations.py
        [~15 min on H100 for CheXagent on 200 OpenI images]
        Output: /data/eval_data_real_hallucinations/annotation_workbook.csv
        STATUS: Requires manual human annotation after this step

Step 7: (OPTIONAL — needs OpenI images on Modal)
        modal run --detach scripts/modal_train_chexzero_fusion.py
        [~15 min on H100]
```

### Estimated Modal Costs:
| Step | GPU | Time | Est. Cost |
|------|-----|------|-----------|
| 1. Progressive NLI | H100 | ~2.5 hr | ~$7.50 |
| 2. DeBERTa fine-tune (optional) | H100 | ~35 min | ~$1.75 |
| 4a. Hypothesis-only | H100 | ~35 min | ~$1.75 |
| 4b. DeBERTa NLI baseline | H100 | ~20 min | ~$1.00 |
| 4c. CheXzero baseline | H100 | ~15 min | ~$0.75 |
| 5. Full eval | H100 | ~10 min | ~$0.50 |
| 6. CheXagent generation | H100 | ~15 min | ~$0.75 |
| 7. CheXzero fusion | H100 | ~15 min | ~$0.75 |
| **Total** | | **~4.5 hr** | **~$15** |

## Tier 6: Writing
- [ ] MLHC 2026 abstract (deadline April 17 — 8 days)
- [ ] NeurIPS 2026 outline (deadline May 6 — 27 days)
- [ ] ICML EIML workshop outline

## Blocked Items
- **RadNLI stage**: Requires PhysioNet credentialing (user needs CITI training)
- **MIMIC-CXR 3rd dataset**: Same PhysioNet block
- **Real hallucination eval**: Needs manual annotation after Step 6
- **CheXzero fusion quality**: Depends on having CXR images on Modal volume

## What's Next
1. User runs Steps 1-6 on Modal (I'll provide exact commands)
2. While GPU jobs run: draft MLHC abstract
3. After results: compile v1 vs v2 comparison table
4. After annotation workbook: distribute for human labeling
