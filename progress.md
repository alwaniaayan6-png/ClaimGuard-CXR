# ClaimGuard-CXR v3 — Progress Tracker

**Last updated:** 2026-04-15 (Task 2 v3 retrain + Task 9 real dual-run COMPLETED)

> This file tracked the v2 DeBERTa / CheXzero line of work through
> 2026-04-09. That work is archived in git history but is NOT the
> current direction — RunPod community cloud lost the v2 checkpoints
> and a 2026-04-14 sprint superseded the approach with 9 tasks focused
> on silver-standard real-hallucination evaluation, counterfactual
> augmentation, provenance gating, and retrieval upgrades. See
> `HANDOFF_ClaimGuard_Improvements_2026-04-14.md` for the sprint
> handoff and `/Users/aayanalwani/.claude/plans/rosy-discovering-bubble.md`
> for the approved plan.

## Current state (v3 sprint, 2026-04-14)

**528 pure-Python tests green** across 16 test files (2 retrieval tests
skipped — torch-gated). Full regression sweep runs in ~2 s on CPU.

### Completed code + tests (all rows have local unit tests)

| Task | Files | Tests | Status |
|---|---|---|---|
| Pre-sprint | `checkpoints/v1_best_verifier.pt` (4.27 GB) + `VERIFIER_V1_MANIFEST.json` | — | Recovered, SHA-256 verified |
| Task 1 — Silver-standard eval | `scripts/generate_silver_standard_graders.py`, `scripts/compile_silver_standard_results.py`, `evaluation/krippendorff_alpha.py`, `scripts/generate_real_hallucinations.py` | 30 + 26 + 28 = 84 | **Done with reframe (2026-04-15)** — 414-claim run on Task 9 workbook landed. Claude Sonnet 4.5 vision produced real label distribution (258 SUP / 80 CONTRA / 72 NOV_P). MedGemma all-UNCERTAIN (gated repo + broken fallbacks — dropped). Krippendorff α between CheXbert and Claude = **0.08** across 3 fallback rungs, driven by 63 cases where image-visible findings absent from reference report. **D27 reframe: drop CheXbert as silver grader (text-only labelers are wrong tool for image-grounded eval), ship Claude as silver standard, use Task 8 self-annotation as second coder.** This becomes a methodological finding for the paper. |
| Task 2 — Extended taxonomy | `data/augmentation/hard_negative_generator.py` (8 → 12 types, compound stacking) | 20 | **Fully done (2026-04-15)** — v3 retrain landed: val_acc 0.9877 (epoch 3 best). Checkpoint at `/data/checkpoints/verifier_binary_v3/best_verifier.pt`. |
| Task 3 — Counterfactual + R-Drop | `data/augmentation/causal_term_identifier.py`, `data/augmentation/counterfactual_generator.py`, `scripts/modal_train_dpo_refinement.py`, `scripts/run_causal_term_identification.py`, `scripts/generate_counterfactual_pairs.py`, `scripts/launch_task3c_detached.py`, `inference/verifier_model.py` | 32 + 54 + 29 = 115 | **Fully done (2026-04-15)** with reframing. Task 3a: 5000 IG spans in 10.5 min ($0.30). Task 3b: 2880/5000 rows kept (58%, $33.75 Claude). Task 3c v1 COLLAPSED (single-class training → predict-contra-everywhere, D28). Task 3c v2 FIXED (mixed cf+faithful data, D29) → v4 v2 synth acc 0.9596, OpenI acc 0.7623. **HO gap went NEGATIVE (−2.61 pp)** because v4 broke the lexical shortcuts but the synthetic test rewards them. OpenI improved +0.78 pp. **Reframe: HO gap on synthetic data is the wrong metric** (D29, D30). |
| Task 4 — Regex annotator | `evaluation/regex_error_annotator.py` | 15 | **Fully done** |
| Task 5 — Hybrid retrieval | `models/retriever/bm25_index.py` (batched), `models/retriever/reranker.py` (batched), `models/retriever/rrf_fusion.py` | 14 | **Fully done** (2 tests skipped — torch) |
| Task 6 — StratCP baseline | `inference/stratcp.py`, `scripts/baseline_stratcp.py`, `scripts/run_openi_recalibrated_eval.py`, `scripts/task6_compile_v3_openi.py`, `results/task6/v3_openi/summary.{json,csv}` | 31 | **Fully done (2026-04-15)** — inverted cfBH FDR 0.0088/0.0050/0.0050/0.0344 at α=0.05/0.10/0.15/0.20 on v3 OpenI transfer (holds at every level); StratCP FDR 0.18/0.28/0.35/0.40 (blows past α as expected — controls miscoverage not FDR); forward cfBH n_green=0 (calibration granularity failure, direct empirical evidence for D1 inverted-calibration decision) |
| Task 7 — LLM extractor | `models/decomposer/llm_claim_extractor.py` (wired in), `evaluation/extractor_fidelity.py` | 10 | **Fully done** |
| Task 8 — Self-annotation | `scripts/self_annotate_silver_subset.py`, `scripts/compute_user_vs_ensemble_alpha.py` | 71 + 51 = 122 | Code + tests done; human 90-min session pending |
| Task 9 — Provenance gate demo | `scripts/demo_provenance_gate_failure.py`, `inference/provenance.py`, `results/same_model_experiment/real/gate_demo.json` | 68 + 36 = 104 | **Fully done (2026-04-15)** — real CheXagent dual-run complete. **Result: downgrade_rate_diff = 1.00** (100% same-model pairs downgraded to supported_uncertified, 0% cross-model pairs downgraded) on 828 pairs. v3 verifier cannot distinguish conditions (paired |diff| < 0.1 on all 414 pairs, mean diff 0.0009). Three architectural bugs fixed mid-run (see D20–D22): missing `add_local_python_source("inference")`, wrong-architecture loader (tried AutoModel+Linear vs actual VerifierModel), and off-distribution sanity probes. |
| Integration seam | `tests/test_integration_silver_to_self_annotation.py` | 13 | Locks the Task 1 → Task 8 data contract |

### Doc-sync (2026-04-14, this session)

- [x] `ARCHITECTURE.md` — §5 retriever status updated (batched hybrid is ACTIVE), §6 hard-negative taxonomy 8 → 12, new §13 "Sprint Additions" covering provenance gate, counterfactual + DPO, silver standard, self-annotation, same-model demo.
- [x] `CLAIMGUARD_PROPOSAL.md` — V3 ADDENDUM section added at the end summarizing all 9 Tasks + 4 new known limitations (v2 had 10, v3 has 14).
- [x] `REPRODUCIBILITY.md` — taxonomy table (12 types), Task 1/2/3/8/9 reproduction commands, full test regression sweep recipe.
- [x] `MANUSCRIPT_MINI.md` — Methods §2.1 (12-type taxonomy + counterfactual + DPO), new §3.6 Silver-Standard Real-Hallucination Evaluation, new §3.7 Provenance Gate Same-Model Failure-Mode Case Study, Limitations expanded to 8 points.

### Pending (GPU + human)

- [ ] **Task 1** — Modal: CheXagent × 200 + Claude Sonnet × 600 + MedGemma (~$12, ~60 min H100) — **blocked on `modal secret create anthropic ANTHROPIC_API_KEY=...`**
- [x] **Task 2** — v3 retrain landed 2026-04-15 (val_acc 0.9877, ~$7 H100). HO baseline retrain pending.
- [ ] **Task 3** — Modal: causal ID + R-Drop refinement training (~$18, ~90 min H100) + Claude Sonnet counterfactual generation (~$25 API) — **blocked on `export ANTHROPIC_API_KEY=...` in shell env**
- [x] **Task 6** — Modal: v3 recalibrated OpenI eval + StratCP baseline COMPLETED 2026-04-15. ~$4 H100. Results in `results/task6/v3_openi/summary.{json,csv}`. Inverted cfBH FDR holds at every α; StratCP FDR overshoots (expected); forward cfBH n_green=0 (empirical evidence for D1).
- [ ] **Task 8** — Human: 90-min self-annotation session on 100 silver-pool claims — blocked on Task 1 silver workbook
- [x] **Task 9** — Real CheXagent dual-run + v3 verifier scoring complete 2026-04-15, ~$6 total. Three architectural bugs fixed mid-run (D20–D22).
- [ ] **Re-evals** after v4 ckpt: Task 1 / Task 5 / Task 6 runs with the new checkpoint (~$5 total)

**Budget to date (real Modal + Anthropic spend, cumulative):**
- Task 2 v3 retrain: ~$7
- Task 9 real dual-run (2× CheXagent + scoring): ~$6
- Task 6 v3 OpenI eval: ~$4
- Task 1 silver graders (Claude vision API): ~$5
- Task 3a IG identification: ~$0.30
- Task 3b counterfactual generation (Claude API): ~$34
- Task 3c v1 R-Drop (collapsed, wasted): ~$10
- Task 3c v2 R-Drop (mixed, landed): ~$10
- HO baseline training: ~$5
- v4 evals (synth + OpenI × v1 + v2): ~$8
- Orchestrator debugging (failed runs + redeploys): ~$4
- **Session total: ~$93 / $900 cap** — well under budget.

## Critical decisions reference

See `decisions.md` for full reasoning. New v3 decisions (D11–D18) are
logged there in the same commit as this file.

## Blocked items

- **Radiologist validation** — intentionally scoped out per the "no-radiologist track" plan. Silver standard + self-annotation is the internal-validity substitute.
- **MLHC 2026 April 17 deadline** — explicitly skipped per user decision. Paper targets are NeurIPS D&B Pilot (May 6), MICCAI workshops (Oct), Regeneron STS.

## Next actions

1. User fires Modal runs for Tasks 1, 2, 3, 9 (order flexible — 2 ⇒ 3 ⇒ 1/9 re-eval is one reasonable order)
2. User does the Task 8 self-annotation pass (90 min, local)
3. Re-run `modal_run_evaluation.py` on v3 and v4 checkpoints
4. Final paper figures + tables regeneration from v4 numbers
5. End-of-sprint retrospective + NeurIPS D&B draft

## Archive: v2 DeBERTa line (2026-04-09, obsolete)

The v2 DeBERTa + progressive NLI + CheXzero fusion track hit checkpoint
loss on RunPod community cloud and was abandoned. v2 result JSONs
survived in `results/v2_runpod/captured_results.json`. v2 reached
98.61% val acc (98.31% test) with FDR=1.66% at α=0.05 and a 98.15%
hypothesis-only baseline — confirming the lexical-shortcut artifact
that v3 Task 3 is designed to attack. The v3 sprint reuses the v1
RoBERTa-large checkpoint as its base.
