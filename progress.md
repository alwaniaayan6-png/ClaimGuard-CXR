# ClaimGuard-CXR v3 — Progress Tracker

**Last updated:** 2026-04-14 (post-sprint)

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
| Task 1 — Silver-standard eval | `scripts/generate_silver_standard_graders.py`, `scripts/compile_silver_standard_results.py`, `evaluation/krippendorff_alpha.py`, `scripts/generate_real_hallucinations.py` | 30 + 26 + 28 = 84 | Code + tests done; Modal run pending |
| Task 2 — Extended taxonomy | `data/augmentation/hard_negative_generator.py` (8 → 12 types, compound stacking) | 20 | Code + tests done; v3 retrain pending |
| Task 3 — Counterfactual + DPO | `data/augmentation/causal_term_identifier.py`, `data/augmentation/counterfactual_generator.py`, `scripts/modal_train_dpo_refinement.py` | 32 + 54 + 29 = 115 | Code + tests done; causal ID + DPO pending |
| Task 4 — Regex annotator | `evaluation/regex_error_annotator.py` | 15 | **Fully done** |
| Task 5 — Hybrid retrieval | `models/retriever/bm25_index.py` (batched), `models/retriever/reranker.py` (batched), `models/retriever/rrf_fusion.py` | 14 | **Fully done** (2 tests skipped — torch) |
| Task 6 — StratCP baseline | `inference/stratcp.py`, `scripts/baseline_stratcp.py`, `scripts/run_openi_recalibrated_eval.py` | 31 | Code + tests done |
| Task 7 — LLM extractor | `models/decomposer/llm_claim_extractor.py` (wired in), `evaluation/extractor_fidelity.py` | 10 | **Fully done** |
| Task 8 — Self-annotation | `scripts/self_annotate_silver_subset.py`, `scripts/compute_user_vs_ensemble_alpha.py` | 71 + 51 = 122 | Code + tests done; human 90-min session pending |
| Task 9 — Provenance gate demo | `scripts/demo_provenance_gate_failure.py`, `inference/provenance.py` (existing) | 68 + 36 = 104 | Code + tests done + reviewer fixes applied; Modal run pending |
| Integration seam | `tests/test_integration_silver_to_self_annotation.py` | 13 | Locks the Task 1 → Task 8 data contract |

### Doc-sync (2026-04-14, this session)

- [x] `ARCHITECTURE.md` — §5 retriever status updated (batched hybrid is ACTIVE), §6 hard-negative taxonomy 8 → 12, new §13 "Sprint Additions" covering provenance gate, counterfactual + DPO, silver standard, self-annotation, same-model demo.
- [x] `CLAIMGUARD_PROPOSAL.md` — V3 ADDENDUM section added at the end summarizing all 9 Tasks + 4 new known limitations (v2 had 10, v3 has 14).
- [x] `REPRODUCIBILITY.md` — taxonomy table (12 types), Task 1/2/3/8/9 reproduction commands, full test regression sweep recipe.
- [x] `MANUSCRIPT_MINI.md` — Methods §2.1 (12-type taxonomy + counterfactual + DPO), new §3.6 Silver-Standard Real-Hallucination Evaluation, new §3.7 Provenance Gate Same-Model Failure-Mode Case Study, Limitations expanded to 8 points.

### Pending (GPU + human)

- [ ] **Task 1** — Modal: CheXagent × 200 + Claude Sonnet × 600 + MedGemma (~$12, ~60 min H100)
- [ ] **Task 2** — Modal: v3 training job + HO baseline retrain (~$14, ~90 min H100)
- [ ] **Task 3** — Modal: causal ID + DPO training (~$43, ~145 min H100) + Claude Sonnet counterfactual generation (~$25 API)
- [ ] **Task 8** — Human: 90-min self-annotation session on 100 silver-pool claims
- [ ] **Task 9** — Modal: 2 × CheXagent dual-run + verifier scoring (~$5, ~60 min H100)
- [ ] **Re-evals** after v3/v4 ckpts: Task 1 / Task 5 / Task 6 runs with the new checkpoint (~$5 total)

**Budget to date:** ~$0 this session (all local CPU work). **Projected
sprint total:** ~$79 of $900 cap.

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
