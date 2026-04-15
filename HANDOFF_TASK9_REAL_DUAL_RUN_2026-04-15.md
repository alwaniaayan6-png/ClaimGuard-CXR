# Handoff: ClaimGuard-CXR Task 9 Real Dual-Run + v3 Sprint Continuation

**Date:** 2026-04-15 ~10:00 ET
**Status:** active — real CheXagent dual-run Modal jobs complete (both sides), orchestrator firing scoring now; user asleep, session running autonomously per instructions.
**Last session model:** claude-opus-4-6 / claude-opus-4-1-20250805 (pre-flight reviews all Opus)

---

## Current State (1-paragraph)

v3 sprint is ~75% done: Task 2 v3 retrain **COMPLETED** (val_acc 0.9877, checkpoint on volume), Task 9 real CheXagent dual-run **COMPLETED** on both sides (run A: 414 claims / 100 images / 0 errors, run B: 401 claims / 100 images / 0 errors, both with real radiology text validated — e.g., *"No pleural effusion or pneumothorax is seen. The cardiac and mediastinal silhouettes are unremarkable."*). The CPU orchestrator is mid-scoring (`fc-01KP8PJD5VJNYS5FNAH4MHB3MC`) and will write `/same_model_experiment/real/gate_demo.json` when done. Tasks 1/3/8 remain blocked on credentials/human time; Task 6 is pending Task 9's scored output. ~10 commits pushed to GitHub this session. Doc-sync green. Preflight reviewer rule now global.

---

## What's Running / In Progress (Modal FunctionCalls)

**ALL IDs live in `/tmp/task9_real_*.json` sidecars — use these first.**

| Name | FunctionCall ID | Kind | Status | ETA |
|---|---|---|---|---|
| CheXagent run A | `fc-01KP8PHQE0T5BP1RR1N29ZK6R1` | H100 | **DONE** (414 claims, 100 imgs, 0 errors) | — |
| CheXagent run B | `fc-01KP8PHQJ79EPPDD63E57YB1D8` | H100 | **DONE** (401 claims, 100 imgs, 0 errors) | — |
| Real orchestrator (scoring) | `fc-01KP8PJD5VJNYS5FNAH4MHB3MC` | CPU → H100 scoring | **RUNNING** | ~10–20 min |
| Plan C scoring (backup, stub data) | `fc-01KP8MBX7XQ3WQHH1FKH3CMWK9` | H100 | RUNNING | ≤15 min |
| Monitor (network-resilient poll) | task_id `b85uy7298` | local Bash | RUNNING | up to 60 min |

**How to check each:**
```bash
cd /Users/aayanalwani/VeriFact/verifact
python3 -c "
import modal
for name, fc_id in [
    ('run_a', 'fc-01KP8PHQE0T5BP1RR1N29ZK6R1'),
    ('run_b', 'fc-01KP8PHQJ79EPPDD63E57YB1D8'),
    ('orchestrator', 'fc-01KP8PJD5VJNYS5FNAH4MHB3MC'),
    ('plan_c', 'fc-01KP8MBX7XQ3WQHH1FKH3CMWK9'),
]:
    try:
        modal.FunctionCall.from_id(fc_id).get(timeout=1)
        print(f'{name}: DONE')
    except TimeoutError:
        print(f'{name}: RUNNING')
    except Exception as e:
        print(f'{name}: {type(e).__name__}: {str(e)[:120]}')
"
```

**Volume outputs to watch for:**
```bash
# Real run (primary story):
modal volume ls claimguard-data /same_model_experiment/real/
# expect eventually: gate_demo.json  (downgrade_rate > 0.5 is the pass bar)

# Plan C backup:
modal volume ls claimguard-data /same_model_experiment/
# expect eventually: gate_demo.json  (synthetic-stub version)
```

**Download the primary result once orchestrator exits:**
```bash
modal volume get claimguard-data /same_model_experiment/real/gate_demo.json /tmp/task9_real/
python3 -c "
import json
with open('/tmp/task9_real/gate_demo.json') as f:
    d = json.load(f)
print('downgrade_rate SAME_MODEL:', d.get('same_model_downgrade_rate'))
print('downgrade_rate INDEPENDENT:', d.get('independent_downgrade_rate'))
print('n_high_score_pre_gate:', d.get('n_high_score_pre_gate'))
print('n_high_score_post_gate:', d.get('n_high_score_post_gate'))
"
```

Expected: same-model condition downgrade_rate > 0.5; independent condition downgrade_rate ≈ 0. If not, the provenance gate experiment failed to reproduce the hypothesis → halt, do not doc-sync, surface to user before anything else.

---

## Completed Tasks (session)

- [x] **Pre-sprint** — v1 checkpoint recovered from Modal volume `/checkpoints/verifier_binary_v2/best_verifier.pt`; SHA-256 pinned; 36/36 provenance tests still green.
- [x] **Task 2 — Extended taxonomy (8 → 12 types) + v3 retrain**
  - 4 new types (`fabricated_measurement`, `fabricated_prior`, `fabricated_temporal`, `compound_perturbation`) registered in `_GENERATORS` / `_LABELS` / `ALL_NEGATIVE_TYPES` in `data/augmentation/hard_negative_generator.py`
  - `scripts/prepare_eval_data.py` wires new types + 60/40 compound split
  - v3 training data: 30,000 examples, 13 hard-negative types, 16 MB
  - v3 eval (cal + test) regenerated
  - v3 retrain on Modal H100 (`scripts/modal_train_verifier_binary.py`): val_acc epoch 1 = 0.9732, epoch 2 = 0.9864, **epoch 3 = 0.9877 [BEST]**, train_ce 0.0313, val_loss 0.058
  - Checkpoint at `/data/checkpoints/verifier_binary_v3/best_verifier.pt` on `claimguard-data`
- [x] **Task 3 architectural fix — DPO → R-Drop consistency**
  - Found the original plan's DPO formulation inverted (chosen=counterfactual rejected=original pulls ORIGINAL down, opposite of intended invariance). Literature dive: the "Dually Self-Improved" citation the plan invoked does not exist in arXiv / OpenAlex / ACL / Semantic Scholar.
  - Replaced with R-Drop (NeurIPS 2021) symmetric-KL consistency regularization: CE(orig,y) + CE(cf,y) + λ_cons·½·[KL(p_orig||stop_grad p_cf) + KL(p_cf||stop_grad p_orig)]
  - `scripts/modal_train_dpo_refinement.py` now has `loss_mode: "consistency"` (default, R-Drop) vs `"dpo"` (legacy, kept for reproducibility only)
  - λ_ce=1.0, λ_cons=0.5; abort path writes `aborted_verifier.pt` + `ABORTED` marker, not silent failure
  - `data/augmentation/counterfactual_generator.py` `validate_preservation` now case-insensitive word-boundary regex `\btoken\b` (was substring → false positives on subtokens)
  - `data/augmentation/causal_term_identifier.py` SEP-index off-by-one fixed (advance past contiguous special tokens for RoBERTa `<s> claim </s></s> evidence </s>`)
  - **Still blocked** on `ANTHROPIC_API_KEY` in shell env for counterfactual generation
- [x] **Task 1 architectural fix — CheXbert ghost**
  - Stanford `stanfordmlgroup/CheXbert` HF upload has no 14-head MLP — grader was silently stamping every row UNCERTAIN.
  - Replaced with rule-based CheXpert-style labeler (`_rule_based_chexpert_label_vector`) in `scripts/generate_silver_standard_graders.py`: NEGATION_CUES + UNCERTAINTY_CUES + SCOPE_TERMINATORS + `_find_in_window_with_scope` (60-char window trimmed at "but"/"however"/"."/";")
  - Claude grader now resizes images via lazy PIL + retry/backoff, anthropic==0.40.0 pinned, transformers==4.50.0 bump for MedGemma compatibility
  - MedGemma loader class rotation (AutoModelForImageTextToText → AutoModelForVision2Seq → AutoModelForCausalLM)
  - **Still blocked** on `modal secret create anthropic ANTHROPIC_API_KEY=...`
- [x] **HO baseline methodology fix**
  - Was DeBERTa-v3-large claim-only at max_length=128 (wrong architecture AND wrong input shape vs v1/v3 verifier)
  - Now RoBERTa-large with (claim, MASKED_EVIDENCE) pair at max_length=512 — matches v1/v3 verifier shape so any HO/full gap is interpretable
  - `scripts/baseline_hypothesis_only.py` default data_path → `/data/verifier_training_data_v3.json`, output_dir → `hypothesis_only_v3`
- [x] **Task 9 — Provenance gate real dual-run**
  - `scripts/demo_provenance_gate_failure.py` (1100+ lines): added `_extract_generator_id` (strict, raises on conflict), `_is_error_sentinel`, `_sanity_check_verifier`, `task9_orchestrator_remote` (CPU Modal function that polls volume for workbooks then fires `demo_provenance_gate_remote.remote(skip_generation=True)`).
  - `scripts/launch_task9_detached.py` (~500 lines, new): detach launcher using `Function.from_name().spawn()` + `.get()` chain with sidecar JSON at `/tmp/task9_function_calls*.json`.
  - `scripts/task9_wait_and_launch.sh` (~110 lines, new): bash wrapper — polls upload PID, verifies volume, fires launcher under `caffeinate -i`. Fixed `$?` after pipe → `${PIPESTATUS[0]}` (the bug that made the wrapper falsely report "COMPLETED successfully" on 2026-04-15 Task 9 v1).
  - `scripts/generate_real_hallucinations.py`: added `image_seed` parameter (decoupled from torch seed), CheXagent deps (einops, cv2, albumentations, torchvision + libgl1 apt), fixed CSV schema detection for both `study_id/report_text` AND OpenI's `deid_patient_id/section_findings+section_impression`.
  - **CheXagent prompt format fix**: initially got 1 claim / 0 claims outputs. Found README-validated format by dumping custom modeling source from HF cache inside Modal container: `' USER: <s>{prompt} ASSISTANT: <s>'`. Smoke4 (3 images) confirmed → 10 claims / 0 errors / real radiology text.
  - Real dual-run fired → both sides complete.
- [x] **Git & doc-sync**
  - Removed blanket `data/` gitignore (was silently excluding ~3000 lines of `data/augmentation/` and `data/preprocessing/` Python)
  - 10+ commits pushed (see "Commits this session" table below)
  - `ARCHITECTURE.md`, `CLAIMGUARD_PROPOSAL.md`, `REPRODUCIBILITY.md`, `MANUSCRIPT_MINI.md`, `progress.md`, `decisions.md` (D11–D19) all updated in-commit with code
  - `~/.claude/CLAUDE.md` — removed "NEVER commit without asking" rule; added **Pre-flight Review Rule** (mandatory Opus reviewer before any expensive remote action)
- [x] **Vault sync** — `~/Vault/wiki/projects/VeriFact.md`, `~/Vault/index.md`, `~/Vault/log.md` all updated through 2026-04-15.

---

## Pending Tasks (priority order)

1. **Wait for orchestrator → download gate_demo.json → validate `downgrade_rate > 0.5`.**
   - Blocked by: orchestrator `fc-01KP8PJD5VJNYS5FNAH4MHB3MC` still running
   - Files to touch: `/same_model_experiment/real/gate_demo.json` (read-only on volume)
   - Pass bar: same-model downgrade_rate > 0.5, independent downgrade_rate ≈ 0, median (claimA, evidenceA) verifier score > 0.9 on claims that disagree with ground-truth OpenI report
   - On pass → commit results table into `CLAIMGUARD_PROPOSAL.md` §Same-model failure-mode case study and `MANUSCRIPT_MINI.md` Limitation (6) expansion
   - On fail → halt, do not doc-sync; investigate whether `_sanity_check_verifier` fires, whether provenance stamping is correct, whether tier classification is right

2. **Task 6 — Recalibrated OpenI + StratCP baseline** (after Task 9 scored output lands)
   - Blocked by: needs v3 verifier scored claims as input
   - Files: `scripts/run_openi_recalibrated_eval.py` (new), `inference/stratcp.py` (new), `scripts/baseline_stratcp.py` (new), `tests/test_stratcp.py` (new)
   - Patient-level 50/50 split on `/Users/aayanalwani/data/openi/openi_cxr_chexpert_schema.csv` via `random.Random(42)`
   - Per-pathology FDR + power, StratCP comparison (medRxiv Feb 2026 algorithm — no public reference implementation, validate on synthetic Gaussian strata first)
   - If our StratCP diverges from paper numbers by > 2 pp → park as partial baseline with caveat
   - Compute ~$2, 10 min H100

3. **Task 1 — Silver-standard graders** (blocked on credentials — user action)
   - Needs: `modal secret create anthropic ANTHROPIC_API_KEY=...`
   - After: `modal run scripts/generate_silver_standard_graders.py --n-images 200`
   - Krippendorff α ≥ 0.80 required (fallback ladder: drop UNCERTAIN → binary S vs rest)
   - Compute ~$12, 1 h H100 + Claude API

4. **Task 3 — Counterfactual + R-Drop refinement** (blocked on credentials — user action)
   - Needs: `export ANTHROPIC_API_KEY=...` in shell env
   - Pipeline: `python3 -m data.augmentation.counterfactual_generator` (Claude Sonnet 4.5 calls) → `modal run scripts/modal_train_dpo_refinement.py --loss-mode consistency`
   - Compute ~$43 ($25 Claude + $18 H100), 2 h
   - Success criterion: HO gap grows from 0.60 pp → ≥ 5 pp (v4 uses evidence, not surface form)

5. **Task 8 — Self-annotation validation (90 min human time)** — blocked on Task 1 silver workbook existing
   - Files: `scripts/self_annotate_silver_subset.py`, `scripts/compute_user_vs_ensemble_alpha.py`
   - User labels 100 claims stratified 20 per class; 4-coder Krippendorff α with 3 silver graders
   - Regression test already in place: `format_prompt` must NOT leak `silver majority=X` to user

6. **Task 4 — Post-hoc regex error annotator** (cheap, parallel-safe)
   - `evaluation/regex_error_annotator.py` (new) — pure-function `annotate(claim: str) -> dict`, metadata only, never changes `TriageResult.final_label`
   - Flags: fabricated_measurement / fabricated_prior / fabricated_date / fabricated_reltime
   - Used by `scripts/compile_silver_standard_results.py` and `scripts/analyze_eval_results.py` for paper error-analysis tables

7. **Task 5 — Batched BM25 + cross-encoder + RRF fusion** (cheap, parallel-safe)
   - `models/retriever/bm25_index.py`: add `search_batch`
   - `models/retriever/reranker.py`: add `rerank_batch(query_passage_pairs, batch_size=64)`
   - `models/retriever/rrf_fusion.py` (new): `rrf_fuse(dense_ranks, sparse_ranks, k=60)`
   - `scripts/modal_build_retrieval_eval.py`: wire the ablation table `{dense_only, sparse_only, dense+sparse_rrf, +rerank}` × `{R@5, R@10, nDCG@10, acc, FDR, power}`
   - Keep `cross-encoder/ms-marco-MiniLM-L-12-v2` — do NOT swap to DeBERTa

8. **Task 7 — LLM claim extractor wire-in**
   - `LLMClaimExtractor.extract_claims()` (line 118 of `models/decomposer/llm_claim_extractor.py`) is already implemented — just not wired into `scripts/prepare_eval_data.py` (line 161) or `demo/app.py`
   - Gate behind `--use-llm-extractor` CLI flag (default off for backward compat)
   - Round-trip fidelity: BLEU-4 + BERTScore-F1 + NLI entailment on N=500 reports

9. **Final smoke + doc-sync + self-check agent sweep** (after all 9 tasks)
   - Full `modal run scripts/modal_run_evaluation.py --checkpoint /data/checkpoints/verifier_binary_v4_rdrop/best_verifier.pt`
   - All verifiers: `python3 tests/test_provenance.py tests/test_hard_negative_generator.py tests/test_krippendorff.py tests/test_causal_term_identifier.py tests/test_regex_error_annotator.py tests/test_retrieval_batching.py tests/test_stratcp.py tests/test_llm_claim_extractor.py`
   - Parallel Opus self-check agents: logic review, requirements review, integration review, data-distribution review (mandatory for Tasks 1/2/3)

---

## Key Files & Locations

| Purpose | Path | Notes |
|---|---|---|
| **Plan (authoritative)** | `/Users/aayanalwani/.claude/plans/rosy-discovering-bubble.md` | 9-task sprint, $900 budget, NeurIPS D&B May 6 |
| v1 checkpoint | `checkpoints/v1_best_verifier.pt` | recovered from Modal volume, ≈1.6 GB, SHA in manifest |
| v3 checkpoint | Modal `/data/checkpoints/verifier_binary_v3/best_verifier.pt` | val_acc 0.9877 |
| v3 training data | Modal `/data/verifier_training_data_v3.json` | 30k examples, 13 hard-negative types, 16 MB |
| v3 eval data | Modal `/data/verifier_eval_v3/{calibration,test}_claims.json` | |
| OpenI images | Modal `/data/openi_images/` + local `/Users/aayanalwani/data/openi/` | 7477 PNGs |
| OpenI CSV | `/Users/aayanalwani/data/openi/openi_cxr_chexpert_schema.csv` | schema: `deid_patient_id/section_findings/section_impression` NOT `study_id/report_text` |
| Task 9 real workbooks | Modal `/data/same_model_experiment/real/annotation_workbook_run_{a,b}.json` | 414 + 401 claims |
| Task 9 real gate_demo | Modal `/data/same_model_experiment/real/gate_demo.json` | **not written yet** — orchestrator mid-scoring |
| Task 9 plan-C gate_demo | Modal `/data/same_model_experiment/gate_demo.json` | **not written yet** — backup, synthetic-stub workbooks |
| Provenance library | `inference/provenance.py` | 388 lines, 36/36 tests green, DO NOT re-plumb |
| Conformal triage | `inference/conformal_triage.py` | `gate_triage_with_provenance()` exists |
| Hard-negative generator | `data/augmentation/hard_negative_generator.py` | 981+ lines, 13 types in `_GENERATORS` |
| Counterfactual generator | `data/augmentation/counterfactual_generator.py` | word-boundary preservation check |
| Causal term identifier | `data/augmentation/causal_term_identifier.py` | SEP off-by-one fixed |
| Silver grader | `scripts/generate_silver_standard_graders.py` | rule-based CheXpert labeler |
| Real hallucination gen | `scripts/generate_real_hallucinations.py` | README-validated prompt, CheXagent deps |
| DPO/R-Drop trainer | `scripts/modal_train_dpo_refinement.py` | `loss_mode="consistency"` default |
| Gate demo | `scripts/demo_provenance_gate_failure.py` | 1100+ lines, orchestrator_remote included |
| Detach launcher | `scripts/launch_task9_detached.py` | sidecar JSON pattern |
| Wait-and-launch wrapper | `scripts/task9_wait_and_launch.sh` | PIPESTATUS[0] bug fixed |
| HO baseline | `scripts/baseline_hypothesis_only.py` | RoBERTa-large + (claim, MASKED_EVIDENCE) pair @ 512 |
| Self-annotation CLI | `scripts/self_annotate_silver_subset.py` | no label leak — locked by 2 regression tests |
| Krippendorff α | `scripts/compute_user_vs_ensemble_alpha.py` | ordinal + bootstrap + rung ladder |

**Sidecar JSONs (current session state):**
- `/tmp/task9_real_calls.json` — real run A/B FunctionCall IDs
- `/tmp/task9_real_orchestrator.json` — real orchestrator FunctionCall ID
- `/tmp/task9_function_calls_v3.json` — earlier v3 launcher handles (pre-smoke4)
- `/tmp/task9_function_calls.json` — v1/v2 launcher handles (historical; keep for audit)

---

## Critical Decisions & Why (this session)

- **DPO → R-Drop consistency (D15).** Original plan's chosen=counterfactual / rejected=original DPO pulls the ORIGINAL's score DOWN, not UP — opposite of invariance. Lit dive found no "Dually Self-Improved" paper (arXiv / OpenAlex / ACL / Semantic Scholar all empty). Replaced with R-Drop symmetric-KL (NeurIPS 2021) which is the actual consistency regularization technique. Citations corrected to R-Drop + UDA (NeurIPS 2020) + Kaushik ICLR 2020 for counterfactual data augmentation.
- **Rule-based CheXpert replaces HF CheXbert (D14).** Stanford `stanfordmlgroup/CheXbert` HF upload lacks the 14-head MLP — `model()` returns hidden states only, grader was silently stamping UNCERTAIN for every row. Rule-based CheXpert-style labeler with scope-terminated negation (60-char window, trimmed at "but"/"however"/"."/";") is a valid substitute given (a) the plan's α≥0.80 bar applies to the ensemble, not any single grader, and (b) this grader is only 1 of 3 in the ensemble.
- **HO baseline RoBERTa-large (claim, MASKED_EVIDENCE) @ 512 (D13).** DeBERTa-v3-large claim-only @ 128 would have been the wrong experiment — if the HO gap grows, you can't tell whether it's because the model learned evidence or because RoBERTa ≠ DeBERTa. Identical architecture + identical tokenization shape + only evidence changes = clean difference.
- **Server-side orchestrator for Task 9 (this session).** Original launcher did `.get()` from local Mac — breaks if Mac sleeps or disconnects. CPU Modal function that polls volume for both workbooks then fires scoring is entirely sleep-resilient, cheap ($0 — CPU + polling, not GPU), and lets the launcher exit cleanly after the spawn chain. This pattern should be reused for any multi-stage Modal job.
- **Nested `.remote()` removed from gate demo.** Demo function used to kick off CheXagent runs from inside a Modal container via `.remote()` — fragile (requires parent `app.run` context) and wasteful (outer H100 idle for 90 min while inner ones ran). Fix: restructured `_launch_chexagent_runs` to run from `main()` on local client.
- **Pre-flight review rule became global (this session).** Per user directive after three parallel Opus reviewers caught ~15 real bugs (inverted DPO preference pairs, HO wrong architecture, MedGemma wrong loader class, CheXbert ghost, CSV schema mismatch, CheXagent missing deps, CheXagent wrong prompt format, `$?` after pipe). Now mandatory in `~/.claude/CLAUDE.md` before any expensive remote action.
- **Binary classification, no label smoothing, inverted calibration on contradicted, global BH.** All four are architectural invariants from the v1 sprint — do NOT revisit. The "inverted" part (calibrate on H0 = hallucinated, not on H1 = faithful) is what lets cfBH give a valid FDR guarantee under the Jin & Candes 2023 framework.

---

## Known Issues / Gotchas

- **CheXagent prompt format is NOT the README's chat template.** The README-validated format is literally `' USER: <s>{prompt} ASSISTANT: <s>'` (leading space, explicit `<s>` tokens). `apply_chat_template` gave 1 claim / 0 claims outputs. Keep the strategy-0 format in `scripts/generate_real_hallucinations.py`.
- **CheXagent needs einops, cv2 (opencv-python-headless), albumentations, torchvision + libgl1 apt in the Modal image.** Without these, the custom modeling code imports fail silently and `model_used=="none"` in 12 seconds — that's the signature of this bug.
- **OpenI CSV schema is `deid_patient_id/section_findings/section_impression`, not `study_id/report_text`.** Handle both schemas in any CSV loader.
- **`transformers==4.40.0` is pinned for CheXagent.** Do NOT bump. Silver grader uses `transformers==4.50.0` in a SEPARATE Modal image — these cannot share.
- **`transformers==4.50.0` is pinned for silver grader (MedGemma).** MedGemma-4B requires `AutoModelForImageTextToText` (try first), fall back to `AutoModelForVision2Seq`, last-ditch `AutoModelForCausalLM`. Do not assume a single class.
- **`anthropic==0.40.0` is pinned.** Newer versions changed the retry/backoff API.
- **`stanfordmlgroup/CheXbert` on HF is a hidden-state-only ghost.** Do not try to use it as a classifier. Use the rule-based labeler in `scripts/generate_silver_standard_graders.py`.
- **`.gitignore` — do NOT add a blanket `data/` rule.** That silently excludes `data/augmentation/` and `data/preprocessing/` Python. Stick to file-extension patterns.
- **`${PIPESTATUS[0]}` not `$?` after a pipe.** `cmd | sed ...` returns sed's exit code (0), masking launcher failures. This bit the 2026-04-15 v1 Task 9 run.
- **Modal detach pattern**: `modal run --detach` for fire-and-forget, `Function.from_name().spawn()` for programmatic chaining with sidecar JSON capture. Use the second when you need to `get()` a FunctionCall later from a different process.
- **`task9_orchestrator_remote` is CPU-only.** If you accidentally give it a GPU, you burn H100 time polling.
- **`CheXagent-8b` OOMs on H100 with batch_size > 1 or beam search.** Stick to bs=1, sampling=True, temperature=0.7, top_p=0.9.
- **The plan's `Task 4` (regex annotator) is DEMOTED to diagnostic metadata only.** It must NOT change `TriageResult.final_label`. Medical-text false-positive risk is too high. Paper error-analysis tables consume the flags; triage decisions do not.
- **User's standing permissions (as of this session):** local git commits with passing tests + doc-sync: OK without asking. `git push` to remote: always ask. No destructive git ops. No `--no-verify`. Stage files by name.
- **Modal secrets needed for blocked tasks**: `anthropic` (for Tasks 1 + 3), and `ANTHROPIC_API_KEY` in local shell env for counterfactual generation.

---

## Environment

- Python: 3.9+ (system Python, no venv in use)
- Modal: authenticated, app keys `claimguard-verifier-binary` (training), `claimguard-evaluation` (eval), `claimguard-demo` (Task 9 gate demo), `claimguard-silver-graders` (Task 1)
- Modal volume: `claimguard-data`
- GPU policy: H100-only unless job is definitely < 15 min
- Critical pins: `transformers==4.40.0` (CheXagent image), `transformers==4.50.0` (silver grader image), `anthropic==0.40.0`, `trl==0.9.x` (DPO/R-Drop)
- Credentials status:
  - `modal token` — set ✓
  - Modal secret `anthropic` — **MISSING** (needed for Task 1 grader)
  - Local `ANTHROPIC_API_KEY` — **MISSING** (needed for Task 3 counterfactual gen)
  - GitHub push — authenticated ✓

---

## Commits This Session (newest first, all pushed)

```
a6603b0 task9: README-validated CheXagent prompt format (strategy 0) + dump fix
a22ffaa task9: add scripts/cat_chexagent_processor.py — source dumper
9586ffa docs: WAKEUP_STATUS_2026-04-15.md — overnight session handoff
d8e89d0 task9: apply_chat_template prompt path + status checker + Plan C fallback
9804c41 task9: CheXagent prompt format fix + server-side orchestrator
1c9663c task9: add CheXagent runtime deps to Modal image (einops, cv2, albumentations, torchvision)
dc2d491 task9: fix OpenI CSV schema mismatch + pipeline status check
1ceffe4 task9: detach-friendly launcher + wait-and-launch pipeline wrapper
d3b9053 v3 sprint: architectural fixes for Task 3 BUG A + Task 1 CheXbert ghost
5637b78 v3 sprint: pre-flight bug fixes for Task 1/2/3 before Modal launch
1907302 v3 sprint: silver-standard eval + counterfactual DPO + 12-type taxonomy + provenance gate demo
```

---

## Second Brain

The Obsidian vault at `~/Vault/` is the persistent knowledge base for all projects. **Read it before reading anything else** — it will tell you what matters.

- `~/Vault/CLAUDE.md` — wiki schema and operating rules (read fully)
- `~/Vault/index.md` — current state of all wiki pages
- `~/Vault/log.md` — last 5 entries show recent activity
- `~/Vault/wiki/projects/VeriFact.md` — the relevant project page (ClaimGuard-CXR)

Vault has been updated through this session. The `decisions.md` log entries D11–D19 cover every architectural decision from the v3 sprint.

---

## Reference Materials — MUST READ FULLY

A fresh session has ZERO project context. Every document below must be read **completely**, not skimmed.

### User's Global Instructions (READ FIRST, always)
- `~/.claude/CLAUDE.md` — **READ FULLY.** User's global preferences, coding rules, model-selection rules, active projects, git rules, pre-flight review rule. Every rule here is mandatory. Do NOT skip.

### Plan (authoritative sprint spec)
- `/Users/aayanalwani/.claude/plans/rosy-discovering-bubble.md` — **READ FULLY.** 9-task sprint, $900 budget, locked user decisions, hard constraints, dependency waves, budget table. This is the contract.

### Architecture & Design Docs
- `ARCHITECTURE.md` — **READ FULLY.** System blueprint. §5 retriever, §6 taxonomy (12 types), §13 sprint additions.
- `CLAIMGUARD_PROPOSAL.md` — **READ FULLY.** Design decisions + V3 ADDENDUM.
- `REPRODUCIBILITY.md` — hard negative taxonomy table, re-run commands.
- `MANUSCRIPT_MINI.md` — paper outline with error-analysis sections.
- `progress.md` — per-task status.
- `decisions.md` — D1 through D19, every architectural decision.

### Literature (read the actual papers)
- **Jin & Candes (JMLR 2023) "Selection by Prediction with Conformal p-values"** — foundation for inverted cfBH procedure. arXiv:2210.01408.
- **Kaushik, Hovy & Lipton (ICLR 2020) "Learning the Difference that Makes a Difference with Counterfactually-Augmented Data"** — basis for Task 3 counterfactual augmentation. arXiv:1909.12434.
- **Liang et al. (NeurIPS 2021) "R-Drop: Regularized Dropout for Neural Networks"** — symmetric-KL consistency loss used in Task 3 refinement. arXiv:2106.14448.
- **Xie et al. (NeurIPS 2020) "Unsupervised Data Augmentation for Consistency Training"** — complementary UDA consistency framing. arXiv:1904.12848.
- **Zhou et al. (medRxiv Feb 2026) "StratCP"** — stratified conformal baseline for Task 6. Zitnik Lab.
- **Chen et al. (2025) Real-world CXR hallucination taxonomy** — motivation for fabricated_measurement/prior/temporal types.

### Existing Codebase — READ BEFORE TOUCHING
| Purpose | Path | Read fully? |
|---|---|---|
| Provenance library | `inference/provenance.py` | **YES** |
| Conformal triage | `inference/conformal_triage.py` | **YES** |
| Hard-negative generator | `data/augmentation/hard_negative_generator.py` | **YES** |
| Counterfactual generator | `data/augmentation/counterfactual_generator.py` | **YES** |
| Causal term identifier | `data/augmentation/causal_term_identifier.py` | **YES** |
| Silver grader | `scripts/generate_silver_standard_graders.py` | **YES** |
| Gate demo | `scripts/demo_provenance_gate_failure.py` | **YES** |
| Real hallucination generator | `scripts/generate_real_hallucinations.py` | **YES** |
| DPO/R-Drop trainer | `scripts/modal_train_dpo_refinement.py` | **YES** |
| HO baseline | `scripts/baseline_hypothesis_only.py` | **YES** |
| Detach launcher | `scripts/launch_task9_detached.py` | Skim for pattern |
| Wait-and-launch wrapper | `scripts/task9_wait_and_launch.sh` | Skim |
| Self-annotation CLI | `scripts/self_annotate_silver_subset.py` | Skim |
| Eval runner | `scripts/modal_run_evaluation.py` | Skim (provenance inlined 212–654) |
| Training data prep | `scripts/prepare_eval_data.py` | Skim |

### Data & Artifacts
- Training data: Modal `/data/verifier_training_data_v3.json` (30k examples, 16 MB)
- Eval data: Modal `/data/verifier_eval_v3/{calibration,test}_claims.json`
- OpenI images: Modal `/data/openi_images/` (7477 PNGs) + `/Users/aayanalwani/data/openi/`
- OpenI CSV: `/Users/aayanalwani/data/openi/openi_cxr_chexpert_schema.csv` (schema: `deid_patient_id/section_findings/section_impression`)
- v1 checkpoint: `checkpoints/v1_best_verifier.pt` (≈1.6 GB, SHA-256 in `checkpoints/VERIFIER_V1_MANIFEST.json`)
- v3 checkpoint: Modal `/data/checkpoints/verifier_binary_v3/best_verifier.pt`
- Task 9 real workbooks: Modal `/data/same_model_experiment/real/annotation_workbook_run_{a,b}.json`
- Task 9 gate demo result: Modal `/data/same_model_experiment/real/gate_demo.json` (**pending**)

---

## How to Resume

Exact steps a fresh Claude session should take, IN ORDER:

1. **Read `~/.claude/CLAUDE.md` FULLY** — user's global rules (git commit rule, pre-flight review rule, model selection, research rules). These rules are mandatory.
2. **Read the vault** — `~/Vault/CLAUDE.md` (schema), `~/Vault/index.md` (state), last 5 entries of `~/Vault/log.md`, then `~/Vault/wiki/projects/VeriFact.md`.
3. **Read this handoff completely** — all sections, all tables.
4. **Read `/Users/aayanalwani/.claude/plans/rosy-discovering-bubble.md` FULLY.**
5. **Read `ARCHITECTURE.md` + `CLAIMGUARD_PROPOSAL.md` + `progress.md` + `decisions.md` FULLY.**
6. **Fetch and read the listed papers** (use WebFetch / arXiv MCP). Do not rely on memory.
7. **Read every codebase file marked "YES" in the Existing Codebase table.**
8. **Check running jobs** (polling commands under "What's Running" above). Confirm the 4 FunctionCalls are still alive or have completed.
9. **Download + inspect `/same_model_experiment/real/gate_demo.json`** if orchestrator has completed. Validate `downgrade_rate > 0.5` in same-model condition.
10. **THEN** start on the next pending task (Task 6 first if Task 9 validates).

**Do NOT start coding before steps 1–9 are complete.** Fresh sessions re-implement existing code, violate design constraints, ignore fixed bugs, and re-introduce bugs the pre-flight reviewer already caught. Every skipped reading step costs 30 min of debugging later.

---

## Self-Check Agents — MANDATORY

After writing any non-trivial code in the resumed session, spawn self-check agents in parallel (Opus, never Sonnet):

1. **Logic review agent** — off-by-ones, wrong signs, shape mismatches, missed edges
2. **Requirements agent** — matches plan + proposal + architecture doc exactly
3. **Integration agent** — does not break existing callers or invariants
4. **Data-distribution agent** (for data pipelines or training code) — no leakage, no label imbalance, eval/test/train distributions match

**Before any Modal GPU launch** — spawn a **pre-flight Opus reviewer** with:
- Concrete file paths
- Numbered, self-contained worry list
- Expected cost range (best/expected/worst)
- Ask for verdict: ready-to-launch / launch-with-fixes / do-not-launch

This rule is in `~/.claude/CLAUDE.md` as of this session. It caught 15+ real bugs this sprint. Never skip.

Report self-check findings to the user before proceeding.

---

## Open Questions for User (for next wake-up)

1. Did the real orchestrator (`fc-01KP8PJD5VJNYS5FNAH4MHB3MC`) complete? If yes, what was the `downgrade_rate`? Do we publish Plan C as backup or drop it?
2. Ready to run `modal secret create anthropic ANTHROPIC_API_KEY=...` and `export ANTHROPIC_API_KEY=...` to unblock Tasks 1 + 3?
3. When do you want to schedule the 90-min Task 8 self-annotation session?
4. Any changes to the NeurIPS D&B May 6 deadline strategy given the v3 sprint progress?

---

## Repro / smoke commands

```bash
# Environment check
cd /Users/aayanalwani/VeriFact/verifact
git log --oneline -15
git status --short

# Running job status
python3 -c "
import modal
for name, fc_id in [
    ('run_a', 'fc-01KP8PHQE0T5BP1RR1N29ZK6R1'),
    ('run_b', 'fc-01KP8PHQJ79EPPDD63E57YB1D8'),
    ('orchestrator', 'fc-01KP8PJD5VJNYS5FNAH4MHB3MC'),
    ('plan_c', 'fc-01KP8MBX7XQ3WQHH1FKH3CMWK9'),
]:
    try:
        modal.FunctionCall.from_id(fc_id).get(timeout=1); print(f'{name}: DONE')
    except TimeoutError:
        print(f'{name}: RUNNING')
    except Exception as e:
        print(f'{name}: {type(e).__name__}: {str(e)[:120]}')
"

# Verifiers still green
python3 tests/test_provenance.py
python3 tests/test_hard_negative_generator.py
python3 tests/test_counterfactual_generator.py
python3 tests/test_causal_term_identifier.py
python3 tests/test_self_annotator_no_leak.py  # Task 8 regression
python3 tests/test_krippendorff.py

# Volume inspection
modal volume ls claimguard-data /same_model_experiment/real/
modal volume ls claimguard-data /checkpoints/verifier_binary_v3/

# Download the primary gate_demo result when it lands
modal volume get claimguard-data /same_model_experiment/real/gate_demo.json /tmp/task9_real/ --force
python3 -c "
import json
with open('/tmp/task9_real/gate_demo.json') as f:
    d = json.load(f)
import json as j
print(j.dumps(d, indent=2)[:2000])
"
```
