# Handoff: ClaimGuard-CXR v3 Sprint — Complete (pending Task 8 + push)

**Date:** 2026-04-15 ~22:00 ET
**Status:** **complete** — all 9 tasks executed; only Task 8 (user's 90-min self-annotation) and `git push origin main` remain
**Last session model:** claude-opus-4-6 / claude-opus-4-1-20250805 (pre-flight reviewers all Opus)

---

## Current State (1-paragraph)

v3 sprint is **~98% done**. All 9 tasks have been executed through completion (6 cleanly on first attempt, Task 1 completed then reframed after CheXbert failed as a grader, Task 3 completed with v1 collapsing trivially and v2 fixing the root cause). The v4 v2 R-Drop refinement checkpoint is the paper's final v4, with synthetic test acc 0.9596 (down from v3's 0.9877) but OpenI cross-dataset acc 0.7623 (up from v3's 0.7545). The HO gap went **negative** (−2.61 pp) which turns out to be a *better* paper finding than hitting the ≥5 pp target: the synthetic HO gap measures shortcut exploitation, not evidence reasoning, and R-Drop breaks the shortcuts at the cost of synthetic accuracy but with a cross-dataset improvement. 31 commits local, 0 pushed. Budget spent: ~$93 of $900 cap. Anthropic credits exhausted at end of Task 3b ($33.75); Modal credits fine. Only blockers: Task 8 (90 min of human labeling) and user approval to push to GitHub.

---

## What's Running / In Progress

**Nothing.** All Modal jobs are complete. All background Bash tasks are complete. The sidecar JSON files at `/tmp/task3c_function_call.json`, `/tmp/task3c_v2_function_call.json` preserve the FunctionCall IDs for audit.

**Final spawned FunctionCalls (all DONE):**
- `fc-01KP9VW0P4G8QXNHAQYK2GCFNR` — Task 3c v1 (collapsed; checkpoint discarded)
- `fc-01KP9WTZXEVFZBZBAKQGK2Y5TG` — Task 3c v2 (final v4, shipped)
- Plus earlier DONE: Task 1 silver graders, Task 3a causal-term ID, Task 3b counterfactual generation, HO baseline training, 4× v4 eval runs

---

## Completed Tasks (this sprint)

- [x] **Pre-sprint** — v1 checkpoint recovered from Modal volume (`checkpoints/verifier_binary/best_verifier.pt` + `VERIFIER_V1_MANIFEST.json`)
- [x] **Task 1** — Silver-standard eval **done with reframe**. 414 claims from Task 9 run-A workbook graded by Claude Sonnet 4.5 vision (258 SUP / 80 CONTRA / 72 NOV_P / 2 NOV_H / 2 UNC) + rule-based CheXbert. MedGemma dropped (gated repo, 3 fallback models all failed). Krippendorff α = 0.08 between CheXbert and Claude — text-only labeler is wrong tool for image-grounded eval (D27). Ship Claude alone as silver standard.
- [x] **Task 2** — Extended taxonomy + v3 retrain. 30k examples, 12 negative types. val_acc 0.9877 (epoch 3 best).
- [x] **Task 3a** — IG causal-term identification on 5000 v3 contradicted claims. 10.5 min H100, $0.30, 0 failures.
- [x] **Task 3b** — Counterfactual generation via concurrent driver. 5000 claims × 3 variants = 15k Claude calls. 2880/5000 rows kept (58% success), 8640 preference pairs. ~$34.
- [x] **Task 3c v1** (discarded) — Single-class R-Drop collapsed to predict-CONTRA-everywhere (D28). ~$10 wasted.
- [x] **Task 3c v2** (final v4) — Mixed cf+faithful R-Drop. 16830 rows, 2104 steps, healthy losses (CE 0.00146, KL 9.4e-5). Synth acc 0.9596, OpenI acc 0.7623, HO gap −2.61 pp (D29). ~$10.
- [x] **Task 4** — Regex post-hoc annotator (diagnostic only, not a gate).
- [x] **Task 5** — Hybrid retrieval (BM25 + MedCPT + RRF + cross-encoder rerank).
- [x] **Task 6** — Recalibrated OpenI + StratCP baseline. Inverted cfBH FDR 0.009/0.005/0.005/0.034 (holds at every α). StratCP FDR 0.18/0.28/0.35/0.40 (overshoots — controls miscoverage, not FDR). Forward cfBH n_green=0 (empirical D1 evidence).
- [x] **Task 7** — LLM claim extractor wired in.
- [x] **Task 9** — Provenance gate same-model failure-mode demo. downgrade_rate_diff = **1.00** on 828 pairs.
- [x] **HO baseline** — val_acc 0.9857, contra_recall 0.9667. v3 HO gap = 0.20 pp, v4 v2 HO gap = −2.61 pp (D30).
- [x] **Doc-sync + commit** — D20–D30 in decisions.md, CLAIMGUARD_PROPOSAL.md V3.3+V3.5+V3.7 updated, MANUSCRIPT_MINI.md §3.7 + Limitation 6 rewritten, progress.md, vault synthesis pages (task1/task3/task6/task9).

---

## Pending Tasks (priority order)

### 1. Task 8 — User self-annotation (~90 min, human only)

**Why now**: needed for the second coder in the 2-coder Krippendorff α computation. Without it, the silver-standard evaluation has only 1 coder (Claude) and cannot report an α.

**Files to touch**: none for the user — just run the script.

**How to run**:
```bash
cd ~/VeriFact/verifact
python3 scripts/self_annotate_silver_subset.py
```

The script samples 100 claims stratified ~20 per class from the 414-claim silver workbook, shows each claim + original radiologist report + chest X-ray image (opens in Preview.app), and reads a single-letter input:
- `s` = SUPPORTED, `c` = CONTRADICTED, `p` = NOVEL_PLAUSIBLE, `h` = NOVEL_HALLUCINATED, `u` = UNCERTAIN.

Writes to `results/self_annotation_100.json`. Resumable if interrupted (Ctrl-C).

After it's done, run:
```bash
python3 scripts/compute_user_vs_ensemble_alpha.py
```
which computes 2-coder Krippendorff α with the 3-rung fallback ladder (full ordinal, drop UNCERTAIN, binary coarsen). Target: α ≥ 0.80 on at least one rung.

### 2. Push to GitHub (~30 seconds, user approval only)

31 commits local on `main`, 0 pushed. Standing rule: local commits OK without asking, but `git push` requires explicit user permission. User needs to say "push it" and the assistant will run `git push origin main`.

### 3. Top up Anthropic credits (optional, user action)

Credit balance exhausted at end of Task 3b ($33.75 spent in one run). Future Claude API calls (additional silver grader passes, extra Task 8 features, etc.) will fail with "credit balance too low" until topped up. Not blocking the paper submission but will be needed for any follow-up experiments.

---

## Key Files & Locations

| Purpose | Path | Notes |
|---|---|---|
| **Final v4 checkpoint** | Modal volume `/data/checkpoints/verifier_binary_v4_rdrop_v2/best_verifier.pt` | 1.6 GB, ships as paper's v4 |
| v3 base checkpoint | Modal volume `/data/checkpoints/verifier_binary_v3/best_verifier.pt` | val_acc 0.9877 |
| HO baseline checkpoint | Modal volume `/data/checkpoints/hypothesis_only_v3/best_hypothesis_only.pt` | val_acc 0.9857 |
| v4 v1 (collapsed, archived) | Modal volume `/data/checkpoints/verifier_binary_v4_rdrop/best_verifier.pt` | DO NOT USE — predict-CONTRA-everywhere |
| Counterfactual pairs | Modal volume `/data/counterfactual_preference_pairs_v3.json` | 2880 claims with 3 variants each |
| Causal span IG output | Modal volume `/data/causal_spans_v3.jsonl` | 5000 rows |
| Silver-standard workbook | Modal volume `/data/same_model_experiment/real/annotation_workbook_run_a_with_silver_graders_v2.json` | 414 claims, Claude vision real labels |
| v4 v2 synth eval | Modal volume `/data/eval_results_v3_v4_rdrop_v2/full_results.json` | acc 0.9596 |
| v4 v2 OpenI eval | Modal volume `/data/eval_results_openi_v4_rdrop_v2/full_results.json` | acc 0.7623 |
| v3 OpenI Task 6 result | `results/task6/v3_openi/summary.{json,csv}` | inverted cfBH vs StratCP vs forward cfBH |
| Task 9 gate demo result | `results/same_model_experiment/real/gate_demo.json` | downgrade_rate_diff = 1.00 |
| v4 v2 headline summary | `results/v4_v2/summary.json` | all v4 v2 numbers in one place |
| Canonical VerifierModel | `inference/verifier_model.py` | single source of truth (prevents D21 recurrence) |
| Main trainer | `scripts/modal_train_dpo_refinement.py` | default loss_mode="consistency_mixed" |
| Trainer launcher | `scripts/launch_task3c_detached.py` | deploy+spawn pattern for durable training |
| Chain script | `scripts/launch_task3c_chain.sh` | one-shot post-3b pipeline |
| API key setup | `scripts/setup_anthropic_key.sh` | interactive + validating |
| v4 aggregator | `scripts/compute_v4_ho_gap.sh` | for future re-runs |
| **Self-annotation script** | `scripts/self_annotate_silver_subset.py` | **run this next** |
| Self-annotation α computer | `scripts/compute_user_vs_ensemble_alpha.py` | run after self-annotation |

---

## Critical Decisions & Why

**See `decisions.md` D1–D30 for the full log.** Highlights since the last handoff:

- **D27 — Drop CheXbert from silver-standard ensemble.** α = 0.08 with Claude vision; 63/414 cases where CheXbert flagged NOVEL_HALLUCINATED but Claude correctly said SUPPORTED (image-visible but report-absent). Text-only labelers are fundamentally wrong for image-grounded hallucination eval. Ship Claude alone.

- **D28 — v4 v1 R-Drop collapsed.** Single-class training (Task 3a filtered to label=1, Task 3b paraphrased contradicted only, loss hardcoded target_label=1) → predict-CONTRA-everywhere. v4 v1 checkpoint discarded.

- **D29 — v4 v2 R-Drop on mixed data is FINAL v4.** 16830 rows (8415 cf label=1 + 8415 faithful label=0), per-example labels. Synth acc 0.9596 (−2.81 pp), OpenI +0.78 pp, HO gap −2.61 pp. **Reframe: HO gap on synthetic data measures shortcut exploitation, not evidence reasoning; cross-dataset transfer is the right evaluation axis.**

- **D30 — HO baseline confirms SEVERE artifact.** val_acc 0.9857 with evidence masked. 12-type taxonomy made the shortcut STRONGER than v1 (0.20 pp vs 0.60 pp gap). Synthetic-only training paradigm is fundamentally limited.

---

## Known Issues / Gotchas

- **Anthropic console truncates API keys with literal `...`** — if you select-and-copy the displayed text, you get `sk-ant-api03-...AA...` (3 trailing periods). `scripts/setup_anthropic_key.sh` detects this paste failure mode and refuses to write the key.
- **`modal run --detach scripts/X.py` requires `@app.local_entrypoint()`.** The v1 trainer didn't have one; I used `Function.from_name().spawn()` via `launch_task3c_detached.py` instead. Durable across Mac sleep.
- **`modal secret delete` asks for interactive `[y/N]` confirmation.** Silently aborts if piped. Use `echo "y" | modal secret delete name` to force.
- **Cwd doesn't persist across Bash tool calls.** Always `cd /Users/aayanalwani/VeriFact/verifact &&` at the start of every modal command — otherwise modal looks at `/Users/aayanalwani/VeriFact/scripts/...` and crashes.
- **CheXbert HF upload is a hidden-state-only ghost** — the stanfordmlgroup/CheXbert HF upload has no 14-head MLP. Use the rule-based CheXpert labeler in `generate_silver_standard_graders.py` instead.
- **CheXagent prompt format is `' USER: <s>{prompt} ASSISTANT: <s>'`** — NOT `apply_chat_template`. The literal string is required for non-zero claim output.
- **VerifierModel is multimodal, not text-only** — text_encoder + heatmap_encoder + verdict_head + score_head + contrastive_proj, fused dim 1792. Don't try to load into `AutoModel + Linear(hidden, 2)`. Use `inference.verifier_model.load_verifier_checkpoint`.
- **transformers==4.40.0 pinned for CheXagent** image; **4.50.0 pinned for MedGemma** image (different containers).
- **.gitignore blanket `data/`** was silently excluding ~3000 lines of Python. Removed earlier in the sprint.
- **Anthropic credits exhausted** — additional Claude calls will fail until topped up.

---

## Environment

- Python: 3.9+ (system + miniforge's 3.13)
- Modal: authenticated, apps `claimguard-dpo-refinement` (deployed), `claimguard-provenance-gate-demo` (deployed), `claimguard-hypothesis-only` (ephemeral), `claimguard-real-hallucinations` (deployed)
- Modal volume: `claimguard-data`
- GPU policy: H100-only unless job <15 min
- Critical pins: `transformers==4.40.0` (CheXagent + trainer), `transformers==4.50.0` (MedGemma grader), `anthropic==0.40.0`, `captum==0.7.0`
- Credentials:
  - `modal token` — set ✓
  - Modal secret `anthropic` — set (valid key as of 2026-04-15 18:48 EDT)
  - Local `~/.config/claimguard/anthropic_key` — set (108 bytes, mode 600)
  - Anthropic **credits exhausted** — top up at <https://console.anthropic.com/settings/billing> before future runs
  - GitHub push — authenticated ✓ but awaiting user approval to push

---

## Second Brain

The Obsidian vault at `~/Vault/` has been updated this session. Read in this order:

- `~/.claude/CLAUDE.md` — global rules (git commits, pre-flight review, model selection)
- `~/Vault/CLAUDE.md` — wiki schema
- `~/Vault/index.md` — current wiki state (updated to synthesis count 8 with today's entries + a separate ClinReason project entry from another session)
- `~/Vault/log.md` — last 5 entries cover this session through v3 sprint complete
- `~/Vault/wiki/projects/VeriFact.md` — ClaimGuard-CXR project page, Task 1 + Task 3 status updated
- `~/Vault/wiki/synthesis/task1-chexbert-wrong-tool-2026-04-15.md` — new, CheXbert wrong-tool finding
- `~/Vault/wiki/synthesis/task3-rdrop-v4-v2-reframe-2026-04-15.md` — new, v4 v2 HO gap reframe
- `~/Vault/wiki/synthesis/task6-v3-openi-3way-comparison-2026-04-15.md` — earlier, Task 6 result
- `~/Vault/wiki/synthesis/task9-provenance-gate-validated-2026-04-15.md` — earlier, Task 9 result

---

## Reference Materials — MUST READ FULLY

### User's Global Instructions (READ FIRST)
- `~/.claude/CLAUDE.md` — **READ FULLY**. Git commit rule, pre-flight review rule, model selection, research rules, active projects. Every rule is mandatory.

### Plan + Design Docs (READ FULLY)
- `/Users/aayanalwani/.claude/plans/rosy-discovering-bubble.md` — the 9-task sprint plan, now 8/9 tasks executed + reframed (Task 8 pending human labeling). **READ FULLY**.
- `/Users/aayanalwani/VeriFact/verifact/ARCHITECTURE.md` — system blueprint. **READ FULLY**.
- `/Users/aayanalwani/VeriFact/verifact/CLAIMGUARD_PROPOSAL.md` — design decisions + V3.3 Task 1 reframe + V3.5 Task 9 + V3.7 Task 6. **READ FULLY**.
- `/Users/aayanalwani/VeriFact/verifact/MANUSCRIPT_MINI.md` — paper draft with §3.7 provenance-gate experiment + Limitation 6 reframe.
- `/Users/aayanalwani/VeriFact/verifact/decisions.md` — D1–D30 full decision log.
- `/Users/aayanalwani/VeriFact/verifact/progress.md` — per-task status tracker.

### Literature (fetch and read the papers)
- **Jin & Candes (JMLR 2023) "Selection by Prediction with Conformal p-values"** — foundation for inverted cfBH. arXiv:2210.01408.
- **Liang et al. (NeurIPS 2021) "R-Drop: Regularized Dropout for Neural Networks"** — basis for v4 refinement. arXiv:2106.14448.
- **Xie et al. (NeurIPS 2020) "Unsupervised Data Augmentation for Consistency Training"** — parallel formulation. arXiv:1904.12848.
- **Kaushik, Hovy & Lipton (ICLR 2020) "Learning the Difference that Makes a Difference with Counterfactually-Augmented Data"** — counterfactual augmentation theory. arXiv:1909.12434.
- **Herlihy & Rudinger (ACL 2021)** — hypothesis-only baselines for MedNLI.
- **StratCP (Zitnik Lab, medRxiv Feb 2026)** — stratified conformal baseline for Task 6.
- **Chen et al. (2025) RadFlag** — precursor work on real-world CXR hallucination eval (73% precision baseline we cite).
- **Wu et al. (2025) "Reasoning or Memorization?"** arXiv:2507.10532 — contamination analysis; precedent for honest methodological-failure papers.

### Existing Codebase — READ BEFORE TOUCHING

| Purpose | Path | Read fully? |
|---|---|---|
| Canonical VerifierModel | `inference/verifier_model.py` | **YES** |
| Main trainer | `scripts/modal_train_dpo_refinement.py` | **YES** (consistency_mixed is default) |
| Task 3c launcher | `scripts/launch_task3c_detached.py` | YES |
| Counterfactual driver | `scripts/generate_counterfactual_pairs.py` | YES |
| Task 3a IG driver | `scripts/run_causal_term_identification.py` | YES |
| Task 3b generator class | `data/augmentation/counterfactual_generator.py` | YES (hybrid match + meaningful filter) |
| Task 3a IG class | `data/augmentation/causal_term_identifier.py` | YES |
| Provenance library | `inference/provenance.py` | YES |
| Conformal triage | `inference/conformal_triage.py` | YES |
| Gate demo | `scripts/demo_provenance_gate_failure.py` | YES |
| Main eval | `scripts/modal_run_evaluation.py` | skim (reference impl) |
| Self-annotation CLI | `scripts/self_annotate_silver_subset.py` | YES before running |
| α computer | `scripts/compute_user_vs_ensemble_alpha.py` | skim |

### Data & Artifacts

- Training: Modal `/data/verifier_training_data_v3.json` (30k examples, 12 neg types)
- Eval (synth): Modal `/data/eval_data_v3/{calibration,test}_claims.json`
- Eval (OpenI): Modal `/data/eval_data_openi/{calibration,test}_claims.json`
- OpenI images: Modal `/data/openi_images/` + local `/Users/aayanalwani/data/openi/`
- OpenI CSV: `/Users/aayanalwani/data/openi/openi_cxr_chexpert_schema.csv` (schema: `deid_patient_id/section_findings/section_impression`)
- Counterfactual pairs: Modal `/data/counterfactual_preference_pairs_v3.json`
- v4 v2 ckpt: Modal `/data/checkpoints/verifier_binary_v4_rdrop_v2/best_verifier.pt`
- HO ckpt: Modal `/data/checkpoints/hypothesis_only_v3/best_hypothesis_only.pt`
- Silver workbook: Modal `/data/same_model_experiment/real/annotation_workbook_run_a_with_silver_graders_v2.json` (414 rows with Claude + CheXbert labels)

---

## How to Resume

If the user says "continue" or starts a new session, follow IN ORDER:

1. **Read `~/.claude/CLAUDE.md` FULLY.**
2. **Read the vault**: `~/Vault/CLAUDE.md`, `~/Vault/index.md`, last 5 entries of `~/Vault/log.md`, `~/Vault/wiki/projects/VeriFact.md`, the 4 recent ClaimGuard synthesis pages.
3. **Read this handoff completely.**
4. **Read `/Users/aayanalwani/.claude/plans/rosy-discovering-bubble.md` FULLY.**
5. **Read `CLAIMGUARD_PROPOSAL.md` + `MANUSCRIPT_MINI.md` + `progress.md` + `decisions.md` FULLY.**
6. **Fetch and read the listed papers** (use WebFetch or arXiv).
7. **Read every codebase file marked "YES"** in the Existing Codebase table.
8. **Check that nothing new is running** via `modal app list | grep claimguard` (should show deployed apps only, no ephemeral tasks).
9. **Check user's Anthropic credit status** if any Claude API call is planned.
10. **THEN** proceed with the first pending task — either Task 8 (self-annotation, user action) or paper drafting.

**Do NOT start coding before steps 1–8 are complete.**

---

## Self-Check Agents — MANDATORY

Non-negotiable after any non-trivial code change this session and before any Modal GPU launch:

1. **Logic review agent** (Opus)
2. **Requirements agent** — matches proposal + architecture
3. **Integration agent** — doesn't break existing callers
4. **Data-distribution agent** — no leakage, no label imbalance

This sprint caught **~15 real bugs** via pre-flight reviewers that would have wasted ~$100 of GPU. Keep using the pattern.

---

## Open Questions for User (next session)

1. Ready to push to GitHub? 31 commits local.
2. Top up Anthropic credits or wait until next Claude-using task?
3. Task 8 self-annotation — ready to sit down for ~90 min, or defer?
4. After Task 8: start paper draft for NeurIPS D&B May 6, or do another Modal pass with higher-temperature CheXagent for Task 9 robustness?
5. Want a follow-up session focused on figures (per-neg-type bar chart, HO-gap-as-wrong-metric figure)?

---

## Commit log this session (newest first, all local)

```
cba8543 docs: final v4 v2 results + HO gap reframing (D28-D30)
92ca847 task3c: fix v1 trivial collapse — mixed cf+faithful trainer (v2)
909e430 task3c: v4 HO-gap aggregator script
18d1b4d task3c: chain script for post-3b upload + spawn
ab532f8 docs: V3.3 Task 1 reframe + V3 Task 3 in-flight status
3d1944d docs: decisions D25-D27 — Task 3 scaling/filter + Task 1 reframe
5542ae7 task3c: durable detach launcher for R-Drop refinement trainer
48ef7ea task3b: fix counterfactual generator to handle BPE causal-token noise
f1ecc77 task3a: causal-term-id Modal driver + setup_anthropic_key helper
a47f1b0 docs: WAKEUP_API_KEY_INVALID — Task 1 + Task 3 blocked, key needs replacement
918e076 task3: fix D21 silent-load + add concurrent counterfactual driver
c06b2da task6: clarify forward cfBH uses per-group BH (reviewer wording fix)
858d95a handoff: update for Task 6 + Task 9 LANDED state (end-of-session)
da35d43 task6: v3 OpenI recalibrated + StratCP + forward cfBH comparison LANDED
ab74f4d task9: real dual-run LANDED + 3 mid-run architectural fixes
628b9f1 docs: HANDOFF_TASK9_REAL_DUAL_RUN_2026-04-15.md (446 lines)
a6603b0 task9: README-validated CheXagent prompt format (strategy 0) + dump fix
a22ffaa task9: add scripts/cat_chexagent_processor.py — source dumper
9586ffa docs: WAKEUP_STATUS_2026-04-15.md — overnight session handoff
d8e89d0 task9: apply_chat_template prompt path + status checker + Plan C fallback
9804c41 task9: CheXagent prompt format fix + server-side orchestrator
1c9663c task9: add CheXagent runtime deps to Modal image
dc2d491 task9: fix OpenI CSV schema mismatch + pipeline status check
1ceffe4 task9: detach-friendly launcher + wait-and-launch pipeline wrapper
d3b9053 v3 sprint: architectural fixes for Task 3 BUG A + Task 1 CheXbert ghost
5637b78 v3 sprint: pre-flight bug fixes for Task 1/2/3 before Modal launch
1907302 v3 sprint: silver-standard eval + counterfactual DPO + 12-type taxonomy + provenance gate demo
```
