# Handoff: ClaimGuard-CXR v6.0 — End of 2026-04-20

**Date:** 2026-04-20 late afternoon
**Status:** compute ~85% complete, paper ~80% drafted, handoff written
**Last session model:** Claude Opus 4.7 (1M context)
**Project root:** `/Users/aayanalwani/VeriFact/verifact/`
**GitHub:** `https://github.com/alwaniaayan6-png/ClaimGuard-CXR` (public, branch `v5-image-grounded`)
**Companion docs:** `PLAN_V6_17DAY_NEURIPS.md`, `ARCHITECTURE_V6_0_NEURIPS_MAIN.md`, `HANDOFF_CLAIMGUARD_V6_2026-04-19.md` (previous)
**Submission target:** NeurIPS 2026 Evaluations & Datasets track, deadline **2026-05-06 AoE** (16 days remaining)

---

## Current state (1-paragraph summary)

The project has produced all major data artifacts and most compute results. v6.0-3site training finished with **IMG 75.25pp @ val_acc 0.954** — the headline number beating v5.3 (69.24pp / 0.925). Three-way cross-site leave-one-out trained on all three held-out configurations (held-out IMG evaluation pending). RadFact and VERT silver labels are complete on both batches (368 CheXagent + 2,339 gated = 2,707 claims total). GREEN silver labels are in a **partial state**: CheXagent batch fully re-parsed (111 SUP / 90 CONTRA / 167 UNCERTAIN); gated batch timed out at 1,142 of 2,339 claims and needs a resume. Paper has v6 content drafted in `mlhc_build/paper.tex` with 4 numeric placeholders remaining (LOO off-diagonal cells, baseline P/R for pending detector re-runs). Gebru 2021 datasheet complete at 262 lines. Poster script with 20+ figure prompts authored. **Five critical bug classes were caught and fixed today** — four via pre-flight review (HO row_idx misalignment, volume commit race, silver_green single-path, GREEN matplotlib dependency) and one via post-hoc output inspection (GREEN parser regex format mismatch). Detailed lessons-learned section below.

---

## What's running / in progress right now

**Nothing.** All Modal jobs have completed or timed out. The next session starts by spawning the GREEN-gated resume job.

To verify:
```bash
modal app list | grep claimguard-v6
modal volume ls claimguard-v5-data /v6_silver/
modal volume ls claimguard-v5-data /v6_rrg/
```

Expected state on volume:
- `/v6_silver/green_labels.jsonl` — 368 rows, CheXagent, re-parsed (uploaded via `modal volume put`)
- `/v6_silver/green_labels_gated.jsonl` — **1,142 rows** (partial, timed-out, needs backfill)
- `/v6_silver/radfact_labels.jsonl` — 368 rows CheXagent, complete
- `/v6_silver/radfact_labels_gated.jsonl` — 2,339 rows gated, complete
- `/v6_silver/vert_labels.jsonl` — 368 rows CheXagent, complete
- `/v6_silver/vert_labels_gated.jsonl` — 2,339 rows gated, complete
- `/v6_silver/claims.jsonl`, `claims_gated.jsonl`, `references.jsonl` — all complete
- `/v6_rrg/generations.jsonl` — 500 CheXagent reports, complete
- `/v6_rrg/generations_gated.jsonl` — 1,000 MAIRA-2 + MedGemma reports, complete
- `/groundbench_v5/all_v6/` — 3-site train/val/cal/test JSONLs, complete
- `/groundbench_v5/all_v6_loo/` — 6 files (train_no_SITE, val_no_SITE for each site), complete
- `/groundbench_v5/ho_filter_weights_v6.jsonl` — computed on v6 3-site train, complete
- `/groundbench_v5/padchest_gr_records.jsonl` — 4,555 records, complete
- `/checkpoints/claimguard_v6/v6_0_3site/` — Day 6 checkpoint, complete
- `/checkpoints/claimguard_v6/v6_0_loo_no_{openi, chestx_det10, padchest-gr}/` — 3 LOO checkpoints, complete

---

## Completed today (delta from 2026-04-19 handoff)

### Data + RRG
- [x] PadChest-GR image download finished (44.3 GB, 46 files, 0 errors)
- [x] 3-site GroundBench-v6 assembled: 40,585 rows across 4 splits
- [x] LOO splits assembled (3 held-out configs): 15k / 21k / 29k train rows
- [x] 500 × 3 RRG reports generated (MAIRA-2, CheXagent-2-3b, MedGemma-4B-IT)
- [x] 2,707 atomic claims decomposed via Claude Haiku on Modal

### Training
- [x] HO filter v6 computed on v6 3-site train (20,560 rows, 85% text-solvable flagged)
- [x] **Day 6 v6.0-3site DONE**: IMG 75.25pp, val_acc 0.954 (Δ+6pp IMG, +3pp acc over v5.3)
- [x] Day 7 LOO × 3: all three held-out configurations trained. **Held-out-site IMG diagnostic not yet evaluated** (need one more pass per checkpoint)

### Silver labeling
- [x] RadFact labels: 368 CheXagent (202/65/101 SUP/CON/UNC) + 2,339 gated (1,396/194/749)
- [x] VERT labels: 368 CheXagent (223/111/34) + 2,339 gated (1,542/363/434)
- [x] GREEN labels CheXagent: fixed parser, re-parsed offline (111/90/167)
- [ ] GREEN labels gated: **1,142 of 2,339 done** — timed out at Modal 2h limit, needs backfill

### Baselines
- [x] Day 8 baseline_eval partial: BiomedCLIP (IMG 8pp / ESG 0pp) + MedGemma (IMG 14pp / ESG 2pp) both show ESG evidence-blindness
- [ ] MAIRA-2 + Claude 3.5 Sonnet baseline calls returned 0.0 — need debugging

### Paper / documentation
- [x] `mlhc_build/paper.tex` v6 additions: 3-site cohort, real-RRG eval, 3-way LOO, baseline landscape, silver-labeling methodology (committed `2518a86`)
- [x] `mlhc_build/datasheet.md` — 262-line Gebru 2021 schema (committed `2518a86`)
- [x] `mlhc_build/poster_figure_script.md` — 8-act narrative linking all 20+ figure prompts
- [x] 20+ figure prompts written in Distill / Nature Methods style: F1, F2, F3, F5, F6 (expanded), F7 (vertical), F8, F9, F10, F11, F13, F14, F15, F16 (NEW, report-level issue), F17 (NEW, safety-bound issue), F18 (NEW, 3 diagnostics), F19 (cfBH, optional since F6 absorbs it), plus objectives-overview and discussion/conclusion figures
- [x] Limitations + future-work bullet drafts (multiple revisions, poster-ready)

### Bug fixes (see Lessons Learned below for context)
- [x] Pre-flight review caught 3 blockers (HO row_idx / volume race / ensemble single-path) → committed `c056609`
- [x] GREEN matplotlib missing → added to image → committed `7e6dca3`
- [x] GREEN parser format mismatch → rewritten for actual `[Clinically Significant Errors]: (a) ...` format → committed `27d3c84`
- [x] Multiple CheXagent image incrementally extended (matplotlib, tensorflow, albumentations pin, opencv, einops, timm, ftfy, regex, protobuf, pyarrow, scikit-learn, scikit-image) as remote_code deps surfaced
- [x] CheXagent FP32 vs BF16 dtype fix (commit `c21fb3c`)
- [x] CheXagent transformers 4.40 vs MAIRA-2 4.51 split into two Modal images (commit `2106cc2`)
- [x] Silver_green had no chex vs gated split → added distinct output paths when spawning (no code change; caller responsibility)
- [x] silver_ensemble rewritten to accept list inputs for union across CheXagent + gated batches (commit `c056609`)
- [x] `volume.reload()` added to `train_v6_loo`, `silver_green`, `silver_ensemble` (commit `c056609`)
- [x] `adversarial_ho_filter=False` for LOO runs (commit `c056609`) — HO weights not aligned to LOO train files

---

## Pending tasks (priority order for next session)

### Tier 1 — blocks everything downstream (~2h)
1. **Backfill GREEN on remaining 1,197 gated claims.** Respawn `silver_green` with a claims file containing ONLY claims not already in `green_labels_gated.jsonl`. Use 4h timeout (not default 2h). Append to existing file. Re-parse offline afterward.
2. **silver_ensemble union.** Spawn with `green_paths=['/data/v6_silver/green_labels.jsonl', '/data/v6_silver/green_labels_gated.jsonl']` etc. Output: `/data/v6_silver/ensemble.jsonl` + `/data/v6_silver/ensemble_stats.json` (Krippendorff α).

### Tier 2 — paper-critical downstream (~4h)
3. **padchest_gr_validate** — Cohen κ of silver ensemble vs PadChest-GR 7,037 radiologist-placed bboxes. Target κ ≥ 0.5.
4. **LOO held-out-site IMG evaluation.** The 3 LOO checkpoints are trained; their IMG on the held-out site still needs computation. Write a small Modal function that loads each checkpoint, runs the evidence-blindness diagnostic on the held-out site's test split, and writes per-LOO IMG. Fills 3 placeholders in Figure 13.
5. **Day 9 hidden-state probe** on MAIRA-2 + MedVersa — `v5/eval/hidden_state_probe.py` ready; orchestrator entrypoint exists.
6. **Day 9 RadFlag replication** on MAIRA-2 — `v5/eval/radflag_replica.py` ready.
7. **Day 10 artifact audit** — text-only RoBERTa pre/post HO filter, measure class-prior-adjusted text-only ceiling. Target Δ ≤ 2pp post-HO.
8. **Day 10 HO-filter-on-real-RRG** activation rate. Test M6 gate from plan: is HO filter activation ≥ 10% on real-RRG hallucinations?
9. **Day 10 cross-site mechanism** — support-score shift KS, HO activation per site, text-only ceiling per site. Input to the F13 mechanistic diagnosis.
10. **Debug MAIRA-2 + Claude baseline failures** — MAIRA-2 processor API mismatch in verifier mode; Claude rate-limit on Opus 4.7.

### Tier 3 — polish and submit (~2-3 days)
11. **Paper number swap.** Replace remaining placeholders in `mlhc_build/paper.tex` with real numbers as they land from Tier 2.
12. **Reproducibility checklist** (NeurIPS E&D required).
13. **Ethics statement final pass.**
14. **Bibliography verification** — compile with `tectonic` and ensure all 25-30 refs render.
15. **Generate figures via image-gen tool** (Nano Banana 2 / FLUX / DALL-E 3 / Imagen 4) using the 20+ prompts in `poster_figure_script.md`. Data figures (F11, F12, F13, F15) are better produced via matplotlib from the actual JSON results.
16. **Final Opus pre-flight review on paper.**
17. **OpenReview submission mechanics** (Day 17).

---

## Key files & locations

### Planning / architecture
- `PLAN_V6_17DAY_NEURIPS.md` — 17-day execution plan with Day 0-17 schedule
- `ARCHITECTURE_V6_0_NEURIPS_MAIN.md` — authoritative v6.0 spec
- `HANDOFF_CLAIMGUARD_V6_2026-04-19.md` — previous handoff (yesterday)
- `HANDOFF_CLAIMGUARD_V6_2026-04-20.md` — this doc

### Paper / poster
- `mlhc_build/paper.tex` — v6 paper draft (~330 lines with v6 additions)
- `mlhc_build/datasheet.md` — Gebru 2021 datasheet
- `mlhc_build/poster_figure_script.md` — 8-act figure narrative + 20+ prompts

### Code — evaluation modules
- `v5/eval/baselines.py` — 7 baseline verifier classes (CheXagent, MAIRA-2, MedGemma, BiomedCLIP, Claude/GPT/Gemini APIs)
- `v5/eval/rrg_generate.py` — 3 RRG generator classes
- `v5/eval/green_labeler.py` — GREEN-RadLlama2-7b silver labeler (parser fixed today)
- `v5/eval/radfact_labeler.py` — RadFact (Haiku decompose + Opus entail)
- `v5/eval/vert_labeler.py` — VERT structured-prompt labeler
- `v5/eval/silver_ensemble.py` — 3-grader ensemble with Krippendorff α (now supports list inputs)
- `v5/eval/hidden_state_probe.py` — ReXTrust-inspired white-box probe
- `v5/eval/radflag_replica.py` — RadFlag black-box temperature-sampling
- `v5/eval/artifact_audit.py` — text-only RoBERTa pre/post HO-filter
- `v5/eval/padchest_gr_validate.py` — silver κ vs radiologist bboxes
- `v5/eval/evidence_blindness.py` — IMG/ESG/IPG diagnostic (v5 vintage, still works)

### Code — data modules
- `v5/data/padchest_gr.py` — loader + translation helper
- `v5/data/groundbench.py` — multi-site assembly
- `v5/data/claim_extractor.py` — rule + LLM claim extraction
- `v5/data/claim_parser.py` — structured claim parsing
- `v5/data/claim_matcher.py` — claim-to-annotation matching with IoU

### Code — Modal orchestrators
- `v5/modal/v6_orchestrator.py` — **primary orchestrator**, ~900 lines, 14 entrypoints:
  - `rrg_generation_chexagent` (image_chexagent, transformers 4.40)
  - `rrg_generation` (image, transformers 4.51, MAIRA-2 + MedGemma)
  - `decompose_rrg_claims` (CPU, Haiku)
  - `silver_green` (H100, GREEN-RadLlama2-7b)
  - `silver_radfact` (CPU, Anthropic API)
  - `silver_vert` (CPU, Anthropic API)
  - `silver_ensemble` (CPU, list-input)
  - `padchest_gr_assemble` (CPU + Haiku)
  - `assemble_v6_3site` (CPU)
  - `assemble_v6_loo_splits` (CPU)
  - `train_v6_3site` (H100, 4h timeout)
  - `train_v6_loo` (H100, 4h timeout, HO-filter disabled per pre-flight)
  - `baseline_eval` (H100, 3h timeout)
  - `hidden_state_probe` (H100:80GB, 3h timeout)
  - `artifact_audit` (H100, 2h timeout)
  - `padchest_gr_validate` (CPU, 30min timeout)
- `v5/modal/ho_filter.py` — HO filter training entrypoint
- `v5/modal/run_v5_pipeline.py` — legacy v5 orchestrator

### Code — training / model
- `v5/train.py` — `V5TrainConfig` + `train_v5` function
- `v5/model.py` — `V5Config` + `ImageGroundedVerifier`
- `v5/losses.py` — 5-term composite loss
- `v5/ho_filter.py` — HO filter implementation
- `v5/conformal.py` — inverted cfBH

### Tests
- `v5/tests/test_silver_ensemble.py` — 14 tests, Krippendorff + decision rule
- `v5/tests/test_silver_labelers.py` — 20 tests, mock Anthropic calls
- `v5/tests/test_padchest_gr_validate.py` — 11 tests, tokenize + Cohen κ
- Plus 32 older tests from v5 era. Current: **77 passing, 2 skipped** (env-dependent).

### Results JSONs (local copies under `v5_final_results/`)
- `TIER3_RESULTS_2026-04-19.md` — v5.0 Tier 3 writeup
- `v5_results.json` — canonical machine-readable
- Per-config diagnostic JSONs

### Documents
- `CLAIMGUARD_PROPOSAL.md` — deprecated, now points to v6.0 arch doc
- `VERIFACT_HANDOFF.md` — v1/v2 vintage
- `ARCHITECTURE_V5_0_EVIDENCE_BLINDNESS.md` — superseded by v6.0

---

## Critical decisions & why

- **NeurIPS 2026 E&D track, not Main** — same May 6 deadline, charter matches diagnostic-work, P(accept) 18-25% vs 3-6% on main. Decided 2026-04-19 in plan.
- **Diagnostic-first reframe (Gururangan style)** — paper's headline is the IMG/ESG/IPG framework, not the verifier architecture. Aggregate accuracy warning is Sections 1-3.
- **Three sites only** (no credentialed expansion) — PadChest-GR + OpenI + ChestX-Det10 give multi-site story without MIMIC-CXR credentialing.
- **Two Modal images** (not one) — CheXagent hard-pins transformers 4.40; MAIRA-2 + MedGemma need 4.51. Reconciled via `image_chexagent` vs `image`.
- **FP32 for CheXagent** — its trust_remote_code image loader emits FP32 regardless of model dtype. BF16 caused hard mismatch errors on every inference.
- **Three silver graders with MIMIC-leakage decoupling** — GREEN (MIMIC-trained) + RadFact + VERT (both MIMIC-free). Krippendorff α reported per subset.
- **HO filter DISABLED for LOO runs** — v5 / v6-3site HO weights don't align positionally to LOO train files. Per pre-flight review. Precedent at `run_v5_pipeline.py:970`.
- **adversarial_ho_filter=True for v6.0-3site** — uses `ho_filter_weights_v6.jsonl` (computed on v6 3-site train, aligned).
- **volume.reload() at every function entry that reads from recently-written paths** — Modal volume commit-on-exit has race semantics; explicit reload is belt+suspenders.
- **silver_green uses distinct `out_jsonl` kwargs for CheXagent vs gated** — default path would interleave writes.
- **silver_ensemble takes list inputs** — concatenates CheXagent + gated batches before calling `combine_labels`.

---

## Mistakes made today + lessons for next session

These cost ~4-6 hours of wall clock today. Future sessions should avoid.

### Mistake 1 — Skipped pre-flight reviews under time pressure
**What happened:** Today I redeployed the v6 orchestrator 5-6 times to fix bugs (CheXagent deps, dtype, image split, green matplotlib). I ran pre-flight Opus reviews on the initial v6 plan and the initial v6 code, but NOT between subsequent fixes. The matplotlib-missing bug in GREEN and the HO-filter row_idx misalignment in LOO slipped through because no pre-flight checked each intermediate deploy.
**Cost:** ~3 hours of GREEN crash-loop debugging, ~$10 of wasted H100 time, one near-miss on shipping an unpublishable LOO checkpoint.
**Lesson:** Pre-flight review is **mandatory before every `modal deploy`**, not just at session start. The rule exists precisely because debugging pressure is when we cut corners. Budget $0.50 of Opus tokens per deploy; that's a 100× safety multiplier vs GPU waste.
**Mechanical fix:** Add a `scripts/preflight.sh` wrapper that spawns an Opus self-check agent on the diff since last commit before calling `modal deploy`. Skip with `--no-preflight` for trivial changes only.

### Mistake 2 — Misread `call.get(timeout=2)` as "still running" when it could be "silently retrying"
**What happened:** GREEN-chex-v2 and GREEN-gated-v2 appeared "still running" for 2-3 hours in my status checks. In reality, they were crash-looping on an ImportError (matplotlib missing), and Modal's retry policy kept the container alive until manual intervention. My `call.get(timeout=2)` returned `TimeoutError` — which I read as "still computing" instead of "possibly retrying silently."
**Cost:** 2-3 hours of apparent progress that was actually no progress.
**Lesson:** `TimeoutError` from `call.get()` means "not yet complete." It does **not** imply "making progress." Always check:
- Modal volume for output file growth (real work produces files)
- Modal logs for actual stdout (retries show tracebacks)
- Container logs for CUDA / GPU activity
**Mechanical fix:** Write a `scripts/verify_job_progress.py` helper that checks volume-file-mtime growth, log-line count growth, AND call status. Report all three. Don't trust any single signal.

### Mistake 3 — Used `[X]` placeholders in figure prompts that rendered as literal text
**What happened:** Several figure prompts (F13, F14, F15) contained bracketed placeholders like `[X1]`, `[pending]`, `[0.XX]` for values I didn't have at write-time. When pasted into image-gen tools, the tools rendered these brackets literally as figure captions (e.g., a heatmap cell saying `[X1] pp`).
**Cost:** User confusion + one round of re-prompting to replace with concrete fake numbers.
**Lesson:** Image-generation tools do not fill in placeholders. Always write prompts with **concrete numbers** — even fake-plausible ones. Real numbers can be swapped in later when they land.
**Mechanical fix:** Before writing any figure prompt, decide: "do I have the real number? If no, invent a plausible one and tag `FAKE` in a comment above the prompt." Never ship a prompt with bracket placeholders to an image-gen tool.

### Mistake 4 — Assumed GREEN's output format without inspecting actual output
**What happened:** I wrote GREEN's `_parse_response` regex for a numbered-list format (`1. false finding: N`) that I assumed from reading the GREEN paper abstract. GREEN actually emits a lettered format under section headers (`[Clinically Significant Errors]: (a) False report: N.`). Every parse returned `score=0.5` → UNCERTAIN. 368 of 368 CheXagent claims came back UNCERTAIN before I noticed.
**Cost:** 2 hours of GREEN wall-clock producing garbage labels. Fixable via offline re-parse of saved raw_responses, so no compute wasted — just delay.
**Lesson:** Before writing a parser for any external model's output, **run the model on ONE example and inspect the raw output format**. Don't infer format from paper text. Models rarely produce exactly what their authors describe in abstracts.
**Mechanical fix:** For every `_parse_response`-style function, include a unit test that hard-codes a real output sample (from HF model card example, or paper's supplementary, or a quick local run). Regression-prevention against format assumptions.

### Mistake 5 — PadChest-GR site name inconsistency (`padchest-gr` vs `padchest_gr`)
**What happened:** `v5/data/padchest_gr.py` sets `site: str = "padchest-gr"` (hyphen). `train_v6_loo` expected `held_out_site="padchest_gr"` (underscore) via its file path template `train_no_{held_out_site}.jsonl`. Mismatch caused a FileNotFoundError on one LOO run that I had to cancel and respawn with corrected naming.
**Cost:** ~30 minutes + one wasted Modal container boot.
**Lesson:** Site/tag/slug strings must be **normalized in one place** and imported from there. A `v5/data/site_names.py` module with canonical constants (`SITES = ["openi", "chestx_det10", "padchest_gr"]`) would prevent this class entirely.
**Mechanical fix:** Add `v5/data/site_constants.py`. Import from everywhere that references a site name. Remove all hardcoded site strings elsewhere.

### Mistake 6 — Did not add matplotlib to the MAIN Modal image initially
**What happened:** I correctly added matplotlib to `image_chexagent` when CheXagent's trust_remote_code required it. But GREEN-RadLlama2-7b's `tokenization_chexagent.py` (shared code file) ALSO requires matplotlib, and GREEN runs on `image` (not `image_chexagent`). I didn't realize the dependency overlap until GREEN crashed.
**Cost:** 2 hours of crash-loop debugging.
**Lesson:** When one trust_remote_code model requires deps, **check whether other trust_remote_code models in other images share the same tokenizer file**. Scan for `auto_map` across all HF model IDs in use.
**Mechanical fix:** Maintain a single `_TRUST_REMOTE_CODE_DEPS` list in `v6_orchestrator.py` that's added to BOTH images whenever a new dep is discovered.

### Mistake 7 — Stale `.pyc` files shipped to Modal via `.add_local_dir`
**What happened:** An older session left `.pyc` bytecode cached with old `translate_reports` symbol in `v5/data/__pycache__/padchest_gr.cpython-313.pyc`. Modal's `.add_local_dir(VERIFACT_ROOT, copy=True)` happily shipped the stale bytecode alongside the fresh `.py` file. Python's import machinery picked up the `.pyc` over the `.py`, running stale code.
**Cost:** 30 minutes of "why isn't my fix applying?" debugging.
**Lesson:** Clean `.pyc` caches before any `modal deploy`:
```bash
find VERIFACT_ROOT -type d -name __pycache__ -exec rm -rf {} +
find VERIFACT_ROOT -name "*.pyc" -delete
```
**Mechanical fix:** Add a pre-deploy hook or include this cleanup in `scripts/preflight.sh` above.

### Mistake 8 — No `volume.reload()` + immediate downstream spawn after volume write
**What happened:** `assemble_v6_loo_splits` wrote 6 files to the Modal volume and returned OK. I immediately spawned 3 `train_v6_loo` calls. Two of three succeeded (LOO-openi, LOO-padchest-gr), one (LOO-chestx) FileNotFoundError'd because its container's volume snapshot predated the write's commit.
**Cost:** 1 hour re-spawn cycle.
**Lesson:** Modal volume writes commit on function exit. Downstream functions starting immediately may see stale state. Either (a) `.get()` the writer call before spawning readers (blocks until commit), or (b) call `volume.reload()` at the top of reader functions.
**Mechanical fix:** Added `volume.reload()` to `train_v6_loo`, `silver_green`, `silver_ensemble` in commit `c056609`. Apply the same pattern to future orchestrator functions.

### Mistake 9 — Spawned two parallel `silver_green` calls without distinct output paths
**What happened:** I planned to run silver_green twice in parallel (one for CheXagent claims, one for gated claims) but both calls used the default `out_jsonl="/data/v6_silver/green_labels.jsonl"`. The two containers would have interleaved writes to the same file OR one would have hit `_skip_if_exists` and produced no output.
**Cost:** Zero — pre-flight review caught it before spawn.
**Lesson:** Any function designed to be called multiple times in parallel MUST take distinct input / output paths as required kwargs, not defaults. Default paths are for single-invocation convenience, not parallelism.
**Mechanical fix:** Add a runtime assertion inside `silver_green` that fails fast if the output path exists AND is non-empty at start time (in tandem with `_skip_if_exists`, which warns but doesn't fail loudly).

### Mistake 10 — HO filter row_idx silently misaligned across v5 → v6 LOO
**What happened:** v5 HO filter weights are positionally indexed to the v5 2-site train file (~25k rows, OpenI + ChestX-Det10). v6 LOO train files are different (2-site combinations including PadChest-GR, different orderings, different row counts). My `train_v6_loo` initially passed the v5 weights path through. The training would proceed "successfully" but with downweights assigned to arbitrary unrelated rows — an unpublishable experimental artifact. Same class of bug as the ClaimGuard DPO inversion from an earlier session.
**Cost:** Zero — pre-flight review caught it before spawn. Would have cost $30-50 of unpublishable GPU time.
**Lesson:** Row-indexed auxiliary signals (HO weights, annotations, captions) are **file-specific**. Transferring them across files only works if the files are identical. Otherwise either (a) recompute on the new file, or (b) disable the signal and document in limitations.
**Mechanical fix:** Fail-fast assertion in `train_v5`: when `ho_filter_weights_path` is passed, verify `len(weights) == len(filtered_rows)` at dataset-init time and raise if mismatched.

---

## Known issues / gotchas (carryover from previous handoffs)

- **Modal H100 capacity** is variable. Jobs sometimes queue for 30+ min before starting. Monitor via `modal app list`.
- **`modal volume rm` requires `-r`** for directories.
- **Anthropic rate limits on Opus 4.7** are 50 rpm per organization. Bulk labeling (>1000 claims) will hit 429s; retry backoff handles it but slows throughput.
- **Anthropic + Claude Code subscription are separate billing** — API calls bill to `console.anthropic.com`, not Claude Code plan.
- **`modal app logs -f` replays full history before tailing live.** Tracebacks from earlier runs will appear in the stream before current-run events. Always check task IDs or request IDs to distinguish.
- **`modal run --detach` requires local client staying alive during image-build handshake.** Use `modal deploy` + `.spawn()` instead; `.spawn()` returns in <2s and is immune to local disconnect.
- **HF tokens** must be the full 37-char `hf_...` string. Truncation to 18 chars produced silent 403s on gated repos (happened earlier in this session).
- **HF gated repo acceptance is per-account, not per-token.** Accepting a license and regenerating a token does NOT require re-accepting licenses.
- **MAIRA-2 needs `transformers==4.51.3`** (pin), not 4.46 or 4.52.
- **CheXagent-2-3b hard-pins `transformers==4.40.0`** in its remote_code. Reconciled by two Modal images.
- **GREEN-RadLlama2-7b's `tokenization_chexagent.py`** (which it shares with CheXagent) requires `matplotlib` at import time. No other CheXagent deps (tensorflow, albumentations, etc.) are needed for GREEN.

---

## Environment

| Setting | Value |
|---|---|
| Python (local) | 3.13 |
| Python (Modal containers) | 3.10 |
| Modal client | 1.4.1 |
| transformers (main image) | 4.51.3 |
| transformers (CheXagent image) | 4.40.0 |
| torch | 2.4.0 |
| GPU | Modal H100 only (H100:80GB for hidden-state probe) |
| Modal app | `claimguard-v6-orchestrator` (deployed) |
| Modal volume | `claimguard-v5-data` |
| Modal secrets | `anthropic`, `huggingface` |
| Key pins | `albumentations==1.4.3`, `albucore==0.0.13`, `open_clip_torch==2.26.1`, `torchxrayvision==1.3.5` |
| Budget state | ~$100 Modal + ~$20 Anthropic spent today; ~$500 Modal remaining |

---

## Second brain (Obsidian vault)

- `~/Vault/CLAUDE.md` — vault operating rules
- `~/Vault/index.md` — master content index
- `~/Vault/log.md` — chronological session log (previous session entry exists for 2026-04-19; 2026-04-20 entry not yet appended — do this during /handoff if run)
- `~/Vault/wiki/projects/VeriFact.md` — project landing page (currently reflects Tier 3 state; needs v6.0-3site headline update)
- `~/Vault/wiki/synthesis/claimguard-v6-neurips-sprint-2026-04-19.md` — v6.0 sprint start synthesis
- `~/Vault/wiki/synthesis/claimguard-v5-tier3-complete-2026-04-19.md` — v5.0 Tier 3 writeup

---

## Reference materials (READ FULLY if unfamiliar)

### User global rules
- `~/.claude/CLAUDE.md` — MANDATORY reading. Model safety rule, pre-flight review rule, doc-sync rule, git rules.

### Architecture + plan (this cycle)
- `ARCHITECTURE_V6_0_NEURIPS_MAIN.md` — authoritative v6.0 spec
- `PLAN_V6_17DAY_NEURIPS.md` — 17-day execution plan with gates + budget
- `mlhc_build/poster_figure_script.md` — figure narrative + prompts
- `mlhc_build/datasheet.md` — Gebru 2021 datasheet

### Literature (must cite in paper)
- **Gururangan et al. 2018 (NAACL)** — hypothesis-only baseline for NLI
- **Poliak et al. 2018 (\*SEM)** — partial-input NLI
- **Mohri & Hashimoto 2024 (ICML)** — conformal factuality
- **Jin & Candès 2023 (JMLR)** — conformal selection with p-values
- **Tibshirani et al. 2019 (NeurIPS)** — weighted conformal under covariate shift
- **Zhang et al. 2024 (ML4H)** — RadFlag
- **Hardy et al. 2025 (AAAI Bridge)** — ReXTrust
- **Bannur et al. 2024 (arXiv:2406.04449)** — MAIRA-2 + RadFact
- **Ostmeier et al. 2024 (EMNLP Findings)** — GREEN
- **Tang et al. 2026 (arXiv:2604.03376)** — VERT
- **Nguyen et al. 2025 (IJCAI, arXiv:2505.00744)** — HEAL-MedVQA — concurrent work, MUST differentiate
- **Feijoo et al. 2024 (arXiv:2411.05085)** — PadChest-GR

---

## How to resume (exact sequence)

1. **Read `~/.claude/CLAUDE.md` fully.**
2. **Read the vault:** `~/Vault/CLAUDE.md` + `~/Vault/index.md` + last 5 entries in `~/Vault/log.md` + `~/Vault/wiki/projects/VeriFact.md`.
3. **Read this handoff fully — every section including mistakes/lessons.**
4. **Read `PLAN_V6_17DAY_NEURIPS.md` and `ARCHITECTURE_V6_0_NEURIPS_MAIN.md` fully.**
5. **Read `v5_final_results/TIER3_RESULTS_2026-04-19.md`** for v5.0 Tier 3 baseline.
6. **Clean pycache:**
   ```bash
   find /Users/aayanalwani/VeriFact/verifact/v5 -type d -name __pycache__ -exec rm -rf {} +
   find /Users/aayanalwani/VeriFact/verifact/v5 -name "*.pyc" -delete
   ```
7. **Check Modal volume state** matches the "Expected state on volume" section above:
   ```bash
   modal volume ls claimguard-v5-data /v6_silver/
   modal volume ls claimguard-v5-data /v6_rrg/
   ```
8. **Re-deploy the orchestrator** (to pick up any uncommitted changes):
   ```bash
   cd /Users/aayanalwani/VeriFact/verifact
   modal deploy v5/modal/v6_orchestrator.py
   ```
9. **Spawn pre-flight Opus review** on the current orchestrator + any code diff since `27d3c84` before any Modal respawn.
10. **Begin Tier 1 tasks** from Pending Tasks section above.

**Do NOT begin Tier 1 before steps 1–9.**

---

## Self-check agents — MANDATORY

Per user CLAUDE.md standing rule, spawn Opus pre-flight / self-check agents:
- **Before any `modal deploy`** — especially if code has changed since last deploy.
- **Before any `modal function spawn`** that costs > $1 of GPU time or > $5 of API calls.
- **After any non-trivial code change** — especially in `v5/eval/*.py`, `v5/train.py`, `v5/losses.py`, `v5/ho_filter.py`.
- **Before any paper edit that changes a numerical claim.**

Today's pre-flight saved an estimated $30-50 of unpublishable GPU time on the LOO HO-weights misalignment bug. Do not skip.

---

## Open questions for user

1. **Compute budget going forward.** ~$500 Modal remaining. Tier 1 (~$5) + Tier 2 (~$70) fit comfortably; Tier 3 (polish, no compute) is ~$0. Ample slack.
2. **Should Claude baseline be re-run?** Requires additional ~$20 Anthropic credits + rate-limit tolerance. Currently marked pending; low-priority since MedGemma/BiomedCLIP numbers already fill the baseline table.
3. **Poster format.** 48" × 36" landscape? Different size per venue requirement?
4. **Figure generation path.** Image-gen tool of choice (Nano Banana / FLUX / DALL-E / Imagen) for illustrations? matplotlib directly for data figures?
5. **Vault update cadence.** Log entry for 2026-04-20 not yet written. Add during next `/handoff` invocation or manually.

---

## Change log

| Date | Version | Author | Change |
|---|---|---|---|
| 2026-04-19 | v6.0 initial | Aayan Alwani | 17-day NeurIPS E&D sprint plan + architecture spec |
| 2026-04-19 | v6.0 pre-flight-patched | Aayan Alwani | Folded Opus pre-flight reviewer fixes into plan |
| 2026-04-19 | v6.0 post-verification | Aayan Alwani | External verifications (ReXTrust, HEAL-MedVQA, MedVersa, PadChest-GR) |
| 2026-04-20 | v6.0 compute-90% | Aayan Alwani | **This handoff.** Day 6-8 complete; silver labeling mostly complete; 5 bug classes caught & fixed; paper + datasheet + poster script drafted |
