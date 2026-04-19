# Handoff: ClaimGuard-CXR v6.0 — NeurIPS 2026 Sprint Code Complete

**Date:** 2026-04-19 ~late
**Status:** plan, architecture, and code committed locally (`2f2fe5b`); 57/57 tests passing; two Opus pre-flight reviews cleared; ready for user to approve Day-0 remote actions.
**Supersedes:** `HANDOFF_CLAIMGUARD_V5_TIER3_COMPLETE_2026-04-19.md`
**Target:** NeurIPS 2026 Evaluations & Datasets track, deadline 2026-05-06 AoE (17 days)

---

## Current state (1-paragraph summary)

v5.0 Tier 3 results are intact (see prior handoff). This session pivoted the project to v6.0: a diagnostic-first reframe for NeurIPS 2026 E&D. All planning docs are finalized (`PLAN_V6_17DAY_NEURIPS.md`, `ARCHITECTURE_V6_0_NEURIPS_MAIN.md`). All 14 new Python modules + 2 test files + Modal orchestrator + reproduction script are written (5,120 lines across 18 files) and committed as `2f2fe5b` on branch `v5-image-grounded`. Two rounds of Opus pre-flight review caught 12 plan-level and 8 code-level issues; all have been applied. Test suite is 57 passing, 2 skipped (env-dependent, intentional). **No Modal jobs spawned yet** — awaiting user approval + Day-0 registrations.

## What's running / in progress

**Nothing.** No Modal jobs active. The v6.0 orchestrator is written but not deployed. User needs to complete Day-0 registration actions before Day-1 code execution proceeds.

To verify nothing is running:
```bash
modal app list | grep claimguard
cat v5_final_results/pipeline_status.json  # last v5.0 Tier 3 status
```

## What you (the next session or the user) must do first

### Day-0 (today / ASAP) — user-gated, cannot be automated:

1. **Submit PadChest-GR RUA** at https://bimcv.cipf.es/bimcv-projects/padchest-gr/ — longest wall-time dependency (historical range 3–14 days). Read the RUA fine print on approval; update `ARCHITECTURE_V6_0_NEURIPS_MAIN.md` §10 release plan if redistribution is restricted.
2. **Submit CheXpert Plus form** at https://aimi.stanford.edu/datasets/chexpert-plus — fallback data source if PadChest-GR is delayed.
3. **Confirm Anthropic API credits loaded** (user reported done in last chat). Silver-labeling Days 3-4 require ~$170 of Opus + Haiku calls.
4. **Approve Modal spend**: the v6 reproduction script will queue ~$443 of H100 + API work. Standing Modal volume / app permissions are in place.

### Day-1 (after Day-0 registrations in flight):

1. Deploy the v6 orchestrator: `cd /Users/aayanalwani/VeriFact/verifact && modal deploy v5/modal/v6_orchestrator.py`
2. Begin pipeline: `bash v5/scripts/reproduce_v6.sh` (fully non-interactive; will wait on PadChest-GR download)

## Completed tasks this session

- [x] **4 parallel deep-research agents** ran (RRG models, silver-labeling, public datasets, NeurIPS bar), ~30 min each, informed every key decision in the plan.
- [x] **PLAN_V6_17DAY_NEURIPS.md** written — Day 0 through Day 17 schedule, 7 stop-the-line gates, 13-row risk table, $443 budget.
- [x] **ARCHITECTURE_V6_0_NEURIPS_MAIN.md** written — authoritative v6.0 spec replacing v5.0.
- [x] **CLAIMGUARD_PROPOSAL.md** header updated with v6.0 pointer (M5 doc-sync).
- [x] **Opus pre-flight #1 (plan)** — caught 5 blocking + 7 major issues; all folded in (RadFact cost, VERT addition for MIMIC leakage, ReXTrust reframing, PadChest-GR timing, parallel silver labeling).
- [x] **Day-0 external verifications** — confirmed no public ReXTrust code; confirmed HEAL-MedVQA uses LobA not ACT; confirmed MedVersa weights at `hyzhou/MedVersa`; confirmed PadChest-GR RUA process.
- [x] **14 new modules + 2 test files + 1 orchestrator + 1 reproduction script** written, totaling ~2,700 LOC: `rrg_generate.py`, `green_labeler.py`, `radfact_labeler.py`, `vert_labeler.py`, `silver_ensemble.py`, `padchest_gr.py`, `padchest_gr_validate.py`, `hidden_state_probe.py`, `radflag_replica.py`, `artifact_audit.py`, fixed `baselines.py`, `v6_orchestrator.py`, `reproduce_v6.sh`, `test_silver_ensemble.py`, `test_padchest_gr_validate.py`.
- [x] **Opus pre-flight #2 (code)** — caught 3 blockers + 5 should-fixes; all applied (V5TrainConfig field mismatch, artifact_audit row_idx key, missing assemble_v6_3site, GREEN per-claim missing_finding, silver ensemble UNCERTAIN-majority, RadFlag polarity, probe claim-token pooling, reproduce_v6.sh interactive prompt).
- [x] **Opus pre-flight #3 (verification)** — re-ran against post-fix code; verdict ready-to-launch, two minor hardening notes applied.
- [x] **Test suite**: 57 passed, 2 skipped (intentional, require env vars). 25 new tests for silver_ensemble + padchest_gr_validate.
- [x] **Local commit**: `2f2fe5b feat(v6.0): NeurIPS 2026 E&D sprint — plan + code for 17-day pipeline`, 18 files, 5,120 lines.
- [x] **Vault updated**: new synthesis page `claimguard-v6-neurips-sprint-2026-04-19.md`, index.md updated, log.md appended.

## Pending tasks (priority order for next session)

1. **User Day-0 actions** (above). All downstream work blocks on these.
2. **Assemble PadChest-GR site branch in `v5/data/groundbench.py`** — the orchestrator expects `/data/groundbench_v5/padchest_gr_records.jsonl` from `padchest_gr_assemble`, and `assemble_v6_3site` concatenates three site JSONLs. You may need a PadChest-GR claim-extraction pass in `groundbench.py` to convert raw records to per-claim rows with `gt_label` assignments. Reference existing OpenI + ChestX-Det10 branches.
3. **Update `mlhc_build/paper.tex`** — paper rewrite around the diagnostic-first reframe (PLAN Day 12). Existing draft has Tier 1 v5.0 numbers as headline; needs v6.0 content.
4. **Draft datasheet** (required for NeurIPS E&D submission) per Gebru et al. 2021 schema. `v5/data/README.md` has partial content; expand.
5. **Spawn Modal jobs per `reproduce_v6.sh`** — once user approves Day-0 work.
6. **Deploy v6 orchestrator**: `modal deploy v5/modal/v6_orchestrator.py`.

## Key files

| Purpose | Path |
|---|---|
| Authoritative architecture (v6.0) | `ARCHITECTURE_V6_0_NEURIPS_MAIN.md` — READ FULLY |
| 17-day execution plan | `PLAN_V6_17DAY_NEURIPS.md` — day-by-day, gates, budget |
| Historical proposal (superseded) | `CLAIMGUARD_PROPOSAL.md` — header points to v6.0 |
| Tier 3 result source-of-truth | `v5_final_results/TIER3_RESULTS_2026-04-19.md` |
| Modal orchestrator (v6) | `v5/modal/v6_orchestrator.py` — 13 entrypoints |
| Reproduction script | `v5/scripts/reproduce_v6.sh` — non-interactive E2E |
| Silver labelers | `v5/eval/{green,radfact,vert}_labeler.py` + `silver_ensemble.py` |
| RRG generators | `v5/eval/rrg_generate.py` (MAIRA-2, CheXagent-2, MedGemma-4B) |
| Fixed VLM verifier baselines | `v5/eval/baselines.py` |
| PadChest-GR loader | `v5/data/padchest_gr.py` |
| ReXTrust-inspired probe | `v5/eval/hidden_state_probe.py` |
| RadFlag replication | `v5/eval/radflag_replica.py` |
| Artifact audit | `v5/eval/artifact_audit.py` |
| PadChest-GR validation | `v5/eval/padchest_gr_validate.py` |
| New tests | `v5/tests/test_silver_ensemble.py`, `v5/tests/test_padchest_gr_validate.py` |

## Critical decisions (recap)

- **NeurIPS E&D target, not Main** — same deadline, charter matches diagnostic work, P(accept) 18-25% post-v6 vs 3-6% Main.
- **Diagnostic-first reframe** — Gururangan 2018 analogue; evidence-blindness is systematic.
- **3 silver graders** — GREEN + RadFact + VERT; VERT added per pre-flight B2 to decouple MIMIC-leakage.
- **3 RRG models for prevalence** — MAIRA-2 + CheXagent-2-3b + MedGemma-4B-IT.
- **PadChest-GR is the third site** — only public CXR dataset with per-sentence radiologist bboxes not from MIMIC.
- **ReXTrust reimplemented not replicated** — no public code; applied to MedVersa AND MAIRA-2 for architecture-independence.
- **Architecture frozen at v5.3** — 17 days does not permit model changes.
- **Drop v5.0-v5.2 from headline** — support-score degeneracy explained via sharpness histogram.

## Known issues / gotchas

- **MedVersa loader** requires installing `medomni` package from rajpurkarlab/MedVersa GitHub. If not installed, `MedVersaExtractor.__init__` raises RuntimeError with clear instructions. `v6_orchestrator.hidden_state_probe` handles this gracefully.
- **transformers pin**: v6_orchestrator image uses `transformers==4.51.3` specifically for MAIRA-2 compatibility. Other modules (RoBERTa-based v5.3 training) work with this version.
- **BIMCV RUA restrictions** on PadChest-GR redistribution may require Mode B release (scripts-only, not labels). `ARCHITECTURE_V6_0` §10 has both modes documented; final mode chosen on RUA read.
- **Modal secret names** assumed: `anthropic`, `huggingface`. Verify with `modal secret list` before deploy.
- **Claude Opus 4.7 model ID** used in labelers: `claude-opus-4-7`. If your workspace uses a dated variant, update `radfact_labeler.py`, `vert_labeler.py`, `radflag_replica.py`.

## Environment

- **Python:** 3.13 (local), 3.10 (Modal containers)
- **Modal client:** 1.4.1
- **Transformers (v6 orchestrator image):** 4.51.3 (MAIRA-2 compat); existing v5 HO-filter image: 4.44.2
- **GPU:** Modal H100 only; H100:80GB for hidden-state probe
- **Modal app (v6):** `claimguard-v6-orchestrator` (not yet deployed)
- **Modal app (v5):** `claimguard-v5-orchestrator` (deployed, idle)
- **Modal volume:** `claimguard-v5-data` (shared between v5 and v6)
- **Secrets required:** `anthropic`, `huggingface`
- **Budget state:** ~$280 of $900 spent in v5; ~$620 remaining; ~$443 projected for v6.

## Second Brain

Updated as of this handoff:
- `~/Vault/CLAUDE.md` — unchanged
- `~/Vault/index.md` — reflects v6.0 commit
- `~/Vault/log.md` — 2026-04-19 session entry appended
- `~/Vault/wiki/projects/VeriFact.md` — still at Tier 3 state; update to v6.0 transition next session
- `~/Vault/wiki/synthesis/claimguard-v6-neurips-sprint-2026-04-19.md` — **new**, full session writeup

## How to resume

1. **Read `~/.claude/CLAUDE.md` fully** (user global rules).
2. **Read the vault** — `~/Vault/CLAUDE.md`, `~/Vault/index.md`, last 5 entries of `~/Vault/log.md`, then `~/Vault/wiki/projects/VeriFact.md` and the v6 synthesis page above.
3. **Read this handoff fully.**
4. **Read `PLAN_V6_17DAY_NEURIPS.md` and `ARCHITECTURE_V6_0_NEURIPS_MAIN.md` fully.**
5. **Read `v5_final_results/TIER3_RESULTS_2026-04-19.md`** for v5.0 results baseline.
6. **Check Day-0 status**: has the user submitted PadChest-GR RUA? Are Anthropic credits loaded? If not, nothing else can proceed.
7. **Only then** begin the Day-2 RRG generation or any Modal spawn.

Do NOT run Modal spawns before steps 1–6 and before confirming Day-0 is complete.

## Self-check agents — still mandatory

Before any new Modal launch:
- Pre-flight Opus review per user CLAUDE.md standing rule (~$0.50 each)
- Post-code-change self-check before commit

Three pre-flight reviews ran this session. The fourth will be needed if further code changes land before launch.
