# ClaimGuard-CXR v6.0 — 17-Day Plan to NeurIPS 2026

**Author:** Aayan Alwani (Laughney Lab, Weill Cornell Medicine)
**Plan date:** 2026-04-19 (pre-flight-reviewed, 2026-04-19; launch-with-fixes verdict, fixes folded in)
**Submission deadline:** 2026-05-06 AoE (NeurIPS 2026 Evaluations & Datasets track — same deadline as main)
**Runway:** 17 days
**Budget remaining:** ~$620 Modal H100
**Companion:** `ARCHITECTURE_V6_0_NEURIPS_MAIN.md`
**Hard constraints:** public data only · no radiologist recruited · no PhysioNet credentialing

---

## Pre-flight review fixes (2026-04-19)

An Opus pre-flight reviewer returned `launch-with-fixes` with 5 blocking + 7 major issues. All fixes folded into the relevant sections below. Summary:

- **B1**: RadFact cost corrected from $50 → **$120** for 1,000 claims (13 LLM calls/report, not 2/claim). Use Claude Haiku for decomposition, Claude Opus for entailment only. §4.
- **B2**: Add **VERT** as third silver grader (Claude Opus backbone, MIMIC-free at inference) to decouple MIMIC-leakage between GREEN (MIMIC-trained) and MAIRA-2 (MIMIC-pretrained). Report per-grader prevalence separately. §3 Day 4, §4.
- **B3 (revised 2026-04-19 post-verification)**: **ReXTrust has no public code or weights** (verified Day 0 — not in rajpurkarlab GitHub, not on HF). True replication is not possible. Final plan: reimplement the hidden-state probe from the paper's description, apply to both **MedVersa** (weights at `hyzhou/MedVersa`, public, non-standard loader) **and MAIRA-2** — giving us an architecture-independence check alongside the primary MedVersa run. Paper row labeled "White-box hidden-state probe (inspired by ReXTrust; reimplemented per paper specification)." §3 Day 9.
- **B4**: Submit PadChest-GR RUA **today, Day 0, not Day 1**. Original PadChest took 3–14 days historically. Submit CheXpert Plus RUA in parallel as fallback. Hardened fallback: NIH-14 + Object-CXR + OpenI as non-RUA 3-site config. §3 Day 1, §5.
- **B5**: Days 3 and 4 collapsed into **one calendar day (Day 3–4 combined)** — GREEN on H100 and RadFact on Claude API run concurrently (no shared resource). Buys full day of buffer for PadChest delay. §3.
- **M1**: Add Day-1 smoke test for MAIRA-2 (10 images) before 500-image spawn. MAIRA-2 actual throughput 15–25 tok/s on H100 (not 45); budget 90 min per 500-image run. §3 Day 1–2.
- **M2 (revised 2026-04-19 post-verification)**: HEAL-MedVQA's mitigation is **LobA** (Localize-before-Answer), not ACT — segmentation training + attention reweighting at inference. Their TPT/VPT are **strictly VQA binary Yes/No** entity-swap tests, no generalization to verification claimed. **Our differentiation is cleaner than originally drafted**: different task (claim verification vs VQA), different metric construction (image-masking distributional gap vs entity-swap accuracy gap), different mitigation (consistency loss + HO filter vs segmentation + attention reweighting). Update Architecture §3.2 accordingly.
- **M3**: Cross-site LOO gets **mechanistic diagnosis** (support-score shift per site, HO-filter activation per site, text-only ceiling per site), not just numerical report. §3 Day 10.
- **M4**: Artifact-audit target changed from "≤ chance + 5pp" to "text-only acc minus majority-class acc ≤ 2pp". §3 Day 10.
- **M5**: Day 1 task added — update `CLAIMGUARD_PROPOSAL.md` header with v6.0 pointer per doc-sync rule. §3 Day 1.
- **M6**: Day 11 experiment added — HO-filter activation rate on real-RRG claims (paraphrastic hallucinations, not negation-flips). §3 Day 11.
- **M7**: Day 1 task added — read PadChest-GR RUA on approval; update §10 release plan (may need to release reproduction scripts, not derived labels). §3 Day 1 (post-approval) + Architecture §10.


---

## 0. Headline decisions

| # | Decision | Rationale |
|---|---|---|
| D1 | **Target NeurIPS 2026 Evaluations & Datasets** (not Main) | Same May 6 deadline; the paper is fundamentally a diagnostic/evaluation contribution; P(accept) 12–18% after fixes vs 3–6% on main with similar work |
| D2 | **Reframe as diagnostic-first paper** | "Evidence-Blindness in Medical Multimodal Verifiers" — analogous to Gururangan 2018's annotation-artifacts framing; establishes a community-facing finding, not yet another verifier |
| D3 | **Add PadChest-GR** (not CheXpert Plus, not ReXGradient-160K) | 46 GB, 48 h RUA approval, bilingual ES/EN, radiologist-placed bboxes per report sentence — only public dataset with claim-level radiologist grounding not derived from MIMIC. Highest value/effort ratio |
| D4 | **Run 3 real RRG models** (MAIRA-2, CheXagent-2-3b, MedGemma-4B-IT) | Top-3 public-weight CXR RRGs in 2025–2026 literature; together they cover grounded-report (MAIRA-2), MIT-licensed (CheXagent-2), and 2025-foundation (MedGemma) positions. Cost: ~$20 total |
| D5 | **Silver-label with GREEN + RadFact ensemble**, not Claude-alone | Two published LLM judges with measured radiologist agreement (GREEN τ=0.63; RadFact τ=0.32 on RaTE-Eval). Report Krippendorff α between them; route disagreements to UNCERTAIN. Defensible without radiologist recruitment |
| D6 | **Validate silver labels against PadChest-GR radiologist bboxes** | PadChest-GR has 7,037 radiologist-placed finding-sentence bboxes — free expert-validation. Measure silver-label agreement against these. Zero incremental cost, strongest reviewer-defensibility move |
| D7 | **Drop three training configs from the headline**, keep v5.3 as "the method" | v5.0–v5.2 have degenerate support-score distributions that break conformal FDR; reviewers read this as brittleness. Add one support-score-histogram figure; discuss in limitations, not headline |
| D8 | **Replicate ReXTrust + RadFlag** as must-beat baselines | Any NeurIPS-class submission in this space without these is incomplete per 2025 reference class |
| D9 | **Keep v5.3 model architecture frozen** | 17 days does not permit architectural changes; adding data + baselines + diagnostic breadth is the highest-leverage work |
| D10 | **Pre-flight every Modal launch** | Per user CLAUDE.md standing rule. ~8 launches × $0.50 = $4 of API tokens to prevent $40+ GPU misfires |
| D11 | **Defer credentialed-dataset extension** (MIMIC-CXR, MS-CXR, etc.) | Listed in future work section only; not pursued this cycle |

---

## 1. Why NeurIPS E&D, not NeurIPS Main

| Axis | NeurIPS Main | NeurIPS E&D | Workshop (GenAI4Health) |
|---|---|---|---|
| Deadline | May 6, 2026 AoE | **May 6, 2026 AoE (same)** | TBA, later |
| Accept rate (2025) | 24.5% | ~25% | ~55% |
| Page limit | 9 + refs + appendix | 9 + refs + appendix | 4–8 typical |
| Charter | Methodological novelty | **Evaluation methodology + datasets + benchmarks** | Applied, lighter novelty bar |
| Reviewer class | Core ML; hostile to clinical | Evaluation-aware; value diagnostic work | Friendly but lower prestige |
| Competing pool | ~21k submissions | ~2k submissions (less diluted) | Hundreds |
| P(accept) for ClaimGuard-v6.0 post-plan | ~15–20% | **~18–25%** | ~60–70% |

NeurIPS 2026 renamed D&B → "Evaluations & Datasets" for 2026 specifically to welcome evaluation-methodology papers. The evidence-blindness diagnostic lives exactly in that charter. No downside: same deadline, same page limit, same author identity preserved — we get a strictly better match without cost.

Fallback ladder if E&D rejects: arXiv release → GenAI4Health workshop → ML4H 2026 Proceedings (Sep 2026) → MICCAI 2027 (Feb 2027).

---

## 2. What changes from v5.0

### 2.1 Scientific framing (paper narrative)

**Before (v5.0):** "We propose ClaimGuard-CXR, an image-grounded verifier with an evidence-blindness mitigation."

**After (v6.0):** "Public medical multimodal verifiers — including our own prior version — are substantially *evidence-blind*: they achieve competitive accuracy without using the image. We introduce three counterfactual diagnostics (IMG, ESG, IPG) adapted from NLI partial-input baselines (Gururangan 2018; Poliak 2018), demonstrate the failure is systematic across seven public detectors on real RRG-model outputs from MAIRA-2, CheXagent-2, and MedGemma-4B, and propose a training-time mitigation (consistency loss + adversarial hypothesis-only filter) that closes the gap 30× in-distribution. We also demonstrate an honest limitation: the mitigation does not cleanly transfer across sites, suggesting evidence-blindness is a training-distribution phenomenon that requires further architectural work. We release ClaimGuard-GroundBench-v6, a benchmark of 30k+ claims across three sites (OpenI, ChestX-Det10, PadChest-GR) with multi-grader silver labels validated against PadChest-GR's radiologist bboxes."

### 2.2 Closest prior-art differentiator (MUST cite)

**HEAL-MedVQA (Nguyen et al., IJCAI 2025, arXiv:2505.00744)** proposed Textual Perturbation Tests (TPT) and Visual Perturbation Tests (VPT) for medical VQA. Superficially similar to our IMG/ESG/IPG but differs materially: (a) VQA, not claim verification; (b) accuracy-based, not distributional-gap based; (c) no training-time mitigation; (d) no conformal FDR coupling.

Positioning: "Concurrent with HEAL-MedVQA, we focus on the distinct claim-verification setting and contribute the first training-distribution mitigation paired with a conformal-guarantee evaluation."

### 2.3 Data roster

| Dataset | v5.0 plan | v6.0 plan (this doc) | Notes |
|---|---|---|---|
| OpenI | used (train + test) | **kept** | existing CheXbert annotations |
| ChestX-Det10 | used (train + test) | **kept** | existing bboxes |
| **PadChest-GR** | not used | **ADD** (third site) | 46 GB, bilingual, grounded-sentence bboxes — registration only |
| CheXpert Plus | v5.0 planned but not run | defer to stretch (day 8 decision gate) | 350 GB download; add only if timeline allows |
| ReXGradient-160K | n/a | **defer (future work)** | HF-gate approval unpredictable in 17 days |
| MIMIC-CXR, MS-CXR, BRAX, etc. | forbidden | **forbidden (unchanged)** | credentialed extension noted as future work |
| NIH ChestX-ray14 | v5.0 planned | not added (no reports) | weakly labeled; low marginal value given PadChest-GR adds reports |
| RSNA / SIIM / Object-CXR | v5.0 planned | not added | detection-only; ChestX-Det10 already covers this role |

### 2.4 Evaluation axes added in v6.0

| Axis | v5.0 | v6.0 |
|---|---|---|
| Synthetic claims (current GroundBench test) | yes | kept |
| **Real RRG-generated claims** (hallucinations from MAIRA-2, CheXagent-2, MedGemma-4B) | no | **added (Δ primary)** |
| **Silver-label agreement (GREEN ∩ RadFact)** | no | **added** |
| **Silver-label validation vs PadChest-GR radiologist bboxes** | no | **added** |
| External baselines (working) | 1 broken (BiomedCLIP) | **7 working**: CheXagent-2 (VLM verifier), MAIRA-2 (VLM verifier), MedGemma-4B (VLM verifier), BiomedCLIP (fixed), RadFlag-replication, ReXTrust-replication, ours v5.3 |
| **Cross-site leave-one-out (3 sites)** | only 2-site | **3-site LOO** |
| **Artifact audit** (text-only train ceiling post-HO-filter) | no | **added** |
| **Support-score sharpness analysis** (why v5.0–v5.2 collapse in conformal) | no | **added (short section)** |

### 2.5 Code changes required

New modules:
- `v5/eval/rrg_generate.py` — run MAIRA-2, CheXagent-2-3b, MedGemma-4B on a fixed eval set, emit JSONL of (image_id, generated_report)
- `v5/eval/claim_decompose.py` — already exists in `v5/data/claim_extractor.py`; reuse with RRG inputs
- `v5/eval/green_labeler.py` — load `StanfordAIMI/GREEN-RadLlama2-7b`, score (reference_report, generated_report) per claim
- `v5/eval/radfact_labeler.py` — RadFact protocol from Microsoft repo, Claude Opus 4.7 as backbone
- `v5/eval/silver_ensemble.py` — combine GREEN + RadFact, compute Krippendorff α per claim, label UNCERTAIN when they disagree
- `v5/eval/rextrust_replica.py` — white-box hidden-state probe, trained on our silver labels
- `v5/eval/radflag_replica.py` — temperature-sampling black-box consistency score
- `v5/eval/padchest_gr_validate.py` — silver vs PadChest-GR bbox-sentence agreement
- `v5/eval/artifact_audit.py` — text-only RoBERTa baseline, pre- and post- HO filter
- `v5/data/padchest_gr.py` — loader + split
- `v5/modal/v6_orchestrator.py` — new Modal app with entrypoints for each new eval

Fixed modules:
- `v5/eval/baselines.py` — fix CheXagent-2-3b loader, fix BiomedCLIP prompt template so zero-shot isn't 41% (below random)

Frozen (not changed):
- `v5/model.py`, `v5/train.py`, `v5/losses.py`, `v5/ho_filter.py`, `v5/conformal.py`, `v5/configs/*.yaml` — architecture and training are locked; v5.3-contrast remains "the method"

---

## 3. Day-by-day schedule

Today = 2026-04-19 (Day 0). Submit 2026-05-06 (Day 17). 17 calendar days; internal buffer on Day 16.

### Day 0 (Apr 19, today, urgent) — Registration push (moved from Day 1 per B4)

Submit all three dataset access requests **today**, not tomorrow. Historical PadChest RUA turnaround was 3–14 days, not 48 h; lost days compound.

- **NOW** — Submit PadChest-GR RUA at https://bimcv.cipf.es/bimcv-projects/padchest-gr/
- **NOW** — Submit CheXpert Plus form at https://aimi.stanford.edu/datasets/chexpert-plus (fallback if PadChest-GR slow)
- **NOW** — Submit NIH ChestX-ray14 metadata download (hardened fallback — no RUA needed, public domain)
- **NOW** — Verify ReXTrust code availability — check https://proceedings.mlr.press/v281/hardy25a.html repo link + Rajpurkar lab GitHub for ReXTrust repository. If available, queue MedVersa weights download. If not, plan reframe of Day-9 row (B3).

### Day 1 (Apr 20, Sun) — Loader fixes + literature dive + compliance

- 09:00 — Spawn second Opus pre-flight review on this (updated) plan + architecture doc (see §7)
- 10:00 — **Read HEAL-MedVQA (arXiv:2505.00744) fully** — note Attention Consistency Training (ACT) for M2 differentiation
- 11:00 — Update `CLAIMGUARD_PROPOSAL.md` header with pointer to ARCHITECTURE_V6_0_NEURIPS_MAIN.md (M5 doc-sync fix)
- 12:00 — Fix CheXagent-2-3b loader in `v5/eval/baselines.py` (`trust_remote_code=True`, Phi-based chat template)
- 14:00 — Fix MedGemma-4B-IT loader (`pipeline("image-text-to-text")`, SigLIP-encoder pinned)
- 15:00 — Fix MAIRA-2 loader (`transformers==4.51.3` hard pin, multi-image + prior-report input)
- 16:00 — Fix BiomedCLIP zero-shot prompt template (CLIP needs class-caption templates, not raw claim strings — explains 41% accuracy)
- 17:00 — **Smoke-test all 4 VLM loaders on 10 OpenI images** locally (MacBook CPU fallback or free Modal debug time) to catch packaging bugs before spawning 500-image jobs (M1)
- 18:00 — If PadChest-GR approved during the day: read RUA fine print carefully; update Architecture §10 release plan if redistribution is restricted (M7)
- End-of-day — all 4 baseline VLM loaders emit coherent output on ≥8/10 smoke-test claims

Cost: $0 Modal (all local or CPU Modal).

### Day 2 (Apr 21, Mon) — RRG model inference

- Pre-flight review on RRG generation script
- Spawn Modal job: MAIRA-2 on 500 OpenI test studies → 500 generated reports (budget 90 min on H100 per M1 correction; real throughput is 15–25 tok/s not 45)
- Spawn Modal job: CheXagent-2-3b on same 500 studies (~30 min)
- Spawn Modal job: MedGemma-4B-IT on same 500 studies (~30 min)
- All three parallel on separate H100 instances
- Cache outputs to `/data/rrg_outputs/v6/{maira2, chexagent2, medgemma4b}.jsonl`

Cost: ~$25 Modal (revised upward per M1: MAIRA-2 at 90 min × $4 = $6; others $3 each).

### Day 3–4 (Apr 22–23, Tue–Wed) — Silver labeling in parallel (B5 + B2 fix)

**Revised per B5**: GREEN (H100) and RadFact (Claude API) run concurrently — no shared resource. Saves one calendar day vs serial plan.

Concurrently:
- **Track A (GPU)**: Run existing LLM claim extractor on each RRG output → ~6,000 atomic claims. Spawn Modal job: GREEN-RadLlama2-7b on all 6,000 claims against OpenI reference reports. Output: `green_labels.jsonl`.
- **Track B (Anthropic API)**: Run RadFact on 1,000-claim subset. Use **Claude Haiku for decomposition** (cheap, ~1,500 calls), **Claude Opus for bidirectional entailment** (~4,000 calls at ~$0.025/call). Revised budget $120, not $50 (B1 fix).
- **Track C (Anthropic API, NEW per B2)**: Run VERT on same 1,000-claim subset using Claude Opus 4.7 backbone (MIMIC-free at inference, decouples silver labels from MAIRA-2's MIMIC pretraining). ~$50.
- **PadChest-GR download**: once approved, start download in background (~2 h for 46 GB)

End of Day 4:
- All 6,000 claims have GREEN labels
- 1,000-claim subset has GREEN + RadFact + VERT triple labels
- Compute pairwise Krippendorff α among (GREEN, RadFact, VERT); claims with any pair disagreement → UNCERTAIN
- Report per-grader prevalence separately in §8 (decouples MIMIC-correlation confound)

Cost: ~$8 (GREEN) + ~$120 (RadFact) + ~$50 (VERT) = **$178 total silver labeling** (was $58 in original plan).

### Day 5 (Apr 24, Thu) — PadChest-GR integration (gate conditional)

**Hard gate per B4**: if PadChest-GR not approved by EOD Apr 23, fall back to CheXpert Plus or NIH-14 + Object-CXR. Do NOT extend this gate day.

If PadChest-GR approved:
- Read RUA fine print; confirm whether derived labels may be redistributed (M7 — likely release reproduction scripts instead of raw labels)
- Implement `v5/data/padchest_gr.py` — loader, English-translation pass (reports are ES primarily; use Claude Haiku to translate findings sentences)
- Claim extraction on PadChest-GR reports → claims
- Claim-to-bbox matching (PadChest-GR already has per-sentence bboxes — identity match, not full IoU pipeline)
- Assemble into GroundBench-v6 as third site

If fallback to CheXpert Plus: same pipeline but using Stanford CheXpert Plus reports (no grounding; drop §4.2 silver-validation-vs-radiologist-bbox subcomponent).

If fallback to NIH-14 + Object-CXR: no reports on either; use 14-class labels + Object-CXR bboxes; reframe paper as US-only 3-site with no bilingual contribution. Silver-label validation drops to self-review only.

Cost: ~$15 (Claude Haiku for translation + claim extraction).

### Day 6 (Apr 25, Fri) — Retrain v5.3 on three sites

- Pre-flight review on the retraining config
- Spawn Modal job: train v5.3-contrast on {OpenI, ChestX-Det10, PadChest-GR} — call it **v6.0**
- 4 h Modal timeout; 3 sites × ~10k claims each ≈ 30k training claims; fits in the same 4 h window
- In parallel: fix-up the loaders/configs for 3-site cross-site experiments

Cost: ~$20 Modal.

### Day 7 (Apr 26, Sat) — Cross-site 3-way LOO + v6.0 diagnostic

- Three LOO training runs, in parallel:
  - Train on {OpenI, ChestX-Det10}, test on PadChest-GR
  - Train on {OpenI, PadChest-GR}, test on ChestX-Det10
  - Train on {ChestX-Det10, PadChest-GR}, test on OpenI
- Each ~2 h on H100 with reduced epochs (this is LOO, not full convergence)
- Evidence-blindness diagnostic on v6.0 and each LOO variant

Cost: ~$30 Modal.

### Day 8 (Apr 27, Sun) — Baseline evaluation suite

- Pre-flight review on baseline harness
- Evaluate as CLAIM VERIFIERS (not RRG) with zero-shot prompt on 500-claim test: CheXagent-2, MAIRA-2, MedGemma-4B, BiomedCLIP-fixed
- Evaluate v5.3 (old) and v6.0 (new) on same
- Run evidence-blindness diagnostic (IMG, ESG, IPG) on all 7 baselines + ours

Cost: ~$35 Modal.

### Day 9 (Apr 28, Mon) — ReXTrust-inspired probe + RadFlag replication (B3 revised post-verification)

**Day-0 verification result**: ReXTrust has no public code or weights. True replication is impossible. Final plan:

- **Reimplement from paper**: `v5/eval/hidden_state_probe.py` — logistic/MLP probe on last-layer hidden states at the verdict token, trained on our silver labels. Follow Hardy et al. 2025 methodology as stated in the paper (arXiv:2412.15264). Honest row label: "White-box hidden-state probe (inspired by ReXTrust; reimplemented)."
- **Apply to MedVersa** (weights public at `hyzhou/MedVersa`): loaded via `registry.get_model_class('medomni')` from their GitHub repo. ~$20 Modal (includes the non-standard-loader setup time).
- **Apply to MAIRA-2**: same probe recipe. Gives us architecture-independence evidence. ~$15.
- **RadFlag replication on MAIRA-2**: `v5/eval/radflag_replica.py` — 10 temperature-sampled generations per claim, consistency score. ~$25.

All three evaluated on the 500-claim test set + evidence-blindness diagnostic (the "prevalence" part of the paper).

Cost: ~$60 Modal.

### Day 10 (Apr 29, Tue) — Artifact audit + support-score sharpness + cross-site mechanism (M3, M4)

- **Artifact audit (M4 fix — tightened target)**: `v5/eval/artifact_audit.py`: train RoBERTa-base text-only (no image, no evidence) on train; report test accuracy. Metric: **text-only acc minus majority-class-prior acc** (not vs chance — base-rate is 70%+ on OpenI CheXbert). Target post-HO-filter: Δ ≤ 2pp. If Δ > 5pp, HO filter is not removing shortcuts effectively and the paper's mitigation story weakens.
- **Support-score sharpness**: plot v5.0, v5.1, v5.2, v5.3, v6.0 support-score distributions on test with KS-distance to uniform reported. Show v5.3 (and v6.0) are the only configs with non-degenerate separation → explains why conformal FDR only fires there. Avoid post-hoc cherry-picking framing.
- **Cross-site mechanism (M3, new)**: for each LOO training configuration, measure:
  - support-score distribution shift (KS statistic) between train-sites and held-out-site
  - HO-filter activation rate (% of held-out claims scored as text-solvable)
  - text-only ceiling per site
  Output: a mechanistic diagnosis figure → "cross-site IMG collapse is caused by X" → forward-looking fix (Y) cited as future work. Converts "we fail to generalize" from publication-poison to diagnostic contribution.

Cost: ~$10 Modal (text-only training + re-inference on LOO test sets).

### Day 11 (Apr 30, Wed) — Silver-label validation + HO-filter-on-real-RRG (M6)

**Silver vs radiologist**:
- For PadChest-GR's 7,037 positive finding sentences: silver label "SUPPORTED"; for sampled absent-findings: silver label "CONTRADICTED"
- Compare silver pipeline's output (GREEN + RadFact + VERT ensemble) vs PadChest-GR's radiologist-placed ground truth
- Report agreement: Cohen κ, precision, recall, per-finding-type breakdown
- Target: κ ≥ 0.5; if below 0.4, escalate per Plan §6 Day-11 gate (reframe required)

**HO-filter on real-RRG claims (M6 fix, new)**:
- Run HO-filter scorer (`v5/ho_filter.py`) on the 6,000 real-RRG claims generated Day 2
- Report HO-filter activation rate (% scored as text-solvable)
- Compare to silver-labeled-CONTRADICTED-without-image subset: are text-solvable claims enriched for hallucinations?
- Target: ≥40% of real hallucinations flagged as HO-solvable. If <10%, the HO filter doesn't transfer to paraphrastic hallucinations — paper's §4 → §7 link breaks and needs reframing.

Cost: ~$15 Modal (validation + HO scoring on 6k claims).

### Day 12 (May 1, Thu) — Paper reframe + writing

- Rewrite abstract + intro around diagnostic-first framing
- Section 1: Motivation — why evidence-blindness in medical multimodal matters (Gururangan analogy + Mohri & Hashimoto 2024 conformal factuality)
- Section 2: Diagnostic framework (IMG, ESG, IPG) — formal definitions + control experiments + thresholds
- Section 3: Prevalence — Table 1 showing all 7 detectors are evidence-blind on real RRG-generated claims
- Section 4: Mitigation — training-time consistency + HO filter; results in-distribution
- Section 5: Limitation — cross-site transfer analysis
- Section 6: Conformal FDR + support-score sharpness
- Section 7: Real-hallucination precision/recall on the 6,000-claim RRG output set
- Section 8: Silver-label validation against PadChest-GR
- Section 9: Related work (HEAL-MedVQA, ReXTrust, RadFlag, FactCheXcker, MedVH, GREEN, RadFact, Mohri & Hashimoto, Gururangan, Poliak, Jin & Candès)
- Section 10: Limitations + future work (credentialed-data extension explicitly deferred)

Cost: $0.

### Day 13 (May 2, Fri) — Writing cont. + tables + figures

- Finalize Table 1 (baseline landscape), Table 2 (IMG/ESG/IPG sweep), Table 3 (LOO cross-site), Table 4 (real-hallucination P/R), Table 5 (silver-label vs PadChest-GR agreement), Table 6 (ablations — already have this from Tier 3)
- Figures: support-score histogram (key figure); conformal FDR curves; architecture diagram
- All figures reproducible from `v5_final_results/*.json` + `v6_final_results/*.json`

Cost: $0.

### Day 14 (May 3, Sat) — Second-pass pre-flight review on paper

- Spawn Opus citation-and-factual-accuracy reviewer on the paper against all result JSONs
- Spawn Opus integrity reviewer: every number cited in the paper must map to a JSON file + a run ID
- Fix any flagged issues

Cost: ~$2 API tokens (reviewer agents are CPU-only).

### Day 15 (May 4, Sun) — Final pass + reproducibility

- Ensure `v5/scripts/reproduce_v6.sh` runs every experiment end-to-end from a clean Modal volume
- Datasheet for the benchmark (required for NeurIPS E&D)
- Ethics statement: public datasets only, no radiologist recruited, silver labels validated against PadChest-GR radiologist ground truth, no MIMIC-CXR or PhysioNet-credentialed data used
- Author statement: single author (Aayan Alwani); Laughney Lab affiliation but no lab co-authors since this is independent methodological work
- MIT-license the code; CC-BY-NC the benchmark metadata (inherits from PadChest-GR RUA; derived labels only, no raw images redistributed)

Cost: $0.

### Day 16 (May 5, Mon) — Buffer day

- Absorb any day-1–15 overruns
- If bored: run v6.0 with 3 seeds for main-table error bars (would spend ~$30 more)
- Last-chance: try CheXpert Plus subset (Stanford AIMI "small" subset of 5k images) if PadChest-GR download/integration finished early

Cost: ≤ $50 Modal (buffer).

### Day 17 (May 6, Tue) — Submit

- Submit to NeurIPS E&D track; OpenReview double-blind per their policy
- arXiv post-embargo or anonymous arXiv per NeurIPS rules
- Git tag `v6.0-neurips-submission`; commit includes code + ARCHITECTURE_V6_0_NEURIPS_MAIN.md + this plan + reproducibility checklist + datasheet
- **No pushes to GitHub before explicit user approval per standing rule**

Cost: $0.

---

## 4. Budget (revised post-pre-flight)

| Phase | Original | Revised | Notes |
|---|---:|---:|---|
| RRG inference (3 models × 500 images) | $20 | **$25** | M1: MAIRA-2 slower than assumed (15–25 tok/s), 90 min budgeted |
| GREEN silver labels (6,000 claims) | $8 | $8 | |
| RadFact silver labels (1,000 claims, Haiku decompose + Opus entailment) | $50 | **$120** | B1: real call count is ~13/report not 2/claim |
| **VERT silver labels (NEW — 1,000 claims, Claude Opus)** | — | **$50** | B2: non-MIMIC third grader to decouple leakage |
| PadChest-GR claim extraction + translation (Haiku) | $15 | $15 | |
| Retrain v6.0-3site on {OpenI, ChestX-Det10, PadChest-GR} | $20 | $20 | |
| 3-way LOO cross-site (3 runs) | $30 | $30 | |
| Baseline evaluation (7 detectors × 4 conditions × 500 claims) | $35 | $35 | |
| ReXTrust + RadFlag replications (incl. MedVersa per B3) | $50 | **$55** | B3: +$5 for proper MedVersa replication |
| Artifact audit + support-score sharpness + cross-site mechanism (M3, M4) | $5 | **$10** | M3: added cross-site mechanism experiments |
| Silver-label validation vs PadChest-GR + HO-filter-on-real-RRG (M6) | $10 | **$15** | M6: added HO-filter real-RRG activation-rate check |
| Pre-flight reviewer API tokens (~10 launches) | $5 | **$8** | added 2 extra pre-flights (this one + second pass) |
| Second-pass citation reviewer | $2 | $2 | |
| Buffer (Day 16 latitude) | $50 | $50 | |
| **Projected total** | **$300** | **$443** | |
| **Budget available** | — | **$620** | |
| **Slack** | — | **$177 (29% margin)** | |

Slack reduced from 53% → 29% after B1+B2 corrections. Still comfortable. If buffer runs thin: drop (a) 3-seed error bars on v6.0 (save $30); (b) the separate VERT third-grader and keep only GREEN+RadFact pair (save $50 but lose B2 leakage-decoupling).

---

## 5. Risks and mitigations

| Risk | Prob | Impact | Mitigation |
|---|---|---|---|
| PadChest-GR RUA approval >48 h | Medium | Critical (breaks plan from Day 5) | Parallel submit CheXpert Plus on Day 1 as fallback. If PadChest-GR delayed past Day 4, drop to 2-site LOO on OpenI + ChestX-Det10 + CheXpert Plus and reframe as US-only cross-site |
| MAIRA-2 integration fails (transformers version pin) | Low | High (one of three RRGs) | Research agent already identified exact pin `transformers==4.51.3`. If still breaks, substitute LLaVA-Rad (Microsoft) or RadVLM (2025) |
| GREEN license issues for redistribution | Low | Low | We don't redistribute GREEN; we only use it as a labeler. Our released labels are derived products; cite GREEN per Llama 2 Community License |
| RadFact API cost overrun via Claude Opus | Low | Medium | Cap at 1,000 claims hard limit; use Claude Haiku fallback if budget tight ($5/1000 claims instead of $50/1000) |
| v6.0 (3-site retrain) underperforms v5.3 | Medium | Medium | Report honestly. Still valuable as a multi-site version. Use v5.3 as primary result in paper; v6.0 as cross-site robustness contribution |
| LOO cross-site still collapses (IMG ≈ 0pp across all LOO directions) | Medium | Medium | Frame as the paper's central honest finding. This is the "evidence-blindness is training-distribution-specific" contribution. Not a bug, a result |
| Anthropic API credits run out mid-RadFact | Low | Medium | User already alerted to add $10–20 on Day 1. Budget includes this |
| Pre-flight reviewer flags DO-NOT-LAUNCH on something critical | Medium | Low-Medium | Budgeted for. Fix, re-review, launch. User CLAUDE.md rule: never override a DO-NOT-LAUNCH verdict |
| Silver labels disagree badly with PadChest-GR radiologist ground truth (κ < 0.4) | Low | High | If this happens, we've discovered the silver pipeline is unreliable — this itself is a publishable negative result for the benchmark contribution. Write up honestly |
| Laterality residual (IPG ≈ 0) stays across all configs | High (known) | Low | Already in v5.0 limitations section. Honest contribution: "mitigation fixes distributional evidence-blindness but not spatial reasoning" |
| Paper too long for 9 pages | Medium | Low | Appendix is unlimited in E&D. Move anything non-critical to appendix |
| CheXagent-2 and MedGemma-4B both refuse to emit Findings-format reports for OpenI images | Low | Medium | Few-shot prompt them with 3 OpenI reports as exemplars. If still failing, fall back to MAIRA-2 + 2 variants of decoding settings |
| NeurIPS 2026 E&D reviewer pool is hostile to medical papers | Low | High | Fallback ladder is explicit: arXiv + GenAI4Health + ML4H 2026 + MICCAI 2027 |

---

## 6. Gates (stop-the-line conditions)

Any of these firing → re-plan, don't push through:

1. **Day 2 gate**: if any RRG model fails to produce coherent reports on OpenI images after debugging, substitute per Risk #2 above. If two or more fail, re-plan with only MAIRA-2 as the RRG source and acknowledge reduced coverage.
2. **Day 4 gate**: if GREEN vs RadFact Krippendorff α < 0.3 (lower bound of acceptable inter-grader agreement), silver-labeling is unreliable; route more claims to RadFact or add VERT as third grader.
3. **Day 5 gate**: if PadChest-GR not approved by EOD Apr 24, commit to CheXpert Plus as data-expansion plan.
4. **Day 7 gate**: if v6.0 accuracy drops >5 pp vs v5.3, investigate before moving on. Either data quality issue or training regression — both fixable.
5. **Day 11 gate**: if silver-label κ against PadChest-GR radiologist labels < 0.4, escalate — report honestly but reframe the paper as "synthetic-claim benchmark with unreliable silver extension to real claims" rather than "validated silver pipeline." Changes the story materially; need user sign-off.
6. **Day 13 gate**: if the paper cannot be structured around the diagnostic-first reframe within the 9-page limit, drop real-hallucination P/R section to appendix and keep the diagnostic + mitigation + cross-site as the headline.
7. **Day 15 gate**: if any reviewer agent flags a numerical inconsistency between paper and JSONs, fix before submission. No "submit with known issues."

---

## 7. Pre-flight review call (run after user approves plan)

Spawn an Opus pre-flight agent before Day 2 with the following pack (user to trigger):

Inputs:
- This plan doc (`PLAN_V6_17DAY_NEURIPS.md`)
- `ARCHITECTURE_V6_0_NEURIPS_MAIN.md`
- The 4 research-agent outputs above
- `v5_final_results/TIER3_RESULTS_2026-04-19.md` (current evidence base)

Ask:
1. Is the NeurIPS E&D positioning defensible given current evidence? Name reviewer objections we haven't addressed.
2. Is the 17-day schedule realistic given parallelism on H100 and Anthropic rate limits?
3. Are there any load-bearing assumptions that haven't been validated (e.g., "MAIRA-2 will run in 4 h on H100", "RadFact Claude Opus will hit ~$50/1000 claims at 2 calls per claim")?
4. Is the budget too tight, too loose, or right?
5. Are there any failure modes that would make the paper unpublishable even if all 17 days succeed (e.g., a licensing clash in benchmark redistribution)?
6. Verdict: **ready-to-launch / launch-with-fixes / do-not-launch**.

Cost: ~$0.50 of Opus API tokens. Standing rule: never launch if verdict is DO-NOT-LAUNCH.

---

## 8. What this plan does NOT do

- **No radiologist recruitment.** All expert grounding comes from PadChest-GR's pre-existing public radiologist bboxes. Future work acknowledges clinical validation is needed for deployment claims.
- **No credentialed-data access** (MIMIC-CXR, MS-CXR, BRAX, CheXmask, RadGraph-XL, ReXVal). Future work: credentialed-dataset extension is a natural follow-up and is sized at ~1 additional month of work.
- **No architectural changes to v5.3.** This plan is additive — data + evaluation + baselines + writing. The verifier itself is frozen.
- **No full CheXpert Plus (350 GB download).** If timeline allows on Day 16 buffer, add a 5k-study subset. Not core to the plan.
- **No ReXGradient-160K.** HF-gate approval timing too unpredictable for 17 days.
- **No GPT-4o / Gemini API baselines.** Cost (~$60) better spent on RadFact-via-Claude and replications. Optional Day 16 buffer activity if under budget.
- **No multi-seed for v5.0–v5.2**. Only v5.3 and v6.0 get 3 seeds if buffer allows.
- **No Figma/Streamlit demos** this cycle. Paper-first.

---

## 9. Change log

| Date | Version | Author | Change |
|---|---|---|---|
| 2026-04-19 | v6.0 (initial) | Aayan Alwani | Initial 17-day plan. Replaces v5.0's 9-month timeline with a NeurIPS 2026 May 6 sprint. Rationale in §0 headline decisions |
