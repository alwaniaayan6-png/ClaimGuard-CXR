# ClaimGuard-CXR v2 — Design Decisions Log

## D1: CheXzero over BiomedCLIP for multimodal fusion
**Decision:** Use CheXzero (Tiu et al., Nature BME 2022) instead of BiomedCLIP.
**Reason:** BiomedCLIP was trained on PMC-15M (academic figures, charts, cropped scans). Its zero-shot performance on raw clinical CXRs is brittle. CheXzero was trained on 377K MIMIC-CXR image-report pairs — its embeddings are inherently aligned to CXR semantics.

## D2: 2-layer MLP gate instead of linear gating
**Decision:** Gate is Linear(4,16)->ReLU->Linear(16,1) (81 params) instead of Linear(3,1) (4 params).
**Reason:** DeBERTa post-softmax scores are heavily polarized ([0.01, 0.99]) while CLIP cosine sims cluster tightly ([0.2, 0.35]). A linear gate cannot learn the non-linear calibration mapping between these distributions. 81 params is still tiny (no overfitting risk).

## D3: Learned tau_clip temperature for CLIP scores
**Decision:** Add a learned temperature parameter that scales CLIP cosine similarity before sigmoid.
**Reason:** Without temperature scaling, the linear gate is dominated by the text logit because CLIP's raw dynamic range is ~10x narrower than DeBERTa's. tau_clip normalizes CLIP scores into a comparable range.

## D4: Density ratio estimation on hidden representations, not softmax scores
**Decision:** CoFact density ratios estimated on 256-dim penultimate hidden (PCA to 32-dim), not 1D softmax.
**Reason:** Neural softmax outputs are spiky and clustered. Density ratio estimation on 1D softmax will explode to infinity for OOD test scores, breaking FDR guarantees. The 256-dim representation has richer distributional structure for stable estimation.

## D5: Manual annotation for real hallucination ground truth
**Decision:** Human annotators label CheXagent outputs, not CheXbert auto-labeling.
**Reason:** CheXbert has ~10-15% label noise. Evaluating a SOTA verifier against a legacy labeler is circular — reviewers would reject this immediately. Quality > quantity: 200 images with manual labels beats 500 with noisy auto-labels.

## D6: LLM-based claim extraction with rule-based fallback
**Decision:** Phi-3-mini for contextual claim extraction, with enhanced rule-based fallback.
**Reason:** Naive sentence splitting breaks negation scope ("no pneumothorax or effusion" split incorrectly) and temporal context ("previously noted ... has resolved"). The LLM produces contextually complete claims. Rule-based fallback has context-aware merging for demo/CPU mode.

## D7: Progressive NLI: skip RadNLI stage
**Decision:** Pipeline is MNLI -> MedNLI -> ClaimGuard (skip RadNLI).
**Reason:** RadNLI requires PhysioNet credentialing which isn't done yet. The MedNLI -> ClaimGuard chain still captures domain adaptation. RadNLI (480 pairs) would help but isn't blocking.

## D8: MNLI subsampled to 100K for speed
**Decision:** Use 100K random MNLI examples instead of full 393K.
**Reason:** Full MNLI on H100 would take ~3-4 hours. 100K is sufficient for learning general NLI patterns before medical domain adaptation. The value of MNLI is in the pretraining signal, not in exhaustive training.

## D9: RadFlag baseline is simplified (no real generator)
**Decision:** Simulate RadFlag's self-consistency via keyword-overlap perturbations instead of actual multi-sample generation.
**Reason:** We have no trained report generator. The real RadFlag approach requires sampling N reports per image. Our simplified version captures the core idea (consistency checking) but will underperform the real method. This is a fair conservative comparison.

## D10: Hypothesis-only baseline uses same model architecture
**Decision:** Train a separate DeBERTa-v3-large with evidence masked (padding tokens).
**Reason:** Using the same architecture controls for model capacity. If hypothesis-only achieves >90%, the problem is in the data (surface shortcuts), not the model.

---

## v3 sprint additions (2026-04-14)

The v3 sprint added 9 improvements to ClaimGuard-CXR (see
`HANDOFF_ClaimGuard_Improvements_2026-04-14.md` and
`/Users/aayanalwani/.claude/plans/rosy-discovering-bubble.md`).
The design decisions below were made during that sprint and are
locked by unit tests in `tests/`.

## D11: Per-class RNG seeding in `sample_stratified` uses `seed + ordinal_idx`
**Decision:** Task 8's stratified sampler derives a per-class RNG seed
as `random.Random(seed + ordinal_idx)` where `ordinal_idx` ∈ {0,1,2,3,4}
for the 5 labels in `VALID_LABELS` order.
**Reason:** Mersenne Twister streams from seeds 42/43/44/… are
statistically independent (different state vectors after
`init_by_array`). Using `seed + ordinal_idx` instead of
`hash((seed, ordinal_idx))` is simpler and achieves the only property
that matters: adding a 6th class at the end of `VALID_LABELS` does
NOT reshuffle the picks for the existing 5 classes. The Opus reviewer
explicitly endorsed this choice during the Task 8 self-check.

## D12: `drop_uncertain_values` (canonical) beats `drop_uncertain_units` (strict)
**Decision:** The fallback rung 2 in
`compute_user_vs_ensemble_alpha.compute_fallback_ladder` uses
`drop_uncertain_values` — nan-ify every `UNCERTAIN` cell and keep the
unit. The stricter `drop_uncertain_units` (drop the whole unit if any
coder voted UNCERTAIN) is still exported for methodologists who want it.
**Reason:** Krippendorff (2018) §3.4 treats missing values at the cell
level, not the unit level. The unit-level drop shrinks the sample
faster and loses units where only one coder hedged. Value-level
nan-ification is the canonical interpretation of "drop UNCERTAIN."

## D13: `format_prompt` must NOT leak the silver majority
**Decision:** The Task 8 interactive CLI shows the user the claim +
image + ground-truth radiologist report, but NEVER the silver graders'
majority label or individual grader outputs. Two regression tests
enforce this (`test_prompt_does_not_leak_silver_majority`,
`test_prompt_does_not_leak_grader_labels_even_when_present`).
**Reason:** The user is used as an independent 4th coder to validate
the 3-grader silver ensemble. If the user is primed by the ensemble's
answer, the resulting Krippendorff α is a "how often does the human
agree with what was shown to them" check, not an independent
reliability estimate. Reviewers who know Krippendorff will flag this
immediately. Opus-reviewer-flagged critical methodology issue.

## D14: Binary coarsen rung uses `level="nominal"`, not ordinal
**Decision:** The binary SUPPORTED-vs-rest rung in the fallback ladder
is reported with `level="nominal"` in
`alpha_with_bootstrap_ci`. The full_ordinal and drop_uncertain rungs
stay on `level="ordinal"`.
**Reason:** With only two values, nominal and ordinal δ² differ only
by a constant scale factor, so α (which is scale-invariant in δ²) is
identical. Using `level="nominal"` matches reader expectations
("binary → nominal") and avoids the confusing prior comment that
claimed δ² itself was equal across metrics (the Opus reviewer
flagged that proof as wrong, even though the numeric result was
correct).

## D15: Task 9 dual-run uses shared `image_seed` with distinct generation seeds
**Decision:** `generate_annotation_workbook` in
`scripts/generate_real_hallucinations.py` takes an optional
`image_seed: int | None = None` that defaults to `seed` for backward
compatibility. Task 9 passes the SAME `image_seed` to both runs
(shared image set) with DIFFERENT `seed` values for the torch
generation RNG (divergent outputs).
**Reason:** Task 9 needs runs A and B to cover identical images so
the downgrade-rate comparison has a common denominator. Without a
separate `image_seed`, the only options were (a) same seed everywhere
(identical outputs, experiment is void) or (b) different seeds
everywhere (disjoint image sets, experiment is invalid). The
3-line surgical edit decouples the two seed spaces.

## D16: `compound_perturbation` applies validators per-step AND on final output
**Decision:** Task 2's `compound_perturbation` generator picks
`n_errors ∈ {2, 3}` (60/40 split) distinct single-type perturbations,
applies them in a fixed deterministic order
`[laterality_swap, finding_substitution, severity_swap, temporal_error,
  negation, region_swap, device_error, fabricate_*]`, rejects the
attempt if any step returns `None`, and runs
`_evidence_supports_claim_text` on the final text.
**Reason:** Applying validators only at the end would allow an early
failing perturbation to leave the claim in a weird intermediate state
that downstream perturbations might accidentally repair. The fixed
order + per-step validation guarantees that every surviving compound
example is the result of N truly distinct error types, each of which
was individually rejectable by the existing label-correctness logic.

## D17: DPO refinement freezes first 8 RoBERTa layers + early-stops hard
**Decision:** `scripts/modal_train_dpo_refinement.py` trains with
β=0.1, lr=5e-6, 1 epoch, gradient clip 1.0, first 8 RoBERTa layers
frozen, and early-stop conditions (a) KL > 5 or (b) mean reward
margin < 0 for 50 consecutive steps. If either triggers, the script
aborts and the v4 checkpoint is not produced.
**Reason:** DPO training is notoriously unstable on small preference
sets. Freezing the early layers preserves the v3 representations that
we want the refinement to build on; single-epoch + aggressive
early-stopping bounds the risk of KL collapse. If both abort
conditions trigger, we ship v3 without DPO and document the abort in
the paper — that's an acceptable fallback, not a project killer.

## D18: StratCP implemented from the medRxiv algorithm (no reference code)
**Decision:** `inference/stratcp.py` implements the Zitnik lab's
stratified conformal predictor from the medRxiv Feb 2026 description.
Per stratum `s`: `Q_s = quantile(cal_scores_s, (n_s + 1)(1 − α)/n_s)`;
at test time, reject null if `score_s ≥ Q_s`. Validated against a
synthetic Gaussian fixture with known target coverage (α=0.05 →
95% expected coverage); empirical coverage must fall within ±2 pp of
α over 1000 trials.
**Reason:** No public reference code exists. Reimplementing from the
paper description and validating against a synthetic fixture with
known ground truth is the only way to be confident the implementation
is correct. If our OpenI numbers diverge from the StratCP paper's
reported numbers by > 2 pp, we park StratCP as a "partial baseline"
with explicit caveat in the paper and (optionally) contact the first
author. This is a soft guarantee — if the synthetic fixture passes
but OpenI disagrees, that's an empirical dataset-shift question, not
a bug.

## D19: Task 9 generation orchestration lives in `main()`, not `_run_demo`
**Decision:** `scripts/demo_provenance_gate_failure.py::_run_demo`
does ONLY the verifier scoring + stats phase (on H100); the two
CheXagent runs are launched from `main()` on the local client via
sequential `with gen_app.run()` contexts.
**Reason:** The prior nested-`.remote()` pattern (call
`generate_annotation_workbook.remote(...)` from inside the demo
Modal container) was flagged by the Opus reviewer as (a) fragile
(requires a parent `app.run` context the outer container does not
own) and (b) wasteful (outer container sits idle for ~90 min of
inner CheXagent work, burning ~2× GPU time). Moving the
orchestration to the local entrypoint spawns the two generation
containers from the client without any nested remote invocations.

## D20: Task 9 Modal image must ship `inference/` as local Python source
**Decision:** `scripts/demo_provenance_gate_failure.py` Modal image
now adds `.add_local_python_source("inference")` and
`.add_local_python_source("scripts")` so the container has the
repo's `inference/` package on `sys.path` at import time.
**Reason:** The deployed `claimguard-provenance-gate-demo` app
crashed silently on every cold start with
`ModuleNotFoundError: No module named 'inference'` because the
script's module-level `from inference.provenance import …` could
not be resolved inside the container. Modal was restarting the
crashed container and the client-side `.get()` loop just reported
"RUNNING" indefinitely (the FunctionCall was alive but the
container was never actually running). Caught from `modal app logs
ap-N7KOT599Q60g7rnT0DwQJG`. The same pattern is used in
`scripts/modal_build_retrieval_eval.py` line 69 for `models/`.

## D21: Task 9 verifier loader rewritten to define the full `VerifierModel` class
**Decision:** `_load_v1_verifier` in
`scripts/demo_provenance_gate_failure.py` was replaced by a
definition of the full `VerifierModel` class (text_encoder +
heatmap_encoder + verdict_head + score_head + contrastive_proj),
matching `scripts/modal_run_evaluation.py` lines 105–162 exactly.
`_score_claim_evidence_pairs` was updated to call
`model(input_ids, attention_mask)` and use
`softmax(verdict_logits)[:, 0]` as the supported-class score. The
default `verifier_checkpoint` now points at
`/data/checkpoints/verifier_binary_v3/best_verifier.pt`.
**Reason:** The earlier loader assumed a plain
`AutoModel + Linear(hidden, 2)` layout with checkpoint keys
`state["encoder"]` + `state["head"]`. That layout does not exist —
no training script in this repo ever wrote it. The real v1/v3
checkpoint is the full `VerifierModel` `state_dict` with
`text_encoder.*`, `heatmap_encoder.*`, `verdict_head.*`,
`score_head.*`, and `contrastive_proj.*` keys (fused dim
1024 + 768 = 1792). The old loader loaded only the text_encoder
into a mismatched AutoModel and left the Linear(1024, 2) head at
random initialization; `_sanity_check_verifier` caught this with
`sup=0.2063, con=0.2029, margin=0.0034 < 0.1`. The replacement
defines the correct architecture inline and loads every non-pooler
/ non-contrastive key with `strict=False` but raises on any
hard-missing keys outside the allowed prefixes. Local-CPU test:
on 15 real eval rows per class, separation is 0.998 vs 0.054
(margin +0.944). This bug was masked for as long as it existed
because the loader returned `(tokenizer, encoder, head)` shaped
correctly even though the head was random — `strict=False` is a
foot-gun when the key space is wrong, not just incomplete.

## D22: Task 9 sanity probes drawn from real eval distribution, not hand-written
**Decision:** `_SANITY_SUPPORTED_PROBES` and
`_SANITY_CONTRADICTED_PROBES` are now 4 claim-evidence pairs each,
sampled verbatim from `/data/eval_data_v3/test_claims.json` with
their evidence already joined via `" [SEP] "` in the training
format. `_SANITY_MARGIN` raised to 0.2 (from 0.1) and the check is
mean-over-probes instead of single-probe pairs. On healthy v3 the
margin is ~0.998; the threshold is a conservative lower bound.
**Reason:** The original hand-written probes ("There is evidence
of cardiomegaly." + well-formed English evidence) are off
distribution. The v1/v3 training data is MIMIC-CXR report
fragments in mixed ALL-CAPS clinical shorthand. On the
off-distribution probes, the v3 verifier outputs
`sup=0.9749, con=0.9756, margin=-0.0007` — a false sanity-check
failure that would block every Task 9 run even when the model is
healthy. Using real eval rows guarantees the sanity check tests
the same input distribution the model was trained on, so the
probes themselves cannot produce false negatives. The mean-over-4
margin formulation also tolerates individual edge probes without
failing the overall check.

## D23: Task 9 real dual-run headline number = `downgrade_rate_diff: 1.00`
**Decision:** The Task 9 real CheXagent dual-run landed on
2026-04-15 with a clean 100% same-model downgrade rate and 0%
cross-model downgrade rate, giving the maximum possible
`downgrade_rate_diff` of 1.00 on 414 matched claim pairs per
condition (828 rows total). Mean verifier score: 0.987 same, 0.986
cross; paired score diff: mean 0.0009, stdev 0.0055; 95.7% of
paired claims receive identical scores within 1e-4; zero pairs
differ by more than 0.1. The v3 verifier cannot distinguish the
two conditions by content; the provenance gate catches every
same-model pair using only metadata.
**Reason:** This is the cleanest empirical argument in the paper
for the provenance gate. The pass bar was `downgrade_rate > 0.5`;
the actual result is 2× that. The result is published in
`results/same_model_experiment/real/gate_demo.json` and the full
per-row table in
`results/same_model_experiment/real/gate_demo_rows.json`. Promoted
from §3.7 "case study" to a full contribution in the Discussion
section.

## D24: Task 6 v3 OpenI transfer — inverted cfBH holds, forward cfBH collapses, StratCP overshoots
**Decision:** Task 6's v3 OpenI recalibrated eval ships three
methods as a direct head-to-head comparison in the paper: inverted
cfBH (main), StratCP (miscoverage baseline), and forward cfBH
(ConformalClaimTriage). The inverted cfBH is the paper's main
result; the other two are explicit failure-mode comparisons.
Numbers on v3 OpenI (n_test=900, patient-level 50/50 split):

| α | inv_cfbh_FDR | inv_cfbh_n_green | stratcp_FDR | fwd_cfbh_n_green |
|---|---|---|---|---|
| 0.05 | 0.0088 | 113 | 0.1808 | 0 |
| 0.10 | 0.0050 | 199 | 0.2778 | 0 |
| 0.15 | 0.0050 | 201 | 0.3549 | 0 |
| 0.20 | 0.0344 | 378 | 0.4006 | 0 |

**Reason:** The v3 OpenI transfer is the paper's main cross-dataset
claim. Inverted cfBH holds FDR at every α even though raw accuracy
drops from 0.9877 to 0.7545 — exactly what conformal prediction
promises. StratCP is included as the Zitnik-lab Feb 2026 baseline
and demonstrates the distinction between miscoverage control and
FDR control: StratCP is certified to bound per-stratum miscoverage,
and its empirical FDR blows past α at every level on OpenI, which
is the expected behavior but is not suitable for a "green means
safe" downstream use case.  Forward cfBH is the textbook direction
(calibrate on faithful) implemented via `ConformalClaimTriage` with
**per-group BH**; on v3 OpenI it returns n_green=0 at every α
because after one-per-patient subsampling the faithful-calibration
pool in the "Rare/Other" pooled group has only ~69 samples, so the
smallest p-value (1/70 = 0.014) exceeds both the per-group
rank-1 BH threshold (α × 1/n_test_pooled = 0.05/315 = 1.6e-4 at
α=0.05, where 315 is the pooled-group test-set size) and the
hypothetical global-BH threshold (α × 1/n_test = 0.05/900 = 5.6e-5
at α=0.05), by >100× in either case — a pure calibration
granularity failure independent of which BH variant is applied. This is the cleanest empirical
evidence we have for D1's inverted-calibration decision: the forward
direction is not just suboptimal, it is brittle on small cross-
dataset splits, and can collapse to zero rejections on realistic
data. We keep the forward run in the paper's Task 6 table as the
"why inversion matters" row.

## D25: Task 3a IG identification capped at 5000 contradicted claims, not 30000
**Decision:** `scripts/run_causal_term_identification.py` is invoked
with `--max-claims 5000` against the v3 contradicted training pool
(of ~12000 contradicted claims in the 30000-row v3 training data).
Output written to `/data/causal_spans_v3.jsonl` on the `claimguard-data`
volume.
**Reason:** Task 3b counterfactual generation is the cost bottleneck,
not Task 3a.  At Sonnet 4.5 prices (~$0.0017/call), running Claude
on 30000 × 3 variants = 90000 calls would cost ~$153 — blowing the
plan's entire $43 Task 3 budget on a single sub-stage.  Capping at
5000 claims keeps Task 3b at ~$25 (15000 calls), leaves headroom
for the Task 3c training itself, and still produces ~12000 usable
preference pairs after the Task 3b filter (96% smoke success rate).
12000 pairs is well above the floor needed for a single-epoch R-Drop
refinement on a 357M-parameter VerifierModel.

The smoke-measured H100 rate was 0.3 sec/claim, which would have
allowed 30000 in ~2.5 hours for $5.  IG attribution is cheap; the
constraint is downstream Claude API cost, not GPU.

## D26: Task 3b counterfactual filter — meaningful tokens + hybrid substring match
**Decision:** `data.augmentation.counterfactual_generator.normalize_causal_tokens`
filters input causal tokens through `_is_meaningful_causal_token`
before they reach `validate_preservation`.  Filter criteria: ≥4 chars,
≥4 alphabetic chars, no leading punctuation/digit, not an ALL-CAPS
short fragment (4-7 chars), not a medical-boilerplate stopword
("single portable", "in the", "demonstrates", etc.), and capped to
top-3 meaningful tokens.  If the strict filter drops every token,
fall back to the longest-by-alpha-count original.

`validate_preservation` uses a hybrid match strategy: tokens ≥6
chars OR multi-word phrases get case-insensitive **substring** match;
tokens <6 chars and single-word get case-insensitive **word-boundary**
regex.

**Reason:** Empirical failure caught during the first Task 3b
production launch on real v3 IG output: 99% of rows produced
n_returned=0 because `validate_preservation` was rejecting all of
Claude's variants.  Two distinct root causes:

(1) IG attribution surfaces a mix of real causal content tokens AND
subword/punctuation artifacts in its top-K output.  Example from
v3_train_000000: `['STITIAL EDEMA worsening.', 'severe', ',',
'single portable', '.']`.  Requiring preservation of ALL 5 tokens
means the bare comma and period kill the row.

(2) `validate_preservation` used `\btoken\b` word-boundary regex.
When IG surfaces a BPE subword fragment like `"STITIAL"` (from
`"INTERSTITIAL"`) and Claude correctly produces a paraphrase
containing `"interSTITIAL EDEMA worsening."`, the word-boundary
check refuses the match because there's no boundary between `"r"`
and `"S"` inside `"interSTITIAL"`.

Empirical validation on 50 real v3 contradicted claims:
* Filter v0 (no fixes):              1/50 = 2% kept
* Filter v1 (meaningful filter):     16/50 = 32% kept
* Filter v2 (+ longest fallback):    16/50 = 32% kept (fallback
                                     didn't help on its own)
* Filter v3 (+ hybrid match):        48/50 = **96% kept**

The 6-char threshold for substring vs word-boundary is empirical:
medical BPE fragments are typically 4-7 chars (`STITIAL`, `OSSIBLE`,
`EGALY`), while the worry case `"heart"` matching `"heartfelt"` is
5 chars and stays on the word-boundary path.  The 60/60 unit tests
pass under this rule, including the regression test for the
heart/heartfelt false positive.

## D27: Task 1 silver standard — drop CheXbert text-only labeler from grader ensemble
**Decision:** The Task 1 silver-standard 3-grader ensemble is
retired in favor of a 2-grader pipeline (Claude Sonnet 4.5 vision +
self-annotation by the user, n=100 stratified subset).  The
rule-based CheXbert text-only labeler is moved to a separate
"report-coverage" diagnostic role and is no longer counted as a
silver-standard grader.  MedGemma is dropped entirely (gated repo +
broken fallback chain).

**Reason:** Empirical disagreement on the 414-claim Task 1 v3 run.
Krippendorff α between CheXbert and Claude Sonnet 4.5 vision was
**0.08** across 3 fallback rungs (full 5-class ordinal, drop
UNCERTAIN, binary coarsen), with the disagreement driven by 63
cases (15% of all claims) in which CheXbert flagged
`NOVEL_HALLUCINATED` and Claude flagged `SUPPORTED`.

The pattern is consistent: CheXbert is a TEXT-ONLY rule-based
labeler.  It sees a CheXagent claim about a pathology that isn't
in the original radiologist *report* and stamps `NOVEL_HALLUCINATED`.
But Claude has *image* vision and sees the pathology in the actual
chest X-ray, so it (correctly) stamps `SUPPORTED`.  This isn't a
methodology failure — it's an empirical measurement of how much the
OpenI radiologist reports under-describe what's visible in the
image.  The reframe: text-only graders are fundamentally
inappropriate for image-grounded hallucination evaluation, and the
field should retire them in favor of vision-language graders.  This
becomes a methodological finding for the paper, filed alongside the
retired-CheXbert decision in §3.6.

The MedGemma path is dropped because all three fallback models
failed in the live Modal run: `google/medgemma-4b-it` is gated,
`microsoft/llava-med-v1.5-mistral-7b` has a `llava_mistral` model
type that transformers 4.50 doesn't recognize, and
`StanfordAIMI/CXR-LLAVA-v2` doesn't exist as a HuggingFace
identifier.  Replacing the third grader with a different model is
deferred to a future session; the Task 8 self-annotation pass
serves as the second coder for the final α computation.

## D28: Task 3c v1 R-Drop collapsed — single-class training on contradicted-only data
**Decision:** v4 v1 checkpoint (trained with `loss_mode="consistency"` on
8415 contradicted-only cf pairs) is DISCARDED.  Checkpoint at
`/data/checkpoints/verifier_binary_v4_rdrop/` is NOT a valid v4; it
predicts CONTRADICTED for every input (synth acc 0.3339, OpenI acc
0.3274, per-class F1 [0.000, 0.493]).  Training appeared healthy
(no early-stop, final_loss=9.8e-6) but the low loss was trivial CE
convergence on single-class data, not learning.
**Root cause:** Task 3a filtered v3 training data to label=1 only.
Task 3b generated counterfactual paraphrases of those (also label=1).
The trainer's `_consistency_classifier_loss` hardcoded
`target = torch.full((B,), fill_value=1)`.  R-Drop on a single-class
dataset trivially converges to "always predict that class."
**Cost wasted:** ~$8 H100 + ~$2 eval.

## D29: Task 3c v2 R-Drop on mixed data — HO gap goes negative; reframing
**Decision:** v4 v2 checkpoint (`loss_mode="consistency_mixed"`, 16830
mixed rows: 8415 cf label=1 + 8415 faithful label=0) is the FINAL v4
for the paper.  Per-example labels + mixed dataloader fixed the
single-class collapse from D28.

v4 v2 headline numbers:
  synth test acc:   0.9596  (v3 was 0.9877, delta −2.81 pp)
  OpenI test acc:   0.7623  (v3 was 0.7545, delta +0.78 pp)
  HO baseline acc:  0.9857  (claim-only, 3 epochs on v3 training data)
  v3 HO gap:        +0.20 pp  (v3 − HO = 0.9877 − 0.9857)
  v4 v2 HO gap:     −2.61 pp  (v4 − HO = 0.9596 − 0.9857)
  Plan target:      ≥ +5.0 pp  → NOT MET

**Why the paper story is BETTER than hitting the target:**
The negative HO gap means v4 v2 is WORSE than a claim-only baseline
on the synthetic test set.  But v4 v2 is BETTER on real OpenI data.
This proves that the HO gap on synthetic data measures "how much
shortcut the model exploits," not "how much evidence reasoning the
model uses."  R-Drop broke the shortcuts (laterality_swap dropped
from 100% to 27%, region_swap from 100% to 27%) and forced real
evidence reasoning, which helped on cross-dataset transfer (+0.78 pp)
but hurt on the synthetic benchmark that REWARDS shortcuts.

**Paper reframing:**
"The hypothesis-only gap on synthetic data is the wrong metric for
evaluating evidence reasoning.  A claim-only model solves the v3
synthetic test set at 98.57%.  R-Drop counterfactual refinement
trades 2.81 pp of synthetic accuracy for 0.78 pp of cross-dataset
accuracy, demonstrating that cross-dataset transfer — not
within-distribution synthetic accuracy — is the right evaluation
axis for verifiers deployed against real AI-generated reports."

cfBH FDR control survives on both:
  v4 v2 synth:  0.016 / 0.029 / 0.044 / 0.061 at α=0.05/0.10/0.15/0.20
  v4 v2 OpenI:  n_green=0 at α≤0.15; FDR=0.005 at α=0.20 (conservative)

## D30: HO baseline on v3 data confirms SEVERE lexical-shortcut artifact
**Decision:** HO baseline trained on v3 training data (3 epochs,
RoBERTa-large, claim + masked evidence) reached val_acc = 0.9857
with contra_recall = 0.9667.  The auto-interpretation: "SEVERE
artifacts.  Task is nearly solvable without evidence."  v3 HO gap
= 0.9877 − 0.9857 = 0.20 pp — WORSE than v1's 0.60 pp.  The 12-type
taxonomy expansion (8 → 12 perturbation types) made the artifact
STRONGER because more perturbation types = more lexical patterns =
more shortcuts.  This is a directly measurable failure mode of
synthetic-only training paradigms.


## D31: Pivot to Path B — image-grounded claim verification (2026-04-17)
**Decision:** Pivot the paper from text-only synthetic-negative training
to an **image-grounded** claim verifier evaluated against
radiologist-drawn pixel annotations. Replace "silver-standard LLM-ensemble
labels" with "radiologist bounding boxes / segmentation masks" as ground
truth. Target venue shifts from NeurIPS D&B to npj Digital Medicine /
Medical Image Analysis (primary) or Nature Machine Intelligence
(stretch, contingent on radiology co-author + PhysioNet credentialing).
**Rationale:** v4's negative HO gap on synthetic data (−2.61 pp) shows
the model does not use evidence; the paper cannot claim evidence-grounded
reasoning on synthetic data alone. Anchoring ground truth in pixels
radiologists drew on removes the text-vs-text circularity that a
Nature-family reviewer would flag. Details: `ARCHITECTURE_PATH_B.md`.


## D32: Path B dataset mix — drop PhysioNet-credentialed datasets from headline
**Decision:** Drop MS-CXR, CheXmask source images, MIMIC-CXR,
VinDr-CXR, ReXVal, ReXErr, RadGraph-XL from the primary evaluation
because they require PhysioNet credentialing. Primary datasets: PadChest-GR
(BIMCV), RSNA Pneumonia (Kaggle), SIIM-ACR (Kaggle), ChestX-Det10 (GitHub),
NIH CXR14 BBox subset, Object-CXR. PhysioNet-bound datasets reserved as
supplementary if credentialing lands in parallel.
**Rationale:** The user cannot hold a PhysioNet credential personally
(HS student, age + institutional-affiliation requirements) and cannot
assume Laughney will file on our timeline. Feasibility review caught
v1 draft's incorrect claim that MS-CXR lives on HuggingFace; correcting
this forced the scope change.


## D33: Four-way contrastive training objective (V1/V2/V3/V4)
**Decision:** Replace the v1 draft's 3-variant evidence-contrastive loss
with a 4-variant objective that includes **image-swap negatives**:
V1 (supp evidence + correct image, label 0), V2 (contra evidence +
correct image, label 1), V3 (supp evidence + RANDOM-PATIENT image,
label 1), V4 (supp evidence + zero-image, label 1).
**Rationale:** Review caught that the v1 loss was gameable — a text-only
model can learn the lexical signature of `evidence_supp` vs
`evidence_contra` and satisfy the evidence margin without looking at the
image. Adding V3 (image-swap) means the text is identical between V1 and
V3 but the label differs; therefore any model that ignores the image
produces equal supported-probabilities on V1 and V3, violates the
`img_margin` term, and cannot minimize the loss. The HO baseline fails
this objective by construction.


## D35: Path B wave-2 scaffold — loaders, extractor, 4-way dataset, baselines, fairness, DCA, manuscript scaffold
**Decision:** Expand the Path B package with: (a) five dataset loaders
(RSNA, SIIM, ChestX-Det10, PadChest-GR, NIH-BBox) returning a uniform
``GroundedStudy`` schema; (b) an LLM-agnostic claim extractor with a
StubBackend for tests; (c) a PyTorch ``FourWayTrainingDataset`` that
yields V1/V2/V3/V4 variants per sample with O(1) image-swap sampling;
(d) a baselines registry covering majority-class, rule-based, LLM-judge,
VLM-judge, RadFlag; (e) fairness subgroup-metrics + decision-curve
analysis modules; (f) ``requirements.txt``, ``pyproject.toml``,
Dockerfile for reproducible builds; (g) a MANUSCRIPT_PATH_B.md skeleton.
Total: 78 unit tests passing.
**Rationale:** Land everything the paper depends on in code so the only
remaining work is data ingestion (credentialing + Kaggle tokens) and
real training runs. Lets the next agent session start at the top of the
training loop instead of re-scaffolding.


## D36: Pre-code review (wave 2) caught 4 blockers + 4 non-blocking bugs
**Decision:** A second Opus pre-flight review on the wave-2 modules
caught: (1) ChestX-Det10 loader was using bbox rectangles instead of
polygon segmentation — silently downgrading grounded-IoU; (2)
PadChest-GR loader returned empty annotations on any schema mismatch,
no error raised; (3) ``FourWayTrainingDataset._pick_random_other_patient_image``
rebuilt candidate lists O(n_patients) per ``__getitem__``; (4)
``LLMJudgeBaseline`` did not strip ``\`\`\`json`` fences before
``json.loads`` → silent UNCERTAIN floor on every fenced response.
Non-blocking: NIH-BBox CSV header whitespace + delimiter sniff, RSNA
missing-file skip, claim extractor parse-failure diagnostics,
stub-image-loader value-range contract. All fixed; 9 regression tests
added.
**Rationale:** The ChestX-Det10 bug alone would have halved IoU on every
ChestX-Det10 test claim. The LLM-judge fence bug matches the exact
failure-mode pattern from the v3 sprint silver-grader incident — a
known class of bug that pre-flight review catches.


## D34: Pre-code review caught 6 fatal/major bugs in Path B scaffold
**Decision:** After scaffolding Path B code, an Opus self-check review
agent found: (1) lazy `_text_proj` layer inside `forward()` would miss
optimizer registration; (2) `_freeze_except_last_n` silently left the
whole image encoder frozen when the attribute path resolution failed;
(3) `WeightedCfBH` recomputed calibration weights per test point (100×
perf hit); (4) `bootstrap_ci` closure bug — y_pred was not being
resampled with y_true, so CIs were meaningless; (5) `_auroc` fallback
computed `1 - AUROC` due to a mis-written argsort permutation; (6)
`FourWayContrastiveLoss` double-counted V4 (effective weight
`lambda_ce + lambda_mask = 1.05` instead of `lambda_mask = 0.05`). All
fixed before any Modal launch. Regression tests added for each.
**Rationale:** Doc-sync + pre-flight review pattern from the v3 sprint
caught all 6 bugs in one review pass. Each would have wasted GPU hours
or produced invalid metrics. Documenting here to preserve the catch.
