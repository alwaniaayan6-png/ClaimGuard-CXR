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
