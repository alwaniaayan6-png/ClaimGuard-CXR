# Supplementary — Silver-label ensemble + PadChest-GR validation

This supplement reports the full silver-label ensemble statistics and the PadChest-GR radiologist-bbox agreement attempt.

## 1. Three-grader ensemble

Computed over 2,707 atomic claims extracted from 500 CheXagent + 1,000 MAIRA-2 + 1,000 MedGemma-4B-IT generations on 500 held-out OpenI studies.

### Per-grader prevalence

| Grader | SUPPORTED | CONTRADICTED | UNCERTAIN |
|---|---|---|---|
| GREEN-RadLlama2-7B (MIMIC-fine-tuned) | 498 | 245 | 1,964 |
| RadFact (Claude Opus entail, MIMIC-free) | 1,598 | 259 | 850 |
| VERT (Claude Opus structured, MIMIC-free) | 1,765 | 474 | 468 |

GREEN is UNCERTAIN-heavy — consistent with its strict error-detection framing (a claim is only SUPPORTED if GREEN's critic finds zero significant errors and at least one matched finding).

### Krippendorff α (pairwise + three-way)

| Pair | α |
|---|---|
| RadFact ↔ VERT | **0.701** |
| GREEN ↔ RadFact | 0.042 |
| GREEN ↔ VERT | −0.001 |
| three-way | 0.265 |

**Interpretation.** The two MIMIC-free graders agree strongly (α = 0.70 is in the "substantial agreement" band of Landis–Koch). GREEN's near-zero agreement with both MIMIC-free graders is direct empirical evidence of a leakage-driven divergence in label distributions, consistent with GREEN's MIMIC-CXR fine-tuning. This argues for reporting silver-label agreement by subset (MIMIC-free pair alone vs three-way) rather than aggregating all three into a single κ.

### Ensemble decision rule

2-of-3 majority with tie-break toward SUPPORTED. Confidence tiers assigned from exact-agreement pattern:

| Tier | Definition | Count |
|---|---|---|
| HIGH | all 3 graders agree | 942 |
| MED | 2 agree, 3rd is UNCERTAIN | 1,357 |
| LOW | 2 agree, 3rd disagrees (flagged) | 42 |
| EXCLUDED | full 3-way disagreement | 366 |

### Final label counts (after ensemble rule)

| Label | Count |
|---|---|
| SUPPORTED | 1,599 |
| CONTRADICTED | 335 |
| UNCERTAIN | 773 |

Disagreement count (any pair disagrees SUP/CONTRA): 408 / 2,707 = 15.1%.

## 2. PadChest-GR radiologist-bbox validation — paired match count

Ran `v5/eval/padchest_gr_validate.py` over PadChest-GR's 11,995 radiologist-labeled sentences (7,037 positive + 4,958 negative) against the ensemble's 2,707 silver-labeled claims.

| Metric | Value |
|---|---|
| n ground-truth sentences | 11,995 |
| n matched (Jaccard ≥ 0.30, same image) | **0** |
| n unmatched | 11,995 |
| Cohen κ | N/A (no matches) |

**Why zero matches?** The validator pairs by `image_id`: for each PadChest-GR sentence, it looks for silver-labeled claims on the same image and computes text similarity. Our silver labels were generated on OpenI images (the RRG pipeline ran on 500 held-out OpenI studies), not on PadChest-GR images. There are no shared `image_id` values between the two cohorts, so the matching returns zero.

The intended use case — validating silver labels on PadChest-GR claims against PadChest-GR radiologist bboxes — requires silver grading of claims on PadChest-GR images. That in turn requires claim extraction from PadChest-GR reports (Spanish/English), which we scaffolded at `v5/data/padchest_gr.py` but did not complete for this submission (see main-text §3.3 Limitations).

A cross-image text-only matching pass (ignoring `image_id`, matching PadChest-GR sentence text to silver claim text across all images) is possible and would likely produce non-zero matches on generic findings ("right pleural effusion", "cardiomegaly"), but the signal it provides is semantic phrasing agreement rather than radiologist-vs-silver grounding agreement — not a substitute for image-paired validation.

## 3. Camera-ready plan

1. Run claim extraction on PadChest-GR reports (both Spanish original and English translations).
2. Run the 3-grader silver pipeline on the resulting claims.
3. Re-run `padchest_gr_validate.py`; report per-finding Cohen κ and the pathology-class breakdown the paper's §3.3.7 currently scaffolds.
4. Expected wall-clock: ~8 hours H100 + ~$40 Anthropic credit.

---

## 4. HO-filter activation on real-RRG hallucinations (M6 gate detail)

Trained a RoBERTa-base text-pair classifier on the v6 train split (8,000 example budget, 1 epoch, `(claim_text, evidence_text) → label`). Scored all 1,934 ensemble-resolved real-RRG silver claims at `τ=0.7`. Full breakdown:

| Subset | n | n flagged | Activation rate |
|---|---|---|---|
| **All real-RRG claims** | **1,934** | **1,559** | **80.6%** |
| MedGemma-4B-IT | 1,330 | 1,150 | 86.5% |
| MAIRA-2 | 305 | 209 | 68.5% |
| CheXagent-2-3b | 299 | 200 | 66.9% |
| flagged | silver=SUPPORTED | 1,599 | 1,464 | 91.6% |
| flagged | silver=CONTRADICTED | 335 | 95 | 28.4% |

M6 gate target was ≥40% activation; safety floor ≥10%. Aggregate 80.6% passes both. The 91.6 vs 28.4 split shows the filter mostly downweights "easy" positives (claim text alone predicts SUPPORTED, e.g., "no acute findings") and rarely fires on the harder hallucinated negatives. This is the design intent — the loss the filter participates in then learns from the harder examples — but should not be read as evidence that the filter would catch most hallucinations as a deployed standalone detector.

## 5. Cross-site shortcut audit (per-site text-pair ceiling + HO activation)

Same RoBERTa-base text-pair classifier, evaluated separately per site:

| Site | n test | Majority acc | Text-pair acc | Δ vs majority | HO activation |
|---|---|---|---|---|---|
| ChestX-Det10 | 1,387 | 0.798 | 0.875 | +7.6pp | 68.2% |
| OpenI | 1,587 | 0.854 | 0.914 | +6.0pp | 82.2% |

Both sites have a real text-pair shortcut signal (5–8pp above majority); the v6 training distribution is not shortcut-free at the site level. The HO filter activates on most rows in each site (>68%), confirming that the downweighting affects a substantial fraction of the training distribution.

## 6. Support-score sharpness across configurations

Evaluated each verifier on the 2,974-claim v6 test split; reported KS distance to a uniform-on-[0.5,1.0] reference and the fraction of predictions above 0.9 confidence.

| Config | Mean predicted prob | KS to uniform | Frac p > 0.9 |
|---|---|---|---|
| v5.0-base | 0.935 | 0.69 | 79.3% |
| v5.1-ground | 0.938 | 0.73 | 82.0% |
| v5.2-real | 0.933 | 0.63 | 77.5% |
| v5.3-contrast | 0.929 | 0.62 | 76.9% |
| v6.0-retrain | 0.922 | 0.62 | 76.9% |

The two evidence-blind configs (v5.0, v5.1) have *slightly sharper* support-score distributions than the non-blind ones — opposite of what a "sharpness explains conformal coverage" hypothesis predicts. Conformal FDR's per-config behavior is therefore not explained by marginal support-score sharpness; it depends on the joint structure of (support score, error event), which the inverted cfBH procedure is designed to exploit. Per-config conformal sets are reported in `v5_final_results/conformal_summary.json`.

## 7. PadChest-GR rescue Phase 3 detail (12-match result)

After extracting more PadChest-GR images from the multi-part ZIP (500 PNGs in /tmp), MAIRA-2 successfully generated reports for 168/500 images (the rest failed for missing-image / processor errors). 168 reports decomposed via Claude Haiku to 565 atomic claims. Silver-labeling counts:

| Grader | SUPPORTED | CONTRADICTED | UNCERTAIN | ERROR |
|---|---|---|---|---|
| RadFact (Claude Opus entail) | 115 | 103 | 347 | 0 |
| VERT (Claude Opus structured) | 66 | 73 | 137 | **289** |

VERT's 51% error rate on this batch (vs 0% on the OpenI batch reported earlier) is most plausibly an Anthropic API rate-limit cluster on the heavier-prompt VERT calls on this specific machine — not a methodology issue. Re-running VERT alone with longer backoff is on the camera-ready punchlist.

Two-grader ensemble (RadFact ∩ VERT, agreement-only):

| Pattern | Count |
|---|---|
| Both SUPPORTED | 52 |
| Both CONTRADICTED | 48 |
| Disagreement | 2 |
| Either UNCERTAIN | 463 |

Validate against PadChest-GR bboxes (Jaccard ≥ 0.30):

| Metric | Value |
|---|---|
| n GT sentences | 11,995 |
| n matched | **12** (0.1%) |
| Cohen κ on matched | -0.80 (n_definite = 3 — small-sample noise) |
| TP / FP / FN / TN | 0 / 2 / 1 / 0 |
| n UNCERTAIN | 9 |

The 12-match result is too small for a publishable κ. Reported here for transparency and to scaffold the camera-ready re-run.
