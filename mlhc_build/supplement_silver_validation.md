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
