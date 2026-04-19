# ClaimGuard-CXR v6.0 — Evidence-Blindness in Medical Multimodal Verifiers

**Status:** authoritative spec for NeurIPS 2026 submission
**Author:** Aayan Alwani (Laughney Lab, Weill Cornell Medicine)
**Date:** 2026-04-19
**Target venue:** **NeurIPS 2026 Evaluations & Datasets track** (primary); Main-track (fallback); GenAI4Health workshop (fallback)
**Deadline:** 2026-05-06 AoE
**Runway:** 17 days
**Budget:** $620 USD Modal remaining (of $900 project cap)
**Clinical collaborators:** none (explicit — no radiologist recruited at any phase)
**Companion:** `PLAN_V6_17DAY_NEURIPS.md` (day-by-day execution plan)

---

## 0. Version disambiguation (READ FIRST)

This project has undergone multiple major pivots. There are now **seven** architecture documents in the repo. Only **this document (v6.0)** is authoritative for the current work.

| Version | Doc | Status | Thesis |
|---|---|---|---|
| v1 | `ARCHITECTURE.md` | deprecated | text-only verifier + cfBH |
| v2 Path B | `ARCHITECTURE_PATH_B.md` | deprecated | image-grounded pivot |
| v2 (rename) | `ARCHITECTURE_V2_IMAGE_GROUNDED.md` | deprecated | duplicate of Path B |
| v3 / v4 (sprint notes inside v1 doc) | — | deprecated | multi-dataset transfer + HO-gap discovery |
| v5 Image-Grounded | `ARCHITECTURE_V5_IMAGE_GROUNDED.md` | deprecated | credentialed-data benchmark, Nature MI target |
| v5.0 Evidence-Blindness | `ARCHITECTURE_V5_0_EVIDENCE_BLINDNESS.md` | deprecated-by-this-doc | diagnose+mitigate evidence-blindness across 8 public datasets; 9-month runway; NeurIPS 2027 |
| **v6.0 NeurIPS 2026** | **this doc** | **authoritative** | **diagnostic-first reframe; NeurIPS 2026 E&D; 17-day sprint; real-RRG evaluation; PadChest-GR added** |

### How v6.0 differs from v5.0

| Axis | v5.0 Evidence-Blindness | v6.0 NeurIPS 2026 (this doc) |
|---|---|---|
| Venue target | NeurIPS 2027 Main (May 2027) | **NeurIPS 2026 E&D (May 6, 2026)** |
| Runway | 9 months | **17 days** |
| Paper framing | Image-grounded verifier + diagnostic | **Diagnostic-first (Gururangan-style)**; verifier is consequence, not headline |
| Training data sites | 8 planned; only OpenI + ChestX-Det10 actually used by Tier 3 | **3 sites: OpenI + ChestX-Det10 + PadChest-GR** (pre-existing + new add) |
| Evaluation claims | synthetic-only (2,906 CheXbert-flipped) | **synthetic + 6,000 real RRG-generated claims** from MAIRA-2, CheXagent-2, MedGemma-4B |
| Silver labels | none (GT = CheXbert automated) | **GREEN-RadLlama2-7b + RadFact-via-Claude-Opus ensemble, validated vs PadChest-GR radiologist bboxes** |
| External VLM baselines | none working (BiomedCLIP at 41% = broken) | **7 working**: CheXagent-2, MAIRA-2, MedGemma-4B, BiomedCLIP (fixed), RadFlag replication, ReXTrust replication, ours |
| Cross-site experiment | 2-way (OpenI↔ChestX-Det10) | **3-way leave-one-out** (OpenI, ChestX-Det10, PadChest-GR) |
| Artifact audit | not done | **Gururangan-style text-only ceiling measurement pre/post HO filter** |
| Support-score sharpness | not measured | **added (explains why conformal FDR only fires on v5.3)** |
| Must-cite concurrent work | not known | **HEAL-MedVQA (Nguyen et al., IJCAI 2025)** — TPT/VPT tests — differentiation required |
| Validation ground truth without radiologist | self-review of 500 claim-matcher outputs | **PadChest-GR's 7,037 radiologist-placed finding-sentence bboxes** |

---

## 1. Executive summary

**Contribution type:** Evaluation methodology + benchmark + companion method (fits NeurIPS 2026 Evaluations & Datasets charter).

**The three sub-claims:**

1. **Diagnosis.** Evidence-blindness — failure of a multimodal verifier to use its image input — is measurable via three counterfactual metrics: image-masking gap (IMG), evidence-shuffling gap (ESG), image-perturbation gap (IPG). We formalize these and calibrate thresholds on fully-image-blind and fully-image-using control models.

2. **Prevalence.** Evidence-blindness is **systematic** across seven public CXR hallucination detectors when evaluated on real generated reports from three public RRG models (MAIRA-2, CheXagent-2-3b, MedGemma-4B-IT). Baselines include zero-shot VLM verifiers, replications of ReXTrust (white-box hidden-state probe) and RadFlag (black-box temperature sampling), and prior ClaimGuard versions. This establishes evidence-blindness as a field-level property, not a single-model artifact. Analogous to Gururangan 2018's findings for NLI.

3. **Mitigation + honest limitation.** A training-time mitigation — consistency loss (penalizes image-masking invariance) + adversarial hypothesis-only filter (down-weights text-solvable training examples) — closes IMG by ~30× in-distribution (2.03pp → 69.24pp) without accuracy regression. However, the mitigation is training-distribution-specific: cross-site 3-way leave-one-out reveals IMG collapses to 0–10pp on held-out sites. We present this as an honest finding about the nature of the intervention, not a bug.

**Supporting contributions:**

- **ClaimGuard-GroundBench-v6**: 30k+ training claims + 500 RRG-generated test claims × 3 RRG sources, spanning three public sites (OpenI US, ChestX-Det10 US-subset, PadChest-GR Spain), bilingual EN/ES, with per-sentence radiologist grounding from PadChest-GR.
- **Silver-labeling protocol**: GREEN + RadFact ensemble with Krippendorff α agreement audit, validated against PadChest-GR's radiologist-placed bboxes (κ, precision, recall reported per finding class).
- **Conformal FDR coupling**: the evidence-blindness diagnostic composes with Jin–Candès 2023 conformal p-value FDR control, producing a single claim-level pipeline with an end-to-end guarantee when both pass.

**What v6.0 is NOT:**
- Not a clinical safety claim; does not argue for deployment.
- Not an attempt to replace radiologist review.
- Not a credentialed-data study. MIMIC-CXR, MS-CXR, BRAX, CheXmask, RadGraph-XL, ReXVal are **forbidden**. Credentialed-dataset extension is explicit future work.

---

## 2. Scope, constraints, assumptions

### 2.1 Hard constraints

| Constraint | Value |
|---|---|
| Total project budget | $900 USD cap; **~$620 remaining** |
| Calendar runway | **17 days** (2026-04-19 → 2026-05-06) |
| GPU class | H100 only (per user global CLAUDE.md policy) |
| Credentialed datasets | **forbidden** |
| Clinician recruitment | **none** at any phase |
| LLM vendors | Anthropic (Claude Opus 4.7 for RadFact; Claude Haiku for translation/extraction) |
| Download-time constraint | datasets that take >1 day to approve or >4 h to download are deferred |
| Model-weight license constraint | research-use-only is acceptable (MAIRA-2, LLaVA-Rad, RadVLM); document in ethics statement |

### 2.2 Soft constraints

- Reproducibility: every artifact ships with pinned Modal image, pinned model SHAs, Zenodo DOI for derived benchmark labels (not raw images).
- Privacy: public datasets only; all already pre-de-identified. Additional Presidio + CXR regex scrub on any extracted text.
- Compute hygiene: every Modal launch preceded by Opus pre-flight logic-review per user CLAUDE.md.
- Doc-sync: ARCHITECTURE_V6_0_NEURIPS_MAIN.md + CLAIMGUARD_PROPOSAL.md updated in the same commit as code changes (standing user rule).
- Budget guardrail: stop-on-exhaust after $560 of $620 remaining (15% buffer).

### 2.3 Assumptions (validated by prior work or research agents)

- **BiomedCLIP ViT-B/16** is adequate for CXR patch embedding with light adapter (verified in v5.0).
- **RoBERTa-large** remains text backbone (verified in v5.0).
- **PadChest-GR RUA** approves within 48 h based on 2025 timelines reported in arXiv:2411.05085.
- **MAIRA-2 on H100** runs at ~45 tok/s; 500 reports at ~200 tok/report = 37 min ≈ $2.50 (verified by RRG-models research agent).
- **GREEN-RadLlama2-7b on H100** processes 1000 pairs in ~25 min ≈ $2–3 (verified).
- **RadFact on Claude Opus 4.7** costs ~$50/1000 claims at 2 LLM calls/claim (verified).
- **CheXagent-2-3b** needs `trust_remote_code=True`; fixable in one session (verified).

### 2.4 Assumption risks (gated in Plan §6)

- PadChest-GR RUA could take >48 h (Day 5 gate in Plan).
- Silver-label κ with PadChest-GR could be <0.4 (Day 11 gate).
- Cross-site LOO could still collapse entirely — in which case the paper frames this as the main honest finding (acceptable, not fatal).

---

## 3. Scientific thesis: evidence-blindness (diagnostic-first reframe)

### 3.1 Formal definitions

Let `f: (x_img, x_claim, x_evidence) → y ∈ {SUPPORTED, CONTRADICTED}` be a multimodal verifier. The three counterfactual diagnostics:

- **Image-masking gap** `IMG(f) = acc(f, D) − acc(f, D^{img=∅})` where `D^{img=∅}` zeros every image.
- **Evidence-shuffling gap** `ESG(f) = acc(f, D) − acc(f, D^{evid=π})` where `π` is a random permutation of evidence across claims.
- **Image-perturbation gap** `IPG(f) = acc(f, D_lat) − acc(f, D_lat^{img=τ})` where `τ` is a horizontal flip and `D_lat` is the laterality-sensitive claim subset.

Evidence-blind if `IMG < 5pp` or `ESG < 5pp`. Spatially-blind if `IPG < 3pp` on `D_lat`.

Thresholds calibrated via two control models:
- **Fully image-blind**: RoBERTa-only, target IMG ≤ 1pp.
- **Fully image-using**: ViT-only, target IMG ≥ 30pp.

Sensitivity reported over thresholds ∈ {3, 5, 7, 10}.

### 3.2 Relation to prior work

- **Gururangan et al. 2018 (NAACL-HLT)** — hypothesis-only baseline for NLI. Direct analogue of IMG.
- **Poliak et al. 2018 (*SEM)** — partial-input diagnostics across 10 NLI datasets. Direct ancestor of our framework.
- **HEAL-MedVQA (Nguyen et al., IJCAI 2025, arXiv:2505.00744)** — concurrent work proposing Textual Perturbation Test (TPT) and Visual Perturbation Test (VPT) + **LobA (Localize-before-Answer)** training-time mitigation for medical VQA. LobA = segmentation training on anatomical masks + attention reweighting at inference + contrastive decoding. TPT/VPT are binary Yes/No entity-swap tests strictly within VQA. **Credit**: HEAL-MedVQA independently proposed a diagnostic+mitigation framework for medical multimodal grounding. **We differentiate cleanly** (post-verification 2026-04-19): (a) **task**: claim verification vs VQA binary questions — different input/output structure; (b) **diagnostic construction**: image-masking distributional gap (IMG) measures reliance on image presence, vs entity-swap Yes→No rate (VPT) measures reliance on specific visual content — different failure modes; (c) **mitigation**: consistency loss + adversarial HO filter on full-image-masking, vs segmentation + attention reweighting on localized regions — different architectural assumptions; (d) **scope**: our paper treats verification under conformal FDR control, which is absent in HEAL-MedVQA; (e) HEAL-MedVQA explicitly does NOT claim generalization beyond VQA, so our extension to verification is novel contribution, not follow-up. Cited prominently in §2 (motivation) and §9 (related work).
- **Mohri & Hashimoto 2024 (ICML)** — conformal factuality guarantees for LMs via entailment sets. Our conformal FDR layer composes with their framework.
- **ReXTrust (Hardy et al., AAAI Bridge 2025)** — white-box hidden-state probe. **Day-0 verification result**: no public code or weights released. We therefore **reimplement** the probe from the paper's description and apply to both **MedVersa** (weights public at `hyzhou/MedVersa`) and **MAIRA-2**. Results row labeled "White-box hidden-state probe (inspired by ReXTrust; reimplemented)." This is honest framing: not a replication claim, but a good-faith reimplementation with two LVLM backbones providing an architecture-independence signal.
- **RadFlag (Zhang et al., ML4H 2024)** — black-box temperature-sampling. Replicated on MAIRA-2.

### 3.3 Why this is a NeurIPS E&D-shaped contribution

- **Evaluation methodology**: core deliverable is a diagnostic framework + benchmark, not a model.
- **Broad implications**: diagnostic is architecture-agnostic; applicable to any multimodal verifier.
- **Field-level finding**: prevalence result ("seven public detectors are evidence-blind") is community-facing, not method-advancing.
- **Conformal guarantee**: Jin–Candès 2023 composition gives a formal claim-level FDR bound when the method passes diagnostic.
- **Honest limitation**: cross-site transfer gap is reported, not hidden; matches the track's maturity expectations.

---

## 4. Data: ClaimGuard-GroundBench v6

### 4.1 Training sites (three, all public, no credentialing)

| Site | Images | Country | Lang | Reports? | Grounding | License | Role |
|---|---|---|---|---|---|---|---|
| **OpenI / Indiana U.** | 3,996 | US | EN | yes (3,955) | some bboxes | NLM / CC-BY-NC-ND | existing — training + cross-site eval |
| **ChestX-Det10** | 3,543 | US (NIH subset) | — | no | pixel masks (10 classes) | Apache-2.0 on annotations | existing — synthesized claims + grounding |
| **PadChest-GR** | 4,555 studies | Spain | ES→EN | yes (bilingual) | **radiologist-placed bboxes per finding sentence** | BIMCV RUA (registration) | **new — third-site training + silver-label validation ground truth** |

Training corpus (approximate): ~30k labeled claims after assembly. The addition of PadChest-GR brings in (a) multilingual coverage, (b) per-sentence expert grounding, (c) a different patient population and imaging equipment.

### 4.2 Evaluation sources

- **Synthetic test** (existing): 2,906 CheXbert-label-flipped claims from OpenI + ChestX-Det10. Used for the headline evidence-blindness diagnostic and conformal FDR.
- **Real-RRG test** (new, v6.0): 500 OpenI test studies run through 3 RRG models → ~6,000 atomic claims after decomposition → silver-labeled via GREEN + RadFact ensemble. Used for real-hallucination precision/recall.
- **PadChest-GR silver-validation set** (new): 7,037 positive finding sentences + sampled absent-findings as natural negatives. Used for silver-label κ vs radiologist ground truth.

### 4.3 Excluded (out of scope)

All PhysioNet-credentialed sources: MIMIC-CXR, MS-CXR, BRAX, CheXmask, RadGraph-XL, ReXVal, Chest ImaGenome, CANDID-PTX (now credentialed), VinDr-CXR (PhysioNet version).

Also deferred:
- **CheXpert Plus** (350 GB; too slow to download + integrate in 17 days; insurance only)
- **ReXGradient-160K** (156 GB, HF-gated with unpredictable approval time)
- **NIH ChestX-ray14** (no reports, low marginal value given PadChest-GR adds reports)

### 4.4 Data pipeline (see §4.2 of v5.0 doc for details; unchanged structure)

New additions for v6.0:
- `v5/data/padchest_gr.py` — loader, Spanish→English translation (Claude Haiku), per-sentence bbox → claim alignment
- Synthesis consolidation: existing `v5/data/claim_synthesizer.py` unchanged
- Assembly: `v5/data/groundbench.py` gets a third site branch

### 4.5 Splits

Patient-stratified, per-site balanced:
- train 70%, val 10%, cal 10%, test 10% (unchanged from v5.0)
- 3-way leave-one-site-out: held-out site contributes all of its test claims; training combines the other two sites' train+val+cal

---

## 5. Model architecture

### 5.1 Frozen from v5.0

The model architecture is unchanged from v5.0 (see `ARCHITECTURE_V5_0_EVIDENCE_BLINDNESS.md` §5). Summary:
- BiomedCLIP ViT-B/16 image encoder (top 4 of 12 blocks unfrozen)
- RoBERTa-large text encoder (top 8 of 24 layers unfrozen)
- Domain-adapter MLP (2-layer, 768→768→768)
- 4-layer bidirectional cross-modal fusion transformer (d=768, heads=12)
- Three heads: verdict (2-way CE), support-score (sigmoid, for conformal), grounding (14×14 patch mask)
- MC-dropout for uncertainty (MC-ECE loss, activated v5.4 only)

**No architectural changes in v6.0.** The 17-day runway does not permit architectural experimentation, and v5.3's accuracy + IMG are already sufficient for the paper's story. All gains in v6.0 come from data expansion + evaluation breadth + diagnostic reframe.

### 5.2 The one exception

For the real-RRG evaluation path, the model is used at inference only. No retraining on RRG-generated claims (that would be cheating — using the test distribution for training).

---

## 6. Training

### 6.1 Frozen from v5.0

Loss, optimizer, schedule, data augmentation — all unchanged. See `ARCHITECTURE_V5_0_EVIDENCE_BLINDNESS.md` §6 for the 5-term composite loss and HO-filter protocol.

### 6.2 New in v6.0

- **v6.0-3site**: retrain v5.3-contrast configuration on {OpenI, ChestX-Det10, PadChest-GR}. This is "the method" in v6.0 results tables.
- **v6.0-LOO-{openi, chestx, padchest}**: three leave-one-out runs, each trained on two of three sites and evaluated on the held-out third. Used for cross-site transfer analysis.

All other training variants (v5.0, v5.1, v5.2, v5.3, v5.4 on OpenI+ChestX-Det10) are kept as historical rows in the appendix table.

---

## 7. Evidence-blindness diagnostic (central contribution)

### 7.1 Protocol unchanged from v5.0 §7

Identical IMG/ESG/IPG computation; identical threshold calibration.

### 7.2 New in v6.0: applied to real-RRG-generated claims

Applied to 500-claim RRG test set (across 3 RRG models). For each detector:
- acc on full input
- acc with image zeroed (for local models) or replaced with solid gray (for API/VLM baselines)
- acc with evidence shuffled within batch
- acc with horizontal flip on laterality-sensitive subset

Reported per detector × RRG source (3 sources) × condition (4 conditions) = 12 numbers per detector × 7 detectors = 84 measurements for Table 1.

### 7.3 New in v6.0: diagnostic on replicated detectors

Explicitly run the diagnostic on **ReXTrust replication** and **RadFlag replication**. This is the "prevalence" argument: if even published 2024–2025 CXR hallucination detectors are evidence-blind under our diagnostic, the finding is field-level.

---

## 8. Evaluation protocol

### 8.1 Baseline landscape (seven detectors)

| Detector | Access | Role | Modal cost |
|---|---|---|---|
| **CheXagent-2-3b** (Stanford, MIT) | local H100 | Medical VLM, as zero-shot verifier | ~$2 |
| **MAIRA-2** (Microsoft, MSRLA) | local H100 | Medical VLM, as zero-shot verifier | ~$3 |
| **MedGemma-4B-IT** (Google, HAI-DEF) | local H100 | Medical VLM, as zero-shot verifier | ~$2 |
| **BiomedCLIP zero-shot** (Microsoft) | local H100 | image-CLIP baseline (with fixed prompt template) | ~$1 |
| **RadFlag replication** | local H100 | Black-box temperature-sampling detector (Zhang et al. 2024) | ~$25 |
| **Hidden-state probe on MedVersa** (ReXTrust-inspired, reimplemented) | local H100 | Reimplemented per Hardy et al. 2025 (no public code); MedVersa weights at `hyzhou/MedVersa` | ~$20 |
| **Hidden-state probe on MAIRA-2** (same recipe, second backbone) | local H100 | Architecture-independence check for the ReXTrust-inspired probe | ~$15 |
| **ClaimGuard v6.0 (ours)** | local H100 | Trained verifier | in training budget |

Each evaluated in 4 conditions (full + IMG + ESG + IPG) on both synthetic test (2,906 claims) and real-RRG test (6,000 claims).

### 8.2 Real-RRG evaluation pipeline

1. Take 500 OpenI test studies (images + ground-truth reports).
2. Run 3 RRGs on each: MAIRA-2, CheXagent-2-3b, MedGemma-4B-IT → 1,500 generated reports.
3. Decompose via LLM claim extractor (Claude Haiku, existing in `v5/data/claim_extractor.py`) → ~6,000 atomic claims.
4. Silver-label each claim via GREEN + RadFact ensemble (§8.3).
5. Route disagreements (Krippendorff α < some threshold per-claim) to UNCERTAIN; exclude from headline P/R.
6. Evaluate each of 7 detectors on the silver-labeled set; report precision/recall/F1 for hallucination detection.

### 8.3 Silver labeling (three-grader ensemble, MIMIC-leakage-aware)

**Grader 1: GREEN-RadLlama2-7b** — EMNLP Findings 2024; τ=0.63 with radiologists on ReXVal; runs locally on H100 at ~$3/1000 claims. **Important caveat**: GREEN was *trained* on MIMIC-CXR pairs, so when labeling outputs from MIMIC-pretrained RRG models (MAIRA-2, LLaVA-Rad) there is a partial distributional correlation. Inference does not require MIMIC access, but label validity on MIMIC-trained RRGs has a known confound.

**Grader 2: RadFact** — Microsoft RadFact framework from MAIRA-2 technical report (arXiv:2406.04449; github.com/microsoft/RadFact; MIT). Decomposes reports into atomic phrases + bidirectional entailment. We use **Claude Haiku for decomposition** (~1,500 calls, cheap) and **Claude Opus 4.7 for entailment** (~4,000 calls, τ=0.46 on RaTE-Eval per VERT benchmark). ~$120/1,000 claims.

**Grader 3 (NEW per pre-flight B2): VERT** (Tang et al., arXiv:2604.03376, April 2026) — structured-prompt LLM judge with Claude Opus 4.7 backbone; τ=0.4633 on RaTE-Eval, highest published. **Critical property**: VERT is **MIMIC-free at both training and inference** (prompt-engineering framework around Claude). Adding VERT decouples the silver-label pipeline from the MIMIC-pretrained-RRG distributional correlation. ~$50/1,000 claims.

**Ensemble protocol**:
- All three graders label each claim binary + confidence.
- Pairwise + three-way Krippendorff α reported.
- Unanimous agreement → label stands.
- 2-of-3 agreement → majority label stands, confidence downgraded.
- Full disagreement → UNCERTAIN; excluded from headline precision/recall (reported separately as "uncertainty rate").
- Target: ≥60% claims unanimously labeled.

**MIMIC-leakage decoupling reporting**: §8 tables report prevalence separately for each grader. If GREEN-only and (RadFact+VERT)-only agree on the prevalence finding, MIMIC-leakage is not a confound. If they disagree, the paper frames GREEN as MIMIC-correlated and RadFact/VERT as the leakage-free evidence.

**Validation against PadChest-GR radiologist bboxes**: for the subset of evaluation where the image comes from PadChest-GR, compute silver-pipeline agreement with the PadChest-GR per-sentence bboxes. Report per-finding Cohen κ and classification metrics. Target: κ ≥ 0.5. PadChest-GR was NOT trained by any grader, so this is a clean external validation.

### 8.4 Conformal FDR protocol

Unchanged from v5.0 §8.3:
- Inverted cfBH (calibrated on CONTRADICTED claims)
- Weighted cfBH (Tibshirani 2019) with per-site weights
- Doubly-robust cfBH (Fannjiang 2024)
- α ∈ {0.05, 0.10, 0.15, 0.20}

v6.0 adds: apply conformal FDR to the real-RRG test set in addition to synthetic. Reports n_green, achieved FDR, power for each.

### 8.5 Ablations (mostly from v5.0 Tier 3, some new)

Already done (from Tier 3):
- Loss drops (5 variants)
- HO threshold sweep (5 thresholds)
- Scale curve (25/50/100%)
- Conformal FDR across configs

New in v6.0:
- **Cross-site 3-way leave-one-out** (3 runs × 1 seed)
- **Cross-site mechanism** (M3): support-score KS shift per site, HO-filter activation rate per site, text-only ceiling per site. Produces a mechanistic diagnosis figure — converts "we fail to generalize" from weakness to diagnostic contribution
- **Artifact audit** (M4 tightened): text-only RoBERTa-base on train, measured pre-HO-filter and post-HO-filter. **Target: text-only acc minus majority-class-prior acc ≤ 2pp post-HO-filter**. Pre-HO baseline (OpenI CheXbert) has majority-class prior ~70%, so "≤ chance + 5pp" is a meaningless bar. Reporting vs class-prior is the correct statistic
- **HO-filter activation on real-RRG claims** (M6, new): measure HO-filter's text-solvability score on the 6,000 real RRG-generated claims. Tests whether the mitigation transfers from synthetic negation-flipped claims (HO's training distribution) to naturally-occurring paraphrastic hallucinations. Target: ≥40% of real hallucinations flagged as HO-solvable. If <10%, the mitigation story needs reframing.
- **Support-score sharpness**: histogram of support-score distributions for v5.0–v5.3 + v6.0, with KS-distance to uniform reported per config. Explains the one-of-four conformal-active configs outcome
- **Silver-label agreement vs PadChest-GR**: κ per finding-class, pre-unanimous and post-majority-vote
- **MIMIC-leakage decoupling**: table reporting GREEN-only vs (RadFact+VERT)-only prevalence findings. If they converge, MIMIC-leakage is not a confound

### 8.6 Handling the no-radiologist constraint (defensibility)

Explicit in-paper framing:
- All ground truth comes from **existing** radiologist annotations released by source datasets (OpenI CheXbert labels, ChestX-Det10 pixel masks, PadChest-GR per-sentence bboxes).
- Silver-label pipeline agreement with PadChest-GR radiologist ground truth is reported quantitatively.
- Claim-to-annotation matcher reliability self-reviewed on 500 claims; protocol published.
- Paper scope is methodological (diagnostic framework), not clinical (deployment). Explicit in Limitations.

No recruited radiologist at any phase.

---

## 9. Compute and budget

See `PLAN_V6_17DAY_NEURIPS.md` §4 for full table.

Summary:
- Projected spend: ~$300 Modal + Anthropic API
- Budget available: $620
- Slack: $320 (53%)

GPU-hours on H100 @ ~$4/hr: approximately 50 GPU-hours total for all v6.0-specific work.

Stop-on-exhaust at $560. All buffer overruns come from the $320 slack first, never from committed line items.

---

## 10. Release artifacts (M7 license fix)

**Important**: the BIMCV Research Use Agreement that governs PadChest-GR may prohibit redistribution of derived labels. Release plan must be finalized only after Day-1 reading of the RUA fine print.

Two release modes, chosen per RUA compliance:

**Mode A — if RUA permits derived-label redistribution**:
- **Code**: GitHub `alwaniaayan6-png/ClaimGuard-CXR` tagged `v6.0-neurips-submission`, MIT license.
- **Benchmark labels**: Zenodo DOI. CC-BY-NC-SA for derived labels and annotations (inherits from most restrictive source: PadChest RUA). **No raw images redistributed.**
- **Model weights**: HuggingFace Hub, MIT-licensed, tagged `v6.0`.

**Mode B — if RUA prohibits redistribution (expected default)**:
- **Code + reproduction scripts ONLY**: GitHub. Scripts regenerate all labels from original source data after the user obtains access independently.
- **No Zenodo label release**. Paper notes: "Reproduction: following [X] steps after obtaining source datasets, all labels regenerate deterministically."
- **Model weights**: HuggingFace Hub, MIT-licensed (labels are inside the model weights too, via training — this is typically permitted, unlike raw-label redistribution).
- **Benchmark access protocol** documented in paper and repo, linking to each source dataset's access page.

Other artifacts (both modes):
- **Evaluation harness**: `v5/scripts/reproduce_v6.sh` reproduces every paper number end-to-end from a clean Modal volume after source data is obtained.
- **Docker image**: pinned Python 3.10 + CUDA 12.1, Modal-compatible.
- **Datasheet** (required for NeurIPS E&D): complete per Gebru et al. 2021 schema.
- **Reproducibility checklist** (NeurIPS standard).

MAIRA-2 (MSRLA) and MedGemma (HAI-DEF) research-only licenses are disclosed in the ethics statement. Our code calls these models via HuggingFace; we do not redistribute their weights.

---

## 11. Risks and mitigations

See `PLAN_V6_17DAY_NEURIPS.md` §5 for the full risk table (13 rows).

Top-3 by impact × probability:
1. **PadChest-GR RUA delay** (Medium × Critical): fallback to CheXpert Plus for 2-way site expansion; accept US-only cross-site story.
2. **Cross-site LOO still collapses** (Medium × Medium): report as central honest finding rather than treating as bug. Paper's positioning supports this outcome.
3. **Silver-label κ vs PadChest-GR < 0.4** (Low × High): reframe as negative result for silver-labeling pipeline; user sign-off required before reframe.

Gate conditions for stop-the-line are in Plan §6.

---

## 12. Timeline

See `PLAN_V6_17DAY_NEURIPS.md` §3 for full day-by-day schedule.

Summary:
- Days 1–2: registrations + RRG inference
- Days 3–5: silver labeling + PadChest-GR integration
- Days 6–7: retrain + cross-site LOO
- Days 8–11: baselines + artifact audit + silver validation
- Days 12–15: writing + figures + reproducibility + pre-flight review
- Day 16: buffer
- Day 17: submit

---

## 13. Success criteria

**NeurIPS 2026 E&D acceptance** requires the combination:
- Our method (v6.0-3site or v5.3): IMG ≥ 6pp, ESG ≥ 6pp on real-RRG test, within 2pp of best baseline on accuracy
- ≥5 of 7 baselines: IMG < 5pp on real-RRG test (establishing prevalence)
- Silver-label κ vs PadChest-GR ≥ 0.5
- Artifact audit: text-only post-HO-filter test acc within 7pp of chance
- Conformal FDR achieves target coverage on ≥1 config (v5.3 or v6.0)

**Minimum publishable outcome**: all of the above except the cross-site LOO may fail; workshop submission to GenAI4Health / ML4H 2026 with the same evidence base would likely accept (~60%+).

**Unacceptable outcome**: baseline replications fail, real-RRG pipeline fails, silver labels unreliable. Gate §6 in Plan catches each of these.

---

## 14. Pointers to code

| Component | File | Status |
|---|---|---|
| Model definition | `v5/model.py` | frozen |
| Training loop | `v5/train.py` | frozen |
| Losses | `v5/losses.py` | frozen |
| HO filter | `v5/ho_filter.py` | frozen |
| Conformal FDR | `v5/conformal.py` | frozen |
| Evidence-blindness diagnostic | `v5/eval/evidence_blindness.py` | frozen |
| Configs | `v5/configs/*.yaml` | frozen |
| Baselines (to fix) | `v5/eval/baselines.py` | **update Day 1** |
| RRG generation | `v5/eval/rrg_generate.py` | **new Day 2** |
| GREEN labeler | `v5/eval/green_labeler.py` | **new Day 3** |
| RadFact labeler | `v5/eval/radfact_labeler.py` | **new Day 4** |
| Silver ensemble | `v5/eval/silver_ensemble.py` | **new Day 4** |
| PadChest-GR loader | `v5/data/padchest_gr.py` | **new Day 5** |
| ReXTrust replication | `v5/eval/rextrust_replica.py` | **new Day 9** |
| RadFlag replication | `v5/eval/radflag_replica.py` | **new Day 9** |
| Artifact audit | `v5/eval/artifact_audit.py` | **new Day 10** |
| Silver vs PadChest-GR validation | `v5/eval/padchest_gr_validate.py` | **new Day 11** |
| v6 orchestrator | `v5/modal/v6_orchestrator.py` | **new Day 2** |
| Reproducibility script | `v5/scripts/reproduce_v6.sh` | **new Day 15** |

---

## 15. Change log

| Date | Version | Author | Change |
|---|---|---|---|
| 2026-04-17 | v5.0 | Aayan Alwani | Initial v5.0 spec (evidence-blindness, 9-month runway, NeurIPS 2027) |
| 2026-04-19 | v6.0 (initial) | Aayan Alwani | NeurIPS 2026 E&D sprint. Diagnostic-first reframe. Added PadChest-GR (third site). Added real-RRG evaluation pipeline via MAIRA-2 + CheXagent-2 + MedGemma-4B. Silver-labeling ensemble with PadChest-GR validation. Cross-site 3-way LOO. 17-day timeline. Companion plan doc |
| 2026-04-19 | v6.0 (pre-flight-patched) | Aayan Alwani | Folded Opus pre-flight reviewer fixes (5 blocking + 7 major). Key changes: (B1) RadFact budget $50→$120 per accurate call model; (B2) added VERT as MIMIC-free third silver grader, decoupling MIMIC-leakage confound; (B3) ReXTrust replication on MedVersa (published backbone), separate MAIRA-2 probe row; (B4) hardened PadChest-GR RUA fallback (CheXpert Plus, NIH-14+Object-CXR); (M2) tightened HEAL-MedVQA differentiation with ACT credit; (M3) added cross-site mechanism analysis; (M4) artifact-audit target vs majority-class baseline not chance; (M6) added HO-filter-on-real-RRG transfer check; (M7) dual-mode release plan (labels vs reproduction scripts) pending RUA fine-print read |
| 2026-04-19 | v6.0 (post-verification) | Aayan Alwani | Day-0 verifications on external claims: (1) ReXTrust has NO public code/weights — `rajpurkarlab` GitHub + HF both checked, not present. Changed B3 from "replicate on MedVersa as published" to "reimplement per paper description, apply to MedVersa AND MAIRA-2" with honest row label. (2) HEAL-MedVQA's training mitigation is **LobA** (Localize-before-Answer = segmentation training + attention reweighting + contrastive decoding), NOT ACT as pre-flight reviewer stated. Corrected §3.2 differentiation. Our differentiation is cleaner than expected: different task (VQA vs verification), different diagnostic construction (entity-swap vs image-masking), different mitigation architecture (segmentation vs consistency-loss). (3) MedVersa weights confirmed public at `hyzhou/MedVersa` with non-standard `registry.get_model_class('medomni')` loader. (4) PadChest-GR RUA URL confirmed at `bimcv.cipf.es/bimcv-projects/padchest-gr/`; 46 GB; no published turnaround time; no institutional-email requirement observed |
