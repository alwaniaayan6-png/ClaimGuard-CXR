# Path B — Status after scaffold session (2026-04-17)

**Commit:** 874f8ae — `path_b: image-grounded claim-verifier scaffold (D31-D34)`, local on `main`, unpushed.

## What shipped this session

### Design
- `ARCHITECTURE_PATH_B.md` v2 — post-feasibility-review. Venue target shifted to npj Digital Medicine / Medical Image Analysis primary; Nature MI only if radiology co-author + PhysioNet credentialing both land.
- `decisions.md` D31–D34 — Path B pivot, dataset scoping, four-way contrastive loss design, 6-bug pre-flight catch.
- `CLAIMGUARD_PROPOSAL.md` header — notes Path B supersedes v1 text-only design.

### Code (`claimguard_nmi/`)
- `configs/ontology.yaml` — 13 canonical findings across English + Spanish aliases; ungroundable-claim-type list; diffuse-finding overrides.
- `configs/datasets.yaml` — per-site access / schema / role; credentialed-bonus section explicitly separated.
- `configs/model.yaml` — BiomedCLIP + RoBERTa + cross-attn + 3 heads, pinned HF revisions, 3-seed ensemble, adversarial HO filter config.
- `grounding/claim_schema.py` — frozen `Claim` dataclass with Enum fields.
- `grounding/grounding.py` — `ground_claim()` with IoU@0.5, soft region prior, negation-aware `GROUNDED_ABSENT_OK`/`_FAIL` split, diffuse-finding size-ratio branch, dataset-schema gate.
- `models/image_grounded_verifier.py` — BiomedCLIP ViT-B/16 vision tower with last-N unfrozen (raises on path-not-found), shared RoBERTa-large text tower, 4-layer bidirectional cross-attention, 3 heads (verdict / score / MC-dropout uncertainty). Text projection built unconditionally in `__init__`.
- `training/contrastive_loss.py` — 4-way loss `L = CE(V1,V2,V3) + λ_evi·hinge + λ_img·hinge + λ_mask·CE(V4)`. V4 no longer double-counted.
- `conformal/inverted_cfbh.py` — one-sided conformal p-values calibrated on contradicted class; BH with empirical FDR / power audit.
- `conformal/weighted_cfbh.py` — Tibshirani 2019 weighted cfBH with gradient-boosted density ratio + isotonic calibration + ESS diagnostic + degenerate-weight fallback flag. Weights precomputed once.
- `eval/metrics.py` — bootstrap CIs via paired resampling across arbitrary aligned arrays; DeLong + McNemar + accuracy + recall + AUROC; AUROC fallback fixed (no longer returns 1−AUROC).
- `eval/runner.py` — multi-site runner skeleton with clear TODO on the inference stub.
- `data/ontology_mapping.py` — surface-form → canonical id resolver, substring fallback, collision detection.
- `scripts/path_b/download_datasets.sh` — public-only download driver (Kaggle CLI + GitHub + BIMCV instructions).

### Tests (`claimguard_nmi/tests/`) — 30/30 passing
- `test_grounding.py` — 10 tests: ungroundable types, schema gating, absent/present claim branches, laterality inversion, soft-prior shape, weighted IoU edge cases.
- `test_conformal.py` — 4 tests: inverted-cfBH FDR control, power, small-cal-set vacuous acceptance, weighted variant.
- `test_contrastive_loss.py` — 4 tests: perfect-model low loss, text-only model fails `img_margin` (HO-shortcut defense), V4 weighting matches `λ_mask` exactly (regression for the double-count bug).
- `test_metrics.py` — 6 tests: bootstrap pairs arrays correctly, detects random pairing, AUROC polarity, mismatched-length rejection, point inside CI.
- `test_ontology_mapping.py` — 5 tests: English / Spanish alias resolution, unknown-string fallback, diffuse flag, ungroundable-claim-type flag.

## What this session did NOT do

- No PhysioNet credentialing filed (requires Laughney signature).
- No datasets downloaded (requires Kaggle token, BIMCV approval, Stanford AIMI creds).
- No GPU runs launched (requires credit top-up + pre-flight review + compute budget authorization).
- No model weights trained.
- No Claude API calls (credits exhausted; Path B labeling switches to GPT-4o + Llama-3.1-405B until topped up).
- No git push (standing rule — user must approve).

## Hard blockers requiring user action

| # | Action | Who | Required before |
|---|---|---|---|
| B1 | Approach Laughney with Path B scope + request PI authorship + Modal budget lift to $2,500 | User | Any Modal GPU run |
| B2 | Weill Cornell IRB exempt-determination letter (Laughney as PI) | User | Benchmark release |
| B3 | Kaggle API token + Stanford AIMI re-auth | User | `scripts/path_b/download_datasets.sh` |
| B4 | BIMCV data-use request for PadChest-GR | User | PadChest-GR ingest |
| B5 | Recruit biostatistician co-author (Weill Cornell Biostats Core consult) | User | Submission |
| B6 | (Optional, NMI-stretch) Radiology consultant / co-author intro from Laughney | User | NMI aim-up |
| B7 | (Optional, NMI-stretch) PhysioNet credentialing so MS-CXR/ReXVal/ReXErr become supplementary | Laughney | Supplementary NMI validation |
| B8 | Anthropic credit top-up | User | Any Claude API call in the labeling ensemble |
| B9 | Approval to `git push origin main` | User | Remote commit history |

## Next agent actions when unblocked

1. Once Laughney signs off on scope + budget — file the Weill Cornell IRB letter draft.
2. Once Kaggle + BIMCV tokens provided — run `download_datasets.sh`, verify each dataset loads into the unified schema.
3. Implement the dataset-specific loaders in `claimguard_nmi/data/loaders/` (one file per dataset, all returning the same `{image_id, patient_id, image_path, annotations, split}` schema).
4. Validate the claim extractor on RadGraph-XL-free-text — target ≥0.95 claim precision before running at scale.
5. Write the training script that consumes the 4-way variants, runs adversarial HO filter 2×, logs MLflow runs with pinned code SHA + data hash + seed.
6. Pre-flight-review the training script before the first GPU launch.
7. Launch first image-grounded verifier training (3 seeds, $50 budget cap) — evaluate HO gap, image-swap gap, and PadChest-GR weighted-κ to clear G2.

## Risks to flag at next handoff

- **Claim extractor quality.** Every downstream result depends on this; if extractor precision is <0.95 we need a new one before any headline numbers.
- **PadChest-GR BIMCV approval latency.** No known SLA; worst case 4–8 weeks. Fallback: RSNA + SIIM + ChestX-Det10 alone are sufficient for a minimum viable headline (3 primary sites instead of 4).
- **BiomedCLIP contamination vs CheXpert.** Run pHash / CLIP-similarity scan of PMC-OA subset against CheXpert test set. If >1% overlap, switch to a from-scratch image encoder or use RadDINO-v2 (and accept the MIMIC pretraining note as a limitation).
- **Compute budget.** Full Path B sweep estimated $500–$1200 depending on retraining rounds. Current $800 remaining is tight.
- **Four-way contrastive loss hyperparameter sensitivity.** λ_img and λ_evi weights are hypothesized from v1 heuristics; may need Optuna sweep on a 10% subset before committing to full training.

## Tests status

    $ python3 -m pytest claimguard_nmi/tests/ -q
    ..............................                                           [100%]
    30 passed in 2.29s

Every module the paper depends on has at least one unit test. No integration test yet — those come online once `download_datasets.sh` has produced real data.
