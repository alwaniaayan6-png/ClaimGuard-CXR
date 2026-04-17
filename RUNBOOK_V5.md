# RUNBOOK — ClaimGuard-CXR v5 (image-grounded)

Companion to `ARCHITECTURE_V5_IMAGE_GROUNDED.md`. Phased, literal commands. Every
phase has a gate: do not proceed past a gate until its criterion is met.

> **Working directory for all Modal commands:** `/Users/aayanalwani/VeriFact/verifact`
> **All `modal run` invocations must be prefixed with `cd /Users/aayanalwani/VeriFact/verifact &&`** — Modal looks at the cwd for relative imports.

---

## Phase 0 — Governance and infrastructure

### 0.1 Confirm budget cap
```bash
# Budget cap stays at $900 (already set on Modal dashboard).
# Do NOT increase — all phases are scoped to fit within $900 total.
```

### 0.2 IRB exempt determination (Weill Cornell)
Draft filed by Laughney. Retrospective analysis of de-identified public data.
Reference the determination letter number in the manuscript.

### 0.3 OSF preregistration
Draft endpoints, sample sizes, analysis plan, fallback ladders. Lock before
starting §4 (GroundBench construction). Link OSF id in manuscript.

### 0.4 Repo hygiene
```bash
cd /Users/aayanalwani/VeriFact/verifact
git checkout -b v5-image-grounded
git add ARCHITECTURE_V5_IMAGE_GROUNDED.md RUNBOOK_V5.md v5/
git commit -m "feat(v5): image-grounded arch doc + full v5 module scaffold"
# push requires user approval per CLAUDE.md
```

### 0.5 Secrets on Modal
Already present: `anthropic`. Also needed for v5:
```bash
modal secret create openai OPENAI_API_KEY=<...>
modal secret create huggingface HF_TOKEN=<...>   # for MAIRA-2, MedGemma access
modal secret create wandb WANDB_API_KEY=<...>
```

### 0.6 Unit tests (no network)
```bash
cd /Users/aayanalwani/VeriFact/verifact
python -m pytest v5/tests/ -x -k "not smoke and not data" -q
```
Expected: parser, matcher, conformal, loss, PII, anatomy-geometry tests pass
with no network or GPU dependency.

**Gate G0:** Budget lift approved, IRB filed, OSF preregistered, secrets set,
unit tests green.

---

## Phase 1 — Data acquisition

Each dataset lands in `/data/<site>/` on the `claimguard-data` Modal volume.

### 1.1 Already available (via Laughney lab)
- CheXpert Plus: `/data/chexpert_plus/`
- OpenI: `/data/openi/`

### 1.2 Downloadable (no credentialing)

MS-CXR (HuggingFace):
```bash
cd /Users/aayanalwani/VeriFact/verifact
python scripts/download_ms_cxr.py --out /Users/aayanalwani/data/ms_cxr
# Then rsync to Modal volume:
modal volume put claimguard-data /Users/aayanalwani/data/ms_cxr /ms_cxr
```

RSNA Pneumonia (Kaggle):
```bash
kaggle competitions download -c rsna-pneumonia-detection-challenge
unzip -d /Users/aayanalwani/data/rsna_pneumonia rsna-pneumonia-detection-challenge.zip
modal volume put claimguard-data /Users/aayanalwani/data/rsna_pneumonia /rsna_pneumonia
```

SIIM-ACR Pneumothorax (Kaggle):
```bash
kaggle competitions download -c siim-acr-pneumothorax-segmentation
unzip -d /Users/aayanalwani/data/siim_acr_pneumothorax siim-acr-pneumothorax-segmentation.zip
modal volume put claimguard-data /Users/aayanalwani/data/siim_acr_pneumothorax /siim_acr_pneumothorax
```

Object-CXR (Github / JF Healthcare):
```bash
git clone https://github.com/jfhealthcare/object-CXR /Users/aayanalwani/data/object_cxr
modal volume put claimguard-data /Users/aayanalwani/data/object_cxr /object_cxr
```

ChestX-Det10 (Github):
```bash
git clone https://github.com/Deepwise-AILab/ChestX-Det10-Dataset /Users/aayanalwani/data/chestx_det10
modal volume put claimguard-data /Users/aayanalwani/data/chestx_det10 /chestx_det10
```

NIH ChestX-ray14 (NIH box share):
```bash
# Follow instructions at https://nihcc.app.box.com/v/ChestXray-NIHCC
# Download images_001.tar.gz through images_012.tar.gz; unpack to /Users/aayanalwani/data/nih_cxr14
modal volume put claimguard-data /Users/aayanalwani/data/nih_cxr14 /nih_cxr14
```

### 1.3 Lightweight registrations (not credentialing)

PadChest (BIMCV): register at https://bimcv.cipf.es/bimcv-projects/padchest/ with
institutional email (Laughney lab affiliation). Download:
```bash
# follow BIMCV instructions once approved
modal volume put claimguard-data /Users/aayanalwani/data/padchest /padchest
```

BRAX (IEEE DataPort): register at https://ieee-dataport.org/open-access/brax-brazilian-labeled-chest-x-ray-dataset
```bash
modal volume put claimguard-data /Users/aayanalwani/data/brax /brax
```

**Gate G1:** All non-credentialed datasets in `/data/` on the Modal volume;
`modal volume ls claimguard-data` lists the expected sites.

---

## Phase 2 — Anatomy masks

Precompute anatomy masks for every image we will evaluate on.

```bash
cd /Users/aayanalwani/VeriFact/verifact
# For each site:
modal run --detach v5/modal/build_anatomy_masks.py::anatomy_entrypoint \
    --site chexpert_plus --limit 0
modal run --detach v5/modal/build_anatomy_masks.py::anatomy_entrypoint \
    --site openi --limit 0
modal run --detach v5/modal/build_anatomy_masks.py::anatomy_entrypoint \
    --site padchest --limit 0
modal run --detach v5/modal/build_anatomy_masks.py::anatomy_entrypoint \
    --site brax --limit 0
modal run --detach v5/modal/build_anatomy_masks.py::anatomy_entrypoint \
    --site ms_cxr --limit 0
```

Verify coverage:
```bash
modal run --detach v5/modal/build_anatomy_masks.py::anatomy_entrypoint \
    --site rsna_pneumonia --limit 5000
```

**Gate G2:** anatomy mask count ≥ 95% of images for CheXpert Plus.

---

## Phase 3 — VLM report generation

Run the five VLM generators over each site's images.

```bash
cd /Users/aayanalwani/VeriFact/verifact
# 3 generators × 3 sites × 2 temps = 18 jobs (budget-safe at $120 cap)
for gen in chexagent-8b maira-2 medgemma-4b ; do
    for site in chexpert_plus openi rsna_pneumonia ; do
        modal run --detach v5/modal/run_vlms.py::run_vlms_entrypoint \
            --generator "$gen" --site "$site" --limit 0 \
            --temperatures 0.3,1.0 --seeds 101,202
    done
done
```

Budget watch: check `python v5/scripts/budget_daily.py --phase 3` before
each batch. Kill immediately if spend hits $132 (110% of cap).

Note: llava-rad and llama-3.2-11b-vision held as ablation-only; add only
if phase 3 spend < $80 after the 18 primary jobs complete.

**Gate G3:** each (site × generator × temperature × seed) has a JSONL under
`/data/vlm_reports/<site>/<generator>/t<t>_s<s>.jsonl` with ≥ 80% success rate.

---

## Phase 4 — GroundBench assembly

Combine images, radiologist annotations, anatomy masks, oracle reports, and
structured claims into ClaimGuard-GroundBench.

### 4.1 Rule-extractor dry run (cheap)
```bash
cd /Users/aayanalwani/VeriFact/verifact
modal run --detach v5/modal/build_groundbench.py::build_groundbench_entrypoint \
    --site chexpert_plus --limit 1000 --use_llm 0
```
Inspect `/data/groundbench_v5/chexpert_plus/summary.json`. Coverage should be
≥ 40% (fraction of claims with non-NO_GT).

### 4.2 LLM-extractor full run (primary)
Budget-cap per site: ~$30. Proceed site-by-site:
```bash
for site in chexpert_plus openi padchest brax ms_cxr rsna_pneumonia siim_acr \
    object_cxr chestx_det10 ; do
    modal run --detach v5/modal/build_groundbench.py::build_groundbench_entrypoint \
        --site "$site" --limit 0 --use_llm 1
done
```

### 4.3 Merge per-site splits into a unified manifest
```bash
modal run --detach v5/modal/build_groundbench.py::build_groundbench_entrypoint \
    --site all --limit 0 --use_llm 0
# (Manual step: concatenate per-site train/cal/test JSONLs into a global one
# at /data/groundbench_v5/all/.)
```

**Gate G4:** `summary.json` per site shows coverage ≥ 40%; labeler κ ≥ 0.70 on
categories kept; PII-scrubbed text in `evidence_text` field.

---

## Phase 5 — Pre-flight review before training

Per CLAUDE.md Pre-flight Review Rule. Spawn an Opus logic-review agent:
```
Files to review:
  - v5/model.py
  - v5/train.py
  - v5/losses.py
  - v5/data/claim_matcher.py
  - v5/data/groundbench.py
  - v5/modal/train_v5.py

Worries:
  1) loss weights mis-weighted or reduction wrong
  2) grounding mask alignment to image patches off-by-one
  3) contrastive pairs leaking correct evidence
  4) MC-dropout not active during uncertainty estimation
  5) data-loader label leakage across splits
  6) BiomedCLIP image normalization stats mismatch
  7) tokenizer max-length truncating claim before evidence
  8) freeze_layers indices mis-indexed for BiomedCLIP ViT

Expected cost: best $5 / expected $15 / worst $40 per training run.

Verdict required: ready-to-launch / launch-with-fixes / do-not-launch.
```

**Do not launch** until verdict is ready-to-launch or fixes applied to DO-NOT-
LAUNCH verdict.

---

## Phase 6 — Training

### 6.1 v5.0-base sanity (2 epochs, classification only)
```bash
cd /Users/aayanalwani/VeriFact/verifact
modal run --detach v5/modal/train_v5.py::train_v5_entrypoint --config v5_0_base
```
Expected: val_acc ≥ v4 parity (~0.95). If below, check data loader.

### 6.2 v5.1-ground (+ grounding supervision)
```bash
modal run --detach v5/modal/train_v5.py::train_v5_entrypoint --config v5_1_ground
```
Gate: grounding IoU on MS-CXR val ≥ 0.30.

### 6.3 v5.2-real (+ real VLM hallucinations + consistency + HO adversarial filter)
```bash
# First run the HO adversarial filter (trains N HO baselines and filters training data):
modal run --detach v5/modal/adversarial_ho_filter.py::filter_entrypoint --n_seeds 3
# Then train:
modal run --detach v5/modal/train_v5.py::train_v5_entrypoint --config v5_2_real
```
Gate: HO gap ≥ 10 pp AND image-masked degradation ≥ 15 pp by epoch 3.

### 6.4 v5.3-contrast (+ contrastive evidence loss)
```bash
modal run --detach v5/modal/train_v5.py::train_v5_entrypoint --config v5_3_contrast
```
Gate: both gaps improve by ≥ 3 pp vs v5.2.

### 6.5 v5.4-final (3 seeds, MC-dropout uncertainty head)
```bash
for seed in 17 41 67 ; do
    SEED=$seed modal run --detach v5/modal/train_v5.py::train_v5_entrypoint \
        --config v5_4_final
done
```
Report mean ± 95% CI on headline numbers. 3 seeds is sufficient for
a Nature MI submission; add seeds 29 and 53 only if variance > 2 pp.

**Gate G6:** v5.4-final mean val-acc ≥ 0.90 with 3-seed variance ≤ 2 pp.

### 6.6 Contingency if HO gap fails
Apply `v5_3_contrast`'s adversarial filter at stronger threshold (0.85 →
0.75). If still fails, reframe paper per arch-doc §2.3 contingency.

---

## Phase 7 — Evaluation

### 7.1 Per-site image-grounded eval
```bash
cd /Users/aayanalwani/VeriFact/verifact
for site in chexpert_plus openi padchest brax ms_cxr rsna_pneumonia siim_acr ; do
    modal run --detach v5/modal/eval_v5.py::eval_v5_entrypoint \
        --ckpt v5_4_final --site "$site"
done
```

### 7.2 Conformal variants (inverted, weighted, doubly-robust)
Already wired into `eval_v5_run`. Output `/data/checkpoints/claimguard_v5/v5_4_final/eval_<site>.json` contains all three.

### 7.3 Fairness / subgroup
Automated inside `eval.fairness` after `evaluate()` returns. Review parity gaps
and flag subgroups > 5 pp.

### 7.4 Scaled provenance experiment
```bash
modal run --detach v5/modal/run_provenance_scaled.py::provenance_entrypoint \
    --site chexpert_plus --n_images 1000 --temperatures 0.3,0.7,1.0,1.2
```
(Script composes the run from already-generated VLM outputs in Phase 3.)

### 7.5 Simulated reader study (MS-CXR)
```bash
modal run --detach v5/modal/simulated_reader_ms_cxr.py::simulated_reader \
    --ckpt v5_4_final
```

**Gate G7:** all six external sites report FDR ≤ α for α ∈ {0.05, 0.10, 0.20}
under at least one conformal variant, with 95% CI documented.

---

## Phase 8 — Release

### 8.1 Model weights → HuggingFace
```bash
# Upload v5.4-final seed-averaged checkpoint + each individual seed.
huggingface-cli upload aayanalwani/claimguard-cxr-v5 \
    /data/checkpoints/claimguard_v5/v5_4_final/best.pt \
    claimguard_v5_4_final_best.pt
```

### 8.2 Data → Zenodo
```bash
# ClaimGuard-GroundBench v1.0 assembled manifest + extracted claims + silver
# labels, with PII scrubbed.
# Create Zenodo deposit, get DOI, include in data card.
```

### 8.3 Code → GitHub release
```bash
cd /Users/aayanalwani/VeriFact/verifact
git tag -a claimguard-v5.0 -m "ClaimGuard-CXR v5.0: image-grounded release"
# push after user approval
```

### 8.4 Docker reproducibility image
```bash
cd /Users/aayanalwani/VeriFact/verifact
docker build -f v5/Dockerfile -t claimguard-cxr-v5:0.1 .
# Publish to Docker Hub / GHCR for reproducers.
```

### 8.5 Third-party reproduction
Send code + data pointers to an independent group. Target: reproducers match
headline numbers within ±1 pp.

**Gate G8:** HF weights live; Zenodo DOI minted; GitHub tag cut; reproducer
confirmed.

---

## Phase 9 — Manuscript

### 9.1 Draft structure
1. Abstract (two contributions + validation summary)
2. Introduction (claim-level hallucination problem + FDR gap + image-grounded pivot)
3. Related work (VeriFact, RadFact, RadFlag, Jin-Candes, Bates, MAIRA-2)
4. Methods (§5 arch doc + §6.x training + §7 eval)
5. Experiments (tables per §7 arch doc)
6. Discussion (limitations honest; threat model for provenance)
7. Conclusion (2 sentences)
8. Methods appendix (theoretical treatment of inverted + weighted + DR cfBH)

### 9.2 Figures
- Fig 1: architecture + provenance tiers
- Fig 2: per-site accuracy + FDR (6 sites × 4 methods) with 95% CIs
- Fig 3: HO-gap + image-masked gap across training stages (v5.0→v5.4)
- Fig 4: ablation heatmap
- Fig 5: ClaimGuard-GroundBench composition + radiologist-anchored coverage
- Fig 6: scaled provenance gate (downgrade vs temperature)

### 9.3 Internal review cycle
Laughney → biostatistician → radiology consultant (if available). Minimum 2
rounds. Freeze only after last-round sign-off.

### 9.4 Compliance checklists
- TRIPOD+AI
- STARD-AI (if diagnostic framing kept)
- Nature MI reproducibility checklist

### 9.5 Submit
Upload to Nature MI portal. Attach OSF preregistration, data/model cards,
reproducer report.

**Gate G9:** manuscript submitted.

---

## Troubleshooting

| Symptom | Check | Fix |
|---|---|---|
| `BiomedCLIP` fails to load | `transformers==4.44.2`, `open-clip-torch==2.26.1` pinned | rebuild Modal image; pin revisions |
| Grounding IoU stays near 0 | MS-CXR box alignment to 14×14 patches | verify `image_patches_side` matches encoder; check normalization |
| HO gap negative | adversarial filter too weak | lower HO-confidence threshold from 0.9 → 0.75; re-filter |
| PadChest text-parsing errors | Spanish vs English column | use `Report_EN`; run dual-translation if absent |
| Modal OOM | batch size 32 → 16; grad_accum 2 → 4 | update config |
| Conformal FDR > alpha | ESS low under weighted | fall back to un-weighted; document per-site |
| Labeler κ < 0.70 | LLM prompt version / model choice | revise prompt; drop bad categories |
| Anatomy mask coverage low | torchxrayvision unavailable on Modal image | add to image; or use midline fallback |

---

## Budget tracking

Hard cap: **$900 total** (existing Modal limit — do not increase).
Run `python v5/scripts/budget_daily.py --all` before each phase to check spend.
Auto-halt if any phase hits 110% of its cap.

| Phase | Cap (USD) | Expected | Scope reduction vs original |
|---|---:|---:|---|
| 3 (VLM gen) | 120 | 80 | 3 generators (MAIRA-2, CheXagent, MedGemma), 3 sites, 2 temps |
| 4 (GroundBench + LLM labeling) | 150 | 110 | GPT-4o-mini labeler (10× cheaper); rule-extractor primary |
| 5 (pre-flight review) | 20 | 15 | Opus agent, CPU-only |
| 6 (training, 3 seeds × 4 configs) | 540 | 480 | Drop 2 seeds, skip v5.1 standalone (fused into v5.2) |
| 7 (evaluation, 4 sites) | 50 | 35 | Core sites only: CheXpert+, OpenI, RSNA, MS-CXR |
| 8 (release) | 20 | 10 | Weights to HF, code tag — no Zenodo until paper accepted |
| **Total** | **900** | **730** | |

Per-phase escalation path: if phase 6 overruns, reduce to 2 seeds and document in manuscript.
