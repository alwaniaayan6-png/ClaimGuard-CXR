# Datasheet for ClaimGuard-GroundBench-v6

Produced per [Gebru et al., 2021](https://arxiv.org/abs/1803.09010), "Datasheets for Datasets," and the NeurIPS 2026 Evaluations & Datasets submission requirements. This datasheet describes the ClaimGuard-GroundBench-v6 benchmark released alongside the ClaimGuard-CXR v6.0 paper.

---

## Motivation

### For what purpose was the dataset created?

ClaimGuard-GroundBench-v6 is an evaluation benchmark for multimodal chest-X-ray (CXR) claim verifiers. Each row is a (chest radiograph image, atomic claim, retrieved text evidence, ground-truth label) tuple. The benchmark supports three experimental contracts:

1. **Verifier training** — 28,361 training claims spanning three public sites.
2. **Evidence-blindness diagnostics** — counterfactual metrics (image-masking gap IMG, evidence-shuffling gap ESG, image-perturbation gap IPG) that measure whether a multimodal verifier actually uses its image input.
3. **Real-hallucination evaluation** — a 2,707-claim evaluation set derived from public RRG model outputs (MAIRA-2, CheXagent-2-3b, MedGemma-4B-IT) on 500 held-out images, silver-labeled by a three-grader ensemble.

The benchmark was created to enable principled evaluation of CXR hallucination detectors without requiring PhysioNet credentialing (no MIMIC-CXR, no MS-CXR, no BRAX), and without recruiting radiologists — both practical barriers for smaller labs.

### Who created this dataset and on behalf of which entity?

Aayan Alwani (Laughney Lab, Weill Cornell Medicine). ClaimGuard-CXR is an independent research project; no external funding or institutional endorsement beyond lab affiliation.

### Who funded the creation of the dataset?

Self-funded compute ($900 total cap, Modal H100 cloud). No grant funding.

---

## Composition

### What do the instances represent?

Each instance is a tuple `(image, claim, evidence, ground_truth_label)` where:

- `image` is a frontal chest radiograph from one of three public sources.
- `claim` is a single-sentence atomic factual assertion about the image (e.g., "mild cardiomegaly is present in the left ventricle").
- `evidence` is retrieved text relevant to the claim (from radiology reports or synthesized templates).
- `ground_truth_label` ∈ {SUPPORTED, CONTRADICTED, NO_GT}.

### How many instances are there in total?

| Split | Count |
|---|---:|
| train | 28,361 |
| val | 4,046 |
| cal (for conformal calibration) | 4,046 |
| test | 4,132 |
| **Total** | **40,585** |

Additional evaluation set: 2,707 real-RRG claims extracted from 1,500 generated reports on 500 OpenI test images.

### Does the dataset contain all possible instances or is it a sample?

It is a sample. The benchmark is the union of three publicly-released CXR datasets that together span two countries (US, Spain), two languages (English, Spanish), and multiple imaging eras and scanners.

### What data does each instance consist of?

For each row:

| Field | Type | Description |
|---|---|---|
| `claim_id` | str | Deterministic hash of (report_id, sentence_index, text). |
| `image_id` | str | Source-dataset image identifier. |
| `image_path` | str | Relative path in the benchmark release. |
| `claim_text` | str | Natural-language claim (≤120 chars typical). |
| `claim_struct` | dict | Parsed: (finding, location, laterality, severity, certainty, polarity). |
| `evidence_text` | str | Retrieved or source-report evidence. |
| `evidence_source_type` | str | ∈ {`oracle_human`, `retrieval`, `synthesized`}. |
| `gt_label` | str | ∈ {SUPPORTED, CONTRADICTED, NO_GT}. |
| `grounding_bbox` | list\|null | [x1, y1, x2, y2] normalized 0–1, if pixel GT available. |
| `site` | str | ∈ {openi, chestx_det10, padchest-gr}. |
| `patient_id` | str | Hashed patient ID from source dataset. |
| `sex`, `age`, `country`, `scanner_manufacturer` | various | Demographic/technical metadata where available. |

### Is there a label or target associated with each instance?

Yes. `gt_label` ∈ {SUPPORTED, CONTRADICTED, NO_GT}. For OpenI and PadChest-GR, labels derive from radiologist-report text parsed via CheXbert-style labelers (OpenI) or radiologist-placed finding sentences (PadChest-GR). For ChestX-Det10, labels derive from the dataset's pixel-level expert annotations.

### Is any information missing from individual instances?

`grounding_bbox` is null for the majority of rows (spatial GT is available for ChestX-Det10 pixel masks, some OpenI bboxes, and all 7,037 positive-assertion PadChest-GR sentences, totaling ~24% of rows). `evidence_text` of type `oracle_human` is available only for rows originating from report-bearing sites (OpenI, PadChest-GR; ~65% of rows).

### Are relationships between individual instances made explicit?

Yes: rows sharing a `patient_id` originate from the same patient. The release includes a `patient_id → row_ids` map. Splits are patient-stratified to prevent leakage.

### Are there recommended data splits?

Yes: patient-stratified 70/10/10/10 for train/val/cal/test. MD5-hash bucketing on `patient_id`. The cal split is held disjoint from val to preserve exchangeability required for conformal FDR control.

### Are there any errors, sources of noise, or redundancies?

Known sources of noise:

1. **CheXbert labelers** (used on OpenI) have ~85–90% F1 against radiologist ground truth on the 14 CheXpert classes.
2. **Synthesized CONTRADICTED claims** (from pixel-annotated sites) are generated by template flipping, which can produce stylistically-dissimilar negatives that invite text-shortcut exploitation. This is the primary motivation for the adversarial hypothesis-only filter and for the evidence-blindness diagnostic framework.
3. **PadChest-GR report translation**: Spanish → English for sentences missing pre-translated English text, done via Claude Haiku. Translation accuracy not independently audited.

### Is the dataset self-contained, or does it link to external sources?

Mode A (if BIMCV RUA permits): Zenodo release contains derived claim-level labels, pipeline-generated claim text, reference-report mappings, and split assignments. Raw images are NOT redistributed; users must obtain images directly from the source datasets (OpenI, ChestX-Det10, PadChest-GR).

Mode B (if RUA prohibits redistribution): the repository contains code that regenerates all labels deterministically from the source datasets after users obtain access themselves.

### Does the dataset contain data that might be considered confidential?

No. All source datasets are publicly released and pre-de-identified.

### Does the dataset contain data that, if viewed directly, might be offensive, insulting, threatening, or might otherwise cause anxiety?

The radiographs show pathology (pneumonia, pneumothorax, cardiomegaly, etc.). Such images are not offensive in the clinical sense but may be unfamiliar to non-medical viewers.

---

## Collection Process

### How was the data collected?

The benchmark is a derived aggregation of three public source datasets, re-processed through the ClaimGuard-CXR pipeline (`v5/data/groundbench.py` + `v5/modal/build_groundbench.py`):

1. **OpenI** — downloaded from NLM (https://openi.nlm.nih.gov).
2. **ChestX-Det10** — downloaded from the Deepwise-AILab GitHub release.
3. **PadChest-GR** — obtained via BIMCV Research Use Agreement (https://bimcv.cipf.es/bimcv-projects/padchest-gr/).

Claim extraction: rule-based primary + Claude Haiku LLM fallback (see `v5/data/claim_extractor.py`). Claim matching to ground truth: ontology-based with IoU ≥ 0.3 threshold for spatial matching (see `v5/data/claim_matcher.py`).

### How was the data associated with each instance acquired?

For OpenI and PadChest-GR: reports + CheXbert-style labels from the source datasets.
For ChestX-Det10: pixel-level pathology annotations from the source dataset.
Claims are algorithmically extracted from reports or synthesized from annotations; extraction code is open-sourced.

### Over what timeframe was the data collected?

Source data:
- OpenI: aggregated from Indiana University radiology exams 2004–2014.
- ChestX-Det10: NIH ChestX-ray14 subset, imaging 1992–2015.
- PadChest-GR: San Juan Hospital (Valencia, Spain), 2009–2017.

Benchmark assembly: April 2026.

### What ethical review processes were conducted?

All source datasets were de-identified and approved for public release by their respective institutional review boards. No additional IRB was required for the aggregation. No human subjects interacted with the ClaimGuard pipeline.

### Were the individuals notified about the data collection?

Notification and consent are the responsibility of the source datasets' original data controllers. All three source datasets are released with their original institutions' permission for public research use.

---

## Preprocessing / Cleaning / Labeling

### Was any preprocessing/cleaning/labeling of the data done?

Extensive. See `v5/data/` module for full pipeline:

1. **Images** resized to 224×224 RGB (PIL bilinear) and anatomical regions pre-computed via `torchxrayvision` anatomy masks.
2. **Reports** parsed via `v5/data/claim_parser.py` (rule-based); fallback to `claim_extractor.llm_extract` via Claude Haiku for reports where rule parsing fails.
3. **Personally identifying information** (MRNs, dates, institution names, phone numbers, emails) scrubbed via Presidio + CXR-specific regex (`v5/data/pii_scrubber.py`).
4. **Spanish → English** translation for PadChest-GR sentences missing pre-translated English text: Claude Haiku.
5. **Ground-truth matching** via `v5/data/claim_matcher.py` with ontology lookup, laterality check, and spatial IoU threshold.

### Was the "raw" data saved in addition to the preprocessed data?

Yes — the release includes original source-dataset identifiers (`image_id`, `report_id`) so users can recover the raw data from the source hosts.

### Is the software used to preprocess the data available?

Yes. MIT-licensed code at https://github.com/alwaniaayan6-png/ClaimGuard-CXR, tag `v6.0-neurips-submission`.

---

## Uses

### Has the dataset been used for any tasks already?

Yes — the ClaimGuard-CXR v6.0 paper trains and evaluates seven detectors on this benchmark: CheXagent-2-3b, MAIRA-2, MedGemma-4B-IT as VLM-verifiers; BiomedCLIP zero-shot; a RadFlag-replicated black-box detector; a ReXTrust-inspired hidden-state probe on MedVersa and MAIRA-2; and ClaimGuard-CXR v6.0 itself (5 training configurations v5.0–v5.4 plus 3 cross-site LOO runs).

### Is there a repository that links to any or all papers or systems that use the dataset?

Yes — results tables and code at the GitHub repo above.

### What (other) tasks could the dataset be used for?

1. **Claim-level hallucination detector training** (the primary use).
2. **Evidence-blindness diagnostic development** for new multimodal medical verifiers.
3. **Conformal FDR procedure evaluation** under claim-level exchangeability.
4. **Cross-site transfer evaluation** for medical multimodal models (3 sites × 2 languages).
5. **Silver-labeling protocol benchmarking** via the PadChest-GR grounded-bbox validation subset.

### Is there anything about the composition of the dataset or the way it was collected and preprocessed that might impact future uses?

- PadChest-GR is **Spain-only**; US-dominant benchmarks will still dominate the training signal (83% US rows). Geographic generalization claims should be scoped accordingly.
- Synthesized CONTRADICTED claims inherit annotation artifacts from label-flipping (see §2 above).
- The benchmark explicitly excludes MIMIC-CXR and all PhysioNet-credentialed sources. Users needing those datasets must run additional data acquisition and cannot inherit our splits.

### Are there tasks for which the dataset should not be used?

**Clinical deployment**. This benchmark supports methodological research into the behavior of CXR hallucination detectors. It does not validate any system for clinical use. Deployment in patient care requires separate IRB-approved prospective validation with radiologist adjudication, which is out of scope for this release.

---

## Distribution

### Will the dataset be distributed to third parties?

Yes, publicly. Release details in §Release above.

### How will the dataset be distributed?

- Derived labels / split assignments / code: Zenodo DOI + GitHub release.
- Raw images: users obtain directly from source dataset hosts. The benchmark release includes source-linking scripts.

### When will the dataset be distributed?

Upon acceptance to NeurIPS 2026 Evaluations & Datasets track (or, pending acceptance, upon the first public preprint release).

### Will the dataset be distributed under a copyright or other IP license?

- **Our derived labels + pipeline code**: MIT (permissive).
- **Source-dataset content**: inherits each source's license (OpenI CC-BY-NC-ND; ChestX-Det10 Apache-2.0 annotations + NIH public domain images; PadChest-GR BIMCV RUA).

### Have any third parties imposed IP-based or other restrictions on the data associated with the instances?

Yes. BIMCV's RUA for PadChest-GR prohibits certain forms of redistribution. Release Mode B is our default: we ship regeneration scripts rather than raw PadChest-GR-derived labels if the RUA's fine print prohibits it. Final mode decided upon RUA re-read (pre-flight item M7).

### Do any export controls or other regulatory restrictions apply to the dataset or to individual instances?

No known export-control concerns for research use.

---

## Maintenance

### Who will be supporting/hosting/maintaining the dataset?

Aayan Alwani, via the GitHub repository. Issues and pull requests welcome.

### How can the owner/curator/manager of the dataset be contacted?

GitHub issues. Email: alwaniaayan6@gmail.com.

### Is there an erratum?

Will be maintained as `ERRATA.md` in the release repo.

### Will the dataset be updated?

We anticipate a `v6.1` after NeurIPS reviewer feedback lands. Long-term, any additional sites (e.g., if MIMIC-CXR or VinDr-CXR credentialing is obtained as future work) would be released as separate `v7.x` line with clear version naming.

### If the dataset relates to people, are there applicable limits on the retention of the data associated with the instances?

Source datasets' original retention policies apply to raw images. Our derived labels are keyed to source identifiers; if a source dataset is withdrawn, our derived labels for that source become unresolvable.

### Will older versions of the dataset continue to be supported/hosted/maintained?

v5.0 and v6.0 will both remain available. Breaking changes (schema, split redefinition) will bump the major version and ship alongside older versions on Zenodo.

### If others want to extend/augment/build on/contribute to the dataset, is there a mechanism for them to do so?

Yes — GitHub pull requests against the benchmark-assembly code. Contributions that add new sites should follow the existing `v5/data/<new_site>.py` loader pattern and re-run `assemble_v6_3site` (or analogous) to produce new derived-label JSONLs.
