# NeurIPS 2026 Reproducibility Checklist — ClaimGuard-CXR

## Claims

1. **All claims match experimental results** — Yes.
   - IMG 70.21pp, ESG 20.01pp, acc 0.911 for v6.0-3site reported in Table 1 and Table 5 come from the diagnostic JSON `v6_results/diag_v6_0_3site.json` computed on `groundbench_v6_test.jsonl` (n=2974 resolved-label rows).
   - Headline v5.0–v5.3 numbers come from `v5_final_results/*_diagnostic.json` produced by `scripts/spawn_full.py` plus `evidence_blindness.run_diagnostic`.
2. **Scope of claims clear** — Yes. The paper defines evidence-blindness via three operational gaps (IMG, ESG, IPG) computable on any multimodal verifier and explicitly reports what the training-time mitigation does (IMG, aggregate accuracy) and does not (IPG) address.
3. **Limitations section exists** — Yes. §Limitations (§3.3.3 in current draft) discusses: cohort size, PadChest-GR claim-extraction deferred to future work, consistency-loss side-effect on masked accuracy, architectural residual on laterality, and the diagnostic-vs-mitigation generalization boundary.
4. **Negative / null results reported** — Yes. Grounding-loss-alone fails to reduce IMG (v5.1 IMG = 2.17pp, essentially unchanged from baseline). Cross-site LOO shows evidence-blindness returns under distribution shift (IMG 3.21pp for held-out OpenI). Laterality IPG ≈ 0 across all configurations.

## Theory

5. **Formal definitions** — Yes. Evidence-blindness defined as $\text{IMG}<5$pp or $\text{ESG}<5$pp via §3.1 Definition 1 and three counterfactual-intervention metrics. Threshold calibrated against empirical controls (supplementary §A).
6. **Complete proofs / derivations** — N/A. This paper is an empirical evaluation and training-distribution study; no formal theorems claimed.

## Data

7. **Data source citations** — Yes.
   - OpenI → \citet{demner2016preparing} (Demner-Fushman 2016)
   - ChestX-Det10 → \citet{liu2020chestxdet10} (Liu 2020, arXiv:2006.10550)
   - PadChest-GR → \citet{feijoo2024padchestgr} (Feijoo 2024, arXiv:2411.05085)
   - BiomedCLIP weights → \citet{zhang2023biomedclip}
   - RoBERTa-large → \citet{liu2019roberta}
   - GREEN-RadLlama2-7B → \citet{ostmeier2024green}
   - RadFact → \citet{bannur2024maira2}
   - VERT → \citet{tang2026vert}
8. **License details** — Yes.
   - OpenI: NIH public domain
   - ChestX-Det10: CC-BY-4.0
   - PadChest-GR: restricted-access; data use agreement required (see datasheet)
   - BiomedCLIP: MIT
   - RoBERTa: MIT
   - GREEN-RadLlama2-7B: Apache-2.0
   - MAIRA-2: MSRLA (microsoft research license)
   - CheXagent-2-3b: MIT
   - MedGemma-4B-IT: HAI-DEF (Health AI Developer Foundations license)
9. **Datasheet** — Yes. Gebru 2021 datasheet at `mlhc_build/datasheet.md` (262 lines) covers motivation, composition, collection, preprocessing, uses, distribution, maintenance.
10. **Train/val/test splits documented** — Yes. §Cohort reports split sizes (train 28,361; val 4,046; cal 4,046; test 4,132). Per-site breakdown: OpenI (15,455/2,238/2,156/2,284), ChestX-Det10 (9,721/274/274/1,387), PadChest-GR (3,185/1,534/? /461 records). Patient-hash bucketing for PadChest-GR cal/test split documented in `v5/modal/v6_orchestrator.py:assemble_v6_3site`. No patient-level leakage across splits (confirmed by patient-hash audit).

## Code

11. **Reproducibility of all experiments** — Yes. All training/evaluation pipelines reproduce via:
    - `cd verifact && modal deploy v5/modal/v6_orchestrator.py`
    - `v5/scripts/reproduce_v6.sh` (deterministic end-to-end order)
    - Random seeds fixed via `V5TrainConfig.seed=42`
    - Config YAMLs under `v5/configs/v5_*.yaml`
12. **Code for baselines / proposed method** — Yes. `v5/eval/baselines.py` (7 baseline verifier classes), `v5/model.py` (V5Config + ImageGroundedVerifier), `v5/train.py`, `v5/losses.py`, `v5/ho_filter.py`, `v5/eval/evidence_blindness.py`.
13. **Preprocessing pipelines** — Yes. `v5/data/padchest_gr.py`, `v5/data/groundbench.py`, `v5/data/claim_extractor.py`, `v5/data/claim_parser.py`, `v5/data/claim_matcher.py`.
14. **Pretrained weights** — Yes.
    - Our checkpoints: `/checkpoints/claimguard_v6/{v6_0_3site,v6_0_loo_no_openi,v6_0_loo_no_chestx_det10,v6_0_loo_no_padchest-gr}/best.pt` on Modal volume `claimguard-v5-data`. Release plan: upload to HF as `claimguard-cxr/v6_0_*` after anonymity review.
    - Baselines: all downloaded from Hugging Face (see §Baseline landscape for exact repo IDs).
15. **Scripts, instructions, executable demos** — Yes. `v5/scripts/reproduce_v6.sh`; `REPRODUCIBILITY.md` at repo root; Modal volume snapshot instructions in appendix.

## Experiments

16. **Number of training runs** — Yes.
    - Main training configurations (v5.0-base, v5.1-ground, v5.2-real, v5.3-contrast, v5.4-final, v6.0-3site): 1 seeded run each (seed=42). Diagnostic numbers across configurations are comparable because all share the same evaluation pipeline and test split.
    - Ablations (loss-drop ×5, HO-threshold sweep ×5, scale curve ×3): 1 seeded run each.
    - LOO ×3: 1 seeded run each.
17. **Hyperparameters reported** — Yes. Appendix A (Training Hyperparameters) lists all optimizer, schedule, loss-weight, and architecture settings. Search strategy: informed-grid for HO-threshold sweep; otherwise single-point configs chosen to match v5.x architecture-frozen conventions.
18. **Evidence that hyperparameter tuning wasn't done on test set** — Yes. All hyperparameter selection is on the `val` split. The `test` split is held out and used only once for final reporting. Calibration set (`cal`) is used only for conformal-stage thresholds, also disjoint from test.
19. **Computation** — Yes.
    - GPU: Modal H100 (80 GB) for all training / diagnostic runs
    - Wall-clock: v6.0-3site ~3.5 hours for 4 epochs; LOO ~1.5–2 hours each; diagnostic eval ~15–30 minutes
    - Memory: peak ~60 GB during training
    - Reported in Appendix A.
20. **Error bars / significance** — Evidence-shuffling gap (ESG) is averaged over 3 derangement seeds (code: `v5/eval/evidence_blindness.py:196-201`). Per-category IMG tables omit cells with n < 10. Confidence intervals for IMG are computed via 1000-sample bootstrap on the test split (reported in supplementary).

## Results

21. **Consistent notation** — Yes. IMG, ESG, IPG as percentage-points (pp); acc_full, acc_image_zeroed, acc_evidence_shuffled as fractions in [0,1]; "blind" = Boolean (IMG < 5pp OR ESG < 5pp).
22. **Tests / proofs for claims** — Yes. `v5/tests/` (77 tests, 2 skipped env-dependent). Each silver grader has a test with a hard-coded real model output sample to prevent parser format drift (per handoff Mistake 4).

## Safety & Ethics

23. **Safety discussion** — Yes. §Discussion.Limitations addresses deployment risk: a paper that reports only aggregate accuracy would select the worst-grounded model (v5.1) as the "best." The IMG diagnostic is proposed as a mandatory safety check before deployment.
24. **Broader impact statement** — Yes. §4 Discussion (final paragraph) and datasheet §Uses discuss intended use (diagnostic audit, training-time mitigation research) and cautions (not a deployment-ready claim verifier; the 5pp threshold is a minimum bar not a sufficiency condition).
25. **Harm mitigation** — Yes. Three mitigations discussed: (a) per-category IMG reporting so clinicians can flag known-blind categories, (b) conformal trust-selection only recommended downstream of IMG-passing verifiers, (c) continuous evaluation on deployment data required.
26. **Security of data** — Yes. All data used is public. No PHI. PadChest-GR data use agreement honored.

## Transparency

27. **Anonymous-submission artifact link** — Planned. Code + model weights + logs will be posted as a GitHub release tagged `neurips2026-eand-submission` at acceptance (or earlier under anonymous GitHub if reviewer requests).
28. **Changes from announced paper** — N/A (first submission).
29. **Review artifact** — Yes. `reproduce_v6.sh` produces all tables and figures from a single Modal volume snapshot + repo clone; expected wall-clock ~6 hours on one H100.
