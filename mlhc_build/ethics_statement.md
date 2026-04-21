# Ethics Statement — ClaimGuard-CXR

## Summary of harms considered and mitigations applied

**The paper proposes a diagnostic framework (evidence-blindness) plus a training-distribution mitigation on a public chest-radiograph benchmark.** We review the safety, fairness, privacy, and deployment-readiness considerations below.

## Data and privacy

- All data used is public: OpenI (NIH public domain, Demner-Fushman 2016), ChestX-Det10 (CC-BY-4.0, Liu 2020), and PadChest-GR (restricted-access under Bilbao data-use agreement, Feijoo 2024). No patient-identifying information is exposed in any of our derived artifacts beyond what is already public in these sources.
- PadChest-GR is behind a DUA. We honor all access-restriction requirements documented in the PadChest-GR distribution agreement. Derived artifacts released by this paper (e.g., sentence-level silver-label Cohen $\kappa$ tables) are aggregate statistics that do not expose any individual patient record.
- No MIMIC-CXR or CheXpert-Plus data are used in training or evaluation, specifically to avoid credentialing-review delays and to preserve an auditable MIMIC-leakage analysis (see §3.7: GREEN was fine-tuned on MIMIC-CXR; RadFact and VERT are MIMIC-free, and we report pairwise Krippendorff $\alpha$ to quantify any leakage-driven correlation).

## Deployment readiness and safety

**The trained verifier in this paper is explicitly not deployment-ready.** The paper's central scientific contribution is a *diagnostic* that flags evidence-blindness, not a clinical-grade verifier. Two safety statements are load-bearing:

1. A verifier that passes the $\text{IMG} \geq 5$pp and $\text{ESG} \geq 5$pp thresholds is *not* certified as deployment-safe by that fact alone; it has merely cleared a minimum-use-of-evidence bar.
2. Residual evidence-blindness on laterality-turning claims (§3.6) is present across every training configuration we attempted. A clinician or clinical AI-system operator should not assume that our v6.0-3site verifier correctly identifies laterality mismatches in report text. This limit is reported transparently in the paper and in the downloadable model card.

## Fairness and subgroup audit

- OpenI is predominantly U.S. patient-source; ChestX-Det10 is a curated public subset; PadChest-GR is Spanish-source. The geographic composition is documented in the datasheet under "Collection" and "Composition."
- Per-site IMG is reported in Table 3 (LOO) to make site-level performance visible. We do not report per-demographic IMG because demographic labels are inconsistently populated across the three sources (age and sex present in PadChest-GR; absent or coarsely binned in OpenI).
- Future work should extend the diagnostic to pediatric, ICU, and non-Western populations, which are under-represented in these three sources. The diagnostic metric is scale-invariant and transfers to those cohorts without modification once data is available.

## Dual-use risk

- The diagnostic (IMG, ESG, IPG) and the training-time mitigation (consistency loss + adversarial HO filter) are applicable only in the context of multimodal image-claim verification. They are not transferable to offensive / intrusive applications.
- The training recipe does not enable synthesis of new medical content. The released artifacts are weights and evaluation code; they do not include a report-generation head.

## Environmental cost

- Total training compute for all configurations reported in this paper: approximately 140 H100-hours across v5.0 through v6.0 configurations, ablations, LOO, hidden-state probe, silver labeling, and baselines. At the Modal published rate (~3.5 kWh per H100-hour under typical utilization and a U.S.-West-2 grid CO2 intensity of ~250 gCO2/kWh), the carbon footprint is approximately 120 kgCO2e. This is reported transparently in the reproducibility checklist. We did not include cost minimization as an evaluation axis but we note that the diagnostic metric itself is compute-light (one forward pass plus three shuffle passes per example).

## Consent

- All sources publish under consent-for-research licenses. Patients contributing to these cohorts provided informed consent at the time of data collection for research use.

## Conflicts of interest

- The authors declare no financial conflict of interest with any of the companies whose model weights are evaluated in this paper (Microsoft / MAIRA-2, Google / MedGemma, Stanford–MIT / CheXagent, Anthropic / Claude). No author holds employment, equity, or advisory relationships with these entities.
