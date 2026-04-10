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
