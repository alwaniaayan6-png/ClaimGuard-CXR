# Resume Prompt — ClaimGuard-CXR 9 Improvements

Copy-paste this block into a new Claude Code session to resume work.

---

I'm continuing work on ClaimGuard-CXR, a radiology report hallucination detector with formal FDR control. This is a continuation of a previous session that ran out of space. The project is at `/Users/aayanalwani/VeriFact/verifact/`.

Before doing ANYTHING, you must complete these steps in order. Do not start writing code, do not start editing files, do not start running scripts until every step below is done.

## Step 1: Read the user's global rules

Read `~/.claude/CLAUDE.md` **in full**. These are my mandatory global preferences for model selection (never Sonnet for reasoning — Opus only), GPU policy (H100 unless definitely <15min), doc-sync rules (update ARCHITECTURE.md + PROPOSAL.md with any code changes), autonomy preferences (never ask me to debug, self-check everything, run overnight if needed), and communication style (direct, concise, no LaTeX in chat). Every rule supersedes your defaults.

## Step 2: Read the handoff doc

Read `HANDOFF_ClaimGuard_Improvements_2026-04-14.md` **completely**. Every section, every table, every list. This is the state of the project and the 9 improvements I want you to make. Do not skim it. The handoff has specific file paths, specific failure modes from the previous session, and specific decisions that must not be revisited.

## Step 3: Read the project docs in full

Read all of the following files completely, not just the summaries:

- `ARCHITECTURE.md` — system blueprint (v1 scope)
- `ARCHITECTURE_V2.md` — v2 planned extensions (multimodal, CoFact)
- `CLAIMGUARD_PROPOSAL.md` — research proposal with every design decision annotated
- `REPRODUCIBILITY.md` — end-to-end pipeline + the new Scope and trust model section
- `MANUSCRIPT_MINI.md` — 5-page paper draft with Limitation (6) on provenance
- `paper/mlhc_abstract.md` — MLHC 2026 Clinical Abstract

## Step 4: Read the existing codebase

Read every file marked "YES — read fully" in the Existing Codebase table of the handoff doc. At minimum that includes:

- `inference/provenance.py` (newest, 300 lines, trust tier contract)
- `inference/conformal_triage.py` (has `gate_triage_with_provenance` helper)
- `scripts/modal_run_evaluation.py` (the most complex file, inverted cfBH + provenance gate + per-tier metrics)
- `scripts/prepare_eval_data.py` (8 hard-neg types + provenance stamping)
- `scripts/modal_build_retrieval_eval.py`
- `scripts/modal_train_verifier_binary.py`
- `tests/test_provenance.py` (36 tests, invariant lock file)
- `data/augmentation/hard_negative_generator.py`
- `demo/app.py`

Do not skim. I will ask you specific questions about these files to verify you read them.

## Step 5: Run the test suite to confirm current state

```bash
cd /Users/aayanalwani/VeriFact/verifact
python3 tests/test_provenance.py
```

All 36 tests should pass. If any fail, stop and fix the regression before doing any new work. Report the test results to me.

## Step 6: Verify the v1 checkpoint exists

```bash
ls -lh checkpoints/v1_best_verifier.pt
```

The file should be 1.6GB. If it's missing, re-download from Modal:

```bash
python3 -m modal volume get claimguard-data /checkpoints/verifier_binary_v2/best_verifier.pt ./checkpoints/v1_best_verifier.pt
```

## Step 7: Read the current literature

This is the step most people skip and it matters most. I want you to do a thorough literature review of the 2024-2026 work on:

1. **Real-world radiology hallucination benchmarks.** Is there a public dataset of VLM-generated radiology reports with radiologist annotation that I don't know about? Specifically search for anything published in 2025 or 2026 (ReXErr was PSB 2025, there may be follow-ups). Check MICCAI, MLHC, NeurIPS, Nature Digital Medicine, Lancet Digital Health, and medRxiv. ArXiv queries for "radiology hallucination benchmark," "chest x-ray factuality," "medical vision language model evaluation."

2. **Conformal prediction for LLM hallucination detection.** The project uses inverted cfBH (Jin & Candes 2023). What has been published since then on conformal procedures for generative AI factuality? Check CoFact (ICLR 2026), StratCP (medRxiv Feb 2026), and anything newer. Are there improvements to the inverted variant I should adopt?

3. **Silver-standard label ensembles for medical text.** Task 1 uses a 3-grader ensemble (CheXbert + vision-LLM + second VLM). Is there published work on how reliable this silver-standard approach is compared to radiologist gold standard? Are there better ensemble strategies?

4. **Same-model self-consistency failure modes in RAG.** The provenance gate addresses a specific failure: same model generating both claim and evidence. Is there published work on this failure mode in RAG systems generally (not just medical)? How are other systems addressing it?

5. **Hard-negative augmentation strategies for NLI.** Task 2 adds fabricated measurements, fabricated priors, and compound perturbations. What does the 2025-2026 literature say about effective hard-negative strategies for NLI-style verification tasks? Is there work on adversarial paraphrase augmentation (task 3) that's better than what I'm planning?

6. **CheXagent, RadFM, CXR-LLaVA, MedGemma.** Which of these are currently the best open VLMs for chest X-ray report generation? The pilot in task 1 depends on choosing a good generator, and I want to use the current SOTA not what was best in 2024.

Use WebFetch or WebSearch. Fetch the actual papers, do not rely on your training data.

**After the literature review, report to me:**

- Are the 9 improvements in the handoff the right ones? Are any of them outdated or superseded by better techniques?
- Are there additional improvements I should add that are obvious from the literature?
- Are any of them worth dropping because they're not a good use of budget?
- Is there a recently-published real-hallucination benchmark I could use instead of building my own silver-standard pilot?
- Does the provenance gate concept have prior art I should cite?

Give me the literature review as a structured report, not a brainstorm. I want to make an informed decision about which of the 9 tasks to execute and in what order before any code is written.

## Step 8: After the literature review, wait for my go-ahead

Do not start implementing anything until I confirm the plan. The literature review may change the priority order or add new tasks. I want to see your findings first.

## Context on the 9 improvements

The handoff doc has them in full detail. Short version so you know what you're evaluating:

1. **Silver-standard real-hallucination evaluation** — the most important one. Run CheXagent on 200 OpenI images, extract claims, label with a 3-grader ensemble (CheXbert + vision LLM + second VLM), filter to high-agreement subset, run v1 verifier on it. First real-world numbers.

2. **Extended synthetic perturbations** — add fabricated measurements, fabricated priors, compound perturbations to the 8 existing hard-negative types. Retrain v1. Honest per-type accuracy breakdown.

3. **Paraphrase-preserving augmentation** — generate meaning-preserving paraphrases of claims and hard negatives to attack the lexical shortcut (hypothesis-only baseline is currently 97.71%). Retrain. Measure HO delta.

4. **Existence-check safety overrides** — deterministic regex pass that flags numeric fabrications and prior-exam fabrications regardless of verifier score. Wires into `inference/conformal_triage.py`.

5. **Retrieval quality improvements** — batched BM25 + cross-encoder reranker. Current eval is dense-only. Should improve retrieval-augmented accuracy 1-3pp.

6. **Recalibrated + stratified OpenI analysis** — calibrate on half of OpenI, evaluate on the other half. Per-pathology stratification. Richer cross-dataset story.

7. **LLM-based claim extractor wired in** — replace regex sentence splitter with Phi-3-mini prompt-based extractor (scaffolded but never used).

8. **Self-annotation validation pass** — I label 100 claims manually, compare to silver-standard labels from task 1, report Cohen's kappa as an internal validity check.

9. **Provenance-gate failure-mode demonstration** — empirically validate the provenance gate by running CheXagent twice (once to generate claim, once to generate same-model "evidence") and showing the gate downgrades all the resulting high-score pairs to supported_uncertified.

Budget: $900 total. No radiologist access. Targeting MLHC Clinical Abstracts (April 17), NeurIPS D&B Pilot Track, MICCAI workshops, and Regeneron STS.

## What not to do

- Don't start coding before steps 1-7 are complete
- Don't use Sonnet for any reasoning or research — Opus only
- Don't retrain on RunPod community cloud (it ate my v2 checkpoints last session)
- Don't assume the v1 RoBERTa architecture is wrong — it works and is the honest baseline
- Don't claim Nature Medicine is achievable at this budget — it's not, and the handoff explicitly rejects that framing
- Don't skip self-check agents after writing code — the cost of bugs in a Modal GPU job is much higher than the cost of an Opus review pass
- Don't hallucinate paper results — the previous session lost v2 checkpoints and the current numbers are the ones in `results/` and `paper/mlhc_abstract.md`. If you want to cite something, cite those.

Start with Step 1.
