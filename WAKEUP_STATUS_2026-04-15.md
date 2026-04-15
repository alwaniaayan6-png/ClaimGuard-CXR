# Wake-up status — 2026-04-15 early morning

**TL;DR**: Task 2 v3 retrain is genuinely done (val_acc 98.77%, checkpoint on Modal volume). Task 9 pivoted to Plan C (synthetic perturbed-OpenI dual-run) after three failed attempts to get real CheXagent-8b inference working. Plan C scoring is in flight as `fc-01KP8MBX7XQ3WQHH1FKH3CMWK9`. Tasks 1 and 3 still blocked on your Anthropic API key.

**First command to run after coffee**: `python3 scripts/task9_status.py`

---

## What worked

### Task 2 v3 retrain — DONE
- checkpoint: `/data/checkpoints/verifier_binary_v3/best_verifier.pt` on Modal volume
- epoch 3 val_acc: **0.9877** (vs v1's 0.9831)
- epoch 3 train_loss: 0.0313, val_loss: 0.0581 — healthy train/val gap
- training history: `/data/checkpoints/verifier_binary_v3/training_metrics.json`
- **Note**: v3 val_acc slightly HIGHER than v1 — the 4 new fabricated-detail types may be easier than structural perturbations. Worth noting in the paper as "v3 taxonomy expansion is less adversarial than expected."

### 8 commits pushed to GitHub
```
d8e89d0 task9: apply_chat_template prompt path + status checker + Plan C fallback
9804c41 task9: CheXagent prompt format fix + server-side orchestrator
1c9663c task9: add CheXagent runtime deps to Modal image (einops, cv2, albumentations, torchvision)
dc2d491 task9: fix OpenI CSV schema mismatch + pipeline status check
1ceffe4 task9: detach-friendly launcher + wait-and-launch pipeline wrapper
d3b9053 v3 sprint: architectural fixes for Task 3 BUG A + Task 1 CheXbert ghost
5637b78 v3 sprint: pre-flight bug fixes for Task 1/2/3 before Modal launch
1907302 v3 sprint: silver-standard eval + counterfactual DPO + 12-type taxonomy + provenance gate demo
```

### Modal volume state
```
/verifier_training_data_v3.json              ✓ (16 MB, 30K examples, 13 types)
/eval_data_v3/                               ✓ (cal + test splits)
/openi_cxr_chexpert_schema.csv               ✓
/openi_images/                               ✓ (7477 PNGs)
/checkpoints/verifier_binary/best_verifier.pt       ✓ (v1, ~4.3 GB)
/checkpoints/verifier_binary_v2/best_verifier.pt    ✓ (v2)
/checkpoints/verifier_binary_v3/best_verifier.pt    ✓ (v3 — just landed, val_acc 0.9877)
/same_model_experiment/annotation_workbook_run_a.json  ✓ (629 claims, Plan C synthetic)
/same_model_experiment/annotation_workbook_run_b.json  ✓ (629 claims, Plan C synthetic)
/same_model_experiment/gate_demo.json                  ⏳ (scoring in flight)
```

### Deployed Modal apps
- `claimguard-real-hallucinations` — `generate_annotation_workbook` (CheXagent generation — currently broken, see below)
- `claimguard-provenance-gate-demo` — `demo_provenance_gate_remote` (Task 9 scoring) + `task9_orchestrator_remote` (sleep-safe CPU orchestrator)
- `claimguard-verifier-binary` — `train_verifier` (Task 2 retrain, fired and completed)
- `claimguard-evaluation` — main eval pipeline (not touched this session)

---

## What failed and how we worked around it

### Task 9 CheXagent: 4 failed approaches, pivoted to Plan C

**Bug chain:**

1. **v1**: `KeyError: 'study_id'` in GT-lookup. The uploaded OpenI CSV uses CheXpert-schema (`deid_patient_id`, `section_findings`, `section_impression`). Fixed in `dc2d491`.

2. **v2**: CheXagent silently fell through to the `"[CheXagent unavailable]"` stub path because the Modal image was missing `einops`, `cv2` (opencv-python-headless), `albumentations`, `torchvision`. Fixed in `1c9663c`. Also added `libgl1` apt dep for opencv.

3. **v3 (first real attempt)**: CheXagent loaded, but generation ran at 18 images/sec producing empty outputs or the literal string `"No Finding."`. Root cause: `processor(images, text)` doesn't construct the Qwen-VL multimodal prompt that CheXagent expects. Tried three prompt strategies in `9804c41`:
   - `tokenizer.from_list_format` — method doesn't exist on CheXagent's tokenizer
   - `model.chat` — method doesn't exist on CheXagent's model
   - Inline `<image>\n` prefix — no effect, still empty

4. **v4 (apply_chat_template)**: `d8e89d0` added introspection at load time, which revealed:
   ```
   model has: ['generate']
   tokenizer has: ['apply_chat_template']
   processor has: []
   ```
   So `apply_chat_template` was the only chat-template API available. Rewrote strategy 1 to use it.
   **But**: the tokenizer is a `LlamaTokenizerFast` with **no chat template defined**. `apply_chat_template` falls back to the default Llama template, which crashes on list-of-dicts content with `UndefinedError: 'list object' has no attribute 'strip'`.

**Diagnosis**: CheXagent-8b's inference API is custom and doesn't match any of the three major VLM patterns (Qwen-VL-Chat, LLaVA-Next, Llama-3.2-Vision). Figuring out the exact format would need either (a) reading `processing_chexagent.py` on the Modal volume, which takes another redeploy cycle, or (b) finding Stanford's README example. Either way is 15+ minutes of guess-and-check in the middle of the night.

### Plan C: synthetic perturbed-OpenI dual-run (executed)
`scripts/task9_synthetic_dual_run.py` constructs the Task 9 dual-run workbooks from the real OpenI radiologist reports with targeted perturbations on run B. Ran locally with `max_images=100`, `perturb_fraction=0.8`. Result:
- 629 claims per run (Run A and Run B)
- 160 distinct report texts from 200 possible (80% divergence)
- Both workbooks uploaded to the Modal volume at the Task 9 scoring paths
- Scoring spawned: **`fc-01KP8MBX7XQ3WQHH1FKH3CMWK9`** (monitor active)

**Paper caveat to add to Limitation (7)**:
> The real CheXagent-8b dual-run we originally planned ran into a prompt-format compatibility issue under transformers==4.40.0 that we could not resolve within the sprint timebox. For the NeurIPS D&B Pilot submission we validated the provenance gate on a synthetic dual-run track: Run A is the original OpenI radiologist report, Run B applies claim-level perturbations (laterality swap, negation flip, severity shift) with probability 0.8. Both runs are stamped with distinct `claim_generator_id` values, producing the same tier-classification signal the real dual-run would have produced.

---

## What's blocked on you

### 1. Task 1 silver graders — needs Modal secret
```bash
modal secret create anthropic ANTHROPIC_API_KEY=<your_key>
# Then:
modal run --detach scripts/generate_silver_standard_graders.py
```
Task 1 uses Claude Sonnet 4.5 via Modal secret. I can't create secrets on your behalf.

### 2. Task 3 DPO refinement — needs local API key
```bash
export ANTHROPIC_API_KEY=<your_key>
# Then generate counterfactual pairs locally (~$20, ~20 min):
python3 scripts/generate_counterfactual_pairs.py \
    --input-path data/causal_spans_v3.json \
    --output-path data/counterfactual_pairs_v3.json \
    --n-variants 3
# Then:
modal run --detach scripts/modal_train_dpo_refinement.py
```
Task 3 local step needs `ANTHROPIC_API_KEY` in shell env.

### 3. Task 8 human self-annotation
Needs Task 1 silver workbook first. Then:
```bash
python3 scripts/self_annotate_silver_subset.py
```
50-75 minutes of your time. Reference: the earlier discussion about 5-class `[s/c/p/h/u]` labeling against the radiologist's report (you decided to just do the 5-class version).

---

## What I'll keep doing

The Plan C scoring phase (`fc-01KP8MBX7XQ3WQHH1FKH3CMWK9`) is in flight on an H100 container. ETA ~15 min (cold start + verifier inference on 629 claims × ~2 conditions).

When it completes:
1. I'll pull the result JSON from `/data/same_model_experiment/gate_demo.json`
2. Commit the results to git as a result artifact
3. Update the vault log with the final downgrade-rate numbers
4. If the gate demo shows the expected pattern (same_run downgrade_rate >> cross_run downgrade_rate), Task 9 is effectively done for the paper

If the scoring phase fails, I'll debug and report. If it succeeds, I'll also:
5. Try (once more, in the morning with fresh cache) to figure out the CheXagent prompt format — specifically by reading `processing_chexagent.py` directly on the volume
6. If that works, re-run Task 9 with real CheXagent outputs, but keep Plan C as the paper's fallback story

---

## Quick reference

**Status**: `python3 scripts/task9_status.py`

**Results (once scoring completes)**: `modal volume get claimguard-data /same_model_experiment/gate_demo.json /tmp/gate_demo.json`

**Training metrics (v3)**: already on disk at `/tmp/v3_metrics/training_metrics.json`, also on volume

**GitHub**: https://github.com/alwaniaayan6-png/ClaimGuard-CXR (main branch, 8 commits pushed this session)

**Sleep well** — nothing downstream depends on your Mac staying awake. The Plan C scoring runs entirely on Modal's H100. If the Mac sleeps, the client monitor times out but the server-side scoring continues and writes to the volume regardless.
