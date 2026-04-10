# ClaimGuard-CXR: Hackathon Demo Plan

**Goal:** Integrate ClaimGuard-CXR into a live, interactive web demo for an AI/Healthcare hackathon. Judges should be able to paste a radiology report and see claim-level hallucination detection with formal FDR guarantees in real time.

**Target timeline:** 24-48 hour hackathon
**Target stack:** Gradio + Hugging Face Spaces + ZeroGPU (free H200)
**Demo URL:** https://huggingface.co/spaces/{username}/ClaimGuard-CXR (after deploy)

---

## 1. What Judges Will See (The Pitch)

### Core Story (2 minutes)
1. **Problem:** AI radiology report generators hallucinate 8-15% of claims. Laterality swaps, negation flips, fabricated findings. Wrong side surgery is a real failure mode.
2. **Existing approaches:** Report-level accept/reject (too coarse) OR claim-level LLM-as-judge (no error guarantees).
3. **Our solution:** Claim-level verification + **provable FDR control**. Every green claim comes with a mathematical guarantee that the fraction of hallucinations is ≤ alpha.
4. **Live demo:** Paste a report → instant color-coded output → drag the safety slider → watch claims change triage with formal guarantees.
5. **Honest limitations:** Show the artifact finding as a feature. We tell judges *exactly* where the model breaks.

### Key Numbers to Show
- **98.61% accuracy** (v2 DeBERTa with progressive NLI)
- **FDR = 1.66% at alpha = 5%** with **99.32% power**
- **32 percentage points above** zero-shot baselines
- **98.15% hypothesis-only** (the artifact finding — transparency wins)

---

## 2. Feature Breakdown (Prioritized)

### Tier 1: MUST-HAVE for demo (Hours 0-12)

**F1. Report Verifier (core feature)**
- Textbox for pasting a radiology report
- Output: `gr.HighlightedText` with per-claim GREEN/YELLOW/RED color coding
- Underneath: detailed table showing claim, triage, confidence score, p-value
- Triggered by "Verify" button

**F2. Interactive FDR Slider**
- `gr.Slider(0.01, 0.20, value=0.05, step=0.01, label="FDR target (alpha)")`
- Dragging it re-runs ONLY the conformal triage (CPU, <100ms)
- Scores are cached — no GPU re-inference on slider change
- Visual feedback: claims change color in real-time

**F3. Pre-loaded Example Reports**
- 8-10 synthetic radiology reports covering:
  - Normal exam (should be all green)
  - Pneumonia with correct findings (should be all green)
  - Pneumonia with **1 laterality swap** inserted (1 claim should go red)
  - Heart failure with **severity swap** (red)
  - Trauma with **fabricated finding** (red)
  - Deliberately ambiguous report (should produce yellows)
- `gr.Examples` component — one click to load

**F4. Summary Stats Panel**
- Live counts: N green / N yellow / N red
- FDR guarantee text: "At α=0.05, ≤5% of green claims are expected to be hallucinated. Observed FDR on test set: 1.66%."

### Tier 2: SHOULD-HAVE (Hours 12-24)

**F5. Baseline Comparison Panel**
- Drop-down: "Compare against: [ClaimGuard | Rule-based | DeBERTa zero-shot NLI | RadFlag]"
- Same report, different color coding per baseline
- Side-by-side view showing why ClaimGuard wins

**F6. Evidence Retrieval Display**
- Click any claim → modal/expander shows retrieved evidence passages
- (Can use synthetic evidence for demo since FAISS index is 3.6GB)
- Shows "the verifier sees this evidence when making a decision"

**F7. Conformal FDR Dashboard**
- Small plotly chart: FDR vs alpha sweep (from the real results)
- Second plot: power vs alpha
- Third plot: reliability diagram (calibration)
- All static, loaded from `results/v2_runpod/captured_results.json`

**F8. Transparent Artifact Section**
- A dedicated tab or expandable panel: "How honest is this model?"
- Shows the hypothesis-only baseline result (98.15%)
- Explains WHY: "Our synthetic perturbations contain lexical shortcuts. We disclose this transparently."
- Judges reward honest limitations — this becomes a feature, not a flaw

### Tier 3: NICE-TO-HAVE (Hours 24-36)

**F9. Upload Your Own Report (PDF/Image)**
- File upload → extract text with OCR (pytesseract for demo)
- Auto-verify → display results

**F10. Claim Extraction Visualizer**
- Show the raw report
- Show extracted atomic claims as bubbles or a tree
- Highlight which sentences map to which claims
- Helps judges see the full pipeline

**F11. "Why this claim?" Explanations**
- For each triaged claim, show top-5 most salient tokens (attention or gradient)
- Visual heat over the claim text

**F12. Shareable Link Generation**
- After verification, generate a unique URL that pre-loads the same report
- For judges to share with each other

---

## 3. Technical Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Hugging Face Spaces                         │
│                  (Gradio Frontend)                           │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
         ▼               ▼               ▼
┌─────────────┐  ┌──────────────┐  ┌─────────────────┐
│ Claim       │  │ DeBERTa-v3   │  │ Conformal       │
│ Extractor   │  │ Verifier     │  │ Triage Engine   │
│ (CPU)       │  │ (@spaces.GPU)│  │ (CPU, NumPy)    │
│ rule-based  │  │ ZeroGPU H200 │  │ inverted cfBH   │
└─────────────┘  └──────────────┘  └─────────────────┘
         │               │               │
         └───────────────┴───────────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ gr.HighlightedText│
                │ + summary + plot  │
                └──────────────────┘
```

### Model Hosting Strategy
- **Primary:** Upload `best_verifier.pt` (~1.7GB DeBERTa-v3-large) to HF Hub as a private model
- Demo loads via `transformers.AutoModel.from_pretrained()` with ZeroGPU allocation via `@spaces.GPU` decorator
- **Problem:** We don't have the checkpoint locally (lived on terminated RunPod pods)
- **Solution 1:** Retrain v2 once more on a **stable** secure cloud pod, save to HF Hub immediately
- **Solution 2:** Use `microsoft/deberta-v3-large` zero-shot without fine-tuning (degraded but deployable)
- **Solution 3:** Train on Hugging Face directly using their training infrastructure

### Data Flow for a Single Verification
```
User pastes report
    ↓
[CPU] extract_claims(report_text) → list of claim strings
    ↓
[GPU, @spaces.GPU] batch_verify(claims) → softmax scores
    ↓
[CPU] conformal_triage(scores, alpha, cal_contra_scores) → green/yellow/red labels
    ↓
[CPU] render HighlightedText + stats
    ↓
Return to user (target latency: <3 seconds for a 10-claim report)
```

### Calibration Data
- Need `cal_contra_scores.npy` — the contradicted calibration scores from the test run
- Bundle with the Spaces repo (few MB, no sensitive data)
- Loaded once at startup

---

## 4. Implementation Phases

### Phase 0: Pre-hackathon Prep (BEFORE the hackathon starts)
- [ ] Retrain v2 DeBERTa on a stable GPU, push checkpoint to HF Hub as private model
- [ ] Generate 10 synthetic example reports covering normal + pathology + hallucinations
- [ ] Save calibration contradicted scores to `cal_contra_scores.npy`
- [ ] Extract FDR/power sweep data to JSON for dashboard

### Phase 1: Core Demo (Hours 0-8)
- [ ] Create HF Space repo with Gradio starter template
- [ ] Port `demo/app.py` to the Space with ZeroGPU decorator
- [ ] Implement F1 (report verifier) + F2 (FDR slider) + F3 (examples)
- [ ] Deploy and verify it loads
- [ ] Test with 5+ example reports end to end

### Phase 2: Polish (Hours 8-16)
- [ ] Custom CSS for dark mode, better colors, logo
- [ ] Add F4 (stats panel) + F7 (FDR dashboard plot)
- [ ] Add F8 (transparent artifact panel — the differentiator)
- [ ] Write About page with problem/method/results
- [ ] Test latency, optimize caching

### Phase 3: Advanced Features (Hours 16-24)
- [ ] Add F5 (baseline comparison)
- [ ] Add F6 (evidence display)
- [ ] Add F10 (claim extraction visualizer) if time permits
- [ ] Record 60-second demo video as backup

### Phase 4: Presentation Prep (Hours 24-36)
- [ ] Practice 2-minute pitch using the live demo
- [ ] Prepare slide deck with:
  - Problem slide
  - Method slide (conformal FDR diagram)
  - Results table
  - Live demo transition
  - Artifact transparency slide
  - Team/acknowledgments
- [ ] Have a backup demo video in case WiFi dies
- [ ] Pre-load the Space in 3 browser tabs (redundancy)

### Phase 5: Buffer (Hours 36-48)
- [ ] Fix bugs found during practice runs
- [ ] Optimize for judge demo path
- [ ] Get sleep

---

## 5. Tech Stack Details

### Frontend
- **Gradio 4.x** — required for HF ZeroGPU, ships with HighlightedText
- **Custom CSS** — dark theme, medical-professional look
- **Plotly** — for FDR/power sweep charts

### Backend
- **Python 3.11** — matches our existing code
- **PyTorch 2.4 + transformers 4.40.0** — match training environment
- **NumPy + scipy** — for conformal procedure
- **Pillow** — if adding image uploads later

### Deployment
- **HF Spaces** — free tier gets 2 vCPU + 16GB RAM + ZeroGPU access
- **ZeroGPU** — 3.5 min/day free GPU quota (H200, 70GB). Enough for ~50 demos/day
- **HF Hub model hosting** — upload `best_verifier.pt` as a private model

### Model Files Structure
```
spaces/ClaimGuard-CXR/
├── app.py                       # Main Gradio app (port from demo/app.py)
├── requirements.txt             # deps
├── models/
│   └── verifier_loader.py       # load DeBERTa from HF Hub
├── utils/
│   ├── claim_extractor.py       # rule-based extraction
│   ├── conformal.py             # inverted cfBH
│   └── examples.py              # pre-loaded reports
├── data/
│   ├── cal_contra_scores.npy    # calibration data
│   ├── examples.json            # synthetic reports
│   └── fdr_sweep.json           # pre-computed for dashboard
├── assets/
│   ├── logo.png
│   └── style.css
└── README.md                    # Space metadata
```

---

## 6. Demo Script (What to Say During Judging)

**[0:00-0:15] Hook**
> "Radiology AI hallucinates 10% of the time. A single laterality swap — 'left' instead of 'right' — can lead to wrong-side surgery. Current systems either reject whole reports or give you confidence scores with no guarantee. Watch this."

**[0:15-0:45] Live Demo Part 1**
- Paste a normal report → all green
- Paste a report with a hidden laterality swap → 1 red claim
- Point to the swap: "ClaimGuard caught it"

**[0:45-1:30] Live Demo Part 2 — The Guarantee**
- Drag the alpha slider from 0.20 → 0.01
- Watch claims move from green to yellow as the guarantee tightens
- "At alpha equals 5 percent, the math says at most 5 percent of greens are wrong. Our test set measured 1.66 percent. Formal FDR control via conformal Benjamini-Hochberg."

**[1:30-2:00] The Honest Differentiator**
- Click to the "Transparency" tab
- Show the 98.15% hypothesis-only result
- "Here's the uncomfortable truth: our synthetic perturbations are too easy. We disclose this. The next step is a real-hallucination test set with radiologist annotation. This is how healthcare AI should be built — with uncertainty, with guarantees, with honesty."

**[2:00+] Q&A**
Expect questions on:
- "Is this deployable?" → Yes, inference is <1 second per claim on a single GPU.
- "How does FDR transfer to new hospitals?" → v1 OpenI transfer experiment: FDR stays ≤ alpha at every level.
- "What about MIMIC-CXR?" → PhysioNet credentialing in progress.
- "Who's the team?" → Single-author high school research at Weill Cornell.

---

## 7. Critical Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| ZeroGPU quota hit mid-demo | Medium | High | Warm the model before demo, have a cached-response fallback |
| HF Spaces slow/down | Low | Critical | Record 90-second demo video as backup, bring laptop with local copy |
| Verifier checkpoint not retrained in time | Medium | High | Fallback: use off-the-shelf `microsoft/deberta-v3-large-mnli` (degraded but functional) |
| Live audience WiFi dies | Medium | Critical | Run local instance on laptop, demo from localhost via HDMI |
| Judges don't understand conformal prediction | High | Medium | Pre-explain with analogy: "Like a 95% confidence interval, but for binary safety decisions" |
| Examples don't show clear hallucinations | Low | Medium | Test every example end-to-end, make sure laterality/negation swaps trigger red |
| 1.7GB model too big for free Space | Low | Medium | Use ONNX quantization (INT8) to reduce to ~450MB |
| Gradio version incompatibility with ZeroGPU | Medium | High | Pin Gradio 4.44.1 (known-good with ZeroGPU) |

---

## 8. Differentiation Strategy

Most hackathon projects have:
- ✗ Cool idea, no working model
- ✗ Working model, no math guarantees
- ✗ Math guarantees, no demo
- ✗ Demo, no honesty about limitations

ClaimGuard-CXR will have:
- ✓ Trained model with real results (98.61%)
- ✓ Formal statistical guarantees (conformal FDR)
- ✓ Live interactive demo (Gradio + ZeroGPU)
- ✓ Transparent artifact analysis (the hypothesis-only finding)
- ✓ Published code (public GitHub)
- ✓ Clear clinical use case (radiologist review triage)

**The honest limitation is the winning move.** Most teams hide their weaknesses. Showing judges exactly where the model breaks demonstrates scientific maturity. Healthcare judges especially reward this.

---

## 9. Post-Hackathon Path

If we win or place:
- Write MLHC 2026 abstract (deadline April 17 — already drafted)
- Write NeurIPS 2026 paper (deadline May 6)
- Apply for Regeneron STS (opens June 1)

If we don't place:
- Demo is still live and citable in the paper
- GitHub stars from hackathon visibility
- Conversations with judges → advisors for paper submission

---

## 10. Immediate Action Items (Before Hackathon)

1. **[1 hour]** Spin up a **secure cloud** RunPod pod, retrain v2 DeBERTa, immediately push `best_verifier.pt` to HF Hub (not to local disk — push straight from the pod)
2. **[30 min]** Save `cal_contra_scores.npy` from the same training run, push to HF Hub dataset
3. **[1 hour]** Write 10 synthetic radiology reports (5 clean + 5 with known hallucinations inserted at known positions)
4. **[30 min]** Create an HF Space repo as a template, test Gradio + ZeroGPU decorator with a hello-world
5. **[2 hours]** Port `demo/app.py` to the Space, replace model loading with HF Hub pull, deploy
6. **[1 hour]** Verify end-to-end demo path with all 10 examples, fix any bugs
7. **[30 min]** Write Space README with clear description for judges who visit the URL

**Total prep time: ~6-7 hours before the hackathon starts.**
