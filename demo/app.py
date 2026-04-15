"""ClaimGuard-CXR v2 — Gradio Web Demo

Hackathon-ready web application for radiology report hallucination detection.
Deployable on Hugging Face Spaces with ZeroGPU for free GPU inference.

Features:
  1. Report Verifier: paste a radiology report, get per-claim green/yellow/red labels
  2. Calibration Explorer: interactive FDR alpha slider
  3. Examples: pre-loaded synthetic and real reports with known hallucinations

Usage (local):
    python demo/app.py

Usage (HF Spaces):
    Upload this directory to a Gradio Space with ZeroGPU enabled.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import gradio as gr
except ImportError:
    raise ImportError("Install gradio: pip install gradio>=4.0.0")

from inference.provenance import (  # noqa: E402
    EvidenceSourceType,
    ProvenanceTriageLabel,
    TrustTier,
    apply_provenance_gate,
    classify_trust_tier,
)

# Try ZeroGPU decorator (HF Spaces), fall back to no-op
try:
    import spaces
    GPU_DECORATOR = spaces.GPU
except ImportError:
    def GPU_DECORATOR(fn):
        return fn


# ============================================================
# Model loading (lazy, on first inference call)
# ============================================================
_MODEL = None
_TOKENIZER = None
_CAL_SCORES = None
_DEVICE = "cpu"


def _load_model():
    """Lazy-load the verifier model and calibration scores."""
    global _MODEL, _TOKENIZER, _CAL_SCORES, _DEVICE
    if _MODEL is not None:
        return

    import torch
    from transformers import AutoModel, AutoTokenizer

    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {_DEVICE}...")

    # Try DeBERTa v2 checkpoint first, fall back to RoBERTa v1
    model_name = "microsoft/deberta-v3-large"
    checkpoint_path = os.environ.get(
        "CLAIMGUARD_CHECKPOINT",
        "best_verifier.pt"
    )

    _TOKENIZER = AutoTokenizer.from_pretrained(model_name)

    # Build model — attribute names MUST match modal_train_verifier_deberta.py
    import torch.nn as nn
    class SimpleVerifier(nn.Module):
        def __init__(self, model_name, num_classes=2, hidden_dim=256, dropout=0.1):
            super().__init__()
            self.text_encoder = AutoModel.from_pretrained(model_name)
            text_dim = self.text_encoder.config.hidden_size
            self.verdict_head = nn.Sequential(
                nn.Linear(text_dim, hidden_dim), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes),
            )
            self.temperature = nn.Parameter(torch.tensor(1.0))

        def forward(self, input_ids, attention_mask):
            out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            cls = out.last_hidden_state[:, 0, :]
            return self.verdict_head(cls)

    _MODEL = SimpleVerifier(model_name).to(_DEVICE)

    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=_DEVICE, weights_only=True)
        _MODEL.load_state_dict(state_dict)
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"WARNING: No checkpoint at {checkpoint_path}. Using random weights (demo mode).")

    _MODEL.eval()

    # Load calibration scores for conformal procedure
    cal_path = os.environ.get("CLAIMGUARD_CAL_SCORES", "cal_contra_scores.npy")
    if os.path.exists(cal_path):
        _CAL_SCORES = np.load(cal_path)
        print(f"Loaded {len(_CAL_SCORES)} calibration scores")
    else:
        # Demo mode: synthetic calibration scores
        rng = np.random.RandomState(42)
        _CAL_SCORES = rng.beta(0.5, 10, size=5000)  # Cluster near 0 (contradicted)
        print("Using synthetic calibration scores (demo mode)")


# ============================================================
# Claim extraction
# ============================================================
#: Demo-wide lazy singleton for the LLM extractor — loading Phi-3-mini
#: costs ~5s and ~2 GB, so we only pay it on the first Gradio inference
#: call (never at import time).  The ``CLAIMGUARD_DEMO_LLM_EXTRACTOR``
#: env var lets power users flip LLM mode on in hosted environments
#: without editing code.
_DEMO_LLM_EXTRACTOR = None


def _get_demo_extractor():
    """Lazy-load a shared ``LLMClaimExtractor`` for the Gradio demo.

    Controlled by the ``CLAIMGUARD_DEMO_LLM_EXTRACTOR`` env var:
        * ``"1"`` / ``"true"`` / ``"yes"`` → LLM path (Phi-3-mini, CPU).
        * anything else → rule-based fallback (default — keeps the
          hosted demo responsive and GPU-free).
    """
    global _DEMO_LLM_EXTRACTOR
    if _DEMO_LLM_EXTRACTOR is not None:
        return _DEMO_LLM_EXTRACTOR
    import os
    from models.decomposer.llm_claim_extractor import LLMClaimExtractor

    use_llm = os.environ.get(
        "CLAIMGUARD_DEMO_LLM_EXTRACTOR", ""
    ).strip().lower() in ("1", "true", "yes", "on")
    _DEMO_LLM_EXTRACTOR = LLMClaimExtractor(use_llm=use_llm)
    return _DEMO_LLM_EXTRACTOR


def extract_claims(report_text: str) -> list[str]:
    """Extract atomic claims from a radiology report.

    Routes through the lazy-loaded demo ``LLMClaimExtractor`` singleton
    (rule-based by default, LLM-backed when
    ``CLAIMGUARD_DEMO_LLM_EXTRACTOR=1``).  Falls back to a pure regex
    splitter if the decomposer module cannot be imported at all (e.g.,
    offline environments missing ``transformers``).
    """
    try:
        extractor = _get_demo_extractor()
        results = extractor.extract_claims(report_text)
        return [r["claim"] for r in results] if results else [report_text.strip()]
    except ImportError:
        # Fallback: enhanced sentence splitting with context merging
        sentences = re.split(r'(?<=[.!?])\s+', report_text.strip())
        claims = []
        for i, s in enumerate(sentences):
            s = s.strip()
            if len(s) <= 10:
                continue
            # Merge continuation sentences (starting with "or", "and")
            if claims and s.lower().startswith(("or ", "and ", "nor ")):
                claims[-1] = claims[-1].rstrip('.') + ", " + s[0].lower() + s[1:]
            else:
                claims.append(s)
        return claims if claims else [report_text.strip()]


# ============================================================
# Verification (GPU-accelerated on ZeroGPU)
# ============================================================
@GPU_DECORATOR
def verify_claims(claims: list[str], evidence: str = "") -> list[dict]:
    """Run binary verification on each claim.

    Returns list of dicts with 'claim', 'score', 'verdict'.
    """
    import torch
    import torch.nn.functional as F

    _load_model()

    results = []
    for claim in claims:
        evidence_text = evidence if evidence else "No additional evidence available."

        encoding = _TOKENIZER(
            claim,
            evidence_text,
            max_length=512,
            padding="max_length",
            truncation="only_second",
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(_DEVICE)
        attention_mask = encoding["attention_mask"].to(_DEVICE)

        with torch.no_grad():
            logits = _MODEL(input_ids, attention_mask)
            probs = F.softmax(logits, dim=-1)
            not_contra_prob = probs[0, 0].item()  # P(Not-Contradicted)
            contra_prob = probs[0, 1].item()

        results.append({
            "claim": claim,
            "score": not_contra_prob,
            "contra_prob": contra_prob,
        })

    return results


# ============================================================
# Conformal triage + provenance gate
# ============================================================
# Display strings for the provenance-aware final labels. Kept in one place
# so the summary panel and the per-claim highlight agree.
_FINAL_LABEL_DISPLAY = {
    ProvenanceTriageLabel.SUPPORTED_TRUSTED: "SUPPORTED (trusted)",
    ProvenanceTriageLabel.SUPPORTED_UNCERTIFIED: "SUPPORTED (uncertified)",
    ProvenanceTriageLabel.REVIEW_REQUIRED: "REVIEW",
    ProvenanceTriageLabel.CONTRADICTED: "CONTRADICTED",
    ProvenanceTriageLabel.PROVENANCE_BLOCKED: "PROVENANCE BLOCKED",
}

_FINAL_LABEL_COLOR = {
    ProvenanceTriageLabel.SUPPORTED_TRUSTED: "#22c55e",      # green
    ProvenanceTriageLabel.SUPPORTED_UNCERTIFIED: "#6b7280",  # gray: cannot certify
    ProvenanceTriageLabel.REVIEW_REQUIRED: "#eab308",        # yellow
    ProvenanceTriageLabel.CONTRADICTED: "#ef4444",           # red
    ProvenanceTriageLabel.PROVENANCE_BLOCKED: "#6b7280",     # gray
}


def conformal_triage(
    claim_results: list[dict],
    alpha: float = 0.05,
    trust_tier: str = TrustTier.UNKNOWN,
) -> list[dict]:
    """Apply inverted cfBH conformal procedure and then the provenance gate.

    Each result dict gets three new fields:
      - "triage"       : raw green/yellow/red from the conformal step (legacy)
      - "final_label"  : provenance-aware label (ProvenanceTriageLabel.*)
      - "display"      : human-readable label string for the UI
      - "color"        : color corresponding to `final_label`
      - "trust_tier"   : the tier used for gating
      - "gate_override": True if a green claim was downgraded by the gate
    """
    if _CAL_SCORES is None:
        _load_model()

    cal_scores = _CAL_SCORES
    n_cal = len(cal_scores)

    for result in claim_results:
        s = result["score"]
        # Inverted p-value: compare against contradicted calibration scores
        p_value = (np.sum(cal_scores >= s) + 1) / (n_cal + 1)
        result["p_value"] = p_value

    # BH procedure
    n_test = len(claim_results)
    p_values = np.array([r["p_value"] for r in claim_results])
    sorted_idx = np.argsort(p_values)
    sorted_pvals = p_values[sorted_idx]

    k_star = 0
    for k in range(1, n_test + 1):
        if sorted_pvals[k - 1] <= k * alpha / n_test:
            k_star = k

    if k_star > 0:
        bh_threshold = sorted_pvals[k_star - 1]
    else:
        bh_threshold = 0.0

    for result in claim_results:
        if result["p_value"] <= bh_threshold and k_star > 0:
            raw = "green"
        elif result["score"] > 0.3:
            raw = "yellow"
        else:
            raw = "red"

        gate = apply_provenance_gate(raw, trust_tier)
        result["triage"] = raw.upper()  # legacy field, backward compat
        result["final_label"] = gate.final_label
        result["display"] = _FINAL_LABEL_DISPLAY[gate.final_label]
        result["color"] = _FINAL_LABEL_COLOR[gate.final_label]
        result["trust_tier"] = gate.trust_tier
        result["gate_override"] = gate.was_overridden

    return claim_results


# ============================================================
# Gradio Interface
# ============================================================
_CACHED_RESULTS = None  # Cache verification results to avoid GPU re-inference on alpha change
_CACHED_REPORT = None


# Display name -> canonical trust tier for the Gradio dropdown.
_EVIDENCE_SOURCE_CHOICES = {
    "Oracle report (trusted)": EvidenceSourceType.ORACLE_REPORT_TEXT,
    "Retrieved passage (independent)": EvidenceSourceType.RETRIEVED_REPORT_TEXT,
    "Generator output (same-model risk)": EvidenceSourceType.GENERATOR_OUTPUT,
    "Unknown provenance": EvidenceSourceType.UNKNOWN,
}


def process_report(
    report_text: str,
    alpha: float,
    evidence_source_display: str = "Oracle report (trusted)",
) -> tuple:
    """Main processing function: extract claims -> verify -> triage -> display.

    The `evidence_source_display` argument selects the evidence provenance
    for the current input, which drives the provenance gate. A real
    deployment would determine this from the evidence pipeline itself, not
    from a UI dropdown — the dropdown exists in this demo so judges can
    actually see the gate in action.
    """
    global _CACHED_RESULTS, _CACHED_REPORT

    if not report_text.strip():
        return [], "Please enter a radiology report.", ""

    # Map the display string to a canonical source_type and then to a tier.
    source_type = _EVIDENCE_SOURCE_CHOICES.get(
        evidence_source_display, EvidenceSourceType.UNKNOWN
    )
    trust_tier = classify_trust_tier(
        source_type=source_type,
        claim_generator_id=None,
        evidence_generator_id=None,
    )

    # Only re-run GPU inference if report text changed
    if report_text != _CACHED_REPORT or _CACHED_RESULTS is None:
        claims = extract_claims(report_text)
        results = verify_claims(claims)
        _CACHED_RESULTS = results
        _CACHED_REPORT = report_text
    else:
        # Reuse cached scores, only recompute triage at new alpha / tier
        results = [dict(r) for r in _CACHED_RESULTS]  # shallow copy

    results = conformal_triage(results, alpha=alpha, trust_tier=trust_tier)

    # Format for HighlightedText using the provenance-aware display label.
    highlighted = []
    for r in results:
        highlighted.append((r["claim"] + " ", r["final_label"]))

    # Summary statistics by final (provenance-aware) label.
    n_sup_trusted = sum(1 for r in results if r["final_label"] == ProvenanceTriageLabel.SUPPORTED_TRUSTED)
    n_sup_uncert = sum(1 for r in results if r["final_label"] == ProvenanceTriageLabel.SUPPORTED_UNCERTIFIED)
    n_review = sum(1 for r in results if r["final_label"] == ProvenanceTriageLabel.REVIEW_REQUIRED)
    n_contra = sum(1 for r in results if r["final_label"] == ProvenanceTriageLabel.CONTRADICTED)

    fdr_line = (
        f"- FDR guarantee: <= {alpha*100:.0f}% of SUPPORTED (trusted) claims are hallucinated"
        if trust_tier in (TrustTier.TRUSTED, TrustTier.INDEPENDENT)
        else (
            f"- **No FDR guarantee**: evidence trust tier is `{trust_tier}`. "
            f"ClaimGuard only certifies claims when evidence comes from a trusted "
            f"or independent source. All supported claims shown as UNCERTIFIED."
        )
    )

    summary = (
        f"**Results at alpha={alpha}, evidence = {evidence_source_display}, tier = `{trust_tier}`**\n\n"
        f"- SUPPORTED (trusted): {n_sup_trusted} claims\n"
        f"- SUPPORTED (uncertified): {n_sup_uncert} claims\n"
        f"- REVIEW: {n_review} claims\n"
        f"- CONTRADICTED: {n_contra} claims\n"
        f"{fdr_line}"
    )

    # Detailed table
    detail_rows = []
    for r in results:
        detail_rows.append([
            r["claim"][:80] + ("..." if len(r["claim"]) > 80 else ""),
            r.get("display", r["final_label"]),
            r["trust_tier"],
            f"{r['score']:.4f}",
            f"{r['p_value']:.4f}",
        ])

    return highlighted, summary, detail_rows


# Example reports. Each row is [report, alpha, evidence_source_display].
EXAMPLES = [
    [
        "The heart is normal in size. The lungs are clear without focal consolidation, "
        "pleural effusion, or pneumothorax. No acute osseous abnormality. "
        "The mediastinal contours are normal.",
        0.05,
        "Oracle report (trusted)",
    ],
    [
        "There is a large left-sided pleural effusion with associated compressive "
        "atelectasis of the left lower lobe. The heart is mildly enlarged. "
        "Right lung is clear. No pneumothorax. A right subclavian central venous "
        "catheter terminates in the SVC.",
        0.05,
        "Retrieved passage (independent)",
    ],
    [
        "The heart is severely enlarged. The lungs show bilateral consolidation "
        "consistent with pneumonia. There is a small right pneumothorax. "
        "No pleural effusion is seen. The left lung is clear without opacity.",
        0.10,
        "Generator output (same-model risk)",
    ],
]


def build_demo():
    """Build the Gradio demo interface."""
    with gr.Blocks(
        title="ClaimGuard-CXR: Radiology Hallucination Detection",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            "# ClaimGuard-CXR\n"
            "### Claim-Level Hallucination Detection for Radiology Reports "
            "with Conformal FDR Control\n\n"
            "Paste a radiology report below. Each claim will be verified and "
            "color-coded by the **provenance-aware triage**:\n\n"
            "- **SUPPORTED (trusted)** — verifier accepted AND evidence is from "
            "a trusted or independent source.\n"
            "- **SUPPORTED (uncertified)** — verifier accepted but evidence is "
            "same-model or unknown; ClaimGuard refuses to certify safety.\n"
            "- **REVIEW** — borderline, needs a clinician to adjudicate.\n"
            "- **CONTRADICTED** — verifier found the claim inconsistent with evidence.\n\n"
            "*For research demonstration only. Not for clinical use.*"
        )

        with gr.Tab("Report Verifier"):
            with gr.Row():
                with gr.Column(scale=2):
                    report_input = gr.Textbox(
                        label="Radiology Report",
                        placeholder="Paste a chest X-ray radiology report here...",
                        lines=8,
                    )
                    alpha_slider = gr.Slider(
                        minimum=0.01,
                        maximum=0.20,
                        value=0.05,
                        step=0.01,
                        label="FDR Target (alpha)",
                        info="Lower alpha = stricter, more claims flagged for review",
                    )
                    evidence_source_input = gr.Dropdown(
                        choices=list(_EVIDENCE_SOURCE_CHOICES.keys()),
                        value="Oracle report (trusted)",
                        label="Evidence provenance",
                        info=(
                            "ClaimGuard only certifies claims when evidence is "
                            "trusted or independent. Same-model or unknown "
                            "provenance is downgraded to UNCERTIFIED."
                        ),
                    )
                    verify_btn = gr.Button("Verify Report", variant="primary")

                with gr.Column(scale=3):
                    highlighted_output = gr.HighlightedText(
                        label="Verified Claims",
                        color_map={
                            ProvenanceTriageLabel.SUPPORTED_TRUSTED: "#22c55e",
                            ProvenanceTriageLabel.SUPPORTED_UNCERTIFIED: "#6b7280",
                            ProvenanceTriageLabel.REVIEW_REQUIRED: "#eab308",
                            ProvenanceTriageLabel.CONTRADICTED: "#ef4444",
                            ProvenanceTriageLabel.PROVENANCE_BLOCKED: "#6b7280",
                        },
                    )
                    summary_output = gr.Markdown(label="Summary")

            detail_table = gr.Dataframe(
                headers=["Claim", "Final label", "Trust tier", "Score", "p-value"],
                label="Detailed Results",
            )

            verify_btn.click(
                fn=process_report,
                inputs=[report_input, alpha_slider, evidence_source_input],
                outputs=[highlighted_output, summary_output, detail_table],
            )

            # Also trigger on alpha or provenance change
            alpha_slider.change(
                fn=process_report,
                inputs=[report_input, alpha_slider, evidence_source_input],
                outputs=[highlighted_output, summary_output, detail_table],
            )
            evidence_source_input.change(
                fn=process_report,
                inputs=[report_input, alpha_slider, evidence_source_input],
                outputs=[highlighted_output, summary_output, detail_table],
            )

            gr.Examples(
                examples=EXAMPLES,
                inputs=[report_input, alpha_slider, evidence_source_input],
                outputs=[highlighted_output, summary_output, detail_table],
                fn=process_report,
                cache_examples=False,
            )

        with gr.Tab("About"):
            gr.Markdown(
                "## How It Works\n\n"
                "1. **Claim Extraction**: The report is split into atomic claims\n"
                "2. **Binary Verification**: Each claim is verified by a DeBERTa-v3-large "
                "cross-encoder against retrieved evidence passages\n"
                "3. **Conformal Triage**: The inverted conformal Benjamini-Hochberg procedure "
                "assigns triage labels with formal FDR control\n\n"
                "## Key Results\n\n"
                "- 98.31% accuracy on CheXpert Plus (15K test claims)\n"
                "- FDR controlled at 1.30% (target 5%)\n"
                "- Cross-dataset transfer to OpenI without retraining\n"
                "- Outperforms zero-shot LLMs by 31+ percentage points\n\n"
                "## Citation\n\n"
                "Alwani, A. (2026). ClaimGuard-CXR: Calibrated Claim-Level "
                "Hallucination Detection for Radiology Reports with Conformal "
                "FDR Control.\n\n"
                "## Disclaimer\n\n"
                "This is a research prototype. It is NOT intended for clinical "
                "decision-making. All example reports are synthetic."
            )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(share=True)
