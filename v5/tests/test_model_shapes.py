"""Smoke test for ImageGroundedVerifier shapes. Requires network access on first
run because BiomedCLIP and RoBERTa weights are pulled from HuggingFace.

Mark skipped in CI unless HF_HUB_OFFLINE=0 and network is available.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("V5_NETWORK_TESTS") != "1",
    reason="Set V5_NETWORK_TESTS=1 to run (pulls large HF weights).",
)


def test_forward_smoke():
    import torch

    from v5.model import V5Config, build_v5_model, build_v5_tokenizer

    cfg = V5Config()
    tok = build_v5_tokenizer(cfg)
    model = build_v5_model(cfg).eval()

    B = 2
    pv = torch.randn(B, 3, 224, 224)
    enc = tok(
        ["A small left pleural effusion is present.", "No acute abnormality."],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=cfg.max_text_tokens,
    )
    with torch.no_grad():
        out = model(pv, enc["input_ids"], enc["attention_mask"], return_grounding=True)
    assert out["verdict_logits"].shape == (B, 2)
    assert out["support_score"].shape == (B,)
    if "grounding_logits" in out:
        assert out["grounding_logits"].shape == (B, 14, 14)
