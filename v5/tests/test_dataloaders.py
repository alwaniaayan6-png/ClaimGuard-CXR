"""Smoke tests for data loaders — verify imports and basic field presence.

Full integration tests require the actual datasets on disk and are gated via
the V5_DATA_TESTS env var.
"""

from __future__ import annotations

import os

import pytest


def test_imports_all_loaders():
    from v5.data import (
        anatomy_masks,
        brax,
        chestx_det10,
        chexpert_plus,
        claim_extractor,
        claim_matcher,
        claim_parser,
        labeler_ensemble,
        groundbench,
        ms_cxr,
        object_cxr,
        openi,
        padchest,
        pii_scrubber,
        rsna_pneumonia,
        siim_pneumothorax,
        vlm_generators,
    )
    # trivial symbol assertions
    assert hasattr(claim_matcher, "ClaimMatcher")
    assert hasattr(claim_parser, "rule_parse")
    assert hasattr(chexpert_plus, "iter_chexpert_plus")
    assert hasattr(ms_cxr, "iter_ms_cxr")
    assert hasattr(groundbench, "assemble_row")
    assert hasattr(anatomy_masks, "compute_anatomy_masks")


@pytest.mark.skipif(
    os.environ.get("V5_DATA_TESTS") != "1",
    reason="Requires datasets on disk.",
)
def test_rsna_loader_basic():
    from pathlib import Path
    from v5.data.rsna_pneumonia import iter_rsna

    root = Path(os.environ.get("RSNA_ROOT", "/data/rsna_pneumonia"))
    recs = list(iter_rsna(root))
    assert len(recs) > 0
