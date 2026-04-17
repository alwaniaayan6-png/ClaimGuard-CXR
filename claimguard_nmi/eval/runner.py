"""Multi-site evaluation runner.

Usage (from project root):

    python -m claimguard_nmi.eval.runner \
        --site padchest_gr \
        --checkpoint /path/to/verifier.pt \
        --conformal inverted \
        --alpha 0.05 \
        --out results/path_b/padchest_gr_inverted_0p05.json

Reads:
  * configs/datasets.yaml for the site's image+label roots
  * configs/ontology.yaml for canonical-finding mapping
  * A pre-built per-site JSONL of (image_id, extracted_claims, annotations)

Writes:
  * JSON summary with per-claim decisions, conformal results, and metrics.

This is a skeleton — model forward integration and VLM claim extraction
are stubbed with clear TODO markers. Fill in as the pipeline matures.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

from claimguard_nmi.conformal import InvertedCfBH, WeightedCfBH
from claimguard_nmi.eval.metrics import compute_verdict_metrics


_REPO_ROOT = Path(__file__).resolve().parents[1]
_DATASETS_YAML = _REPO_ROOT / "configs" / "datasets.yaml"


def load_site_config(site: str) -> dict:
    with open(_DATASETS_YAML, "r") as fh:
        cfg = yaml.safe_load(fh)
    primary = cfg.get("datasets", {}).get(site)
    if primary is None:
        raise KeyError(f"unknown site '{site}' (not in configs/datasets.yaml)")
    return primary


def _stub_load_scored_claims(site: str, checkpoint: str) -> dict:
    """Placeholder returning the expected shape.

    Replace with actual verifier inference over the site's (image, claim) pairs.
    Expected return dict:

        {
          "claim_ids": np.ndarray[str] shape (N,),
          "y_true": np.ndarray[int] shape (N,) with 0/1,
          "y_supported_prob": np.ndarray[float] shape (N,),
          "y_contradicted_prob": np.ndarray[float] shape (N,),
          "y_pred": np.ndarray[int] shape (N,),
          "calibration_supported_prob_on_contradicted": np.ndarray[float],
          "features": np.ndarray[float] shape (N, d) — for weighted cfBH
          "calibration_features": np.ndarray[float] shape (N_cal, d),
        }
    """
    raise NotImplementedError(
        "_stub_load_scored_claims must be replaced with a real verifier-inference "
        "pipeline that loads the site's (image, claim) pairs, runs the image-grounded "
        "verifier, and returns scored outputs in the schema documented above."
    )


def run(site: str, checkpoint: str, conformal: str, alpha: float, out_path: Path) -> None:
    site_cfg = load_site_config(site)
    scored = _stub_load_scored_claims(site, checkpoint)  # raises until wired up

    if conformal == "inverted":
        proc = InvertedCfBH().fit(scored["calibration_supported_prob_on_contradicted"])
        result = proc.predict(scored["y_supported_prob"], alpha=alpha)
        result = InvertedCfBH.audit(result, scored["y_true"])
    elif conformal == "weighted":
        proc = WeightedCfBH().fit(
            scored["calibration_supported_prob_on_contradicted"],
            scored["calibration_features"],
            scored["features"],
        )
        result = proc.predict(scored["y_supported_prob"], scored["features"], alpha=alpha)
    else:
        raise ValueError(f"unknown conformal variant: {conformal}")

    metrics = compute_verdict_metrics(
        scored["y_true"], scored["y_pred"], scored["y_contradicted_prob"],
    )

    out = {
        "site": site,
        "site_config": site_cfg,
        "checkpoint": checkpoint,
        "conformal": conformal,
        "alpha": alpha,
        "n_total": int(result.n_total),
        "n_green": int(result.n_green),
        "fdr": getattr(result, "fdr", None),
        "power": getattr(result, "power", None),
        "bh_threshold": float(result.bh_threshold),
        "metrics": {k: asdict(v) for k, v in metrics.items()},
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--site", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--conformal", choices=["inverted", "weighted"], default="inverted")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    run(args.site, args.checkpoint, args.conformal, args.alpha, Path(args.out))


if __name__ == "__main__":
    main()
