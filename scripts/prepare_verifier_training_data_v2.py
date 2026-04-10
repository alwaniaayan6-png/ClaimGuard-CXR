"""Generate v2 verifier training data (30K balanced examples).

Uses the SAME generator as the eval data to guarantee distribution match:
same 8 hard-negative types, same class balance, same label-correctness
validators (C5 + C6 fixes). Committed for the NeurIPS 2026 reproducibility
artifact — reviewers can re-run this to reproduce verifier_training_data.json
byte-identical (with seed=42).

Usage:
    python3 scripts/prepare_verifier_training_data_v2.py \\
        --data-path /Users/aayanalwani/data/claimguard/chexpert-plus/df_chexpert_plus_240401.csv \\
        --splits-dir /Users/aayanalwani/data/claimguard/splits_chexpert_plus/ \\
        --output /Users/aayanalwani/data/claimguard/verifier_training_data_v2.json \\
        --n-claims 10000 \\
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

# Allow `from scripts...` imports when run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.prepare_eval_data import create_eval_claims  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Generate v2 verifier training data")
    parser.add_argument("--data-path", required=True,
                        help="Path to df_chexpert_plus_240401.csv")
    parser.add_argument("--splits-dir", required=True,
                        help="Path to splits directory (needs train_patients.csv)")
    parser.add_argument("--output", required=True,
                        help="Output JSON path for training data")
    parser.add_argument("--n-claims", type=int, default=10000,
                        help="Number of claims per class (supported/contradicted/insufficient)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load CheXpert Plus
    logger.info(f"Loading CheXpert Plus from {args.data_path}...")
    df = pd.read_csv(args.data_path, low_memory=False)
    df["deid_patient_id"] = df["deid_patient_id"].astype(str)
    logger.info(f"Loaded {len(df)} report rows")

    # Load TRAINING patient split
    train_split_file = Path(args.splits_dir) / "train_patients.csv"
    if not train_split_file.exists():
        raise FileNotFoundError(
            f"Training split not found at {train_split_file}. "
            f"Generate patient splits first via data/preprocessing/patient_splits.py"
        )
    train_pids = set(
        pd.read_csv(train_split_file)["patient_id"].astype(str).tolist()
    )
    train_df = df[df["deid_patient_id"].isin(train_pids)]
    logger.info(f"Training split: {len(train_df)} reports, {len(train_pids)} patients")

    # Generate claims using the canonical generator (same as eval data)
    claims = create_eval_claims(
        train_df,
        n_supported=args.n_claims,
        n_contradicted=args.n_claims,
        n_insufficient=args.n_claims,
        seed=args.seed,
    )

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(claims, f, indent=2)
    logger.info(f"Saved {len(claims)} training claims to {out_path}")


if __name__ == "__main__":
    main()
