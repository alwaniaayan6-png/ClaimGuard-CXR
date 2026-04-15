"""Patient-level data splits for ClaimGuard-CXR.

Creates reproducible 60/15/25 train/calibration/test splits at the patient level.
Every image from the same patient falls in the same partition. Stratified by sex
and study count to maintain demographic balance across splits.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def load_chexpert_plus_metadata(data_root: str | Path) -> pd.DataFrame:
    """Load CheXpert Plus metadata CSV.

    Args:
        data_root: Path to the CheXpert Plus dataset root directory.

    Returns:
        DataFrame with columns: patient_id, study_id, image_path, report_path, sex, ...
    """
    data_root = Path(data_root)

    # CheXpert Plus stores metadata in a CSV or parquet
    candidates = [
        data_root / "metadata.csv",
        data_root / "chexpert_plus.csv",
        data_root / "train.csv",
        data_root / "df_chexpert_plus.csv",
    ]
    meta_path = None
    for p in candidates:
        if p.exists():
            meta_path = p
            break

    if meta_path is None:
        raise FileNotFoundError(
            f"No metadata CSV found in {data_root}. "
            f"Looked for: {[str(c) for c in candidates]}"
        )

    logger.info(f"Loading metadata from {meta_path}")
    df = pd.read_csv(meta_path)

    # Standardize column names
    col_map = {}
    for col in df.columns:
        lower = col.lower().strip()
        if "patient" in lower or "subject" in lower:
            col_map[col] = "patient_id"
        elif lower == "sex" or lower == "gender":
            col_map[col] = "sex"
        elif "study" in lower and "id" in lower:
            col_map[col] = "study_id"
        elif lower == "path" or lower == "image_path":
            col_map[col] = "image_path"

    if col_map:
        df = df.rename(columns=col_map)

    if "patient_id" not in df.columns:
        # Try to extract patient_id from path (CheXpert format: patient12345/study1/...)
        if "image_path" in df.columns or "Path" in df.columns:
            path_col = "image_path" if "image_path" in df.columns else "Path"
            df["patient_id"] = df[path_col].apply(
                lambda x: str(x).split("/")[0] if pd.notna(x) else None
            )
        else:
            raise ValueError("Cannot find or derive patient_id column in metadata")

    logger.info(f"Loaded {len(df)} records from {len(df['patient_id'].unique())} patients")
    return df


def create_patient_splits(
    metadata: pd.DataFrame,
    train_frac: float = 0.60,
    cal_frac: float = 0.15,
    test_frac: float = 0.25,
    seed: int = 42,
    stratify_col: Optional[str] = "sex",
) -> dict[str, list[str]]:
    """Create patient-level splits with stratification.

    Args:
        metadata: DataFrame with at least a 'patient_id' column.
        train_frac: Fraction of patients for training.
        cal_frac: Fraction for calibration.
        test_frac: Fraction for testing.
        seed: Random seed for reproducibility.
        stratify_col: Column to stratify by (None for no stratification).

    Returns:
        Dict with keys 'train', 'calibration', 'test', each mapping to a list of patient IDs.
    """
    assert abs(train_frac + cal_frac + test_frac - 1.0) < 1e-6, \
        f"Fractions must sum to 1.0, got {train_frac + cal_frac + test_frac}"

    # Get unique patients with stratification info
    patients = metadata.groupby("patient_id").agg(
        num_studies=("patient_id", "count"),
        **({stratify_col: (stratify_col, "first")} if stratify_col and stratify_col in metadata.columns else {}),
    ).reset_index()

    patient_ids = patients["patient_id"].values

    # Build stratification variable
    stratify = None
    if stratify_col and stratify_col in patients.columns:
        stratify = patients[stratify_col].fillna("Unknown").values
        # Bin study counts for stratification
        patients["study_bucket"] = pd.qcut(
            patients["num_studies"], q=3, labels=["low", "med", "high"], duplicates="drop"
        )
        stratify = (
            patients[stratify_col].fillna("Unknown").astype(str)
            + "_"
            + patients["study_bucket"].astype(str)
        ).values

        # Remove groups with fewer than 2 members (sklearn requirement)
        from collections import Counter
        counts = Counter(stratify)
        stratify = np.array([s if counts[s] >= 2 else "other" for s in stratify])

    # First split: train vs rest
    rest_frac = cal_frac + test_frac
    train_ids, rest_ids = train_test_split(
        patient_ids,
        test_size=rest_frac,
        random_state=seed,
        stratify=stratify,
    )

    # Build stratify labels for the rest split
    rest_strat = None
    if stratify is not None:
        # Map rest_ids back to their stratify labels
        id_to_strat = dict(zip(patient_ids, stratify))
        rest_strat = np.array([id_to_strat[pid] for pid in rest_ids])

    # Second split: calibration vs test
    cal_relative = cal_frac / rest_frac
    cal_ids, test_ids = train_test_split(
        rest_ids,
        test_size=1.0 - cal_relative,
        random_state=seed,
        stratify=rest_strat,
    )

    splits = {
        "train": sorted(train_ids.tolist()),
        "calibration": sorted(cal_ids.tolist()),
        "test": sorted(test_ids.tolist()),
    }

    logger.info(
        f"Split {len(patient_ids)} patients: "
        f"train={len(splits['train'])} ({len(splits['train'])/len(patient_ids):.1%}), "
        f"cal={len(splits['calibration'])} ({len(splits['calibration'])/len(patient_ids):.1%}), "
        f"test={len(splits['test'])} ({len(splits['test'])/len(patient_ids):.1%})"
    )

    return splits


def verify_no_leakage(splits: dict[str, list[str]]) -> bool:
    """Assert zero patient overlap between all split pairs.

    Args:
        splits: Dict of split_name -> list of patient IDs.

    Returns:
        True if no leakage detected.

    Raises:
        AssertionError if any overlap is found.
    """
    names = list(splits.keys())
    assert len(names) >= 2, f"Need at least 2 splits to check leakage, got {len(names)}"
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            set_a = set(splits[names[i]])
            set_b = set(splits[names[j]])
            overlap = set_a & set_b
            assert len(overlap) == 0, (
                f"LEAKAGE: {len(overlap)} patients shared between "
                f"'{names[i]}' and '{names[j]}': {list(overlap)[:5]}..."
            )
    logger.info("No leakage detected across splits")
    return True


def save_splits(splits: dict[str, list[str]], output_dir: str | Path, seed: int = 42) -> None:
    """Save split files to disk.

    Args:
        splits: Dict of split_name -> list of patient IDs.
        output_dir: Directory to save split files.
        seed: Random seed used (saved in metadata).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save individual split files
    for split_name, patient_ids in splits.items():
        path = output_dir / f"{split_name}_patients.csv"
        pd.DataFrame({"patient_id": patient_ids}).to_csv(path, index=False)
        logger.info(f"Saved {len(patient_ids)} patients to {path}")

    # Save metadata
    meta = {
        "seed": seed,
        "counts": {k: len(v) for k, v in splits.items()},
        "total_patients": sum(len(v) for v in splits.values()),
    }
    meta_path = output_dir / "split_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Saved split metadata to {meta_path}")


def load_splits(split_dir: str | Path) -> dict[str, list[str]]:
    """Load previously saved splits from disk.

    Args:
        split_dir: Directory containing split CSV files.

    Returns:
        Dict of split_name -> list of patient IDs.
    """
    split_dir = Path(split_dir)
    splits = {}
    for split_name in ["train", "calibration", "test"]:
        path = split_dir / f"{split_name}_patients.csv"
        if not path.exists():
            raise FileNotFoundError(f"Split file not found: {path}")
        df = pd.read_csv(path)
        splits[split_name] = df["patient_id"].astype(str).tolist()
        logger.info(f"Loaded {len(splits[split_name])} {split_name} patients from {path}")
    return splits


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Create patient-level splits for ClaimGuard-CXR")
    parser.add_argument("--data-root", type=str, required=True, help="Path to CheXpert Plus root")
    parser.add_argument("--output-dir", type=str, default="./splits", help="Output directory for split files")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    meta = load_chexpert_plus_metadata(args.data_root)
    splits = create_patient_splits(meta, seed=args.seed)
    verify_no_leakage(splits)
    save_splits(splits, args.output_dir, seed=args.seed)
