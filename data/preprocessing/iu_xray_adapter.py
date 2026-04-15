"""IU X-Ray dataset adapter for ClaimGuard-CXR.

Converts the IU X-Ray HuggingFace format into the same interface as
the CheXpert Plus loader, enabling pipeline development and testing
before CheXpert Plus access is obtained.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Map IU X-Ray MeSH/Problems to CheXpert-like labels
_MESH_TO_CHEXPERT = {
    "Cardiomegaly": "Cardiomegaly",
    "Pulmonary Edema": "Edema",
    "Consolidation": "Consolidation",
    "Pneumonia": "Pneumonia",
    "Atelectasis": "Atelectasis",
    "Pneumothorax": "Pneumothorax",
    "Pleural Effusion": "Pleural Effusion",
    "Fracture": "Fracture",
    "Opacity": "Lung Opacity",
    "Mass": "Lung Lesion",
    "Nodule": "Lung Lesion",
    "Emphysema": "Lung Opacity",
    "Pulmonary Fibrosis": "Lung Opacity",
    "normal": "No Finding",
}


def load_iu_xray_as_chexpert_format(
    csv_path: str | Path,
) -> pd.DataFrame:
    """Load IU X-Ray CSV and convert to CheXpert Plus-compatible format.

    Args:
        csv_path: Path to iu_xray_reports.csv.

    Returns:
        DataFrame with columns matching CheXpert Plus expectations:
        patient_id, study_id, report, findings, impression, sex, and
        CheXpert label columns.
    """
    df = pd.read_csv(csv_path)

    # Map columns to standard format
    result = pd.DataFrame()
    result["patient_id"] = df["uid"].astype(str)
    result["study_id"] = df["uid"].astype(str)
    result["report"] = df.apply(
        lambda row: _combine_report(row.get("findings", ""), row.get("impression", "")),
        axis=1,
    )
    result["findings"] = df["findings"].fillna("")
    result["impression"] = df["impression"].fillna("")
    result["indication"] = df["indication"].fillna("")
    result["sex"] = "Unknown"  # IU X-Ray doesn't have demographics

    # Parse MeSH terms into CheXpert-like labels
    chexpert_labels = [
        "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
        "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
        "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
        "Pleural Other", "Fracture", "Support Devices",
    ]

    for label in chexpert_labels:
        result[label] = 0  # default negative

    for idx, row in df.iterrows():
        mesh = str(row.get("mesh", ""))
        problems = str(row.get("problems", ""))
        combined = mesh + ";" + problems

        if combined.strip() in ("", "nan;nan", "normal;normal"):
            result.at[idx, "No Finding"] = 1
            continue

        for term in combined.split(";"):
            term = term.strip()
            if not term or term == "nan":
                continue
            # Check against mapping
            for mesh_key, chexpert_label in _MESH_TO_CHEXPERT.items():
                if mesh_key.lower() in term.lower():
                    result.at[idx, chexpert_label] = 1
                    break

    logger.info(
        f"Loaded {len(result)} IU X-Ray reports in CheXpert-compatible format. "
        f"Label distribution: "
        + ", ".join(f"{l}={result[l].sum()}" for l in chexpert_labels if result[l].sum() > 0)
    )

    return result


def _combine_report(findings: str, impression: str) -> str:
    """Combine findings and impression into a full report."""
    parts = []
    f = str(findings).strip() if pd.notna(findings) else ""
    i = str(impression).strip() if pd.notna(impression) else ""
    if f and f != "nan":
        parts.append(f"FINDINGS: {f}")
    if i and i != "nan":
        parts.append(f"IMPRESSION: {i}")
    return " ".join(parts) if parts else ""


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="/Users/aayanalwani/data/claimguard/iu-xray/iu_xray_reports.csv")
    parser.add_argument("--output", type=str, default="/Users/aayanalwani/data/claimguard/iu-xray/iu_xray_chexpert_format.csv")
    args = parser.parse_args()

    df = load_iu_xray_as_chexpert_format(args.csv)
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} records to {args.output}")
