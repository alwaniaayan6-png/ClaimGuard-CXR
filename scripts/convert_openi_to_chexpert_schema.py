"""Convert OpenI (Indiana University) XML reports into CheXpert Plus CSV schema.

This allows prepare_eval_data.py to run unchanged on OpenI for cross-dataset
evaluation. Output CSV has columns matching CheXpert Plus's structure.

OpenI schema:
  - 3,955 XML reports with FINDINGS, IMPRESSION, MeSH labels (major + automatic)
  - Each report has unique uId (CXR1, CXR2, ...) used as patient_id

CheXpert Plus schema (expected columns):
  - deid_patient_id (str)
  - section_findings (str)
  - section_impression (str)
  - pathology columns (Atelectasis, Cardiomegaly, etc. as 0/1/-1)

Usage:
  python3 scripts/convert_openi_to_chexpert_schema.py \
    --xml-dir /Users/aayanalwani/data/openi/ecgen-radiology \
    --output /Users/aayanalwani/data/openi/openi_cxr_chexpert_schema.csv
"""

from __future__ import annotations

import argparse
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd


# Map OpenI MeSH labels -> CheXpert 14-label taxonomy
MESH_TO_CHEXPERT = {
    # Normal
    "normal": "No Finding",
    "no finding": "No Finding",
    # Opacity / Lung Opacity
    "opacity": "Lung Opacity",
    "opacities": "Lung Opacity",
    "infiltrates": "Lung Opacity",
    "infiltrate": "Lung Opacity",
    # Atelectasis
    "atelectasis": "Atelectasis",
    "atelectases": "Atelectasis",
    # Cardiomegaly
    "cardiomegaly": "Cardiomegaly",
    # Consolidation
    "consolidation": "Consolidation",
    # Edema
    "edema": "Edema",
    "pulmonary edema": "Edema",
    # Effusion
    "pleural effusion": "Pleural Effusion",
    "pleural effusions": "Pleural Effusion",
    "effusion": "Pleural Effusion",
    # Pneumothorax
    "pneumothorax": "Pneumothorax",
    # Pneumonia
    "pneumonia": "Pneumonia",
    # Lesion / Nodule / Mass
    "nodule": "Lung Lesion",
    "mass": "Lung Lesion",
    "lesion": "Lung Lesion",
    # Fracture
    "fracture": "Fracture",
    "fractures": "Fracture",
    # Devices
    "catheter": "Support Devices",
    "tube": "Support Devices",
    "pacemaker": "Support Devices",
    "sternotomy": "Support Devices",
    "line": "Support Devices",
}

CHEXPERT_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion",
    "Pneumonia", "Pneumothorax", "Lung Opacity", "Lung Lesion", "Fracture",
    "Support Devices", "Enlarged Cardiomediastinum", "No Finding",
]


def extract_report(xml_path: Path) -> dict | None:
    """Parse one OpenI XML file into a dict."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError:
        return None

    # Extract uId from filename since some reports miss the attribute
    uid_elem = root.find("uId")
    if uid_elem is not None and uid_elem.get("id"):
        uid = uid_elem.get("id")
    else:
        uid = f"CXR{xml_path.stem}"

    # Extract findings + impression
    findings = None
    impression = None
    for ab in root.iter("AbstractText"):
        label = ab.get("Label", "").upper()
        text = (ab.text or "").strip()
        if label == "FINDINGS":
            findings = text
        elif label == "IMPRESSION":
            impression = text

    if not findings and not impression:
        return None

    # Extract MeSH labels (both major + automatic)
    mesh_labels = set()
    for mesh in root.iter("MeSH"):
        for child in mesh.iter():
            if child.tag in {"major", "automatic"}:
                text = (child.text or "").strip().lower()
                if text:
                    mesh_labels.add(text)

    # Map to CheXpert labels (1 = present, 0 = absent or unknown)
    chexpert_row = {lbl: 0 for lbl in CHEXPERT_LABELS}
    for mesh in mesh_labels:
        # Strip modifiers like "/mild" or "/severe"
        base = mesh.split("/")[0].strip()
        if base in MESH_TO_CHEXPERT:
            chexpert_row[MESH_TO_CHEXPERT[base]] = 1

    # Clean up XXXX placeholders (OpenI's redaction markers)
    def clean(s):
        if not s:
            return ""
        return re.sub(r"XXXX", "redacted", s).strip()

    return {
        "deid_patient_id": uid,  # Use CXR id as patient id
        "section_findings": clean(findings),
        "section_impression": clean(impression),
        **chexpert_row,
        "source": "openi",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml-dir", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    xml_files = sorted(Path(args.xml_dir).glob("*.xml"))
    print(f"Found {len(xml_files)} OpenI XML reports")

    rows = []
    skipped = 0
    for p in xml_files:
        rec = extract_report(p)
        if rec is None:
            skipped += 1
            continue
        rows.append(rec)

    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} reports to {args.output} (skipped {skipped})")
    print(f"\nColumn summary:")
    print(df.dtypes)
    print(f"\nLabel distribution:")
    for lbl in CHEXPERT_LABELS:
        if lbl in df.columns:
            n = int(df[lbl].sum())
            print(f"  {lbl}: {n}/{len(df)} ({100*n/len(df):.1f}%)")


if __name__ == "__main__":
    main()
