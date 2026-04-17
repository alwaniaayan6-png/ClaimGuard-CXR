#!/usr/bin/env bash
# Download all PUBLIC datasets for ClaimGuard-Bench-Grounded (Path B).
#
# Credentialed datasets (MS-CXR, CheXmask source images, MIMIC-CXR,
# VinDr-CXR, ReXVal, ReXErr) are NOT downloaded here. Those require
# PhysioNet credentialed access under Laughney's signature.
#
# Environment variables to set beforehand:
#   CLAIMGUARD_DATA_ROOT       — root directory for all datasets
#   KAGGLE_USERNAME            — from https://www.kaggle.com/settings
#   KAGGLE_KEY                 — from same
#   BIMCV_TOKEN                — PadChest-GR data-use approval token (email)
#
# Usage:
#   bash scripts/path_b/download_datasets.sh
#
# Each dataset is skipped if its target directory already exists.

set -euo pipefail

ROOT="${CLAIMGUARD_DATA_ROOT:?set CLAIMGUARD_DATA_ROOT to a writable directory}"
mkdir -p "$ROOT"
cd "$ROOT"

echo "ClaimGuard Path B — public dataset download"
echo "Root: $ROOT"
echo

# ---------------------------------------------------------------------------
# 1. RSNA Pneumonia Detection Challenge (Kaggle)
# ---------------------------------------------------------------------------
if [ ! -d "$ROOT/rsna_pneumonia" ]; then
    echo "[1/6] RSNA Pneumonia — Kaggle"
    mkdir -p "$ROOT/rsna_pneumonia"
    cd "$ROOT/rsna_pneumonia"
    if ! command -v kaggle >/dev/null; then
        echo "  kaggle CLI not found; install with: pip install kaggle"
        exit 1
    fi
    kaggle competitions download -c rsna-pneumonia-detection-challenge
    unzip -q rsna-pneumonia-detection-challenge.zip
    cd "$ROOT"
else
    echo "[1/6] RSNA Pneumonia — already present, skipping"
fi

# ---------------------------------------------------------------------------
# 2. SIIM-ACR Pneumothorax (Kaggle)
# ---------------------------------------------------------------------------
if [ ! -d "$ROOT/siim_pneumothorax" ]; then
    echo "[2/6] SIIM-ACR Pneumothorax — Kaggle"
    mkdir -p "$ROOT/siim_pneumothorax"
    cd "$ROOT/siim_pneumothorax"
    kaggle competitions download -c siim-acr-pneumothorax-segmentation
    unzip -q siim-acr-pneumothorax-segmentation.zip
    cd "$ROOT"
else
    echo "[2/6] SIIM-ACR Pneumothorax — already present, skipping"
fi

# ---------------------------------------------------------------------------
# 3. NIH ChestX-ray14 (Box Labels subset)
# ---------------------------------------------------------------------------
if [ ! -d "$ROOT/nih_cxr14" ]; then
    echo "[3/6] NIH CXR14 — Stanford AIMI mirror"
    mkdir -p "$ROOT/nih_cxr14"
    cd "$ROOT/nih_cxr14"
    # BBox CSV (small)
    curl -fsSL https://nihcc.app.box.com/v/ChestXray-NIHCC/file/0_BBox_List_2017.csv \
        -o BBox_List_2017.csv || echo "  WARN: BBox CSV download failed; set BOX mirror path"
    # Image archive (47 GB; comment out if not desired for scaffolding stage)
    # curl -fsSL https://nihcc.app.box.com/v/ChestXray-NIHCC/file/images.tar.gz \
    #     -o images.tar.gz && tar xzf images.tar.gz
    cd "$ROOT"
else
    echo "[3/6] NIH CXR14 — already present, skipping"
fi

# ---------------------------------------------------------------------------
# 4. ChestX-Det10 (GitHub release)
# ---------------------------------------------------------------------------
if [ ! -d "$ROOT/chestxdet10" ]; then
    echo "[4/6] ChestX-Det10 — GitHub"
    git clone --depth 1 https://github.com/Deepwise-AILab/ChestX-Det10-Dataset.git \
        "$ROOT/chestxdet10"
else
    echo "[4/6] ChestX-Det10 — already present, skipping"
fi

# ---------------------------------------------------------------------------
# 5. Object-CXR (MIDL 2020)
# ---------------------------------------------------------------------------
if [ ! -d "$ROOT/object_cxr" ]; then
    echo "[5/6] Object-CXR — JF Healthcare"
    mkdir -p "$ROOT/object_cxr"
    # Hosted on Dianxin Cloud / academictorrents mirror; user must verify link.
    echo "  MANUAL STEP: download Object-CXR from https://jfhealthcare.github.io/object-CXR/"
    echo "              into $ROOT/object_cxr"
else
    echo "[5/6] Object-CXR — already present, skipping"
fi

# ---------------------------------------------------------------------------
# 6. PadChest-GR (BIMCV, institutional-email data-use request)
# ---------------------------------------------------------------------------
if [ ! -d "$ROOT/padchest_gr" ]; then
    echo "[6/6] PadChest-GR — BIMCV"
    mkdir -p "$ROOT/padchest_gr"
    if [ -z "${BIMCV_TOKEN:-}" ]; then
        echo "  BIMCV_TOKEN not set; PadChest-GR requires a data-use request."
        echo "  Apply at: https://bimcv.cipf.es/bimcv-projects/padchest-gr/"
        echo "  After approval, place downloaded zips under $ROOT/padchest_gr"
    else
        # Placeholder; BIMCV issues per-request URLs in the approval email.
        echo "  BIMCV_TOKEN set; manual URL not encoded here — follow email instructions"
    fi
else
    echo "[6/6] PadChest-GR — already present, skipping"
fi

echo
echo "Download step complete. Next: bash scripts/path_b/build_unified_index.sh"
