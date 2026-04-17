#!/usr/bin/env bash
# ClaimGuard-CXR v5 — Phase 1 data acquisition
#
# Downloads all freely accessible (no-credentials) datasets.
# Run from the repo root:  bash v5/scripts/download_datasets.sh
#
# Prerequisites:
#   pip install kaggle
#   Place Kaggle API token at ~/.kaggle/kaggle.json  (kaggle.com > Account > API)
#   conda or pip install gdown (for Google Drive items if needed)
#
# What this does NOT download (requires PhysioNet credential):
#   MS-CXR, CheXpert-Plus, BRAX — obtain separately via PhysioNet / BIMCV

set -euo pipefail
DATA_ROOT="${DATA_ROOT:-$HOME/data}"

log() { echo "[$(date +%H:%M:%S)] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

# ─────────────────────────────────────────────────────────────────────────────
# 1. Check prerequisites
# ─────────────────────────────────────────────────────────────────────────────
command -v kaggle >/dev/null 2>&1 || die "kaggle CLI not found. Run: pip install kaggle"
[[ -f "$HOME/.kaggle/kaggle.json" ]] || die "~/.kaggle/kaggle.json missing. See kaggle.com/Account > API"
chmod 600 "$HOME/.kaggle/kaggle.json"

# ─────────────────────────────────────────────────────────────────────────────
# 2. RSNA Pneumonia Detection Challenge (Kaggle)
#    ~4 GB compressed  |  30,227 DICOM CXRs  |  pneumonia bboxes
# ─────────────────────────────────────────────────────────────────────────────
RSNA_DIR="$DATA_ROOT/rsna_pneumonia"
if [[ ! -d "$RSNA_DIR" || "$(ls -A "$RSNA_DIR" 2>/dev/null | wc -l)" -lt 10 ]]; then
    log "Downloading RSNA Pneumonia..."
    mkdir -p "$RSNA_DIR"
    kaggle competitions download -c rsna-pneumonia-detection-challenge -p "$RSNA_DIR"
    log "Unzipping RSNA..."
    unzip -q "$RSNA_DIR/rsna-pneumonia-detection-challenge.zip" -d "$RSNA_DIR"
    rm "$RSNA_DIR/rsna-pneumonia-detection-challenge.zip"
    log "RSNA done: $(find "$RSNA_DIR" -name '*.dcm' | wc -l) DICOMs"
else
    log "RSNA already present at $RSNA_DIR — skipping"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 3. SIIM-ACR Pneumothorax Segmentation (Kaggle)
#    ~8 GB compressed  |  12,047 CXRs  |  pneumothorax RLE masks
# ─────────────────────────────────────────────────────────────────────────────
SIIM_DIR="$DATA_ROOT/siim_acr_pneumothorax"
if [[ ! -d "$SIIM_DIR" || "$(ls -A "$SIIM_DIR" 2>/dev/null | wc -l)" -lt 10 ]]; then
    log "Downloading SIIM-ACR Pneumothorax..."
    mkdir -p "$SIIM_DIR"
    kaggle competitions download -c siim-acr-pneumothorax-segmentation -p "$SIIM_DIR"
    log "Unzipping SIIM..."
    unzip -q "$SIIM_DIR/siim-acr-pneumothorax-segmentation.zip" -d "$SIIM_DIR"
    rm "$SIIM_DIR/siim-acr-pneumothorax-segmentation.zip"
    log "SIIM done: $(find "$SIIM_DIR" -name '*.dcm' | wc -l) DICOMs"
else
    log "SIIM already present at $SIIM_DIR — skipping"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 4. Object-CXR (GitHub / JF Healthcare)
#    ~1.2 GB  |  9,000 CXRs  |  foreign object bboxes  |  JPEG
# ─────────────────────────────────────────────────────────────────────────────
OBJCXR_DIR="$DATA_ROOT/object_cxr"
if [[ ! -d "$OBJCXR_DIR" || "$(ls -A "$OBJCXR_DIR" 2>/dev/null | wc -l)" -lt 5 ]]; then
    log "Downloading Object-CXR..."
    git clone https://github.com/jfhealthcare/object-CXR "$OBJCXR_DIR" || {
        # fallback: direct HuggingFace datasets mirror
        python3 -c "
from datasets import load_dataset
ds = load_dataset('jfhealthcare/object_cxr')
print(ds)
"
    }
    log "Object-CXR done"
else
    log "Object-CXR already present at $OBJCXR_DIR — skipping"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 5. ChestX-Det10 (GitHub / Deepwise AI Lab)
#    ~3.5 GB  |  3,543 CXRs  |  10-class finding bboxes  |  PNG
# ─────────────────────────────────────────────────────────────────────────────
DETX_DIR="$DATA_ROOT/chestx_det10"
if [[ ! -d "$DETX_DIR" || "$(ls -A "$DETX_DIR" 2>/dev/null | wc -l)" -lt 5 ]]; then
    log "Downloading ChestX-Det10..."
    git clone https://github.com/Deepwise-AILab/ChestX-Det10-Dataset "$DETX_DIR"
    log "ChestX-Det10 done"
else
    log "ChestX-Det10 already present at $DETX_DIR — skipping"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 6. NIH ChestX-ray14 (NIH Box)
#    ~45 GB total (12 tarballs)  |  112,120 CXRs  |  14-class labels
#    NOTE: This downloads images only; bbox annotations are a separate file.
# ─────────────────────────────────────────────────────────────────────────────
NIH_DIR="$DATA_ROOT/nih_cxr14"
if [[ ! -d "$NIH_DIR" || "$(find "$NIH_DIR" -name '*.png' 2>/dev/null | wc -l)" -lt 1000 ]]; then
    log "Downloading NIH ChestX-ray14..."
    mkdir -p "$NIH_DIR"
    BASE_URL="https://nihcc.box.com/shared/static"
    declare -a URLS=(
        "vfJNXcgly21l85MkeabznkjpDSsu2Dco.tar.gz"
        "i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.tar.gz"
        "f1t00wrtdk94satdfb9olcolqx20z2jp.tar.gz"
        "0aowwzs5lhjrceb3qp67ahp0rd1l1etg.tar.gz"
        "v5e3goj22zr574iadmu9uzber59uqcfj.tar.gz"
        "asi7ikud9jwnkrnkj99jnpfkjdes7l6l.tar.gz"
        "jn1b4d1w3dh7d5s7tcpc1ilyh1sbtxla.tar.gz"
        "gmlwbgdenklh4nfifbgbovhstkao0h4c.tar.gz"
        "dvqmanyrqqigkgt401uscpvnaqr722uq.tar.gz"
        "hrqdzvj52fqwoipqmhtv4m55q066wudt.tar.gz"
        "bnmw8eq9oy589boevkm2fi9rl3mzplk4.tar.gz"
        "p5ts5i4km8ykf32v2nrknkzwp6wfez6w.tar.gz"
    )
    for url in "${URLS[@]}"; do
        fname="${url%%.*}.tar.gz"
        log "  Downloading NIH shard $url..."
        curl -s -L "$BASE_URL/$url" -o "$NIH_DIR/$fname"
        tar xzf "$NIH_DIR/$fname" -C "$NIH_DIR" && rm "$NIH_DIR/$fname"
    done
    # Bbox annotations file
    curl -s -L "https://nihcc.box.com/shared/static/x0cmgjy0jkk0hn6sduv9f84lkpsbyl43.csv" \
        -o "$NIH_DIR/BBox_List_2017.csv"
    # Data entry csv
    curl -s -L "https://nihcc.box.com/shared/static/uriayc85gfqga9gzcd60q51bdn03ztzq.csv" \
        -o "$NIH_DIR/Data_Entry_2017.csv"
    log "NIH done: $(find "$NIH_DIR" -name '*.png' | wc -l) PNGs"
else
    log "NIH ChestX-ray14 already present at $NIH_DIR — skipping"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 7. OpenI Indiana (open access)
#    ~3 MB JSON + DICOM  |  3,996 CXRs  |  radiologist reports
# ─────────────────────────────────────────────────────────────────────────────
OPENI_DIR="$DATA_ROOT/openi"
if [[ ! -d "$OPENI_DIR" || "$(ls -A "$OPENI_DIR" 2>/dev/null | wc -l)" -lt 3 ]]; then
    log "Downloading OpenI Indiana..."
    mkdir -p "$OPENI_DIR"
    curl -s -L "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_dcm.tgz" \
        -o "$OPENI_DIR/NLMCXR_dcm.tgz"
    curl -s -L "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz" \
        -o "$OPENI_DIR/NLMCXR_reports.tgz"
    tar xzf "$OPENI_DIR/NLMCXR_dcm.tgz" -C "$OPENI_DIR"
    tar xzf "$OPENI_DIR/NLMCXR_reports.tgz" -C "$OPENI_DIR"
    rm "$OPENI_DIR/NLMCXR_dcm.tgz" "$OPENI_DIR/NLMCXR_reports.tgz"
    log "OpenI done: $(find "$OPENI_DIR" -name '*.dcm' | wc -l) DICOMs"
else
    log "OpenI already present at $OPENI_DIR — skipping"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
log "=== Phase 1 data acquisition complete ==="
log "Data root: $DATA_ROOT"
echo ""
echo "Datasets requiring separate access (not downloaded here):"
echo "  MS-CXR          — PhysioNet (MIMIC-CXR access required)"
echo "  CheXpert-Plus   — Already in Laughney lab at /data/chexpert_plus"
echo "  PadChest        — BIMCV registration: https://bimcv.cipf.es"
echo "  BRAX            — IEEE DataPort registration"
echo ""
echo "Next: modal volume put claimguard-v5-data \$HOME/data/<site> /<site>"
