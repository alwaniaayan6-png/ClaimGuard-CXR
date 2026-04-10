#!/bin/bash
# Download CheXpert Plus dataset (publicly available, no credentialing needed)
# Run from the verifact/ directory: bash scripts/download_data.sh
#
# CheXpert Plus: 223K CXR image-report pairs from Stanford
# ~400GB for full images, ~2GB for reports/metadata only

set -e

DATA_DIR="${DATA_ROOT:-$HOME/data/claimguard}"
mkdir -p "$DATA_DIR"

echo "============================================"
echo "ClaimGuard-CXR Data Download Script"
echo "============================================"
echo "Data directory: $DATA_DIR"
echo ""

# -----------------------------------------------
# Option 1: Download reports/metadata only (fast, ~2GB)
# This is enough for: decomposer training, retriever index,
# verifier training (text only), conformal calibration
# -----------------------------------------------
echo "[1/3] Downloading CheXpert Plus metadata and reports..."
echo "  Source: Stanford AIMI (no account needed)"
echo ""
echo "  >>> MANUAL STEP REQUIRED <<<"
echo "  Go to: https://stanfordaimi.azurewebsites.net/datasets/5158c524-d3ab-4e02-96e9-6ee9efc110a1"
echo "  Download the metadata CSV and reports files"
echo "  Place them in: $DATA_DIR/chexpert-plus/"
echo ""

# If you have the azcopy tool (Azure CLI), you can automate:
# azcopy copy "https://stanfordaimi.blob.core.windows.net/datasets/CheXpert-Plus" "$DATA_DIR/chexpert-plus" --recursive

# -----------------------------------------------
# Option 2: IU X-Ray (small, ~1GB, open access)
# For external validation
# -----------------------------------------------
echo "[2/3] Downloading IU X-Ray dataset..."
IU_DIR="$DATA_DIR/iu-xray"
mkdir -p "$IU_DIR"

if [ ! -f "$IU_DIR/indiana_reports.csv" ]; then
    echo "  Downloading IU X-Ray reports..."
    # IU X-Ray is hosted on multiple mirrors
    # Try kaggle first, then direct download
    if command -v kaggle &> /dev/null; then
        kaggle datasets download -d raddar/chest-xrays-indiana-university -p "$IU_DIR" --unzip
    else
        echo "  Install kaggle CLI (pip install kaggle) or download manually from:"
        echo "  https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university"
        echo "  Place in: $IU_DIR/"
    fi
else
    echo "  IU X-Ray already downloaded"
fi

# -----------------------------------------------
# Option 3: Model weights (auto-downloaded by HuggingFace)
# Just pre-cache them to avoid download during training
# -----------------------------------------------
echo "[3/3] Pre-caching model weights..."
python3 -c "
from transformers import AutoModel, AutoTokenizer
print('  Caching DeBERTa-v3-large...')
AutoTokenizer.from_pretrained('microsoft/deberta-v3-large')
AutoModel.from_pretrained('microsoft/deberta-v3-large')
print('  Caching MedCPT encoders...')
AutoTokenizer.from_pretrained('ncbi/MedCPT-Query-Encoder')
AutoModel.from_pretrained('ncbi/MedCPT-Query-Encoder')
AutoModel.from_pretrained('ncbi/MedCPT-Article-Encoder')
print('  Done caching model weights.')
" 2>/dev/null || echo "  (Some model downloads may require running with GPU — will auto-download during training)"

echo ""
echo "============================================"
echo "Download complete!"
echo ""
echo "Next steps:"
echo "  1. Download CheXpert Plus manually from the Stanford AIMI link above"
echo "  2. Place data in $DATA_DIR/chexpert-plus/"
echo "  3. Run: python data/preprocessing/patient_splits.py --data-root $DATA_DIR/chexpert-plus --output-dir $DATA_DIR/splits"
echo "  4. Upload splits to Modal volume: modal volume put claimguard-data $DATA_DIR/splits /splits"
echo "============================================"
