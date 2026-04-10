#!/bin/bash
# ClaimGuard-CXR v2 — RunPod H100 Setup
# Run this ONCE after SSH'ing into your RunPod pod.
# Installs all dependencies and creates directory structure.

set -e

echo "=== ClaimGuard-CXR RunPod Setup ==="
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Directory structure
mkdir -p /workspace/data/eval_data
mkdir -p /workspace/checkpoints/progressive_nli
mkdir -p /workspace/checkpoints/hypothesis_only
mkdir -p /workspace/results/baselines
mkdir -p /workspace/results/eval_deberta_v2

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --quiet --upgrade pip
pip install --quiet \
    torch>=2.1.0 \
    transformers==4.40.0 \
    accelerate>=0.27.0 \
    datasets>=2.18.0 \
    numpy>=1.24.0 \
    pandas>=2.0.0 \
    scikit-learn>=1.3.0 \
    scipy>=1.11.0 \
    tqdm>=4.66.0 \
    sentencepiece>=0.1.99 \
    Pillow>=10.0.0 \
    pyyaml>=6.0.1

echo ""
echo "=== Setup complete ==="
echo "Directory structure:"
ls -la /workspace/data/
echo ""
echo "Next: upload data with scp, then run: bash runpod/run_all.sh"
