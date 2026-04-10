#!/bin/bash
# ClaimGuard-CXR v2 — Master Execution Script for RunPod H100
#
# Runs the full text-only pipeline sequentially:
#   1. Progressive NLI (MNLI -> MedNLI -> ClaimGuard): ~2.5 hrs
#   2. Full v2 evaluation with conformal FDR: ~10 min
#   3. All baselines (hypothesis-only, DeBERTa NLI, RadFlag): ~1 hr
#   4. Summary comparison table
#
# Total: ~3.5-4 hours on H100. Estimated cost: ~$10 at $2.49/hr.
#
# Prerequisites:
#   - RunPod H100 pod with SSH access
#   - Data uploaded to /workspace/data/
#   - Dependencies installed via setup_runpod.sh
#
# Usage:
#   cd /workspace/verifact
#   bash runpod/run_all.sh 2>&1 | tee /workspace/results/run_all.log

set -e  # Exit on first error

WORKSPACE="/workspace"
DATA_DIR="$WORKSPACE/data"
CKPT_DIR="$WORKSPACE/checkpoints"
RESULTS_DIR="$WORKSPACE/results"
SCRIPTS_DIR="$WORKSPACE/verifact/runpod"

echo "=============================================="
echo "ClaimGuard-CXR v2 — Full Pipeline"
echo "=============================================="
echo "Start time: $(date)"
echo ""

# Verify data exists
echo "Checking data files..."
for f in "$DATA_DIR/verifier_training_data_v2.json" \
         "$DATA_DIR/eval_data/calibration_claims.json" \
         "$DATA_DIR/eval_data/test_claims.json"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Missing $f"
        echo "Upload data first. See upload instructions below."
        echo ""
        echo "From your Mac:"
        echo "  scp /Users/aayanalwani/data/claimguard/verifier_training_data_v2.json RUNPOD:$DATA_DIR/"
        echo "  scp /Users/aayanalwani/data/claimguard/eval_data/*.json RUNPOD:$DATA_DIR/eval_data/"
        exit 1
    fi
done
echo "All data files present."
echo ""

# GPU check
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "No GPU detected (CPU mode)"
echo ""

# Create output dirs
mkdir -p "$CKPT_DIR/progressive_nli" "$RESULTS_DIR/baselines" "$RESULTS_DIR/eval_deberta_v2"

# ============================================
# STEP 1: Progressive NLI Training
# ============================================
echo "=============================================="
echo "STEP 1: Progressive NLI (MNLI -> MedNLI -> ClaimGuard)"
echo "Estimated time: 2-3 hours"
echo "=============================================="

PROGRESSIVE_CKPT="$CKPT_DIR/progressive_nli/final/best_verifier.pt"
if [ -f "$PROGRESSIVE_CKPT" ]; then
    echo "Checkpoint already exists at $PROGRESSIVE_CKPT. Skipping training."
else
    python "$SCRIPTS_DIR/run_progressive_nli.py" \
        --data-path "$DATA_DIR/verifier_training_data_v2.json" \
        --output-dir "$CKPT_DIR/progressive_nli" \
        --model-name "microsoft/deberta-v3-large" \
        --mnli-epochs 1 \
        --mednli-epochs 3 \
        --claimguard-epochs 5 \
        --batch-size 16 \
        --grad-accum 4 \
        --mnli-subsample 100000

    if [ ! -f "$PROGRESSIVE_CKPT" ]; then
        echo "ERROR: Progressive NLI training failed. No checkpoint produced."
        exit 1
    fi
fi
echo ""
echo "Step 1 complete. Checkpoint: $PROGRESSIVE_CKPT"
echo "Time: $(date)"
echo ""

# ============================================
# STEP 2: Full v2 Evaluation
# ============================================
echo "=============================================="
echo "STEP 2: Full v2 Evaluation (DeBERTa + Conformal FDR)"
echo "Estimated time: 10-15 minutes"
echo "=============================================="

python "$SCRIPTS_DIR/run_eval.py" \
    --checkpoint "$PROGRESSIVE_CKPT" \
    --cal-claims "$DATA_DIR/eval_data/calibration_claims.json" \
    --test-claims "$DATA_DIR/eval_data/test_claims.json" \
    --output-dir "$RESULTS_DIR/eval_deberta_v2" \
    --model-type v2 \
    --batch-size 32

echo ""
echo "Step 2 complete. Results: $RESULTS_DIR/eval_deberta_v2/full_results.json"
echo "Time: $(date)"
echo ""

# ============================================
# STEP 3: All Baselines
# ============================================
echo "=============================================="
echo "STEP 3: Baselines (Hypothesis-only, DeBERTa NLI, RadFlag)"
echo "Estimated time: 1-1.5 hours"
echo "=============================================="

python "$SCRIPTS_DIR/run_baselines.py" \
    --training-data "$DATA_DIR/verifier_training_data_v2.json" \
    --test-claims "$DATA_DIR/eval_data/test_claims.json" \
    --output-dir "$RESULTS_DIR/baselines"

echo ""
echo "Step 3 complete. Results in: $RESULTS_DIR/baselines/"
echo "Time: $(date)"
echo ""

# ============================================
# STEP 4: Summary
# ============================================
echo "=============================================="
echo "PIPELINE COMPLETE"
echo "=============================================="
echo "End time: $(date)"
echo ""
echo "=== Results Summary ==="
echo ""

# Print v2 eval results
if [ -f "$RESULTS_DIR/eval_deberta_v2/full_results.json" ]; then
    echo "--- ClaimGuard v2 (DeBERTa + Progressive NLI) ---"
    python -c "
import json
with open('$RESULTS_DIR/eval_deberta_v2/full_results.json') as f:
    r = json.load(f)
c = r['classification']
print(f\"  Accuracy:      {c['accuracy']:.4f}\")
print(f\"  Macro F1:      {c['macro_f1']:.4f}\")
print(f\"  Contra Recall: {c['contra_recall']:.4f}\")
print(f\"  AUROC:         {c['auroc']:.4f}\")
print(f\"  ECE:           {c['ece']:.4f}\")
print(f\"  Temperature:   {r['temperature']:.4f}\")
print()
print('  Conformal FDR:')
for alpha, cr in r['conformal'].items():
    print(f\"    alpha={alpha}: FDR={cr['fdr']:.4f}, power={cr['power']:.4f}, n_green={cr['n_green']}\")
"
    echo ""
fi

# Print baseline results
echo "--- Baselines ---"
for f in "$RESULTS_DIR/baselines"/*.json; do
    if [ -f "$f" ]; then
        python -c "
import json
with open('$f') as fh:
    r = json.load(fh)
print(f\"  {r['method']}: acc={r['accuracy']:.4f}, contra_recall={r['contra_recall']:.4f}, auroc={r.get('auroc', 'N/A')}\")
"
    fi
done

echo ""
echo "All results saved to $RESULTS_DIR/"
echo "Download to Mac: scp -r RUNPOD:$RESULTS_DIR/ ~/VeriFact/verifact/results/"
