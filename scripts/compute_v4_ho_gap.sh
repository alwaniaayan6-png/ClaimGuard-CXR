#!/usr/bin/env bash
# Compute the v3 → v4 hypothesis-only gap delta after Task 3c
# (R-Drop refinement) and the parallel HO baseline training have both
# completed.
#
# Inputs (all on the claimguard-data Modal volume):
#   /data/checkpoints/verifier_binary_v3/best_verifier.pt
#       → v3 base checkpoint (val_acc 0.9877)
#   /data/checkpoints/verifier_binary_v4_rdrop/best_verifier.pt
#       → v4 R-Drop refined checkpoint (this script's main input)
#   /data/checkpoints/hypothesis_only_v3/best_hypothesis_only.pt
#       → fresh RoBERTa-large trained on (claim, masked_evidence)
#   /data/checkpoints/hypothesis_only_v3/hypothesis_only_results.json
#       → contains the HO best val accuracy
#
# This script:
#   1. Downloads the v4 checkpoint, the HO results JSON, and the v4
#      training history JSON.
#   2. Reads HO val_acc from hypothesis_only_results.json.
#   3. Runs `modal run` against the v4 checkpoint on the v3 synthetic
#      eval set to get v4 full accuracy.
#   4. Runs the same on the OpenI cross-dataset eval to get v4 OpenI
#      accuracy and per-α cfBH FDR.
#   5. Computes:
#        v3_HO_gap = 0.9877 - HO_acc   (v3 reference number)
#        v4_HO_gap = v4_acc - HO_acc   (the headline)
#   6. Prints a final summary block + writes results/v4/summary.json.
#
# Usage:
#   bash scripts/compute_v4_ho_gap.sh
#
# Must be run from the verifact repo root.

set -euo pipefail

REPO_ROOT="/Users/aayanalwani/VeriFact/verifact"
cd "$REPO_ROOT"

V3_VAL_ACC="0.9877"  # Locked from Task 2 retrain (epoch 3 best)

echo "===================================================="
echo " Task 3c v4 HO-gap aggregator"
echo "===================================================="
echo

# ---- Step 1: download artifacts ----
mkdir -p /tmp/v4
echo "[1/6] Downloading v4 + HO artifacts from Modal volume..."

modal volume get claimguard-data /checkpoints/verifier_binary_v4_rdrop/best_verifier.pt /tmp/v4/v4_best_verifier.pt --force 2>&1 | tail -3 || {
    echo "ERROR: v4 checkpoint not yet on volume.  Task 3c may still be running." >&2
    echo "Check: python3 -c 'import modal; print(modal.FunctionCall.from_id(\"fc-01KP9VW0P4G8QXNHAQYK2GCFNR\").get(timeout=1))'" >&2
    exit 1
}
modal volume get claimguard-data /checkpoints/verifier_binary_v4_rdrop/training_history.json /tmp/v4/training_history.json --force 2>&1 | tail -3 || {
    echo "WARN: v4 training_history.json missing, continuing"
}
modal volume get claimguard-data /checkpoints/hypothesis_only_v3/hypothesis_only_results.json /tmp/v4/hypothesis_only_results.json --force 2>&1 | tail -3 || {
    echo "ERROR: HO baseline results not yet on volume.  HO training may still be running." >&2
    exit 2
}
echo "  OK"
echo

# ---- Step 2: parse HO baseline ----
echo "[2/6] Reading HO baseline accuracy..."
HO_ACC=$(python3 -c "
import json
with open('/tmp/v4/hypothesis_only_results.json') as f:
    d = json.load(f)
# Try common keys
for key in ['val_acc', 'best_val_acc', 'final_val_acc', 'val_accuracy', 'best_validation_accuracy']:
    if key in d:
        print(d[key])
        break
else:
    print('UNKNOWN')
")
echo "  HO val_acc = $HO_ACC"
if [[ "$HO_ACC" == "UNKNOWN" ]]; then
    echo "ERROR: could not find val_acc in HO results JSON.  Inspect manually:" >&2
    cat /tmp/v4/hypothesis_only_results.json >&2
    exit 3
fi
echo

# ---- Step 3: fire v4 modal eval on synthetic v3 test set ----
echo "[3/6] Firing v4 eval on synthetic v3 test set (Modal H100, ~10 min)..."
modal run scripts/modal_run_evaluation.py \
    --cal-claims-path /data/eval_data_v3/calibration_claims.json \
    --test-claims-path /data/eval_data_v3/test_claims.json \
    --output-dir /data/eval_results_v3_v4_rdrop \
    --verifier-path /data/checkpoints/verifier_binary_v4_rdrop/best_verifier.pt \
    --num-classes 2 \
    > /tmp/v4/synthetic_eval.log 2>&1
echo "  done; log at /tmp/v4/synthetic_eval.log"
echo

# ---- Step 4: fire v4 modal eval on OpenI test set ----
echo "[4/6] Firing v4 eval on OpenI cross-dataset test set..."
modal run scripts/modal_run_evaluation.py \
    --cal-claims-path /data/eval_data_openi/calibration_claims.json \
    --test-claims-path /data/eval_data_openi/test_claims.json \
    --output-dir /data/eval_results_openi_v4_rdrop \
    --verifier-path /data/checkpoints/verifier_binary_v4_rdrop/best_verifier.pt \
    --num-classes 2 \
    > /tmp/v4/openi_eval.log 2>&1
echo "  done; log at /tmp/v4/openi_eval.log"
echo

# ---- Step 5: parse final accuracies ----
echo "[5/6] Parsing v4 results from Modal eval output..."
modal volume get claimguard-data /eval_results_v3_v4_rdrop/full_results.json /tmp/v4/v4_synthetic.json --force 2>&1 | tail -3
modal volume get claimguard-data /eval_results_openi_v4_rdrop/full_results.json /tmp/v4/v4_openi.json --force 2>&1 | tail -3

V4_SYNTH_ACC=$(python3 -c "
import json
with open('/tmp/v4/v4_synthetic.json') as f:
    d = json.load(f)
print(d['verifier_metrics']['test']['accuracy'])
")
V4_OPENI_ACC=$(python3 -c "
import json
with open('/tmp/v4/v4_openi.json') as f:
    d = json.load(f)
print(d['verifier_metrics']['test']['accuracy'])
")
echo "  v4 synthetic test acc: $V4_SYNTH_ACC"
echo "  v4 OpenI test acc:     $V4_OPENI_ACC"
echo

# ---- Step 6: compute HO gaps + summary ----
echo "[6/6] Computing HO gaps and writing summary..."
python3 << PY
import json
v3_acc = $V3_VAL_ACC
v4_acc = $V4_SYNTH_ACC
v4_openi = $V4_OPENI_ACC
ho_acc = $HO_ACC

v3_ho_gap = v3_acc - ho_acc
v4_ho_gap = v4_acc - ho_acc

# Read OpenI cfBH FDR per α
with open('/tmp/v4/v4_openi.json') as f:
    openi = json.load(f)
fdr_by_alpha = {
    k: v['fdr']
    for k, v in openi.get('conformal_results', {}).items()
}

summary = {
    'v3_synthetic_acc': v3_acc,
    'v4_synthetic_acc': v4_acc,
    'v4_synthetic_acc_delta': v4_acc - v3_acc,
    'v4_openi_acc': v4_openi,
    'ho_baseline_acc': ho_acc,
    'v3_ho_gap_pp': round((v3_ho_gap) * 100, 2),
    'v4_ho_gap_pp': round((v4_ho_gap) * 100, 2),
    'ho_gap_growth_pp': round((v4_ho_gap - v3_ho_gap) * 100, 2),
    'v4_openi_fdr_by_alpha': fdr_by_alpha,
    'plan_target_ho_gap_pp': 5.0,
    'meets_target': (v4_ho_gap * 100) >= 5.0,
}

import os
os.makedirs('results/v4', exist_ok=True)
with open('results/v4/summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print()
print('=' * 60)
print(' v4 HO-GAP HEADLINE')
print('=' * 60)
print(f'  v3 synthetic acc:      {v3_acc:.4f}')
print(f'  v4 synthetic acc:      {v4_acc:.4f}  (delta {(v4_acc-v3_acc)*100:+.2f} pp)')
print(f'  v4 OpenI acc:          {v4_openi:.4f}')
print(f'  HO baseline acc:       {ho_acc:.4f}')
print(f'  v3 HO gap:             {v3_ho_gap*100:.2f} pp')
print(f'  v4 HO gap:             {v4_ho_gap*100:.2f} pp')
print(f'  HO gap growth:         {(v4_ho_gap-v3_ho_gap)*100:+.2f} pp')
print(f'  Plan target (>=5 pp):  {"PASS" if (v4_ho_gap*100) >= 5 else "FAIL"}')
print()
print('  v4 OpenI cfBH FDR by alpha:')
for k, v in sorted(fdr_by_alpha.items()):
    print(f'    {k}: {v:.4f}')
print()
print('  Summary written to results/v4/summary.json')
PY
