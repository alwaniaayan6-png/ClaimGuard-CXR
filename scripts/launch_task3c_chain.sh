#!/usr/bin/env bash
# Task 3b → Task 3c chain.
#
# Run this AFTER Task 3b counterfactual generation completes.  It:
#
#   1. Verifies the local counterfactual_pairs_v3.json exists and is
#      not empty.
#   2. Uploads it to the claimguard-data Modal volume.
#   3. Verifies the upload landed.
#   4. Spawns the R-Drop refinement trainer via the deploy+spawn
#      launcher (so the call survives local Mac sleep).
#   5. Prints the spawned FunctionCall ID.
#
# Usage:
#   bash scripts/launch_task3c_chain.sh
#
# Or with custom paths:
#   LOCAL_JSON=/tmp/task3b/counterfactual_pairs_v3.json \
#     bash scripts/launch_task3c_chain.sh

set -euo pipefail

LOCAL_JSON="${LOCAL_JSON:-/tmp/task3b/counterfactual_pairs_v3.json}"
REMOTE_PATH="${REMOTE_PATH:-/counterfactual_preference_pairs_v3.json}"
VOLUME_NAME="${VOLUME_NAME:-claimguard-data}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/checkpoints/verifier_binary_v4_rdrop}"
BASE_CKPT="${BASE_CKPT:-/data/checkpoints/verifier_binary_v3/best_verifier.pt}"

echo "===================================================="
echo " Task 3b → Task 3c chain launcher"
echo "===================================================="
echo "  local JSON:    $LOCAL_JSON"
echo "  remote path:   /data$REMOTE_PATH"
echo "  output dir:    $OUTPUT_DIR"
echo "  base ckpt:     $BASE_CKPT"
echo

# ---- Step 1: verify local file ----
echo "[1/5] Verifying local counterfactual JSON..."
if [[ ! -f "$LOCAL_JSON" ]]; then
    echo "ERROR: local file not found at $LOCAL_JSON" >&2
    echo "Did Task 3b complete?  Check /tmp/task3b_full.log" >&2
    exit 1
fi
LOCAL_SIZE=$(stat -f%z "$LOCAL_JSON" 2>/dev/null || stat -c%s "$LOCAL_JSON")
echo "  local file size: $LOCAL_SIZE bytes"
if [[ "$LOCAL_SIZE" -lt 1000 ]]; then
    echo "ERROR: local file is suspiciously small (<1KB)." >&2
    echo "Task 3b probably aborted — check the log." >&2
    exit 2
fi
N_ROWS=$(python3 -c "import json; d=json.load(open('$LOCAL_JSON')); print(len(d))")
echo "  preference pair rows: $N_ROWS"
if [[ "$N_ROWS" -lt 100 ]]; then
    echo "ERROR: only $N_ROWS preference pair rows — too few for R-Drop training." >&2
    exit 3
fi
echo "  OK"
echo

# ---- Step 2: upload to Modal volume ----
echo "[2/5] Uploading to Modal volume $VOLUME_NAME:$REMOTE_PATH ..."
modal volume put --force "$VOLUME_NAME" "$LOCAL_JSON" "$REMOTE_PATH"
echo "  OK"
echo

# ---- Step 3: verify upload ----
echo "[3/5] Verifying upload landed..."
if ! modal volume ls "$VOLUME_NAME" "$REMOTE_PATH" >/dev/null 2>&1; then
    # Try listing parent dir
    PARENT=$(dirname "$REMOTE_PATH")
    BASENAME=$(basename "$REMOTE_PATH")
    if ! modal volume ls "$VOLUME_NAME" "$PARENT" 2>&1 | grep -q "$BASENAME"; then
        echo "ERROR: uploaded file not visible on volume after put." >&2
        echo "Check 'modal volume ls $VOLUME_NAME $PARENT'." >&2
        exit 4
    fi
fi
echo "  OK"
echo

# ---- Step 4: ensure trainer is deployed ----
echo "[4/5] Verifying claimguard-dpo-refinement is deployed..."
if ! modal app list 2>&1 | grep -q "claimguard-dpo-refinement.*deployed"; then
    echo "  trainer not deployed — deploying now..."
    modal deploy scripts/modal_train_dpo_refinement.py
fi
echo "  OK"
echo

# ---- Step 5: spawn the trainer ----
echo "[5/5] Spawning R-Drop refinement trainer..."
python3 scripts/launch_task3c_detached.py \
    --preference-data "/data$REMOTE_PATH" \
    --base-checkpoint "$BASE_CKPT" \
    --output-dir "$OUTPUT_DIR" \
    --loss-mode consistency
echo
echo "===================================================="
echo " Task 3c spawned successfully"
echo "===================================================="
echo " The training run will continue on Modal H100"
echo " regardless of what happens to this Mac.  Check the"
echo " sidecar JSON at /tmp/task3c_function_call.json for"
echo " the FunctionCall ID needed to poll for completion."
echo
