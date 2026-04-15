#!/usr/bin/env bash
# Task 9 detach-friendly wait-and-launch wrapper.
#
# Fired once by the user with:
#
#     nohup bash scripts/task9_wait_and_launch.sh > /tmp/task9_pipeline.log 2>&1 &
#     disown
#
# Behavior:
#   1. Waits for the OpenI images upload (modal volume put) to complete.
#   2. Verifies the images landed on the volume.
#   3. Fires the detach launcher via `python3 scripts/launch_task9_detached.py`
#      which in turn calls Function.from_name().spawn() + .get() for the
#      three Task 9 stages (run A generation, run B generation, scoring).
#   4. Logs everything to /tmp/task9_pipeline.log.
#
# The combination of nohup + disown + caffeinate (inside) keeps the
# launcher alive while the Mac stays on AC power. If the Mac goes to
# sleep on battery, caffeinate prevents it. If the Mac powers off
# entirely, the launcher dies mid-chain — but the sidecar JSON at
# /tmp/task9_function_calls.json preserves the spawned Modal
# FunctionCall handles so a follow-up process can resume with:
#
#     python3 -c "import modal; modal.FunctionCall.from_id('$CALL_ID').get()"

set -u  # fail on unset variables, but don't exit on error so we log failures

UPLOAD_PID="${1:-91885}"  # default to the currently-running upload
REPO_ROOT="/Users/aayanalwani/VeriFact/verifact"
LAUNCHER="scripts/launch_task9_detached.py"
PIPELINE_LOG="/tmp/task9_pipeline.log"
UPLOAD_LOG="/tmp/openi_upload.log"
HANDLES_JSON="/tmp/task9_function_calls.json"

log() {
    printf '%s | %s\n' "$(date -Iseconds)" "$*"
}

log "=== Task 9 wait-and-launch pipeline started ==="
log "Upload PID to watch: $UPLOAD_PID"
log "Repo root:           $REPO_ROOT"
log "Launcher:            $LAUNCHER"
log "Pipeline log:        $PIPELINE_LOG"
log "Handles JSON:        $HANDLES_JSON"

cd "$REPO_ROOT" || { log "ERROR: cannot cd to $REPO_ROOT"; exit 1; }

# ---- Step 1: wait for the upload process to exit ----
log "Stage 1: waiting for upload PID $UPLOAD_PID to exit..."
wait_start=$(date +%s)
while kill -0 "$UPLOAD_PID" 2>/dev/null; do
    sleep 30
    elapsed=$(( $(date +%s) - wait_start ))
    log "  still waiting (upload elapsed: ${elapsed}s)"
done
wait_elapsed=$(( $(date +%s) - wait_start ))
log "Upload PID $UPLOAD_PID exited after ${wait_elapsed}s of waiting."
log "Last 10 lines of upload log:"
tail -10 "$UPLOAD_LOG" 2>&1 | sed 's/^/    /'

# ---- Step 2: verify the volume actually has the images ----
log "Stage 2: verifying /openi_images/ on claimguard-data volume..."
volume_count=$(modal volume ls claimguard-data /openi_images/ 2>/dev/null | wc -l | tr -d ' ')
log "Volume /openi_images/ entry count: $volume_count"
if [ "$volume_count" -lt 100 ]; then
    log "ERROR: volume has fewer than 100 entries — upload likely failed."
    log "Check $UPLOAD_LOG for the modal volume put error."
    log "You may need to retry the upload manually."
    exit 2
fi
log "Volume verification OK."

# ---- Step 3: fire the detach launcher ----
log "Stage 3: launching the Task 9 spawn chain via $LAUNCHER..."
log "This will spawn run A + run B generation (parallel on Modal H100)"
log "then the scoring phase. Expected total wall time: ~50-80 min."
log "Launcher progress goes to $PIPELINE_LOG (this file)."
log ""

# Run inside caffeinate so the Mac doesn't sleep mid-.get().
# stdin redirected from /dev/null so the launcher doesn't block on tty.
caffeinate -i python3 "$LAUNCHER" \
    --handles-path "$HANDLES_JSON" \
    < /dev/null 2>&1 | sed 's/^/LAUNCHER | /'
# ${PIPESTATUS[0]} is python3's exit code, NOT sed's.  Without this,
# any launcher failure would be masked by sed's success exit code 0,
# and the pipeline wrapper would falsely report "COMPLETED
# successfully" even when the launcher crashed.  This bit me on the
# 2026-04-15 Task 9 run when the generation code hit a KeyError and
# the wrapper declared victory.
launcher_exit=${PIPESTATUS[0]}

log ""
log "Launcher exited with code $launcher_exit"
log "Handles JSON: $HANDLES_JSON"
if [ -f "$HANDLES_JSON" ]; then
    log "Handles JSON contents:"
    cat "$HANDLES_JSON" 2>&1 | sed 's/^/    /'
fi

if [ "$launcher_exit" -eq 0 ]; then
    log "=== Task 9 pipeline COMPLETED successfully ==="
    log "Results at /data/same_model_experiment/gate_demo.json on the volume."
    log "Download with:"
    log "    modal volume get claimguard-data /same_model_experiment/ /tmp/task9_results/"
else
    log "=== Task 9 pipeline FAILED at stage 3 (launcher exit code $launcher_exit) ==="
    log "Check $HANDLES_JSON for the Modal FunctionCall handles to resume."
fi

exit $launcher_exit
