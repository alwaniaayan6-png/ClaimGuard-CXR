#!/usr/bin/env bash
# End-to-end reproduction of every v6.0 result from a clean Modal volume.
#
# Prerequisites (see PLAN_V6_17DAY_NEURIPS.md Day 0):
#   * PadChest-GR RUA approved; raw data uploaded to /data/padchest_gr_raw
#     on the claimguard-v5-data Modal volume
#   * Anthropic API credits loaded (checked via secret `anthropic`)
#   * HuggingFace token present with MAIRA-2 + MedGemma access accepted
#     (Modal secret `huggingface`)
#   * v5 Tier 3 artifacts present (run by previous session; available at
#     /data/groundbench_v5/all/ and /data/checkpoints/claimguard_v5/)
#
# This script is idempotent: each step is skipped if its output already exists,
# so partial re-runs are safe.
#
# Runtime: approximately 48 hours of wall-clock for the full pipeline when
# run serially. Steps can be run in parallel by spawning multiple workflows
# at once (see PLAN_V6 Day 3-4 concurrency pattern).

set -euo pipefail

APP=claimguard-v6-orchestrator
VOL=claimguard-v5-data
cd "$(dirname "$0")/../.."

echo "=== v6.0 reproduction script ==="
date

echo
echo "Step 0: Deploy the v6 Modal orchestrator"
modal deploy v5/modal/v6_orchestrator.py

echo
echo "Step 1 (Day 2): RRG generation on 500 OpenI test images"
python -c "
import modal
f = modal.Function.from_name('$APP', 'rrg_generation')
call = f.spawn(max_images=500)
print('spawned rrg_generation:', call.object_id)
"

echo
echo "Step 2 (Day 3-4): Silver labeling (parallel)"
python -c "
import modal
g = modal.Function.from_name('$APP', 'silver_green')
call_g = g.spawn()
print('spawned silver_green:', call_g.object_id)
r = modal.Function.from_name('$APP', 'silver_radfact')
call_r = r.spawn(n_claims=1000)
print('spawned silver_radfact:', call_r.object_id)
v = modal.Function.from_name('$APP', 'silver_vert')
call_v = v.spawn(n_claims=1000)
print('spawned silver_vert:', call_v.object_id)
"

echo
echo "Wait-for-silver-labels gate: poll Modal volume for all three output JSONLs."
python -c "
import modal, time, sys
vol = modal.Volume.from_name('$VOL')
outputs = [
    '/v6_silver/green_labels.jsonl',
    '/v6_silver/radfact_labels.jsonl',
    '/v6_silver/vert_labels.jsonl',
]
deadline = time.time() + 60*60*4
while time.time() < deadline:
    try:
        files = {e.path for e in vol.iterdir('/v6_silver')}
    except Exception:
        files = set()
    missing = [o for o in outputs if o not in files]
    if not missing:
        print('all three silver-label JSONLs present; proceeding')
        sys.exit(0)
    print(f'still waiting on: {missing}')
    time.sleep(120)
print('timeout waiting for silver labels'); sys.exit(2)
"

echo
echo "Step 3 (Day 4 end): Silver ensemble combine"
python -c "
import modal
f = modal.Function.from_name('$APP', 'silver_ensemble')
call = f.spawn()
print('spawned silver_ensemble:', call.object_id)
"

echo
echo "Step 4 (Day 5): PadChest-GR assembly"
python -c "
import modal
f = modal.Function.from_name('$APP', 'padchest_gr_assemble')
call = f.spawn(translate=True)
print('spawned padchest_gr_assemble:', call.object_id)
call.get()
print('padchest_gr_assemble complete')
"

echo
echo "Step 5 (Day 6): Assemble 3-site GroundBench-v6 (Modal entrypoint, blocking)"
python -c "
import modal
f = modal.Function.from_name('$APP', 'assemble_v6_3site')
call = f.spawn()
print('spawned assemble_v6_3site:', call.object_id)
result = call.get()
print('assemble_v6_3site result:', result)
"

echo
echo "Step 6 (Day 6): Train v6.0-3site"
python -c "
import modal
f = modal.Function.from_name('$APP', 'train_v6_3site')
call = f.spawn()
print('spawned train_v6_3site:', call.object_id)
"

echo
echo "Step 7 (Day 7): 3-way leave-one-site-out training (parallel)"
python -c "
import modal
f = modal.Function.from_name('$APP', 'train_v6_loo')
for site in ('openi', 'chestx_det10', 'padchest_gr'):
    call = f.spawn(held_out_site=site)
    print(f'spawned train_v6_loo held_out={site}:', call.object_id)
"

echo
echo "Step 8 (Day 8): Baseline evaluation (7 detectors x 4 conditions)"
python -c "
import modal
f = modal.Function.from_name('$APP', 'baseline_eval')
call = f.spawn(max_rows=500)
print('spawned baseline_eval:', call.object_id)
"

echo
echo "Step 9 (Day 9): Hidden-state probe (MAIRA-2 then MedVersa)"
python -c "
import modal
f = modal.Function.from_name('$APP', 'hidden_state_probe')
for backbone in ('maira-2', 'medversa'):
    call = f.spawn(backbone=backbone)
    print(f'spawned hidden_state_probe backbone={backbone}:', call.object_id)
"

echo
echo "Step 10 (Day 10): Artifact audit"
python -c "
import modal
f = modal.Function.from_name('$APP', 'artifact_audit')
call = f.spawn()
print('spawned artifact_audit:', call.object_id)
"

echo
echo "Step 11 (Day 11): PadChest-GR radiologist validation"
python -c "
import modal
f = modal.Function.from_name('$APP', 'padchest_gr_validate')
call = f.spawn()
print('spawned padchest_gr_validate:', call.object_id)
"

echo
echo "=== reproduction pipeline spawned ==="
echo "Monitor: modal app logs $APP -f"
echo "Results: modal volume ls $VOL /v6_results/"
echo "Final status: modal volume get $VOL /v6_results/status.json ./status.json"
