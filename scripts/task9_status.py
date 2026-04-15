"""Task 9 status checker — wake-up utility.

Run this any time to see where the in-flight Task 9 runs stand.
Works regardless of whether the local launcher is alive, because
everything is tracked via Modal FunctionCall IDs saved to sidecar
JSONs.

Usage:

    python3 scripts/task9_status.py

    # Or point at a specific sidecar:
    python3 scripts/task9_status.py \\
        --sidecar /tmp/task9_function_calls_v3.json

Prints a table like:

    SIDECAR: /tmp/task9_function_calls_v3.json
      mode              = blocking_generation_in_flight
      saved_at          = 2026-04-15T05:53:31Z

      Run A (chexagent generation)
        id        = fc-01KP7V7C8S2PZ6ZNVWAE35ETVJ
        status    = completed
        result    = {'total_claims': 1, 'model_used': 'StanfordAIMI/CheXagent-8b', ...}

      Run B (chexagent generation)
        id        = fc-01KP7V7CBH8KDQ0DR0Q3VE11WV
        status    = completed
        result    = {'total_claims': 0, 'model_used': 'StanfordAIMI/CheXagent-8b', ...}

      Scoring (provenance gate demo)
        id        = (not yet spawned)

    ORCHESTRATOR: /tmp/task9_orchestrator_handle.json
      call_id   = fc-01KP7W6R2SXNYPQ05VMFJQ2Q49
      status    = running
      poll_note = server-side polling loop, waiting for workbooks

    SMOKE TEST: /tmp/task9_smoke_call.json
      call_id   = fc-01KP8KFNQYVBXEEZRJQP9G74SG
      status    = completed
      result    = {'total_claims': 14, 'total_reports': 5, ...}

    VOLUME CONTENTS (/same_model_experiment):
      annotation_workbook_run_a.json      3.2 KB (1 claim — BROKEN, rerun needed)
      annotation_workbook_run_b.json        42 B (0 claims — BROKEN, rerun needed)
      smoke/annotation_workbook_smoke.json 41 KB (14 claims — healthy)

    RECOMMENDED NEXT ACTION:
      Smoke test produced 14 non-empty claims → prompt fix worked.
      Fire the full run via:
          nohup python3 scripts/launch_task9_detached.py \\
              --fire-and-forget > /tmp/task9_fire_full.log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

DEFAULT_SIDECARS = (
    "/tmp/task9_function_calls.json",        # original v1 attempt
    "/tmp/task9_function_calls_v3.json",     # v3 re-fire
    "/tmp/task9_orchestrator_handle.json",   # CPU orchestrator
    "/tmp/task9_smoke_call.json",            # 5-image smoke test
)

VOLUME_NAME = "claimguard-data"
SAME_MODEL_PATH = "/same_model_experiment/"


def _status_of_call(call_id: str) -> dict[str, Any]:
    """Return ``{status, result, error}`` for a FunctionCall id.

    Uses Modal's poll API — non-blocking.  Possible status values:
        * ``completed`` — result available
        * ``running``   — still in flight
        * ``failed``    — raised an exception; error field populated
        * ``missing``   — FunctionCall not found on server
    """
    try:
        import modal
    except ImportError:
        return {"status": "unknown", "error": "modal SDK not installed"}

    try:
        call = modal.FunctionCall.from_id(call_id)
    except Exception as e:  # noqa: BLE001
        return {"status": "missing", "error": str(e)}

    # Use a 1-second timeout as a cheap "is it done?" poll.
    try:
        result = call.get(timeout=1)
        return {"status": "completed", "result": result}
    except TimeoutError:
        return {"status": "running"}
    except Exception as e:  # noqa: BLE001
        return {"status": "failed", "error": str(e)[:400]}


def _fmt_kv(key: str, value: Any, indent: int = 4) -> str:
    pad = " " * indent
    if isinstance(value, dict):
        head = f"{pad}{key}:"
        body = "\n".join(
            f"{pad}  {k}: {v}" for k, v in value.items()
        )
        return head + "\n" + body
    return f"{pad}{key:<12} = {value}"


def _report_sidecar(path: str) -> None:
    if not os.path.exists(path):
        return
    print(f"SIDECAR: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:  # noqa: BLE001
        print(f"  (could not parse: {e})")
        print()
        return

    if "mode" in data:
        print(f"  mode              = {data.get('mode', '?')}")
    if "saved_at" in data:
        print(f"  saved_at          = {data.get('saved_at', '?')}")
    print()

    call_fields = [
        ("call_a_id", "Run A (chexagent generation)"),
        ("call_b_id", "Run B (chexagent generation)"),
        ("scoring_id", "Scoring (provenance gate demo)"),
        ("orchestrator_call_id", "Orchestrator (server-side chain)"),
        ("smoke_call_id", "Smoke test (5-image CheXagent)"),
    ]
    for key, label in call_fields:
        cid = data.get(key)
        if not cid:
            continue
        print(f"  {label}")
        print(f"    id        = {cid}")
        s = _status_of_call(cid)
        print(f"    status    = {s.get('status', '?')}")
        if s.get("error"):
            print(f"    error     = {s['error'][:300]}")
        if s.get("result") is not None:
            result = s["result"]
            if isinstance(result, dict):
                # Print a condensed one-line summary of key fields
                keys_of_interest = [
                    "total_claims", "total_reports", "generation_errors",
                    "model_used", "status", "stop_reason",
                    "downgrade_rate_diff", "n_pairs",
                ]
                summary = {k: result.get(k) for k in keys_of_interest if k in result}
                if summary:
                    print(f"    result    = {json.dumps(summary)}")
                else:
                    print(f"    result    = {str(result)[:200]}...")
            else:
                print(f"    result    = {str(result)[:200]}")
        print()


def _report_volume_workbooks() -> None:
    """List the same_model_experiment/ dir on the volume with sizes."""
    print(f"VOLUME CONTENTS ({SAME_MODEL_PATH}):")
    import subprocess
    try:
        out = subprocess.run(
            ["modal", "volume", "ls", VOLUME_NAME, SAME_MODEL_PATH],
            capture_output=True, text=True, timeout=30,
        )
    except Exception as e:  # noqa: BLE001
        print(f"  (modal volume ls failed: {e})")
        print()
        return
    if out.returncode != 0:
        print(f"  (volume path not found or empty — {out.stderr.strip()[:200]})")
        print()
        return
    lines = [l for l in out.stdout.splitlines() if l.strip()]
    if not lines:
        print("  (empty)")
    else:
        for line in lines[:20]:
            print(f"  {line}")
    print()


def _recommended_next_action(sidecars: list[str]) -> str:
    """Heuristic: what should the user do next based on sidecar state."""
    # 1. Smoke test state wins if present.  Check the most-recent
    # smoke sidecar first (sort by mtime descending).
    smoke_sidecars = sorted(
        [p for p in sidecars if "smoke" in p and os.path.exists(p)],
        key=lambda p: os.path.getmtime(p),
        reverse=True,
    )
    for p in smoke_sidecars:
        try:
            with open(p, "r") as f:
                data = json.load(f)
            cid = data.get("smoke_call_id")
            if not cid:
                continue
            s = _status_of_call(cid)
            if s.get("status") == "completed":
                result = s.get("result") or {}
                if isinstance(result, dict):
                    n_claims = result.get("total_claims", 0)
                    n_errors = result.get("generation_errors", 0)
                    n_reports = result.get("total_reports", 0)
                    if n_reports == 0:
                        return f"{p}: 0 reports generated — something very wrong"
                    error_rate = n_errors / n_reports
                    if error_rate > 0.5:
                        return (
                            f"{p}: {n_errors}/{n_reports} errors "
                            f"({int(error_rate*100)}%) — CheXagent prompt path still "
                            f"broken.\nModal logs: `modal app logs claimguard-real-hallucinations | tail -100`"
                        )
                    if n_claims > n_reports:  # more claims than images → good, reports have sentences
                        return (
                            f"{p}: {n_claims} claims from {n_reports} "
                            f"images, {n_errors} errors — prompt fix WORKED.\n"
                            f"Fire the full run:\n"
                            f"    python3 scripts/launch_task9_detached.py --fire-and-forget"
                        )
                    return (
                        f"{p}: {n_claims} claims from {n_reports} "
                        f"images, {n_errors} errors — mixed signal. "
                        f"Check Modal logs to inspect actual outputs:\n"
                        f"    modal app logs claimguard-real-hallucinations | tail -50"
                    )
            elif s.get("status") == "running":
                return f"{p}: smoke test still running — check back in a few min"
            elif s.get("status") == "failed":
                return f"{p}: smoke test FAILED — {s.get('error', '?')[:300]}"
        except Exception as e:  # noqa: BLE001
            return f"{p}: parse error: {e}"

    # 2. Fall back to generic advice
    return (
        "No smoke test state found. Run the smoke test first:\n"
        "    python3 -c \"import modal; "
        "orch=modal.Function.from_name('claimguard-real-hallucinations', "
        "'generate_annotation_workbook'); "
        "call=orch.spawn(openi_images_dir='/data/openi_images', "
        "openi_reports_csv='/data/openi_cxr_chexpert_schema.csv', "
        "output_dir='/data/same_model_experiment/smoke', max_images=5, "
        "seed=999, image_seed=42, sampling=True, temperature=0.7, "
        "top_p=0.9, run_id='smoke', "
        "workbook_filename='annotation_workbook_smoke.json'); "
        "print(call.object_id)\""
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Task 9 status checker — shows state of in-flight "
                    "generation/scoring runs and recommends next action.",
    )
    parser.add_argument(
        "--sidecar", action="append", default=None,
        help="Sidecar JSON path(s) to inspect. Defaults: all known paths in /tmp.",
    )
    parser.add_argument(
        "--skip-volume", action="store_true",
        help="Skip the volume-listing step (faster).",
    )
    args = parser.parse_args(argv)

    sidecars = args.sidecar or list(DEFAULT_SIDECARS)
    any_found = False
    for path in sidecars:
        if os.path.exists(path):
            _report_sidecar(path)
            any_found = True
    if not any_found:
        print("No sidecar files found at the default paths.\n")
        print("  Checked:", *sidecars, sep="\n    ")
        print()

    if not args.skip_volume:
        _report_volume_workbooks()

    print("RECOMMENDED NEXT ACTION:")
    print("  " + _recommended_next_action(sidecars).replace("\n", "\n  "))
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
