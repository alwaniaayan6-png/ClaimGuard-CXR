"""Daily budget aggregator for ClaimGuard-CXR v5 Modal jobs.

Pulls spend data from the Modal API and checks against per-phase caps.
Auto-halts (prints a hard STOP line) if any phase exceeds 110% of its cap.

Usage:
    python v5/scripts/budget_daily.py [--phase 3]
    python v5/scripts/budget_daily.py --all

Requires:
    pip install modal
    MODAL_TOKEN_ID and MODAL_TOKEN_SECRET in env (or `modal token set`)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

# Per-phase caps in USD (from RUNBOOK §Budget tracking)
PHASE_CAPS = {
    3: 300,   # VLM generation
    4: 500,   # GroundBench + LLM labeling
    6: 800,   # Training (5 seeds × 5 configs)
    7: 200,   # Evaluation
    8: 100,   # Release
}
TOTAL_CAP = 1_900
HALT_FACTOR = 1.10  # auto-halt if phase > 110% of cap

LOG_FILE = Path(__file__).parent.parent.parent / "budget_log.jsonl"


def _get_modal_spend(since_hours: int = 24) -> list[dict]:
    """Fetch Modal app usage. Returns list of {app, cost_usd, calls} dicts."""
    try:
        import modal.runner  # type: ignore

        client = modal.runner.get_client()  # type: ignore
        cutoff = datetime.now(timezone.utc) - timedelta(hours=since_hours)
        # Modal SDK v0.62+ has client.usage.list() — adjust if API changes
        records = []
        try:
            for item in client.usage.list(since=cutoff):  # type: ignore
                records.append(
                    {
                        "app": getattr(item, "app_name", "unknown"),
                        "cost_usd": float(getattr(item, "cost", 0.0)),
                        "calls": int(getattr(item, "call_count", 0)),
                        "ts": str(getattr(item, "created_at", cutoff)),
                    }
                )
        except AttributeError:
            pass  # SDK version without usage API — fall back to manual
        return records
    except Exception as e:
        print(f"[budget] Modal API unavailable: {e}. Using log file only.", file=sys.stderr)
        return []


def _load_log() -> list[dict]:
    if not LOG_FILE.exists():
        return []
    with open(LOG_FILE) as f:
        return [json.loads(line) for line in f if line.strip()]


def _append_log(records: list[dict]) -> None:
    with open(LOG_FILE, "a") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _phase_tag(app_name: str) -> Optional[int]:
    """Infer phase from Modal app name convention."""
    mapping = {
        "vlm": 3,
        "groundbench": 4,
        "train": 6,
        "eval": 7,
        "release": 8,
    }
    name = app_name.lower()
    for kw, phase in mapping.items():
        if kw in name:
            return phase
    return None


def report(phase: Optional[int] = None) -> dict:
    """Aggregate spend by phase and check against caps."""
    live = _get_modal_spend(since_hours=7 * 24)  # last 7 days
    _append_log(live)
    all_records = _load_log()

    phase_spend: dict[int, float] = {p: 0.0 for p in PHASE_CAPS}
    uncategorized = 0.0

    for r in all_records:
        p = _phase_tag(r.get("app", ""))
        cost = float(r.get("cost_usd", 0.0))
        if p is not None:
            phase_spend[p] = phase_spend.get(p, 0.0) + cost
        else:
            uncategorized += cost

    total = sum(phase_spend.values()) + uncategorized
    halt = False
    lines = [
        f"\n{'='*54}",
        f"  ClaimGuard-CXR v5 — Budget Report  {datetime.now():%Y-%m-%d %H:%M}",
        f"{'='*54}",
    ]

    for p, cap in sorted(PHASE_CAPS.items()):
        if phase is not None and p != phase:
            continue
        spent = phase_spend.get(p, 0.0)
        pct = 100 * spent / cap
        flag = ""
        if spent > cap * HALT_FACTOR:
            flag = "  <<<  HALT — EXCEEDS 110% CAP"
            halt = True
        elif spent > cap * 0.8:
            flag = "  (!)"
        lines.append(f"  Phase {p:2d}: ${spent:7.2f} / ${cap:5d}  ({pct:5.1f}%){flag}")

    lines += [
        f"{'─'*54}",
        f"  Total   : ${total:7.2f} / ${TOTAL_CAP:5d}  ({100*total/TOTAL_CAP:5.1f}%)",
        f"  Uncat.  : ${uncategorized:7.2f}",
        f"{'='*54}",
    ]

    if halt:
        lines.append("\n  *** STOP — Phase cap exceeded. Do not launch more jobs.")
        lines.append("  *** Escalate to user for re-authorization.\n")

    print("\n".join(lines))
    return {
        "phase_spend": phase_spend,
        "total": total,
        "halt": halt,
        "ts": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily Modal budget aggregator")
    parser.add_argument("--phase", type=int, default=None, help="Show only this phase")
    parser.add_argument("--all", action="store_true", help="Show all phases (default)")
    args = parser.parse_args()
    result = report(phase=args.phase)
    if result["halt"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
