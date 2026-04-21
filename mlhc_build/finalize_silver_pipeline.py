"""Post-GREEN-backfill finalization for the silver-label pipeline.

Runs:
  1. silver_ensemble union (3 graders × 2 batches CheXagent+gated)
  2. padchest_gr_validate Cohen kappa
  3. prints a markdown table suitable for dropping into the paper supplement
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import modal


def spawn_and_wait(fn_name: str, **kwargs) -> dict:
    print(f"[finalize] spawning {fn_name} kwargs={list(kwargs.keys())}")
    f = modal.Function.from_name("claimguard-v6-orchestrator", fn_name)
    call = f.spawn(**kwargs)
    print(f"[finalize]   call id: {call.object_id}")
    t0 = time.time()
    while True:
        try:
            result = call.get(timeout=5)
            elapsed = time.time() - t0
            print(f"[finalize]   DONE in {elapsed:.1f}s")
            return result
        except TimeoutError:
            elapsed = time.time() - t0
            if elapsed > 60 * 30 * 10:  # 5 hour safety
                raise RuntimeError(f"{fn_name} exceeded 5h timeout")
            print(f"[finalize]   still running @ {elapsed:.0f}s...")
            continue


def main():
    # 1. silver_ensemble: union across CheXagent + gated batches
    green_paths = [
        "/data/v6_silver/green_labels.jsonl",
        "/data/v6_silver/green_labels_gated.jsonl",
        "/data/v6_silver/green_labels_gated_remainder.jsonl",
    ]
    radfact_paths = [
        "/data/v6_silver/radfact_labels.jsonl",
        "/data/v6_silver/radfact_labels_gated.jsonl",
    ]
    vert_paths = [
        "/data/v6_silver/vert_labels.jsonl",
        "/data/v6_silver/vert_labels_gated.jsonl",
    ]

    ens = spawn_and_wait(
        "silver_ensemble",
        green_paths=green_paths,
        radfact_paths=radfact_paths,
        vert_paths=vert_paths,
        out_jsonl="/data/v6_silver/ensemble.jsonl",
        out_stats="/data/v6_silver/ensemble_stats.json",
    )
    print("[finalize] ensemble stats:", json.dumps(ens, default=str, indent=2)[:1200])

    # 2. padchest_gr_validate
    pg = spawn_and_wait(
        "padchest_gr_validate",
        padchest_gr_records="/data/groundbench_v5/padchest_gr_records.jsonl",
        ensemble_jsonl="/data/v6_silver/ensemble.jsonl",
        out_jsonl="/data/v6_results/padchest_gr_validation.jsonl",
        out_stats="/data/v6_results/padchest_gr_validation_stats.json",
    )
    print("[finalize] padchest_gr validate stats:", json.dumps(pg, default=str, indent=2)[:1200])

    # 3. pull locally and emit markdown
    out_dir = Path("/tmp/cg_state")
    out_dir.mkdir(exist_ok=True)
    import subprocess
    subprocess.run(
        [
            "modal", "volume", "get",
            "claimguard-v5-data",
            "/v6_silver/ensemble_stats.json",
            str(out_dir / "ensemble_stats.json"),
            "--force",
        ],
        check=True,
    )
    subprocess.run(
        [
            "modal", "volume", "get",
            "claimguard-v5-data",
            "/v6_results/padchest_gr_validation_stats.json",
            str(out_dir / "padchest_gr_validation_stats.json"),
            "--force",
        ],
        check=True,
    )

    ens_stats = json.loads((out_dir / "ensemble_stats.json").read_text())
    pg_stats = json.loads((out_dir / "padchest_gr_validation_stats.json").read_text())

    md_lines = [
        "# Silver-label + PadChest-GR validation summary",
        "",
        f"- **ensemble size**: {ens_stats.get('n_total', '?')} claims",
        f"- **unanimous SUP**: {ens_stats.get('n_unanimous_supported', '?')}",
        f"- **unanimous CON**: {ens_stats.get('n_unanimous_contradicted', '?')}",
        f"- **unanimous UNC**: {ens_stats.get('n_unanimous_uncertain', '?')}",
        f"- **Krippendorff $\\alpha$**: {ens_stats.get('krippendorff_alpha', '?')}",
        "",
        "## PadChest-GR radiologist-bbox agreement",
        "",
        f"- **n matched**: {pg_stats.get('n_matched', '?')}",
        f"- **Cohen $\\kappa$**: {pg_stats.get('cohen_kappa', '?')}",
        f"- **precision**: {pg_stats.get('precision', '?')}",
        f"- **recall**: {pg_stats.get('recall', '?')}",
    ]
    out_md = Path(
        "/Users/aayanalwani/VeriFact/verifact/mlhc_build/supplement_silver_validation.md"
    )
    out_md.write_text("\n".join(md_lines) + "\n")
    print(f"[finalize] wrote {out_md}")


if __name__ == "__main__":
    main()
