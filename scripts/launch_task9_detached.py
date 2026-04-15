"""Task 9 detach-friendly launcher.

Fires the full Task 9 dual-run + scoring chain from the local Mac,
using deployed Modal Functions + spawn()/get() so the chain survives
the user closing their laptop mid-run.

Usage
-----

One-time setup (idempotent; re-running just overwrites the deployed
functions):

    modal deploy scripts/generate_real_hallucinations.py
    modal deploy scripts/demo_provenance_gate_failure.py

Then fire the chain:

    nohup caffeinate -i python3 scripts/launch_task9_detached.py \\
        > /tmp/task9_launcher.log 2>&1 &
    disown

``caffeinate -i`` keeps the Mac from sleeping while the launcher is
running — needed because the launcher does ``.get()`` waits that
must stay connected to Modal.  The Mac needs power (closing the lid
with battery WILL kill it), but closing the lid with AC connected is
fine.  ``nohup`` + ``disown`` detach from the shell so closing the
terminal doesn't kill the launcher.

If you're going to fully shut down the Mac, use the alternative
"fire-and-forget" mode below, which dispatches all three jobs
immediately without waiting:

    python3 scripts/launch_task9_detached.py --fire-and-forget

In fire-and-forget mode the launcher:

    1. Spawns run-A generation (returns a FunctionCall handle)
    2. Spawns run-B generation in parallel
    3. Writes both handles to a JSON sidecar
    4. Prints the handles and exits
    5. Does NOT launch the scoring phase — the user must do that
       manually after both runs complete, via:

            python3 scripts/demo_provenance_gate_failure.py --skip-generation

Why this design
---------------
* ``modal run --detach`` only works for a single function call, not a
  chain.  We have three dependent stages (run A, run B, scoring) so
  we need a chainable primitive.
* ``modal deploy`` + ``Function.from_name().spawn()`` is the chainable
  primitive: ``.spawn()`` returns a ``FunctionCall`` handle whose
  ``.get()`` can be called later (even from a different Python
  process) via ``FunctionCall.from_id(handle_id).get()``.
* The sidecar JSON preserves the handles so a follow-up process can
  resume even if the launcher dies mid-chain.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.demo_provenance_gate_failure import (  # noqa: E402
    GateDemoConfig,
    RUN_A_GENERATOR_ID,
    RUN_A_ID,
    RUN_B_GENERATOR_ID,
    RUN_B_ID,
)

logger = logging.getLogger("task9_launcher")


GEN_APP_NAME = "claimguard-real-hallucinations"
GEN_FUNC_NAME = "generate_annotation_workbook"

DEMO_APP_NAME = "claimguard-provenance-gate-demo"
DEMO_FUNC_NAME = "demo_provenance_gate_remote"

DEFAULT_HANDLES_JSON = "/tmp/task9_function_calls.json"


def _build_gen_kwargs(
    config: GateDemoConfig,
    run_id: str,
    seed: int,
    workbook_path: str,
) -> dict:
    """Build the kwargs for one call to ``generate_annotation_workbook``.

    Mirrors the argument layout that ``_launch_chexagent_runs`` in
    ``demo_provenance_gate_failure.py`` uses, so a downstream scoring
    pass with ``skip_generation=True`` finds the workbooks at the
    expected paths.
    """
    return {
        "openi_images_dir": config.openi_images_dir,
        "openi_reports_csv": config.openi_reports_csv,
        "output_dir": os.path.dirname(workbook_path),
        "max_images": config.max_images,
        "seed": seed,
        "image_seed": config.image_seed,
        "sampling": True,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "run_id": run_id,
        "workbook_filename": os.path.basename(workbook_path),
    }


def _lookup_functions():
    """Look up the deployed Modal Functions.

    Raises ``RuntimeError`` with a useful message if the apps haven't
    been deployed yet — the user needs to run ``modal deploy`` first.
    """
    try:
        import modal
    except ImportError as e:
        raise RuntimeError(
            "modal Python SDK is required for the detached launcher. "
            "Install with `pip install modal` and re-run."
        ) from e

    try:
        gen_func = modal.Function.from_name(GEN_APP_NAME, GEN_FUNC_NAME)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            f"Could not look up {GEN_APP_NAME}::{GEN_FUNC_NAME}. "
            f"Deploy it first with:\n"
            f"    modal deploy scripts/generate_real_hallucinations.py\n"
            f"Original error: {e}"
        ) from e

    try:
        demo_func = modal.Function.from_name(DEMO_APP_NAME, DEMO_FUNC_NAME)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            f"Could not look up {DEMO_APP_NAME}::{DEMO_FUNC_NAME}. "
            f"Deploy it first with:\n"
            f"    modal deploy scripts/demo_provenance_gate_failure.py\n"
            f"Original error: {e}"
        ) from e

    return gen_func, demo_func


def _save_handles(
    path: str,
    *,
    call_a_id: str | None,
    call_b_id: str | None,
    scoring_id: str | None,
    config: GateDemoConfig,
    mode: str,
) -> None:
    """Write handles + config to a sidecar JSON so a follow-up process
    can resume the chain after a crash."""
    parent = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(parent, exist_ok=True)
    payload = {
        "mode": mode,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gen_app": GEN_APP_NAME,
        "gen_func": GEN_FUNC_NAME,
        "demo_app": DEMO_APP_NAME,
        "demo_func": DEMO_FUNC_NAME,
        "call_a_id": call_a_id,
        "call_b_id": call_b_id,
        "scoring_id": scoring_id,
        "config": json.loads(config.to_json()),
    }
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)
    logger.info("Saved function call handles to %s", path)


def run_chain_blocking(
    config: GateDemoConfig,
    *,
    handles_path: str = DEFAULT_HANDLES_JSON,
) -> dict:
    """Fire the full Task 9 chain, blocking until every stage finishes.

    Steps:
        1. Look up the deployed Modal functions.
        2. Spawn ``generate_annotation_workbook`` for run A.
        3. Spawn ``generate_annotation_workbook`` for run B in parallel.
        4. Save both handles to the sidecar JSON immediately so a
           later resume is possible if the launcher crashes here.
        5. ``.get()`` run A (blocks ~30–45 min).
        6. ``.get()`` run B (blocks ~30–45 min, often in parallel with A).
        7. Spawn ``demo_provenance_gate_remote`` with
           ``skip_generation=True`` (since A and B both wrote their
           workbooks to the volume in steps 5 and 6).
        8. Update the sidecar with the scoring call id.
        9. ``.get()`` the scoring call (~5 min).
        10. Print the final stats dict.

    Uses ``Function.from_name().spawn()`` so each stage is a
    fire-and-forget call on Modal's side; ``.get()`` on the local
    client is a thin wait that doesn't consume GPU.  If the launcher
    dies between stages, the saved handles in ``handles_path`` let a
    recovery process pick up from where it left off via
    ``FunctionCall.from_id(handle_id).get()``.
    """
    gen_func, demo_func = _lookup_functions()

    # --- Stage 1: spawn both generation runs in parallel ---
    logger.info(
        "Spawning run A generation (seed=%d, image_seed=%d, max_images=%d)",
        config.seed_run_a, config.image_seed, config.max_images,
    )
    call_a = gen_func.spawn(
        **_build_gen_kwargs(
            config,
            run_id=RUN_A_ID,
            seed=config.seed_run_a,
            workbook_path=config.workbook_a_path,
        )
    )
    logger.info("Run A spawned: %s", getattr(call_a, "object_id", "?"))

    logger.info(
        "Spawning run B generation (seed=%d, image_seed=%d, max_images=%d)",
        config.seed_run_b, config.image_seed, config.max_images,
    )
    call_b = gen_func.spawn(
        **_build_gen_kwargs(
            config,
            run_id=RUN_B_ID,
            seed=config.seed_run_b,
            workbook_path=config.workbook_b_path,
        )
    )
    logger.info("Run B spawned: %s", getattr(call_b, "object_id", "?"))

    _save_handles(
        handles_path,
        call_a_id=getattr(call_a, "object_id", None),
        call_b_id=getattr(call_b, "object_id", None),
        scoring_id=None,
        config=config,
        mode="blocking_generation_in_flight",
    )

    # --- Stage 2: wait for both generations ---
    logger.info("Waiting for run A to complete (this blocks ~30-45 min)...")
    result_a = call_a.get()
    logger.info("Run A complete: %s", result_a)

    logger.info("Waiting for run B to complete...")
    result_b = call_b.get()
    logger.info("Run B complete: %s", result_b)

    # --- Stage 3: scoring with skip_generation ---
    scoring_config = GateDemoConfig(
        verifier_checkpoint=config.verifier_checkpoint,
        openi_images_dir=config.openi_images_dir,
        openi_reports_csv=config.openi_reports_csv,
        workbook_a_path=config.workbook_a_path,
        workbook_b_path=config.workbook_b_path,
        output_dir=config.output_dir,
        max_images=config.max_images,
        image_seed=config.image_seed,
        seed_run_a=config.seed_run_a,
        seed_run_b=config.seed_run_b,
        temperature=config.temperature,
        top_p=config.top_p,
        high_score_threshold=config.high_score_threshold,
        hf_backbone=config.hf_backbone,
        max_length=config.max_length,
        batch_size=config.batch_size,
        skip_generation=True,  # critical: don't re-generate
    )
    logger.info(
        "Spawning scoring phase (skip_generation=True, workbook_a=%s, workbook_b=%s)",
        config.workbook_a_path, config.workbook_b_path,
    )
    call_scoring = demo_func.spawn(scoring_config.to_json())
    logger.info("Scoring spawned: %s", getattr(call_scoring, "object_id", "?"))

    _save_handles(
        handles_path,
        call_a_id=getattr(call_a, "object_id", None),
        call_b_id=getattr(call_b, "object_id", None),
        scoring_id=getattr(call_scoring, "object_id", None),
        config=config,
        mode="scoring_in_flight",
    )

    # --- Stage 4: wait for scoring ---
    logger.info("Waiting for scoring phase to complete (~5-10 min)...")
    scoring_result = call_scoring.get()
    logger.info("Scoring complete")

    _save_handles(
        handles_path,
        call_a_id=getattr(call_a, "object_id", None),
        call_b_id=getattr(call_b, "object_id", None),
        scoring_id=getattr(call_scoring, "object_id", None),
        config=config,
        mode="complete",
    )

    return {
        "run_a": result_a,
        "run_b": result_b,
        "scoring": scoring_result,
        "handles_path": handles_path,
    }


def fire_and_forget(
    config: GateDemoConfig,
    *,
    handles_path: str = DEFAULT_HANDLES_JSON,
) -> dict:
    """Spawn both generation runs, save handles, and exit immediately.

    Use this mode when the launcher cannot stay alive to chain the
    scoring phase (e.g. fully shutting down the Mac).  After both
    generations complete on Modal's servers (visible via
    ``modal app logs`` or checking the volume), the user must
    manually fire the scoring phase:

        python3 scripts/demo_provenance_gate_failure.py --skip-generation

    Returns the handle dict for quick inspection.
    """
    gen_func, _ = _lookup_functions()

    logger.info(
        "[fire-and-forget] Spawning run A generation (seed=%d, image_seed=%d)",
        config.seed_run_a, config.image_seed,
    )
    call_a = gen_func.spawn(
        **_build_gen_kwargs(
            config,
            run_id=RUN_A_ID,
            seed=config.seed_run_a,
            workbook_path=config.workbook_a_path,
        )
    )
    logger.info("Run A spawned: %s", getattr(call_a, "object_id", "?"))

    logger.info(
        "[fire-and-forget] Spawning run B generation (seed=%d, image_seed=%d)",
        config.seed_run_b, config.image_seed,
    )
    call_b = gen_func.spawn(
        **_build_gen_kwargs(
            config,
            run_id=RUN_B_ID,
            seed=config.seed_run_b,
            workbook_path=config.workbook_b_path,
        )
    )
    logger.info("Run B spawned: %s", getattr(call_b, "object_id", "?"))

    _save_handles(
        handles_path,
        call_a_id=getattr(call_a, "object_id", None),
        call_b_id=getattr(call_b, "object_id", None),
        scoring_id=None,
        config=config,
        mode="fire_and_forget",
    )

    info = {
        "call_a_id": getattr(call_a, "object_id", None),
        "call_b_id": getattr(call_b, "object_id", None),
        "handles_path": handles_path,
        "next_step": (
            "After runs complete, run: python3 "
            "scripts/demo_provenance_gate_failure.py --skip-generation"
        ),
    }
    logger.info("fire-and-forget complete: %s", info)
    return info


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    parser = argparse.ArgumentParser(
        description=(
            "Task 9 detach-friendly launcher.  Deploys + spawns the "
            "dual-run gate demo chain so it survives the local Mac "
            "closing."
        ),
    )
    parser.add_argument(
        "--fire-and-forget",
        action="store_true",
        help=(
            "Spawn both generation runs and exit immediately. "
            "After they complete, manually run "
            "scripts/demo_provenance_gate_failure.py --skip-generation."
        ),
    )
    parser.add_argument(
        "--handles-path",
        default=DEFAULT_HANDLES_JSON,
        help="Sidecar JSON for saved FunctionCall handles.",
    )
    # All GateDemoConfig fields are overridable from the CLI.
    defaults = GateDemoConfig()
    parser.add_argument(
        "--verifier-checkpoint",
        default=defaults.verifier_checkpoint,
    )
    parser.add_argument(
        "--openi-images-dir", default=defaults.openi_images_dir,
    )
    parser.add_argument(
        "--openi-reports-csv", default=defaults.openi_reports_csv,
    )
    parser.add_argument(
        "--workbook-a-path", default=defaults.workbook_a_path,
    )
    parser.add_argument(
        "--workbook-b-path", default=defaults.workbook_b_path,
    )
    parser.add_argument("--output-dir", default=defaults.output_dir)
    parser.add_argument(
        "--max-images", type=int, default=defaults.max_images,
    )
    parser.add_argument(
        "--image-seed", type=int, default=defaults.image_seed,
    )
    parser.add_argument(
        "--seed-run-a", type=int, default=defaults.seed_run_a,
    )
    parser.add_argument(
        "--seed-run-b", type=int, default=defaults.seed_run_b,
    )
    parser.add_argument(
        "--temperature", type=float, default=defaults.temperature,
    )
    parser.add_argument("--top-p", type=float, default=defaults.top_p)
    args = parser.parse_args(argv)

    config = GateDemoConfig(
        verifier_checkpoint=args.verifier_checkpoint,
        openi_images_dir=args.openi_images_dir,
        openi_reports_csv=args.openi_reports_csv,
        workbook_a_path=args.workbook_a_path,
        workbook_b_path=args.workbook_b_path,
        output_dir=args.output_dir,
        max_images=args.max_images,
        image_seed=args.image_seed,
        seed_run_a=args.seed_run_a,
        seed_run_b=args.seed_run_b,
        temperature=args.temperature,
        top_p=args.top_p,
        skip_generation=False,  # unused in fire-and-forget path
    )
    logger.info("Task 9 launcher config: %s", config.to_json())

    try:
        if args.fire_and_forget:
            result = fire_and_forget(
                config, handles_path=args.handles_path,
            )
        else:
            result = run_chain_blocking(
                config, handles_path=args.handles_path,
            )
    except RuntimeError as e:
        logger.error("Task 9 launcher failed: %s", e)
        return 1

    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "DEFAULT_HANDLES_JSON",
    "DEMO_APP_NAME",
    "DEMO_FUNC_NAME",
    "GEN_APP_NAME",
    "GEN_FUNC_NAME",
    "fire_and_forget",
    "main",
    "run_chain_blocking",
]
