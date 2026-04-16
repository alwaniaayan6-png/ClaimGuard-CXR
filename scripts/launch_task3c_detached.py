"""Task 3c — durable detach launcher for the R-Drop refinement trainer.

Wraps the spawn-based pattern that the trainer's existing local
entrypoint (``scripts/modal_train_dpo_refinement.py:main``) does NOT
support — that entrypoint uses ``with app.run(): fn.remote(...)``
which blocks the local Python process for the entire ~30 minute
training run.  If the local dies (network blip, Mac sleep, terminal
close) the remote dies with it.

This launcher uses ``modal.Function.from_name(...).spawn(...)`` to
hand the call off to Modal's backend, then exits cleanly.  The
training run continues on Modal's H100 cluster regardless of what
happens to the local Mac.

Prereq: the trainer must be deployed first via::

    modal deploy scripts/modal_train_dpo_refinement.py

Then fire the launcher::

    python3 scripts/launch_task3c_detached.py \\
        --preference-data /data/counterfactual_preference_pairs_v3.json \\
        --output-dir /data/checkpoints/verifier_binary_v4_rdrop \\
        --base-checkpoint /data/checkpoints/verifier_binary_v3/best_verifier.pt

The launcher writes the spawned ``FunctionCall.object_id`` to a
sidecar JSON at ``/tmp/task3c_function_call.json`` so a follow-up
process can resume the wait via::

    python3 -c "
    import modal
    fc = modal.FunctionCall.from_id('fc-...')
    print(fc.get())  # blocks until done
    "

The launcher does NOT wait for the call to complete — it exits as
soon as the spawn is acknowledged by Modal.  Polling for completion
is the caller's job.

Why this design instead of ``modal run --detach``: the trainer
script doesn't have an ``@app.local_entrypoint()`` decorator (it has
a regular ``main()`` that uses ``with app.run():`` instead), so
``modal run --detach scripts/modal_train_dpo_refinement.py`` doesn't
work — Modal can't find an entrypoint to detach.  Adding the
decorator would require restructuring the trainer's argparse flow,
which is more risk than this small launcher script.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

# Make the in-repo trainer config importable.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.modal_train_dpo_refinement import DPOTrainingConfig

logger = logging.getLogger("task3c_launcher")

APP_NAME = "claimguard-dpo-refinement"
FUNC_NAME = "train_dpo_refinement_remote"
DEFAULT_HANDLES_JSON = "/tmp/task3c_function_call.json"


def _lookup_function():
    """Resolve the deployed trainer function by name.

    Raises with a clear hint if the user forgot to ``modal deploy``.
    """
    try:
        import modal
    except ImportError as e:
        raise RuntimeError(
            "modal SDK required.  pip install modal and re-run."
        ) from e

    try:
        return modal.Function.from_name(APP_NAME, FUNC_NAME)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            f"Could not look up {APP_NAME}::{FUNC_NAME}.  Deploy it "
            f"first with:\n"
            f"    modal deploy scripts/modal_train_dpo_refinement.py\n"
            f"Original error: {e}"
        ) from e


def _save_handle(path: str, call_id: str, config: DPOTrainingConfig) -> None:
    """Write the spawned FunctionCall ID + config to a sidecar JSON."""
    parent = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(parent, exist_ok=True)
    payload = {
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "app": APP_NAME,
        "func": FUNC_NAME,
        "call_id": call_id,
        "config": {
            "base_checkpoint": config.base_checkpoint,
            "output_dir": config.output_dir,
            "preference_data": config.preference_data,
            "loss_mode": config.loss_mode,
            "num_epochs": config.num_epochs,
            "batch_size": config.batch_size,
            "lr": config.lr,
            "freeze_first_n_layers": config.freeze_first_n_layers,
            "ce_weight": config.ce_weight,
            "consistency_weight": config.consistency_weight,
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Task 3c R-Drop refinement detach launcher."
    )
    defaults = DPOTrainingConfig()
    parser.add_argument(
        "--preference-data",
        default=defaults.preference_data,
        help="Modal-volume path to the counterfactual preference "
             "pairs JSON (output of generate_counterfactual_pairs.py).",
    )
    parser.add_argument(
        "--base-checkpoint",
        default=defaults.base_checkpoint,
        help="Modal-volume path to the v3 base checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        default="/data/checkpoints/verifier_binary_v4_rdrop",
        help="Modal-volume directory where v4 best_verifier.pt and "
             "training_history.json will land.",
    )
    parser.add_argument(
        "--loss-mode",
        choices=["consistency_mixed", "consistency", "dpo"],
        default="consistency_mixed",
        help="'consistency_mixed' (default, post-2026-04-15 fix): "
             "R-Drop with mixed cf+faithful data and per-example "
             "labels. 'consistency' is the broken single-class path "
             "that collapsed v4 v1. 'dpo' is the legacy DPO path with "
             "the chosen/rejected inversion bug. Use consistency_mixed.",
    )
    parser.add_argument(
        "--num-epochs", type=int, default=defaults.num_epochs,
    )
    parser.add_argument(
        "--batch-size", type=int, default=defaults.batch_size,
    )
    parser.add_argument(
        "--lr", type=float, default=defaults.lr,
    )
    parser.add_argument(
        "--freeze-first-n-layers", type=int,
        default=defaults.freeze_first_n_layers,
    )
    parser.add_argument(
        "--ce-weight", type=float, default=defaults.ce_weight,
    )
    parser.add_argument(
        "--consistency-weight", type=float,
        default=defaults.consistency_weight,
    )
    parser.add_argument(
        "--ce-blowup-threshold", type=float,
        default=defaults.ce_blowup_threshold,
    )
    parser.add_argument(
        "--seed", type=int, default=defaults.seed,
    )
    parser.add_argument(
        "--handles-json",
        default=DEFAULT_HANDLES_JSON,
        help="Where to save the FunctionCall handle for resume.",
    )
    args = parser.parse_args()

    config = DPOTrainingConfig(
        base_checkpoint=args.base_checkpoint,
        output_dir=args.output_dir,
        preference_data=args.preference_data,
        loss_mode=args.loss_mode,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        freeze_first_n_layers=args.freeze_first_n_layers,
        ce_weight=args.ce_weight,
        consistency_weight=args.consistency_weight,
        ce_blowup_threshold=args.ce_blowup_threshold,
        seed=args.seed,
    )
    logger.info("DPO config: %s", config)

    fn = _lookup_function()
    logger.info("Spawning %s::%s ...", APP_NAME, FUNC_NAME)
    handle = fn.spawn(config.to_json())
    call_id = handle.object_id
    logger.info("Spawned call_id=%s", call_id)

    _save_handle(args.handles_json, call_id, config)
    logger.info("Saved handle to %s", args.handles_json)

    print()
    print("=" * 60)
    print(" Task 3c R-Drop refinement spawned (detached)")
    print("=" * 60)
    print(f" call_id:     {call_id}")
    print(f" handle file: {args.handles_json}")
    print()
    print(" To poll for completion:")
    print(f"   python3 -c \"import modal; "
          f"print(modal.FunctionCall.from_id('{call_id}').get())\"")
    print()
    print(" To check progress without blocking:")
    print(f"   modal app logs <app_id>  # see modal app list")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
