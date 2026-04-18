"""Modal entrypoint: adversarial hypothesis-only (HO) filter.

Trains a text-only RoBERTa-large classifier on (claim, evidence) -> label and
writes per-row training weights to /data/groundbench_v5/ho_filter_weights.jsonl.
Subsequent v5 training runs read those weights directly, so the HO filter is
computed ONCE and reused across the v5.2/v5.3/v5.4 configs.

Usage:
    cd /Users/aayanalwani/VeriFact/verifact &&
    modal run v5/modal/ho_filter.py::run_entrypoint
"""

from __future__ import annotations

from pathlib import Path

import modal

VERIFACT_ROOT = Path(__file__).resolve().parent.parent.parent if Path(__file__).resolve().parent.name == "modal" else Path("/root/verifact")

app = modal.App("claimguard-v5-ho-filter")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "torch==2.4.0",
        "transformers==4.44.2",
        "numpy==1.26.4",
        "pyarrow==17.0.0",
        "pyyaml==6.0.2",
    )
    .add_local_dir(str(VERIFACT_ROOT), remote_path="/root/verifact", copy=True)
)

volume = modal.Volume.from_name("claimguard-v5-data")


@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 60 * 2,
    volumes={"/data": volume},
)
def run_ho_filter_on_modal(
    train_jsonl: str = "/data/groundbench_v5/all/groundbench_v5_train.jsonl",
    output_weights_path: str = "/data/groundbench_v5/ho_filter_weights.jsonl",
    confidence_threshold: float = 0.7,
    downweight: float = 0.2,
    n_epochs: int = 1,
    seed: int = 17,
) -> dict:
    import sys

    sys.path.insert(0, "/root/verifact")

    from pathlib import Path as P
    import torch

    from v5.ho_filter import run_ho_filter
    from v5.model import build_v5_tokenizer, V5Config

    tokenizer = build_v5_tokenizer(V5Config())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    summary = run_ho_filter(
        train_jsonl=P(train_jsonl),
        output_weights_path=P(output_weights_path),
        tokenizer=tokenizer,
        device=device,
        confidence_threshold=confidence_threshold,
        downweight=downweight,
        n_epochs=n_epochs,
        seed=seed,
    )
    print(summary)
    return summary


@app.local_entrypoint()
def run_entrypoint(
    train_jsonl: str = "/data/groundbench_v5/all/groundbench_v5_train.jsonl",
    confidence_threshold: float = 0.7,
    downweight: float = 0.2,
    n_epochs: int = 1,
) -> None:
    print(
        run_ho_filter_on_modal.remote(
            train_jsonl=train_jsonl,
            confidence_threshold=confidence_threshold,
            downweight=downweight,
            n_epochs=n_epochs,
        )
    )
