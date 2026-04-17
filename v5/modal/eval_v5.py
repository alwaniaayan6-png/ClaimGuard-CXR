"""Modal entrypoint: evaluate a v5 checkpoint on a specified site.

Usage:
    modal run --detach v5/modal/eval_v5.py::eval_v5_entrypoint \\
        --ckpt v5_3_contrast --site chexpert_plus
"""

from __future__ import annotations

from pathlib import Path

import modal

VERIFACT_ROOT = Path(__file__).resolve().parents[2]

app = modal.App("claimguard-v5-eval")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        "transformers==4.44.2",
        "open-clip-torch==2.26.1",
        "hydra-core==1.3.2",
        "omegaconf==2.3.0",
        "scikit-learn==1.5.1",
        "numpy==1.26.4",
        "pandas==2.2.2",
        "pyarrow==17.0.0",
        "pillow==10.4.0",
        "scipy==1.14.1",
        "pyyaml==6.0.2",
    )
    .add_local_dir(str(VERIFACT_ROOT), remote_path="/root/verifact", copy=True)
)

volume = modal.Volume.from_name("claimguard-v5-data")


@app.function(image=image, gpu="H100", timeout=60 * 60 * 4, volumes={"/data": volume})
def eval_v5_run(ckpt_name: str, site: str) -> dict:
    import json
    import sys
    from pathlib import Path as P

    sys.path.insert(0, "/root/verifact")

    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    with initialize_config_dir(config_dir="/root/verifact/v5/configs", version_base=None):
        base = compose(config_name="base")
        sites = compose(config_name="eval_sites")
        cfg = OmegaConf.merge(base, sites)
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    site_cfg = cfg_dict["sites"][site]
    from v5.eval.image_grounded import EvalConfig, evaluate

    ckpt_path = P(cfg_dict["paths"]["out_root"]) / ckpt_name / "best.pt"
    out_path = P(cfg_dict["paths"]["out_root"]) / ckpt_name / f"eval_{site}.json"

    eval_cfg = EvalConfig(
        ckpt_path=ckpt_path,
        test_jsonl=P(site_cfg["test"]),
        cal_jsonl=P(site_cfg["cal"]),
        image_root=P(site_cfg["image_root"]),
        out_json=out_path,
        site_name=site,
        alphas=tuple(cfg_dict["conformal"]["alphas"]),
        conformal_variants=tuple(cfg_dict["conformal"]["variants"]),
        feature_columns=tuple(cfg_dict["conformal"]["feature_columns"]),
        bootstrap_replicates=int(cfg_dict["conformal"]["bootstrap_replicates"]),
        apply_provenance_gate=bool(cfg_dict["conformal"]["apply_provenance_gate"]),
    )
    result = evaluate(eval_cfg)
    return {"site": site, "out_json": str(out_path), "summary": result.metrics}


@app.local_entrypoint()
def eval_v5_entrypoint(ckpt: str = "v5_3_contrast", site: str = "chexpert_plus") -> None:
    print(eval_v5_run.remote(ckpt_name=ckpt, site=site))
