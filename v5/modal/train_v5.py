"""Modal entrypoint: train v5 (any config).

Usage:
    cd /Users/aayanalwani/VeriFact/verifact &&
    modal run --detach v5/modal/train_v5.py::train_v5_entrypoint --config v5_2_real

Arguments are passed via --config <name>; configs live in v5/configs/.
"""

from __future__ import annotations

from pathlib import Path

import modal

VERIFACT_ROOT = Path(__file__).resolve().parents[2]

app = modal.App("claimguard-v5-train")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        "transformers==4.44.2",
        "open-clip-torch==2.26.1",
        "sentencepiece==0.2.0",
        "accelerate==0.33.0",
        "datasets==2.20.0",
        "peft==0.12.0",
        "wandb==0.17.7",
        "hydra-core==1.3.2",
        "omegaconf==2.3.0",
        "scikit-learn==1.5.1",
        "numpy==1.26.4",
        "pandas==2.2.2",
        "pyarrow==17.0.0",
        "pillow==10.4.0",
        "presidio-analyzer==2.2.356",
        "presidio-anonymizer==2.2.356",
        "anthropic==0.34.2",
        "openai==1.45.0",
        "scipy==1.14.1",
        "pyyaml==6.0.2",
    )
    .add_local_dir(str(VERIFACT_ROOT), remote_path="/root/verifact", copy=True)
)

volume = modal.Volume.from_name("claimguard-v5-data")

SECRETS: list[modal.Secret] = []
try:
    SECRETS.append(modal.Secret.from_name("wandb"))
except Exception:
    pass


@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 60 * 12,
    volumes={"/data": volume},
    secrets=SECRETS,
)
def train_v5_run(config_name: str) -> dict:
    import json
    import os
    import sys
    from pathlib import Path as P

    sys.path.insert(0, "/root/verifact")

    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    # Load config
    with initialize_config_dir(
        config_dir=str(P("/root/verifact/v5/configs").resolve()), version_base=None
    ):
        cfg = compose(config_name=config_name)
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Build V5TrainConfig + V5Config
    from v5.train import V5TrainConfig, train_v5
    from v5.model import V5Config
    from v5.losses import LossWeights

    lw = LossWeights(**cfg_dict["loss"])
    tcfg = V5TrainConfig(
        train_jsonl=P(cfg_dict["paths"]["groundbench_root"]) / "all" / "groundbench_v5_train.jsonl",
        val_jsonl=P(cfg_dict["paths"]["groundbench_root"]) / "all" / "groundbench_v5_cal.jsonl",
        out_dir=P(cfg_dict["paths"]["out_root"]) / config_name,
        image_root=P(cfg_dict["paths"]["data_root"]),
        batch_size=cfg_dict["train"]["batch_size"],
        grad_accum=cfg_dict["train"]["grad_accum"],
        epochs=cfg_dict["train"]["epochs"],
        lr_encoders=cfg_dict["train"]["lr_encoders"],
        lr_heads=cfg_dict["train"]["lr_heads"],
        weight_decay=cfg_dict["train"]["weight_decay"],
        warmup_steps=cfg_dict["train"]["warmup_steps"],
        mixed_precision=cfg_dict["train"]["mixed_precision"],
        image_masked_prob=cfg_dict["train"]["image_masked_prob"],
        contrast_prob=cfg_dict["train"]["contrast_prob"],
        adversarial_ho_filter=cfg_dict["train"]["adversarial_ho_filter"],
        seed=cfg_dict["train"]["seed"],
        loss_weights=lw,
        use_wandb=cfg_dict["train"]["use_wandb"],
        wandb_project=cfg_dict["train"]["wandb_project"],
        wandb_run_name=cfg_dict["train"].get("wandb_run_name"),
        freeze_image_layers=cfg_dict["model"]["freeze_image_layers"],
        freeze_text_layers=cfg_dict["model"]["freeze_text_layers"],
        grounding_enabled=cfg_dict["model"]["grounding_enabled"],
    )
    model_cfg = V5Config(
        image_backbone=cfg_dict["model"]["image_backbone"],
        image_backbone_revision=cfg_dict["model"].get("image_backbone_revision"),
        text_backbone=cfg_dict["model"]["text_backbone"],
        text_backbone_revision=cfg_dict["model"].get("text_backbone_revision"),
        shared_dim=cfg_dict["model"]["shared_dim"],
        fusion_layers=cfg_dict["model"]["fusion_layers"],
        fusion_heads=cfg_dict["model"]["fusion_heads"],
        fusion_ffn_dim=cfg_dict["model"]["fusion_ffn_dim"],
        fusion_dropout=cfg_dict["model"]["fusion_dropout"],
        mc_dropout_p=cfg_dict["model"]["mc_dropout_p"],
        num_verdict_classes=cfg_dict["model"]["num_verdict_classes"],
        image_patches_side=cfg_dict["model"]["image_patches_side"],
        max_text_tokens=cfg_dict["model"]["max_text_tokens"],
        freeze_image_layers=cfg_dict["model"]["freeze_image_layers"],
        freeze_text_layers=cfg_dict["model"]["freeze_text_layers"],
        grounding_enabled=cfg_dict["model"]["grounding_enabled"],
        uncertainty_samples=cfg_dict["model"]["uncertainty_samples"],
    )

    stats = train_v5(tcfg, model_cfg)
    return {"epochs": [s.__dict__ for s in stats], "out_dir": str(tcfg.out_dir)}


@app.local_entrypoint()
def train_v5_entrypoint(config: str = "v5_2_real") -> None:
    result = train_v5_run.remote(config_name=config)
    print(result)
