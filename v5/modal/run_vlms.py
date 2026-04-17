"""Modal entrypoint: run the 5 VLM generators across a pool of CXR images.

Produces `/data/vlm_reports/<site>/<generator>/<temperature>_<seed>.jsonl`
with one generated report per row, plus metadata for provenance gating.
"""

from __future__ import annotations

from pathlib import Path

import modal

VERIFACT_ROOT = Path(__file__).resolve().parents[2]

app = modal.App("claimguard-v5-vlms")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        "transformers==4.44.2",
        "accelerate==0.33.0",
        "sentencepiece==0.2.0",
        "numpy==1.26.4",
        "pandas==2.2.2",
        "pillow==10.4.0",
        "pydicom==2.4.4",
    )
    .add_local_dir(str(VERIFACT_ROOT), remote_path="/root/verifact", copy=True)
)

volume = modal.Volume.from_name("claimguard-v5-data")

secrets: list[modal.Secret] = []
try:
    secrets.append(modal.Secret.from_name("huggingface", required=False))
except Exception:
    pass


@app.function(image=image, gpu="H100", timeout=60 * 60 * 8, volumes={"/data": volume}, secrets=secrets)
def run_one_generator(generator: str, site: str, limit: int, temperatures: list[float], seeds: list[int]) -> dict:
    import json
    import sys
    from pathlib import Path as P

    from PIL import Image

    sys.path.insert(0, "/root/verifact")
    from v5.data.vlm_generators import GENERATORS

    out_root = P("/data/vlm_reports") / site / generator
    out_root.mkdir(parents=True, exist_ok=True)

    cls = GENERATORS.get(generator)
    if cls is None:
        raise ValueError(f"Unknown generator {generator}")
    adapter = cls()

    # Simple image-iterator by walking site root for images.
    site_root = P("/data") / site
    img_paths: list[P] = []
    for ext in (".png", ".jpg", ".jpeg", ".dcm"):
        img_paths.extend(sorted(site_root.rglob(f"*{ext}")))
    if limit:
        img_paths = img_paths[:limit]

    n_written = 0
    for t in temperatures:
        for s in seeds:
            out_path = out_root / f"t{t}_s{s}.jsonl"
            with out_path.open("w") as f:
                for ip in img_paths:
                    try:
                        if ip.suffix == ".dcm":
                            import pydicom

                            dcm = pydicom.dcmread(str(ip))
                            arr = dcm.pixel_array
                            img = Image.fromarray(arr.astype("uint8")).convert("RGB")
                        else:
                            img = Image.open(ip).convert("RGB")
                        img.filename = str(ip)  # stash path for image_id
                        report = adapter(img, temperature=float(t), seed=int(s))
                        f.write(json.dumps(
                            {
                                "image_path": str(ip),
                                "image_id": ip.stem,
                                "generator_id": report.generator_id,
                                "generator_version": report.generator_version,
                                "temperature": report.temperature,
                                "seed": report.seed,
                                "report_text": report.report_text,
                                "latency_sec": report.latency_sec,
                                "prompt_version": report.prompt_version,
                            }
                        ) + "\n")
                        n_written += 1
                    except Exception as exc:  # noqa: BLE001
                        print(f"[{generator}] failed on {ip}: {exc}")
    return {"generator": generator, "site": site, "n_written": n_written, "out_root": str(out_root)}


@app.local_entrypoint()
def run_vlms_entrypoint(
    generator: str = "chexagent-8b",
    site: str = "openi",
    limit: int = 100,
    temperatures: str = "0.3,0.7,1.0,1.2",
    seeds: str = "101,202",
) -> None:
    t = [float(x) for x in temperatures.split(",")]
    s = [int(x) for x in seeds.split(",")]
    print(run_one_generator.remote(generator=generator, site=site, limit=limit, temperatures=t, seeds=s))
