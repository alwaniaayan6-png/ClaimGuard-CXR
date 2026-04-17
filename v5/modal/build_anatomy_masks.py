"""Modal entrypoint: precompute anatomy masks for a dataset.

Walks the dataset's images directory, computes 8-region anatomy masks via
`v5.data.anatomy_masks.compute_anatomy_masks`, and writes per-image PNGs to
/data/anatomy_masks_v5/<image_id>/<region>.png.
"""

from __future__ import annotations

from pathlib import Path

import modal

VERIFACT_ROOT = Path(__file__).resolve().parents[2]

app = modal.App("claimguard-v5-anatomy")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        "numpy==1.26.4",
        "pillow==10.4.0",
        "scipy==1.14.1",
        "torchxrayvision==1.3.5",
        "scikit-image==0.24.0",
    )
    .add_local_dir(str(VERIFACT_ROOT), remote_path="/root/verifact", copy=True)
)

volume = modal.Volume.from_name("claimguard-v5-data")


@app.function(image=image, gpu="H100", timeout=60 * 60 * 6, volumes={"/data": volume})
def build_anatomy_masks(site: str, limit: int = 0) -> dict:
    import sys
    from pathlib import Path as P

    sys.path.insert(0, "/root/verifact")
    from v5.data.anatomy_masks import compute_anatomy_masks, save_anatomy_masks

    site_root = P("/data") / site
    out_root = P("/data/anatomy_masks_v5")
    out_root.mkdir(parents=True, exist_ok=True)

    n = 0
    for ext in (".png", ".jpg", ".jpeg"):
        for img in site_root.rglob(f"*{ext}"):
            if limit and n >= limit:
                break
            try:
                masks = compute_anatomy_masks(img)
                save_anatomy_masks(masks, out_root)
                n += 1
            except Exception as exc:  # noqa: BLE001
                print(f"[{site}] anatomy_masks failed on {img}: {exc}")
    return {"site": site, "n_written": n, "out_root": str(out_root)}


@app.local_entrypoint()
def anatomy_entrypoint(site: str = "chexpert_plus", limit: int = 0) -> None:
    print(build_anatomy_masks.remote(site=site, limit=limit))
