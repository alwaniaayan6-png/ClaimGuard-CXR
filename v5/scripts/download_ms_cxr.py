"""Download MS-CXR from HuggingFace and cache locally.

Usage:
    python v5/scripts/download_ms_cxr.py --out /data/ms_cxr

Requires:
    pip install datasets huggingface_hub pillow tqdm
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm


def _check_deps() -> None:
    missing = []
    for pkg in ("datasets", "PIL", "tqdm"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        sys.exit(f"Missing packages: {', '.join(missing)}. Run: pip install datasets pillow tqdm")


def download(out: Path, split: str = "all") -> None:
    """Stream MS-CXR from HF datasets and save images + manifest to *out*."""
    from datasets import load_dataset  # type: ignore

    out.mkdir(parents=True, exist_ok=True)
    img_dir = out / "images"
    img_dir.mkdir(exist_ok=True)

    ds = load_dataset("microsoft/ms_cxr", split=split if split != "all" else None)
    if split == "all":
        # concatenate all splits
        from datasets import concatenate_datasets  # type: ignore

        ds = concatenate_datasets(list(ds.values()))

    manifest = []
    for row in tqdm(ds, desc="MS-CXR download", unit="img"):
        img_id: str = row["dicom_id"]
        img_path = img_dir / f"{img_id}.png"
        if not img_path.exists():
            row["image"].save(img_path)
        record = {
            "dicom_id": img_id,
            "image_path": str(img_path),
            "label_text": row.get("label_text", ""),
            "category_name": row.get("category_name", ""),
            "x": row.get("x", None),
            "y": row.get("y", None),
            "w": row.get("w", None),
            "h": row.get("h", None),
            "split": row.get("split", split),
        }
        manifest.append(record)

    manifest_path = out / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved {len(manifest)} records → {manifest_path}")


def main() -> None:
    _check_deps()
    parser = argparse.ArgumentParser(description="Download MS-CXR dataset")
    parser.add_argument("--out", type=Path, default=Path("/data/ms_cxr"), help="Output directory")
    parser.add_argument(
        "--split",
        choices=["train", "validation", "test", "all"],
        default="all",
        help="Which split(s) to download",
    )
    args = parser.parse_args()
    download(args.out, args.split)


if __name__ == "__main__":
    main()
