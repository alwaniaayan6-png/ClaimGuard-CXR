"""SIIM-ACR Pneumothorax Segmentation loader.

12,047 CXRs with radiologist-drawn pixel masks for pneumothorax.
Access: Kaggle (SIIM-ACR Pneumothorax Segmentation) — public.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from .claim_matcher import Annotation


@dataclass
class SIIMRecord:
    image_id: str
    image_path: Path
    rle: str | None   # run-length-encoded mask, " -1" if no pneumothorax
    image_width: int = 1024
    image_height: int = 1024


def rle_decode(rle: str, shape: tuple[int, int]) -> np.ndarray:
    """Decode SIIM's space-separated (start length)* RLE encoding."""
    if rle is None or rle.strip() in ("-1", ""):
        return np.zeros(shape, dtype=np.uint8)
    s = [int(x) for x in rle.split()]
    starts, lengths = s[::2], s[1::2]
    H, W = shape
    mask = np.zeros(H * W, dtype=np.uint8)
    for start, length in zip(starts, lengths, strict=False):
        mask[start : start + length] = 1
    # SIIM encodes in Fortran (column-major) order
    return mask.reshape((W, H)).T


def iter_siim(root: Path) -> Iterator[SIIMRecord]:
    import pandas as pd

    df = pd.read_csv(root / "train-rle.csv")
    for _, row in df.iterrows():
        yield SIIMRecord(
            image_id=str(row["ImageId"]),
            image_path=root / "images" / f"{row['ImageId']}.png",
            rle=row[" EncodedPixels"] if " EncodedPixels" in row else row.get("EncodedPixels"),
        )


def annotations_for_record(rec: SIIMRecord) -> list[Annotation]:
    has_ptx = rec.rle is not None and rec.rle.strip() not in ("-1", "")
    if not has_ptx:
        return [
            Annotation(
                image_id=rec.image_id,
                finding="pneumothorax",
                source="siim_acr",
                is_structured_negative=True,  # explicit absence
            )
        ]
    mask = rle_decode(rec.rle, (rec.image_height, rec.image_width))
    return [
        Annotation(
            image_id=rec.image_id,
            finding="pneumothorax",
            mask=mask,
            source="siim_acr",
        )
    ]
