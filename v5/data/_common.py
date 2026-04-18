"""Shared utilities for v5 data loaders."""

from __future__ import annotations

import csv
import hashlib
import logging
from pathlib import Path
from typing import Iterable, Iterator

logger = logging.getLogger(__name__)


def stable_id(*parts: str) -> str:
    """Deterministic short id derived from arbitrary string parts."""
    h = hashlib.sha1("::".join(parts).encode("utf-8")).hexdigest()
    return h[:16]


def read_jsonl(path: Path) -> Iterator[dict]:
    import json

    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(records: Iterable[dict], path: Path) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def read_csv(path: Path) -> Iterator[dict]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def ensure_split(
    records: list[dict],
    key: str,
    *,
    train_frac: float = 0.7,
    val_frac: float = 0.1,
    cal_frac: float = 0.1,
    seed: int = 17,
) -> dict[str, list[dict]]:
    """Group-aware split on `key` (patient/report/image) into train/val/cal/test.

    val is used for early stopping and hyperparameter selection.
    cal is used for conformal calibration and must be disjoint from val to
    preserve exchangeability guarantees. Never collapse val and cal (v4 bug).
    """
    import random

    if train_frac + val_frac + cal_frac >= 1.0:
        raise ValueError(
            f"train_frac + val_frac + cal_frac must be < 1.0 (got "
            f"{train_frac}+{val_frac}+{cal_frac}={train_frac+val_frac+cal_frac}). "
            "Remainder is the test split."
        )

    rng = random.Random(seed)
    groups = sorted({r[key] for r in records})
    rng.shuffle(groups)
    n = len(groups)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    n_cal = int(n * cal_frac)
    train_keys = set(groups[:n_train])
    val_keys = set(groups[n_train : n_train + n_val])
    cal_keys = set(groups[n_train + n_val : n_train + n_val + n_cal])
    test_keys = set(groups[n_train + n_val + n_cal :])

    out: dict[str, list[dict]] = {"train": [], "val": [], "cal": [], "test": []}
    for r in records:
        v = r[key]
        if v in train_keys:
            out["train"].append(r)
        elif v in val_keys:
            out["val"].append(r)
        elif v in cal_keys:
            out["cal"].append(r)
        else:
            out["test"].append(r)
    return out
