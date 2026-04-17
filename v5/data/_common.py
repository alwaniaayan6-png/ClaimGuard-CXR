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
    train_frac: float = 0.8,
    cal_frac: float = 0.1,
    seed: int = 17,
) -> dict[str, list[dict]]:
    """Group-aware split on `key` (patient/report/image) into train/cal/test."""
    import random

    rng = random.Random(seed)
    groups = sorted({r[key] for r in records})
    rng.shuffle(groups)
    n_train = int(len(groups) * train_frac)
    n_cal = int(len(groups) * cal_frac)
    train_keys = set(groups[:n_train])
    cal_keys = set(groups[n_train : n_train + n_cal])
    test_keys = set(groups[n_train + n_cal :])
    out: dict[str, list[dict]] = {"train": [], "cal": [], "test": []}
    for r in records:
        v = r[key]
        if v in train_keys:
            out["train"].append(r)
        elif v in cal_keys:
            out["cal"].append(r)
        else:
            out["test"].append(r)
    return out
