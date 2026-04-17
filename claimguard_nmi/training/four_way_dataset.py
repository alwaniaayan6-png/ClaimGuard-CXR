"""Four-way training dataset and collate function.

For each training example, we need four (image, claim, evidence) variants
that together force both the text and image pathways to carry signal.
See ARCHITECTURE_PATH_B.md §4.2 and ``contrastive_loss.py``.

The dataset takes a list of ``TrainingPair`` records plus an image pool
for random-patient image-swap negatives (V3). V4 synthesizes a zero
image on the fly. V1 uses the supplied supporting-evidence text with
the correct image; V2 uses contradicting-evidence text with the
correct image.

The collate function stacks all four variants into a single batch-of-
batches so the model forwards once per variant and the loss module
consumes a dict keyed by variant.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError as e:
    raise ImportError(
        "torch is required for claimguard_nmi.training.four_way_dataset"
    ) from e


@dataclass
class TrainingPair:
    """One training record with supp and contra evidence pre-generated."""
    claim_text: str
    evidence_supp: str
    evidence_contra: str
    image_path: Path
    patient_id: str
    claim_finding: str      # canonical id
    claim_laterality: str
    claim_region: str
    # For V3 image swap negative, we'll pull a random image from a different patient
    # at __getitem__ time; the dataset doesn't store it.


#: Image-loader contract: a ``(3, H, W)`` float32 tensor in ``[0, 1]`` per-channel.
#: The BiomedCLIP preprocessor subsequently normalizes to its training range. Any
#: real loader (DICOM, PNG, JPG) must emit values in this range so that model
#: behavior is identical between training and inference.
IMAGE_VALUE_RANGE = "unit_interval_[0,1]"


def _stub_image_loader(path: Path, size: Tuple[int, int] = (224, 224)) -> "torch.Tensor":
    """Deterministic placeholder returning a path-hashed (3, H, W) tensor in [0, 1].

    Production code must supply a real loader (DICOM + PNG + preprocessor)
    that obeys ``IMAGE_VALUE_RANGE``.
    """
    import hashlib

    rng = np.random.default_rng(
        int(hashlib.md5(str(path).encode()).hexdigest()[:8], 16)
    )
    arr = rng.uniform(low=0.0, high=1.0, size=(3, *size)).astype(np.float32)
    return torch.from_numpy(arr)


class FourWayTrainingDataset(Dataset):
    """PyTorch dataset that yields one TrainingPair per index, as a dict of tensors.

    The DataLoader's collate_fn calls ``collate_four_way`` to stack variants.

    Parameters
    ----------
    pairs : list of TrainingPair
    tokenizer : callable(str) -> dict with 'input_ids', 'attention_mask'
    image_loader : callable(Path) -> (3, H, W) torch.Tensor
    image_pool_by_patient : dict[str, list[Path]]
        Other-patient images to draw V3 swap negatives from.
    max_text_length : int
        Sequence length cap for the tokenizer.
    rng_seed : int
    """

    def __init__(
        self,
        pairs: List[TrainingPair],
        tokenizer: Callable[[str], Dict[str, "torch.Tensor"]],
        image_loader: Callable[[Path], "torch.Tensor"] = _stub_image_loader,
        image_pool_by_patient: Optional[Dict[str, List[Path]]] = None,
        max_text_length: int = 128,
        rng_seed: int = 17,
    ):
        self.pairs = list(pairs)
        self.tokenizer = tokenizer
        self.image_loader = image_loader
        self.image_pool = image_pool_by_patient or {}
        self.max_text_length = max_text_length
        self._rng = np.random.default_rng(rng_seed)

        # Flat, order-matched arrays so V3 sampling is O(1).
        self._all_image_paths: List[Path] = []
        self._patient_of_path: List[str] = []
        for pid, imgs in self.image_pool.items():
            for p in imgs:
                self._all_image_paths.append(p)
                self._patient_of_path.append(pid)
        self._n_images = len(self._all_image_paths)

    def __len__(self) -> int:
        return len(self.pairs)

    _MAX_REJECTION_RETRIES = 4

    def _pick_random_other_patient_image(self, own_patient: str) -> Path:
        """Pick an image path that is NOT from the current patient.

        v2 review fix: previously O(n_patients * n_per_patient) per call via
        list comprehension. Now O(1) via rejection sampling on pre-flattened
        arrays. Fallback: if all retries return own patient (only happens
        when pool has a single patient), return whatever we picked.
        """
        if self._n_images == 0:
            return Path("/dev/null")
        for _ in range(self._MAX_REJECTION_RETRIES):
            idx = int(self._rng.integers(0, self._n_images))
            if self._patient_of_path[idx] != own_patient:
                return self._all_image_paths[idx]
        # Last-resort: return any path (test must ensure >= 2 distinct patients)
        idx = int(self._rng.integers(0, self._n_images))
        return self._all_image_paths[idx]

    def __getitem__(self, idx: int) -> Dict[str, "torch.Tensor"]:
        pair = self.pairs[idx]
        claim_tok = self.tokenizer(pair.claim_text)
        supp_tok = self.tokenizer(pair.evidence_supp)
        contra_tok = self.tokenizer(pair.evidence_contra)

        img_correct = self.image_loader(pair.image_path)
        swap_path = self._pick_random_other_patient_image(pair.patient_id)
        img_swap = self.image_loader(swap_path)
        img_mask = torch.zeros_like(img_correct)

        return {
            "claim_ids": claim_tok["input_ids"],
            "claim_mask": claim_tok["attention_mask"],
            "evidence_supp_ids": supp_tok["input_ids"],
            "evidence_supp_mask": supp_tok["attention_mask"],
            "evidence_contra_ids": contra_tok["input_ids"],
            "evidence_contra_mask": contra_tok["attention_mask"],
            "image_correct": img_correct,
            "image_swap": img_swap,
            "image_mask": img_mask,
            # Labels per variant.
            "label_v1": torch.tensor(0, dtype=torch.long),
            "label_v2": torch.tensor(1, dtype=torch.long),
            "label_v3": torch.tensor(1, dtype=torch.long),
            "label_v4": torch.tensor(1, dtype=torch.long),
        }


def collate_four_way(batch: List[Dict[str, "torch.Tensor"]]) -> Dict[str, "torch.Tensor"]:
    """Collate a list of per-sample dicts into stacked-tensor batch dict."""
    stacked: Dict[str, "torch.Tensor"] = {}
    for key in batch[0]:
        items = [b[key] for b in batch]
        if items[0].dim() == 0:
            stacked[key] = torch.stack(items, dim=0)
        else:
            stacked[key] = torch.stack(items, dim=0)
    return stacked
