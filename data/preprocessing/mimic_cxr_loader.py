"""CheXpert Plus data loader for ClaimGuard-CXR.

Loads CheXpert Plus images and reports with patient-level split filtering.
Despite the filename (kept for compatibility with the handoff doc), this
module is designed for CheXpert Plus as the primary dataset.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Callable, Literal, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)

# Default normalization (ImageNet stats, standard for pretrained ViTs)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def parse_report(report_text: str) -> dict[str, str]:
    """Extract FINDINGS and IMPRESSION sections from a radiology report.

    Args:
        report_text: Raw report text.

    Returns:
        Dict with keys 'findings', 'impression', 'full'. Missing sections are empty strings.
    """
    if not isinstance(report_text, str):
        return {"findings": "", "impression": "", "full": ""}

    text = report_text.strip()
    result = {"findings": "", "impression": "", "full": text}

    # Try to extract FINDINGS section
    findings_match = re.search(
        r"(?:FINDINGS|Findings)[:\s]*\n?(.*?)(?=(?:IMPRESSION|Impression|$))",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if findings_match:
        result["findings"] = findings_match.group(1).strip()

    # Try to extract IMPRESSION section
    impression_match = re.search(
        r"(?:IMPRESSION|Impression)[:\s]*\n?(.*?)$",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if impression_match:
        result["impression"] = impression_match.group(1).strip()

    # If no sections found, treat entire text as findings
    if not result["findings"] and not result["impression"]:
        result["findings"] = text
        result["impression"] = text

    return result


def build_transforms(
    image_size: int = 384,
    augment: bool = False,
    mean: list[float] = IMAGENET_MEAN,
    std: list[float] = IMAGENET_STD,
) -> Callable:
    """Build image transforms for training or inference.

    Args:
        image_size: Target image size (square).
        augment: Whether to apply data augmentation (training only).
        mean: Normalization mean.
        std: Normalization std.

    Returns:
        torchvision transform pipeline.
    """
    if augment:
        return transforms.Compose([
            transforms.Resize(image_size + 32),
            transforms.RandomCrop(image_size),
            # NO horizontal flip for CXR — laterality matters!
            transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])


class CheXpertPlusDataset(Dataset):
    """PyTorch Dataset for CheXpert Plus images and reports.

    Supports patient-level split filtering to ensure no data leakage.

    Args:
        data_root: Path to CheXpert Plus root directory.
        patient_ids: List of patient IDs to include (from split file).
        image_size: Target image resolution.
        section: Which report section to return ('findings', 'impression', 'both', 'full').
        transform: Optional custom transform (overrides default).
        augment: Whether to apply data augmentation.
        max_samples: Maximum number of samples (for debugging).
    """

    def __init__(
        self,
        data_root: str | Path,
        patient_ids: list[str],
        image_size: int = 384,
        section: Literal["findings", "impression", "both", "full"] = "findings",
        transform: Optional[Callable] = None,
        augment: bool = False,
        max_samples: Optional[int] = None,
    ):
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.section = section
        self.transform = transform or build_transforms(image_size, augment=augment)

        # Load metadata and filter to specified patients
        self._load_and_filter(patient_ids, max_samples)

    def _load_and_filter(self, patient_ids: list[str], max_samples: Optional[int]) -> None:
        """Load metadata CSV and filter to the given patient split."""
        from .patient_splits import load_chexpert_plus_metadata

        meta = load_chexpert_plus_metadata(self.data_root)
        patient_set = set(str(pid) for pid in patient_ids)
        meta["patient_id"] = meta["patient_id"].astype(str)

        self.records = meta[meta["patient_id"].isin(patient_set)].reset_index(drop=True)

        if max_samples is not None:
            self.records = self.records.head(max_samples)

        logger.info(
            f"CheXpertPlusDataset: {len(self.records)} records from "
            f"{self.records['patient_id'].nunique()} patients "
            f"(filtered from {len(meta)} total records)"
        )

    def _load_image(self, idx: int) -> Image.Image:
        """Load and convert a single image to RGB."""
        row = self.records.iloc[idx]

        # Find image path column
        path_col = None
        for col in ["image_path", "Path", "path", "dicom_path"]:
            if col in row.index:
                path_col = col
                break

        if path_col is None:
            raise ValueError(f"No image path column found. Available: {list(row.index)}")

        img_path = self.data_root / str(row[path_col])

        # Handle DICOM vs standard image formats
        if img_path.suffix.lower() in (".dcm", ".dicom"):
            import pydicom
            dcm = pydicom.dcmread(str(img_path))
            pixel_array = dcm.pixel_array.astype(np.float32)
            # Normalize to 0-255
            prange = pixel_array.max() - pixel_array.min()
            if prange < 1e-6:
                logger.warning(f"Uniform pixel values in DICOM: {img_path}")
            pixel_array = (pixel_array - pixel_array.min()) / (prange + 1e-8) * 255
            img = Image.fromarray(pixel_array.astype(np.uint8)).convert("RGB")
        else:
            img = Image.open(img_path).convert("RGB")

        return img

    def _load_report(self, idx: int) -> str:
        """Load report text for a given index."""
        row = self.records.iloc[idx]
        report_text = ""

        # Try inline report text column first
        for col in ["report", "Report", "report_text", "text", "impression", "findings"]:
            if col in row.index and pd.notna(row[col]):
                report_text = str(row[col])
                break
        else:
            # Try loading from report file
            for col in ["report_path", "report_file"]:
                if col in row.index and pd.notna(row[col]):
                    report_path = self.data_root / str(row[col])
                    if report_path.exists():
                        report_text = report_path.read_text().strip()
                        break

        sections = parse_report(report_text)

        if self.section == "findings":
            return sections["findings"]
        elif self.section == "impression":
            return sections["impression"]
        elif self.section == "both":
            parts = []
            if sections["findings"]:
                parts.append(sections["findings"])
            if sections["impression"]:
                parts.append(sections["impression"])
            return " ".join(parts)
        else:
            return sections["full"]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        """Return a single sample.

        Returns:
            Dict with keys:
                - 'image': tensor of shape (3, H, W)
                - 'report': string
                - 'patient_id': string
                - 'study_id': string (if available)
                - 'labels': dict of CheXpert labels (if available)
        """
        img = self._load_image(idx)
        img_tensor = self.transform(img)
        report = self._load_report(idx)

        row = self.records.iloc[idx]
        sample = {
            "image": img_tensor,
            "report": report,
            "patient_id": str(row["patient_id"]),
        }

        if "study_id" in row.index:
            sample["study_id"] = str(row["study_id"])

        # Include CheXpert labels if available
        chexpert_labels = [
            "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
            "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
            "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
            "Pleural Other", "Fracture", "Support Devices",
        ]
        labels = {}
        for label in chexpert_labels:
            if label in row.index:
                val = row[label]
                # CheXpert convention: 1=positive, 0=negative, -1=uncertain, NaN=unmentioned
                if pd.isna(val):
                    labels[label] = -2  # unmentioned
                else:
                    labels[label] = int(val)
        if labels:
            sample["labels"] = labels

        return sample


def get_dataloader(
    data_root: str | Path,
    patient_ids: list[str],
    batch_size: int = 8,
    image_size: int = 384,
    section: str = "findings",
    augment: bool = False,
    num_workers: int = 4,
    shuffle: bool = True,
    max_samples: Optional[int] = None,
) -> torch.utils.data.DataLoader:
    """Create a DataLoader for CheXpert Plus.

    Args:
        data_root: Path to CheXpert Plus root.
        patient_ids: Patient IDs for this split.
        batch_size: Batch size.
        image_size: Target image resolution.
        section: Report section to return.
        augment: Whether to augment.
        num_workers: DataLoader workers.
        shuffle: Whether to shuffle.
        max_samples: Limit dataset size (for debugging).

    Returns:
        PyTorch DataLoader.
    """
    dataset = CheXpertPlusDataset(
        data_root=data_root,
        patient_ids=patient_ids,
        image_size=image_size,
        section=section,
        augment=augment,
        max_samples=max_samples,
    )

    def collate_fn(batch):
        """Custom collate to handle mixed types (tensors + strings)."""
        images = torch.stack([b["image"] for b in batch])
        reports = [b["report"] for b in batch]
        patient_ids = [b["patient_id"] for b in batch]

        result = {
            "image": images,
            "report": reports,
            "patient_id": patient_ids,
        }

        if all("labels" in b for b in batch):
            # Stack label dicts into a dict of tensors (only if ALL samples have labels)
            label_keys = batch[0]["labels"].keys()
            result["labels"] = {
                k: torch.tensor([b["labels"][k] for b in batch])
                for k in label_keys
            }

        return result

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
    )
