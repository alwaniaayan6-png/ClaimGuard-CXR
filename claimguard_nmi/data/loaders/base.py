"""Base loader contract for per-dataset ingestion.

Every site loader returns a stream of ``GroundedStudy`` records so
downstream grounding, evaluation, and conformal code does not need to
know which dataset it's processing.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

import numpy as np

from claimguard_nmi.grounding.grounding import Annotation


@dataclass
class GroundedStudy:
    """One (image, annotations, optional report) record from a dataset.

    Attributes
    ----------
    study_id : str
        Stable id unique within a dataset (often = filename stem).
    patient_id : str | None
        Patient id if the dataset exposes one. Used for patient-disjoint
        splits. When None, the loader falls back to study-disjoint splits.
    image_path : Path
        Absolute filesystem path to the CXR image.
    image_shape : tuple[int, int] | None
        (H, W). Loaders may populate lazily; None means "read from file on demand".
    annotations : dict[str, list[Annotation]]
        Canonical-finding-keyed radiologist annotations. Empty dict means
        "this image has no positive annotations drawn by radiologists"
        which is semantically different from "this finding is absent from
        the image" — the dataset_label_schema gate in grounding.py
        enforces that distinction.
    dataset_label_schema : frozenset[str]
        Which canonical findings the dataset's radiologists were asked to
        draw. Absence of a finding not in the schema means UNKNOWN, not ABSENT.
    report_text : str | None
        Radiologist-written reference report, if any.
    split : str
        'train' | 'cal' | 'test'.
    metadata : dict
        Arbitrary per-study metadata (age, sex, scanner, etc.) for fairness
        stratification.
    """
    study_id: str
    patient_id: Optional[str]
    image_path: Path
    image_shape: Optional[tuple] = None
    annotations: Dict[str, List[Annotation]] = field(default_factory=dict)
    dataset_label_schema: frozenset = field(default_factory=frozenset)
    report_text: Optional[str] = None
    split: str = "test"
    metadata: Dict[str, object] = field(default_factory=dict)


class BaseLoader(ABC):
    """Abstract interface every site loader implements."""

    #: Site name matching configs/datasets.yaml.
    site_name: str = ""

    @property
    @abstractmethod
    def label_schema(self) -> frozenset:
        """Canonical findings the dataset's annotators drew for."""

    @abstractmethod
    def iter_studies(self) -> Iterator[GroundedStudy]:
        """Yield all studies in the dataset."""

    # ------------------------------------------------------------------
    # Convenience wrappers for downstream training / eval code.
    # ------------------------------------------------------------------
    def iter_split(self, split: str) -> Iterator[GroundedStudy]:
        for study in self.iter_studies():
            if study.split == split:
                yield study

    def counts(self) -> Dict[str, int]:
        cnt = {"train": 0, "cal": 0, "test": 0, "total": 0}
        for s in self.iter_studies():
            cnt[s.split] = cnt.get(s.split, 0) + 1
            cnt["total"] += 1
        return cnt
