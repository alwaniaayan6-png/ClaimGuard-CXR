"""ClaimGuard-GroundBench assembler.

Unifies radiologist-annotated sources into a single Parquet manifest. For each
(image, claim) pair, writes:
  - image_id, image_path, source_site
  - claim_id, claim_text, claim_struct (parsed)
  - gt_label (SUPPORTED / CONTRADICTED / NO_GT / NO_FINDING)
  - matching_annotation_id, spatial_overlap, matcher_reason
  - silver_label (from LLM ensemble) — secondary
  - evidence_text, evidence_source_type, evidence_trust_tier, provenance_metadata
  - VLM generator metadata (if applicable)
  - subgroup metadata (sex, age, scanner, country)

The output split follows patient-/image-disjoint train/cal/test.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from ._common import ensure_split, write_jsonl
from .claim_matcher import AnatomyMask, Annotation, ClaimMatcher, MatchResult
from .claim_parser import StructuredClaim

logger = logging.getLogger(__name__)


@dataclass
class GroundBenchRow:
    image_id: str
    image_path: str
    source_site: str  # chexpert_plus | openi | padchest | brax | msc_xr | rsna | siim | object_cxr | chestx_det10
    claim_id: str
    claim_text: str
    claim_struct: dict
    gt_label: str
    gt_coverage: bool
    matching_annotation_id: str | None
    spatial_overlap: float
    matcher_reason: str
    silver_label: str | None = None
    silver_agreement: float | None = None
    evidence_text: str | None = None
    evidence_source_type: str = "unknown"
    evidence_trust_tier: str = "unknown"
    claim_generator_id: str = "oracle"
    evidence_generator_id: str | None = None
    generator_temperature: float | None = None
    generator_seed: int | None = None
    patient_id: str | None = None
    sex: str | None = None
    age: float | None = None
    scanner_manufacturer: str | None = None
    country: str | None = None


@dataclass
class GroundBenchConfig:
    out_dir: Path
    matcher: ClaimMatcher
    run_silver: bool = False


def assemble_row(
    structured: StructuredClaim,
    annotations: list[Annotation],
    anatomy: AnatomyMask | None,
    *,
    image_path: Path,
    source_site: str,
    evidence_text: str | None = None,
    patient_id: str | None = None,
    sex: str | None = None,
    age: float | None = None,
    scanner_manufacturer: str | None = None,
    country: str | None = None,
    matcher: ClaimMatcher | None = None,
) -> GroundBenchRow:
    matcher = matcher or ClaimMatcher()
    result: MatchResult = matcher.match(structured, annotations, anatomy)
    return GroundBenchRow(
        image_id=structured.report_id if not structured.claim_id else anatomy.image_id if anatomy else structured.report_id,
        image_path=str(image_path),
        source_site=source_site,
        claim_id=structured.claim_id,
        claim_text=structured.raw_text,
        claim_struct=structured.to_dict(),
        gt_label=result.label.value,
        gt_coverage=result.coverage,
        matching_annotation_id=result.matching_annotation_id,
        spatial_overlap=result.spatial_overlap,
        matcher_reason=result.reason,
        evidence_text=evidence_text,
        evidence_source_type=structured.evidence_source_type,
        evidence_trust_tier="trusted" if structured.evidence_source_type == "oracle_human" else "unknown",
        claim_generator_id=structured.generator_id,
        evidence_generator_id=None,
        generator_temperature=structured.generator_temperature,
        generator_seed=structured.generator_seed,
        patient_id=patient_id,
        sex=sex,
        age=age,
        scanner_manufacturer=scanner_manufacturer,
        country=country,
    )


def split_and_write(
    rows: list[GroundBenchRow],
    out_dir: Path,
    *,
    key: str = "patient_id",
    seed: int = 17,
) -> None:
    dicts = [asdict(r) for r in rows]
    # For rows missing patient_id, fall back to image_id so rows aren't lost.
    for d in dicts:
        if d.get(key) is None:
            d[key] = d["image_id"]
    splits = ensure_split(dicts, key=key, seed=seed)
    for split_name, records in splits.items():
        p = out_dir / f"groundbench_v5_{split_name}.jsonl"
        write_jsonl(records, p)
        logger.info("wrote %d rows to %s", len(records), p)


def aggregate_summary(rows: list[GroundBenchRow]) -> dict:
    """Return basic statistics on the assembled GroundBench."""
    from collections import Counter

    total = len(rows)
    label_counts = Counter(r.gt_label for r in rows)
    coverage = sum(r.gt_coverage for r in rows) / max(total, 1)
    per_site = Counter(r.source_site for r in rows)
    per_gen = Counter(r.claim_generator_id for r in rows)
    return {
        "n_rows": total,
        "label_distribution": dict(label_counts),
        "coverage_fraction": coverage,
        "per_site": dict(per_site),
        "per_generator": dict(per_gen),
    }
