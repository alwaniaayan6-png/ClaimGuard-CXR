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
    image_id = anatomy.image_id if anatomy is not None else structured.report_id
    return GroundBenchRow(
        image_id=image_id,
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


def aggregate_groundbench(
    groundbench_root: Path,
    *,
    sites: list[str] | None = None,
    splits: tuple[str, ...] = ("train", "val", "cal", "test"),
) -> dict:
    """Concat per-site JSONL splits into groundbench_root/all/{split}.jsonl.

    Reads `groundbench_root/<site>/groundbench_v5_<split>.jsonl` for every
    site in `sites` (auto-discovered if None) and writes the unioned rows to
    `groundbench_root/all/groundbench_v5_<split>.jsonl`.

    This is the aggregation step that v5 training loads from. Without it,
    `/data/groundbench_v5/all/` does not exist and training cannot find the
    unioned manifest.
    """
    from ._common import read_jsonl, write_jsonl

    groundbench_root = Path(groundbench_root)
    if sites is None:
        sites = sorted(
            p.name
            for p in groundbench_root.iterdir()
            if p.is_dir() and p.name != "all"
        )

    all_dir = groundbench_root / "all"
    all_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict] = {}
    for split in splits:
        unioned: list[dict] = []
        per_site_counts: dict[str, int] = {}
        for site in sites:
            jsonl = groundbench_root / site / f"groundbench_v5_{split}.jsonl"
            if not jsonl.exists():
                logger.warning("missing %s; skipping", jsonl)
                per_site_counts[site] = 0
                continue
            rows = list(read_jsonl(jsonl))
            unioned.extend(rows)
            per_site_counts[site] = len(rows)
        out_path = all_dir / f"groundbench_v5_{split}.jsonl"
        write_jsonl(unioned, out_path)
        logger.info("aggregated %d rows to %s", len(unioned), out_path)
        summary[split] = {"total": len(unioned), "per_site": per_site_counts}

    manifest = all_dir / "aggregate_manifest.json"
    import json as _json

    manifest.write_text(_json.dumps(summary, indent=2))
    return summary
