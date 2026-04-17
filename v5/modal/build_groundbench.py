"""Modal entrypoint: assemble ClaimGuard-GroundBench from all data sources.

For each site:
  1. Iterate data loader over images.
  2. Collect radiologist annotations from all relevant sources.
  3. Load or compute anatomy masks.
  4. For each report/claim, extract claims + parse them.
  5. Run claim-to-annotation matcher to assign GT.
  6. Write per-site JSONL manifests (train/cal/test split).

Heavy step is (4) + (5) which hits the LLM claim extractor and parser.
Budget and API-key configuration is per-site to allow partial runs.
"""

from __future__ import annotations

from pathlib import Path

import modal

VERIFACT_ROOT = Path(__file__).resolve().parents[2]

app = modal.App("claimguard-v5-groundbench")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        "transformers==4.44.2",
        "hydra-core==1.3.2",
        "omegaconf==2.3.0",
        "numpy==1.26.4",
        "pandas==2.2.2",
        "pyarrow==17.0.0",
        "pillow==10.4.0",
        "scipy==1.14.1",
        "pyyaml==6.0.2",
        "torchxrayvision==1.3.5",
        "anthropic==0.34.2",
        "openai==1.45.0",
        "scikit-image==0.24.0",
    )
    .add_local_dir(str(VERIFACT_ROOT), remote_path="/root/verifact", copy=True)
)

volume = modal.Volume.from_name("claimguard-v5-data")

secrets: list[modal.Secret] = [modal.Secret.from_name("anthropic")]


@app.function(image=image, cpu=8.0, memory=32_000, timeout=60 * 60 * 6, volumes={"/data": volume}, secrets=secrets)
def build_groundbench_for_site(site: str, limit: int = 0, use_llm_extractor: bool = False) -> dict:
    """Assemble GroundBench for one site."""
    import sys

    sys.path.insert(0, "/root/verifact")

    from pathlib import Path as P

    from v5.data.claim_extractor import llm_extract, rule_extract
    from v5.data.claim_parser import llm_parse, rule_parse, load_ontology
    from v5.data.claim_matcher import ClaimMatcher, Annotation
    from v5.data.anatomy_masks import compute_anatomy_masks
    from v5.data.groundbench import assemble_row, split_and_write, aggregate_summary
    from v5.data import chexpert_plus as ds_chexpert
    from v5.data import openi as ds_openi
    from v5.data import padchest as ds_padchest
    from v5.data import brax as ds_brax
    from v5.data import rsna_pneumonia as ds_rsna
    from v5.data import siim_pneumothorax as ds_siim
    from v5.data import object_cxr as ds_objcxr
    from v5.data import chestx_det10 as ds_cx10

    ontology = load_ontology()
    matcher = ClaimMatcher()
    out_root = P("/data/groundbench_v5") / site
    out_root.mkdir(parents=True, exist_ok=True)

    # Dispatcher
    cfg = {
        "chexpert_plus": (ds_chexpert.iter_chexpert_plus, ds_chexpert.annotations_for_record, "/data/chexpert_plus"),
        "openi": (ds_openi.iter_openi, ds_openi.annotations_for_record, "/data/openi"),
        "padchest": (ds_padchest.iter_padchest, ds_padchest.annotations_for_record, "/data/padchest"),
        "brax": (ds_brax.iter_brax, ds_brax.annotations_for_record, "/data/brax"),
        "rsna_pneumonia": (ds_rsna.iter_rsna, ds_rsna.annotations_for_record, "/data/rsna_pneumonia"),
        "siim_acr": (ds_siim.iter_siim, ds_siim.annotations_for_record, "/data/siim_acr_pneumothorax"),
        "object_cxr": (ds_objcxr.iter_object_cxr, ds_objcxr.annotations_for_record, "/data/object_cxr"),
        "chestx_det10": (ds_cx10.iter_chestx_det10, ds_cx10.annotations_for_record, "/data/chestx_det10"),
    }
    if site not in cfg:
        raise ValueError(f"Unknown site {site}; valid: {list(cfg)}")
    iter_fn, ann_fn, root = cfg[site]

    rows = []
    summary_counts = {"images": 0, "claims": 0}

    for idx, rec in enumerate(iter_fn(P(root))):
        if limit and idx >= limit:
            break
        annotations: list[Annotation] = ann_fn(rec)
        report_text = ""
        patient_id = None
        age = None
        sex = None
        scanner = None
        country = None
        if site == "chexpert_plus":
            report_text = f"{rec.report_findings}\n\n{rec.report_impression}"
            patient_id = rec.patient_id
            country = "US"
        elif site == "openi":
            report_text = f"{rec.report_findings}\n\n{rec.report_impression}"
            country = "US"
        elif site == "padchest":
            report_text = rec.report_en or rec.report_raw
            country = "ES"
        elif site == "brax":
            report_text = rec.report_en or rec.report_pt
            country = "BR"
        elif site in {"rsna_pneumonia", "siim_acr", "object_cxr", "chestx_det10"}:
            report_text = ""  # no reports; skip claim extraction
        summary_counts["images"] += 1

        # Extract claims (only for sites that have reports)
        if report_text.strip():
            extractor_fn = llm_extract if use_llm_extractor else rule_extract
            parser_fn = llm_parse if use_llm_extractor else rule_parse
            try:
                claims = extractor_fn(report_text, report_id=str(rec.image_id))
            except Exception as exc:  # noqa: BLE001
                print(f"[{site}] extractor failed on {rec.image_id}: {exc}")
                continue
            for cl in claims:
                try:
                    structured = parser_fn(
                        cl.claim_text, cl.claim_id, cl.report_id, ontology=ontology,
                        evidence_source_type="oracle_human", generator_id="oracle",
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"[{site}] parser failed on claim {cl.claim_id}: {exc}")
                    continue
                try:
                    anatomy = compute_anatomy_masks(rec.image_path)
                except Exception as exc:  # noqa: BLE001
                    print(f"[{site}] anatomy_masks failed on {rec.image_id}: {exc}")
                    anatomy = None
                row = assemble_row(
                    structured,
                    annotations,
                    anatomy,
                    image_path=rec.image_path,
                    source_site=site,
                    evidence_text=report_text,
                    patient_id=patient_id,
                    sex=sex,
                    age=age,
                    scanner_manufacturer=scanner,
                    country=country,
                    matcher=matcher,
                )
                rows.append(row)
                summary_counts["claims"] += 1
        # Sites without reports: still produce "dummy" anchor rows for use as
        # image-grounded test-set (claims will come from VLMs in later passes).

    split_and_write(rows, out_root, seed=17)
    summary = aggregate_summary(rows)
    summary.update(summary_counts)
    (out_root / "summary.json").write_text(__import__("json").dumps(summary, indent=2, default=str))
    return summary


@app.local_entrypoint()
def build_groundbench_entrypoint(site: str = "chexpert_plus", limit: int = 0, use_llm: bool = False) -> None:
    print(build_groundbench_for_site.remote(site=site, limit=limit, use_llm_extractor=use_llm))
