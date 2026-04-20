"""Modal orchestrator for the v6.0 NeurIPS 2026 sprint.

Provides server-side entrypoints for every v6-specific workflow so the local
client can disconnect safely (all jobs built via ``modal deploy`` + ``.spawn()``
per the V5_TIER3_RUNBOOK pattern).

Each entrypoint is idempotent: if its primary output file exists and is
non-empty, the step logs and skips. Failures in one step do not halt others;
each writes its own summary JSON under ``/data/v6_results/`` and the local
collector picks up whatever completed.

Deployment:

    cd /Users/aayanalwani/VeriFact/verifact
    modal deploy v5/modal/v6_orchestrator.py

Spawn a single entrypoint:

    python -c "import modal; \
        f = modal.Function.from_name('claimguard-v6-orchestrator', 'rrg_generation'); \
        call = f.spawn(); print(call.object_id)"

See ``v5/scripts/reproduce_v6.sh`` for the deterministic end-to-end order.
"""

from __future__ import annotations

from pathlib import Path

import modal

VERIFACT_ROOT = (
    Path(__file__).resolve().parent.parent.parent
    if Path(__file__).resolve().parent.name == "modal"
    else Path("/root/verifact")
)

app = modal.App("claimguard-v6-orchestrator")


# CheXagent-2-3b's remote-code modeling file hard-checks ``transformers==4.40.0``
# and also needs a specific older albumentations/albucore pair. MAIRA-2
# requires ``transformers>=4.48,<4.52`` (tested on 4.51.3) and MedGemma-4B-IT
# uses the modern image-text-to-text pipeline that also needs newer
# transformers. The two version ranges are mutually exclusive, so we build
# TWO images.
#
# - image_chexagent: transformers==4.40.0 + CheXagent's CV dep stack
# - image_recent:    transformers==4.51.3 + modern silver labelers + MAIRA-2
#                    + MedGemma-4B + everything else

_chexagent_deps = (
    "matplotlib==3.9.2",
    "tensorflow-cpu==2.17.0",
    "albumentations==1.4.3",
    "albucore==0.0.13",
    "opencv-python-headless==4.10.0.84",
    "einops==0.8.0",
    "timm==1.0.9",
    "ftfy==6.2.3",
    "regex==2024.9.11",
    "protobuf==4.25.5",
)

image_chexagent = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        "transformers==4.40.0",
        "sentencepiece==0.2.0",
        "accelerate==0.30.1",
        "numpy==1.26.4",
        "pandas==2.2.2",
        "pyarrow==17.0.0",
        "pillow==10.4.0",
        "scipy==1.14.1",
        "pyyaml==6.0.2",
        "anthropic==0.42.0",
        "huggingface_hub==0.23.5",
        "scikit-learn==1.5.1",
        "scikit-image==0.24.0",
        *_chexagent_deps,
    )
    .add_local_dir(str(VERIFACT_ROOT), remote_path="/root/verifact", copy=True)
)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        "transformers==4.51.3",
        "open-clip-torch==2.26.1",
        "sentencepiece==0.2.0",
        "accelerate==0.33.0",
        "scikit-learn==1.5.1",
        "numpy==1.26.4",
        "pandas==2.2.2",
        "pyarrow==17.0.0",
        "pillow==10.4.0",
        "scipy==1.14.1",
        "pyyaml==6.0.2",
        "torchxrayvision==1.3.5",
        "anthropic==0.42.0",
        "scikit-image==0.24.0",
        "huggingface_hub>=0.30.0,<1.0",
        # GREEN-RadLlama2-7b's trust_remote_code modeling file imports
        # matplotlib + einops + timm at module-load time. Include here
        # so silver_green doesn't crash on import. (Same class of failure
        # as CheXagent; different container, same root cause.)
        "matplotlib==3.9.2",
        "einops==0.8.0",
        "timm==1.0.9",
    )
    .add_local_dir(str(VERIFACT_ROOT), remote_path="/root/verifact", copy=True)
)

volume = modal.Volume.from_name("claimguard-v5-data", create_if_missing=True)

secrets: list[modal.Secret] = []
for name in ("anthropic", "huggingface"):
    try:
        secrets.append(modal.Secret.from_name(name))
    except Exception:
        pass


_H100 = "H100"
# Modal H100 SKU is 80 GB by default; higher memory or multi-GPU would use
# "H100:2" (2 GPUs), not "H100:80GB".
_H100_80 = "H100"

_STATUS_PATH = "/data/v6_results/status.json"


def _write_status(step: str, payload: dict) -> None:
    import json
    from pathlib import Path as P
    P("/data/v6_results").mkdir(parents=True, exist_ok=True)
    status_path = P(_STATUS_PATH)
    data = {}
    if status_path.exists():
        try:
            data = json.loads(status_path.read_text())
        except Exception:
            data = {}
    data[step] = payload
    status_path.write_text(json.dumps(data, indent=2, default=str))


def _skip_if_exists(path: str, step: str) -> bool:
    from pathlib import Path as P
    p = P(path)
    if p.exists() and p.stat().st_size > 0:
        import json
        payload = {"status": "skipped", "reason": "output exists", "output": str(p)}
        _write_status(step, payload)
        print(f"[v6 orchestrator] {step}: output exists at {p}, skipping")
        return True
    return False


# ---------------------------------------------------------------------------
# Day 2: RRG generation
# ---------------------------------------------------------------------------

def _load_manifest(manifest_jsonl: str) -> list[dict]:
    import json
    rows: list[dict] = []
    with open(manifest_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows.append({"image_id": row.get("image_id") or row.get("row_id"),
                          "image_path": row["image_path"]})
    seen: set[str] = set()
    dedup: list[dict] = []
    for r in rows:
        iid = str(r["image_id"])
        if iid in seen:
            continue
        seen.add(iid)
        dedup.append(r)
    return dedup


@app.function(
    image=image_chexagent,
    gpu=_H100,
    timeout=60 * 60 * 2,
    volumes={"/data": volume},
    secrets=secrets,
)
def rrg_generation_chexagent(
    manifest_jsonl: str = "/data/groundbench_v5/all/groundbench_v5_test.jsonl",
    out_jsonl: str = "/data/v6_rrg/generations.jsonl",
    max_images: int = 500,
) -> dict:
    """CheXagent-2-3b only, on the transformers-4.40.0 image. Safe to run
    immediately — CheXagent weights are MIT-licensed and require no gated access."""
    import sys
    sys.path.insert(0, "/root/verifact")
    from pathlib import Path as P

    from v5.eval.rrg_generate import CheXagentV2Generator, run_rrg_sweep
    try:
        generators = [CheXagentV2Generator(device="cuda")]
    except Exception as exc:
        payload = {"status": "failed", "reason": f"chexagent load failed: {exc}"}
        _write_status("rrg_generation_chexagent", payload)
        return payload

    counts = run_rrg_sweep(
        generators=generators,
        manifest=_load_manifest(manifest_jsonl),
        image_root=P("/data"),
        out_jsonl=P(out_jsonl),
        max_images=max_images,
    )
    payload = {"status": "ok", "counts": counts, "output": out_jsonl}
    _write_status("rrg_generation_chexagent", payload)
    return payload


@app.function(
    image=image,
    gpu=_H100,
    timeout=60 * 60 * 3,
    volumes={"/data": volume},
    secrets=secrets,
)
def rrg_generation(
    manifest_jsonl: str = "/data/groundbench_v5/all/groundbench_v5_test.jsonl",
    out_jsonl: str = "/data/v6_rrg/generations.jsonl",
    max_images: int = 500,
) -> dict:
    """MAIRA-2 + MedGemma-4B-IT on the transformers-4.51.3 image. Requires
    the user to have accepted the HF gated-repo licenses; otherwise both
    loaders fall through with 403 and the function emits an empty counts
    dict.
    """
    import sys
    sys.path.insert(0, "/root/verifact")
    from pathlib import Path as P

    from v5.eval.rrg_generate import MAIRA2Generator, MedGemma4BGenerator, run_rrg_sweep
    generators = []
    for name, ctor in [("maira-2", MAIRA2Generator), ("medgemma-4b-it", MedGemma4BGenerator)]:
        try:
            generators.append(ctor(device="cuda"))
            print(f"[rrg_generation] loaded {name}")
        except Exception as exc:
            print(f"[rrg_generation] {name} load failed: {exc}")

    if not generators:
        payload = {"status": "failed", "reason": "no gated RRG generators loaded — check HF licenses"}
        _write_status("rrg_generation", payload)
        return payload

    counts = run_rrg_sweep(
        generators=generators,
        manifest=_load_manifest(manifest_jsonl),
        image_root=P("/data"),
        out_jsonl=P(out_jsonl),
        max_images=max_images,
    )
    payload = {"status": "ok", "counts": counts, "output": out_jsonl}
    _write_status("rrg_generation", payload)
    return payload


# ---------------------------------------------------------------------------
# Day 3: silver labeling (GREEN on H100)
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    cpu=2,
    timeout=60 * 60 * 2,
    volumes={"/data": volume},
    secrets=secrets,
)
def decompose_rrg_claims(
    rrg_jsonl: str = "/data/v6_rrg/generations.jsonl",
    manifest_jsonl: str = "/data/groundbench_v5/all/groundbench_v5_test.jsonl",
    out_claims_jsonl: str = "/data/v6_silver/claims.jsonl",
    out_refs_jsonl: str = "/data/v6_silver/references.jsonl",
) -> dict:
    """Decompose RRG-generated reports into atomic claims via Claude Haiku.

    Also emits a per-image reference_report JSONL sourced from the test
    manifest's ``evidence_text`` field (where ``evidence_source_type=oracle_human``),
    which is the original OpenI radiologist report. Both files feed the
    silver labelers (green/radfact/vert).
    """
    import sys
    sys.path.insert(0, "/root/verifact")
    from pathlib import Path as P
    import json

    if _skip_if_exists(out_claims_jsonl, "decompose_rrg_claims"):
        return {"status": "skipped"}

    from v5.data.claim_extractor import llm_extract, rule_extract

    refs: dict[str, str] = {}
    with open(manifest_jsonl) as f:
        for line in f:
            r = json.loads(line)
            iid = str(r.get("image_id") or r.get("row_id", ""))
            if not iid or iid in refs:
                continue
            if r.get("evidence_source_type") != "oracle_human":
                continue
            text = str(r.get("evidence_text", "")).strip()
            if text:
                refs[iid] = text

    P(out_refs_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(out_refs_jsonl, "w") as f:
        for iid, ref in refs.items():
            f.write(json.dumps({"image_id": iid, "reference_report": ref}) + "\n")
    print(f"[decompose] wrote {len(refs)} references")

    claims_out: list[dict] = []
    n_gens, n_rule_fallback, n_skipped_no_ref = 0, 0, 0
    with open(rrg_jsonl) as f:
        for line in f:
            g = json.loads(line)
            if g.get("error") or not g.get("generated_report"):
                continue
            image_id = str(g["image_id"])
            if image_id not in refs:
                n_skipped_no_ref += 1
                continue
            report_id = f"{image_id}__{g['model']}"
            try:
                claims = llm_extract(g["generated_report"], report_id=report_id,
                                     provider="anthropic",
                                     model="claude-haiku-4-5-20251001")
            except Exception as exc:
                print(f"[decompose] llm_extract failed for {image_id}: {exc}; using rule")
                claims = rule_extract(g["generated_report"], report_id=report_id)
                n_rule_fallback += 1
            for c in claims:
                claims_out.append({
                    "claim_id": c.claim_id,
                    "image_id": image_id,
                    "rrg_model": g["model"],
                    "claim_text": c.claim_text,
                })
            n_gens += 1
            if n_gens % 50 == 0:
                print(f"[decompose] {n_gens} reports processed, {len(claims_out)} claims")

    with open(out_claims_jsonl, "w") as f:
        for c in claims_out:
            f.write(json.dumps(c) + "\n")
    payload = {"status": "ok", "n_claims": len(claims_out), "n_reports": n_gens,
               "n_rule_fallback": n_rule_fallback, "n_skipped_no_ref": n_skipped_no_ref,
               "n_references": len(refs)}
    _write_status("decompose_rrg_claims", payload)
    return payload


@app.function(
    image=image,
    gpu=_H100,
    timeout=60 * 60 * 2,
    volumes={"/data": volume},
    secrets=secrets,
)
def silver_green(
    claims_jsonl: str = "/data/v6_silver/claims.jsonl",
    references_jsonl: str = "/data/v6_silver/references.jsonl",
    out_jsonl: str = "/data/v6_silver/green_labels.jsonl",
) -> dict:
    import sys
    sys.path.insert(0, "/root/verifact")
    from pathlib import Path as P
    import json

    if _skip_if_exists(out_jsonl, "silver_green"):
        return {"status": "skipped"}

    from v5.eval.green_labeler import GreenLabeler, run_green_sweep
    references: dict[str, str] = {}
    with open(references_jsonl) as f:
        for line in f:
            r = json.loads(line)
            references[str(r["image_id"])] = str(r.get("reference_report", ""))
    claims: list[dict] = []
    with open(claims_jsonl) as f:
        for line in f:
            claims.append(json.loads(line))
    labeler = GreenLabeler(device="cuda")
    counts = run_green_sweep(labeler, claims, references, P(out_jsonl))
    payload = {"status": "ok", "counts": counts, "output": out_jsonl}
    _write_status("silver_green", payload)
    return payload


# ---------------------------------------------------------------------------
# Day 4: silver labeling (RadFact + VERT, Anthropic API only, CPU container)
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    cpu=2,
    timeout=60 * 60 * 2,
    volumes={"/data": volume},
    secrets=secrets,
)
def silver_radfact(
    claims_jsonl: str = "/data/v6_silver/claims.jsonl",
    references_jsonl: str = "/data/v6_silver/references.jsonl",
    out_jsonl: str = "/data/v6_silver/radfact_labels.jsonl",
    n_claims: int = 1000,
) -> dict:
    import sys
    sys.path.insert(0, "/root/verifact")
    from pathlib import Path as P
    import json

    if _skip_if_exists(out_jsonl, "silver_radfact"):
        return {"status": "skipped"}

    from v5.eval.radfact_labeler import RadFactLabeler, run_radfact_sweep
    references: dict[str, str] = {}
    with open(references_jsonl) as f:
        for line in f:
            r = json.loads(line)
            references[str(r["image_id"])] = str(r.get("reference_report", ""))
    claims: list[dict] = []
    with open(claims_jsonl) as f:
        for line in f:
            claims.append(json.loads(line))
            if len(claims) >= n_claims:
                break
    labeler = RadFactLabeler()
    counts = run_radfact_sweep(labeler, claims, references, P(out_jsonl))
    payload = {"status": "ok", "counts": counts, "output": out_jsonl}
    _write_status("silver_radfact", payload)
    return payload


@app.function(
    image=image,
    cpu=2,
    timeout=60 * 60 * 2,
    volumes={"/data": volume},
    secrets=secrets,
)
def silver_vert(
    claims_jsonl: str = "/data/v6_silver/claims.jsonl",
    references_jsonl: str = "/data/v6_silver/references.jsonl",
    out_jsonl: str = "/data/v6_silver/vert_labels.jsonl",
    n_claims: int = 1000,
) -> dict:
    import sys
    sys.path.insert(0, "/root/verifact")
    from pathlib import Path as P
    import json

    if _skip_if_exists(out_jsonl, "silver_vert"):
        return {"status": "skipped"}

    from v5.eval.vert_labeler import VertLabeler, run_vert_sweep
    references: dict[str, str] = {}
    with open(references_jsonl) as f:
        for line in f:
            r = json.loads(line)
            references[str(r["image_id"])] = str(r.get("reference_report", ""))
    claims: list[dict] = []
    with open(claims_jsonl) as f:
        for line in f:
            claims.append(json.loads(line))
            if len(claims) >= n_claims:
                break
    labeler = VertLabeler()
    counts = run_vert_sweep(labeler, claims, references, P(out_jsonl))
    payload = {"status": "ok", "counts": counts, "output": out_jsonl}
    _write_status("silver_vert", payload)
    return payload


@app.function(
    image=image,
    cpu=2,
    timeout=60 * 30,
    volumes={"/data": volume},
)
def silver_ensemble(
    green_jsonl: str = "/data/v6_silver/green_labels.jsonl",
    radfact_jsonl: str = "/data/v6_silver/radfact_labels.jsonl",
    vert_jsonl: str = "/data/v6_silver/vert_labels.jsonl",
    out_jsonl: str = "/data/v6_silver/ensemble.jsonl",
    out_stats: str = "/data/v6_silver/ensemble_stats.json",
) -> dict:
    import sys
    sys.path.insert(0, "/root/verifact")
    from pathlib import Path as P
    import json

    if _skip_if_exists(out_jsonl, "silver_ensemble"):
        return {"status": "skipped"}

    from v5.eval.silver_ensemble import combine_labels
    stats = combine_labels(P(green_jsonl), P(radfact_jsonl), P(vert_jsonl), P(out_jsonl))
    P(out_stats).write_text(json.dumps(stats, indent=2, default=str))
    payload = {"status": "ok", "stats": stats, "output": out_jsonl}
    _write_status("silver_ensemble", payload)
    return payload


# ---------------------------------------------------------------------------
# Day 5: PadChest-GR assembly
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    cpu=4,
    timeout=60 * 60 * 4,
    volumes={"/data": volume},
    secrets=secrets,
)
def padchest_gr_assemble(
    raw_root: str = "/data/padchest_gr_raw",
    out_jsonl: str = "/data/groundbench_v5/padchest_gr_records.jsonl",
    max_records: int | None = None,
    translate: bool = True,
) -> dict:
    import sys
    sys.path.insert(0, "/root/verifact")
    from pathlib import Path as P

    if _skip_if_exists(out_jsonl, "padchest_gr_assemble"):
        return {"status": "skipped"}

    from v5.data.padchest_gr import load_records, translate_missing_en_sentences, records_to_jsonl
    records = load_records(P(raw_root), max_records=max_records)
    if translate:
        n = translate_missing_en_sentences(records)
        print(f"translated {n} missing EN sentences via Claude Haiku")
    count = records_to_jsonl(records, P(out_jsonl))
    payload = {"status": "ok", "n_records": count, "output": out_jsonl}
    _write_status("padchest_gr_assemble", payload)
    return payload


# ---------------------------------------------------------------------------
# Day 6-7: v6.0 training + 3-way LOO
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu=_H100,
    timeout=60 * 60 * 4,
    volumes={"/data": volume},
    secrets=secrets,
)
def assemble_v6_3site(
    openi_dir: str = "/data/groundbench_v5/openi",
    chestx_dir: str = "/data/groundbench_v5/chestx_det10",
    padchest_gr_records: str = "/data/groundbench_v5/padchest_gr_records.jsonl",
    out_dir: str = "/data/groundbench_v5/all_v6",
) -> dict:
    """Concatenate per-split JSONLs from each site into v6 train/val/cal/test.

    The existing v5 site dirs ship per-split JSONLs already (train/val/cal/test),
    so we preserve those split boundaries rather than re-hash-bucketing. For
    PadChest-GR (which ships its own official 'train'/'validation'/'test'
    split field), we map: train->train, validation->val, test split into
    cal+test via patient-hash (since we need a calibration set that PadChest-GR
    doesn't provide).

    Must run BEFORE train_v6_3site; reproduce_v6.sh calls this blocking.
    """
    import sys
    sys.path.insert(0, "/root/verifact")
    from pathlib import Path as P
    import json
    import hashlib

    if _skip_if_exists(str(P(out_dir) / "groundbench_v6_train.jsonl"), "assemble_v6_3site"):
        return {"status": "skipped"}

    def _load_jsonl(p: P) -> list[dict]:
        rows: list[dict] = []
        if not p.exists():
            print(f"[assemble_v6_3site] MISSING {p} — skipping")
            return rows
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    per_site: dict[str, dict[str, int]] = {}
    split_rows: dict[str, list[dict]] = {s: [] for s in ("train", "val", "cal", "test")}

    for site_name, site_dir in (("openi", openi_dir), ("chestx_det10", chestx_dir)):
        site_path = P(site_dir)
        site_counts: dict[str, int] = {}
        for split in ("train", "val", "cal", "test"):
            jp = site_path / f"groundbench_v5_{split}.jsonl"
            rows = _load_jsonl(jp)
            for r in rows:
                r.setdefault("site", site_name)
            split_rows[split].extend(rows)
            site_counts[split] = len(rows)
        per_site[site_name] = site_counts

    # PadChest-GR: use its official split if present, else hash-bucket.
    pg_rows = _load_jsonl(P(padchest_gr_records))
    pg_counts = {"train": 0, "val": 0, "cal": 0, "test": 0, "skipped": 0}

    def _hash_bucket(pid: str) -> int:
        return int(hashlib.md5(pid.encode()).hexdigest(), 16) % 100

    for r in pg_rows:
        r.setdefault("site", "padchest_gr")
        official = str(r.get("split") or "").lower()
        if official == "train":
            dest = "train"
        elif official in ("validation", "val"):
            dest = "val"
        elif official == "test":
            pid = str(r.get("patient_id") or r.get("study_id") or r.get("image_id") or "")
            dest = "cal" if pid and _hash_bucket(pid) < 50 else "test"
        else:
            pid = str(r.get("patient_id") or r.get("study_id") or r.get("image_id") or "")
            if not pid:
                pg_counts["skipped"] += 1
                continue
            b = _hash_bucket(pid)
            dest = "train" if b < 70 else ("val" if b < 80 else ("cal" if b < 90 else "test"))
        split_rows[dest].append(r)
        pg_counts[dest] += 1
    per_site["padchest_gr"] = pg_counts

    sites_present = [k for k, v in per_site.items() if sum(v.values()) > 0]
    if len(sites_present) < 2:
        return {"status": "failed", "reason": "fewer than 2 sites present",
                "per_site": per_site}
    if len(sites_present) < 3:
        print(f"[assemble_v6_3site] WARNING: only {len(sites_present)} sites present "
              f"({sites_present}) — the 3-site headline cannot be reported")

    out = P(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    split_counts: dict[str, int] = {}
    for split, rows in split_rows.items():
        out_path = out / f"groundbench_v6_{split}.jsonl"
        with open(out_path, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        split_counts[split] = len(rows)

    payload = {"status": "ok", "split_counts": split_counts, "per_site": per_site,
               "sites_present": sites_present, "out_dir": str(out)}
    _write_status("assemble_v6_3site", payload)
    return payload


@app.function(
    image=image,
    gpu=_H100,
    timeout=60 * 60 * 4,
    volumes={"/data": volume},
    secrets=secrets,
)
def train_v6_3site(
    train_jsonl: str = "/data/groundbench_v5/all_v6/groundbench_v6_train.jsonl",
    val_jsonl: str = "/data/groundbench_v5/all_v6/groundbench_v6_val.jsonl",
    image_root: str = "/data",
    out_ckpt: str = "/data/checkpoints/claimguard_v6/v6_0_3site",
    ho_weights_path: str = "/data/groundbench_v5/ho_filter_weights.jsonl",
) -> dict:
    import sys
    sys.path.insert(0, "/root/verifact")
    from pathlib import Path as P
    import dataclasses

    from v5.train import train_v5, V5TrainConfig
    from v5.model import V5Config

    cfg = V5TrainConfig(
        train_jsonl=P(train_jsonl),
        val_jsonl=P(val_jsonl),
        out_dir=P(out_ckpt),
        image_root=P(image_root),
        use_wandb=False,
        adversarial_ho_filter=True,
        ho_filter_weights_path=P(ho_weights_path),
    )
    stats = train_v5(cfg, V5Config())
    summary = [dataclasses.asdict(s) for s in stats] if stats else []
    _write_status("train_v6_3site", {"status": "ok", "summary": summary, "ckpt": out_ckpt})
    return {"summary": summary, "ckpt": out_ckpt}


@app.function(
    image=image,
    cpu=2,
    timeout=60 * 30,
    volumes={"/data": volume},
)
def assemble_v6_loo_splits(
    v6_train: str = "/data/groundbench_v5/all_v6/groundbench_v6_train.jsonl",
    v6_val: str = "/data/groundbench_v5/all_v6/groundbench_v6_val.jsonl",
    v6_cal: str = "/data/groundbench_v5/all_v6/groundbench_v6_cal.jsonl",
    out_dir: str = "/data/groundbench_v5/all_v6_loo",
) -> dict:
    """Build held-one-site-out training + validation splits for 3-way LOO.

    Reads the already-assembled v6 train+val+cal rows, partitions by the
    ``site`` field, and for each site s writes:
        {out_dir}/train_no_{s}.jsonl + val_no_{s}.jsonl
    containing the other two sites' rows. Within each held-out config, the
    rows are bucketed 90/10 into train/val by MD5-hash on patient_id. Test
    rows from each held-out site are left alone (they come from the original
    test split and serve as the evaluation set for their respective LOO
    configuration).
    """
    import sys
    sys.path.insert(0, "/root/verifact")
    from pathlib import Path as P
    import json
    import hashlib
    from collections import defaultdict

    def _read(path: str) -> list[dict]:
        rows: list[dict] = []
        pp = P(path)
        if not pp.exists():
            return rows
        with open(pp) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    all_rows = _read(v6_train) + _read(v6_val) + _read(v6_cal)
    by_site: dict[str, list[dict]] = defaultdict(list)
    for r in all_rows:
        s = r.get("site") or r.get("source_site") or "unknown"
        by_site[s].append(r)

    per_site_input_counts = {s: len(rs) for s, rs in by_site.items()}
    present = [s for s, rs in by_site.items() if rs and s != "unknown"]
    if len(present) < 2:
        return {"status": "failed", "reason": "fewer than 2 sites present",
                "per_site_input_counts": per_site_input_counts}

    def _patient_key(r: dict) -> str:
        return str(r.get("patient_id") or r.get("study_id") or r.get("image_id") or "")

    def _bucket(pid: str) -> int:
        if not pid:
            return 0
        return int(hashlib.md5(pid.encode()).hexdigest(), 16) % 100

    out = P(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    per_site_counts: dict[str, dict] = {}
    for held_out in present:
        included = [s for s in present if s != held_out]
        merged: list[dict] = []
        for s in included:
            merged.extend(by_site[s])
        train_rows: list[dict] = []
        val_rows: list[dict] = []
        for r in merged:
            pid = _patient_key(r)
            if not pid:
                continue
            if _bucket(pid) < 90:
                train_rows.append(r)
            else:
                val_rows.append(r)
        train_path = out / f"train_no_{held_out}.jsonl"
        val_path = out / f"val_no_{held_out}.jsonl"
        with open(train_path, "w") as f:
            for r in train_rows:
                f.write(json.dumps(r) + "\n")
        with open(val_path, "w") as f:
            for r in val_rows:
                f.write(json.dumps(r) + "\n")
        per_site_counts[held_out] = {
            "included": included,
            "n_train": len(train_rows),
            "n_val": len(val_rows),
        }
        print(f"[loo] held_out={held_out} · train={len(train_rows)} · val={len(val_rows)}")

    payload = {"status": "ok", "per_site_counts": per_site_counts,
               "sites_present": present, "per_site_input_counts": per_site_input_counts,
               "out_dir": str(out)}
    _write_status("assemble_v6_loo_splits", payload)
    return payload


@app.function(
    image=image,
    gpu=_H100,
    timeout=60 * 60 * 4,
    volumes={"/data": volume},
    secrets=secrets,
)
def train_v6_loo(held_out_site: str, image_root: str = "/data",
                  ho_weights_path: str = "/data/groundbench_v5/ho_filter_weights.jsonl") -> dict:
    """Train with one of {'openi','chestx_det10','padchest_gr'} held out."""
    import sys
    sys.path.insert(0, "/root/verifact")
    from pathlib import Path as P
    import dataclasses

    from v5.train import train_v5, V5TrainConfig
    from v5.model import V5Config

    train_jsonl = P(f"/data/groundbench_v5/all_v6_loo/train_no_{held_out_site}.jsonl")
    val_jsonl = P(f"/data/groundbench_v5/all_v6_loo/val_no_{held_out_site}.jsonl")
    out_ckpt = P(f"/data/checkpoints/claimguard_v6/v6_0_loo_no_{held_out_site}")
    cfg = V5TrainConfig(
        train_jsonl=train_jsonl,
        val_jsonl=val_jsonl,
        out_dir=out_ckpt,
        image_root=P(image_root),
        use_wandb=False,
        adversarial_ho_filter=True,
        ho_filter_weights_path=P(ho_weights_path),
        epochs=2,
    )
    stats = train_v5(cfg, V5Config())
    summary = [dataclasses.asdict(s) for s in stats] if stats else []
    _write_status(f"train_v6_loo_{held_out_site}", {"status": "ok", "summary": summary,
                                                        "held_out_site": held_out_site})
    return {"summary": summary, "held_out_site": held_out_site}


# ---------------------------------------------------------------------------
# Day 8: baseline evaluation
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu=_H100,
    timeout=60 * 60 * 3,
    volumes={"/data": volume},
    secrets=secrets,
)
def baseline_eval(
    test_jsonl: str = "/data/groundbench_v5/all/groundbench_v5_test.jsonl",
    image_root: str = "/data",
    out_dir: str = "/data/v6_results/baselines",
    max_rows: int = 500,
) -> dict:
    import sys
    sys.path.insert(0, "/root/verifact")
    from pathlib import Path as P

    from v5.eval.baselines import available_baselines, run_baseline_diagnostic, _load_test_subset

    rows = _load_test_subset(P(test_jsonl), P(image_root), max_rows=max_rows)
    baselines = available_baselines(device="cuda")
    P(out_dir).mkdir(parents=True, exist_ok=True)
    results = {}
    for b in baselines:
        out_path = P(out_dir) / f"baseline_{b.name}_diagnostic.json"
        try:
            result = run_baseline_diagnostic(b, rows, P(image_root), out_path=out_path)
            results[b.name] = {"img": result.img_gap_pp, "esg": result.esg_gap_pp,
                               "acc": result.acc_full, "blind": result.evidence_blind}
        except Exception as exc:
            results[b.name] = {"error": str(exc)}
    _write_status("baseline_eval", {"status": "ok", "results": results})
    return results


# ---------------------------------------------------------------------------
# Day 9: hidden-state probe + RadFlag replication
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu=_H100_80,
    timeout=60 * 60 * 3,
    volumes={"/data": volume},
    secrets=secrets,
)
def hidden_state_probe(
    silver_jsonl: str = "/data/v6_silver/ensemble.jsonl",
    backbone: str = "maira-2",
    out_dir: str = "/data/v6_results/probe",
) -> dict:
    import sys
    sys.path.insert(0, "/root/verifact")
    from pathlib import Path as P
    import json
    import numpy as np

    from v5.eval.hidden_state_probe import (
        MAIRA2Extractor, MedVersaExtractor, extract_embeddings, train_probe, ProbeTrainConfig,
    )
    rows: list[dict] = []
    with open(silver_jsonl) as f:
        for line in f:
            r = json.loads(line)
            if r.get("final_label") not in {"SUPPORTED", "CONTRADICTED"}:
                continue
            rows.append({
                "claim_id": r["claim_id"],
                "image_path": r.get("image_path", f"{r['image_id']}.png"),
                "claim_text": r["claim_text"],
                "label": r["final_label"],
            })
    if backbone == "maira-2":
        extractor = MAIRA2Extractor(device="cuda")
    elif backbone == "medversa":
        extractor = MedVersaExtractor(device="cuda")
    else:
        raise ValueError(f"unknown backbone {backbone}")

    out_dir_p = P(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    split = int(0.8 * len(rows))
    train_rows, test_rows = rows[:split], rows[split:]
    X_tr, y_tr, _ = extract_embeddings(extractor, train_rows, P("/data"), out_dir_p / f"emb_{backbone}_train.npz")
    X_te, y_te, _ = extract_embeddings(extractor, test_rows, P("/data"), out_dir_p / f"emb_{backbone}_test.npz")
    probe, metrics = train_probe(
        X_tr, y_tr, X_te, y_te,
        cfg=ProbeTrainConfig(arch="logistic"),
        backbone_name=backbone,
        device="cuda",
    )
    import dataclasses
    metrics_d = dataclasses.asdict(metrics)
    (out_dir_p / f"probe_metrics_{backbone}.json").write_text(json.dumps(metrics_d, indent=2))
    _write_status(f"hidden_state_probe_{backbone}", {"status": "ok", "metrics": metrics_d})
    return metrics_d


# ---------------------------------------------------------------------------
# Day 10: artifact audit
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu=_H100,
    timeout=60 * 60 * 2,
    volumes={"/data": volume},
    secrets=secrets,
)
def artifact_audit(
    train_jsonl: str = "/data/groundbench_v5/all/groundbench_v5_train.jsonl",
    test_jsonl: str = "/data/groundbench_v5/all/groundbench_v5_test.jsonl",
    ho_weights_jsonl: str = "/data/groundbench_v5/ho_filter_weights.jsonl",
    out_json: str = "/data/v6_results/artifact_audit.json",
) -> dict:
    import sys
    sys.path.insert(0, "/root/verifact")
    from pathlib import Path as P
    import json
    import dataclasses

    from v5.eval.artifact_audit import train_text_only_classifier

    pre = train_text_only_classifier(P(train_jsonl), P(test_jsonl), config_name="pre_ho_filter")
    post = train_text_only_classifier(
        P(train_jsonl), P(test_jsonl),
        weights_jsonl=P(ho_weights_jsonl),
        config_name="post_ho_filter",
    )
    payload = {"pre_ho_filter": dataclasses.asdict(pre), "post_ho_filter": dataclasses.asdict(post)}
    P(out_json).parent.mkdir(parents=True, exist_ok=True)
    P(out_json).write_text(json.dumps(payload, indent=2))
    _write_status("artifact_audit", {"status": "ok", "result": payload})
    return payload


# ---------------------------------------------------------------------------
# Day 11: silver vs PadChest-GR radiologist agreement
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    cpu=2,
    timeout=60 * 30,
    volumes={"/data": volume},
)
def padchest_gr_validate(
    padchest_gr_records: str = "/data/groundbench_v5/padchest_gr_records.jsonl",
    ensemble_jsonl: str = "/data/v6_silver/ensemble.jsonl",
    out_jsonl: str = "/data/v6_results/padchest_gr_validation.jsonl",
    out_stats: str = "/data/v6_results/padchest_gr_validation_stats.json",
) -> dict:
    import sys
    sys.path.insert(0, "/root/verifact")
    from pathlib import Path as P
    import json

    from v5.eval.padchest_gr_validate import validate

    stats = validate(P(padchest_gr_records), P(ensemble_jsonl), P(out_jsonl))
    P(out_stats).write_text(json.dumps(stats, indent=2, default=str))
    _write_status("padchest_gr_validate", {"status": "ok", "stats": stats})
    return stats


# ---------------------------------------------------------------------------
# Local entrypoints (convenience)
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def print_status():
    import json
    fn = modal.Function.from_name("claimguard-v6-orchestrator", "baseline_eval")
    print(f"app deployed. Use modal.Function.from_name(...).spawn() to trigger. ")
    print(f"entrypoints: rrg_generation, silver_green, silver_radfact, silver_vert, "
          f"silver_ensemble, padchest_gr_assemble, train_v6_3site, train_v6_loo, "
          f"baseline_eval, hidden_state_probe, artifact_audit, padchest_gr_validate")
