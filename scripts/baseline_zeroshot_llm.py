"""Zero-shot LLM judge baseline for ClaimGuard-CXR.

Sends (claim, evidence) pairs to LLM judge via the LLM API and
asks whether the claim is CONTRADICTED by the evidence. Parses JSON
responses and computes the same metrics as the trained verifier.

Uses concurrent requests (concurrency=10) for speed.
Reads API key from ~/.api-keys/llm or LLM_API_KEY env.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
# TODO: Replace with your LLM SDK (OpenAI, LiteLLM, etc.)
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score, average_precision_score,
)

MODEL = "your-llm-model-here"
CONCURRENCY = 10

SYSTEM = ("Output only a JSON object with two keys, nothing else: "
          '{"contradicted": true|false, "confidence": 0.0..1.0}. '
          "Set contradicted=true only if the evidence directly contradicts the claim "
          "(negation, laterality swap, severity swap, temporal swap, finding mismatch, "
          "device/line mismatch, region swap). Set contradicted=false otherwise.")


def load_key() -> str:
    if os.environ.get("LLM_API_KEY"):
        return os.environ["LLM_API_KEY"]
    f = Path.home() / ".api-keys" / "llm"
    if f.exists():
        return f.read_text().strip()
    raise RuntimeError("No API key in env or ~/.api-keys/llm")


def parse_json(text: str) -> tuple[bool, float] | None:
    m = re.search(r'\{[^}]+\}', text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        contra = bool(obj.get("contradicted", False))
        conf = float(obj.get("confidence", 0.5))
        return contra, max(0.0, min(1.0, conf))
    except Exception:
        return None


async def classify(client, claim_dict, sem):
    async with sem:
        ev = claim_dict["evidence"]
        if isinstance(ev, list):
            ev = " ".join(ev)
        user = f"Claim: {claim_dict['claim']}\nEvidence: {ev}"
        for attempt in range(3):
            try:
                resp = await client.messages.create(
                    model=MODEL, max_tokens=64, system=SYSTEM,
                    messages=[{"role": "user", "content": user}],
                )
                parsed = parse_json(resp.content[0].text)
                if parsed is None:
                    await asyncio.sleep(0.2)
                    continue
                contra, conf = parsed
                return {
                    "true": 1 if claim_dict["label"] == 1 else 0,
                    "pred": 1 if contra else 0,
                    "contra_score": conf if contra else (1 - conf),
                    "err": None,
                }
            except Exception as e:
                if attempt == 2:
                    return {"true": 1 if claim_dict["label"] == 1 else 0,
                            "pred": 0, "contra_score": 0.5,
                            "err": f"{type(e).__name__}: {str(e)[:100]}"}
                await asyncio.sleep(1.0 * (attempt + 1))


async def run(claims, name):
    os.environ["LLM_API_KEY"] = load_key()
    client = None  # TODO: Instantiate your LLM client
    sem = asyncio.Semaphore(CONCURRENCY)

    print(f"=== {name} ===", flush=True)
    print(f"n={len(claims)}, model={MODEL}, concurrency={CONCURRENCY}", flush=True)

    t0 = time.time()
    tasks = [classify(client, c, sem) for c in claims]
    results = []
    done_count = 0
    for coro in asyncio.as_completed(tasks):
        r = await coro
        results.append(r)
        done_count += 1
        if done_count % 100 == 0 or done_count == len(tasks):
            dt = time.time() - t0
            rate = done_count / dt
            eta = (len(tasks) - done_count) / rate if rate > 0 else 0
            n_err = sum(1 for r in results if r["err"])
            print(f"  {done_count}/{len(tasks)} ({rate:.1f}/s, ETA {eta/60:.1f}min, errors={n_err})",
                  flush=True)

    total_time = time.time() - t0
    print(f"Completed in {total_time/60:.2f} min", flush=True)

    labels = np.array([r["true"] for r in results])
    preds = np.array([r["pred"] for r in results])
    not_contra_score = 1.0 - np.array([r["contra_score"] for r in results])
    n_err = sum(1 for r in results if r["err"])

    acc = accuracy_score(labels, preds)
    mf1 = f1_score(labels, preds, average="macro")
    per_class_f1 = f1_score(labels, preds, average=None, zero_division=0).tolist()
    prec = precision_score(labels, preds, average=None, zero_division=0).tolist()
    rec = recall_score(labels, preds, average=None, zero_division=0).tolist()
    cm = confusion_matrix(labels, preds, labels=[0, 1]).tolist()
    try:
        auroc = roc_auc_score((labels == 0).astype(int), not_contra_score)
        ap = average_precision_score((labels == 0).astype(int), not_contra_score)
    except ValueError:
        auroc, ap = float("nan"), float("nan")

    print(f"  Accuracy:      {acc:.4f}", flush=True)
    print(f"  Macro F1:      {mf1:.4f}", flush=True)
    print(f"  Contra F1:     {per_class_f1[1]:.4f}", flush=True)
    print(f"  Contra Prec:   {prec[1]:.4f}", flush=True)
    print(f"  Contra Recall: {rec[1]:.4f}", flush=True)
    print(f"  AUROC:         {auroc:.4f}", flush=True)
    print(f"  Confusion:     [[TN={cm[0][0]}, FP={cm[0][1]}], [FN={cm[1][0]}, TP={cm[1][1]}]]", flush=True)
    print(f"  Errors:        {n_err}/{len(labels)}", flush=True)

    return {
        "name": name, "model": MODEL, "n_claims": len(claims), "n_errors": n_err,
        "wall_clock_min": float(total_time/60),
        "accuracy": float(acc), "macro_f1": float(mf1),
        "f1_notcontra": float(per_class_f1[0]), "f1_contra": float(per_class_f1[1]),
        "precision_contra": float(prec[1]), "recall_contra": float(rec[1]),
        "auroc": float(auroc), "average_precision": float(ap),
        "confusion_matrix": cm,
    }


async def main():
    with open("/tmp/oracle_res/test_claims.json") as f:
        all_claims = json.load(f)
    rng = random.Random(42)
    by_label = {0: [], 1: [], 2: []}
    for c in all_claims:
        by_label[c["label"]].append(c)
    sampled = []
    for lbl in [0, 1, 2]:
        sampled.extend(rng.sample(by_label[lbl], min(500, len(by_label[lbl]))))
    rng.shuffle(sampled)
    print(f"Subsampled {len(sampled)} claims (500/class stratified)", flush=True)

    result = await run(sampled, "LLM judge (zero-shot, n=1500 stratified)")
    out_path = Path("/Users/aayanalwani/VeriFact/verifact/figures/baseline_zeroshot_llm.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps([result], indent=2))
    print(f"\nSaved: {out_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
