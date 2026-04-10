# Data Regeneration Guide — ClaimGuard-CXR v2

This document describes how to reproduce the verifier training + evaluation data
for ClaimGuard-CXR (v2, post-bug-fixes) from scratch.

## Inputs required

- `df_chexpert_plus_240401.csv` — CheXpert Plus metadata CSV (~387 MB, 223 K reports)
- `splits_chexpert_plus/{train,calibration,test}_patients.csv` — 60/15/25 patient splits (seed=42)

## What v2 fixes (vs. v1 data)

| Bug ID | Fix |
|---|---|
| C5 | `create_eval_claims` enforces *different patient* AND *different pathology* for "Insufficient Evidence" claims (was: random index, same-patient leakage). |
| C6 | `_evidence_supports_claim_text` validator rejects contradicted pairings where the perturbed claim is still supported by its original evidence. |
| M8 | `should_swap_laterality` called before applying `laterality_swap` — no more midline-finding laterality swaps. |
| H11 | 3 new hard-negative types added: `region_swap`, `device_line_error`, `omission_as_support`. Full proposal 8-type taxonomy. |

## Reproduce v2 training data (30K claims)

```bash
python3 scripts/prepare_verifier_training_data_v2.py \
    --data-path /path/to/df_chexpert_plus_240401.csv \
    --splits-dir /path/to/splits_chexpert_plus/ \
    --output /path/to/verifier_training_data_v2.json \
    --n-claims 10000 \
    --seed 42
```

Produces: 10,000 supported + 10,000 contradicted + 10,000 insufficient = 30K claims,
with the 10,000 contradicted balanced across all 8 hard-negative types (~1,250 each,
subject to per-type claim eligibility).

## Reproduce v2 eval data (30K claims: 15K cal + 15K test)

```bash
python3 scripts/prepare_eval_data.py \
    --data-path /path/to/df_chexpert_plus_240401.csv \
    --splits-dir /path/to/splits_chexpert_plus/ \
    --output-dir /path/to/eval_data/ \
    --n-claims 5000 \
    --seed 42
```

Produces `calibration_claims.json` and `test_claims.json`, each with
5,000 supported + 5,000 contradicted + 5,000 insufficient = 15K claims.

## Upload to Modal volume

```bash
modal volume put claimguard-data /path/to/verifier_training_data_v2.json /data/verifier_training_data.json
modal volume put claimguard-data /path/to/eval_data/calibration_claims.json /data/eval_data/calibration_claims.json
modal volume put claimguard-data /path/to/eval_data/test_claims.json /data/eval_data/test_claims.json
```

## Verify determinism

With `seed=42`, re-running either script should produce byte-identical JSON
(verified via `sha256sum`). Cluster iteration order is stabilized via
`sorted(set(...))` at the generator boundary.

## Per-type distribution (approximate, varies with claim eligibility)

From test runs on CheXpert Plus calibration split:

| Negative type | Fraction of contradicted claims |
|---|---|
| negation | ~14% |
| laterality_swap | ~12% (drops for midline-dominant splits) |
| severity_swap | ~12% |
| temporal_error | ~12% |
| finding_substitution | ~12% |
| region_swap | ~13% |
| device_line_error | ~12% (drops for non-ICU-heavy splits) |
| omission_as_support | ~13% |

All types hit the balanced quota (`type_quota = n_contradicted // 8`) whenever
enough claims have the required features.
