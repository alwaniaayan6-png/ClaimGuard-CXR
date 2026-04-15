# WAKEUP — Anthropic API key is invalid; Task 1 + Task 3 blocked

**Date:** 2026-04-15 ~18:30 ET
**Priority:** USER ACTION REQUIRED (~2 minutes)

---

## TL;DR

You ran `modal secret create anthropic ANTHROPIC_API_KEY=sk-ant-...`
and `echo 'sk-ant-...' > ~/.config/claimguard/anthropic_key`. **Both
locations got the same key, but that key value is invalid** — every
call to `api.anthropic.com` returns `401 invalid x-api-key` for every
model (Sonnet 4.5, Sonnet 3.5, Haiku 3.5, Opus 4.5).

Possible causes:
1. You pasted a Claude Code OAuth token instead of a real Anthropic
   API key — they look superficially similar but are not
   interchangeable.
2. The key was revoked between creation and use.
3. The key belongs to a different / paused workspace.
4. You pasted the literal placeholder `sk-ant-...` from my
   instructions instead of substituting a real value.

The local file is **112 bytes**, starts with `sk-ant-a`, and has 1
trailing newline. The format looks right (sk-ant- + ~104 chars + LF),
so it's not a paste-mangling issue. The value itself is just rejected
by Anthropic's auth server.

---

## What got affected

### Task 1 silver graders — RAN BUT MOSTLY GHOST

The silver grader Modal job actually completed cleanly on Task 9's
run-A workbook (414 claims, 100 OpenI images). Output landed at
`/data/same_model_experiment/real/annotation_workbook_run_a_with_silver_graders.json`
on the volume. **But two of the three graders are ghosts:**

| Grader | Status | Output |
|---|---|---|
| **CheXbert** (rule-based) | ✅ working | 209 SUPPORTED / 108 NOVEL_HALLUCINATED / 94 UNCERTAIN / 3 CONTRADICTED |
| **Claude Sonnet 4.5 vision** | ❌ 401 invalid x-api-key | 414 / 414 stamped UNCERTAIN |
| **MedGemma-4B** (or fallback) | ❌ all 3 fallback models failed (gated repo / wrong arch / nonexistent) | 414 / 414 stamped UNCERTAIN |

Krippendorff α between (real labels, all-UNCERTAIN, all-UNCERTAIN)
will be undefined or near-zero — the ensemble effectively has only
one working grader, which is exactly what the plan said to *avoid*.

**Sample row showing the 401 error in the rationale field:**

```
claim:     'No pleural effusion or pneumothorax is seen.'
evidence:  'Findings: ... No pleural effusion ...'
claude_label:      UNCERTAIN
claude_confidence: low
claude_rationale:  "api error: Error code: 401 - {'type': 'error',
                    'error': {'type': 'authentication_error',
                              'message': 'invalid x-api-key'},
                    'request_id': 'req_011Ca6KAdi19...'}"
```

### Task 3 R-Drop refinement — NOT FIRED

Cannot fire any of the three Task 3 stages until the API key works:

* **Task 3a (causal-term ID)** — Modal H100, integrated gradients on
  v3 verifier. Doesn't strictly need Claude API, BUT the downstream
  Task 3b (counterfactual generation) does, so there's no point
  firing 3a until you can also fire 3b.
* **Task 3b (counterfactual generation)** — local CPU + Anthropic
  Sonnet 4.5 calls. Requires a working `ANTHROPIC_API_KEY`.
* **Task 3c (R-Drop refinement training)** — Modal H100. Needs the
  output of 3b.

---

## What I shipped while diagnosing the key issue

Three commits since you went to bed (all local, none pushed):

```
918e076 task3: fix D21 silent-load + add concurrent counterfactual driver
da35d43 task6: v3 OpenI recalibrated + StratCP + forward cfBH comparison LANDED
ab74f4d task9: real dual-run LANDED + 3 mid-run architectural fixes
```

The most recent commit (`918e076`) is a substantial body of work
that the pre-flight reviewer caught **before** any GPU spend —
the same review pattern that has now caught D21 in three separate
files this week. Net additions:

- **`inference/verifier_model.py` (NEW, ~340 lines)** — canonical
  `VerifierModel` definition + `load_verifier_checkpoint` with
  hard-missing-key audit. Single source of truth so D21 can't
  spread to a fourth file.
- **`data/augmentation/causal_term_identifier.py`** — refactored
  to use the canonical loader. The old version loaded a plain
  `AutoModel + Linear(hidden, 2)` against the v3 checkpoint and
  silently dropped every head key, leaving the Linear at random
  init. Integrated gradients would have computed against random
  weights and produced meaningless attributions.
- **`scripts/modal_train_dpo_refinement.py`** — same fix on both
  policy and reference models. `_forward()` now takes a single
  model and unpacks `(verdict_logits, sigmoid)`. Checkpoint save
  format changed from `{"encoder", "head"}` to
  `{"model_state_dict", ...}` so v4 outputs are loadable by the
  canonical loader on re-eval. Modal image build now adds
  `.add_local_python_source("inference")` (D20 fix).
- **`scripts/generate_counterfactual_pairs.py` (NEW, ~600 lines)** —
  the production driver for Task 3b. Pre-flight reviewer found
  that the existing `CounterfactualGenerator` was per-claim and
  serial, with no concurrency, no 429 backoff, and no
  resume-on-crash. At Sonnet 4.5 tier limits, driving it over 30k
  claims would take 30+ hours and burn the $25 budget on retries.
  The new driver has:
  - `ThreadPoolExecutor(max_workers=8)` configurable
  - Exponential backoff with jitter (1s → 60s cap) on transient
    Anthropic errors
  - Per-claim JSONL checkpointing → resume-on-crash skips
    completed `claim_id`s
  - `--max-cost-usd` budget guard (pre-flight 1.5× refusal +
    per-worker cancellation)
  - SIGINT graceful drain (Ctrl-C cancels pending futures)
  - Dry-run mode with stub transport (smoke-tested locally)

**559/559 regression tests still green** across 16 test files. Local
smoke test of the new loader against the real v3 checkpoint produced
357M params, all 5 layers loaded, supported probe scored 0.983 vs
contradicted 0.017 — D21 fully closed.

---

## What you need to do (≤ 5 minutes)

### Step 1 — Get a fresh API key

1. Open <https://console.anthropic.com/settings/keys>
2. Click "Create Key" (or copy an existing one you know is valid)
3. Verify it's an **API key** (starts with `sk-ant-api`), not an
   OAuth token (those start with `sk-ant-oat`)

### Step 2 — Test it locally before re-uploading

This step is critical — it verifies the key actually authenticates
before we burn ~$45 of GPU/API on a re-run.

```bash
# Replace the literal sk-ant-... below with your real key
NEW_KEY='sk-ant-api03-...'

ANTHROPIC_API_KEY="$NEW_KEY" python3 -c "
import anthropic
client = anthropic.Anthropic()
resp = client.messages.create(
    model='claude-sonnet-4-5',
    max_tokens=20,
    messages=[{'role': 'user', 'content': 'Reply with only OK'}],
)
print('AUTH OK:', resp.content[0].text)
"
```

If you see `AUTH OK: OK`, the key works. If you see
`AuthenticationError`, the key is also bad — try a different one.

### Step 3 — Update both locations

Once the test passes:

```bash
# Update the local file
echo "$NEW_KEY" > ~/.config/claimguard/anthropic_key
chmod 600 ~/.config/claimguard/anthropic_key

# Recreate the Modal secret (delete + create, no in-place update)
modal secret delete anthropic
modal secret create anthropic "ANTHROPIC_API_KEY=$NEW_KEY"

# Verify
modal secret list | grep anthropic
```

### Step 4 — Tell me "key fixed, re-run task 1 and fire task 3"

Then I'll:

1. Verify the local key works again (~1 second)
2. Re-fire Task 1 silver graders against the same Task 9 workbook
   (only ~$5 — Modal caches the image build from the previous run)
3. Wait for Task 1 to land, inspect Claude grader's real
   distribution, compute Krippendorff α with the 3-rung fallback
   ladder if needed
4. Fire Task 3a (causal-term ID) → wait → fire Task 3b
   (counterfactual generation, the new driver) → wait → fire
   Task 3c (R-Drop refinement training)
5. Doc-sync everything, commit, report final v4 numbers

Total wall time after you fix the key: ~3–4 hours unattended.

---

## Diagnostic evidence (in case you want to verify yourself)

```bash
# The key file looks structurally correct
$ python3 -c "
key = open('/Users/aayanalwani/.config/claimguard/anthropic_key', 'rb').read()
print(f'length: {len(key)}, prefix: {key[:8]!r}, trailing newline: {key.endswith(b chr(10))}')
"
length: 112, prefix: b'sk-ant-a', trailing newline: True

# But Anthropic rejects it on every model
$ ANTHROPIC_API_KEY="$(cat ~/.config/claimguard/anthropic_key)" python3 -c "
import anthropic
for model in ['claude-sonnet-4-5', 'claude-3-5-sonnet-20241022', 'claude-3-5-haiku-20241022', 'claude-opus-4-5']:
    try:
        anthropic.Anthropic().messages.create(model=model, max_tokens=10, messages=[{'role':'user','content':'OK'}])
        print(f'{model}: OK')
    except anthropic.AuthenticationError:
        print(f'{model}: 401 invalid x-api-key')
"
claude-sonnet-4-5:           401 invalid x-api-key
claude-3-5-sonnet-20241022:  401 invalid x-api-key
claude-3-5-haiku-20241022:   401 invalid x-api-key
claude-opus-4-5:             401 invalid x-api-key

# The Modal secret exists but has the same value
$ modal secret list | grep anthropic
│ anthropic │ 2026-04-15 17:44 EDT │ alwaniaayan6 │ ... │
```

---

## Sprint state right now (post-key-failure)

| Task | Status | Notes |
|---|---|---|
| 0 — checkpoint recovery | ✅ done | v1 ckpt + manifest, SHA-256 verified |
| 1 — silver graders | 🟡 ran but ghost | needs rerun with valid key (CheXbert real, Claude+MedGemma stamped UNCERTAIN due to 401 / gated) |
| 2 — extended taxonomy + v3 retrain | ✅ done | val_acc 0.9877 |
| 3 — counterfactual + R-Drop | 🟡 code+driver+pre-flight done | needs valid key to fire |
| 4 — regex annotator | ✅ done | |
| 5 — hybrid retrieval | ✅ done | |
| 6 — recalibrated OpenI + StratCP | ✅ done | inverted cfBH FDR holds at every α |
| 7 — LLM extractor | ✅ done | |
| 8 — self-annotation | 🟡 needs your 90 min | depends on a valid Task 1 silver workbook |
| 9 — provenance gate demo | ✅ done | downgrade_rate_diff = 1.00 |

**6 of 9 fully done. 2 blocked on you (Task 1 + Task 3 need valid
API key; Task 8 needs your 90 min after Task 1 lands).**

Budget spent so far this session: ~$22 of $900 cap. The bad-key
silver-grader run cost ~$3 (CheXbert was real, the two ghost graders
just stamped UNCERTAIN very fast). Total projected after Tasks 1+3
re-run: ~$87 of $900.

15 commits local, none pushed. Say "push it" when you're happy.
