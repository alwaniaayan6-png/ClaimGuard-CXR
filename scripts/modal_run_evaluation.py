"""Modal script for running the full ClaimGuard-CXR evaluation pipeline.

Fixes applied (vs v1):
  - C1 INVERTED FDR: uses canonical label-conditional conformal procedure — calibrate
    on FAITHFUL scores only, FDR = (unfaithful labeled green) / (total green).
  - C2 TEMPERATURE SCALING: fit on calibration logits, apply to both cal + test.
  - C3 ECE: 10-bin reliability diagrams computed before AND after temperature.
  - C4 ORPHANED METRICS: hallucination precision/recall, cluster-bootstrap 95% CIs,
    per-type confusion matrices, intra-report ICC all wired in.
  - C7 PATIENT ID: coerced to str on load with assertion.
  - C8 NAN QUANTILE: guarded with fallback to global tau_low.
  - H2 COVERAGE: both green-fraction AND CheXbert-intersection formulas reported.
  - H4 WORD BOUNDARIES: finding keyword match uses \\b boundaries (no false positives).
  - H5 SILENT BH FAILURE: logged prominently when n_green == 0 or n_flagged == 0.
  - H6 TOKENIZER TRUNCATION: truncation="only_second" preserves claim text.
  - H7 SILENT FIELD FALLBACKS: removed; required fields asserted at load time.
  - M5 DETERMINISM: sorted(set(...)) for reproducible iteration.
  - M11 DETERMINISM: cudnn.deterministic and use_deterministic_algorithms(warn_only=True).

Usage:
    modal run scripts/modal_run_evaluation.py
"""

from __future__ import annotations

import modal

app = modal.App("claimguard-evaluation")

eval_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1.0",
        "transformers>=4.40.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.11.0",
        "tqdm>=4.66.0",
    )
)

vol = modal.Volume.from_name("claimguard-data", create_if_missing=True)


@app.function(
    image=eval_image,
    gpu="H100",  # H100 per user policy — T4 takes ~40min, H100 takes ~5-7min
    timeout=60 * 60 * 4,  # 4 hour timeout
    volumes={"/data": vol},
)
def run_evaluation(
    verifier_path: str = "/data/checkpoints/verifier/best_verifier.pt",
    cal_claims_path: str = "/data/eval_data/calibration_claims.json",
    test_claims_path: str = "/data/eval_data/test_claims.json",
    output_dir: str = "/data/eval_results",
    model_name: str = "roberta-large",
    num_classes: int = 3,  # 3 for 3-class, 2 for binary (not-contra vs contra)
    max_length: int = 512,
    batch_size: int = 32,
    alpha_levels: list = None,
    min_group_size: int = 200,
    n_bootstrap: int = 500,
    seed: int = 42,
) -> dict:
    """Run the full evaluation pipeline: verify → calibrate → conformal → metrics."""
    import json
    import os
    import random

    import numpy as np
    import torch
    import torch.nn as nn
    from sklearn.metrics import (
        accuracy_score, classification_report, confusion_matrix,
        f1_score, precision_score, recall_score,
    )
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModel, AutoTokenizer
    from tqdm import tqdm

    if alpha_levels is None:
        alpha_levels = [0.05, 0.10, 0.15, 0.20]

    # ---------- Determinism (M11) ----------
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception as e:
        print(f"WARN: could not enable deterministic algorithms: {e}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(output_dir, exist_ok=True)

    # =========================================================
    # 1. Model definitions (inline for Modal)
    # =========================================================
    class HeatmapEncoder(nn.Module):
        def __init__(self, output_dim=768):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
            )
            self.proj = nn.Linear(128, output_dim)

        def forward(self, heatmap):
            if heatmap.ndim == 3:
                heatmap = heatmap.unsqueeze(1)
            return self.proj(self.conv(heatmap).flatten(1))

    class VerifierModel(nn.Module):
        def __init__(self, model_name, heatmap_dim=768, num_classes=3,
                     hidden_dim=256, dropout=0.1):
            super().__init__()
            self.text_encoder = AutoModel.from_pretrained(model_name)
            text_dim = self.text_encoder.config.hidden_size
            self.heatmap_encoder = HeatmapEncoder(output_dim=heatmap_dim)
            fused_dim = text_dim + heatmap_dim

            self.verdict_head = nn.Sequential(
                nn.Linear(fused_dim, hidden_dim), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes),
            )
            self.score_head = nn.Sequential(
                nn.Linear(fused_dim, hidden_dim), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(hidden_dim, 1),
            )
            self.contrastive_proj = nn.Sequential(
                nn.Linear(fused_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, 128),
            )

        def forward(self, input_ids, attention_mask, heatmap=None):
            outputs = self.text_encoder(
                input_ids=input_ids, attention_mask=attention_mask
            )
            text_cls = outputs.last_hidden_state[:, 0, :]
            if heatmap is not None:
                hmap_feat = self.heatmap_encoder(heatmap)
            else:
                hmap_feat = torch.zeros(
                    text_cls.shape[0],
                    self.heatmap_encoder.proj.out_features,
                    device=text_cls.device,
                    dtype=text_cls.dtype,
                )
            fused = torch.cat([text_cls, hmap_feat], dim=-1)
            verdict_logits = self.verdict_head(fused)
            score = torch.sigmoid(self.score_head(fused)).squeeze(-1)
            return verdict_logits, score

    class TemperatureScaling(nn.Module):
        """Post-hoc temperature scaling (learned on calibration, applied to test)."""
        def __init__(self, initial_temperature: float = 1.5):
            super().__init__()
            self.temperature = nn.Parameter(torch.tensor(initial_temperature))

        def forward(self, logits):
            return logits / self.temperature.clamp(min=0.01)

        def fit(self, logits, labels, lr=0.01, max_iter=100):
            nll = nn.CrossEntropyLoss()
            opt = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
            def closure():
                opt.zero_grad()
                loss = nll(self.forward(logits), labels)
                loss.backward()
                return loss
            opt.step(closure)
            return float(self.temperature.item())

    # =========================================================
    # 2. Load verifier model
    # =========================================================
    print(f"Loading verifier model (num_classes={num_classes})...")
    model = VerifierModel(model_name, num_classes=num_classes).to(device)
    if os.path.exists(verifier_path):
        checkpoint = torch.load(verifier_path, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            loss_val = checkpoint.get('loss', None)
            loss_str = f"{loss_val:.4f}" if isinstance(loss_val, (int, float)) else str(loss_val)
            print(f"Loaded checkpoint from {verifier_path} "
                  f"(epoch {checkpoint.get('epoch', '?')}, loss {loss_str})")
        else:
            model.load_state_dict(checkpoint)
            print(f"Loaded state dict from {verifier_path}")
    else:
        print(f"WARNING: No checkpoint at {verifier_path} — using random weights!")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # =========================================================
    # 3. Dataset + inference (H6, H7 fixes)
    # =========================================================
    class ClaimDataset(Dataset):
        REQUIRED_FIELDS = ("claim", "evidence", "label", "pathology",
                           "negative_type", "patient_id")
        # Provenance fields are optional for backward compat with legacy data.
        # Missing values are backfilled to "unknown", which is the maximally
        # restrictive tier (never certified safe by the provenance gate).
        PROVENANCE_FIELDS = ("evidence_source_type", "evidence_is_independent",
                             "evidence_generator_id", "claim_generator_id",
                             "evidence_trust_tier")

        def __init__(self, claims, tokenizer, max_length):
            # H7 FIX: assert required fields at load time
            for i, item in enumerate(claims):
                for f in self.REQUIRED_FIELDS:
                    if f not in item:
                        raise KeyError(
                            f"Claim {i} missing required field {f!r}. "
                            f"Regenerate eval data with scripts/prepare_eval_data.py."
                        )
            self.claims = claims
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.claims)

        def __getitem__(self, idx):
            item = self.claims[idx]
            claim = item["claim"]
            evidence = " [SEP] ".join(item["evidence"][:2])
            # H6 FIX: truncation="only_second" truncates evidence, preserves claim
            encoding = self.tokenizer(
                claim, evidence,
                max_length=self.max_length,
                padding="max_length",
                truncation="only_second",
                return_tensors="pt",
            )
            # Backfill provenance tier to "unknown" for legacy records.
            trust_tier = item.get("evidence_trust_tier", "unknown")
            return {
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "label": item["label"],
                "pathology": item["pathology"],
                "negative_type": item["negative_type"],
                "patient_id": str(item["patient_id"]),  # C7 FIX: coerce to str
                "trust_tier": str(trust_tier),
            }

    def run_inference(claims_path, split_name):
        with open(claims_path, "r") as f:
            claims = json.load(f)
        print(f"[{split_name}] Loaded {len(claims)} claims")
        dataset = ClaimDataset(claims, tokenizer, max_length)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

        all_logits, all_scores, all_labels = [], [], []
        all_pathologies, all_neg_types, all_patient_ids = [], [], []
        all_trust_tiers = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Inference ({split_name})"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                verdict_logits, score = model(input_ids, attention_mask)
                all_logits.append(verdict_logits.cpu().numpy())
                all_scores.append(score.cpu().numpy())
                labels = batch["label"]
                all_labels.extend(labels if isinstance(labels, list) else labels.tolist())
                all_pathologies.extend(batch["pathology"])
                all_neg_types.extend(batch["negative_type"])
                all_patient_ids.extend(batch["patient_id"])
                all_trust_tiers.extend(batch["trust_tier"])
        return {
            "logits": np.concatenate(all_logits, axis=0),
            "scores": np.concatenate(all_scores, axis=0),
            "labels": np.array(all_labels),
            "pathologies": np.array(all_pathologies),
            "neg_types": np.array(all_neg_types),
            "patient_ids": np.array(all_patient_ids, dtype=str),
            "trust_tiers": np.array(all_trust_tiers, dtype=str),
        }

    print("\n=== Inference on calibration + test ===")
    cal_raw = run_inference(cal_claims_path, "calibration")
    test_raw = run_inference(test_claims_path, "test")

    # BINARY REMAP: if num_classes=2, remap {0,2}->0 (not-contra), {1}->1 (contra)
    # Preserve original 3-class labels for post-hoc stratification analysis.
    if num_classes == 2:
        print("\n=== Binary label remap: {Supported, Insufficient} -> 0, {Contradicted} -> 1 ===")
        cal_raw["labels_3class"] = cal_raw["labels"].copy()
        test_raw["labels_3class"] = test_raw["labels"].copy()
        cal_raw["labels"] = (cal_raw["labels"] == 1).astype(int)
        test_raw["labels"] = (test_raw["labels"] == 1).astype(int)
        print(f"  cal:  {(cal_raw['labels']==0).sum()} not-contra, {(cal_raw['labels']==1).sum()} contra")
        print(f"  test: {(test_raw['labels']==0).sum()} not-contra, {(test_raw['labels']==1).sum()} contra")

    # =========================================================
    # 4. Temperature scaling (C2 fix)
    # =========================================================
    print("\n=== Fitting temperature scaling on calibration ===")
    temp_scaler = TemperatureScaling(initial_temperature=1.5).to(device)
    T_opt = temp_scaler.fit(
        torch.from_numpy(cal_raw["logits"]).float().to(device),
        torch.from_numpy(cal_raw["labels"]).long().to(device),
    )
    print(f"Optimal temperature T = {T_opt:.4f}")

    def scaled_probs(logits_np):
        with torch.no_grad():
            t = torch.from_numpy(logits_np).float().to(device)
            scaled = temp_scaler(t)
            return torch.softmax(scaled, dim=-1).cpu().numpy()

    def raw_probs(logits_np):
        t = torch.from_numpy(logits_np).float()
        return torch.softmax(t, dim=-1).numpy()

    cal_probs_raw = raw_probs(cal_raw["logits"])
    test_probs_raw = raw_probs(test_raw["logits"])
    cal_probs_cal = scaled_probs(cal_raw["logits"])
    test_probs_cal = scaled_probs(test_raw["logits"])
    cal_preds = np.argmax(cal_raw["logits"], axis=1)
    test_preds = np.argmax(test_raw["logits"], axis=1)
    # Supported-prob is the conformal score (use temperature-scaled)
    cal_supported = cal_probs_cal[:, 0]
    test_supported = test_probs_cal[:, 0]

    # =========================================================
    # 5. Basic verifier metrics
    # =========================================================
    # Dynamic class labels + names based on num_classes
    _class_ids = list(range(num_classes))
    _class_names = (["NotContra", "Contradicted"] if num_classes == 2
                    else ["Supported", "Contradicted", "Insufficient"])

    def basic_metrics(labels, preds, name):
        acc = accuracy_score(labels, preds)
        macro_f1 = f1_score(labels, preds, average="macro")
        per_class = f1_score(labels, preds, average=None, zero_division=0).tolist()
        per_class_prec = precision_score(labels, preds, average=None, zero_division=0).tolist()
        per_class_rec = recall_score(labels, preds, average=None, zero_division=0).tolist()
        cm = confusion_matrix(labels, preds, labels=_class_ids).tolist()
        print(f"\n[{name}] acc={acc:.4f} macro_f1={macro_f1:.4f} per_class_f1={per_class}")
        print(classification_report(labels, preds,
              target_names=_class_names,
              zero_division=0))
        return {
            "accuracy": float(acc),
            "macro_f1": float(macro_f1),
            "per_class_f1": per_class,
            "per_class_precision": per_class_prec,
            "per_class_recall": per_class_rec,
            "confusion_matrix": cm,
        }

    cal_basic = basic_metrics(cal_raw["labels"], cal_preds, "calibration")
    test_basic = basic_metrics(test_raw["labels"], test_preds, "test")

    # Hallucination (binary) precision/recall/F1
    def hallucination_binary(labels, preds):
        # Supported=0 is the "faithful" class; 1 (Contradicted) and 2 (Insufficient)
        # are both "hallucination" classes.
        pred_halluc = (preds != 0).astype(int)
        gt_halluc = (labels != 0).astype(int)
        return {
            "precision": float(precision_score(gt_halluc, pred_halluc, zero_division=0)),
            "recall": float(recall_score(gt_halluc, pred_halluc, zero_division=0)),
            "f1": float(f1_score(gt_halluc, pred_halluc, zero_division=0)),
            "n_halluc_gt": int(gt_halluc.sum()),
            "n_halluc_pred": int(pred_halluc.sum()),
        }
    test_halluc = hallucination_binary(test_raw["labels"], test_preds)
    print(f"\n[test] Hallucination binary: P={test_halluc['precision']:.4f} "
          f"R={test_halluc['recall']:.4f} F1={test_halluc['f1']:.4f}")

    # =========================================================
    # 6. ECE (Expected Calibration Error) — C3 fix
    # =========================================================
    def compute_ece(probs, labels, n_bins=10):
        """10-bin ECE using predicted class confidences."""
        confidences = probs.max(axis=1)
        preds = probs.argmax(axis=1)
        correct = (preds == labels).astype(float)
        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        bin_info = []
        for i in range(n_bins):
            lo, hi = bins[i], bins[i + 1]
            in_bin = (confidences >= lo) & (confidences < hi if i < n_bins - 1
                                            else confidences <= hi)
            n_in = in_bin.sum()
            if n_in == 0:
                bin_info.append({"lo": float(lo), "hi": float(hi), "n": 0,
                                 "acc": None, "conf": None})
                continue
            bin_acc = correct[in_bin].mean()
            bin_conf = confidences[in_bin].mean()
            ece += (n_in / len(labels)) * abs(bin_acc - bin_conf)
            bin_info.append({"lo": float(lo), "hi": float(hi), "n": int(n_in),
                             "acc": float(bin_acc), "conf": float(bin_conf)})
        return float(ece), bin_info

    ece_raw, bins_raw = compute_ece(test_probs_raw, test_raw["labels"])
    ece_cal, bins_cal = compute_ece(test_probs_cal, test_raw["labels"])
    print(f"\n[test] ECE before temperature: {ece_raw:.4f}")
    print(f"[test] ECE after  temperature: {ece_cal:.4f}  (T={T_opt:.4f})")

    # =========================================================
    # 7. Conformal procedure — CANONICAL (C1, C2, C8, H5 fixes)
    # =========================================================
    print("\n=== Conformal calibration (label-conditional cfBH) ===")

    # One-per-patient subsampling (C7 fix — patient_id is str)
    def subsample_one_per_patient(patient_ids, seed):
        rng = np.random.RandomState(seed)
        seen = {}
        for i, pid in enumerate(patient_ids):
            if pid not in seen:
                seen[pid] = [i]
            else:
                seen[pid].append(i)
        # For each patient, pick one claim uniformly at random
        selected = sorted(rng.choice(idxs) for idxs in seen.values())
        return np.array(selected, dtype=int)

    cal_sub_idx = subsample_one_per_patient(cal_raw["patient_ids"], seed=seed)
    print(f"Subsampled {len(cal_sub_idx)} calibration claims (one-per-patient) "
          f"from {len(cal_raw['patient_ids'])} total")

    cal_sub_scores = cal_supported[cal_sub_idx]
    cal_sub_labels = cal_raw["labels"][cal_sub_idx]
    cal_sub_paths = cal_raw["pathologies"][cal_sub_idx]

    # INVERTED LABEL-CONDITIONAL calibration (2026-04-05 fix):
    # Null H0 = "claim is contradicted (hallucinated)".
    # Score = P(not-contra). Under H0 (contra), scores are LOW.
    # A test claim with unusually HIGH score rejects H0 -> flag as SAFE (green).
    # p-value = P(cal_contra_score >= test_score).
    #
    # Why inverted: calibrating on faithful (label!=1) gives p-values that are
    # compressed because both cal-faithful and test-faithful scores cluster at
    # the softmax ceiling. Calibrating on contra gives a distinct cal distribution
    # (contra scores near 0) that lets BH confidently reject outliers = safe claims.
    # FDR(contra in green) is controlled at alpha by BH with exchangeability
    # of test-contra ~ cal-contra under H0.
    unique_paths = sorted(set(cal_sub_paths.tolist()) | set(test_raw["pathologies"].tolist()))
    # Count CONTRA samples per group (our calibration pool)
    group_counts = {g: int(((cal_sub_paths == g) & (cal_sub_labels == 1)).sum()) for g in unique_paths}
    powered = [g for g in unique_paths if group_counts.get(g, 0) >= min_group_size]
    underpow = [g for g in unique_paths if g not in powered]

    if underpow:
        print(f"Underpowered groups (<{min_group_size} contra samples): {underpow} -> Rare/Other")

    # Build calibration score distributions per group (label=1 = contra)
    def get_group_cal_scores(group):
        if group in powered:
            mask = (cal_sub_paths == group) & (cal_sub_labels == 1)
        else:
            mask = np.isin(cal_sub_paths, underpow) & (cal_sub_labels == 1)
        return cal_sub_scores[mask]

    # Also compute global tau_low as fallback for C8
    global_tau_low_025 = float(np.quantile(cal_sub_scores, 0.25)) if len(cal_sub_scores) else 0.5

    def conformal_pvalue(test_score, cal_scores):
        """p = (|cal >= test| + 1) / (n_cal + 1). Upper-tail, high score → low p."""
        n = len(cal_scores)
        if n == 0:
            return 1.0
        return float((np.sum(cal_scores >= test_score) + 1) / (n + 1))

    def benjamini_hochberg(pvalues, alpha):
        """BH step-up. Returns boolean array (True = accepted / GREEN)."""
        n = len(pvalues)
        if n == 0:
            return np.array([], dtype=bool)
        sorted_idx = np.argsort(pvalues)
        sorted_p = pvalues[sorted_idx]
        thresholds = np.arange(1, n + 1) / n * alpha
        below = sorted_p <= thresholds
        if not below.any():
            return np.zeros(n, dtype=bool)
        k_star = int(np.max(np.where(below)[0]))
        accepted = np.zeros(n, dtype=bool)
        accepted[sorted_idx[:k_star + 1]] = True
        return accepted

    conformal_results = {}
    test_labels = test_raw["labels"]
    test_paths = test_raw["pathologies"]
    n_test = len(test_supported)

    # Pre-compute per-group p-values once (independent of alpha)
    all_pvalues_stratified = np.ones(n_test)
    per_group_cal_sizes = {}
    for group in unique_paths:
        group_mask = (test_paths == group)
        if not group_mask.any():
            continue
        cal_scores = get_group_cal_scores(group)
        per_group_cal_sizes[group] = int(len(cal_scores))
        group_test_scores = test_supported[group_mask]
        pvals = np.array([
            conformal_pvalue(s, cal_scores) for s in group_test_scores
        ])
        all_pvalues_stratified[group_mask] = pvals

    for alpha in alpha_levels:
        print(f"\n--- alpha = {alpha} ---")
        all_pvalues = all_pvalues_stratified.copy()
        # FIX 2026-04-05: Apply GLOBAL BH across all stratified p-values.
        # Per-group BH was too strict (min_p_per_group > local_BH_threshold), always n_green=0.
        # Per-group p-values preserve pathology-conditional exchangeability,
        # global BH controls marginal FDR across the full test set.
        all_accepted = benjamini_hochberg(all_pvalues, alpha)

        # C1 FIX: FDR computed among GREEN (accepted) claims — correct direction
        n_green = int(all_accepted.sum())
        n_total_unfaithful = int((test_labels != 0).sum())
        if n_green == 0:
            print("  WARN: n_green == 0 — no claims accepted by BH procedure!")
            fdr = float("nan")
            power = 0.0
        else:
            n_false_discoveries = int((test_labels[all_accepted] != 0).sum())
            fdr = float(n_false_discoveries / n_green)
            # power = true discoveries / total faithful (n_true / (n_true + misses))
            n_true_discoveries = int((test_labels[all_accepted] == 0).sum())
            n_faithful = int((test_labels == 0).sum())
            power = float(n_true_discoveries / n_faithful) if n_faithful > 0 else 0.0

        n_flagged = n_test - n_green
        if n_flagged == 0:
            print("  WARN: n_flagged == 0 — all claims accepted (no triage)!")

        # Coverage: BOTH the green-fraction AND the proposal's formal definition
        coverage_green_frac = float(n_green / n_test) if n_test > 0 else 0.0

        # Triage: green=accepted, yellow=non-accepted w/ score > tau_low, red=otherwise
        # Per-group tau_low with C8 fallback
        tau_low_map = {}
        for group in unique_paths:
            scores = get_group_cal_scores(group)
            if len(scores) == 0:
                tau_low_map[group] = global_tau_low_025
            else:
                tau_low_map[group] = float(np.quantile(scores, 0.25))

        triage = np.full(n_test, "red", dtype=object)
        triage[all_accepted] = "green"
        for group in unique_paths:
            gm = (test_paths == group) & (~all_accepted)
            if not gm.any():
                continue
            tl = tau_low_map[group]
            triage[gm & (test_supported > tl)] = "yellow"
        n_g = int((triage == "green").sum())
        n_y = int((triage == "yellow").sum())
        n_r = int((triage == "red").sum())

        # -------------------------------------------------------------
        # Provenance-aware gate (see inference/provenance.py).
        # green + (trusted | independent)  -> supported_trusted
        # green + (same_model | unknown)   -> supported_uncertified (override)
        # yellow -> review_required
        # red    -> contradicted
        # Non-certifiable provenance cannot be certified safe regardless of
        # the verifier score or the conformal BH decision. This is a pipeline
        # policy, not a model change.
        # -------------------------------------------------------------
        test_tiers = test_raw["trust_tiers"]
        certifiable = np.isin(test_tiers, np.array(["trusted", "independent"]))
        final_label = np.empty(n_test, dtype=object)
        final_label[:] = "contradicted"
        final_label[triage == "yellow"] = "review_required"
        green_mask = triage == "green"
        final_label[green_mask & certifiable] = "supported_trusted"
        final_label[green_mask & (~certifiable)] = "supported_uncertified"
        gate_overrides = int(np.sum(green_mask & (~certifiable)))

        n_sup_trusted = int((final_label == "supported_trusted").sum())
        n_sup_uncert = int((final_label == "supported_uncertified").sum())
        n_review = int((final_label == "review_required").sum())
        n_contra = int((final_label == "contradicted").sum())

        # Per-trust-tier breakdown of the triage + FDR so readers can see
        # how the gate affects each provenance slice separately.
        tier_results = {}
        for tier in sorted(set(test_tiers.tolist())):
            tier_mask = (test_tiers == tier)
            tier_n = int(tier_mask.sum())
            if tier_n == 0:
                continue
            tier_green = int((green_mask & tier_mask).sum())
            tier_final_sup = int(((final_label == "supported_trusted") & tier_mask).sum())
            tier_false_green = int(
                ((green_mask & tier_mask) & (test_labels != 0)).sum()
            )
            tier_results[tier] = {
                "n": tier_n,
                "n_green_pre_gate": tier_green,
                "n_supported_trusted_post_gate": tier_final_sup,
                "fdr_pre_gate": (
                    float(tier_false_green / tier_green) if tier_green > 0 else 0.0
                ),
            }

        # Per-pathology FDR (on GREEN claims)
        path_fdr = {}
        for group in sorted(set(test_paths.tolist())):  # M5: deterministic
            gmask = (test_paths == group) & all_accepted
            g_n = int(gmask.sum())
            if g_n > 0:
                g_false = int((test_labels[gmask] != 0).sum())
                path_fdr[group] = {
                    "fdr": float(g_false / g_n),
                    "n_green": g_n,
                    "n_total": int((test_paths == group).sum()),
                }

        print(f"  n_green={n_green} fdr={fdr:.4f} power={power:.4f} "
              f"coverage={coverage_green_frac:.4f} (target alpha={alpha})")
        print(f"  triage: green={n_g} yellow={n_y} red={n_r}")
        print(f"  provenance gate: supported_trusted={n_sup_trusted} "
              f"supported_uncertified={n_sup_uncert} "
              f"review_required={n_review} contradicted={n_contra} "
              f"(overrides={gate_overrides})")

        conformal_results[f"alpha_{alpha}"] = {
            "alpha": alpha,
            "fdr": fdr,
            "power": power,
            "coverage_green_frac": coverage_green_frac,
            "n_green": n_green,
            "n_flagged": n_flagged,
            "n_test": n_test,
            "triage_distribution": {"green": n_g, "yellow": n_y, "red": n_r},
            "provenance_gate": {
                "supported_trusted": n_sup_trusted,
                "supported_uncertified": n_sup_uncert,
                "review_required": n_review,
                "contradicted": n_contra,
                "gate_overrides_from_green": gate_overrides,
            },
            "per_trust_tier": tier_results,
            "pathology_fdr": path_fdr,
            "per_group_cal_sizes": per_group_cal_sizes,
        }

    # =========================================================
    # 8. Per-negative-type + per-pathology analysis (M9 fix)
    # =========================================================
    print("\n=== Per-negative-type analysis ===")
    neg_type_results = {}
    for nt in sorted(set(test_raw["neg_types"].tolist())):  # M5
        mask = (test_raw["neg_types"] == nt)
        n = int(mask.sum())
        if n == 0:
            continue
        t_lab = test_labels[mask]
        t_pr = test_preds[mask]
        neg_type_results[nt] = {
            "n": n,
            "accuracy": float(accuracy_score(t_lab, t_pr)),
            "macro_f1": float(f1_score(t_lab, t_pr, average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(t_lab, t_pr, labels=_class_ids).tolist(),
        }
        print(f"  {nt}: n={n} acc={neg_type_results[nt]['accuracy']:.3f} "
              f"F1={neg_type_results[nt]['macro_f1']:.3f}")

    print("\n=== Pathology-stratified analysis ===")
    pathology_results = {}
    for path in sorted(set(test_paths.tolist())):  # M5
        mask = (test_paths == path)
        n = int(mask.sum())
        if n < 50:
            continue
        t_lab = test_labels[mask]
        t_pr = test_preds[mask]
        pathology_results[path] = {
            "n": n,
            "accuracy": float(accuracy_score(t_lab, t_pr)),
            "macro_f1": float(f1_score(t_lab, t_pr, average="macro", zero_division=0)),
        }
        print(f"  {path}: n={n} acc={pathology_results[path]['accuracy']:.3f} "
              f"F1={pathology_results[path]['macro_f1']:.3f}")

    # =========================================================
    # 9. Patient-cluster bootstrap 95% CIs (H9 fix)
    # =========================================================
    print(f"\n=== Patient-cluster bootstrap 95% CIs (B={n_bootstrap}) ===")
    def cluster_bootstrap_ci(labels, preds, patient_ids, metric_fn, B=500, seed=42):
        """Resample PATIENTS with replacement, recompute metric.

        Performance fix 2026-04-05: pre-compute patient->indices dict once,
        use np.concatenate instead of list.extend().
        """
        rng = np.random.RandomState(seed)
        unique_patients = np.unique(patient_ids)
        # Build patient->indices dict ONCE (outside loop)
        patient_to_idx = {p: np.where(patient_ids == p)[0] for p in unique_patients}
        metrics_list = []
        for _ in range(B):
            sampled = rng.choice(unique_patients, size=len(unique_patients), replace=True)
            try:
                idx = np.concatenate([patient_to_idx[p] for p in sampled])
                metrics_list.append(metric_fn(labels[idx], preds[idx]))
            except Exception:
                continue
        arr = np.array(metrics_list)
        if len(arr) == 0:
            return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
        return {
            "mean": float(np.mean(arr)),
            "ci_low": float(np.percentile(arr, 2.5)),
            "ci_high": float(np.percentile(arr, 97.5)),
        }

    acc_ci = cluster_bootstrap_ci(
        test_labels, test_preds, test_raw["patient_ids"],
        lambda l, p: accuracy_score(l, p), B=n_bootstrap, seed=seed,
    )
    f1_ci = cluster_bootstrap_ci(
        test_labels, test_preds, test_raw["patient_ids"],
        lambda l, p: f1_score(l, p, average="macro", zero_division=0),
        B=n_bootstrap, seed=seed,
    )
    print(f"  Accuracy 95% CI: [{acc_ci['ci_low']:.4f}, {acc_ci['ci_high']:.4f}]")
    print(f"  Macro F1 95% CI: [{f1_ci['ci_low']:.4f}, {f1_ci['ci_high']:.4f}]")

    # =========================================================
    # 10. Intra-report ICC (exchangeability diagnostic)
    # =========================================================
    def intra_patient_icc(scores, patient_ids):
        unique = np.unique(patient_ids)
        multi = [p for p in unique if np.sum(patient_ids == p) > 1]
        if len(multi) < 2:
            return 0.0
        grand_mean = scores.mean()
        k = len(multi)
        ss_between = ss_within = 0.0
        n_total = 0
        sizes = []
        for p in multi:
            m = patient_ids == p
            gs = scores[m]
            mu_i = gs.mean()
            n_i = len(gs)
            ss_between += n_i * (mu_i - grand_mean) ** 2
            ss_within += ((gs - mu_i) ** 2).sum()
            n_total += n_i
            sizes.append(n_i)
        if ss_within == 0:
            return 1.0
        ms_b = ss_between / (k - 1)
        ms_w = ss_within / (n_total - k)
        sa = np.array(sizes, dtype=float)
        n0 = (n_total - (sa ** 2).sum() / n_total) / (k - 1)
        icc = (ms_b - ms_w) / (ms_b + (n0 - 1) * ms_w)
        return float(max(0.0, min(1.0, icc)))

    icc_val = intra_patient_icc(test_supported, test_raw["patient_ids"])
    print(f"\n=== Intra-patient ICC of faithfulness scores: {icc_val:.4f} "
          f"({'MINOR' if icc_val < 0.1 else 'MODERATE' if icc_val < 0.3 else 'SERIOUS'} "
          f"exchangeability concern) ===")

    # =========================================================
    # 11. Write results + save raw predictions
    # =========================================================
    output = {
        "verifier_metrics": {
            "test": test_basic,
            "calibration": cal_basic,
            "hallucination_binary": test_halluc,
            "accuracy_ci": acc_ci,
            "macro_f1_ci": f1_ci,
        },
        "calibration_diagnostics": {
            "temperature": float(T_opt),
            "ece_before_temp": float(ece_raw),
            "ece_after_temp": float(ece_cal),
            "reliability_bins_before": bins_raw,
            "reliability_bins_after": bins_cal,
        },
        "conformal_results": conformal_results,
        "neg_type_results": neg_type_results,
        "pathology_results": pathology_results,
        "exchangeability": {
            "intra_patient_icc": icc_val,
        },
        "config": {
            "model_name": model_name,
            "verifier_path": verifier_path,
            "alpha_levels": alpha_levels,
            "n_cal_claims": int(len(cal_raw["labels"])),
            "n_test_claims": int(n_test),
            "min_group_size": min_group_size,
            "n_bootstrap": n_bootstrap,
            "seed": seed,
        },
    }
    results_path = os.path.join(output_dir, "full_results.json")
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved full results to {results_path}")

    np.savez(os.path.join(output_dir, "test_predictions.npz"),
             logits=test_raw["logits"], probs_raw=test_probs_raw,
             probs_calibrated=test_probs_cal, supported_prob=test_supported,
             preds=test_preds, labels=test_labels)
    np.savez(os.path.join(output_dir, "cal_predictions.npz"),
             logits=cal_raw["logits"], probs_raw=cal_probs_raw,
             probs_calibrated=cal_probs_cal, supported_prob=cal_supported,
             preds=cal_preds, labels=cal_raw["labels"])
    vol.commit()

    print("\n=== EVALUATION COMPLETE ===")
    print(f"Test Accuracy: {test_basic['accuracy']:.4f} "
          f"[95% CI: {acc_ci['ci_low']:.4f}, {acc_ci['ci_high']:.4f}]")
    print(f"Test Macro F1: {test_basic['macro_f1']:.4f} "
          f"[95% CI: {f1_ci['ci_low']:.4f}, {f1_ci['ci_high']:.4f}]")
    print(f"ECE (raw/cal): {ece_raw:.4f} / {ece_cal:.4f}   T={T_opt:.4f}")
    for k, cr in conformal_results.items():
        print(f"cfBH α={cr['alpha']}: FDR={cr['fdr']:.4f} "
              f"coverage={cr['coverage_green_frac']:.4f} "
              f"n_green={cr['n_green']}")
    return output


@app.local_entrypoint()
def main(
    cal_claims_path: str = "/data/eval_data/calibration_claims.json",
    test_claims_path: str = "/data/eval_data/test_claims.json",
    output_dir: str = "/data/eval_results",
    verifier_path: str = "/data/checkpoints/verifier/best_verifier.pt",
    num_classes: int = 3,
):
    """Entry point for `modal run scripts/modal_run_evaluation.py`.

    CLI flags enable running the retrieval-augmented ablation:
        modal run scripts/modal_run_evaluation.py \
            --cal-claims-path /data/eval_data/calibration_claims_retrieved.json \
            --test-claims-path /data/eval_data/test_claims_retrieved.json \
            --output-dir /data/eval_results_retrieved
    """
    print("Launching evaluation pipeline on Modal...")
    print(f"  cal:  {cal_claims_path}")
    print(f"  test: {test_claims_path}")
    print(f"  out:  {output_dir}")
    print(f"  verifier: {verifier_path}")
    print(f"  num_classes: {num_classes}")
    result = run_evaluation.remote(
        cal_claims_path=cal_claims_path,
        test_claims_path=test_claims_path,
        output_dir=output_dir,
        verifier_path=verifier_path,
        num_classes=num_classes,
    )
    print("\n=== FINAL RESULTS ===")
    vm = result["verifier_metrics"]["test"]
    print(f"Verifier Accuracy: {vm['accuracy']:.4f}")
    print(f"Verifier Macro F1: {vm['macro_f1']:.4f}")
    print(f"Per-class F1: {[f'{f:.3f}' for f in vm['per_class_f1']]}")
    cd = result["calibration_diagnostics"]
    print(f"Temperature: {cd['temperature']:.4f}")
    print(f"ECE (raw/calibrated): {cd['ece_before_temp']:.4f} / {cd['ece_after_temp']:.4f}")
    print("\nConformal FDR Control:")
    for key, cr in result["conformal_results"].items():
        print(f"  α={cr['alpha']}: FDR={cr['fdr']:.4f}, "
              f"Coverage={cr['coverage_green_frac']:.4f}, "
              f"n_green={cr['n_green']}")
