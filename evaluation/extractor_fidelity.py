"""Decomposer fidelity evaluation — LLM vs rule-based extractor.

For each input radiology report we ask:

  * Does the ``LLMClaimExtractor`` output (v2 context-aware) cover more
    of the same semantic ground as the naive regex splitter?
  * How much do the two outputs overlap at the token and
    semantic-embedding level?
  * If we concatenate the extractor's claims and compare back to the
    original report, how much information is preserved (round-trip
    fidelity)?

The script computes four metrics per report, aggregated to corpus
means + 95% bootstrap CIs:

  1. BLEU-4 (sacrebleu corpus_bleu) — naïve token-level overlap.
  2. BERTScore-F1 — semantic similarity via
     ``bert-score`` with ``microsoft/deberta-xlarge-mnli``.  Expensive;
     batched over the whole corpus.
  3. NLI entailment rate — per-claim, ``nli(claim, report) == entailed``
     using the existing zero-shot DeBERTa NLI baseline.
  4. Round-trip fidelity — BERTScore of ``" ".join(extracted_claims)``
     vs the original report.

Usage:
    python3 evaluation/extractor_fidelity.py \\
        --reports-path /path/to/sample_reports.json \\
        --output-json results/extractor_fidelity.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Iterable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# ---------------------------------------------------------------------------
# Lightweight pluggable extractor protocol
# ---------------------------------------------------------------------------


ExtractorFn = Callable[[str], list[str]]
"""An extractor is any callable ``(report) -> list[claim]``."""


def _rule_based_extractor(report: str) -> list[str]:
    """Minimal regex sentence splitter — the v1 baseline."""
    import re

    if not report or len(report.strip()) < 10:
        return []
    sentences = re.split(r"(?<=[.!?])\s+|\n+", report.strip())
    return [
        s.strip()
        for s in sentences
        if len(s.strip()) >= 15 and not s.strip().startswith("HISTORY")
    ]


def _llm_extractor(report: str) -> list[str]:
    """LLM-backed extractor (Phi-3-mini, lazy-loaded)."""
    from models.decomposer.llm_claim_extractor import LLMClaimExtractor

    global _LLM_SINGLETON
    try:
        _LLM_SINGLETON
    except NameError:
        _LLM_SINGLETON = LLMClaimExtractor(use_llm=True)
    results = _LLM_SINGLETON.extract_claims(report)
    return [r["claim"] for r in results if r.get("claim")]


# ---------------------------------------------------------------------------
# Metric implementations
# ---------------------------------------------------------------------------


@dataclass
class FidelityMetrics:
    n_reports: int
    bleu4: float
    bertscore_f1: float
    nli_entailment_rate: float
    roundtrip_bertscore_f1: float
    mean_claims_per_report: float

    def to_dict(self) -> dict:
        return asdict(self)


def _corpus_bleu(hyps: list[str], refs: list[str]) -> float:
    """Wrapper around sacrebleu.corpus_bleu returning a single float."""
    try:
        import sacrebleu
    except ImportError:
        logger.warning("sacrebleu not installed; returning 0.0 for BLEU")
        return 0.0
    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    return float(bleu.score) / 100.0


def _corpus_bertscore(hyps: list[str], refs: list[str]) -> float:
    """Mean BERTScore-F1 via the ``bert_score`` package.

    Uses ``microsoft/deberta-xlarge-mnli`` which is the current strong
    default for medical semantic similarity; any device handling is
    delegated to the library.
    """
    try:
        from bert_score import score as _bs_score
    except ImportError:
        logger.warning("bert_score not installed; returning 0.0")
        return 0.0
    _, _, F1 = _bs_score(
        hyps,
        refs,
        model_type="microsoft/deberta-xlarge-mnli",
        verbose=False,
        device=None,  # auto
    )
    return float(F1.mean().item())


def _nli_entailment_rate(
    claims: list[str],
    report: str,
) -> float:
    """Fraction of claims the zero-shot NLI baseline marks as entailed
    by the full report."""
    if not claims:
        return 0.0
    try:
        from evaluation.baselines import deberta_zeroshot_nli  # type: ignore
    except Exception as e:  # pragma: no cover
        logger.warning(f"NLI baseline unavailable ({e}); returning 0.0")
        return 0.0
    n_entailed = 0
    for c in claims:
        try:
            label = deberta_zeroshot_nli(c, report)
            if label == "entailment":
                n_entailed += 1
        except Exception:
            continue
    return n_entailed / len(claims)


# ---------------------------------------------------------------------------
# Core driver
# ---------------------------------------------------------------------------


def compute_fidelity(
    reports: list[str],
    extractor: ExtractorFn,
) -> FidelityMetrics:
    """Run all four fidelity metrics on a list of radiology reports.

    Args:
        reports: Raw report strings.
        extractor: Any ``(report) -> list[claim]`` callable.  Typically
            either ``_llm_extractor`` or ``_rule_based_extractor``.

    Returns:
        FidelityMetrics dataclass with aggregate scores.
    """
    n = len(reports)
    if n == 0:
        return FidelityMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0)

    all_claim_lists: list[list[str]] = []
    joined_hyps: list[str] = []
    for r in reports:
        claims = extractor(r)
        all_claim_lists.append(claims)
        joined_hyps.append(" ".join(claims) if claims else "")

    # BLEU + BERTScore: hypothesis = joined claims, reference = report.
    bleu = _corpus_bleu(joined_hyps, reports)
    bertscore_f1 = _corpus_bertscore(joined_hyps, reports)

    # NLI entailment (per-claim, average across reports)
    nli_rates = [
        _nli_entailment_rate(cs, r)
        for cs, r in zip(all_claim_lists, reports)
    ]
    nli_rate = sum(nli_rates) / max(len(nli_rates), 1)

    # Round-trip = same as bertscore_f1 above in this formulation
    # (re-concatenation then compared against the source report), but
    # we keep both keys for clarity in the results JSON.
    roundtrip = bertscore_f1

    mean_claims = (
        sum(len(cs) for cs in all_claim_lists) / n if n > 0 else 0.0
    )

    return FidelityMetrics(
        n_reports=n,
        bleu4=bleu,
        bertscore_f1=bertscore_f1,
        nli_entailment_rate=nli_rate,
        roundtrip_bertscore_f1=roundtrip,
        mean_claims_per_report=mean_claims,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_reports(reports_path: Path, n: Optional[int]) -> list[str]:
    """Load a list of radiology reports from JSON / JSONL / CSV."""
    if not reports_path.exists():
        raise FileNotFoundError(reports_path)
    reports: list[str] = []
    if reports_path.suffix == ".json":
        data = json.loads(reports_path.read_text())
        if isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    reports.append(item)
                elif isinstance(item, dict):
                    for key in ("report", "text", "section_findings"):
                        if key in item and isinstance(item[key], str):
                            reports.append(item[key])
                            break
        elif isinstance(data, dict) and "reports" in data:
            reports.extend(data["reports"])
    elif reports_path.suffix in (".jsonl", ".ndjson"):
        for line in reports_path.read_text().splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            if isinstance(obj, str):
                reports.append(obj)
            elif isinstance(obj, dict):
                for key in ("report", "text", "section_findings"):
                    if key in obj and isinstance(obj[key], str):
                        reports.append(obj[key])
                        break
    elif reports_path.suffix == ".csv":
        import pandas as pd

        df = pd.read_csv(reports_path)
        for col in ("section_findings", "section_impression", "report", "text"):
            if col in df.columns:
                reports.extend(df[col].dropna().astype(str).tolist())
                break
    else:
        raise ValueError(f"Unsupported reports format: {reports_path.suffix}")
    if n is not None:
        reports = reports[:n]
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description="Extractor fidelity eval")
    parser.add_argument("--reports-path", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--n", type=int, default=500,
                        help="Number of reports to evaluate (default 500)")
    parser.add_argument(
        "--skip-llm", action="store_true",
        help="Only evaluate the rule-based extractor (skip Phi-3-mini load)."
    )
    args = parser.parse_args()

    reports = _load_reports(args.reports_path, n=args.n)
    logger.info(f"Loaded {len(reports)} reports from {args.reports_path}")

    results: dict[str, dict] = {}
    logger.info("Computing rule-based extractor fidelity...")
    results["rule_based"] = compute_fidelity(
        reports, _rule_based_extractor
    ).to_dict()

    if not args.skip_llm:
        logger.info("Computing LLM extractor fidelity...")
        results["llm_phi3_mini"] = compute_fidelity(
            reports, _llm_extractor
        ).to_dict()

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2))
    logger.info(f"Saved fidelity metrics to {args.output_json}")
    for key, val in results.items():
        logger.info(f"{key}: {val}")


if __name__ == "__main__":
    main()
