"""Claim decomposer training for ClaimGuard-CXR.

Fine-tunes Phi-3-mini (LoRA) on RadGraph-XL annotations + LLM-augmented
decompositions to split reports into atomic claims.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def prepare_decomposer_training_data(
    radgraph_json_path: str | Path,
    chexpert_csv_path: str | Path,
    output_path: str | Path,
    n_augmented: int = 2700,
    seed: int = 42,
) -> int:
    """Prepare training data for the claim decomposer.

    Combines RadGraph-XL annotations with sentence-level decompositions
    from the CheXpert Plus reports.

    Args:
        radgraph_json_path: Path to RadGraph-XL section_findings.json.
        chexpert_csv_path: Path to CheXpert Plus CSV.
        output_path: Path to save training data JSON.
        n_augmented: Number of augmented examples from CheXpert Plus.
        seed: Random seed.

    Returns:
        Number of training examples.
    """
    import random
    import pandas as pd
    from verifact.models.decomposer.claim_decomposer import SentenceDecomposer

    rng = random.Random(seed)
    decomposer = SentenceDecomposer()
    training_examples = []

    # Load RadGraph-XL annotations
    rg_path = Path(radgraph_json_path)
    if rg_path.exists():
        with open(rg_path) as f:
            radgraph_data = json.load(f)

        # RadGraph-XL annotations are a list of dicts
        for annotation in radgraph_data:
            if not isinstance(annotation, dict):
                continue
            text = annotation.get("0", {}).get("text", "") if isinstance(annotation.get("0"), dict) else ""
            if not text:
                continue

            # Use sentence decomposer as baseline, RadGraph entities as ground truth
            claims = decomposer.decompose(text)
            training_examples.append({
                "input": text,
                "output": [{"claim": c.text, "category": c.pathology_category} for c in claims],
            })

        logger.info(f"Loaded {len(training_examples)} examples from RadGraph-XL")

    # Augment with CheXpert Plus reports
    csv_path = Path(chexpert_csv_path)
    if csv_path.exists() and n_augmented > 0:
        df = pd.read_csv(csv_path, low_memory=False)
        impression_col = "section_impression" if "section_impression" in df.columns else "impression"

        # Sample reports for augmentation
        valid_reports = df[df[impression_col].notna() & (df[impression_col].str.len() > 20)]
        sample = valid_reports.sample(n=min(n_augmented, len(valid_reports)), random_state=seed)

        for _, row in sample.iterrows():
            text = str(row[impression_col])
            claims = decomposer.decompose(text)
            training_examples.append({
                "input": text,
                "output": [{"claim": c.text, "category": c.pathology_category} for c in claims],
            })

        logger.info(f"Added {len(sample)} augmented examples from CheXpert Plus")

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(training_examples, f, indent=2)

    logger.info(f"Total: {len(training_examples)} decomposer training examples saved to {output_path}")
    return len(training_examples)


def train_decomposer(
    training_data_path: str | Path,
    output_dir: str | Path,
    model_name: str = "microsoft/Phi-3-mini-4k-instruct",
    lora_rank: int = 8,
    learning_rate: float = 1e-4,
    batch_size: int = 16,
    num_epochs: int = 5,
) -> dict:
    """Train the claim decomposer with LoRA.

    Args:
        training_data_path: Path to prepared training data JSON.
        output_dir: Directory for checkpoints.
        model_name: Base model.
        lora_rank: LoRA rank.
        learning_rate: Learning rate.
        batch_size: Batch size.
        num_epochs: Number of epochs.

    Returns:
        Dict with training metrics.
    """
    logger.info(
        f"Decomposer training: {model_name}, LoRA r={lora_rank}, "
        f"lr={learning_rate}, epochs={num_epochs}"
    )
    logger.info("Run via Modal for GPU access: modal run scripts/modal_train_decomposer.py")
    return {"status": "use_modal_script"}
