"""Build retrieval indices for ClaimGuard-CXR.

Constructs both FAISS (dense) and BM25 (sparse) indices from
training split reports. Enforces strict data isolation — only
training patients are indexed.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def extract_passages_from_reports(
    df: pd.DataFrame,
    section: str = "section_impression",
    min_length: int = 10,
) -> tuple[list[str], list[str]]:
    """Extract individual sentences from reports as retrieval passages.

    Args:
        df: DataFrame with report text columns.
        section: Column name to extract from.
        min_length: Minimum character length for a passage.

    Returns:
        Tuple of (passages, passage_ids).
    """
    import re

    passages = []
    passage_ids = []

    for idx, row in df.iterrows():
        text = str(row.get(section, ""))
        if not text or text == "nan" or len(text) < min_length:
            continue

        patient_id = str(row.get("deid_patient_id", row.get("patient_id", idx)))

        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+|\n+', text.strip())
        for i, sent in enumerate(sentences):
            sent = sent.strip()
            if len(sent) >= min_length:
                passages.append(sent)
                passage_ids.append(f"{patient_id}_s{i}")

    logger.info(f"Extracted {len(passages)} passages from {len(df)} reports")
    return passages, passage_ids


def build_dense_index(
    passages: list[str],
    output_path: str | Path,
    model_name: str = "ncbi/MedCPT-Article-Encoder",
    batch_size: int = 64,
) -> None:
    """Build a FAISS dense retrieval index.

    Args:
        passages: List of passages to index.
        output_path: Path to save the FAISS index.
        model_name: HuggingFace model for encoding passages.
        batch_size: Encoding batch size.
    """
    from .medcpt_encoder import MedCPTRetriever

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    retriever = MedCPTRetriever()
    embeddings = retriever.encode_passages(passages, batch_size=batch_size)

    # Build FAISS index
    try:
        import faiss

        dim = embeddings.shape[1]
        nlist = min(1024, len(passages) // 10)  # IVF clusters

        if len(passages) < 1000:
            # Too few for IVF, use flat index
            index = faiss.IndexFlatIP(dim)
            index.add(embeddings.astype(np.float32))
        else:
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
            index.train(embeddings.astype(np.float32))
            index.add(embeddings.astype(np.float32))
            index.nprobe = 32

        faiss.write_index(index, str(output_path))
        logger.info(f"Saved FAISS index ({len(passages)} vectors, dim={dim}) to {output_path}")

    except ImportError:
        # Save raw embeddings if FAISS not available
        np.save(str(output_path).replace('.index', '.npy'), embeddings)
        logger.warning("FAISS not installed. Saved raw embeddings instead.")


def build_bm25_index(
    passages: list[str],
    passage_ids: list[str],
    output_path: str | Path,
) -> None:
    """Build a BM25 sparse index.

    Args:
        passages: List of passages to index.
        passage_ids: IDs for each passage.
        output_path: Path to save the pickled index.
    """
    from .bm25_index import BM25Index

    index = BM25Index(k1=1.5, b=0.75)
    index.build(passages, passage_ids)
    index.save(output_path)


def build_all_indices(
    data_path: str | Path,
    splits_dir: str | Path,
    output_dir: str | Path,
    section: str = "section_impression",
) -> dict:
    """Build all retrieval indices from training split data.

    Args:
        data_path: Path to CheXpert Plus CSV.
        splits_dir: Path to patient split files.
        output_dir: Directory to save indices.

    Returns:
        Dict with index paths and statistics.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data + training split
    df = pd.read_csv(data_path, low_memory=False)
    train_patients = pd.read_csv(
        Path(splits_dir) / "train_patients.csv"
    )["patient_id"].astype(str).tolist()

    pid_col = "deid_patient_id" if "deid_patient_id" in df.columns else "patient_id"
    df[pid_col] = df[pid_col].astype(str)
    train_df = df[df[pid_col].isin(set(train_patients))]

    logger.info(f"Building indices from {len(train_df)} training reports ({len(train_patients)} patients)")

    # Extract passages
    passages, passage_ids = extract_passages_from_reports(train_df, section=section)

    # Save passage text for later lookup
    passages_path = output_dir / "passages.json"
    with open(passages_path, "w") as f:
        json.dump({"passages": passages, "passage_ids": passage_ids}, f)

    # Build BM25
    bm25_path = output_dir / "bm25_index.pkl"
    build_bm25_index(passages, passage_ids, bm25_path)

    # Build FAISS (skip if MedCPT not available)
    faiss_path = output_dir / "faiss_index.index"
    try:
        build_dense_index(passages, faiss_path)
    except Exception as e:
        logger.warning(f"Could not build FAISS index: {e}")
        faiss_path = None

    return {
        "n_passages": len(passages),
        "n_patients": len(train_patients),
        "bm25_path": str(bm25_path),
        "faiss_path": str(faiss_path) if faiss_path else None,
        "passages_path": str(passages_path),
    }


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--splits-dir", required=True)
    parser.add_argument("--output-dir", default="./indices")
    args = parser.parse_args()

    result = build_all_indices(args.data_path, args.splits_dir, args.output_dir)
    print(f"Built indices: {result}")
