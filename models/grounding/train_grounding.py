"""Visual grounding module training for ClaimGuard-CXR.

Trains the cross-attention grounding module on ChestImagenome
bounding box annotations and MS-CXR phrase grounding pairs.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def prepare_grounding_data(
    chestimagenome_dir: str | Path,
    mscxr_dir: str | Path,
    output_path: str | Path,
) -> int:
    """Prepare grounding training data from annotation sources.

    Converts ChestImagenome bounding boxes and MS-CXR phrase-bounding box
    pairs into a uniform format: (image_id, claim_text, target_heatmap).

    Args:
        chestimagenome_dir: Path to ChestImagenome annotations.
        mscxr_dir: Path to MS-CXR annotations.
        output_path: Path to save prepared data.

    Returns:
        Number of training examples.
    """
    logger.info("Grounding data preparation requires ChestImagenome + MS-CXR annotations")
    logger.info("These are available via PhysioNet — download when credentialing clears")
    return 0


def train_grounding(
    training_data_path: str | Path,
    vision_encoder_path: str | Path,
    output_dir: str | Path,
    learning_rate: float = 5e-5,
    batch_size: int = 32,
    num_epochs: int = 10,
) -> dict:
    """Train the visual grounding module.

    Args:
        training_data_path: Path to prepared grounding data.
        vision_encoder_path: Path to frozen vision encoder weights.
        output_dir: Directory for checkpoints.
        learning_rate: Learning rate.
        batch_size: Batch size.
        num_epochs: Number of epochs.

    Returns:
        Dict with training metrics (IoU, pointing game accuracy).
    """
    logger.info(
        f"Grounding training: lr={learning_rate}, batch={batch_size}, "
        f"epochs={num_epochs}"
    )
    logger.info("Run via Modal for GPU access")
    return {"status": "use_modal_script"}
