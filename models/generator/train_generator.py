"""Generator training script for ClaimGuard-CXR.

Trains the RadJEPA + Phi-3-mini report generator on CheXpert Plus.
This is a local/Modal-agnostic training script — the Modal wrapper
is in scripts/modal_train_generator.py.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def train_generator(config: dict) -> dict:
    """Train the report generator.

    This function is called by the Modal training script with all
    hyperparameters passed via config dict.

    Args:
        config: Training configuration dict.

    Returns:
        Dict with training metrics.
    """
    # Import here to avoid loading heavy deps at module level
    from .vision_encoder import get_vision_encoder
    from .report_decoder import ReportDecoder

    logger.info("Generator training — see scripts/modal_train_generator.py for Modal execution")
    logger.info(f"Config: {config}")

    # The actual training loop is in modal_train_generator.py
    # This module provides the model construction helpers
    return {"status": "use_modal_script"}


def build_generator(
    vision_encoder_name: str = "RadJEPA",
    decoder_model: str = "microsoft/Phi-3-mini-4k-instruct",
    image_size: int = 384,
    adapter_bottleneck: int = 32,
    lora_rank: int = 16,
) -> tuple:
    """Build the generator components.

    Returns:
        Tuple of (vision_encoder, report_decoder).
    """
    from .vision_encoder import get_vision_encoder
    from .report_decoder import ReportDecoder

    vision_encoder = get_vision_encoder(
        name=vision_encoder_name,
        image_size=image_size,
        adapter_bottleneck=adapter_bottleneck,
    )

    decoder = ReportDecoder(
        model_name=decoder_model,
        vision_dim=vision_encoder.output_dim,
        lora_rank=lora_rank,
    )

    return vision_encoder, decoder
