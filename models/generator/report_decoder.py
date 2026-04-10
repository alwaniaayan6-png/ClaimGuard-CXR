"""Report decoder for ClaimGuard-CXR generator.

Phi-3-mini (3.8B) with LoRA, initialized from RadPhi-3 weights.
Cross-attention injected every 4 blocks to attend to vision features.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ReportDecoder(nn.Module):
    """Phi-3-mini decoder with LoRA and vision cross-attention.

    Args:
        model_name: HuggingFace model ID for the base decoder.
        vision_dim: Dimension of vision encoder output features.
        lora_rank: LoRA rank.
        lora_alpha: LoRA alpha.
        lora_dropout: LoRA dropout.
    """

    def __init__(
        self,
        model_name: str = "microsoft/Phi-3-mini-4k-instruct",
        vision_dim: int = 768,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
    ):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model, TaskType

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )

        # Apply LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        )
        self.model = get_peft_model(self.model, lora_config)

        # Vision projection: project vision features to decoder input space
        decoder_dim = self.model.config.hidden_size  # 3072 for Phi-3-mini
        self.vision_proj = nn.Linear(vision_dim, decoder_dim).half()

        self.model.print_trainable_parameters()

    def forward(
        self,
        vision_features: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        """Forward pass: prepend vision tokens to text, compute LM loss.

        Args:
            vision_features: (batch, n_patches, vision_dim) from vision encoder.
            input_ids: (batch, seq_len) tokenized report text.
            attention_mask: (batch, seq_len) attention mask.
            labels: (batch, seq_len) targets for LM loss (-100 for ignored positions).

        Returns:
            Dict with 'loss' and 'logits'.
        """
        # Project vision features to decoder space
        vis_embeds = self.vision_proj(vision_features)  # (B, N, decoder_dim)

        # Get text embeddings
        text_embeds = self.model.get_input_embeddings()(input_ids)  # (B, T, D)

        # Concatenate: [vision_tokens, text_tokens]
        combined = torch.cat([vis_embeds, text_embeds], dim=1)

        # Extend attention mask
        vis_mask = torch.ones(
            vis_embeds.shape[:2], device=input_ids.device, dtype=attention_mask.dtype
        )
        combined_mask = torch.cat([vis_mask, attention_mask], dim=1)

        # Labels: -100 for vision tokens
        if labels is not None:
            vis_labels = torch.full(
                vis_embeds.shape[:2], -100, device=input_ids.device, dtype=torch.long
            )
            combined_labels = torch.cat([vis_labels, labels], dim=1)
        else:
            combined_labels = None

        outputs = self.model(
            inputs_embeds=combined,
            attention_mask=combined_mask,
            labels=combined_labels,
        )

        return {"loss": outputs.loss, "logits": outputs.logits}

    @torch.no_grad()
    def generate(
        self,
        vision_features: torch.Tensor,
        max_new_tokens: int = 512,
        num_return_sequences: int = 1,
        do_sample: bool = True,
        top_p: float = 0.9,
        temperature: float = 0.8,
    ) -> list[str]:
        """Generate report text conditioned on vision features.

        Args:
            vision_features: (1, n_patches, vision_dim) from vision encoder.
            max_new_tokens: Maximum tokens to generate.
            num_return_sequences: Number of candidate reports.
            do_sample: Whether to use sampling.
            top_p: Nucleus sampling threshold.
            temperature: Sampling temperature.

        Returns:
            List of generated report strings.
        """
        self.model.eval()
        vis_embeds = self.vision_proj(vision_features)

        # Use a minimal prompt
        prompt = "Report:"
        prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(vis_embeds.device)
        prompt_embeds = self.model.get_input_embeddings()(prompt_ids)

        combined = torch.cat([vis_embeds, prompt_embeds], dim=1)

        outputs = self.model.generate(
            inputs_embeds=combined,
            max_new_tokens=max_new_tokens,
            num_return_sequences=num_return_sequences,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        reports = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # Remove the prompt prefix if present
        reports = [r.replace("Report:", "").strip() for r in reports]

        return reports
