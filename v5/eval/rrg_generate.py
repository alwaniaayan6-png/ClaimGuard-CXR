"""Run public-weight CXR report generators and emit JSONL.

Each generator takes a single PIL frontal chest radiograph and produces a
free-text radiology report (findings section). Outputs are cached per
(model, image_id) and written to a JSONL stream so downstream silver labelers
(``green_labeler``, ``radfact_labeler``, ``vert_labeler``) can consume them
without loading any model themselves.

The three generators implemented:

* ``CheXagentV2Generator`` — ``StanfordAIMI/CheXagent-2-3b`` (MIT license).
* ``MedGemma4BGenerator`` — ``google/medgemma-4b-it`` (HAI-DEF, gated).
* ``MAIRA2Generator`` — ``microsoft/maira-2`` (MSRLA research-only,
  ``transformers`` pinned 4.48-4.51 range, tested on 4.51.3).

Each generator exposes ``.generate(pil)`` returning a ``GenerationResult``.
A helper ``run_rrg_sweep(...)`` iterates over a manifest of test images and
writes one row per (model, image) to the output JSONL.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    model: str
    image_id: str
    generated_report: str
    generation_time_s: float
    n_tokens: int
    error: str | None = None


def _load_pil(image_path: Path, size: int = 518) -> Image.Image:
    img = Image.open(image_path).convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.BILINEAR)
    return img


class RRGGenerator:
    name: str

    def generate(self, pil: Image.Image) -> GenerationResult:
        raise NotImplementedError


def _hf_token() -> str | None:
    """Return whichever HF auth env var is set (the huggingface Modal secret
    name may expose any of these)."""
    import os
    for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN",
              "HUGGINGFACE_TOKEN"):
        if os.environ.get(k):
            return os.environ[k]
    return None


class CheXagentV2Generator(RRGGenerator):
    """CheXagent-2-3b as a findings-section generator.

    CheXagent-2-3b follows the Qwen-VL-style tokenizer-based image-input
    convention: images are passed *by path* through
    ``tokenizer.from_list_format([{'image': path}, {'text': prompt}])`` rather
    than via ``AutoProcessor``. The model does not ship a preprocessor_config,
    so ``AutoProcessor`` will raise. We therefore save the PIL to a temp file
    and call the tokenizer's custom path-based interface per the model card
    example.
    """

    name = "chexagent-2-3b"
    model_id = "StanfordAIMI/CheXagent-2-3b"
    max_new_tokens = 256

    def __init__(self, device: torch.device | str = "cuda"):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        kwargs: dict[str, Any] = {"trust_remote_code": True}
        token = _hf_token()
        if token:
            kwargs["token"] = token
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, **kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, device_map="auto", torch_dtype=torch.bfloat16, **kwargs,
        ).eval()

    @torch.no_grad()
    def generate(self, pil: Image.Image, image_id: str = "") -> GenerationResult:
        import tempfile
        t0 = time.time()
        prompt = (
            "Describe the findings of the frontal chest radiograph in a structured "
            "radiology report. Use the style: 'Lungs: ... Heart: ... Pleura: ... "
            "Bones: ... Other: ...'. Do not include an impression section."
        )
        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
            pil.save(tmp.name, format="PNG")
            query = self.tokenizer.from_list_format([
                {"image": tmp.name},
                {"text": prompt},
            ])
            conv = [
                {"from": "system", "value": "You are a helpful radiology assistant."},
                {"from": "human", "value": query},
            ]
            input_ids = self.tokenizer.apply_chat_template(
                conv, add_generation_prompt=True, return_tensors="pt"
            )
            output = self.model.generate(
                input_ids.to(self.device),
                do_sample=False,
                num_beams=1,
                temperature=1.0,
                top_p=1.0,
                use_cache=True,
                max_new_tokens=self.max_new_tokens,
            )[0]
            gen = output[input_ids.size(1):]
            text = self.tokenizer.decode(gen, skip_special_tokens=True).strip()
            if text.endswith("<|endoftext|>"):
                text = text[: -len("<|endoftext|>")].rstrip()
        return GenerationResult(
            model=self.name,
            image_id=image_id,
            generated_report=text,
            generation_time_s=time.time() - t0,
            n_tokens=int(gen.numel()),
        )


class MedGemma4BGenerator(RRGGenerator):
    """MedGemma-4B-IT as a findings-section generator (HF pipeline)."""

    name = "medgemma-4b-it"
    model_id = "google/medgemma-4b-it"
    max_new_tokens = 256

    def __init__(self, device: torch.device | str = "cuda"):
        from transformers import pipeline
        kwargs: dict[str, Any] = {"torch_dtype": torch.bfloat16}
        token = _hf_token()
        if token:
            kwargs["token"] = token
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.pipe = pipeline(
            "image-text-to-text",
            model=self.model_id,
            device=self.device,
            **kwargs,
        )

    def generate(self, pil: Image.Image, image_id: str = "") -> GenerationResult:
        t0 = time.time()
        messages = [
            {"role": "system", "content": [{"type": "text", "text": (
                "You are a radiologist. Produce a structured Findings section for the chest "
                "radiograph. Format: 'Lungs: ... Heart: ... Pleura: ... Bones: ... Other: ...'. "
                "Do not include an impression."
            )}]},
            {"role": "user", "content": [
                {"type": "image", "image": pil},
                {"type": "text", "text": "Findings:"},
            ]},
        ]
        try:
            out = self.pipe(text=messages, max_new_tokens=self.max_new_tokens, do_sample=False)
        except Exception as exc:
            return GenerationResult(
                model=self.name,
                image_id=image_id,
                generated_report="",
                generation_time_s=time.time() - t0,
                n_tokens=0,
                error=f"pipeline_failed: {exc}",
            )
        text = ""
        if out and isinstance(out, list) and out[0].get("generated_text"):
            gt = out[0]["generated_text"]
            if isinstance(gt, list) and gt:
                last = gt[-1]
                if isinstance(last, dict) and last.get("role") == "assistant":
                    content = last.get("content")
                    if isinstance(content, list) and content:
                        text = content[0].get("text", "") if isinstance(content[0], dict) else str(content[0])
                    else:
                        text = str(content or "")
            else:
                text = str(gt)
        return GenerationResult(
            model=self.name,
            image_id=image_id,
            generated_report=text.strip(),
            generation_time_s=time.time() - t0,
            n_tokens=len(text.split()),
        )


class MAIRA2Generator(RRGGenerator):
    """MAIRA-2 as a grounded findings generator (frontal-only, no prior-report).

    MAIRA-2 exposes a custom processor method
    ``format_and_preprocess_reporting_input`` for multi-field input. We run it
    in simplified findings-generation mode with only the frontal image and
    minimal context fields. Requires ``transformers`` pinned to the 4.48-4.51
    range; 4.51.3 is the tested version.
    """

    name = "maira-2"
    model_id = "microsoft/maira-2"
    max_new_tokens = 300

    def __init__(self, device: torch.device | str = "cuda"):
        from transformers import AutoProcessor, AutoModelForCausalLM
        kwargs: dict[str, Any] = {"trust_remote_code": True}
        token = _hf_token()
        if token:
            kwargs["token"] = token
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.processor = AutoProcessor.from_pretrained(self.model_id, **kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=torch.bfloat16, **kwargs,
        ).to(self.device).eval()

    @torch.no_grad()
    def generate(self, pil: Image.Image, image_id: str = "") -> GenerationResult:
        t0 = time.time()
        try:
            inputs = self.processor.format_and_preprocess_reporting_input(
                current_frontal=pil,
                current_lateral=None,
                prior_frontal=None,
                prior_report=None,
                indication="",
                technique="",
                comparison="",
                return_tensors="pt",
                get_grounding=False,
            )
        except Exception as exc:
            logger.warning("MAIRA-2 format_and_preprocess failed: %s — falling back to generic", exc)
            prompt = "Describe the findings of the frontal chest radiograph."
            inputs = self.processor(images=pil, text=prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        if "pixel_values" in inputs and hasattr(inputs["pixel_values"], "to"):
            inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)
        out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        n_input = inputs["input_ids"].shape[-1] if "input_ids" in inputs else 0
        gen = out[0, n_input:] if n_input else out[0]
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        text = tokenizer.decode(gen, skip_special_tokens=True).strip()
        return GenerationResult(
            model=self.name,
            image_id=image_id,
            generated_report=text,
            generation_time_s=time.time() - t0,
            n_tokens=int(gen.numel()) if hasattr(gen, "numel") else len(text.split()),
        )


def run_rrg_sweep(
    generators: list[RRGGenerator],
    manifest: Iterable[dict],
    image_root: Path,
    out_jsonl: Path,
    *,
    max_images: int | None = None,
    log_every: int = 25,
) -> dict[str, int]:
    """Iterate the manifest, run each generator on each image, write JSONL.

    Args:
        generators: list of loaded RRGGenerator instances.
        manifest: iterable of dicts with keys ``image_id``, ``image_path``.
        image_root: base directory for resolving relative image paths.
        out_jsonl: append-mode JSONL output path.
        max_images: if set, stops after this many images per generator.
        log_every: progress logging cadence.

    Returns:
        Dict mapping generator name -> successful-generation count.
    """
    out_jsonl = Path(out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    manifest_list = list(manifest)
    if max_images is not None:
        manifest_list = manifest_list[:max_images]
    counts: dict[str, int] = {g.name: 0 for g in generators}
    with open(out_jsonl, "a") as fh:
        for i, row in enumerate(manifest_list):
            image_id = str(row["image_id"])
            ipath = Path(row["image_path"])
            if not ipath.is_absolute():
                ipath = image_root / ipath
            try:
                pil = _load_pil(ipath)
            except Exception as exc:
                logger.warning("image load failed for %s: %s", image_id, exc)
                continue
            for gen in generators:
                try:
                    result = gen.generate(pil, image_id=image_id)
                except Exception as exc:
                    result = GenerationResult(
                        model=gen.name,
                        image_id=image_id,
                        generated_report="",
                        generation_time_s=0.0,
                        n_tokens=0,
                        error=f"generate_failed: {exc.__class__.__name__}: {exc}",
                    )
                fh.write(json.dumps(asdict(result)) + "\n")
                fh.flush()
                if result.error is None and result.generated_report:
                    counts[gen.name] += 1
            if log_every and (i + 1) % log_every == 0:
                logger.info("rrg_sweep progress %d/%d images; counts=%s", i + 1, len(manifest_list), counts)
    return counts


def build_default_generators(device: torch.device | str = "cuda") -> list[RRGGenerator]:
    """Instantiate the three canonical v6 generators, logging any failures."""
    out: list[RRGGenerator] = []
    for name, ctor in [
        ("chexagent-2-3b", lambda: CheXagentV2Generator(device=device)),
        ("medgemma-4b-it", lambda: MedGemma4BGenerator(device=device)),
        ("maira-2", lambda: MAIRA2Generator(device=device)),
    ]:
        try:
            out.append(ctor())
            logger.info("RRG generator %s loaded", name)
        except Exception as exc:
            logger.exception("RRG generator %s FAILED to load: %s", name, exc)
    return out
