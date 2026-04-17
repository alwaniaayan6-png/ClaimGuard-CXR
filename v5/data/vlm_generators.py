"""Orchestrate the 5 VLM generators over a pool of CXR images.

Generators:
- MAIRA-2           (microsoft/maira-2)
- CheXagent-8b      (StanfordAIMI/CheXagent-8b)
- MedGemma-4B       (google/medgemma-4b-it; gated)
- LLaVA-Rad / LLaVA-Med (microsoft/llava-med-v1.5-mistral-7b)
- Llama-3.2-11B-Vision

Each generator is wrapped with a thin adapter that accepts a PIL image + an
optional prompt and returns a generated report string + metadata (temperature,
seed, generator_id, generator_version). Modal is responsible for GPU
scheduling; this module is the adapter layer only.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol

logger = logging.getLogger(__name__)


@dataclass
class GeneratedReport:
    image_id: str
    generator_id: str
    generator_version: str
    temperature: float
    seed: int
    prompt_version: str
    report_text: str
    latency_sec: float
    raw_output: dict[str, Any] = field(default_factory=dict)


class VLMAdapter(Protocol):
    generator_id: str
    generator_version: str

    def __call__(
        self,
        image: Any,
        *,
        prompt: str,
        temperature: float,
        seed: int,
    ) -> GeneratedReport: ...


# ---------------------------------------------------------------------------
# Concrete adapters. Each lazy-loads its model on first call and caches it.
# ---------------------------------------------------------------------------


class MAIRA2Adapter:
    generator_id = "maira-2"
    generator_version = "microsoft/maira-2"

    def __init__(self) -> None:
        self._model = None
        self._processor = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        self._processor = AutoProcessor.from_pretrained(self.generator_version, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.generator_version,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )

    def __call__(
        self,
        image: Any,
        *,
        prompt: str = "Generate a radiology report for this chest X-ray.",
        temperature: float = 0.7,
        seed: int = 42,
    ) -> GeneratedReport:
        self._load()
        import torch

        torch.manual_seed(seed)
        assert self._processor is not None and self._model is not None
        start = time.time()
        inputs = self._processor(text=prompt, images=image, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs, max_new_tokens=512, do_sample=True, temperature=temperature,
            )
        text = self._processor.batch_decode(out, skip_special_tokens=True)[0]
        return GeneratedReport(
            image_id=str(getattr(image, "filename", "")),
            generator_id=self.generator_id,
            generator_version=self.generator_version,
            temperature=temperature,
            seed=seed,
            prompt_version="v5.0-default",
            report_text=text,
            latency_sec=time.time() - start,
        )


class CheXagentAdapter:
    generator_id = "chexagent-8b"
    generator_version = "StanfordAIMI/CheXagent-8b"

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._processor = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.generator_version, trust_remote_code=True)
        self._processor = AutoProcessor.from_pretrained(self.generator_version, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.generator_version,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )

    def __call__(
        self,
        image: Any,
        *,
        prompt: str = "Generate a chest X-ray report.",
        temperature: float = 0.7,
        seed: int = 42,
    ) -> GeneratedReport:
        self._load()
        import torch

        torch.manual_seed(seed)
        assert self._tokenizer is not None and self._processor is not None and self._model is not None
        # CheXagent README-validated prompt format:
        #   " USER: <s>{prompt} ASSISTANT: <s>"
        formatted = f" USER: <s>{prompt} ASSISTANT: <s>"
        start = time.time()
        inputs = self._processor(text=formatted, images=image, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-3),
                top_p=0.9,
            )
        text = self._tokenizer.decode(out[0], skip_special_tokens=True)
        # Strip echoed prompt if present
        if formatted.strip() in text:
            text = text.split(formatted.strip(), 1)[-1].strip()
        return GeneratedReport(
            image_id=str(getattr(image, "filename", "")),
            generator_id=self.generator_id,
            generator_version=self.generator_version,
            temperature=temperature,
            seed=seed,
            prompt_version="v5.0-readme",
            report_text=text,
            latency_sec=time.time() - start,
        )


class MedGemmaAdapter:
    generator_id = "medgemma-4b"
    generator_version = "google/medgemma-4b-it"

    def __init__(self) -> None:
        self._model = None
        self._processor = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        try:
            self._processor = AutoProcessor.from_pretrained(self.generator_version, trust_remote_code=True)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.generator_version,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map="auto",
            )
        except Exception as exc:
            raise RuntimeError(f"MedGemma not accessible ({exc}); fall back to MedGemma-27B or skip")

    def __call__(
        self,
        image: Any,
        *,
        prompt: str = "Write a chest X-ray report.",
        temperature: float = 0.7,
        seed: int = 42,
    ) -> GeneratedReport:
        self._load()
        import torch

        torch.manual_seed(seed)
        assert self._processor is not None and self._model is not None
        start = time.time()
        inputs = self._processor(text=prompt, images=image, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=temperature)
        text = self._processor.batch_decode(out, skip_special_tokens=True)[0]
        return GeneratedReport(
            image_id=str(getattr(image, "filename", "")),
            generator_id=self.generator_id,
            generator_version=self.generator_version,
            temperature=temperature,
            seed=seed,
            prompt_version="v5.0-default",
            report_text=text,
            latency_sec=time.time() - start,
        )


class LLaVARadAdapter:
    generator_id = "llava-rad"
    generator_version = "microsoft/llava-med-v1.5-mistral-7b"

    def __init__(self) -> None:
        self._model = None
        self._processor = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        self._processor = AutoProcessor.from_pretrained(self.generator_version, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.generator_version,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )

    def __call__(
        self,
        image: Any,
        *,
        prompt: str = "Describe the findings in this chest X-ray in report format.",
        temperature: float = 0.7,
        seed: int = 42,
    ) -> GeneratedReport:
        self._load()
        import torch

        torch.manual_seed(seed)
        assert self._processor is not None and self._model is not None
        start = time.time()
        inputs = self._processor(text=prompt, images=image, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=temperature)
        text = self._processor.batch_decode(out, skip_special_tokens=True)[0]
        return GeneratedReport(
            image_id=str(getattr(image, "filename", "")),
            generator_id=self.generator_id,
            generator_version=self.generator_version,
            temperature=temperature,
            seed=seed,
            prompt_version="v5.0-default",
            report_text=text,
            latency_sec=time.time() - start,
        )


class Llama32VisionAdapter:
    generator_id = "llama-3.2-11b-vision"
    generator_version = "meta-llama/Llama-3.2-11B-Vision-Instruct"

    def __init__(self) -> None:
        self._model = None
        self._processor = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoProcessor, MllamaForConditionalGeneration

        self._processor = AutoProcessor.from_pretrained(self.generator_version)
        self._model = MllamaForConditionalGeneration.from_pretrained(
            self.generator_version,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    def __call__(
        self,
        image: Any,
        *,
        prompt: str = "You are a radiology assistant. Write a clinical chest X-ray report based on this image.",
        temperature: float = 0.7,
        seed: int = 42,
    ) -> GeneratedReport:
        self._load()
        import torch

        torch.manual_seed(seed)
        assert self._processor is not None and self._model is not None
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        input_text = self._processor.apply_chat_template(messages, add_generation_prompt=True)
        start = time.time()
        inputs = self._processor(image, input_text, add_special_tokens=False, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=temperature)
        text = self._processor.decode(out[0], skip_special_tokens=True)
        return GeneratedReport(
            image_id=str(getattr(image, "filename", "")),
            generator_id=self.generator_id,
            generator_version=self.generator_version,
            temperature=temperature,
            seed=seed,
            prompt_version="v5.0-chat-template",
            report_text=text,
            latency_sec=time.time() - start,
        )


GENERATORS: dict[str, type] = {
    "maira-2": MAIRA2Adapter,
    "chexagent-8b": CheXagentAdapter,
    "medgemma-4b": MedGemmaAdapter,
    "llava-rad": LLaVARadAdapter,
    "llama-3.2-11b-vision": Llama32VisionAdapter,
}


def run_all_generators(
    image: Any,
    *,
    generator_names: Iterable[str] | None = None,
    temperatures: Iterable[float] = (0.7,),
    seeds: Iterable[int] = (42,),
    prompt: str | None = None,
) -> list[GeneratedReport]:
    """Run each configured generator × temperatures × seeds combinations."""
    generator_names = list(generator_names or GENERATORS.keys())
    out: list[GeneratedReport] = []
    for name in generator_names:
        cls = GENERATORS.get(name)
        if cls is None:
            logger.warning("Unknown generator %s", name)
            continue
        try:
            adapter = cls()
            for t in temperatures:
                for s in seeds:
                    try:
                        out.append(adapter(image, temperature=t, seed=int(s), **({"prompt": prompt} if prompt else {})))
                    except Exception as exc:  # noqa: BLE001
                        logger.exception("%s failed on one (t=%s, s=%s): %s", name, t, s, exc)
        except Exception as exc:  # noqa: BLE001
            logger.exception("%s load failed: %s", name, exc)
    return out
