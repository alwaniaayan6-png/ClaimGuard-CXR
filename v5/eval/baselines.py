"""Evidence-blindness diagnostic on external baseline verifiers.

Runs the three counterfactual conditions (image-zeroed, evidence-shuffled,
laterality-flipped) against a set of baseline systems. The baseline pool
auto-adapts to available Modal secrets and installed weights:

* Always-on (no external secrets required):
    - CheXagent-8b (local, HuggingFace)
    - LLaVA-Med-v1.5-7b (local)
    - MedVLM-v0.1 (local)
    - BiomedCLIP zero-shot (local, open_clip)
* Anthropic secret present: Claude 3.5 Sonnet via API
* OpenAI secret present: GPT-4o via API
* Google secret present: Gemini 1.5 Pro via API

Each baseline is treated as a callable verifier taking an image tensor, claim
string, evidence string, and returning a verdict in {SUPPORTED, CONTRADICTED}
(mapped to {0, 1} for comparison with ground truth). The four-condition
diagnostic is computed the same way as ``v5.eval.evidence_blindness`` so all
numbers are directly comparable.

Output: one ``baseline_<name>_diagnostic.json`` per baseline, matching the
schema of the per-config diagnostics produced by the orchestrator.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading: reuse GroundBenchDataset but only up to a subset of the test
# split to keep API cost bounded.
# ---------------------------------------------------------------------------


def _load_test_subset(
    val_jsonl: Path,
    image_root: Path,
    max_rows: int = 1000,
    seed: int = 17,
) -> list[dict]:
    """Read up to ``max_rows`` resolved-GT rows from the benchmark for baseline eval."""
    rows: list[dict] = []
    with open(val_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("gt_label") in {"SUPPORTED", "CONTRADICTED"}:
                rows.append(r)
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:max_rows]


def _load_image_tensor(path: Path, size: int = 224) -> torch.Tensor:
    """Return a (3, size, size) float tensor in [0, 1] for local-VLM pipelines."""
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    import numpy as np

    arr = np.asarray(img).astype("float32") / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)
    return t


def _image_to_png_bytes(image_tensor: torch.Tensor, *, zeroed: bool, flipped: bool) -> bytes:
    """Convert a (3,H,W) tensor to PNG bytes, optionally zeroed or h-flipped."""
    import numpy as np

    if zeroed:
        arr = np.full((image_tensor.shape[-2], image_tensor.shape[-1], 3), 128, dtype="uint8")
    else:
        if flipped:
            image_tensor = torch.flip(image_tensor, dims=[-1])
        arr = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Baseline interface
# ---------------------------------------------------------------------------


@dataclass
class BaselineResult:
    name: str
    n_test: int
    acc_full: float
    acc_image_zeroed: float
    acc_evidence_shuffled: float
    acc_laterality_flipped: float | None
    img_gap_pp: float
    esg_gap_pp: float
    ipg_gap_pp: float | None
    evidence_blind: bool
    threshold_pp: float = 5.0


class BaselineVerifier:
    """Minimal interface every baseline must implement."""

    name: str

    def __init__(self, name: str):
        self.name = name

    def predict(
        self,
        image: torch.Tensor,
        claim: str,
        evidence: str,
        *,
        zero_image: bool,
        flip_image: bool,
    ) -> int:
        """Return 0 if SUPPORTED, 1 if CONTRADICTED."""
        raise NotImplementedError

    def predict_batch(self, *args, **kwargs) -> list[int]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# API baselines (Anthropic, OpenAI, Google) — only instantiated when secrets exist
# ---------------------------------------------------------------------------


_API_SYSTEM_PROMPT = (
    "You are a radiology claim verifier. "
    "Given a chest X-ray image, a claim about the image, and supporting evidence text, "
    "return exactly one word: SUPPORTED if the image supports the claim, "
    "CONTRADICTED otherwise. No other output."
)


def _api_parse(reply: str) -> int:
    r = reply.strip().upper()
    if r.startswith("SUPPORTED"):
        return 0
    if r.startswith("CONTRADICTED"):
        return 1
    # fallback: any "yes"/"true"/"support" → 0, else → 1
    return 0 if any(k in r for k in ("YES", "TRUE", "SUPPORT", "AGREE")) else 1


class ClaudeBaseline(BaselineVerifier):
    def __init__(self):
        super().__init__("claude-3-5-sonnet")
        import anthropic
        self.client = anthropic.Anthropic()
        self.model_id = "claude-3-5-sonnet-20241022"

    def predict(self, image, claim, evidence, *, zero_image, flip_image):
        # Anthropic API rate-limits 50 rpm on Opus / 60 rpm on Sonnet 3.5.
        # Add explicit 429 retry-with-backoff so a single rate-limit error
        # doesn't zero out the whole baseline run.
        import anthropic as _ah
        png = _image_to_png_bytes(image, zeroed=zero_image, flipped=flip_image)
        b64 = base64.b64encode(png).decode("ascii")
        for attempt in range(5):
            try:
                msg = self.client.messages.create(
                    model=self.model_id,
                    max_tokens=10,
                    system=_API_SYSTEM_PROMPT,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "image", "source": {"type": "base64",
                                                          "media_type": "image/png", "data": b64}},
                            {"type": "text", "text": f"Claim: {claim}\nEvidence: {evidence}\nVerdict:"},
                        ],
                    }],
                )
                return _api_parse(msg.content[0].text if msg.content else "")
            except _ah.RateLimitError:
                time.sleep(2.0 * (2 ** attempt))
            except _ah.APIStatusError as exc:
                if getattr(exc, "status_code", None) in (429, 529):
                    time.sleep(2.0 * (2 ** attempt))
                    continue
                raise
        raise RuntimeError("Claude baseline: 5 rate-limit retries exhausted")


class OpenAIBaseline(BaselineVerifier):
    def __init__(self):
        super().__init__("gpt-4o")
        from openai import OpenAI
        self.client = OpenAI()
        self.model_id = "gpt-4o"

    def predict(self, image, claim, evidence, *, zero_image, flip_image):
        png = _image_to_png_bytes(image, zeroed=zero_image, flipped=flip_image)
        b64 = base64.b64encode(png).decode("ascii")
        resp = self.client.chat.completions.create(
            model=self.model_id,
            max_tokens=10,
            messages=[
                {"role": "system", "content": _API_SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": f"Claim: {claim}\nEvidence: {evidence}\nVerdict:"},
                ]},
            ],
        )
        return _api_parse(resp.choices[0].message.content or "")


class GeminiBaseline(BaselineVerifier):
    def __init__(self):
        super().__init__("gemini-1-5-pro")
        import google.generativeai as genai
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        self.model = genai.GenerativeModel(
            "gemini-1.5-pro",
            system_instruction=_API_SYSTEM_PROMPT,
        )

    def predict(self, image, claim, evidence, *, zero_image, flip_image):
        png = _image_to_png_bytes(image, zeroed=zero_image, flipped=flip_image)
        pil = Image.open(io.BytesIO(png))
        resp = self.model.generate_content([pil, f"Claim: {claim}\nEvidence: {evidence}\nVerdict:"])
        return _api_parse(resp.text or "")


# ---------------------------------------------------------------------------
# Local open-weight baselines
# ---------------------------------------------------------------------------


def _tensor_to_pil(image: torch.Tensor, *, zero_image: bool, flip_image: bool) -> "Image.Image":
    import numpy as np
    if zero_image:
        arr = np.full((image.shape[-2], image.shape[-1], 3), 128, dtype="uint8")
        return Image.fromarray(arr)
    img_t = torch.flip(image, dims=[-1]) if flip_image else image
    arr = (img_t.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype("uint8")
    return Image.fromarray(arr)


_BIOMEDCLIP_FINDINGS = [
    "atelectasis", "cardiomegaly", "consolidation", "edema",
    "enlarged cardiomediastinum", "fracture", "lung lesion", "lung opacity",
    "pleural effusion", "pleural other", "pneumonia", "pneumothorax",
    "support devices", "no finding",
]


class BiomedCLIPZeroShotBaseline(BaselineVerifier):
    """Zero-shot claim verification via polarity-aware CheXzero-style caption matching.

    The prior implementation compared the image to "This image supports/contradicts the claim"
    templates, which are not informative captions for a CLIP model — hence the 41% accuracy
    (below random). This implementation parses the claim to extract a finding and polarity,
    then compares the image to "{finding}" vs "no {finding}" captions — the standard CheXzero
    protocol. For unparseable claims, falls back to a finding-list classification over the
    14 CheXpert categories and picks the best match via substring overlap.
    """

    def __init__(self, device: torch.device | str = "cuda"):
        super().__init__("biomedclip-zero-shot")
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        )
        tokenizer = open_clip.get_tokenizer(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        )
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.model = model.to(self.device).eval()
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self._caption_cache: dict[str, torch.Tensor] = {}

    def _extract_finding_polarity(self, claim: str) -> tuple[str, bool]:
        """Return (finding, is_positive). Positive means the claim asserts presence."""
        try:
            from v5.data.claim_parser import parse_claim
            parsed = parse_claim(claim)
            finding = (parsed.finding or "").lower().strip() if parsed else ""
            polarity = parsed.polarity if parsed else "positive"
            is_positive = polarity != "negative"
        except Exception:
            finding = ""
            is_positive = True
        if not finding:
            lc = claim.lower()
            for f in _BIOMEDCLIP_FINDINGS:
                if f != "no finding" and f in lc:
                    finding = f
                    break
            if not finding:
                finding = "abnormality"
            is_positive = not any(neg in lc for neg in ("no ", "without ", "absence", "rule out"))
        return finding, is_positive

    @torch.no_grad()
    def _encode_caption(self, text: str) -> torch.Tensor:
        if text in self._caption_cache:
            return self._caption_cache[text]
        tok = self.tokenizer([text]).to(self.device)
        feat = F.normalize(self.model.encode_text(tok), dim=-1)
        self._caption_cache[text] = feat
        return feat

    @torch.no_grad()
    def predict(self, image, claim, evidence, *, zero_image, flip_image):
        pil = _tensor_to_pil(image, zero_image=zero_image, flip_image=flip_image)
        inp = self.preprocess(pil).unsqueeze(0).to(self.device)
        img_feat = F.normalize(self.model.encode_image(inp), dim=-1)
        finding, is_positive = self._extract_finding_polarity(claim)
        cap_present = self._encode_caption(f"radiograph showing {finding}")
        cap_absent = self._encode_caption(f"radiograph with no evidence of {finding}")
        sim_present = float((img_feat @ cap_present.T).squeeze())
        sim_absent = float((img_feat @ cap_absent.T).squeeze())
        image_says_present = sim_present >= sim_absent
        if is_positive:
            return 0 if image_says_present else 1
        return 1 if image_says_present else 0


def _baselines_hf_token() -> str | None:
    for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN",
              "HUGGINGFACE_TOKEN"):
        if os.environ.get(k):
            return os.environ[k]
    return None


class CheXagentV2Baseline(BaselineVerifier):
    """CheXagent-2-3b (Stanford AIMI, MIT) as a zero-shot claim verifier.

    Uses the Qwen-VL-style tokenizer-based image-input interface per the
    CheXagent-2-3b model card — ``tokenizer.from_list_format`` + chat
    template; the repo does not ship a preprocessor_config, so AutoProcessor
    cannot be used.
    """

    model_id = "StanfordAIMI/CheXagent-2-3b"
    max_new_tokens = 8

    def __init__(self, device: torch.device | str = "cuda"):
        super().__init__("chexagent-2-3b")
        from transformers import AutoTokenizer, AutoModelForCausalLM
        kwargs: dict[str, Any] = {"trust_remote_code": True}
        token = _baselines_hf_token()
        if token:
            kwargs["token"] = token
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, **kwargs)
        # Default dtype (FP32). See CheXagentV2Generator for rationale: the
        # remote-code image loader emits FP32 tensors regardless of model
        # dtype, so explicit bfloat16 casting produces a type-mismatch at
        # inference.
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, device_map="auto", **kwargs,
        ).eval()

    @torch.no_grad()
    def predict(self, image, claim, evidence, *, zero_image, flip_image):
        import tempfile
        pil = _tensor_to_pil(image, zero_image=zero_image, flip_image=flip_image)
        prompt_text = (
            "Given the chest radiograph, a claim, and supporting evidence, "
            "output exactly one word: SUPPORTED if the claim is consistent with the image, "
            "CONTRADICTED otherwise.\n"
            f"Claim: {claim}\nEvidence: {evidence}\nVerdict:"
        )
        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
            pil.save(tmp.name, format="PNG")
            query = self.tokenizer.from_list_format([
                {"image": tmp.name},
                {"text": prompt_text},
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
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
            )[0]
            gen = output[input_ids.size(1):]
            text = self.tokenizer.decode(gen, skip_special_tokens=True)
        return _api_parse(text[:40])


class MedGemma4BBaseline(BaselineVerifier):
    """MedGemma-4B-IT (Google, HAI-DEF) as a zero-shot claim verifier.

    Modern HF image-text-to-text pipeline; SigLIP image encoder + Gemma-3 LM.
    Single-image input only.
    """

    model_id = "google/medgemma-4b-it"
    max_new_tokens = 8

    def __init__(self, device: torch.device | str = "cuda"):
        super().__init__("medgemma-4b-it")
        from transformers import pipeline
        kwargs: dict[str, Any] = {"torch_dtype": torch.bfloat16}
        token = _baselines_hf_token()
        if token:
            kwargs["token"] = token
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.pipe = pipeline(
            "image-text-to-text",
            model=self.model_id,
            device=self.device,
            **kwargs,
        )

    def predict(self, image, claim, evidence, *, zero_image, flip_image):
        pil = _tensor_to_pil(image, zero_image=zero_image, flip_image=flip_image)
        messages = [
            {"role": "system", "content": [{"type": "text", "text": _API_SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "image", "image": pil},
                {"type": "text", "text": f"Claim: {claim}\nEvidence: {evidence}\nVerdict:"},
            ]},
        ]
        out = self.pipe(text=messages, max_new_tokens=self.max_new_tokens, do_sample=False)
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
        return _api_parse(text[:40])


class MAIRA2Baseline(BaselineVerifier):
    """MAIRA-2 (Microsoft, MSRLA research-only) as a zero-shot claim verifier.

    MAIRA-2 is trained for grounded report generation, not verification. We use it
    in findings-generation mode with a frontal-only input (skipping prior-report
    conditioning and lateral view for simplicity), then parse its textual output for
    a SUPPORTED / CONTRADICTED verdict. Requires transformers pinned to 4.51.3 per
    Microsoft's recommended range (4.48–4.51).
    """

    model_id = "microsoft/maira-2"
    max_new_tokens = 32  # short SUPPORTED/CONTRADICTED + a couple words; 8 was truncating

    def __init__(self, device: torch.device | str = "cuda"):
        super().__init__("maira-2")
        from transformers import AutoProcessor, AutoModelForCausalLM
        kwargs: dict[str, Any] = {"trust_remote_code": True}
        token = _baselines_hf_token()
        if token:
            kwargs["token"] = token
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.processor = AutoProcessor.from_pretrained(self.model_id, **kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=torch.bfloat16, **kwargs,
        ).to(self.device).eval()

    @torch.no_grad()
    def predict(self, image, claim, evidence, *, zero_image, flip_image):
        pil = _tensor_to_pil(image, zero_image=zero_image, flip_image=flip_image)
        prompt_text = (
            "You are a radiology claim verifier. Given the frontal chest radiograph, "
            "a claim, and supporting evidence, output exactly one word: SUPPORTED if the "
            "claim is consistent with the image, CONTRADICTED otherwise.\n"
            f"Claim: {claim}\nEvidence: {evidence}\nVerdict:"
        )
        # MAIRA-2's generic processor(images=, text=) interface is not exposed
        # by the trust_remote_code processor — it requires the custom
        # format_and_preprocess_reporting_input call. Use that with claim+evidence
        # injected as the indication field, and parse the generated findings text
        # for SUPPORTED/CONTRADICTED.
        try:
            inputs = self.processor.format_and_preprocess_reporting_input(
                current_frontal=pil,
                current_lateral=None,
                prior_frontal=None,
                prior_report=None,
                indication=prompt_text,
                technique="",
                comparison="",
                return_tensors="pt",
                get_grounding=False,
            )
        except Exception:
            # last-ditch fallback for processor variants
            inputs = self.processor(images=pil, text=prompt_text, return_tensors="pt")
        inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        if "pixel_values" in inputs and hasattr(inputs["pixel_values"], "to"):
            inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)
        out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        n_input = inputs["input_ids"].shape[-1] if "input_ids" in inputs else 0
        gen = out[0, n_input:] if n_input else out[0]
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        text = tokenizer.decode(gen, skip_special_tokens=True)
        return _api_parse(text[:80])


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _shuffle_evidence(rows: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    shuffled: list[dict] = []
    for i in range(len(rows)):
        j = idx[i] if idx[i] != i else idx[(i + 1) % len(rows)]
        copy = dict(rows[i])
        copy["evidence_text"] = rows[j].get("evidence_text") or ""
        shuffled.append(copy)
    return shuffled


def _is_laterality_claim(text: str) -> bool:
    lc = text.lower()
    return any(t in lc for t in (" left ", " right ", "bilateral", "left-", "right-"))


def run_baseline_diagnostic(
    baseline: BaselineVerifier,
    rows: list[dict],
    image_root: Path,
    *,
    n_shuffle_seeds: int = 1,
    out_path: Path | None = None,
    threshold_pp: float = 5.0,
    progress_every: int = 50,
) -> BaselineResult:
    """Run the four-condition diagnostic on a single baseline.

    Args:
        baseline: instance of a BaselineVerifier subclass.
        rows: list of GroundBench rows with image_path, claim_text, evidence_text, gt_label.
        image_root: root for resolving relative image paths.
        n_shuffle_seeds: ESG is averaged over this many random derangements (API cost scales).
        out_path: if provided, write the result as JSON here.
        threshold_pp: IMG/ESG threshold for the evidence-blind classification.

    Returns: BaselineResult summary.
    """
    name = baseline.name
    total = len(rows)
    logger.info("baseline=%s running diagnostic on %d rows", name, total)

    def _score(transform_rows: list[dict], *, zero: bool = False, flip: bool = False) -> float:
        correct = 0
        for i, row in enumerate(transform_rows):
            image = _load_image_tensor(Path(row["image_path"]))
            claim = row["claim_text"]
            evid = row.get("evidence_text") or ""
            y = 1 if row["gt_label"] == "CONTRADICTED" else 0
            try:
                pred = baseline.predict(image, claim, evid, zero_image=zero, flip_image=flip)
            except Exception as exc:
                logger.warning("baseline %s predict failed at row %d: %s", name, i, exc)
                continue
            if pred == y:
                correct += 1
            if progress_every and (i + 1) % progress_every == 0:
                logger.info("%s progress: %d/%d", name, i + 1, total)
        return correct / max(1, total)

    acc_full = _score(rows)
    acc_zero = _score(rows, zero=True)
    shuf_accs = [_score(_shuffle_evidence(rows, seed=17 + s)) for s in range(n_shuffle_seeds)]
    acc_shuf = float(sum(shuf_accs) / max(1, len(shuf_accs)))
    lat_rows = [r for r in rows if _is_laterality_claim(r.get("claim_text", ""))]
    if lat_rows:
        acc_lat = _score(lat_rows)
        acc_flip = _score(lat_rows, flip=True)
        ipg_gap = (acc_lat - acc_flip) * 100.0
    else:
        acc_flip = None
        ipg_gap = None

    img_gap = (acc_full - acc_zero) * 100.0
    esg_gap = (acc_full - acc_shuf) * 100.0
    result = BaselineResult(
        name=name,
        n_test=total,
        acc_full=acc_full,
        acc_image_zeroed=acc_zero,
        acc_evidence_shuffled=acc_shuf,
        acc_laterality_flipped=acc_flip,
        img_gap_pp=img_gap,
        esg_gap_pp=esg_gap,
        ipg_gap_pp=ipg_gap,
        evidence_blind=(img_gap < threshold_pp) or (esg_gap < threshold_pp),
        threshold_pp=threshold_pp,
    )
    logger.info("baseline=%s IMG=%.2fpp ESG=%.2fpp IPG=%s blind=%s",
                name, img_gap, esg_gap, f"{ipg_gap:.2f}" if ipg_gap is not None else "-",
                result.evidence_blind)
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(asdict(result), indent=2, default=str))
    return result


def available_baselines(device: torch.device | str = "cuda") -> list[BaselineVerifier]:
    """Return baselines whose prerequisites are present in the current env.

    Each baseline's init is tried unconditionally; failures are logged with
    full traceback so it is obvious in the logs which piece is missing (API
    key, gated weights, bad model ID, etc.).
    """
    out: list[BaselineVerifier] = []

    def _try(name: str, ctor):
        try:
            out.append(ctor())
            logger.info("baseline %s initialized", name)
        except Exception as exc:
            logger.exception("baseline %s init FAILED: %s", name, exc)

    # API baselines — attempt unconditionally; the SDK constructor will raise
    # a clean error if the env var is missing, which the try/except catches.
    _try("claude-3-5-sonnet", ClaudeBaseline)
    if os.environ.get("OPENAI_API_KEY"):
        _try("gpt-4o", OpenAIBaseline)
    if os.environ.get("GOOGLE_API_KEY"):
        _try("gemini-1-5-pro", GeminiBaseline)

    _try("biomedclip-zero-shot", lambda: BiomedCLIPZeroShotBaseline(device=device))
    _try("chexagent-2-3b", lambda: CheXagentV2Baseline(device=device))
    _try("medgemma-4b-it", lambda: MedGemma4BBaseline(device=device))
    _try("maira-2", lambda: MAIRA2Baseline(device=device))
    return out
