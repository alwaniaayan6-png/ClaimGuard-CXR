"""Task 3a — Identify causal token spans in (claim, evidence) contradicted pairs.

This module is one of three files implementing the ACL 2025 "Dually
Self-Improved" counterfactual augmentation recipe for ClaimGuard-CXR.
Given a trained v3 verifier and a contradicted-label (claim, evidence)
pair, we want the top-k token spans that the model relies on to emit
the contradicted logit.  These spans are then pinned into the
counterfactual-generation prompt (Task 3b) as tokens that MUST be
preserved verbatim, while everything else is paraphrased.  The DPO
trainer (Task 3c) uses these (original, paraphrase) pairs to teach
v4 that the causal tokens — not the surface form — determine the
verdict.

Why captum LayerIntegratedGradients?
-----------------------------------
Attention rollout alone is noisy: the last-layer CLS attention is
dominated by a handful of high-frequency tokens regardless of
label.  Integrated gradients give a path-integral estimate of each
input embedding's contribution to the output logit, which is both
more faithful (Sundararajan et al. 2017) and trivially decomposable
across claim / evidence sides.

Implementation design
---------------------
We intentionally split the class from the pure scoring helpers so the
helpers can be unit-tested without captum / torch.  The heavy class
``CausalTermIdentifier`` is optional — if captum or torch isn't
available at import time, the class is still importable but raises on
instantiation.  Callers that only want to test the span-extraction
logic can import ``score_to_spans`` / ``merge_contiguous_spans``
directly.

Public API
----------
    * ``CausalSpan`` — dataclass for a single top-k span.
    * ``score_to_spans(tokens, offsets, scores, source_ids, top_k)``
      — pure function: select top-k spans from token-level scores.
    * ``merge_contiguous_spans(spans)`` — merge adjacent spans
      sharing the same ``source`` into longer phrase-level spans.
    * ``CausalTermIdentifier(model_path, ...)`` — captum wrapper.
    * ``CausalTermIdentifier.identify(claim, evidence, top_k=5)``
      — returns a sorted list of ``CausalSpan`` objects.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Literal, Optional, Sequence

logger = logging.getLogger(__name__)

SpanSource = Literal["claim", "evidence"]


@dataclass(frozen=True)
class CausalSpan:
    """A single top-k causal span for the counterfactual generator.

    Attributes:
        text: The span's surface text (not lowercased, whitespace
            preserved).
        source: ``"claim"`` or ``"evidence"`` — which side of the
            cross-encoder input the span came from.
        score: Summed absolute attribution score across the span's
            tokens.  Higher = more causal.
        start_char: Inclusive character offset in the source string.
        end_char: Exclusive character offset in the source string.
        token_indices: The token indices in the tokenizer output
            sequence that contributed to this span.  Mostly useful
            for debugging; the counterfactual generator only reads
            ``text`` / ``source`` / ``score``.
    """

    text: str
    source: SpanSource
    score: float
    start_char: int
    end_char: int
    token_indices: tuple[int, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Pure scoring helpers — unit-testable without captum / torch.
# ---------------------------------------------------------------------------


def score_to_spans(
    tokens: Sequence[str],
    offsets: Sequence[tuple[int, int]],
    scores: Sequence[float],
    source_ids: Sequence[SpanSource],
    source_texts: dict[SpanSource, str],
    top_k: int = 5,
) -> list[CausalSpan]:
    """Extract the top-k causal spans from per-token attribution scores.

    Args:
        tokens: Decoded token strings (not subword ids).  Special
            tokens such as ``<s>``, ``</s>`` should have zero offsets
            in ``offsets`` and empty strings here — the caller is
            responsible for masking them out.
        offsets: Character offsets ``(start, end)`` within the source
            text the token belongs to.  Zero-length tuples (0, 0)
            mark non-source-aligned special tokens and are ignored.
        scores: Per-token causal score (e.g. ``|integrated_grad|``
            summed across the embedding dim).  Must have the same
            length as ``tokens``.
        source_ids: Per-token ``"claim"`` / ``"evidence"`` label.
        source_texts: Mapping ``{"claim": str, "evidence": str}``
            used to resolve ``(start, end)`` back into span text.
        top_k: How many spans to return.

    Returns:
        A list of up to ``top_k`` ``CausalSpan`` objects, sorted by
        decreasing ``score``.  Ties are broken by token index (stable).

    Raises:
        ValueError: if any of the input sequences disagree in length.
    """
    n = len(tokens)
    if not (len(offsets) == n and len(scores) == n and len(source_ids) == n):
        raise ValueError(
            "tokens/offsets/scores/source_ids must have the same length"
        )
    if top_k <= 0:
        return []

    # Build one span per real token.  Adjacent spans within the same
    # source side can be merged later by ``merge_contiguous_spans``.
    candidates: list[CausalSpan] = []
    for i in range(n):
        start, end = offsets[i]
        if end <= start:
            continue  # special token
        src = source_ids[i]
        if src not in source_texts:
            continue
        text = source_texts[src][start:end]
        if not text.strip():
            continue
        candidates.append(
            CausalSpan(
                text=text,
                source=src,
                score=float(abs(scores[i])),
                start_char=int(start),
                end_char=int(end),
                token_indices=(i,),
            )
        )

    # Sort by (−score, token_index_sum) for stable top-k.
    candidates.sort(
        key=lambda s: (-s.score, s.token_indices[0] if s.token_indices else 0)
    )
    return candidates[:top_k]


def merge_contiguous_spans(
    spans: Iterable[CausalSpan],
    source_texts: dict[SpanSource, str],
    join_whitespace: bool = True,
) -> list[CausalSpan]:
    """Merge adjacent same-source spans into phrase-level spans.

    Two spans merge if they share a source AND the second starts at
    the first's end character (or at +1 to allow a single whitespace
    separator when ``join_whitespace`` is set).

    The merged span's ``score`` is the sum of the sources', and its
    ``text`` is re-sliced from ``source_texts`` to include any
    intervening whitespace.

    Input spans are processed in their provided order — pass in the
    top-k output of ``score_to_spans`` (sorted by score) if you want
    the highest-scoring spans as merge seeds.
    """
    sorted_spans = sorted(
        spans, key=lambda s: (s.source, s.start_char)
    )
    merged: list[CausalSpan] = []
    for span in sorted_spans:
        if not merged:
            merged.append(span)
            continue
        last = merged[-1]
        gap = span.start_char - last.end_char
        same_source = span.source == last.source
        if same_source and (gap == 0 or (join_whitespace and gap <= 1)):
            new_start = last.start_char
            new_end = span.end_char
            src_text = source_texts[span.source]
            new_text = src_text[new_start:new_end]
            merged[-1] = CausalSpan(
                text=new_text,
                source=span.source,
                score=last.score + span.score,
                start_char=new_start,
                end_char=new_end,
                token_indices=tuple(
                    list(last.token_indices) + list(span.token_indices)
                ),
            )
        else:
            merged.append(span)

    merged.sort(key=lambda s: -s.score)
    return merged


def split_tokens_by_sep(
    special_mask: Sequence[int],
    sep_index: Optional[int],
) -> list[SpanSource]:
    """Label each token with its input side based on the SEP index.

    Args:
        special_mask: 1 for special tokens (CLS/SEP/PAD), 0 otherwise.
            Same length as the token sequence.
        sep_index: The index of the first SEP token in the sequence,
            which separates claim from evidence in a RoBERTa-style
            cross-encoder input.  If ``None``, the entire sequence is
            labeled ``"claim"``.

    Returns:
        List of ``"claim"`` / ``"evidence"`` labels per token.  Special
        tokens are labeled ``"claim"`` but should be filtered out by
        ``score_to_spans`` based on their offsets being ``(0, 0)``.
    """
    labels: list[SpanSource] = []
    seen_sep = False
    for i, is_special in enumerate(special_mask):
        if sep_index is not None and i >= sep_index:
            seen_sep = True
        labels.append("evidence" if seen_sep else "claim")
    return labels


# ---------------------------------------------------------------------------
# Heavy captum wrapper — optional, only used from the Modal trainer path.
# ---------------------------------------------------------------------------


class CausalTermIdentifier:
    """Run Layer Integrated Gradients over a v3 verifier to find top-k spans.

    This class is imported lazily by the DPO data-prep pipeline.  If
    ``torch`` / ``transformers`` / ``captum`` are not installed at
    instantiation time, the constructor raises a clear ``ImportError``
    with installation instructions.
    """

    def __init__(
        self,
        model_path: str,
        *,
        hf_backbone: str = "roberta-large",
        target_label: int = 1,
        device: Optional[str] = None,
        n_ig_steps: int = 50,
        top_k: int = 5,
    ) -> None:
        """Args:
        model_path: Path to the trained v3 verifier ``.pt`` file.
        hf_backbone: HuggingFace model id to initialise the
            encoder (default ``"roberta-large"``).
        target_label: Which class to attribute toward.  For the v3
            binary head, 1 = contradicted (the class we want to
            explain).
        device: ``"cuda"`` / ``"cpu"``; defaults to auto-detect.
        n_ig_steps: Integrated-gradients path-length steps.  50
            matches the Sundararajan et al. 2017 default.
        top_k: How many spans to keep per (claim, evidence) pair.
        """
        try:
            import torch
            from captum.attr import LayerIntegratedGradients
        except ImportError as e:  # noqa: BLE001
            raise ImportError(
                "CausalTermIdentifier requires torch, transformers, and "
                "captum. Install with `pip install torch transformers "
                "captum`."
            ) from e

        # D21 fix (2026-04-15): import the canonical VerifierModel
        # loader from inference.verifier_model so we load the FULL
        # multimodal architecture (text_encoder + heatmap_encoder +
        # verdict_head + score_head + contrastive_proj) instead of
        # the broken plain AutoModel + Linear(hidden, 2) layout that
        # silently dropped the head weights.  The previous loader
        # built integrated gradients against a random-init Linear
        # head, producing meaningless attributions.  See
        # decisions.md D21 + D25 for the full incident report.
        try:
            from inference.verifier_model import load_verifier_checkpoint
        except ImportError as e:
            raise ImportError(
                "CausalTermIdentifier requires inference/verifier_model.py. "
                "If running in a Modal container, ensure the image was "
                "built with `.add_local_python_source(\"inference\")` "
                "(D20 fix)."
            ) from e

        self._torch = torch
        self._lig_cls = LayerIntegratedGradients

        self.model_path = model_path
        self.hf_backbone = hf_backbone
        self.target_label = int(target_label)
        self.top_k = int(top_k)
        self.n_ig_steps = int(n_ig_steps)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # ---- Load the FULL VerifierModel (D21 fix) ----
        # load_verifier_checkpoint raises RuntimeError if any non-
        # allowed key is missing after load_state_dict, so a wrong
        # architecture or stale checkpoint format will fail loudly
        # at construction time, not silently produce garbage
        # attributions later.
        self.tokenizer, self.model = load_verifier_checkpoint(
            checkpoint_path=model_path,
            hf_backbone=hf_backbone,
            device=self.device,
            num_classes=2,
        )
        # The model is in .eval() by default from the loader. We
        # don't switch to .train() because IG only needs grad-on
        # against inputs, not parameters.

        # ---- Set up LIG on the word-embedding layer ----
        # The text encoder is now a wrapped HF model under
        # self.model.text_encoder. The embedding layer is at
        # text_encoder.embeddings.word_embeddings (RoBERTa
        # convention) OR text_encoder.roberta.embeddings.* (rare
        # alternate). We probe both.
        text_enc = self.model.text_encoder
        if hasattr(text_enc, "embeddings"):
            self._embedding_layer = text_enc.embeddings.word_embeddings
        elif hasattr(text_enc, "roberta"):
            self._embedding_layer = (
                text_enc.roberta.embeddings.word_embeddings
            )
        else:
            raise RuntimeError(
                "Cannot locate word_embeddings on text_encoder of type "
                f"{type(text_enc).__name__}. Expected HF RoBERTa-style "
                "model with .embeddings.word_embeddings."
            )
        self._lig = self._lig_cls(self._forward_logits, self._embedding_layer)

    def _forward_logits(
        self,
        input_ids: "Any",  # torch.Tensor
        attention_mask: "Any",
    ):
        """Forward pass returning the 2-class logits for LIG.

        Calls the FULL VerifierModel forward (text_encoder +
        heatmap_encoder zero-fill path + verdict_head). Returns the
        verdict_logits tensor of shape (batch, 2). The zero heatmap
        is automatically supplied by VerifierModel.forward when
        heatmap=None.
        """
        verdict_logits, _sigmoid_score = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return verdict_logits

    def identify(
        self,
        claim: str,
        evidence: str,
        *,
        top_k: Optional[int] = None,
        max_length: int = 256,
    ) -> list[CausalSpan]:
        """Return the top-k causal spans from a (claim, evidence) pair.

        The cross-encoder is fed as ``[CLS] claim [SEP] evidence [SEP]``.
        Spans are returned sorted by decreasing |attribution|.  The
        list is deduped by character span so no (claim, evidence) pair
        produces duplicate entries even if the tokenizer splits a
        word across subwords.
        """
        torch = self._torch
        k = top_k if top_k is not None else self.top_k

        enc = self.tokenizer(
            claim,
            evidence,
            return_tensors="pt",
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        offsets = enc["offset_mapping"][0].tolist()
        special_mask = enc["special_tokens_mask"][0].tolist()

        # Baseline: all PAD tokens (the canonical IG baseline for NLP).
        pad_id = self.tokenizer.pad_token_id or 0
        baseline = torch.full_like(input_ids, pad_id)

        attributions = self._lig.attribute(
            inputs=input_ids,
            baselines=baseline,
            additional_forward_args=(attention_mask,),
            target=self.target_label,
            n_steps=self.n_ig_steps,
            return_convergence_delta=False,
        )
        # Sum across embedding dim → per-token scalar.
        per_token = attributions.sum(dim=-1)[0].detach().cpu().tolist()

        # Determine the SEP index (first evidence-side token).  For
        # RoBERTa the structure is ``<s> claim </s></s> evidence </s>``,
        # so after the claim there are TWO contiguous special tokens
        # (`</s>` and `</s>`) before the first evidence subword.  The
        # 2026-04-14 pre-flight reviewer flagged that the prior code
        # did ``sep_index = i + 1`` where ``i`` was the first
        # special_mask==1 position after the CLS, which landed on the
        # SECOND `</s>` (still a special token) rather than the first
        # evidence subword.  That caused the first evidence token to
        # be silently mislabeled as claim-side in
        # ``split_tokens_by_sep``.
        #
        # The fix: after finding the first special position after CLS,
        # advance past ALL contiguous special tokens so ``sep_index``
        # points at the first real evidence subword.
        sep_index: Optional[int] = None
        for i in range(1, len(special_mask)):
            if special_mask[i] == 1:
                j = i + 1
                while j < len(special_mask) and special_mask[j] == 1:
                    j += 1
                sep_index = j  # first non-special token after the SEP run
                break

        sources = split_tokens_by_sep(special_mask, sep_index)

        # Re-align offsets: the tokenizer returns global offsets that
        # count in the concatenated claim+evidence string.  Offsets
        # beyond the SEP should be interpreted as starting fresh in
        # the evidence; HuggingFace helpfully resets them for the
        # second sequence, so we can index directly.
        source_texts: dict[SpanSource, str] = {
            "claim": claim,
            "evidence": evidence,
        }

        # Decode tokens for debugging / merge phase.
        tokens = [
            self.tokenizer.decode([tid])
            for tid in input_ids[0].cpu().tolist()
        ]

        spans = score_to_spans(
            tokens=tokens,
            offsets=offsets,
            scores=per_token,
            source_ids=sources,
            source_texts=source_texts,
            top_k=max(k * 3, k),  # oversample, merge, then trim
        )
        merged = merge_contiguous_spans(spans, source_texts=source_texts)
        return merged[:k]


__all__ = [
    "CausalSpan",
    "CausalTermIdentifier",
    "merge_contiguous_spans",
    "score_to_spans",
    "split_tokens_by_sep",
]
