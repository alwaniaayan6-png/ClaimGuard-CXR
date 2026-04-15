"""Task 3b — Counterfactual paraphrase generator for contradicted claims.

This is the second of three files implementing the ACL 2025 "Dually
Self-Improved" counterfactual augmentation recipe for ClaimGuard-CXR.
Upstream, ``causal_term_identifier.py`` produces a ranked list of
``CausalSpan`` objects marking the token spans a v3 verifier depends
on for its contradicted verdict.  This module consumes those spans and
asks Claude Sonnet 4.5 to paraphrase everything around them:

    * CAUSAL tokens are pinned verbatim (case-insensitive substring
      check) — these are what the verifier must keep learning from.
    * Non-causal surface form is rephrased so that the v4 DPO trainer
      (Task 3c) sees (original, paraphrase) pairs with identical label
      but different lexical "shortcuts."  That is the signal we want
      v4 to generalize across.

Why a separate transport hook?
------------------------------
The Anthropic SDK is heavy and requires a paid API key; we don't want
unit tests to ever hit the real network.  ``CounterfactualGenerator``
accepts an optional ``transport: Callable[[str], str]`` that takes the
fully-rendered prompt and returns the raw response string.  If omitted,
a default transport is built from ``anthropic.Anthropic`` (which is
imported lazily so tests that pass a transport don't need the SDK
installed).

Public API
----------
    * ``CounterfactualVariant`` — dataclass for a single generated
      counterfactual (text + preserved tokens + edit distance).
    * ``build_prompt(claim, causal_tokens, n_variants)`` — pure
      function; returns the prompt string.
    * ``parse_variants_json(response_text)`` — pure; tolerant JSON
      parser (handles preambles, markdown fences, partial arrays).
    * ``validate_preservation(variant_text, required_tokens)`` —
      returns the list of tokens that were NOT preserved.
    * ``normalize_causal_tokens(tokens)`` — strip / dedupe.
    * ``levenshtein_distance(a, b)`` — edit distance for minimality
      scoring (ranks variants by closeness to the original).
    * ``CounterfactualGenerator(model, ..., transport=None)`` — heavy
      class that drives the Anthropic API (or an injected transport).
    * ``CounterfactualGenerator.generate(claim, causal_spans, n)`` —
      returns up to ``n`` validated ``CounterfactualVariant`` objects,
      sorted by increasing edit distance.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence, Union

from data.augmentation.causal_term_identifier import CausalSpan

logger = logging.getLogger(__name__)

# Type alias: either a pre-computed list of CausalSpan objects (the
# Task 3a output) or a raw list of token strings for tests / manual use.
CausalInput = Union[Sequence[CausalSpan], Sequence[str]]


@dataclass(frozen=True)
class CounterfactualVariant:
    """A single validated counterfactual paraphrase.

    Attributes:
        text: The paraphrase text.
        preserved_tokens: The causal tokens that the variant was
            validated against (all confirmed present via substring
            match before the variant was accepted).
        edit_distance: Levenshtein distance from the original claim.
            Lower = more minimal edit.
        warnings: Optional human-readable notes from the validator
            (e.g. "token preserved only case-insensitively").
    """

    text: str
    preserved_tokens: tuple[str, ...]
    edit_distance: int
    warnings: tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Pure helpers — unit-testable without Anthropic / network.
# ---------------------------------------------------------------------------


def normalize_causal_tokens(tokens: Sequence[str]) -> list[str]:
    """Strip leading/trailing whitespace and deduplicate tokens.

    RoBERTa-style BPE tokens often carry a leading space (e.g.
    ``" enlarged"``).  Stripping produces cleaner prompts and avoids
    confusing the LLM.  Duplicates are dropped case-sensitively to keep
    the prompt short.
    """
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in tokens:
        if not isinstance(raw, str):
            continue
        clean = raw.strip()
        if not clean or clean in seen:
            continue
        normalized.append(clean)
        seen.add(clean)
    return normalized


def build_prompt(
    claim: str,
    causal_tokens: Sequence[str],
    n_variants: int = 3,
) -> str:
    """Render the counterfactual-generation prompt.

    The prompt is deliberately strict:

    * causal tokens MUST appear verbatim
    * clinical semantics (laterality, severity, negation) MUST match
    * non-causal surface form SHOULD be paraphrased
    * output MUST be a JSON array of exactly ``n_variants`` strings

    Args:
        claim: The contradicted claim to paraphrase.
        causal_tokens: List of causal token strings (already normalized
            by ``normalize_causal_tokens``).
        n_variants: Number of paraphrases to request.

    Returns:
        A single prompt string suitable for passing to the Anthropic
        messages API or to a test transport.
    """
    causal_json = json.dumps(list(causal_tokens), ensure_ascii=False)
    return (
        "You are augmenting a radiology-claim verifier training set "
        "with counterfactual paraphrases. Given a single claim and a "
        "list of CAUSAL tokens that the verifier relies on to predict "
        "CONTRADICTED, produce exactly "
        f"{n_variants} minimally-edited paraphrase variants that:\n\n"
        "1. MUST contain every causal token verbatim (exact substring, "
        "case-insensitive allowed).\n"
        "2. SHOULD paraphrase the non-causal surface form — reorder "
        "words, swap synonyms, adjust grammar.\n"
        "3. MUST preserve the claim's clinical meaning exactly — same "
        "finding, same laterality, same severity, same negation "
        "polarity. Do not introduce new findings.\n"
        "4. SHOULD differ from the original claim and from each other.\n"
        "5. MUST output ONLY a valid JSON array of exactly "
        f"{n_variants} strings. No preamble, no markdown fences, no "
        "trailing commentary.\n\n"
        f"Claim:\n{claim}\n\n"
        f"Causal tokens (preserve verbatim):\n{causal_json}\n\n"
        f"Response (JSON array of {n_variants} strings):"
    )


# Greedy ``[...]`` finder.  Non-greedy would break on variants
# containing ``]`` characters (e.g. nested JSON artifacts in LLM output)
# — we match the widest possible array span, then defer to
# ``json.loads`` for final validation.  Multiline so it spans the whole
# response blob.
_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def parse_variants_json(
    response_text: Optional[str],
) -> list[str]:
    """Parse a list of paraphrase strings out of a raw LLM response.

    Tolerates:
        * Clean JSON array (``["a", "b", "c"]``)
        * Preamble text before the array
        * Markdown code fences (triple-backtick json ... triple-backtick)
        * Trailing prose after the array

    Does NOT tolerate:
        * Non-list JSON (object, string, number)
        * Missing / empty response
        * Zero non-empty string items

    Args:
        response_text: Raw response string from the LLM.

    Returns:
        A list of non-empty paraphrase strings (stripped).

    Raises:
        ValueError: if the response is empty, unparseable, or contains
            no valid strings.
    """
    if not response_text or not response_text.strip():
        raise ValueError("empty response")

    text = response_text.strip()

    # Strip markdown fences: ```json ... ``` or ``` ... ```.
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]  # drop opening fence
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    parsed: Any
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Fall back to extracting the first array we see in the blob.
        match = _JSON_ARRAY_RE.search(text)
        if match is None:
            raise ValueError(
                f"no JSON array found in response: {text[:200]!r}"
            )
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError as e:
            raise ValueError(f"malformed JSON array: {e}") from e

    if not isinstance(parsed, list):
        raise ValueError(
            f"expected JSON list, got {type(parsed).__name__}"
        )

    result: list[str] = []
    for item in parsed:
        if not isinstance(item, str):
            continue
        clean = item.strip()
        if clean:
            result.append(clean)

    if not result:
        raise ValueError("no non-empty string items in response")

    return result


def validate_preservation(
    variant_text: str,
    required_tokens: Sequence[str],
) -> list[str]:
    """Return the list of required tokens that are NOT in ``variant_text``.

    Match is case-insensitive **word-boundary** regex
    (``\\btoken\\b``).  A prior implementation used plain substring
    matching (``token.lower() in variant.lower()``), which the 2026-04-14
    pre-flight reviewer flagged as too lax: short causal tokens like
    ``"no"`` would match ``"normal"`` or ``"nodule"``, and the token
    ``"is"`` would match literally any sentence containing ``"is"``.
    Under the old rule, Claude could produce a variant that looked
    nothing like the original and the preservation check would still
    pass.

    Word-boundary matching fixes this: ``\\bno\\b`` only matches ``"no"``
    as a standalone word, not as a prefix of ``"normal"``.  This is
    slightly stricter than the old behavior — in particular, a token
    written in the original as ``"enlarged"`` will not match a variant
    that uses ``"enlarged."`` (with trailing punctuation) — but Python's
    ``\\b`` considers punctuation a word boundary, so punctuation-
    adjacent matches still work correctly.

    For multi-word tokens (e.g. ``"heart is enlarged"``), we use
    ``re.escape`` on the whole phrase and wrap it in ``\\b`` on each
    side.  This preserves the semantic of "this exact phrase must
    appear somewhere in the variant."
    """
    if not required_tokens:
        return []
    missing: list[str] = []
    for token in required_tokens:
        if not token:
            continue
        pattern = r"\b" + re.escape(token) + r"\b"
        if not re.search(pattern, variant_text, flags=re.IGNORECASE):
            missing.append(token)
    return missing


def levenshtein_distance(a: str, b: str) -> int:
    """Standard O(|a|·|b|) Levenshtein edit distance.

    Used to rank counterfactual variants by proximity to the original
    claim (smaller = more minimal edit, which is what we want for the
    DPO trainer so the preference margin is driven by meaning, not by
    random surface-form drift).
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    # Ensure |a| >= |b| for minimal working memory (optional micro-opt).
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        current = [i + 1]
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else 1
            current.append(
                min(
                    previous[j + 1] + 1,  # deletion
                    current[j] + 1,       # insertion
                    previous[j] + cost,   # substitution
                )
            )
        previous = current
    return previous[-1]


# ---------------------------------------------------------------------------
# Heavy class — Anthropic transport with retry loop.
# ---------------------------------------------------------------------------


class CounterfactualGenerator:
    """Drive Claude Sonnet 4.5 to produce minimally-edited counterfactuals.

    The class is deliberately transport-agnostic: pass ``transport=<fn>``
    to short-circuit the Anthropic SDK for unit tests or for using a
    different provider.  The default transport lazily imports
    ``anthropic`` on first instantiation.
    """

    def __init__(
        self,
        *,
        model: str = "claude-sonnet-4-5",
        api_key: Optional[str] = None,
        max_retries: int = 3,
        max_tokens: int = 1024,
        transport: Optional[Callable[[str], str]] = None,
    ) -> None:
        """Args:
        model: Anthropic model id.  Default is the locked-in Task 1
            silver-standard grader.
        api_key: Optional explicit key; otherwise uses the SDK's env.
        max_retries: Re-prompts if the first batch doesn't produce
            ``n_variants`` validated paraphrases.  3 is empirically
            enough for radiology claims.
        max_tokens: Upper bound on response length.  1024 is ample for
            3 × ~30-word variants.
        transport: Optional callable ``(prompt) -> response_text``.
            If ``None``, builds a default transport from the Anthropic
            SDK (which is imported lazily).
        """
        self.model = model
        self.max_retries = int(max_retries)
        self.max_tokens = int(max_tokens)

        if transport is not None:
            self._transport = transport
            return

        try:
            import anthropic  # noqa: WPS433 — lazy import is intentional
        except ImportError as e:  # noqa: BLE001
            raise ImportError(
                "CounterfactualGenerator requires the `anthropic` SDK "
                "for the default transport. Install with "
                "`pip install anthropic`, or pass `transport=<callable>` "
                "to inject a custom transport (e.g. for unit tests)."
            ) from e

        client = (
            anthropic.Anthropic(api_key=api_key)
            if api_key is not None
            else anthropic.Anthropic()
        )

        def _default_transport(prompt: str) -> str:
            resp = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            # ``resp.content`` is a list of blocks; pick the first text block.
            for block in resp.content:
                text = getattr(block, "text", None)
                if text:
                    return text
            return ""

        self._transport = _default_transport

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def generate(
        self,
        claim: str,
        causal_spans: CausalInput,
        *,
        n_variants: int = 3,
    ) -> list[CounterfactualVariant]:
        """Return up to ``n_variants`` validated counterfactual variants.

        Args:
            claim: The original contradicted claim.
            causal_spans: Either a list of ``CausalSpan`` objects (the
                standard path, from Task 3a) or a plain list of token
                strings (for tests / manual use).
            n_variants: Target number of variants to keep.

        Returns:
            A list of ``CounterfactualVariant`` objects, sorted by
            increasing edit distance (most minimal edit first).  May be
            shorter than ``n_variants`` if validation repeatedly failed
            — callers should check ``len(result)`` and decide whether
            to skip the example from the DPO training set.
        """
        if not isinstance(claim, str) or not claim.strip():
            return []
        if n_variants <= 0:
            return []

        causal_tokens = self._extract_tokens(causal_spans)
        if not causal_tokens:
            logger.warning(
                "CounterfactualGenerator.generate called with empty "
                "causal tokens for claim=%r; skipping.",
                claim[:80],
            )
            return []

        collected: list[CounterfactualVariant] = []
        seen_texts: set[str] = {claim.strip()}
        attempts_remaining = self.max_retries + 1

        while attempts_remaining > 0 and len(collected) < n_variants:
            attempts_remaining -= 1
            prompt = build_prompt(claim, causal_tokens, n_variants)
            try:
                raw = self._transport(prompt)
            except Exception as e:  # noqa: BLE001
                logger.warning("Transport error: %s", e)
                continue

            try:
                variants = parse_variants_json(raw)
            except ValueError as e:
                logger.info("Parse failure on attempt: %s", e)
                continue

            for v in variants:
                if v in seen_texts:
                    continue
                missing = validate_preservation(v, causal_tokens)
                if missing:
                    logger.debug(
                        "Variant dropped — missing tokens %r in %r",
                        missing, v[:80],
                    )
                    continue
                dist = levenshtein_distance(claim, v)
                if dist == 0:
                    continue  # identical, not a counterfactual
                collected.append(
                    CounterfactualVariant(
                        text=v,
                        preserved_tokens=tuple(causal_tokens),
                        edit_distance=dist,
                    )
                )
                seen_texts.add(v)
                if len(collected) >= n_variants:
                    break

        collected.sort(key=lambda cv: cv.edit_distance)
        return collected[:n_variants]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_tokens(spans: CausalInput) -> list[str]:
        """Flatten ``CausalSpan``s or plain strings into a normalized list."""
        if not spans:
            return []
        tokens: list[str] = []
        for item in spans:
            if isinstance(item, CausalSpan):
                tokens.append(item.text)
            elif isinstance(item, str):
                tokens.append(item)
            # Silently ignore unknown types — the caller shouldn't be
            # feeding us mixed data, but if they do we stay permissive.
        return normalize_causal_tokens(tokens)


__all__ = [
    "CausalInput",
    "CounterfactualGenerator",
    "CounterfactualVariant",
    "build_prompt",
    "levenshtein_distance",
    "normalize_causal_tokens",
    "parse_variants_json",
    "validate_preservation",
]
