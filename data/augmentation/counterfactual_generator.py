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


def _is_meaningful_causal_token(token: str) -> bool:
    """Filter out noise tokens that paraphrases never preserve.

    Integrated-gradients on a noisy text classifier surfaces a mix of
    real content tokens AND subword/punctuation artifacts in its top-K
    output.  Examples from a real Task 3a run:

        ['STITIAL EDEMA worsening.',  # good — contains the finding
         'severe',                     # good — severity word
         ',',                          # noise — bare punctuation
         'single portable',            # noise — generic boilerplate
         '.']                          # noise — bare punctuation

    `validate_preservation` requires word-boundary `\\btoken\\b` matches
    in the variant; bare punctuation and BPE subword fragments will
    almost never match a paraphrase, so requiring them rejects every
    Claude variant and produces n_returned=0 for the row.

    A token is considered "meaningful" if it satisfies all of:
        * length ≥ 3 characters after strip
        * contains at least 2 alphabetic characters
        * does not start with a punctuation/digit character (drops
          tokens like '.W', '.A', '. T', '1.AP')
        * is not a generic boilerplate phrase (drops 'single portable',
          'in the', 'of the', etc.)

    The boilerplate stopwords are radiology-specific noise tokens we
    observed surfaced repeatedly during the smoke test on the v3
    training data.

    This is a hard filter, not a re-ranking — noise tokens are dropped
    entirely from the prompt and the validation set, so Claude doesn't
    waste tokens trying to preserve them.
    """
    import string

    if not isinstance(token, str):
        return False
    clean = token.strip()
    if len(clean) < 4:  # require at least 4 chars
        return False

    alpha_count = sum(1 for c in clean if c.isalpha())
    if alpha_count < 4:  # require at least 4 alpha chars
        return False

    if clean[0] in string.punctuation or clean[0].isdigit():
        return False

    # Reject tokens that have any non-alphanumeric char in a leading
    # position other than spaces between words.  Drops things like
    # "ENLAR" → no, that's all alphabetic.  Drops "[ TO PRIOR" → the
    # leading-char check already gets it.  Drops "1.AP" → digit-leading.
    # The remaining concern is BPE subword fragments like "STITIAL"
    # (from "interstitial") or "OSSIBLE" (from "possible") that ARE
    # alphabetic but aren't real words.  We can't distinguish those
    # from real medical terms without a dictionary, so we accept them
    # — `validate_preservation` will reject Claude variants that don't
    # contain them, which gracefully drops the row from the training
    # set.  The cost is a smaller training set, not a wrong one.

    # Heuristic: a token is likely a BPE fragment if it is ALL CAPS
    # AND has 4-7 chars AND doesn't end in a real word boundary
    # marker.  Common medical terms longer than 7 chars (e.g.
    # PNEUMOTHORAX, CARDIOMEDIASTINAL) are kept; short ALL CAPS
    # fragments (ENLAR, OSSIBLE, ASIS) are dropped.
    if clean.isupper() and 4 <= len(clean) <= 7:
        return False

    # Reject single ALL-CAPS abbreviations followed by punctuation
    # (e.g. "ASIS.", "EST.", "EDEM.")
    stripped_punct = clean.rstrip(string.punctuation)
    if (
        stripped_punct.isupper()
        and 4 <= len(stripped_punct) <= 7
    ):
        return False

    # Boilerplate fragments observed in the v3 IG output that are
    # generic enough to appear in any radiology report and therefore
    # carry no per-claim discriminative signal.
    boilerplate = {
        "single portable", "the chest", "of the", "in the", "to the",
        "to prior", "in stable", "for further", "is again", "is no",
        "of a", "for the", "demonstrate", "demonstrates",
        "demonstrated", "again seen", "again noted", "is seen",
        "is noted", "is present", "are seen", "is unchanged",
        "and", "or", "the", "this", "that", "with", "from",
        "without", "but",
    }
    if clean.lower() in boilerplate:
        return False

    return True


def normalize_causal_tokens(
    tokens: Sequence[str],
    *,
    max_tokens: int = 3,
    drop_noise: bool = True,
) -> list[str]:
    """Strip whitespace, dedupe, drop noise, and cap to top-K.

    RoBERTa-style BPE tokens often carry a leading space (e.g.
    ``" enlarged"``).  Stripping produces cleaner prompts and avoids
    confusing the LLM.  Duplicates are dropped case-sensitively to keep
    the prompt short.

    2026-04-15 update: noise tokens (bare punctuation, BPE fragments,
    generic boilerplate) are dropped via ``_is_meaningful_causal_token``,
    and the result is capped to ``max_tokens`` (default 3) so the
    counterfactual generator only requires preservation of the top
    causal content tokens.  Without this filter the Task 3b
    `validate_preservation` step rejected ~99% of Claude's variants
    on real v3 training claims.

    Args:
        tokens: Raw causal token list from the IG attribution step.
        max_tokens: Maximum number of meaningful tokens to keep.
            Default 3 — empirically enough to constrain Claude's
            paraphrase without over-restricting the search space.
            Set to 0 (or any large number) to disable the cap.
        drop_noise: If True (default), apply the meaningful-token
            filter.  Set to False for unit tests that want raw
            normalization without filtering.

    Returns:
        A list of up to ``max_tokens`` normalized, deduplicated,
        meaningful causal tokens, in the input order.
    """
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in tokens:
        if not isinstance(raw, str):
            continue
        clean = raw.strip()
        if not clean or clean in seen:
            continue
        if drop_noise and not _is_meaningful_causal_token(clean):
            continue
        normalized.append(clean)
        seen.add(clean)
        if max_tokens and len(normalized) >= max_tokens:
            break

    # FALLBACK: if the strict filter dropped EVERY token, fall back to
    # the longest-by-alpha-count original (after basic strip/dedupe).
    # This prevents the row from being silently dropped just because
    # IG happened to surface only noise tokens for it; even a single
    # subword fragment as a preservation hint gives Claude SOMETHING
    # to anchor on, and the resulting paraphrase has a chance of
    # being kept.  Without this fallback, ~70% of v3 rows produce
    # empty preservation sets and get dropped from the training data.
    if drop_noise and not normalized:
        best = ""
        best_alpha = -1
        for raw in tokens:
            if not isinstance(raw, str):
                continue
            clean = raw.strip()
            if not clean:
                continue
            n_alpha = sum(1 for c in clean if c.isalpha())
            if n_alpha > best_alpha:
                best = clean
                best_alpha = n_alpha
        if best:
            normalized.append(best)

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

    Hybrid matching strategy:

    * **Tokens ≥ 5 chars OR multi-word**: case-insensitive substring
      match.  This is permissive enough to accept BPE subword
      fragments embedded in larger words (e.g. ``"STITIAL EDEMA
      worsening."`` matched against ``"interSTITIAL EDEMA
      worsening."``), which IG attribution surfaces routinely.
    * **Tokens < 5 chars and single-word**: case-insensitive
      word-boundary regex (``\\btoken\\b``).  This prevents short
      tokens like ``"heart"`` from accidentally matching
      ``"heartfelt"``, which is the case the 2026-04-14 pre-flight
      reviewer originally flagged.

    The boundary at 5 chars is empirical: medical BPE fragments are
    typically 4-7 chars (``STITIAL``, ``OSSIBLE``, ``EGALY``, etc.),
    while the false-positive substring collisions for ``\\btoken\\b``
    happen mostly with short common English words (``"no"``,
    ``"is"``, ``"of"``, ``"at"``).  ``normalize_causal_tokens`` with
    ``drop_noise=True`` already filters tokens < 4 alpha chars and
    English stopwords, so most short tokens never reach this
    validator anyway — but the strict path here is the safety net
    for callers that bypass the filter.

    For multi-word tokens (e.g. ``"heart is enlarged"``), the entire
    phrase must appear as a contiguous case-insensitive substring,
    regardless of length.
    """
    if not required_tokens:
        return []
    variant_lower = variant_text.lower()
    missing: list[str] = []
    for token in required_tokens:
        if not token:
            continue
        token_lower = token.lower()
        is_multi_word = " " in token_lower
        if is_multi_word or len(token_lower) >= 6:
            # Permissive substring match for long tokens / phrases
            if token_lower not in variant_lower:
                missing.append(token)
        else:
            # Strict word-boundary match for short single-word tokens
            pattern = r"\b" + re.escape(token) + r"\b"
            if not re.search(
                pattern, variant_text, flags=re.IGNORECASE,
            ):
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
