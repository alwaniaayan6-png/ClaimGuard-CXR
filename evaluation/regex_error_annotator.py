"""Post-hoc regex error annotator — diagnostic metadata only.

Tags each radiology claim with flags indicating the presence of
structural error surfaces (fabricated measurements, priors, dates,
relative temporals).  **These flags are additive metadata; they never
change a claim's triage label.**  False-positive rates are too high
for regex alone to be safe as a gate in a medical pipeline — but the
flags are cheap, deterministic, and useful for the paper's error
analysis tables.

Task 4 (demoted) per the v3 sprint plan.  Consumed by
``scripts/compile_silver_standard_results.py`` and
``scripts/analyze_eval_results.py`` to populate paper tables.

Public API:
    ``annotate(claim)``         — return a dict ``{"regex_flags": [...]}``.
    ``annotate_all(claims)``    — batched; returns a list of dicts.
    ``FLAG_NAMES``              — tuple of possible flag names.
    ``PATTERNS``                — compiled regex patterns (module-level).
"""

from __future__ import annotations

import re
from typing import Iterable


#: Regex matching an explicit numeric measurement (mm or cm).  Matches
#: both simple forms ("3 mm"), decimal forms ("1.2 cm"), hyphenated
#: forms ("6-mm"), and dimension-pair forms ("3 x 4 mm").
MEAS_REGEX = re.compile(
    r"\b\d+(?:\.\d+)?"
    r"(?:\s*[x×-]\s*\d+(?:\.\d+)?)?"
    r"\s*[- ]?"
    r"(?:mm|cm|millimeter|centimeter)s?\b",
    re.IGNORECASE,
)

#: Regex matching a comparison-to-prior-exam phrase.  Covers the
#: common "compared to the prior study / exam / film" family as well
#: as "since the previous X" and "from the last Y".
PRIOR_REGEX = re.compile(
    r"\b(?:compared (?:to|with)|since|from|relative to|interval (?:change|increase|decrease|worsening|improvement))"
    r"\s+(?:the\s+)?"
    r"(?:prior|previous|last)\s+"
    r"(?:study|exam|film|image|report|radiograph|imaging|chest x-?ray|cxr)\b",
    re.IGNORECASE,
)

#: Regex matching an explicit calendar date.  Both numeric
#: (MM/DD/YYYY, with 2- or 4-digit years) and written-out month-name
#: forms.  Does not attempt to validate real dates.
DATE_REGEX = re.compile(
    r"\b(?:"
    r"\d{1,2}/\d{1,2}/\d{2,4}"
    r"|(?:january|february|march|april|may|june|july|august|september|october|november|december)"
    r"\s+\d{1,2}(?:,\s*\d{4})?"
    r")\b",
    re.IGNORECASE,
)

#: Regex matching a relative temporal reference such as "3 days ago",
#: "yesterday", "last week".  Note the overlap with PRIOR_REGEX — a
#: claim like "compared to last week's study" fires both flags, which
#: is fine for diagnostic purposes.
RELTIME_REGEX = re.compile(
    r"\b(?:"
    r"\d+\s+(?:day|week|month|year)s?\s+ago"
    r"|yesterday"
    r"|last\s+(?:week|month|year)"
    r"|over\s+the\s+(?:past|prior|last)\s+\d+\s+(?:day|week|month|year)s?"
    r")\b",
    re.IGNORECASE,
)


#: Mapping of flag names → pattern.  Update PATTERNS (and the
#: ``annotate`` docstring) if you add a new flag.
PATTERNS: dict[str, re.Pattern[str]] = {
    "fabricated_measurement": MEAS_REGEX,
    "fabricated_prior": PRIOR_REGEX,
    "fabricated_date": DATE_REGEX,
    "fabricated_relative_time": RELTIME_REGEX,
}

#: Ordered tuple of every flag name the annotator can emit.  Stable
#: across releases — paper tables can rely on this ordering.
FLAG_NAMES: tuple[str, ...] = tuple(PATTERNS.keys())


def annotate(claim: str) -> dict:
    """Annotate a single claim with structural error-surface flags.

    Args:
        claim: The claim text.  Empty or non-string inputs return an
            empty flag list.

    Returns:
        A dict ``{"regex_flags": [flag_name, ...]}`` where each element
        is a name from :data:`FLAG_NAMES`.  Multiple flags can be set;
        order follows :data:`FLAG_NAMES`.

    This function is deterministic and free of I/O.  It MUST NOT be
    used to change a claim's triage label — the flags are diagnostic
    metadata only.
    """
    if not isinstance(claim, str) or not claim:
        return {"regex_flags": []}

    flags: list[str] = []
    for name, pattern in PATTERNS.items():
        if pattern.search(claim):
            flags.append(name)
    return {"regex_flags": flags}


def annotate_all(claims: Iterable[str]) -> list[dict]:
    """Apply :func:`annotate` to an iterable of claims."""
    return [annotate(c) for c in claims]


def has_any_flag(claim: str) -> bool:
    """Shortcut: True iff :func:`annotate` returns at least one flag."""
    return bool(annotate(claim)["regex_flags"])


def count_flags(claims: Iterable[str]) -> dict[str, int]:
    """Count how many claims fire each flag.

    Useful for quickly producing the paper's error-surface distribution
    table from a full results JSON.

    Args:
        claims: Iterable of claim strings.

    Returns:
        A dict mapping each flag name (from :data:`FLAG_NAMES`) to the
        number of claims that fired it.  Includes zero counts for
        unseen flags so the result shape is stable across datasets.
    """
    counts: dict[str, int] = {name: 0 for name in FLAG_NAMES}
    for c in claims:
        for flag in annotate(c)["regex_flags"]:
            counts[flag] += 1
    return counts
