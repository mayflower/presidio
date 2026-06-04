"""Experimental, eval-only text normalization for robustness evaluation.

These helpers are local to the evaluation harness and are **not** used by the
Presidio analyzer at runtime. They let you measure how much OCR-style noise and
spaced-out formatting light preprocessing would recover, without changing
production behavior. Pure standard library (``re``), so they unit-test offline.

Because some normalizations change the text length (collapsing a spaced e-mail),
:func:`normalize_text` returns the normalized text together with a ``SpanMap``
that maps each output character back to the original offsets; :func:`map_span`
uses it to translate a predicted span back to original coordinates so it can be
scored against gold spans authored on the original text.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

#: Leet / OCR digit -> letter substitutions, applied only between ASCII letters.
_LEET_DIGIT_MAP = {
    "0": "o",
    "1": "i",
    "3": "e",
    "4": "a",
    "5": "s",
    "7": "t",
    "8": "b",
    "9": "g",
}

#: Matches an e-mail whose parts are separated by stray spaces, e.g.
#: ``john . smith @ example . com``.
_SPACED_EMAIL = re.compile(
    r"[A-Za-z0-9_%+\-]+(?:\s*\.\s*[A-Za-z0-9_%+\-]+)*"
    r"\s*@\s*"
    r"[A-Za-z0-9\-]+(?:\s*\.\s*[A-Za-z0-9\-]+)+"
)

#: One ``(orig_start, orig_end)`` entry per normalized character.
SpanMap = List[Tuple[int, int]]


def _is_ascii_letter(char: str) -> bool:
    """Return whether ``char`` is a single ASCII letter."""
    return len(char) == 1 and char.isascii() and char.isalpha()


def normalize_ocr_noise(text: str) -> str:
    """Map leet / OCR digits back to letters when flanked by ASCII letters.

    Only a digit with an ASCII letter on *both* sides is substituted (e.g.
    ``J0hn`` -> ``John``, ``Sm1th`` -> ``Smith``), which leaves pure-numeric
    tokens such as phone numbers and account IDs untouched. Length-preserving, so
    offsets are unchanged.
    """
    if not text:
        return text
    chars = list(text)
    length = len(chars)
    for index in range(length):
        replacement = _LEET_DIGIT_MAP.get(chars[index])
        if replacement is None:
            continue
        left = text[index - 1] if index > 0 else ""
        right = text[index + 1] if index + 1 < length else ""
        if _is_ascii_letter(left) and _is_ascii_letter(right):
            chars[index] = replacement
    return "".join(chars)


def _collapse_spaced_emails(text: str) -> Tuple[str, List[int]]:
    """Collapse whitespace inside spaced e-mails; return (text, kept_indices)."""
    drop = set()
    for match in _SPACED_EMAIL.finditer(text):
        for index in range(match.start(), match.end()):
            if text[index].isspace():
                drop.add(index)
    if not drop:
        return text, list(range(len(text)))
    out_chars = []
    kept = []
    for index, char in enumerate(text):
        if index in drop:
            continue
        out_chars.append(char)
        kept.append(index)
    return "".join(out_chars), kept


def normalize_text(
    text: str, ocr: bool = False, spaced_email: bool = False
) -> Tuple[str, SpanMap]:
    """Normalize ``text`` and return it with a map back to original offsets.

    OCR substitution runs first (length-preserving); spaced-e-mail collapsing
    runs second (length-changing). The returned ``SpanMap`` has one
    ``(orig_start, orig_end)`` entry per output character.
    """
    work = normalize_ocr_noise(text) if ocr else text
    if spaced_email:
        out, kept = _collapse_spaced_emails(work)
    else:
        out, kept = work, list(range(len(work)))
    span_map: SpanMap = [(kept[i], kept[i] + 1) for i in range(len(out))]
    return out, span_map


def map_span(start: int, end: int, span_map: Optional[SpanMap]) -> Tuple[int, int]:
    """Map a normalized ``[start, end)`` span back to original offsets.

    When ``span_map`` is ``None`` (no length-changing normalization) the offsets
    are returned unchanged. Out-of-range inputs are clamped defensively.
    """
    if span_map is None or start >= end:
        return start, end
    start = max(0, min(start, len(span_map) - 1))
    end = max(start + 1, min(end, len(span_map)))
    return span_map[start][0], span_map[end - 1][1]
