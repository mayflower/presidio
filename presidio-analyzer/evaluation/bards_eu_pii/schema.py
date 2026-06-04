"""JSONL schema and loader for Bards EU-PII evaluation data.

Each line of the dataset is one example::

    {"id": "example-1", "language": "de",
     "text": "Kontaktieren Sie Max Mueller unter max@example.de",
     "spans": [{"start": 17, "end": 28, "entity_type": "PERSON"},
               {"start": 35, "end": 50, "entity_type": "EMAIL_ADDRESS"}]}

Pure standard library so it stays offline and importable without Presidio.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Union

#: Required keys on every example and span object.
_EXAMPLE_KEYS = ("id", "language", "text", "spans")
_SPAN_KEYS = ("start", "end", "entity_type")


@dataclass(frozen=True)
class Span:
    """A labeled character span within an example's text."""

    start: int
    end: int
    entity_type: str


@dataclass(frozen=True)
class EvalExample:
    """A single evaluation example: an id, language, text and gold spans."""

    id: str
    language: str
    text: str
    spans: List[Span]


def _parse_span(raw: Dict[str, Any], example_id: str, text_length: int) -> Span:
    """Parse and validate one span dict, raising ``ValueError`` on bad data."""
    for key in _SPAN_KEYS:
        if key not in raw:
            raise ValueError(f"example {example_id!r}: span missing {key!r}: {raw!r}")
    start, end, entity_type = raw["start"], raw["end"], raw["entity_type"]
    if not isinstance(start, int) or not isinstance(end, int):
        raise ValueError(
            f"example {example_id!r}: span offsets must be integers: {raw!r}"
        )
    if start < 0 or end < start or end > text_length:
        raise ValueError(
            f"example {example_id!r}: span offsets out of range "
            f"(text length {text_length}): {raw!r}"
        )
    if not isinstance(entity_type, str) or not entity_type:
        raise ValueError(
            f"example {example_id!r}: span entity_type must be a non-empty string: "
            f"{raw!r}"
        )
    return Span(start=start, end=end, entity_type=entity_type)


def parse_example(raw: Dict[str, Any]) -> EvalExample:
    """Parse and validate one example dict into an ``EvalExample``."""
    for key in _EXAMPLE_KEYS:
        if key not in raw:
            raise ValueError(f"example missing required field {key!r}: {raw!r}")
    text = raw["text"]
    if not isinstance(text, str):
        raise ValueError(f"example {raw['id']!r}: text must be a string")
    spans = [_parse_span(span, raw["id"], len(text)) for span in raw["spans"]]
    return EvalExample(
        id=str(raw["id"]),
        language=str(raw["language"]),
        text=text,
        spans=spans,
    )


def iter_jsonl(path: Union[str, Path]) -> Iterator[EvalExample]:
    """Yield ``EvalExample`` objects from a JSONL file, skipping blank lines."""
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                raw = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            yield parse_example(raw)


def load_jsonl(path: Union[str, Path]) -> List[EvalExample]:
    """Load all examples from a JSONL file into a list."""
    return list(iter_jsonl(path))
