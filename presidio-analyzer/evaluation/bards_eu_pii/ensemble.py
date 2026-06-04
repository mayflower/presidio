"""Span-combination logic for ensembling NER backends in the evaluation harness.

Pure standard library and duck-typed: the functions accept any objects exposing
``start``, ``end``, ``entity_type`` and (optionally) ``score`` — Presidio
``RecognizerResult`` objects in the CLI, or lightweight stand-ins in tests — so
the ensemble logic unit-tests offline without importing Presidio or a model.
"""

from __future__ import annotations

from typing import List, Sequence


def _overlaps(first, second) -> bool:
    """Return whether two spans share an entity type and overlap in characters."""
    return (
        first.entity_type == second.entity_type
        and first.start < second.end
        and second.start < first.end
    )


def _dedup_sort_key(span):
    """Return a deterministic ordering key; the smallest sorts first.

    Prefers higher score, then earlier start, then a longer span, then entity
    type, so duplicate removal keeps a single stable representative regardless of
    input order.
    """
    score = getattr(span, "score", 0.0) or 0.0
    return (-score, span.start, -(span.end - span.start), span.entity_type)


def dedup_spans(spans: Sequence) -> List:
    """Remove overlapping same-entity-type duplicates, deterministically.

    Within a group of mutually overlapping same-type spans the one that sorts
    first by :func:`_dedup_sort_key` is kept and the rest are dropped. The result
    is ordered by ``(start, end, entity_type)`` so it does not depend on input
    order.
    """
    kept: List = []
    for span in sorted(spans, key=_dedup_sort_key):
        if not any(_overlaps(span, keep) for keep in kept):
            kept.append(span)
    return sorted(kept, key=lambda s: (s.start, s.end, s.entity_type))


def union_spans(span_sets: Sequence[Sequence]) -> List:
    """Combine every backend's spans, dropping overlapping same-type duplicates."""
    flattened = [span for span_set in span_sets for span in span_set]
    return dedup_spans(flattened)


def intersection_spans(span_sets: Sequence[Sequence]) -> List:
    """Keep only spans the backends agree on (overlap + same type in every set).

    A span is kept when it overlaps a same-type span in every *other* set;
    agreeing spans are then collapsed to one representative each. With a single
    set this is just :func:`dedup_spans`.
    """
    sets = [list(span_set) for span_set in span_sets]
    if not sets:
        return []
    if len(sets) == 1:
        return dedup_spans(sets[0])
    agreed: List = []
    for index, span_set in enumerate(sets):
        others = [other for position, other in enumerate(sets) if position != index]
        for span in span_set:
            if all(any(_overlaps(span, o) for o in other) for other in others):
                agreed.append(span)
    return dedup_spans(agreed)
