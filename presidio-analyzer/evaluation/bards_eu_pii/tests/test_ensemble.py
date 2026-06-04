"""Offline tests for ensemble span combination (synthetic RecognizerResults)."""
from presidio_analyzer import RecognizerResult

from ensemble import dedup_spans, intersection_spans, union_spans


def R(entity_type, start, end, score=0.8):
    """Build a synthetic RecognizerResult."""
    return RecognizerResult(entity_type=entity_type, start=start, end=end, score=score)


def _triples(spans):
    """Reduce spans to a sorted list of (start, end, entity_type) tuples."""
    return sorted((s.start, s.end, s.entity_type) for s in spans)


# --------------------------------------------------------------------------- #
# union
# --------------------------------------------------------------------------- #
def test_union_keeps_all_non_duplicate_spans():
    sets = [
        [R("PERSON", 0, 5), R("LOCATION", 10, 15)],
        [R("EMAIL_ADDRESS", 20, 27)],
    ]
    assert _triples(union_spans(sets)) == [
        (0, 5, "PERSON"),
        (10, 15, "LOCATION"),
        (20, 27, "EMAIL_ADDRESS"),
    ]


def test_union_dedups_exact_duplicate_keeping_highest_score():
    sets = [[R("PERSON", 0, 5, 0.7)], [R("PERSON", 0, 5, 0.9)]]
    out = union_spans(sets)
    assert len(out) == 1
    assert out[0].score == 0.9


def test_union_dedups_overlapping_same_type():
    sets = [[R("PERSON", 0, 5, 0.7)], [R("PERSON", 1, 6, 0.9)]]
    out = union_spans(sets)
    assert len(out) == 1
    assert (out[0].start, out[0].end, out[0].score) == (1, 6, 0.9)


def test_union_keeps_different_entity_types_at_same_span():
    sets = [[R("PERSON", 0, 5)], [R("LOCATION", 0, 5)]]
    assert _triples(union_spans(sets)) == [(0, 5, "LOCATION"), (0, 5, "PERSON")]


def test_union_empty():
    assert union_spans([]) == []
    assert union_spans([[], []]) == []


# --------------------------------------------------------------------------- #
# intersection / agreement
# --------------------------------------------------------------------------- #
def test_intersection_keeps_only_overlapping_spans():
    sets = [
        [R("PERSON", 0, 5, 0.9), R("LOCATION", 10, 15)],
        [R("PERSON", 1, 6, 0.7)],
    ]
    out = intersection_spans(sets)
    # PERSON agrees (0-5 overlaps 1-6); LOCATION has no match in the other set.
    assert _triples(out) == [(0, 5, "PERSON")]


def test_intersection_drops_non_agreeing_entity_types():
    sets = [[R("PERSON", 0, 5)], [R("LOCATION", 0, 5)]]
    assert intersection_spans(sets) == []


def test_intersection_requires_all_sets_to_agree():
    sets = [
        [R("PERSON", 0, 5)],
        [R("PERSON", 1, 6)],
        [R("PERSON", 100, 105)],  # far away: no overlap with the others
    ]
    assert intersection_spans(sets) == []


def test_intersection_single_set_is_dedup():
    sets = [[R("PERSON", 0, 5, 0.5), R("PERSON", 1, 6, 0.9)]]
    out = intersection_spans(sets)
    assert len(out) == 1
    assert out[0].score == 0.9


def test_intersection_empty():
    assert intersection_spans([]) == []


# --------------------------------------------------------------------------- #
# determinism
# --------------------------------------------------------------------------- #
def test_dedup_is_order_independent():
    spans = [
        R("PERSON", 0, 5, 0.7),
        R("PERSON", 1, 6, 0.9),
        R("LOCATION", 10, 15, 0.5),
    ]
    out1 = dedup_spans(spans)
    out2 = dedup_spans(list(reversed(spans)))
    assert _triples(out1) == _triples(out2)
    persons = [s for s in out1 if s.entity_type == "PERSON"]
    assert len(persons) == 1 and persons[0].score == 0.9  # highest-score kept


def test_union_is_order_independent():
    s1 = [R("PERSON", 0, 5, 0.7)]
    s2 = [R("PERSON", 0, 5, 0.9)]
    assert _triples(union_spans([s1, s2])) == _triples(union_spans([s2, s1]))
    assert union_spans([s1, s2])[0].score == 0.9
    assert union_spans([s2, s1])[0].score == 0.9
