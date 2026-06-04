"""Offline unit tests for the evaluation metrics (no model / network)."""
import math

from metrics import (
    Counts,
    EvalSpan,
    ScoredExample,
    evaluate,
    f_beta,
    precision_recall,
    spans_match,
    tally_example,
)


def S(start, end, entity_type):
    """Shorthand to build an EvalSpan."""
    return EvalSpan(start, end, entity_type)


# --------------------------------------------------------------------------- #
# span matching
# --------------------------------------------------------------------------- #
def test_exact_match_requires_identical_offsets_and_type():
    assert spans_match(S(0, 4, "PERSON"), S(0, 4, "PERSON"), overlap=False)
    assert not spans_match(S(0, 4, "PERSON"), S(0, 5, "PERSON"), overlap=False)
    assert not spans_match(S(0, 4, "PERSON"), S(0, 4, "LOCATION"), overlap=False)


def test_overlap_match_requires_intersection_and_type():
    # Different boundaries but overlapping ranges + same type -> overlap match.
    assert spans_match(S(0, 10, "PERSON"), S(3, 7, "PERSON"), overlap=True)
    assert not spans_match(S(0, 10, "PERSON"), S(3, 7, "LOCATION"), overlap=True)
    # Touching but not overlapping (half-open) -> no match.
    assert not spans_match(S(0, 4, "PERSON"), S(4, 8, "PERSON"), overlap=True)


# --------------------------------------------------------------------------- #
# tally_example
# --------------------------------------------------------------------------- #
def test_tally_perfect_exact():
    gold = [S(0, 4, "PERSON"), S(10, 17, "EMAIL_ADDRESS")]
    pred = [S(0, 4, "PERSON"), S(10, 17, "EMAIL_ADDRESS")]
    counts = tally_example(gold, pred, overlap=False)
    assert counts["PERSON"] == Counts(tp=1, fp=0, fn=0)
    assert counts["EMAIL_ADDRESS"] == Counts(tp=1, fp=0, fn=0)


def test_tally_boundary_mismatch_exact_vs_overlap():
    gold = [S(0, 10, "PERSON")]
    pred = [S(2, 9, "PERSON")]  # off-by boundaries
    exact = tally_example(gold, pred, overlap=False)
    assert exact["PERSON"] == Counts(tp=0, fp=1, fn=1)
    overlap = tally_example(gold, pred, overlap=True)
    assert overlap["PERSON"] == Counts(tp=1, fp=0, fn=0)


def test_tally_false_positive_and_negative():
    gold = [S(0, 4, "PERSON"), S(5, 9, "LOCATION")]
    pred = [S(0, 4, "PERSON"), S(20, 24, "PERSON")]
    counts = tally_example(gold, pred, overlap=False)
    assert counts["PERSON"] == Counts(tp=1, fp=1, fn=0)
    assert counts["LOCATION"] == Counts(tp=0, fp=0, fn=1)


def test_tally_duplicate_pred_one_matches_rest_fp():
    gold = [S(0, 4, "PERSON")]
    pred = [S(0, 4, "PERSON"), S(0, 4, "PERSON")]
    counts = tally_example(gold, pred, overlap=False)
    assert counts["PERSON"] == Counts(tp=1, fp=1, fn=0)


# --------------------------------------------------------------------------- #
# precision / recall / f-beta
# --------------------------------------------------------------------------- #
def test_precision_recall_guards_zero():
    assert precision_recall(Counts(0, 0, 0)) == (0.0, 0.0)
    assert precision_recall(Counts(tp=1, fp=1, fn=3)) == (0.5, 0.25)


def test_f_beta_weights_recall_more_for_beta_2():
    precision, recall = 0.5, 1.0
    f1 = f_beta(precision, recall, 1.0)
    f2 = f_beta(precision, recall, 2.0)
    # With recall > precision, F2 (recall-weighted) exceeds F1.
    assert f2 > f1
    assert math.isclose(f1, 2 / 3, rel_tol=1e-9)
    assert math.isclose(f2, 0.8333333333, rel_tol=1e-6)


def test_f_beta_zero_when_no_signal():
    assert f_beta(0.0, 0.0, 1.0) == 0.0
    assert f_beta(0.0, 0.0, 2.0) == 0.0


# --------------------------------------------------------------------------- #
# evaluate (aggregation)
# --------------------------------------------------------------------------- #
def _examples():
    return [
        ScoredExample(
            language="de",
            gold=[S(0, 4, "PERSON"), S(10, 17, "EMAIL_ADDRESS")],
            pred=[S(0, 4, "PERSON"), S(10, 17, "EMAIL_ADDRESS")],
        ),
        ScoredExample(
            language="en",
            gold=[S(0, 5, "PERSON")],
            pred=[S(0, 4, "PERSON")],  # overlap but not exact
        ),
    ]


def test_evaluate_exact_micro_and_per_entity():
    result = evaluate(_examples())
    exact = result["modes"]["exact"]
    # 3 gold spans, 2 exact TP (the de example), the en PERSON is fp+fn.
    assert exact["micro"]["tp"] == 2
    assert exact["micro"]["fp"] == 1
    assert exact["micro"]["fn"] == 1
    assert exact["per_entity"]["EMAIL_ADDRESS"]["f1"] == 1.0
    assert exact["per_entity"]["PERSON"]["tp"] == 1


def test_evaluate_overlap_beats_exact():
    result = evaluate(_examples())
    exact_f1 = result["modes"]["exact"]["micro"]["f1"]
    overlap_f1 = result["modes"]["overlap"]["micro"]["f1"]
    # The en PERSON span matches under overlap, so overlap F1 is higher.
    assert overlap_f1 > exact_f1
    assert result["modes"]["overlap"]["micro"]["tp"] == 3


def test_evaluate_per_language():
    result = evaluate(_examples())
    per_lang = result["modes"]["exact"]["per_language"]
    assert set(per_lang) == {"de", "en"}
    assert per_lang["de"]["f1"] == 1.0  # perfect
    assert per_lang["en"]["tp"] == 0  # exact miss


def test_evaluate_reports_requested_betas():
    result = evaluate(_examples(), betas=(1.0, 2.0, 0.5))
    assert result["betas"] == [1.0, 2.0, 0.5]
    micro = result["modes"]["overlap"]["micro"]
    assert set(micro["fbeta"]) == {"1.0", "2.0", "0.5"}


def test_evaluate_empty_is_safe():
    result = evaluate([])
    assert result["examples"] == 0
    assert result["modes"]["exact"]["micro"]["f1"] == 0.0
    assert result["modes"]["exact"]["per_entity"] == {}
