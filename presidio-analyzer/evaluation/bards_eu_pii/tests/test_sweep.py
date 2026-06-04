"""Offline tests for threshold-sweep selection logic (mocked summaries only)."""
from sweep import (
    ZERO_SUMMARY,
    micro_summary,
    recommend_thresholds,
    summary_row,
    threshold_entry,
)


def row(threshold, precision, recall, f1, f2):
    """Build a mock sweep row."""
    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f2": f2,
    }


# A typical sweep: precision rises and recall falls as the threshold rises.
SWEEP_ROWS = [
    row(0.10, 0.40, 0.99, 0.57, 0.80),
    row(0.20, 0.55, 0.95, 0.70, 0.88),  # highest F2  -> high_recall
    row(0.30, 0.70, 0.88, 0.78, 0.84),
    row(0.35, 0.88, 0.85, 0.80, 0.82),  # highest precision w/ recall>=0.8
    row(0.40, 0.82, 0.84, 0.83, 0.835),  # highest F1  -> balanced
    row(0.50, 0.95, 0.60, 0.74, 0.65),  # highest precision overall but recall<0.8
]


# --------------------------------------------------------------------------- #
# recommend_thresholds
# --------------------------------------------------------------------------- #
def test_balanced_picks_highest_f1():
    rec = recommend_thresholds(SWEEP_ROWS)
    assert rec["balanced"]["threshold"] == 0.40
    assert rec["balanced"]["criterion"] == "highest F1"


def test_high_recall_picks_highest_f2():
    rec = recommend_thresholds(SWEEP_ROWS)
    assert rec["high_recall"]["threshold"] == 0.20
    assert rec["high_recall"]["criterion"] == "highest F2"


def test_high_precision_respects_recall_floor():
    rec = recommend_thresholds(SWEEP_ROWS, min_recall_for_high_precision=0.80)
    # 0.50 has the best precision (0.95) but recall 0.60 < floor, so it is excluded;
    # the best eligible precision is 0.35 (0.88, recall 0.85).
    assert rec["high_precision"]["threshold"] == 0.35
    assert rec["high_precision"]["precision"] == 0.88


def test_high_precision_none_when_floor_unmet():
    rec = recommend_thresholds(SWEEP_ROWS, min_recall_for_high_precision=0.999)
    assert rec["high_precision"] is None


def test_high_precision_floor_is_echoed():
    rec = recommend_thresholds(SWEEP_ROWS, min_recall_for_high_precision=0.7)
    assert rec["min_recall_for_high_precision"] == 0.7
    # With a 0.7 floor, 0.50 (recall 0.60) is still excluded but everything
    # else qualifies; the max precision among eligible is still 0.35.
    assert rec["high_precision"]["threshold"] == 0.35


def test_balanced_tie_break_prefers_lower_threshold():
    rows = [
        row(0.30, 0.80, 0.85, 0.80, 0.82),
        row(0.50, 0.85, 0.85, 0.80, 0.82),  # identical f1 and recall
    ]
    rec = recommend_thresholds(rows)
    assert rec["balanced"]["threshold"] == 0.30


def test_high_precision_tie_break_prefers_higher_threshold():
    rows = [
        row(0.30, 0.90, 0.85, 0.80, 0.80),
        row(0.60, 0.90, 0.85, 0.80, 0.80),  # identical precision and recall
    ]
    rec = recommend_thresholds(rows)
    assert rec["high_precision"]["threshold"] == 0.60


def test_empty_rows_return_none_profiles():
    rec = recommend_thresholds([])
    assert rec["balanced"] is None
    assert rec["high_recall"] is None
    assert rec["high_precision"] is None


def test_recommendation_does_not_mutate_rows():
    rows = [row(0.4, 0.8, 0.8, 0.8, 0.8)]
    recommend_thresholds(rows)
    assert "criterion" not in rows[0]  # selection annotated a copy, not the row


# --------------------------------------------------------------------------- #
# summary helpers
# --------------------------------------------------------------------------- #
def test_micro_summary_extracts_f2_from_fbeta():
    metrics = {
        "precision": 0.5,
        "recall": 0.4,
        "f1": 0.44,
        "fbeta": {"1.0": 0.44, "2.0": 0.41},
    }
    assert micro_summary(metrics) == {
        "precision": 0.5,
        "recall": 0.4,
        "f1": 0.44,
        "f2": 0.41,
    }


def test_micro_summary_none_is_zero():
    assert micro_summary(None) == ZERO_SUMMARY
    assert micro_summary({}) == ZERO_SUMMARY


def test_summary_row_includes_threshold():
    metrics = {"precision": 0.6, "recall": 0.5, "f1": 0.55, "fbeta": {"2.0": 0.52}}
    assert summary_row(0.4, metrics) == {
        "threshold": 0.4,
        "precision": 0.6,
        "recall": 0.5,
        "f1": 0.55,
        "f2": 0.52,
    }


FAKE_RESULT = {
    "modes": {
        "exact": {
            "micro": {"precision": 0.6, "recall": 0.5, "f1": 0.55, "fbeta": {"2.0": 0.52}},
            "per_entity": {
                "PERSON": {"precision": 0.7, "recall": 0.6, "f1": 0.65, "fbeta": {"2.0": 0.62}}
            },
        },
        "overlap": {
            "micro": {"precision": 0.8, "recall": 0.7, "f1": 0.75, "fbeta": {"2.0": 0.72}},
            "per_entity": {
                "PERSON": {"precision": 0.9, "recall": 0.85, "f1": 0.87, "fbeta": {"2.0": 0.86}}
            },
        },
    }
}


def test_threshold_entry_micro():
    entry = threshold_entry(0.4, FAKE_RESULT)
    assert entry["threshold"] == 0.4
    assert entry["exact"]["precision"] == 0.6
    assert entry["overlap"]["f2"] == 0.72


def test_threshold_entry_for_entity():
    entry = threshold_entry(0.4, FAKE_RESULT, entity="PERSON")
    assert entry["overlap"]["precision"] == 0.9
    assert entry["exact"]["f1"] == 0.65


def test_threshold_entry_missing_entity_is_zero():
    entry = threshold_entry(0.4, FAKE_RESULT, entity="LOCATION")
    assert entry["exact"] == ZERO_SUMMARY
    assert entry["overlap"] == ZERO_SUMMARY
