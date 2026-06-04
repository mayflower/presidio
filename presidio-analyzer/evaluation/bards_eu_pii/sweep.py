"""Threshold-sweep selection logic for the Bards EU-PII evaluation harness.

Pure standard library: given per-threshold metric summaries it recommends
``balanced`` / ``high_recall`` / ``high_precision`` thresholds, so the selection
logic can be unit-tested without any model inference. The CLI in
``evaluate_bards_eu_pii.py`` builds the rows from real model runs; the functions
here only read plain dicts.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

#: A zero-valued metric summary (used when an entity has no spans at a threshold).
ZERO_SUMMARY: Dict[str, float] = {
    "precision": 0.0,
    "recall": 0.0,
    "f1": 0.0,
    "f2": 0.0,
}


def micro_summary(metrics: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """Flatten an ``evaluate`` metrics entry into precision / recall / F1 / F2.

    Returns a zero summary when ``metrics`` is falsy (e.g. an entity that had no
    spans at a given threshold).
    """
    if not metrics:
        return dict(ZERO_SUMMARY)
    return {
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "f2": metrics.get("fbeta", {}).get("2.0", 0.0),
    }


def summary_row(
    threshold: float, metrics: Optional[Dict[str, Any]]
) -> Dict[str, float]:
    """Build one sweep row: the threshold plus flattened P / R / F1 / F2."""
    row: Dict[str, float] = {"threshold": threshold}
    row.update(micro_summary(metrics))
    return row


def threshold_entry(
    threshold: float, evaluate_result: Dict[str, Any], entity: Optional[str] = None
) -> Dict[str, Any]:
    """Build a per-threshold output entry with exact and overlap summaries.

    When ``entity`` is given, the per-entity metrics are used instead of micro
    (missing entities fall back to a zero summary).
    """
    entry: Dict[str, Any] = {"threshold": threshold}
    for mode in ("exact", "overlap"):
        mode_metrics = evaluate_result["modes"][mode]
        if entity is None:
            metrics = mode_metrics["micro"]
        else:
            metrics = mode_metrics["per_entity"].get(entity)
        entry[mode] = micro_summary(metrics)
    return entry


def _select(
    rows: List[Dict[str, float]], metric: str, prefer_higher_threshold: bool
) -> Dict[str, float]:
    """Return the row maximizing ``metric`` with deterministic tie-breaks.

    Ties are broken first by higher recall, then by the threshold: lower when
    ``prefer_higher_threshold`` is false (recall-leaning profiles), higher when
    true (the precision-leaning profile).
    """
    sign = 1.0 if prefer_higher_threshold else -1.0
    return max(rows, key=lambda r: (r[metric], r["recall"], sign * r["threshold"]))


def _labeled(row: Dict[str, float], criterion: str) -> Dict[str, Any]:
    """Return a copy of ``row`` annotated with the selection criterion."""
    labeled: Dict[str, Any] = dict(row)
    labeled["criterion"] = criterion
    return labeled


def recommend_thresholds(
    rows: List[Dict[str, float]],
    min_recall_for_high_precision: float = 0.8,
) -> Dict[str, Any]:
    """Recommend balanced / high-recall / high-precision thresholds from rows.

    Each row must carry ``threshold``, ``precision``, ``recall``, ``f1`` and
    ``f2``. Selection rules:

    - ``balanced``: highest F1 (ties: higher recall, then lower threshold).
    - ``high_recall``: highest F2 (ties: higher recall, then lower threshold).
    - ``high_precision``: highest precision among rows whose recall is at least
      ``min_recall_for_high_precision`` (ties: higher recall, then higher
      threshold); ``None`` when no row clears the recall floor.
    """
    recommendations: Dict[str, Any] = {
        "min_recall_for_high_precision": min_recall_for_high_precision,
        "balanced": None,
        "high_recall": None,
        "high_precision": None,
    }
    if not rows:
        return recommendations

    recommendations["balanced"] = _labeled(
        _select(rows, "f1", prefer_higher_threshold=False), "highest F1"
    )
    recommendations["high_recall"] = _labeled(
        _select(rows, "f2", prefer_higher_threshold=False), "highest F2"
    )
    eligible = [
        row for row in rows if row["recall"] >= min_recall_for_high_precision
    ]
    if eligible:
        recommendations["high_precision"] = _labeled(
            _select(eligible, "precision", prefer_higher_threshold=True),
            f"highest precision with recall >= {min_recall_for_high_precision}",
        )
    return recommendations
