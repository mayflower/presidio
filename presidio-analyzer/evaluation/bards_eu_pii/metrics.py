"""Offline span-matching metrics for Bards EU-PII evaluation.

Pure Python (standard library only) so the metrics can be unit-tested without
importing Presidio, downloading a model, or touching the network. Computes
exact-span and overlap-span precision / recall / F1 (and F-beta), broken down per
entity type and per language, plus micro averages.
"""

from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Tuple

#: One labeled span. ``start``/``end`` are half-open character offsets.
EvalSpan = namedtuple("EvalSpan", ["start", "end", "entity_type"])

#: One scored example: its language plus gold and predicted spans.
ScoredExample = namedtuple("ScoredExample", ["language", "gold", "pred"])

#: Default F-beta values reported alongside F1 (F1 and recall-weighted F2).
DEFAULT_BETAS: Tuple[float, ...] = (1.0, 2.0)


@dataclass
class Counts:
    """True-positive / false-positive / false-negative tallies for one bucket."""

    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, other: "Counts") -> None:
        """Accumulate another tally into this one in place."""
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn


def spans_match(gold: EvalSpan, pred: EvalSpan, overlap: bool) -> bool:
    """Return whether a gold and predicted span count as a match.

    Entity types must be equal. For exact matching the offsets must be identical;
    for overlap matching the half-open character ranges must intersect.
    """
    if gold.entity_type != pred.entity_type:
        return False
    if overlap:
        return gold.start < pred.end and pred.start < gold.end
    return gold.start == pred.start and gold.end == pred.end


def tally_example(
    gold: Iterable[EvalSpan], pred: Iterable[EvalSpan], overlap: bool
) -> Dict[str, Counts]:
    """Greedily match one example's spans and return per-entity-type counts.

    Each predicted span matches at most one gold span (the first available, in
    order) and vice versa. Unmatched gold spans are false negatives; unmatched
    predicted spans are false positives.
    """
    gold = list(gold)
    pred = list(pred)
    counts: Dict[str, Counts] = {}

    def bucket(entity_type: str) -> Counts:
        return counts.setdefault(entity_type, Counts())

    matched_pred = set()
    for gold_span in gold:
        hit = False
        for index, pred_span in enumerate(pred):
            if index in matched_pred:
                continue
            if spans_match(gold_span, pred_span, overlap):
                matched_pred.add(index)
                bucket(gold_span.entity_type).tp += 1
                hit = True
                break
        if not hit:
            bucket(gold_span.entity_type).fn += 1
    for index, pred_span in enumerate(pred):
        if index not in matched_pred:
            bucket(pred_span.entity_type).fp += 1
    return counts


def precision_recall(counts: Counts) -> Tuple[float, float]:
    """Return (precision, recall) for a tally, guarding zero denominators."""
    predicted = counts.tp + counts.fp
    support = counts.tp + counts.fn
    precision = counts.tp / predicted if predicted else 0.0
    recall = counts.tp / support if support else 0.0
    return precision, recall


def f_beta(precision: float, recall: float, beta: float) -> float:
    """Return the F-beta score; beta > 1 weights recall, beta < 1 precision."""
    if precision == 0.0 and recall == 0.0:
        return 0.0
    beta_sq = beta * beta
    denominator = beta_sq * precision + recall
    if denominator == 0.0:
        return 0.0
    return (1.0 + beta_sq) * precision * recall / denominator


def _beta_key(beta: float) -> str:
    """Return a stable string key for a beta value (e.g. 2.0 -> ``'2.0'``)."""
    return str(float(beta))


def metrics_for_counts(
    counts: Counts, betas: Iterable[float] = DEFAULT_BETAS, ndigits: int = 6
) -> Dict[str, Any]:
    """Return a rounded metrics dict (P/R/F1/F-beta and raw counts) for a tally."""
    precision, recall = precision_recall(counts)
    return {
        "precision": round(precision, ndigits),
        "recall": round(recall, ndigits),
        "f1": round(f_beta(precision, recall, 1.0), ndigits),
        "fbeta": {
            _beta_key(beta): round(f_beta(precision, recall, beta), ndigits)
            for beta in betas
        },
        "tp": counts.tp,
        "fp": counts.fp,
        "fn": counts.fn,
        "support": counts.tp + counts.fn,
        "predicted": counts.tp + counts.fp,
    }


def _score_mode(
    examples: Iterable[ScoredExample],
    overlap: bool,
    betas: Iterable[float],
    ndigits: int,
) -> Dict[str, Any]:
    """Aggregate per-entity, per-language and micro metrics for one match mode."""
    per_entity: Dict[str, Counts] = {}
    per_language: Dict[str, Counts] = {}
    micro = Counts()
    for example in examples:
        counts = tally_example(example.gold, example.pred, overlap)
        for entity_type, entity_counts in counts.items():
            per_entity.setdefault(entity_type, Counts()).add(entity_counts)
            per_language.setdefault(example.language, Counts()).add(entity_counts)
            micro.add(entity_counts)
    return {
        "micro": metrics_for_counts(micro, betas, ndigits),
        "per_entity": {
            entity_type: metrics_for_counts(per_entity[entity_type], betas, ndigits)
            for entity_type in sorted(per_entity)
        },
        "per_language": {
            language: metrics_for_counts(per_language[language], betas, ndigits)
            for language in sorted(per_language)
        },
    }


def evaluate(
    examples: Iterable[ScoredExample],
    betas: Iterable[float] = DEFAULT_BETAS,
    ndigits: int = 6,
) -> Dict[str, Any]:
    """Evaluate scored examples, returning exact and overlap metrics.

    The returned dict is JSON-serializable and deterministic (sorted keys,
    rounded floats), so it is stable enough to commit or diff in CI.
    """
    examples = list(examples)
    betas = tuple(float(beta) for beta in betas)
    return {
        "examples": len(examples),
        "betas": list(betas),
        "modes": {
            "exact": _score_mode(examples, False, betas, ndigits),
            "overlap": _score_mode(examples, True, betas, ndigits),
        },
    }
