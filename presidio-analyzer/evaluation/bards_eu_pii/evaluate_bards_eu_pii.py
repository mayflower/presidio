r"""CLI to evaluate ``BardsEuPiiRecognizer`` quality on a local JSONL dataset.

Optional and never imported by the unit tests: Presidio and the model are
imported lazily, only when predictions are actually produced, so ``--help`` and
the metrics import stay light and offline.

Example::

    python evaluate_bards_eu_pii.py \
        --input sample_data.jsonl \
        --languages de,en,fr,it \
        --mode hybrid \
        --threshold 0.4 \
        --output results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ensemble import intersection_spans, union_spans
from label_maps import to_eval_bucket
from metrics import EvalSpan, ScoredExample, evaluate
from preprocess import map_span, normalize_text
from schema import EvalExample, load_jsonl
from sweep import recommend_thresholds, summary_row, threshold_entry

#: Evaluation buckets owned by the deterministic recognizers in hybrid mode.
#: In a multi-backend comparison the NER backends defer these to the
#: deterministic layer so the comparison is about free-text NER quality.
_STRUCTURED_EVAL_BUCKETS = frozenset(
    {"EMAIL_ADDRESS", "PHONE_NUMBER", "IP_ADDRESS", "CREDIT_CARD", "URL", "IBAN_CODE"}
)


def _split_csv(value: Optional[str]) -> Optional[List[str]]:
    """Split a comma-separated option into a list, or return ``None``."""
    if not value:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


def _parse_floats(value: Optional[str]) -> Optional[List[float]]:
    """Parse a comma-separated list of floats into a sorted list, or ``None``."""
    items = _split_csv(value)
    if not items:
        return None
    try:
        return sorted(float(item) for item in items)
    except ValueError as exc:
        raise SystemExit(f"invalid float in '{value}': {exc}") from exc


def _load_json_arg(value: Optional[str], name: str) -> Optional[Any]:
    """Parse a JSON-valued CLI option, raising a clear error on bad JSON."""
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--{name}: invalid JSON: {exc}") from exc


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate BardsEuPiiRecognizer on a local JSONL dataset."
    )
    parser.add_argument(
        "--input", required=True, help="Path to the evaluation JSONL file."
    )
    parser.add_argument(
        "--languages",
        default=None,
        help="Comma-separated languages to keep (default: all in the file).",
    )
    parser.add_argument(
        "--mode",
        choices=("standard", "hybrid"),
        default="standard",
        help="standard = BardsEuPiiRecognizer; hybrid = .hybrid() + deterministic.",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.4, help="Global confidence threshold."
    )
    parser.add_argument(
        "--entities",
        default=None,
        help="Comma-separated evaluation buckets to restrict to (gold and pred).",
    )
    parser.add_argument(
        "--mapping-profile",
        default=None,
        help="Named mapping profile (presidio_standard, gdpr_sensitive, ...).",
    )
    parser.add_argument(
        "--labels-to-ignore",
        default=None,
        help="Comma-separated model labels to drop before mapping.",
    )
    parser.add_argument(
        "--thresholds-by-entity",
        default=None,
        help='JSON object of per-entity thresholds, e.g. \'{"PERSON": 0.3}\'.',
    )
    parser.add_argument(
        "--thresholds-by-language",
        default=None,
        help='JSON object of per-language/entity thresholds.',
    )
    parser.add_argument(
        "--label-map",
        default=None,
        help="JSON object overriding the default prediction->bucket map.",
    )
    parser.add_argument(
        "--betas",
        default="1.0,2.0",
        help="Comma-separated F-beta values to report (default: 1.0,2.0).",
    )
    parser.add_argument(
        "--threshold-sweep",
        default=None,
        help=(
            "Comma-separated thresholds to sweep, e.g. 0.1,0.2,0.3,0.4,0.5. "
            "Evaluates once per threshold and recommends balanced / high_recall "
            "/ high_precision thresholds. Replaces --threshold when set."
        ),
    )
    parser.add_argument(
        "--sweep-entity",
        default=None,
        help="Also report/optimize this evaluation bucket separately in a sweep.",
    )
    parser.add_argument(
        "--sweep-mode",
        choices=("exact", "overlap"),
        default="overlap",
        help="Match mode the sweep recommendations optimize (default: overlap).",
    )
    parser.add_argument(
        "--min-recall-for-high-precision",
        type=float,
        default=0.80,
        help="Recall floor for the high_precision recommendation (default: 0.80).",
    )
    parser.add_argument(
        "--normalize-ocr-noise",
        action="store_true",
        help=(
            "EXPERIMENTAL (eval-only): map leet/OCR digits back to letters "
            "between ASCII letters (J0hn -> John) before analysis."
        ),
    )
    parser.add_argument(
        "--normalize-spaced-email",
        action="store_true",
        help=(
            "EXPERIMENTAL (eval-only): collapse whitespace inside spaced e-mails "
            "(john . smith @ example . com) before analysis."
        ),
    )
    parser.add_argument(
        "--backend",
        action="append",
        choices=("bards", "gliner", "huggingface"),
        default=None,
        help=(
            "NER backend(s) to evaluate; repeat to compare several. Defaults to "
            "'bards'. 'gliner'/'huggingface' are optional and require their "
            "extras/models. Multiple backends are combined via --ensemble."
        ),
    )
    parser.add_argument(
        "--ensemble",
        choices=("union", "intersection"),
        default=None,
        help=(
            "How to combine multiple backends: 'union' (all non-duplicate spans) "
            "or 'intersection' (only spans the backends agree on). Defaults to "
            "'union' when more than one backend is given."
        ),
    )
    parser.add_argument(
        "--gliner-model",
        default=None,
        help="GLiNER model id for --backend gliner (default: the recognizer's own).",
    )
    parser.add_argument(
        "--huggingface-model",
        default=None,
        help="HuggingFace model id; required for --backend huggingface.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write the results JSON to (also printed as a summary).",
    )
    return parser.parse_args(argv)


def _build_recognizer(language: str, args: argparse.Namespace):
    """Build a ``BardsEuPiiRecognizer`` for one language from the CLI args."""
    from presidio_analyzer.predefined_recognizers import BardsEuPiiRecognizer

    kwargs: Dict[str, Any] = {
        "supported_language": language,
        "threshold": args.threshold,
    }
    if args.mapping_profile:
        kwargs["mapping_profile"] = args.mapping_profile
    labels_to_ignore = _split_csv(args.labels_to_ignore)
    if labels_to_ignore:
        kwargs["labels_to_ignore"] = labels_to_ignore
    by_entity = _load_json_arg(args.thresholds_by_entity, "thresholds-by-entity")
    if by_entity:
        kwargs["thresholds_by_entity"] = by_entity
    by_language = _load_json_arg(args.thresholds_by_language, "thresholds-by-language")
    if by_language:
        kwargs["thresholds_by_language"] = by_language

    if args.mode == "hybrid":
        return BardsEuPiiRecognizer.hybrid(**kwargs)
    return BardsEuPiiRecognizer(**kwargs)


def _build_deterministic_recognizers() -> list:
    """Build the deterministic recognizers that own structured PII in hybrid mode.

    Recognizers whose optional dependencies are missing are skipped with a
    warning so the harness still runs (e.g. ``PhoneRecognizer`` needs
    ``phonenumbers``).
    """
    from presidio_analyzer import predefined_recognizers as pr

    recognizers = []
    for class_name in (
        "EmailRecognizer",
        "CreditCardRecognizer",
        "IpRecognizer",
        "UrlRecognizer",
        "IbanRecognizer",
        "PhoneRecognizer",
    ):
        factory = getattr(pr, class_name, None)
        if factory is None:
            continue
        try:
            recognizers.append(factory())
        except Exception as exc:  # noqa: BLE001 - best-effort, reported to stderr
            print(f"warning: skipping {class_name}: {exc}", file=sys.stderr)
    return recognizers


def _normalize_for_eval(text: str, args: argparse.Namespace):
    """Apply the experimental eval-only normalizations to ``text``.

    Returns ``(analysis_text, span_map)``. ``span_map`` is ``None`` when no
    length-changing normalization runs, meaning predicted offsets already align
    with the original text (OCR substitution is length-preserving).
    """
    ocr = args.normalize_ocr_noise
    spaced = args.normalize_spaced_email
    if not ocr and not spaced:
        return text, None
    normalized, span_map = normalize_text(text, ocr=ocr, spaced_email=spaced)
    if not spaced:
        return normalized, None
    return normalized, span_map


def _predict_spans(
    example: EvalExample,
    bards,
    deterministic: list,
    label_map: Optional[Dict[str, str]],
    keep: Optional[set],
    args: argparse.Namespace,
) -> List[EvalSpan]:
    """Run the recognizers on one example and return bucketed predicted spans."""
    text, span_map = _normalize_for_eval(example.text, args)
    results = list(bards.analyze(text, list(bards.supported_entities)))
    for recognizer in deterministic:
        results.extend(
            recognizer.analyze(
                text, list(recognizer.supported_entities), nlp_artifacts=None
            )
        )
    spans = []
    for result in results:
        bucket = to_eval_bucket(result.entity_type, label_map)
        if keep is not None and bucket not in keep:
            continue
        start, end = map_span(result.start, result.end, span_map)
        spans.append(EvalSpan(start, end, bucket))
    return spans


def _gold_spans(example: EvalExample, keep: Optional[set]) -> List[EvalSpan]:
    """Return an example's gold spans, optionally restricted to ``keep`` buckets."""
    return [
        EvalSpan(span.start, span.end, span.entity_type)
        for span in example.spans
        if keep is None or span.entity_type in keep
    ]


def _build_config(args: argparse.Namespace, languages, keep, betas) -> Dict[str, Any]:
    """Build a deterministic echo of the run configuration (no timestamps)."""
    sweep_thresholds = _parse_floats(args.threshold_sweep)
    config: Dict[str, Any] = {
        "input": str(args.input),
        "mode": args.mode,
        "threshold": None if sweep_thresholds else args.threshold,
        "languages": sorted(set(languages)) if languages else None,
        "entities": sorted(keep) if keep else None,
        "mapping_profile": args.mapping_profile,
        "labels_to_ignore": sorted(_split_csv(args.labels_to_ignore) or []) or None,
        "thresholds_by_entity": _load_json_arg(
            args.thresholds_by_entity, "thresholds-by-entity"
        ),
        "thresholds_by_language": _load_json_arg(
            args.thresholds_by_language, "thresholds-by-language"
        ),
        "betas": betas,
        "threshold_sweep": sweep_thresholds,
        "sweep_mode": args.sweep_mode if sweep_thresholds else None,
        "sweep_entity": args.sweep_entity if sweep_thresholds else None,
        "min_recall_for_high_precision": (
            args.min_recall_for_high_precision if sweep_thresholds else None
        ),
        "normalize_ocr_noise": args.normalize_ocr_noise,
        "normalize_spaced_email": args.normalize_spaced_email,
    }
    return config


def _format_summary(result: Dict[str, Any]) -> str:
    """Format a short human-readable summary of the micro metrics."""
    examples = result["metrics"]["examples"]
    mode_label = result["config"]["mode"]
    lines = [f"examples: {examples}  mode: {mode_label}"]
    for mode in ("exact", "overlap"):
        micro = result["metrics"]["modes"][mode]["micro"]
        lines.append(
            f"  {mode:7s} micro  P={micro['precision']:.3f}  "
            f"R={micro['recall']:.3f}  F1={micro['f1']:.3f}  "
            f"F2={micro['fbeta'].get('2.0', float('nan')):.3f}"
        )
    return "\n".join(lines)


def _run_single(
    examples: List[EvalExample],
    args: argparse.Namespace,
    keep: Optional[set],
    label_map: Optional[Dict[str, str]],
    betas: List[float],
    languages: Optional[List[str]],
) -> Dict[str, Any]:
    """Run a single-threshold evaluation and return the result dict."""
    deterministic = (
        _build_deterministic_recognizers() if args.mode == "hybrid" else []
    )
    recognizer_by_language: Dict[str, Any] = {}
    scored: List[ScoredExample] = []
    for example in examples:
        bards = recognizer_by_language.get(example.language)
        if bards is None:
            bards = _build_recognizer(example.language, args)
            recognizer_by_language[example.language] = bards
        pred = _predict_spans(example, bards, deterministic, label_map, keep, args)
        gold = _gold_spans(example, keep)
        scored.append(ScoredExample(example.language, gold, pred))

    return {
        "config": _build_config(args, languages, keep, betas),
        "metrics": evaluate(scored, betas=betas),
    }


def _build_sweep_recognizer(language: str, args: argparse.Namespace, floor: float):
    """Build a recognizer for a sweep: global threshold at ``floor``, no maps.

    The sweep optimizes the single *global* threshold, so per-entity/per-language
    threshold maps are intentionally not applied here; the recognizer is built at
    the lowest swept threshold and predictions are filtered by score afterwards.
    """
    from presidio_analyzer.predefined_recognizers import BardsEuPiiRecognizer

    kwargs: Dict[str, Any] = {"supported_language": language, "threshold": floor}
    if args.mapping_profile:
        kwargs["mapping_profile"] = args.mapping_profile
    labels_to_ignore = _split_csv(args.labels_to_ignore)
    if labels_to_ignore:
        kwargs["labels_to_ignore"] = labels_to_ignore

    if args.mode == "hybrid":
        return BardsEuPiiRecognizer.hybrid(**kwargs)
    return BardsEuPiiRecognizer(**kwargs)


def _bucket_span(
    result,
    label_map: Optional[Dict[str, str]],
    keep: Optional[set],
    span_map,
) -> Optional[EvalSpan]:
    """Map a recognizer result to a bucketed ``EvalSpan`` (or drop it)."""
    bucket = to_eval_bucket(result.entity_type, label_map)
    if keep is not None and bucket not in keep:
        return None
    start, end = map_span(result.start, result.end, span_map)
    return EvalSpan(start, end, bucket)


def _filter_and_bucket(
    model_results: list,
    deterministic_results: list,
    threshold: float,
    label_map: Optional[Dict[str, str]],
    keep: Optional[set],
    span_map,
) -> List[EvalSpan]:
    """Keep model results scoring >= ``threshold`` (deterministic kept always)."""
    spans: List[EvalSpan] = []
    for result in model_results:
        if result.score < threshold:
            continue
        span = _bucket_span(result, label_map, keep, span_map)
        if span is not None:
            spans.append(span)
    for result in deterministic_results:
        span = _bucket_span(result, label_map, keep, span_map)
        if span is not None:
            spans.append(span)
    return spans


def _run_sweep(
    examples: List[EvalExample],
    args: argparse.Namespace,
    keep: Optional[set],
    label_map: Optional[Dict[str, str]],
    betas: List[float],
    languages: Optional[List[str]],
    sweep_thresholds: List[float],
) -> Dict[str, Any]:
    """Run a threshold sweep and return the result dict with recommendations.

    The model runs once per example at the lowest swept threshold; each threshold
    then re-scores the cached predictions by filtering on prediction score, so
    "evaluate once per threshold" happens without re-running inference.
    """
    betas = sorted(set(betas) | {1.0, 2.0})  # F1 and F2 are always needed
    floor = min(sweep_thresholds)
    deterministic = (
        _build_deterministic_recognizers() if args.mode == "hybrid" else []
    )

    recognizer_by_language: Dict[str, Any] = {}
    cached = []
    for example in examples:
        bards = recognizer_by_language.get(example.language)
        if bards is None:
            bards = _build_sweep_recognizer(example.language, args, floor)
            recognizer_by_language[example.language] = bards
        text, span_map = _normalize_for_eval(example.text, args)
        model_results = list(bards.analyze(text, list(bards.supported_entities)))
        deterministic_results = []
        for recognizer in deterministic:
            deterministic_results.extend(
                recognizer.analyze(
                    text,
                    list(recognizer.supported_entities),
                    nlp_artifacts=None,
                )
            )
        cached.append(
            (
                example.language,
                _gold_spans(example, keep),
                model_results,
                deterministic_results,
                span_map,
            )
        )

    threshold_entries: List[Dict[str, Any]] = []
    micro_rows: List[Dict[str, float]] = []
    entity_entries: List[Dict[str, Any]] = []
    entity_rows: List[Dict[str, float]] = []
    for threshold in sweep_thresholds:
        scored = [
            ScoredExample(
                language,
                gold,
                _filter_and_bucket(
                    model_results,
                    deterministic_results,
                    threshold,
                    label_map,
                    keep,
                    span_map,
                ),
            )
            for language, gold, model_results, deterministic_results, span_map in cached
        ]
        metrics_result = evaluate(scored, betas=betas)
        threshold_entries.append(threshold_entry(threshold, metrics_result))
        micro = metrics_result["modes"][args.sweep_mode]["micro"]
        micro_rows.append(summary_row(threshold, micro))
        if args.sweep_entity:
            entity_entries.append(
                threshold_entry(threshold, metrics_result, entity=args.sweep_entity)
            )
            entity_metrics = metrics_result["modes"][args.sweep_mode][
                "per_entity"
            ].get(args.sweep_entity)
            entity_rows.append(summary_row(threshold, entity_metrics))

    result: Dict[str, Any] = {
        "config": _build_config(args, languages, keep, betas),
        "sweep": {
            "mode": args.sweep_mode,
            "scope": "micro",
            "thresholds": threshold_entries,
            "recommendations": recommend_thresholds(
                micro_rows, args.min_recall_for_high_precision
            ),
        },
    }
    if args.sweep_entity:
        result["sweep_entity"] = {
            "entity": args.sweep_entity,
            "mode": args.sweep_mode,
            "thresholds": entity_entries,
            "recommendations": recommend_thresholds(
                entity_rows, args.min_recall_for_high_precision
            ),
        }
    return result


def _format_reco(recommendations: Dict[str, Any]) -> List[str]:
    """Format the three recommendation profiles into aligned lines."""
    lines = []
    for profile in ("balanced", "high_recall", "high_precision"):
        reco = recommendations.get(profile)
        if reco is None:
            lines.append(f"  {profile:14s} (none: no threshold met the recall floor)")
        else:
            lines.append(
                f"  {profile:14s} thr={reco['threshold']:.2f}  "
                f"P={reco['precision']:.3f}  R={reco['recall']:.3f}  "
                f"F1={reco['f1']:.3f}  F2={reco['f2']:.3f}"
            )
    return lines


def _format_sweep_summary(result: Dict[str, Any]) -> str:
    """Format a human-readable summary of a threshold sweep."""
    sweep = result["sweep"]
    mode = sweep["mode"]
    lines = [f"threshold sweep ({mode} micro):", "  thr    P      R      F1     F2"]
    for entry in sweep["thresholds"]:
        metrics = entry[mode]
        lines.append(
            f"  {entry['threshold']:.2f}  {metrics['precision']:.3f}  "
            f"{metrics['recall']:.3f}  {metrics['f1']:.3f}  {metrics['f2']:.3f}"
        )
    lines.append("recommendations (micro):")
    lines.extend(_format_reco(sweep["recommendations"]))
    if "sweep_entity" in result:
        entity = result["sweep_entity"]
        lines.append(f"recommendations ({entity['entity']}):")
        lines.extend(_format_reco(entity["recommendations"]))
    return "\n".join(lines)


def _build_backend(backend: str, language: str, args: argparse.Namespace):
    """Build one NER backend recognizer, with actionable errors when optional.

    ``bards`` is always available on this branch; ``gliner`` and ``huggingface``
    are optional and raise ``SystemExit`` with installation guidance if their
    dependency or model is missing.
    """
    if backend == "bards":
        from presidio_analyzer.predefined_recognizers import BardsEuPiiRecognizer

        kwargs: Dict[str, Any] = {
            "supported_language": language,
            "threshold": args.threshold,
        }
        if args.mapping_profile:
            kwargs["mapping_profile"] = args.mapping_profile
        return BardsEuPiiRecognizer(**kwargs)

    if backend == "gliner":
        try:
            from presidio_analyzer.predefined_recognizers import GLiNERRecognizer

            kwargs = {"supported_language": language, "threshold": args.threshold}
            if args.gliner_model:
                kwargs["model_name"] = args.gliner_model
            return GLiNERRecognizer(**kwargs)
        except ImportError as exc:
            raise SystemExit(
                "--backend gliner requires the 'gliner' package "
                "(pip install gliner). Original error: " + str(exc)
            ) from exc

    if backend == "huggingface":
        if not args.huggingface_model:
            raise SystemExit(
                "--backend huggingface requires --huggingface-model MODEL_ID."
            )
        try:
            from presidio_analyzer.predefined_recognizers import (
                HuggingFaceNerRecognizer,
            )

            return HuggingFaceNerRecognizer(
                supported_language=language,
                model_name=args.huggingface_model,
                threshold=args.threshold,
            )
        except ImportError as exc:
            raise SystemExit(
                "--backend huggingface requires the 'transformers' extra "
                "(pip install 'presidio-analyzer[transformers]'). "
                "Original error: " + str(exc)
            ) from exc

    raise SystemExit(f"Unknown backend: {backend!r}")


def _bucketed_results(
    results,
    label_map: Optional[Dict[str, str]],
    keep: Optional[set],
    span_map,
    drop_structured: bool,
):
    """Map recognizer results to bucket-labeled, original-offset results.

    Returns a list of ``RecognizerResult`` whose ``entity_type`` is the
    evaluation bucket (so the ensemble agrees across backends) and whose offsets
    are mapped back to the original text. When ``drop_structured`` is set, buckets
    owned by the deterministic layer are dropped (hybrid mode).
    """
    from presidio_analyzer import RecognizerResult

    out = []
    for result in results:
        bucket = to_eval_bucket(result.entity_type, label_map)
        if drop_structured and bucket in _STRUCTURED_EVAL_BUCKETS:
            continue
        if keep is not None and bucket not in keep:
            continue
        start, end = map_span(result.start, result.end, span_map)
        out.append(
            RecognizerResult(
                entity_type=bucket, start=start, end=end, score=result.score
            )
        )
    return out


def _run_comparison(
    examples: List[EvalExample],
    args: argparse.Namespace,
    keep: Optional[set],
    label_map: Optional[Dict[str, str]],
    betas: List[float],
    languages: Optional[List[str]],
    backends: List[str],
) -> Dict[str, Any]:
    """Evaluate one or more NER backends, combining them via the ensemble mode.

    In hybrid mode the deterministic recognizers own the structured identifiers
    (shared across backends); the NER backends are combined via ``--ensemble``
    (default ``union`` for multiple backends).
    """
    ensemble_mode = args.ensemble or "union"
    hybrid = args.mode == "hybrid"
    deterministic = _build_deterministic_recognizers() if hybrid else []

    recognizers: Dict[tuple, Any] = {}
    scored: List[ScoredExample] = []
    for example in examples:
        text, span_map = _normalize_for_eval(example.text, args)

        per_backend = []
        for backend in backends:
            key = (backend, example.language)
            recognizer = recognizers.get(key)
            if recognizer is None:
                recognizer = _build_backend(backend, example.language, args)
                recognizers[key] = recognizer
            results = recognizer.analyze(text, list(recognizer.supported_entities))
            per_backend.append(
                _bucketed_results(results, label_map, keep, span_map, hybrid)
            )

        if ensemble_mode == "intersection":
            ner = intersection_spans(per_backend)
        else:
            ner = union_spans(per_backend)

        structured = []
        for recognizer in deterministic:
            det_results = recognizer.analyze(
                text, list(recognizer.supported_entities), nlp_artifacts=None
            )
            structured.extend(
                _bucketed_results(det_results, label_map, keep, span_map, False)
            )

        pred = [
            EvalSpan(span.start, span.end, span.entity_type)
            for span in ner + structured
        ]
        gold = _gold_spans(example, keep)
        scored.append(ScoredExample(example.language, gold, pred))

    config = _build_config(args, languages, keep, betas)
    config["backends"] = backends
    config["ensemble"] = ensemble_mode if len(backends) > 1 else None
    config["gliner_model"] = args.gliner_model
    config["huggingface_model"] = args.huggingface_model
    return {"config": config, "metrics": evaluate(scored, betas=betas)}


def _format_comparison_summary(result: Dict[str, Any]) -> str:
    """Format a short summary that names the backends and ensemble mode."""
    config = result["config"]
    backends = ",".join(config["backends"])
    head = (
        f"backends: {backends}  ensemble: {config['ensemble']}  "
        f"mode: {config['mode']}"
    )
    return head + "\n" + _format_summary(result)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the evaluation (single threshold, sweep or comparison) and report."""
    args = parse_args(argv)
    examples = load_jsonl(args.input)

    languages = _split_csv(args.languages)
    if languages:
        wanted = set(languages)
        examples = [example for example in examples if example.language in wanted]

    keep = set(_split_csv(args.entities) or []) or None
    label_map = _load_json_arg(args.label_map, "label-map")
    betas = [float(beta) for beta in (_split_csv(args.betas) or ["1.0", "2.0"])]

    if not examples:
        raise SystemExit("No examples to evaluate (check --input / --languages).")

    backends = args.backend or ["bards"]
    use_comparison = backends != ["bards"] or args.ensemble is not None
    sweep_thresholds = _parse_floats(args.threshold_sweep)

    if sweep_thresholds and use_comparison:
        raise SystemExit(
            "--threshold-sweep is only supported for the default single Bards "
            "backend; drop --backend/--ensemble to sweep."
        )

    if sweep_thresholds:
        result = _run_sweep(
            examples, args, keep, label_map, betas, languages, sweep_thresholds
        )
        summary = _format_sweep_summary(result)
    elif use_comparison:
        result = _run_comparison(
            examples, args, keep, label_map, betas, languages, backends
        )
        summary = _format_comparison_summary(result)
    else:
        result = _run_single(examples, args, keep, label_map, betas, languages)
        summary = _format_summary(result)

    serialized = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(serialized + "\n", encoding="utf-8")
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
