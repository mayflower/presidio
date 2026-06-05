r"""Throughput / latency benchmark for Bards EU-PII on CPU (PyTorch vs ONNX).

Measures how fast a recognizer processes documents — it does **not** score
quality (use ``evaluate_bards_eu_pii.py`` for that). Run it twice (``--backend
pytorch`` and ``--backend onnx``) to prove the ONNX CPU speed-up on local
hardware. Optional and offline-by-default: Presidio and the model are imported
lazily, so importing this module (and its unit tests) stays light.

Example::

    python benchmark_bards_cpu.py --input sample_data.jsonl \\
        --backend onnx --mode hybrid --warmup 1 --repeat 20 --output onnx.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

Doc = Tuple[str, str]  # (language, text)


def _split_csv(value: Optional[str]) -> Optional[List[str]]:
    """Split a comma-separated option into a list, or return ``None``."""
    if not value:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


def _load_docs(path: str, languages: List[str]) -> List[Doc]:
    """Load benchmark documents from a JSONL eval file or a plain text file.

    ``.jsonl`` inputs are read leniently (each line needs a ``text`` field;
    ``language`` defaults to the first ``--languages`` entry). Any other file is
    treated as plain text, one document per non-blank line. JSONL docs are
    filtered to the requested languages.
    """
    file_path = Path(path)
    default_language = languages[0] if languages else "en"
    docs: List[Doc] = []
    if file_path.suffix == ".jsonl":
        with file_path.open(encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                obj = json.loads(stripped)
                text = obj.get("text")
                if text:
                    docs.append((obj.get("language", default_language), text))
        wanted = set(languages)
        docs = [doc for doc in docs if doc[0] in wanted]
    else:
        with file_path.open(encoding="utf-8") as handle:
            docs = [
                (default_language, line.rstrip("\n"))
                for line in handle
                if line.strip()
            ]
    return docs


def _build_ner(backend: str, mode: str, language: str):
    """Build the NER recognizer for one language (lazy Presidio import)."""
    from presidio_analyzer.predefined_recognizers import (
        BardsEuPiiOnnxRecognizer,
        BardsEuPiiRecognizer,
    )

    cls = BardsEuPiiOnnxRecognizer if backend == "onnx" else BardsEuPiiRecognizer
    try:
        if mode == "hybrid":
            return cls.hybrid(supported_language=language)
        return cls(supported_language=language)
    except ImportError as exc:
        extra = "bards-onnx" if backend == "onnx" else "transformers"
        raise SystemExit(
            f"--backend {backend} could not load the model. Install it with "
            f"pip install 'presidio-analyzer[{extra}]'. Original error: {exc}"
        ) from exc


def _build_deterministic() -> list:
    """Build the deterministic recognizers that own structured PII in hybrid mode."""
    from presidio_analyzer import predefined_recognizers as pr

    recognizers = []
    for class_name in (
        "EmailRecognizer",
        "PhoneRecognizer",
        "CreditCardRecognizer",
        "IpRecognizer",
        "UrlRecognizer",
        "IbanRecognizer",
    ):
        factory = getattr(pr, class_name, None)
        if factory is None:
            continue
        try:
            recognizers.append(factory())
        except Exception as exc:  # noqa: BLE001 - best-effort, reported to stderr
            print(f"warning: skipping {class_name}: {exc}", file=sys.stderr)
    return recognizers


def _analyze_one(ner, deterministic: list, text: str) -> None:
    """Run one document through the NER backend and the deterministic layer."""
    ner.analyze(text, list(ner.supported_entities))
    for recognizer in deterministic:
        recognizer.analyze(
            text, list(recognizer.supported_entities), nlp_artifacts=None
        )


def _percentile(ordered: Sequence[float], q: float) -> float:
    """Return the ``q``-th percentile of a pre-sorted sequence (linear interp)."""
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    rank = (q / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def _onnx_info(ner) -> Dict[str, Any]:
    """Return ONNX Runtime / thread settings for the report (onnx backend)."""
    return {
        "provider": getattr(ner, "_onnx_provider", None),
        "onnx_model_subfolder": getattr(ner, "_onnx_model_subfolder", None),
        "onnx_model_file": getattr(ner, "_onnx_model_file", None),
        "intra_op_num_threads": getattr(ner, "_onnx_intra_op_num_threads", None),
        "inter_op_num_threads": getattr(ner, "_onnx_inter_op_num_threads", None),
        "env_ORT_INTRA_OP_THREADS": os.getenv("ORT_INTRA_OP_THREADS"),
        "env_ORT_INTER_OP_THREADS": os.getenv("ORT_INTER_OP_THREADS"),
    }


def _maybe_rss_mb() -> Optional[float]:
    """Return the process RSS in MB if psutil is available, else ``None``."""
    try:
        import psutil
    except ImportError:
        return None
    return round(psutil.Process().memory_info().rss / (1024 * 1024), 1)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark Bards EU-PII CPU throughput/latency (PyTorch vs ONNX)."
    )
    parser.add_argument(
        "--input", required=True, help="JSONL eval file or plain text (one doc/line)."
    )
    parser.add_argument(
        "--backend",
        choices=("pytorch", "onnx"),
        default="onnx",
        help="NER backend to benchmark (default: onnx).",
    )
    parser.add_argument(
        "--mode",
        choices=("standard", "hybrid"),
        default="standard",
        help="standard = NER only; hybrid = NER + deterministic recognizers.",
    )
    parser.add_argument(
        "--languages",
        default="en",
        help="Comma-separated languages (default: en). Plain-text docs use the first.",
    )
    parser.add_argument(
        "--warmup", type=int, default=1, help="Untimed warmup passes (default: 1)."
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Timed passes over the input (default: 1).",
    )
    parser.add_argument(
        "--output", default=None, help="Write the results JSON to this path."
    )
    return parser.parse_args(argv)


def _format_summary(result: Dict[str, Any]) -> str:
    """Format a short human-readable summary of the benchmark result."""
    lines = [
        f"backend: {result['backend']}  mode: {result['mode']}  "
        f"examples: {result['examples']}  samples: {result['samples']}",
        f"  {result['docs_per_second']:.1f} docs/s  "
        f"{result['chars_per_second']:.0f} chars/s  "
        f"total {result['total_seconds']:.3f}s",
        f"  p50={result['p50_ms']:.2f}ms  p95={result['p95_ms']:.2f}ms"
        + (f"  p99={result['p99_ms']:.2f}ms" if "p99_ms" in result else ""),
    ]
    if "rss_mb" in result:
        lines.append(f"  rss={result['rss_mb']:.0f} MB")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the benchmark and write / print the results."""
    args = parse_args(argv)
    languages = _split_csv(args.languages) or ["en"]
    docs = _load_docs(args.input, languages)
    if not docs:
        raise SystemExit("No documents to benchmark (check --input / --languages).")

    warmup = max(0, args.warmup)
    repeat = max(1, args.repeat)
    ner_by_language = {
        language: _build_ner(args.backend, args.mode, language)
        for language in {doc[0] for doc in docs}
    }
    deterministic = _build_deterministic() if args.mode == "hybrid" else []

    for _ in range(warmup):
        for language, text in docs:
            _analyze_one(ner_by_language[language], deterministic, text)

    samples: List[float] = []
    chars_processed = 0
    for _ in range(repeat):
        for language, text in docs:
            start = time.perf_counter()
            _analyze_one(ner_by_language[language], deterministic, text)
            samples.append(time.perf_counter() - start)
            chars_processed += len(text)

    total_seconds = sum(samples)
    ordered = sorted(samples)
    result: Dict[str, Any] = {
        "backend": args.backend,
        "mode": args.mode,
        "languages": languages,
        "examples": len(docs),
        "total_chars": sum(len(text) for _, text in docs),
        "warmup": warmup,
        "repeat": repeat,
        "samples": len(samples),
        "total_seconds": round(total_seconds, 6),
        "docs_per_second": (
            round(len(samples) / total_seconds, 3) if total_seconds else 0.0
        ),
        "chars_per_second": (
            round(chars_processed / total_seconds, 1) if total_seconds else 0.0
        ),
        "p50_ms": round(_percentile(ordered, 50) * 1000, 3),
        "p95_ms": round(_percentile(ordered, 95) * 1000, 3),
        "onnx": (
            _onnx_info(next(iter(ner_by_language.values())))
            if args.backend == "onnx"
            else None
        ),
    }
    if len(samples) >= 100:
        result["p99_ms"] = round(_percentile(ordered, 99) * 1000, 3)
    rss = _maybe_rss_mb()
    if rss is not None:
        result["rss_mb"] = rss

    serialized = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(serialized + "\n", encoding="utf-8")
    print(_format_summary(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
