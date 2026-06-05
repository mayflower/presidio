"""Offline tests for benchmark_bards_cpu (CLI parsing / building / report only).

No model is loaded: the recognizer builders are monkeypatched, so the timing
harness, loaders and report assembly are exercised against fakes.
"""

import json
import types

import pytest

import benchmark_bards_cpu as bench


def _fake_ner():
    """A stand-in NER recognizer with the attributes the benchmark touches."""
    return types.SimpleNamespace(
        supported_entities=["PERSON"],
        analyze=lambda text, entities, **kwargs: [],
        _onnx_provider="CPUExecutionProvider",
        _onnx_model_subfolder="onnx",
        _onnx_model_file="model_quantized.onnx",
        _onnx_intra_op_num_threads=2,
        _onnx_inter_op_num_threads=1,
    )


# --------------------------------------------------------------------------- #
# argument parsing
# --------------------------------------------------------------------------- #
def test_parse_args_defaults():
    args = bench.parse_args(["--input", "x.jsonl"])
    assert args.backend == "onnx"
    assert args.mode == "standard"
    assert args.warmup == 1
    assert args.repeat == 1


def test_parse_args_rejects_unknown_backend():
    with pytest.raises(SystemExit):
        bench.parse_args(["--input", "x", "--backend", "tensorflow"])


# --------------------------------------------------------------------------- #
# input loading
# --------------------------------------------------------------------------- #
def test_load_docs_jsonl(tmp_path):
    p = tmp_path / "data.jsonl"
    p.write_text(
        '{"text": "Max Müller", "language": "de"}\n'
        '{"text": "Jane Doe", "language": "en"}\n\n',
        encoding="utf-8",
    )
    docs = bench._load_docs(str(p), ["de", "en"])
    assert docs == [("de", "Max Müller"), ("en", "Jane Doe")]


def test_load_docs_jsonl_filters_languages():
    pass  # covered below with a temp file


def test_load_docs_jsonl_language_filter(tmp_path):
    p = tmp_path / "data.jsonl"
    p.write_text(
        '{"text": "a", "language": "de"}\n{"text": "b", "language": "fr"}\n',
        encoding="utf-8",
    )
    assert bench._load_docs(str(p), ["de"]) == [("de", "a")]


def test_load_docs_plain_text(tmp_path):
    p = tmp_path / "notes.txt"
    p.write_text("line one\n\nline two\n", encoding="utf-8")
    assert bench._load_docs(str(p), ["fr"]) == [("fr", "line one"), ("fr", "line two")]


# --------------------------------------------------------------------------- #
# percentile helper
# --------------------------------------------------------------------------- #
def test_percentile():
    assert bench._percentile([], 50) == 0.0
    assert bench._percentile([5.0], 95) == 5.0
    assert bench._percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5
    assert bench._percentile(list(range(1, 101)), 95) == pytest.approx(95.05)


# --------------------------------------------------------------------------- #
# backend routing
# --------------------------------------------------------------------------- #
def test_build_ner_routes_onnx(monkeypatch):
    import presidio_analyzer.predefined_recognizers as pr
    from unittest.mock import MagicMock

    onnx_cls, pt_cls = MagicMock(), MagicMock()
    monkeypatch.setattr(pr, "BardsEuPiiOnnxRecognizer", onnx_cls)
    monkeypatch.setattr(pr, "BardsEuPiiRecognizer", pt_cls)

    bench._build_ner("onnx", "standard", "en")
    onnx_cls.assert_called_once_with(supported_language="en")
    pt_cls.assert_not_called()

    onnx_cls.reset_mock()
    bench._build_ner("onnx", "hybrid", "de")
    onnx_cls.hybrid.assert_called_once_with(supported_language="de")


def test_build_ner_routes_pytorch(monkeypatch):
    import presidio_analyzer.predefined_recognizers as pr
    from unittest.mock import MagicMock

    onnx_cls, pt_cls = MagicMock(), MagicMock()
    monkeypatch.setattr(pr, "BardsEuPiiOnnxRecognizer", onnx_cls)
    monkeypatch.setattr(pr, "BardsEuPiiRecognizer", pt_cls)

    bench._build_ner("pytorch", "standard", "en")
    pt_cls.assert_called_once_with(supported_language="en")
    onnx_cls.assert_not_called()


def test_build_ner_missing_extra_exits(monkeypatch):
    import presidio_analyzer.predefined_recognizers as pr

    def boom(**kwargs):
        raise ImportError("no optimum")

    monkeypatch.setattr(pr, "BardsEuPiiOnnxRecognizer", boom)
    with pytest.raises(SystemExit, match=r"bards-onnx"):
        bench._build_ner("onnx", "standard", "en")


# --------------------------------------------------------------------------- #
# end-to-end report (mocked recognizers, no model)
# --------------------------------------------------------------------------- #
def test_main_writes_report(tmp_path, monkeypatch, capsys):
    inp = tmp_path / "in.jsonl"
    inp.write_text(
        '{"text": "Max Müller", "language": "en"}\n'
        '{"text": "Jane Doe in Berlin", "language": "en"}\n',
        encoding="utf-8",
    )
    out = tmp_path / "out.json"
    monkeypatch.setattr(bench, "_build_ner", lambda backend, mode, lang: _fake_ner())
    monkeypatch.setattr(bench, "_build_deterministic", lambda: [])

    rc = bench.main(
        [
            "--input",
            str(inp),
            "--backend",
            "onnx",
            "--repeat",
            "3",
            "--output",
            str(out),
        ]
    )
    assert rc == 0
    result = json.loads(out.read_text())
    assert result["backend"] == "onnx"
    assert result["mode"] == "standard"
    assert result["examples"] == 2
    assert result["total_chars"] == len("Max Müller") + len("Jane Doe in Berlin")
    assert result["samples"] == 6  # 2 docs * repeat 3
    assert result["docs_per_second"] > 0
    assert result["chars_per_second"] > 0
    assert "p50_ms" in result and "p95_ms" in result
    assert result["onnx"]["provider"] == "CPUExecutionProvider"
    assert result["onnx"]["intra_op_num_threads"] == 2
    # a summary line is printed
    assert "backend: onnx" in capsys.readouterr().out


def test_main_p99_with_enough_samples(tmp_path, monkeypatch):
    inp = tmp_path / "in.txt"
    inp.write_text("\n".join(f"line {i}" for i in range(5)) + "\n", encoding="utf-8")
    out = tmp_path / "out.json"
    monkeypatch.setattr(bench, "_build_ner", lambda backend, mode, lang: _fake_ner())

    bench.main(
        [
            "--input",
            str(inp),
            "--backend",
            "pytorch",
            "--repeat",
            "20",
            "--output",
            str(out),
        ]
    )
    result = json.loads(out.read_text())
    assert result["samples"] == 100  # 5 docs * 20
    assert "p99_ms" in result
    assert result["onnx"] is None  # pytorch backend
