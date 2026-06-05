"""Offline tests for the ``bards_onnx`` backend wiring in evaluate_bards_eu_pii.

No model is loaded: the recognizer classes are monkeypatched on
``presidio_analyzer.predefined_recognizers`` (the harness imports them lazily
inside ``_build_backend``), and ``main`` is driven against a tiny JSONL file
with the run functions stubbed, so only the routing/argument logic is exercised.
"""

from unittest.mock import MagicMock

import pytest

import evaluate_bards_eu_pii as ev


# --------------------------------------------------------------------------- #
# argparse choice
# --------------------------------------------------------------------------- #
def test_bards_onnx_is_a_valid_backend_choice():
    args = ev.parse_args(
        ["--input", "x.jsonl", "--backend", "bards", "--backend", "bards_onnx"]
    )
    assert args.backend == ["bards", "bards_onnx"]


def test_unknown_backend_is_rejected():
    with pytest.raises(SystemExit):
        ev.parse_args(["--input", "x.jsonl", "--backend", "tensorflow"])


# --------------------------------------------------------------------------- #
# _build_backend routing
# --------------------------------------------------------------------------- #
def test_build_backend_routes_bards_onnx(monkeypatch):
    import presidio_analyzer.predefined_recognizers as pr

    onnx_cls = MagicMock()
    monkeypatch.setattr(pr, "BardsEuPiiOnnxRecognizer", onnx_cls)

    args = ev.parse_args(["--input", "x.jsonl", "--threshold", "0.3"])
    ev._build_backend("bards_onnx", "de", args)
    onnx_cls.assert_called_once_with(supported_language="de", threshold=0.3)


def test_build_backend_bards_unchanged(monkeypatch):
    import presidio_analyzer.predefined_recognizers as pr

    bards_cls = MagicMock()
    monkeypatch.setattr(pr, "BardsEuPiiRecognizer", bards_cls)

    args = ev.parse_args(["--input", "x.jsonl"])
    ev._build_backend("bards", "en", args)
    bards_cls.assert_called_once_with(supported_language="en", threshold=0.4)


def test_build_backend_onnx_passes_mapping_profile(monkeypatch):
    import presidio_analyzer.predefined_recognizers as pr

    onnx_cls = MagicMock()
    monkeypatch.setattr(pr, "BardsEuPiiOnnxRecognizer", onnx_cls)

    args = ev.parse_args(["--input", "x.jsonl", "--mapping-profile", "gdpr_sensitive"])
    ev._build_backend("bards_onnx", "fr", args)
    onnx_cls.assert_called_once_with(
        supported_language="fr", threshold=0.4, mapping_profile="gdpr_sensitive"
    )


def test_build_backend_onnx_missing_extra_exits(monkeypatch):
    import presidio_analyzer.predefined_recognizers as pr

    def boom(**kwargs):
        raise ImportError("No module named 'optimum'")

    monkeypatch.setattr(pr, "BardsEuPiiOnnxRecognizer", boom)
    args = ev.parse_args(["--input", "x.jsonl"])
    with pytest.raises(SystemExit, match=r"bards-onnx"):
        ev._build_backend("bards_onnx", "en", args)


# --------------------------------------------------------------------------- #
# main() routing (no model: run functions stubbed)
# --------------------------------------------------------------------------- #
_ONE_EXAMPLE = '{"id": "1", "language": "en", "text": "hi", "spans": []}\n'


def test_default_bards_backend_uses_single_path(tmp_path, monkeypatch):
    inp = tmp_path / "data.jsonl"
    inp.write_text(_ONE_EXAMPLE, encoding="utf-8")

    called = []
    monkeypatch.setattr(
        ev, "_run_single", lambda *a, **k: called.append("single") or {}
    )
    monkeypatch.setattr(
        ev,
        "_run_comparison",
        lambda *a, **k: pytest.fail("comparison path used for the default backend"),
    )
    monkeypatch.setattr(ev, "_format_summary", lambda result: "ok")

    assert ev.main(["--input", str(inp)]) == 0
    assert called == ["single"]


def test_bards_plus_onnx_triggers_comparison(tmp_path, monkeypatch):
    inp = tmp_path / "data.jsonl"
    inp.write_text(_ONE_EXAMPLE, encoding="utf-8")

    seen = {}

    def fake_comparison(examples, args, keep, label_map, betas, languages, backends):
        seen["backends"] = backends
        return {"config": {"backends": backends, "ensemble": None, "mode": args.mode}}

    monkeypatch.setattr(ev, "_run_comparison", fake_comparison)
    monkeypatch.setattr(
        ev,
        "_run_single",
        lambda *a, **k: pytest.fail("single path used for a multi-backend run"),
    )
    monkeypatch.setattr(ev, "_format_comparison_summary", lambda result: "ok")

    rc = ev.main(["--input", str(inp), "--backend", "bards", "--backend", "bards_onnx"])
    assert rc == 0
    assert seen["backends"] == ["bards", "bards_onnx"]
