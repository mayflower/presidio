"""Tests for the optional BardsEuPiiOnnxRecognizer.

The unit tests never download a model and never require optimum / onnxruntime:
the ONNX load path's lazy imports are faked in ``sys.modules`` (fake
``onnxruntime`` and ``optimum.onnxruntime`` modules) and
``transformers.AutoTokenizer.from_pretrained`` / ``transformers.pipeline`` are
patched, so the real ``load()`` code (session options, ``from_pretrained``,
pipeline build, the cache, the ignored-label filter, thresholding) is exercised
against fakes. A single opt-in integration test downloads the real quantized
ONNX model and is skipped unless ``PRESIDIO_RUN_BARDS_EU_PII_ONNX_INTEGRATION=1``.
"""

import importlib
import os
import sys
import types
from unittest.mock import MagicMock

import pytest

from presidio_analyzer.predefined_recognizers import BardsEuPiiOnnxRecognizer
from presidio_analyzer.predefined_recognizers.ner import (
    bards_eu_pii_onnx_recognizer as onnx_module,
)

# The PyTorch pipeline factory lives in the parent module; patching it there
# lets the comparison tests build a real BardsEuPiiRecognizer without a download.
HF_MODULE = "presidio_analyzer.predefined_recognizers.ner.huggingface_ner_recognizer"


def _pred(entity_group, start, end, word, score=0.95):
    """Build a HuggingFace token-classification prediction dict (aggregated)."""
    return {
        "entity_group": entity_group,
        "score": score,
        "word": word,
        "start": start,
        "end": end,
    }


class _FakeSessionOptions:
    """Stand-in for onnxruntime.SessionOptions capturing the settings applied."""

    def __init__(self):
        self.graph_optimization_level = None
        self.intra_op_num_threads = None
        self.inter_op_num_threads = None


@pytest.fixture
def fake_ort(monkeypatch):
    """Fake the ONNX lazy imports so no optimum/onnxruntime/model is needed.

    Injects fake ``onnxruntime`` and ``optimum.onnxruntime`` modules and patches
    ``transformers.AutoTokenizer.from_pretrained`` and ``transformers.pipeline``.
    Yields a namespace exposing the patched mocks plus the fake (callable)
    pipeline, whose ``return_value`` tests set to crafted predictions.
    """
    transformers = importlib.import_module("transformers")
    onnx_module._ORT_MODEL_CACHE.clear()

    # fake onnxruntime
    fake_ort_mod = types.ModuleType("onnxruntime")
    fake_ort_mod.SessionOptions = _FakeSessionOptions
    fake_ort_mod.GraphOptimizationLevel = types.SimpleNamespace(
        ORT_ENABLE_ALL="ORT_ENABLE_ALL"
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort_mod)

    # fake optimum.onnxruntime.ORTModelForTokenClassification
    model = MagicMock(name="ort_model")
    ort_model_cls = MagicMock(name="ORTModelForTokenClassification")
    ort_model_cls.from_pretrained = MagicMock(return_value=model)
    fake_optimum = types.ModuleType("optimum")
    fake_optimum_ort = types.ModuleType("optimum.onnxruntime")
    fake_optimum_ort.ORTModelForTokenClassification = ort_model_cls
    fake_optimum.onnxruntime = fake_optimum_ort
    monkeypatch.setitem(sys.modules, "optimum", fake_optimum)
    monkeypatch.setitem(sys.modules, "optimum.onnxruntime", fake_optimum_ort)

    # patch transformers AutoConfig + AutoTokenizer.from_pretrained + pipeline
    config = MagicMock(name="config")
    config_loader = MagicMock(return_value=config)
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", config_loader)
    tokenizer = MagicMock(name="tokenizer")
    tokenizer_loader = MagicMock(return_value=tokenizer)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", tokenizer_loader)
    pipe = MagicMock(name="pipeline_callable", return_value=[])
    pipeline_factory = MagicMock(name="pipeline_factory", return_value=pipe)
    monkeypatch.setattr(transformers, "pipeline", pipeline_factory)

    yield types.SimpleNamespace(
        pipe=pipe,
        pipeline_factory=pipeline_factory,
        ort_model_cls=ort_model_cls,
        ort_model=model,
        config_loader=config_loader,
        config=config,
        tokenizer_loader=tokenizer_loader,
        tokenizer=tokenizer,
    )
    onnx_module._ORT_MODEL_CACHE.clear()


# --------------------------------------------------------------------------- #
# Importability
# --------------------------------------------------------------------------- #
def test_importable_from_both_paths():
    from presidio_analyzer.predefined_recognizers import (
        BardsEuPiiOnnxRecognizer as FromPackage,
    )
    from presidio_analyzer.predefined_recognizers.ner import (
        BardsEuPiiOnnxRecognizer as FromNer,
    )

    assert FromPackage is FromNer is BardsEuPiiOnnxRecognizer
    import presidio_analyzer.predefined_recognizers as pr
    import presidio_analyzer.predefined_recognizers.ner as ner

    assert "BardsEuPiiOnnxRecognizer" in pr.__all__
    assert "BardsEuPiiOnnxRecognizer" in ner.__all__


# --------------------------------------------------------------------------- #
# Construction / defaults / parity with BardsEuPiiRecognizer
# --------------------------------------------------------------------------- #
def test_defaults(fake_ort):
    rec = BardsEuPiiOnnxRecognizer()
    assert rec.name == "BardsEuPiiOnnxRecognizer"
    assert rec.model_name == onnx_module.DEFAULT_EU_PII_MODEL
    assert rec.REQUIRES_TORCH is False
    assert rec.aggregation_strategy == "first"
    assert rec._onnx_model_subfolder == "onnx"
    assert rec._onnx_model_file == "model_quantized.onnx"
    assert rec._onnx_provider == "CPUExecutionProvider"


def _build_pytorch(monkeypatch, **kwargs):
    """Build a BardsEuPiiRecognizer with the PyTorch pipeline factory mocked."""
    from presidio_analyzer.predefined_recognizers import BardsEuPiiRecognizer

    monkeypatch.setattr(
        f"{HF_MODULE}.hf_pipeline", MagicMock(return_value=MagicMock(return_value=[]))
    )
    return BardsEuPiiRecognizer(**kwargs)


def test_supported_entities_match_pytorch(fake_ort, monkeypatch):
    onnx_default = set(BardsEuPiiOnnxRecognizer().supported_entities)
    pytorch_default = set(_build_pytorch(monkeypatch).supported_entities)
    assert onnx_default == pytorch_default
    # ...and with a mapping profile that changes the effective mapping.
    onnx_gdpr = set(
        BardsEuPiiOnnxRecognizer(mapping_profile="gdpr_sensitive").supported_entities
    )
    pytorch_gdpr = set(
        _build_pytorch(monkeypatch, mapping_profile="gdpr_sensitive").supported_entities
    )
    assert onnx_gdpr == pytorch_gdpr
    assert "NRP" not in onnx_gdpr  # collapsed trio kept descriptive


def test_hybrid_merges_structured_with_caller_labels(fake_ort):
    from presidio_analyzer.predefined_recognizers.ner.bards_eu_pii_recognizer import (
        STRUCTURED_LABELS_WITH_DETERMINISTIC_RECOGNIZERS as STRUCTURED,
    )

    rec = BardsEuPiiOnnxRecognizer.hybrid(labels_to_ignore=["FINANCIAL_AMOUNT"])
    assert type(rec) is BardsEuPiiOnnxRecognizer
    assert STRUCTURED <= rec.labels_to_ignore
    assert "FINANCIAL_AMOUNT" in rec.labels_to_ignore
    # Same merged set the PyTorch hybrid() would produce.
    assert rec.labels_to_ignore == set(STRUCTURED) | {"FINANCIAL_AMOUNT"}


def test_bio_prefixed_labels_to_ignore_normalized(fake_ort):
    rec = BardsEuPiiOnnxRecognizer(
        labels_to_ignore=["B-EMAIL_ADDRESS", "I-PERSON_NAME"]
    )
    assert "EMAIL_ADDRESS" in rec.labels_to_ignore
    assert "PERSON_NAME" in rec.labels_to_ignore
    assert "EMAIL_ADDRESS" not in rec.supported_entities


# --------------------------------------------------------------------------- #
# Inference exercises the inherited mapping / ignore / threshold logic
# --------------------------------------------------------------------------- #
def test_person_name_maps_to_person(fake_ort):
    fake_ort.pipe.return_value = [_pred("PERSON_NAME", 0, 10, "John Smith")]
    rec = BardsEuPiiOnnxRecognizer()
    res = rec.analyze("John Smith went home", ["PERSON"])
    assert len(res) == 1 and res[0].entity_type == "PERSON"


def test_labels_to_ignore_drops_prediction(fake_ort):
    fake_ort.pipe.return_value = [_pred("EMAIL_ADDRESS", 0, 7, "a@b.com")]
    rec = BardsEuPiiOnnxRecognizer(labels_to_ignore=["EMAIL_ADDRESS"])
    assert rec.analyze("a@b.com", ["EMAIL_ADDRESS"]) == []


def test_thresholds_by_entity_applied(fake_ort):
    fake_ort.pipe.return_value = [_pred("PERSON_NAME", 0, 10, "John Smith", score=0.5)]
    rec = BardsEuPiiOnnxRecognizer(thresholds_by_entity={"PERSON": 0.8})
    assert rec.analyze("John Smith", ["PERSON"]) == []  # 0.5 < 0.8


def test_thresholds_by_language_applied(fake_ort):
    rec = BardsEuPiiOnnxRecognizer(
        supported_language="de",
        thresholds_by_language={"de": {"PERSON": 0.9}},
    )
    fake_ort.pipe.return_value = [_pred("PERSON_NAME", 0, 10, "Max Muller", score=0.6)]
    assert rec.analyze("Max Muller", ["PERSON"]) == []  # 0.6 < de PERSON 0.9
    fake_ort.pipe.return_value = [_pred("PERSON_NAME", 0, 10, "Max Muller", score=0.95)]
    assert len(rec.analyze("Max Muller", ["PERSON"])) == 1


# --------------------------------------------------------------------------- #
# ONNX session loading details
# --------------------------------------------------------------------------- #
def test_loads_quantized_onnx_with_enable_all(fake_ort):
    BardsEuPiiOnnxRecognizer()
    fake_ort.ort_model_cls.from_pretrained.assert_called_once()
    _, kwargs = fake_ort.ort_model_cls.from_pretrained.call_args
    assert kwargs["subfolder"] == "onnx"
    assert kwargs["file_name"] == "model_quantized.onnx"
    assert kwargs["provider"] == "CPUExecutionProvider"
    assert kwargs["session_options"].graph_optimization_level == "ORT_ENABLE_ALL"
    # Config is loaded from the repo root (the onnx/ subfolder has no config.json)
    # and passed explicitly, so Optimum does not try to read it from the subfolder.
    fake_ort.config_loader.assert_called_once_with(onnx_module.DEFAULT_EU_PII_MODEL)
    assert kwargs["config"] is fake_ort.config
    # pipeline built with the inherited aggregation strategy
    _, pkwargs = fake_ort.pipeline_factory.call_args
    assert pkwargs["aggregation_strategy"] == "first"


def test_thread_counts_set_on_session_options(fake_ort):
    BardsEuPiiOnnxRecognizer(onnx_intra_op_num_threads=3, onnx_inter_op_num_threads=2)
    _, kwargs = fake_ort.ort_model_cls.from_pretrained.call_args
    assert kwargs["session_options"].intra_op_num_threads == 3
    assert kwargs["session_options"].inter_op_num_threads == 2


# --------------------------------------------------------------------------- #
# ORT thread env parsing
# --------------------------------------------------------------------------- #
def test_threads_from_env(fake_ort, monkeypatch):
    monkeypatch.setenv("ORT_INTRA_OP_THREADS", "3")
    monkeypatch.setenv("ORT_INTER_OP_THREADS", "2")
    rec = BardsEuPiiOnnxRecognizer()
    assert rec._onnx_intra_op_num_threads == 3
    assert rec._onnx_inter_op_num_threads == 2


def test_explicit_threads_override_env(fake_ort, monkeypatch):
    monkeypatch.setenv("ORT_INTRA_OP_THREADS", "3")
    rec = BardsEuPiiOnnxRecognizer(onnx_intra_op_num_threads=8)
    assert rec._onnx_intra_op_num_threads == 8


def test_non_integer_env_ignored_with_warning(fake_ort, monkeypatch, caplog):
    monkeypatch.setenv("ORT_INTRA_OP_THREADS", "lots")
    with caplog.at_level("WARNING", logger="presidio-analyzer"):
        rec = BardsEuPiiOnnxRecognizer()
    assert rec._onnx_intra_op_num_threads is None
    assert "ORT_INTRA_OP_THREADS" in caplog.text


def test_non_positive_env_ignored_with_warning(fake_ort, monkeypatch, caplog):
    monkeypatch.setenv("ORT_INTER_OP_THREADS", "0")
    with caplog.at_level("WARNING", logger="presidio-analyzer"):
        rec = BardsEuPiiOnnxRecognizer()
    assert rec._onnx_inter_op_num_threads is None
    assert "ORT_INTER_OP_THREADS" in caplog.text


# --------------------------------------------------------------------------- #
# Missing-extra error
# --------------------------------------------------------------------------- #
def test_missing_extra_raises_actionable_error(monkeypatch):
    onnx_module._ORT_MODEL_CACHE.clear()
    # Force the optimum import inside the load path to fail regardless of what is
    # installed, then assert the wrapped error points users at the extra.
    monkeypatch.setitem(sys.modules, "optimum.onnxruntime", None)
    with pytest.raises(ImportError, match=r"presidio-analyzer\[bards-onnx\]"):
        BardsEuPiiOnnxRecognizer(model_name="unique/model-for-error-test")


# --------------------------------------------------------------------------- #
# Shared ORT-model cache + per-instance isolation
# --------------------------------------------------------------------------- #
def test_model_cached_across_languages(fake_ort):
    BardsEuPiiOnnxRecognizer(supported_language="en")
    BardsEuPiiOnnxRecognizer(supported_language="de")
    # The heavy ORT model + tokenizer is loaded once and shared...
    assert fake_ort.ort_model_cls.from_pretrained.call_count == 1
    assert fake_ort.tokenizer_loader.call_count == 1
    assert len(onnx_module._ORT_MODEL_CACHE) == 1
    # ...but each recognizer gets its own pipeline built around it.
    assert fake_ort.pipeline_factory.call_count == 2


def test_labels_to_ignore_isolated_between_instances(fake_ort):
    fake_ort.pipe.return_value = [_pred("EMAIL_ADDRESS", 0, 7, "a@b.com")]
    ignoring = BardsEuPiiOnnxRecognizer(
        supported_language="en", labels_to_ignore=["EMAIL_ADDRESS"]
    )
    keeping = BardsEuPiiOnnxRecognizer(supported_language="de")
    # Same cached model (loaded once), but the ignore filter is per recognizer.
    assert fake_ort.ort_model_cls.from_pretrained.call_count == 1
    assert "EMAIL_ADDRESS" not in ignoring.supported_entities
    assert "EMAIL_ADDRESS" in keeping.supported_entities
    assert ignoring.analyze("a@b.com", ["EMAIL_ADDRESS"]) == []
    assert [r.entity_type for r in keeping.analyze("a@b.com", ["EMAIL_ADDRESS"])] == [
        "EMAIL_ADDRESS"
    ]


# --------------------------------------------------------------------------- #
# YAML loader instantiates the class from the deployment config
# --------------------------------------------------------------------------- #
def test_yaml_loader_from_bards_hybrid_config(fake_ort):
    from pathlib import Path

    import presidio_analyzer
    from presidio_analyzer.recognizer_registry import RecognizerRegistryProvider

    conf = (
        Path(presidio_analyzer.__file__).parent
        / "conf"
        / "bards_hybrid_recognizers.yaml"
    )
    registry = RecognizerRegistryProvider(
        conf_file=str(conf)
    ).create_recognizer_registry()
    onnx = [
        r
        for r in registry.recognizers
        if type(r).__name__ == "BardsEuPiiOnnxRecognizer"
    ]
    assert {r.supported_language for r in onnx} == {"en", "de", "fr", "it"}
    assert onnx[0]._onnx_provider == "CPUExecutionProvider"
    assert {"EMAIL_ADDRESS", "PHONE_NUMBER", "IP_ADDRESS"} <= onnx[0].labels_to_ignore


# --------------------------------------------------------------------------- #
# Opt-in integration test (downloads the real quantized ONNX model)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    os.getenv("PRESIDIO_RUN_BARDS_EU_PII_ONNX_INTEGRATION") != "1",
    reason=(
        "Set PRESIDIO_RUN_BARDS_EU_PII_ONNX_INTEGRATION=1 to run the real ONNX "
        "model integration test (downloads the quantized model, needs the "
        "bards-onnx extra and network access)."
    ),
)
def test_integration_real_onnx_model_detects_person_and_email():
    pytest.importorskip("optimum", reason="optimum is not installed")
    pytest.importorskip("onnxruntime", reason="onnxruntime is not installed")

    rec = BardsEuPiiOnnxRecognizer(threshold=0.3)
    text = "Contact John Smith at john.smith@example.com"
    found = {r.entity_type for r in rec.analyze(text, rec.supported_entities)}
    assert "PERSON" in found
    assert "EMAIL_ADDRESS" in found
