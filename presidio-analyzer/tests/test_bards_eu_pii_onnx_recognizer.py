"""Tests for the optional BardsEuPiiOnnxRecognizer.

The unit tests never download a model and never require torch / optimum: the
load path's lazy imports are faked in ``sys.modules`` (a fake ``onnxruntime`` and
a fake ``huggingface_hub.snapshot_download``) and
``transformers.AutoTokenizer`` / ``transformers.AutoConfig`` are patched, so the
real ``load()`` code (session options, the torch-free ONNX pipeline, BIO
aggregation, the cache, the ignored-label filter, thresholding) runs against
fakes. The fake tokenizer + session are driven by ``fake_ort.prime(...)`` so a
test controls exactly which spans the model "predicts". A single opt-in
integration test downloads the real quantized ONNX model and is skipped unless
``PRESIDIO_RUN_BARDS_EU_PII_ONNX_INTEGRATION=1``.
"""

import importlib
import os
import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest

from presidio_analyzer.predefined_recognizers import BardsEuPiiOnnxRecognizer
from presidio_analyzer.predefined_recognizers.ner import (
    bards_eu_pii_onnx_recognizer as onnx_module,
)

# The PyTorch pipeline factory lives in the parent module; patching it there
# lets the comparison tests build a real BardsEuPiiRecognizer without a download.
HF_MODULE = "presidio_analyzer.predefined_recognizers.ner.huggingface_ner_recognizer"

# A small BIO label set the fakes expose as the model's id2label.
_LABELS = [
    "O",
    "B-PERSON_NAME",
    "I-PERSON_NAME",
    "B-EMAIL_ADDRESS",
    "I-EMAIL_ADDRESS",
    "B-ETHNIC_ORIGIN",
    "B-HEALTH_DATA",
]
_LABEL2ID = {label: idx for idx, label in enumerate(_LABELS)}


class _FakeSessionOptions:
    """Stand-in for onnxruntime.SessionOptions capturing the settings applied."""

    def __init__(self):
        self.graph_optimization_level = None
        self.intra_op_num_threads = None
        self.inter_op_num_threads = None


class _NamedIO:
    """Minimal ONNX input/output descriptor (only ``.name`` is used)."""

    def __init__(self, name):
        self.name = name


def _logits_for(tokens):
    """Build a ``[1, n, num_labels]`` logits array whose softmax yields ``prob``.

    ``tokens`` is a list of ``(entity, start, end, prob)``; the row for each token
    is shaped so ``softmax(row)[B-entity] == prob`` exactly.
    """
    num_labels = len(_LABELS)
    rows = []
    for entity, _start, _end, prob in tokens:
        row = np.zeros(num_labels, dtype=np.float32)
        row[_LABEL2ID[f"B-{entity}"]] = np.log(prob * (num_labels - 1) / (1.0 - prob))
        rows.append(row)
    if not rows:
        rows = [np.zeros(num_labels, dtype=np.float32)]  # a single "O" token
    return np.array([rows], dtype=np.float32)


@pytest.fixture
def fake_ort(monkeypatch):
    """Fake the ONNX load path so no torch/optimum/onnxruntime/model is needed.

    Injects a fake ``onnxruntime`` module and ``huggingface_hub.snapshot_download``
    and patches ``transformers.AutoTokenizer`` / ``transformers.AutoConfig``. The
    fake tokenizer + session are driven by the shared ``state`` list, which tests
    set via ``prime(...)``. Yields a namespace exposing the fakes plus ``prime``.
    """
    transformers = importlib.import_module("transformers")
    onnx_module._ORT_SESSION_CACHE.clear()

    state = {"tokens": []}  # list of (entity, start, end, prob) for the next call
    created = {"sessions": [], "options": []}

    class _FakeInferenceSession:
        def __init__(self, path, sess_options=None, providers=None):
            self.path = path
            self.sess_options = sess_options
            self.providers = providers
            created["sessions"].append(self)

        def get_inputs(self):
            return [_NamedIO("input_ids"), _NamedIO("attention_mask")]

        def get_outputs(self):
            return [_NamedIO("logits")]

        def run(self, _output_names, _feeds):
            return [_logits_for(state["tokens"])]

    fake_ort_mod = types.ModuleType("onnxruntime")

    def _session_options():
        opts = _FakeSessionOptions()
        created["options"].append(opts)
        return opts

    fake_ort_mod.SessionOptions = _session_options
    fake_ort_mod.GraphOptimizationLevel = types.SimpleNamespace(
        ORT_ENABLE_ALL="ORT_ENABLE_ALL"
    )
    fake_ort_mod.InferenceSession = _FakeInferenceSession
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort_mod)

    # fake huggingface_hub.snapshot_download -> a local dir (no network)
    import huggingface_hub

    model_dir = "/fake/snapshot/dir"
    snapshot_loader = MagicMock(return_value=model_dir)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_loader)

    # fake tokenizer: offsets come from the primed tokens; ids are placeholders
    def fake_tokenizer_call(text, **kwargs):
        tokens = state["tokens"]
        offsets = [[start, end] for (_e, start, end, _p) in tokens] or [[0, 0]]
        n = len(offsets)
        return {
            "input_ids": np.ones((1, n), dtype=np.int64),
            "attention_mask": np.ones((1, n), dtype=np.int64),
            "offset_mapping": np.array([offsets], dtype=np.int64),
        }

    tokenizer = MagicMock(name="tokenizer", side_effect=fake_tokenizer_call)
    tokenizer_loader = MagicMock(return_value=tokenizer)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", tokenizer_loader)

    config = types.SimpleNamespace(id2label=dict(enumerate(_LABELS)))
    config_loader = MagicMock(return_value=config)
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", config_loader)

    def prime(tokens):
        """Set the (entity, start, end, prob) spans the next analyze() predicts."""
        state["tokens"] = list(tokens)

    yield types.SimpleNamespace(
        prime=prime,
        snapshot_loader=snapshot_loader,
        model_dir=model_dir,
        config_loader=config_loader,
        tokenizer_loader=tokenizer_loader,
        tokenizer=tokenizer,
        created=created,
    )
    onnx_module._ORT_SESSION_CACHE.clear()


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
    assert rec.labels_to_ignore == set(STRUCTURED) | {"FINANCIAL_AMOUNT"}


def test_bio_prefixed_labels_to_ignore_normalized(fake_ort):
    rec = BardsEuPiiOnnxRecognizer(
        labels_to_ignore=["B-EMAIL_ADDRESS", "I-PERSON_NAME"]
    )
    assert "EMAIL_ADDRESS" in rec.labels_to_ignore
    assert "PERSON_NAME" in rec.labels_to_ignore
    assert "EMAIL_ADDRESS" not in rec.supported_entities


# --------------------------------------------------------------------------- #
# Inference exercises the torch-free ONNX pipeline + inherited mapping/thresholds
# --------------------------------------------------------------------------- #
def test_person_name_maps_to_person(fake_ort):
    fake_ort.prime([("PERSON_NAME", 0, 10, 0.95)])
    rec = BardsEuPiiOnnxRecognizer()
    res = rec.analyze("John Smith went home", ["PERSON"])
    assert len(res) == 1 and res[0].entity_type == "PERSON"
    assert (res[0].start, res[0].end) == (0, 10)


def test_consecutive_subwords_merge_into_one_span(fake_ort):
    # The model BIO-tags every sub-word; consecutive same-entity tokens merge.
    fake_ort.prime([("PERSON_NAME", 0, 4, 0.99), ("PERSON_NAME", 5, 10, 0.97)])
    rec = BardsEuPiiOnnxRecognizer()
    res = rec.analyze("John Smith", ["PERSON"])
    assert len(res) == 1
    assert (res[0].start, res[0].end) == (0, 10)


def test_labels_to_ignore_drops_prediction(fake_ort):
    fake_ort.prime([("EMAIL_ADDRESS", 0, 7, 0.95)])
    rec = BardsEuPiiOnnxRecognizer(labels_to_ignore=["EMAIL_ADDRESS"])
    assert rec.analyze("a@b.com", ["EMAIL_ADDRESS"]) == []


def test_thresholds_by_entity_applied(fake_ort):
    fake_ort.prime([("PERSON_NAME", 0, 10, 0.5)])
    rec = BardsEuPiiOnnxRecognizer(thresholds_by_entity={"PERSON": 0.8})
    assert rec.analyze("John Smith", ["PERSON"]) == []  # 0.5 < 0.8


def test_thresholds_by_language_applied(fake_ort):
    rec = BardsEuPiiOnnxRecognizer(
        supported_language="de",
        thresholds_by_language={"de": {"PERSON": 0.9}},
    )
    fake_ort.prime([("PERSON_NAME", 0, 10, 0.6)])
    assert rec.analyze("Max Muller", ["PERSON"]) == []  # 0.6 < de PERSON 0.9
    fake_ort.prime([("PERSON_NAME", 0, 10, 0.95)])
    assert len(rec.analyze("Max Muller", ["PERSON"])) == 1


# --------------------------------------------------------------------------- #
# ONNX session loading details
# --------------------------------------------------------------------------- #
def test_loads_quantized_onnx_with_enable_all(fake_ort):
    BardsEuPiiOnnxRecognizer()
    # The repo is resolved to a local snapshot dir (so the load never calls the
    # HF repo-tree API and a baked, offline image loads from cache); only the
    # config, tokenizer and the single quantized ONNX file are fetched.
    fake_ort.snapshot_loader.assert_called_once()
    s_args, s_kwargs = fake_ort.snapshot_loader.call_args
    assert s_args[0] == onnx_module.DEFAULT_EU_PII_MODEL
    assert "onnx/model_quantized.onnx" in s_kwargs["allow_patterns"]

    # One InferenceSession built from the local quantized ONNX path with
    # ORT_ENABLE_ALL and the CPU provider.
    assert len(fake_ort.created["sessions"]) == 1
    session = fake_ort.created["sessions"][0]
    assert session.path == "/fake/snapshot/dir/onnx/model_quantized.onnx"
    assert session.providers == ["CPUExecutionProvider"]
    assert session.sess_options.graph_optimization_level == "ORT_ENABLE_ALL"
    # config + tokenizer loaded from the local dir (default tokenizer == model)
    fake_ort.config_loader.assert_called_once_with(fake_ort.model_dir)
    fake_ort.tokenizer_loader.assert_called_once_with(fake_ort.model_dir)


def test_thread_counts_set_on_session_options(fake_ort):
    BardsEuPiiOnnxRecognizer(onnx_intra_op_num_threads=3, onnx_inter_op_num_threads=2)
    opts = fake_ort.created["sessions"][0].sess_options
    assert opts.intra_op_num_threads == 3
    assert opts.inter_op_num_threads == 2


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
    onnx_module._ORT_SESSION_CACHE.clear()
    # Force the onnxruntime import inside the load path to fail regardless of what
    # is installed, then assert the wrapped error points users at the extra.
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    with pytest.raises(ImportError, match=r"presidio-analyzer\[bards-onnx\]"):
        BardsEuPiiOnnxRecognizer(model_name="unique/model-for-error-test")


# --------------------------------------------------------------------------- #
# Shared ORT-session cache + per-instance isolation
# --------------------------------------------------------------------------- #
def test_model_cached_across_languages(fake_ort):
    BardsEuPiiOnnxRecognizer(supported_language="en")
    BardsEuPiiOnnxRecognizer(supported_language="de")
    # The heavy ORT session + tokenizer is built once and shared...
    assert len(fake_ort.created["sessions"]) == 1
    assert fake_ort.tokenizer_loader.call_count == 1
    assert len(onnx_module._ORT_SESSION_CACHE) == 1


def test_labels_to_ignore_isolated_between_instances(fake_ort):
    ignoring = BardsEuPiiOnnxRecognizer(
        supported_language="en", labels_to_ignore=["EMAIL_ADDRESS"]
    )
    keeping = BardsEuPiiOnnxRecognizer(supported_language="de")
    # Same cached session (built once), but the ignore filter is per recognizer.
    assert len(fake_ort.created["sessions"]) == 1
    assert "EMAIL_ADDRESS" not in ignoring.supported_entities
    assert "EMAIL_ADDRESS" in keeping.supported_entities
    fake_ort.prime([("EMAIL_ADDRESS", 0, 7, 0.95)])
    assert ignoring.analyze("a@b.com", ["EMAIL_ADDRESS"]) == []
    fake_ort.prime([("EMAIL_ADDRESS", 0, 7, 0.95)])
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
    pytest.importorskip("onnxruntime", reason="onnxruntime is not installed")

    rec = BardsEuPiiOnnxRecognizer(threshold=0.3)
    text = "Contact John Smith at john.smith@example.com"
    found = {r.entity_type for r in rec.analyze(text, rec.supported_entities)}
    assert "PERSON" in found
    assert "EMAIL_ADDRESS" in found
