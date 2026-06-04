"""Tests for the optional BardsEuPiiRecognizer.

The unit tests never download a model: the HuggingFace ``pipeline`` factory
(used by the parent ``HuggingFaceNerRecognizer``) is patched with a mock that
returns crafted token-classification predictions. A single opt-in integration
test downloads the real model and is skipped unless
``PRESIDIO_RUN_BARDS_EU_PII_INTEGRATION=1`` is set.
"""
import os
from unittest.mock import MagicMock, patch

import pytest

from presidio_analyzer import RecognizerRegistry
from presidio_analyzer.predefined_recognizers import BardsEuPiiRecognizer
from presidio_analyzer.predefined_recognizers.ner.bards_eu_pii_recognizer import (
    DEFAULT_EU_PII_MODEL,
    EU_PII_ENTITY_MAPPING,
)

# The pipeline factory lives in the parent module; patching it there prevents any
# model download regardless of which subclass constructs the recognizer.
HF_MODULE = "presidio_analyzer.predefined_recognizers.ner.huggingface_ner_recognizer"


@pytest.fixture
def mock_pipeline():
    """Patch the HuggingFace pipeline factory; yield the mock pipeline callable.

    Set ``mock_pipeline.return_value`` to the list of prediction dicts the
    "model" should emit for a chunk.
    """
    pipe = MagicMock()
    pipe.return_value = []
    with patch(f"{HF_MODULE}.hf_pipeline") as factory:
        factory.return_value = pipe
        yield pipe


def _pred(entity_group, start, end, word, score=0.95):
    """Build a HuggingFace token-classification prediction dict (aggregated)."""
    return {
        "entity_group": entity_group,
        "score": score,
        "word": word,
        "start": start,
        "end": end,
    }


# --------------------------------------------------------------------------- #
# Entity mapping constant (no recognizer construction / no model needed)
# --------------------------------------------------------------------------- #
def test_mapping_has_all_35_model_labels():
    assert len(EU_PII_ENTITY_MAPPING) == 35


def test_mapping_is_read_only():
    with pytest.raises(TypeError):
        EU_PII_ENTITY_MAPPING["PERSON_NAME"] = "X"  # type: ignore[index]


@pytest.mark.parametrize(
    "label, entity",
    [
        ("PERSON_NAME", "PERSON"),
        ("PERSON_ALIAS", "PERSON"),
        ("EMAIL_ADDRESS", "EMAIL_ADDRESS"),
        ("PHONE_NUMBER", "PHONE_NUMBER"),
        ("PAYMENT_CARD", "CREDIT_CARD"),
        ("GEO_LOCATION", "LOCATION"),
        ("POSTAL_ADDRESS", "LOCATION"),
        ("DATE_OF_BIRTH", "DATE_TIME"),
        ("IDENTIFYING_LINK", "URL"),
        ("ORGANIZATION_NAME", "ORGANIZATION"),
    ],
)
def test_mapping_faithful_standard_entities(label, entity):
    assert EU_PII_ENTITY_MAPPING[label] == entity


@pytest.mark.parametrize(
    "label", ["ETHNIC_ORIGIN", "RELIGION_OR_BELIEF", "POLITICAL_OPINION"]
)
def test_mapping_special_categories_collapse_to_nrp(label):
    assert EU_PII_ENTITY_MAPPING[label] == "NRP"


@pytest.mark.parametrize(
    "label",
    [
        "HEALTH_DATA",
        "BIOMETRIC_DATA",
        "CRIMINAL_OFFENCE_DATA",
        "SEXUAL_ORIENTATION",
        "TRADE_UNION_MEMBERSHIP",
        "BANK_ACCOUNT_IDENTIFIER",
        "VEHICLE_IDENTIFIER",
    ],
)
def test_mapping_descriptive_passthrough(label):
    """Labels with no faithful Presidio standard keep their own entity type."""
    assert EU_PII_ENTITY_MAPPING[label] == label


# --------------------------------------------------------------------------- #
# Construction / supported entities
# --------------------------------------------------------------------------- #
def test_defaults(mock_pipeline):
    rec = BardsEuPiiRecognizer()
    assert rec.name == "BardsEuPiiRecognizer"
    assert rec.model_name == DEFAULT_EU_PII_MODEL
    assert rec.supported_language == "en"
    assert rec.threshold == 0.4
    # "first" (not "simple") keeps subword entities like e-mails as one span.
    assert rec.aggregation_strategy == "first"


def test_supported_entities_derived_from_mapping(mock_pipeline):
    rec = BardsEuPiiRecognizer()
    supported = set(rec.supported_entities)
    # NRP is present; the collapsed source labels are not.
    assert "NRP" in supported
    assert {"PERSON", "EMAIL_ADDRESS", "CREDIT_CARD", "LOCATION"} <= supported
    assert "ETHNIC_ORIGIN" not in supported
    assert "RELIGION_OR_BELIEF" not in supported
    assert "POLITICAL_OPINION" not in supported


# --------------------------------------------------------------------------- #
# analyze() with mocked predictions
# --------------------------------------------------------------------------- #
def test_person_name_maps_to_person(mock_pipeline):
    text = "John Smith went home"
    mock_pipeline.return_value = [_pred("PERSON_NAME", 0, 10, "John Smith")]
    rec = BardsEuPiiRecognizer()
    res = rec.analyze(text, ["PERSON"])
    assert len(res) == 1
    assert res[0].entity_type == "PERSON"
    assert (res[0].start, res[0].end) == (0, 10)


def test_email_and_phone(mock_pipeline):
    text = "mail a@b.com call 555"
    mock_pipeline.return_value = [
        _pred("EMAIL_ADDRESS", 5, 12, "a@b.com"),
        _pred("PHONE_NUMBER", 18, 21, "555"),
    ]
    rec = BardsEuPiiRecognizer()
    res = rec.analyze(text, ["EMAIL_ADDRESS", "PHONE_NUMBER"])
    types = {r.entity_type for r in res}
    assert types == {"EMAIL_ADDRESS", "PHONE_NUMBER"}


def test_special_category_detected_as_nrp(mock_pipeline):
    text = "she is католичка"
    mock_pipeline.return_value = [_pred("RELIGION_OR_BELIEF", 7, 16, "католичка")]
    rec = BardsEuPiiRecognizer()
    res = rec.analyze(text, ["NRP"])
    assert len(res) == 1 and res[0].entity_type == "NRP"


def test_descriptive_label_passes_through(mock_pipeline):
    text = "diagnosis diabetes"
    mock_pipeline.return_value = [_pred("HEALTH_DATA", 10, 18, "diabetes")]
    rec = BardsEuPiiRecognizer()
    res = rec.analyze(text, ["HEALTH_DATA"])
    assert len(res) == 1 and res[0].entity_type == "HEALTH_DATA"


def test_score_below_threshold_is_dropped(mock_pipeline):
    text = "John Smith"
    mock_pipeline.return_value = [_pred("PERSON_NAME", 0, 10, "John Smith", score=0.2)]
    rec = BardsEuPiiRecognizer()  # default threshold 0.4
    assert rec.analyze(text, ["PERSON"]) == []


def test_requested_entities_filter(mock_pipeline):
    """A supported-but-unrequested entity is filtered out."""
    text = "John Smith a@b.com"
    mock_pipeline.return_value = [
        _pred("PERSON_NAME", 0, 10, "John Smith"),
        _pred("EMAIL_ADDRESS", 11, 18, "a@b.com"),
    ]
    rec = BardsEuPiiRecognizer()
    res = rec.analyze(text, ["PERSON"])
    assert [r.entity_type for r in res] == ["PERSON"]


# --------------------------------------------------------------------------- #
# Overrides
# --------------------------------------------------------------------------- #
def test_custom_label_mapping_override(mock_pipeline):
    text = "John Smith"
    mock_pipeline.return_value = [_pred("PERSON_NAME", 0, 10, "John Smith")]
    rec = BardsEuPiiRecognizer(label_mapping={"PERSON_NAME": "CUSTOM"})
    res = rec.analyze(text, ["CUSTOM"])
    assert len(res) == 1 and res[0].entity_type == "CUSTOM"


def test_custom_model_name_override(mock_pipeline):
    rec = BardsEuPiiRecognizer(model_name="some/other-model")
    assert rec.model_name == "some/other-model"


# --------------------------------------------------------------------------- #
# Registry loading behavior
# --------------------------------------------------------------------------- #
def test_absent_from_default_registry():
    """The default registry must not load (or download) the opt-in recognizer."""
    registry = RecognizerRegistry()
    registry.load_predefined_recognizers()
    names = {type(r).__name__ for r in registry.recognizers}
    assert "BardsEuPiiRecognizer" not in names


def test_enabled_via_yaml_registry(mock_pipeline):
    """A registry config that lists it loads it (pipeline mocked, no download)."""
    from presidio_analyzer.recognizer_registry import RecognizerRegistryProvider

    cfg = {
        "supported_languages": ["en"],
        "recognizers": [
            {
                "name": "BardsEuPiiRecognizer",
                "type": "predefined",
                "supported_languages": ["en"],
            }
        ],
    }
    registry = RecognizerRegistryProvider(
        registry_configuration=cfg
    ).create_recognizer_registry()
    by_name = {type(r).__name__: r for r in registry.recognizers}
    assert "BardsEuPiiRecognizer" in by_name
    # Omitted YAML fields must fall back to the subclass defaults, not None.
    rec = by_name["BardsEuPiiRecognizer"]
    assert rec.model_name == DEFAULT_EU_PII_MODEL
    assert rec.threshold == 0.4
    assert "NRP" in rec.supported_entities


# --------------------------------------------------------------------------- #
# Opt-in integration test (downloads the real model)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    os.getenv("PRESIDIO_RUN_BARDS_EU_PII_INTEGRATION") != "1",
    reason=(
        "Set PRESIDIO_RUN_BARDS_EU_PII_INTEGRATION=1 to run the real-model "
        "integration test (downloads the model and requires network access)."
    ),
)
def test_integration_real_model_detects_person_and_email():
    pytest.importorskip("transformers", reason="transformers is not installed")
    pytest.importorskip("torch", reason="torch is not installed")

    rec = BardsEuPiiRecognizer(threshold=0.3)
    text = "Contact John Smith at john.smith@example.com"
    results = rec.analyze(text, rec.supported_entities)
    found = {r.entity_type for r in results}
    assert "PERSON" in found
    assert "EMAIL_ADDRESS" in found
