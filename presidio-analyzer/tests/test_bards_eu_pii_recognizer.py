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
    EXPECTED_EU_PII_MODEL_LABELS,
    MAPPING_PROFILE_GDPR_SENSITIVE,
    MAPPING_PROFILE_HIGH_RECALL,
    MAPPING_PROFILE_PRESERVE_MODEL_LABELS,
    MAPPING_PROFILE_PRESIDIO_STANDARD,
    STRUCTURED_LABELS_WITH_DETERMINISTIC_RECOGNIZERS,
    get_eu_pii_entity_mapping,
    normalize_model_label,
    validate_eu_pii_mapping_labels,
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
# labels_to_ignore (hybrid setups)
# --------------------------------------------------------------------------- #
def test_ignored_label_dropped_even_when_entity_requested(mock_pipeline):
    """An ignored label is removed even if its mapped entity is requested."""
    rec = BardsEuPiiRecognizer(labels_to_ignore=["EMAIL_ADDRESS"])
    mock_pipeline.return_value = [_pred("EMAIL_ADDRESS", 0, 7, "a@b.com")]
    assert rec.analyze("a@b.com", ["EMAIL_ADDRESS"]) == []


def test_ignored_label_not_in_supported_entities(mock_pipeline):
    rec = BardsEuPiiRecognizer(labels_to_ignore=["EMAIL_ADDRESS"])
    assert "EMAIL_ADDRESS" not in rec.supported_entities
    # PERSON (from PERSON_NAME/PERSON_ALIAS) is unaffected.
    assert "PERSON" in rec.supported_entities


def test_ignored_label_dropped_when_all_supported_requested(mock_pipeline):
    """Ignored labels stay dropped even when every supported entity is requested.

    This is the sentinel bug guard: AnalyzerEngine expands ``entities=None`` to
    every supported entity, so an ignored label must not surface either then or
    in an unfiltered call.
    """
    rec = BardsEuPiiRecognizer(labels_to_ignore=["EMAIL_ADDRESS"])
    mock_pipeline.return_value = [
        _pred("EMAIL_ADDRESS", 0, 7, "a@b.com"),
        _pred("PERSON_NAME", 8, 18, "John Smith"),
    ]
    # Expanded to all supported entities.
    res = rec.analyze("a@b.com John Smith", list(rec.supported_entities))
    types = {r.entity_type for r in res}
    assert "EMAIL_ADDRESS" not in types
    assert "PERSON" in types
    # No entity filter at all (the recognizer keeps unrequested/unmapped hits).
    res_unfiltered = rec.analyze("a@b.com John Smith", [])
    assert "EMAIL_ADDRESS" not in {r.entity_type for r in res_unfiltered}


def test_custom_labels_to_ignore_descriptive(mock_pipeline):
    rec = BardsEuPiiRecognizer(labels_to_ignore=["HEALTH_DATA"])
    assert "HEALTH_DATA" not in rec.supported_entities
    mock_pipeline.return_value = [_pred("HEALTH_DATA", 0, 8, "diabetes")]
    assert rec.analyze("diabetes", []) == []


def test_bio_prefixed_ignored_label_normalizes(mock_pipeline):
    """A ``B-``/``I-`` prefixed ignore label matches the aggregated base label."""
    rec = BardsEuPiiRecognizer(labels_to_ignore=["B-EMAIL_ADDRESS"])
    assert "EMAIL_ADDRESS" in rec.labels_to_ignore  # stored normalized
    assert "EMAIL_ADDRESS" not in rec.supported_entities
    mock_pipeline.return_value = [_pred("EMAIL_ADDRESS", 0, 7, "a@b.com")]
    assert rec.analyze("a@b.com", []) == []


def test_hybrid_ignores_structured_labels(mock_pipeline):
    rec = BardsEuPiiRecognizer.hybrid()
    # The structured model labels map to these Presidio entities; none survive.
    for entity in ("EMAIL_ADDRESS", "PHONE_NUMBER", "IP_ADDRESS", "CREDIT_CARD", "URL"):
        assert entity not in rec.supported_entities
    # Free-text NER entities remain.
    assert {"PERSON", "LOCATION", "NRP"} <= set(rec.supported_entities)

    mock_pipeline.return_value = [
        _pred("EMAIL_ADDRESS", 0, 7, "a@b.com"),
        _pred("PHONE_NUMBER", 8, 11, "555"),
        _pred("PERSON_NAME", 12, 22, "John Smith"),
    ]
    res = rec.analyze("a@b.com 555 John Smith", [])
    assert {r.entity_type for r in res} == {"PERSON"}


def test_hybrid_merges_extra_labels_to_ignore(mock_pipeline):
    """hybrid() adds extra labels on top of the structured defaults."""
    rec = BardsEuPiiRecognizer.hybrid(labels_to_ignore=["FINANCIAL_AMOUNT"])
    assert STRUCTURED_LABELS_WITH_DETERMINISTIC_RECOGNIZERS <= rec.labels_to_ignore
    assert "FINANCIAL_AMOUNT" in rec.labels_to_ignore
    assert "FINANCIAL_AMOUNT" not in rec.supported_entities


def test_no_labels_to_ignore_keeps_full_mapping(mock_pipeline):
    """The default (no ignore) recognizer advertises the full entity set."""
    rec = BardsEuPiiRecognizer()
    assert rec.labels_to_ignore == frozenset()
    assert "EMAIL_ADDRESS" in rec.supported_entities


# --------------------------------------------------------------------------- #
# Per-entity / per-language thresholds
# --------------------------------------------------------------------------- #
def test_global_threshold_unchanged_without_maps(mock_pipeline):
    """No threshold maps => parent's single global threshold, behavior intact."""
    rec = BardsEuPiiRecognizer()  # default 0.4
    assert rec.threshold == 0.4
    assert rec._has_custom_thresholds is False
    mock_pipeline.return_value = [_pred("PERSON_NAME", 0, 10, "John Smith", score=0.3)]
    # 0.3 < 0.4 -> dropped, exactly as before this feature existed.
    assert rec.analyze("John Smith", ["PERSON"]) == []


def test_entity_threshold_overrides_global(mock_pipeline):
    """A per-entity threshold (higher than global) suppresses a mid-score hit."""
    rec = BardsEuPiiRecognizer(thresholds_by_entity={"PERSON": 0.8})
    mock_pipeline.return_value = [
        _pred("PERSON_NAME", 0, 10, "John Smith", score=0.5),
        _pred("EMAIL_ADDRESS", 11, 18, "a@b.com", score=0.5),
    ]
    res = rec.analyze("John Smith a@b.com", ["PERSON", "EMAIL_ADDRESS"])
    # PERSON needs >= 0.8 (dropped at 0.5); EMAIL_ADDRESS still uses global 0.4.
    assert [r.entity_type for r in res] == ["EMAIL_ADDRESS"]


def test_entity_threshold_below_global_keeps_low_score(mock_pipeline):
    """A per-entity threshold below the global keeps an otherwise-dropped hit.

    Proves the parent's pre-filter floor is lowered to the minimum configured
    threshold so the post-filter can still see the low-score prediction.
    """
    rec = BardsEuPiiRecognizer(thresholds_by_entity={"PERSON": 0.2})
    mock_pipeline.return_value = [_pred("PERSON_NAME", 0, 10, "John Smith", score=0.3)]
    res = rec.analyze("John Smith", ["PERSON"])
    assert len(res) == 1 and res[0].entity_type == "PERSON"


def test_language_threshold_overrides_entity_threshold(mock_pipeline):
    """The per-language/entity threshold takes precedence over the per-entity one."""
    rec = BardsEuPiiRecognizer(
        supported_language="en",
        thresholds_by_entity={"PERSON": 0.5},
        thresholds_by_language={"en": {"PERSON": 0.9}},
    )
    mock_pipeline.return_value = [_pred("PERSON_NAME", 0, 10, "John Smith", score=0.6)]
    # 0.6 passes the entity threshold (0.5) but not the language one (0.9).
    assert rec.analyze("John Smith", ["PERSON"]) == []

    mock_pipeline.return_value = [_pred("PERSON_NAME", 0, 10, "John Smith", score=0.95)]
    assert len(rec.analyze("John Smith", ["PERSON"])) == 1


def test_language_threshold_only_applies_to_its_language(mock_pipeline):
    """A language sub-map for another language does not affect this instance."""
    rec = BardsEuPiiRecognizer(
        supported_language="en",
        thresholds_by_entity={"PERSON": 0.5},
        thresholds_by_language={"de": {"PERSON": 0.9}},
    )
    mock_pipeline.return_value = [_pred("PERSON_NAME", 0, 10, "John Smith", score=0.6)]
    # 'de' map is irrelevant for an 'en' instance, so the entity threshold (0.5)
    # applies and 0.6 is kept.
    assert len(rec.analyze("John Smith", ["PERSON"])) == 1


def test_threshold_keyed_by_mapped_entity_not_raw_label(mock_pipeline):
    """Thresholds key on the mapped Presidio entity, not the raw model label."""
    # PERSON_NAME maps to PERSON: a "PERSON" threshold applies.
    rec_mapped = BardsEuPiiRecognizer(thresholds_by_entity={"PERSON": 0.8})
    mock_pipeline.return_value = [_pred("PERSON_NAME", 0, 10, "John Smith", score=0.5)]
    assert rec_mapped.analyze("John Smith", ["PERSON"]) == []

    # A threshold keyed on the raw label "PERSON_NAME" never applies; the global
    # threshold (0.4) is used instead, so 0.5 is kept.
    rec_raw = BardsEuPiiRecognizer(thresholds_by_entity={"PERSON_NAME": 0.8})
    mock_pipeline.return_value = [_pred("PERSON_NAME", 0, 10, "John Smith", score=0.5)]
    assert len(rec_raw.analyze("John Smith", ["PERSON"])) == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"thresholds_by_entity": {"PERSON": 1.5}},
        {"thresholds_by_entity": {"PERSON": -0.1}},
        {"thresholds_by_entity": {"PERSON": "high"}},
        {"thresholds_by_entity": {"PERSON": True}},
        {"thresholds_by_language": {"en": {"PERSON": 2.0}}},
        {"thresholds_by_language": {"en": "not-a-map"}},
    ],
)
def test_invalid_threshold_values_raise(mock_pipeline, kwargs):
    with pytest.raises(ValueError):
        BardsEuPiiRecognizer(**kwargs)


def test_thresholds_resolution_method(mock_pipeline):
    """The resolver returns the most specific configured threshold."""
    rec = BardsEuPiiRecognizer(
        supported_language="en",
        threshold=0.4,
        thresholds_by_entity={"PERSON": 0.6, "LOCATION": 0.5},
        thresholds_by_language={"en": {"PERSON": 0.9}},
    )
    assert rec._resolve_threshold("PERSON") == 0.9  # language wins
    assert rec._resolve_threshold("LOCATION") == 0.5  # entity wins
    assert rec._resolve_threshold("ORGANIZATION") == 0.4  # global fallback


# --------------------------------------------------------------------------- #
# Mapping profiles
# --------------------------------------------------------------------------- #
def test_default_profile_equals_current_mapping():
    """The default profile is byte-for-byte the existing mapping."""
    assert get_eu_pii_entity_mapping() == dict(EU_PII_ENTITY_MAPPING)
    assert (
        get_eu_pii_entity_mapping(MAPPING_PROFILE_PRESIDIO_STANDARD)
        == dict(EU_PII_ENTITY_MAPPING)
    )


def test_gdpr_sensitive_keeps_special_categories_separate():
    mapping = get_eu_pii_entity_mapping(MAPPING_PROFILE_GDPR_SENSITIVE)
    for label in ("ETHNIC_ORIGIN", "RELIGION_OR_BELIEF", "POLITICAL_OPINION"):
        assert mapping[label] == label
    # NRP came only from that trio, so it is gone now.
    assert "NRP" not in set(mapping.values())
    # Faithful built-ins are untouched.
    assert mapping["PERSON_NAME"] == "PERSON"


def test_preserve_model_labels_maps_each_label_to_itself():
    mapping = get_eu_pii_entity_mapping(MAPPING_PROFILE_PRESERVE_MODEL_LABELS)
    assert set(mapping) == set(EU_PII_ENTITY_MAPPING)
    assert all(key == value for key, value in mapping.items())


def test_high_recall_maps_proper_name_to_person():
    mapping = get_eu_pii_entity_mapping(MAPPING_PROFILE_HIGH_RECALL)
    assert mapping["PROPER_NAME"] == "PERSON"
    # Other mappings stay conservative (identical to presidio_standard).
    standard = get_eu_pii_entity_mapping(MAPPING_PROFILE_PRESIDIO_STANDARD)
    assert {k: v for k, v in mapping.items() if k != "PROPER_NAME"} == {
        k: v for k, v in standard.items() if k != "PROPER_NAME"
    }


def test_get_mapping_returns_fresh_mutable_copy():
    """The helper must not return the read-only module constant."""
    mapping = get_eu_pii_entity_mapping()
    mapping["PERSON_NAME"] = "X"  # must not raise
    assert EU_PII_ENTITY_MAPPING["PERSON_NAME"] == "PERSON"  # constant intact


def test_invalid_mapping_profile_raises():
    with pytest.raises(ValueError, match="Unknown mapping_profile"):
        get_eu_pii_entity_mapping("bogus")


def test_recognizer_uses_mapping_profile(mock_pipeline):
    rec = BardsEuPiiRecognizer(mapping_profile=MAPPING_PROFILE_GDPR_SENSITIVE)
    assert "NRP" not in rec.supported_entities
    assert "ETHNIC_ORIGIN" in rec.supported_entities
    # The special category is now detected under its own entity type.
    mock_pipeline.return_value = [_pred("ETHNIC_ORIGIN", 0, 5, "Roma")]
    res = rec.analyze("Roma", ["ETHNIC_ORIGIN"])
    assert len(res) == 1 and res[0].entity_type == "ETHNIC_ORIGIN"


def test_explicit_label_mapping_overrides_profile(mock_pipeline):
    """An explicit label_mapping wins over mapping_profile."""
    rec = BardsEuPiiRecognizer(
        label_mapping={"PERSON_NAME": "PERSON"},
        mapping_profile=MAPPING_PROFILE_GDPR_SENSITIVE,
    )
    assert rec.label_mapping == {"PERSON_NAME": "PERSON"}


def test_invalid_mapping_profile_raises_on_construction(mock_pipeline):
    with pytest.raises(ValueError, match="Unknown mapping_profile"):
        BardsEuPiiRecognizer(mapping_profile="bogus")


# --------------------------------------------------------------------------- #
# Model-label drift guard (offline)
# --------------------------------------------------------------------------- #
def test_expected_labels_count_is_35():
    assert len(EXPECTED_EU_PII_MODEL_LABELS) == 35


def test_default_mapping_covers_expected_labels():
    """The built-in mapping's keys exactly equal the pinned expected set."""
    assert set(EU_PII_ENTITY_MAPPING) == set(EXPECTED_EU_PII_MODEL_LABELS)
    # And the validator accepts it (and every built-in profile, same keys).
    validate_eu_pii_mapping_labels(dict(EU_PII_ENTITY_MAPPING))
    for profile in (
        MAPPING_PROFILE_PRESIDIO_STANDARD,
        MAPPING_PROFILE_GDPR_SENSITIVE,
        MAPPING_PROFILE_PRESERVE_MODEL_LABELS,
        MAPPING_PROFILE_HIGH_RECALL,
    ):
        validate_eu_pii_mapping_labels(get_eu_pii_entity_mapping(profile))


def test_validate_detects_missing_labels():
    incomplete = dict(EU_PII_ENTITY_MAPPING)
    del incomplete["PERSON_NAME"]
    with pytest.raises(ValueError, match="missing expected labels"):
        validate_eu_pii_mapping_labels(incomplete)


def test_validate_detects_unknown_labels():
    extended = dict(EU_PII_ENTITY_MAPPING)
    extended["BRAND_NEW_LABEL"] = "PERSON"
    with pytest.raises(ValueError, match="unknown labels"):
        validate_eu_pii_mapping_labels(extended)


@pytest.mark.parametrize(
    "raw, base",
    [
        ("B-PERSON_NAME", "PERSON_NAME"),
        ("I-PERSON_NAME", "PERSON_NAME"),
        ("U-LOCATION", "LOCATION"),
        ("L-LOCATION", "LOCATION"),
        ("PERSON_NAME", "PERSON_NAME"),  # no prefix -> unchanged
        ("O", "O"),  # non-entity label unchanged
    ],
)
def test_normalize_model_label_strips_bio_prefixes(raw, base):
    assert normalize_model_label(raw) == base


def test_normalize_model_label_custom_prefixes():
    assert normalize_model_label("X-PERSON", prefixes=("X-",)) == "PERSON"
    # Default prefixes do not strip a custom one.
    assert normalize_model_label("X-PERSON") == "X-PERSON"


def test_builtin_mapping_validated_at_construction(mock_pipeline):
    """The drift guard runs for the built-in mapping (here it passes)."""
    rec = BardsEuPiiRecognizer()  # would raise if keys drifted from expected
    assert set(rec.label_mapping) == set(EXPECTED_EU_PII_MODEL_LABELS)


def test_custom_mapping_not_validated_by_default(mock_pipeline):
    """A partial custom mapping is allowed without opting into validation."""
    rec = BardsEuPiiRecognizer(label_mapping={"PERSON_NAME": "PERSON"})
    assert rec.label_mapping == {"PERSON_NAME": "PERSON"}


def test_custom_mapping_validation_opt_in_raises(mock_pipeline):
    with pytest.raises(ValueError, match="pinned model label set"):
        BardsEuPiiRecognizer(
            label_mapping={"PERSON_NAME": "PERSON"}, validate_mapping=True
        )


def test_validate_mapping_false_skips_builtin(mock_pipeline):
    """validate_mapping=False is an escape hatch even for the built-in path."""
    # Constructs without error; nothing to assert beyond no raise.
    BardsEuPiiRecognizer(validate_mapping=False)


def test_labels_to_ignore_does_not_break_builtin_validation(mock_pipeline):
    """Validation runs on the full base mapping, before labels_to_ignore."""
    # hybrid() ignores 5 structured labels; the drift check still passes because
    # it validates the full base mapping, not the post-ignore one.
    rec = BardsEuPiiRecognizer.hybrid()
    assert "EMAIL_ADDRESS" not in rec.label_mapping  # dropped from active mapping
    assert STRUCTURED_LABELS_WITH_DETERMINISTIC_RECOGNIZERS <= rec.labels_to_ignore


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


def test_labels_to_ignore_via_yaml_registry(mock_pipeline):
    """A YAML config can pass labels_to_ignore through to the recognizer."""
    from presidio_analyzer.recognizer_registry import RecognizerRegistryProvider

    cfg = {
        "supported_languages": ["en"],
        "recognizers": [
            {
                "name": "BardsEuPiiRecognizer",
                "type": "predefined",
                "supported_languages": ["en"],
                "labels_to_ignore": ["EMAIL_ADDRESS", "PHONE_NUMBER"],
            }
        ],
    }
    registry = RecognizerRegistryProvider(
        registry_configuration=cfg
    ).create_recognizer_registry()
    rec = {type(r).__name__: r for r in registry.recognizers}["BardsEuPiiRecognizer"]
    assert {"EMAIL_ADDRESS", "PHONE_NUMBER"} <= rec.labels_to_ignore
    assert "EMAIL_ADDRESS" not in rec.supported_entities
    assert "PHONE_NUMBER" not in rec.supported_entities


def test_thresholds_via_yaml_registry(mock_pipeline):
    """A YAML config can pass both threshold maps through to the recognizer."""
    from presidio_analyzer.recognizer_registry import RecognizerRegistryProvider

    cfg = {
        "supported_languages": ["en"],
        "recognizers": [
            {
                "name": "BardsEuPiiRecognizer",
                "type": "predefined",
                "supported_languages": ["en"],
                "thresholds_by_entity": {"PERSON": 0.7},
                "thresholds_by_language": {"en": {"LOCATION": 0.9}},
            }
        ],
    }
    registry = RecognizerRegistryProvider(
        registry_configuration=cfg
    ).create_recognizer_registry()
    rec = {type(r).__name__: r for r in registry.recognizers}["BardsEuPiiRecognizer"]
    assert rec.thresholds_by_entity == {"PERSON": 0.7}
    assert rec.thresholds_by_language == {"en": {"LOCATION": 0.9}}
    assert rec._resolve_threshold("LOCATION") == 0.9
    assert rec._resolve_threshold("PERSON") == 0.7


def test_mapping_profile_via_yaml_registry(mock_pipeline):
    """A YAML config can select a mapping profile."""
    from presidio_analyzer.recognizer_registry import RecognizerRegistryProvider

    cfg = {
        "supported_languages": ["en"],
        "recognizers": [
            {
                "name": "BardsEuPiiRecognizer",
                "type": "predefined",
                "supported_languages": ["en"],
                "mapping_profile": MAPPING_PROFILE_GDPR_SENSITIVE,
            }
        ],
    }
    registry = RecognizerRegistryProvider(
        registry_configuration=cfg
    ).create_recognizer_registry()
    rec = {type(r).__name__: r for r in registry.recognizers}["BardsEuPiiRecognizer"]
    assert rec.label_mapping["ETHNIC_ORIGIN"] == "ETHNIC_ORIGIN"
    assert "NRP" not in rec.supported_entities


# --------------------------------------------------------------------------- #
# Opt-in integration tests (touch the real Hugging Face model)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    os.getenv("PRESIDIO_RUN_BARDS_EU_PII_INTEGRATION") != "1",
    reason=(
        "Set PRESIDIO_RUN_BARDS_EU_PII_INTEGRATION=1 to run the model-label "
        "drift check (downloads only the model config.json, needs network)."
    ),
)
def test_integration_model_labels_drift_check():
    """Compare the live model's labels against the pinned expected set.

    Loads only the HF *config* (config.json), reads ``id2label``, strips BIO
    prefixes and drops ``O``, then asserts the resulting base-label set matches
    :data:`EXPECTED_EU_PII_MODEL_LABELS`. A mismatch is an early warning that the
    remote checkpoint's taxonomy changed and the mapping needs updating.
    """
    pytest.importorskip("transformers", reason="transformers is not installed")
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(DEFAULT_EU_PII_MODEL)
    id2label = dict(config.id2label)
    base_labels = {
        normalize_model_label(label)
        for label in id2label.values()
        if normalize_model_label(label) != "O"
    }
    expected = set(EXPECTED_EU_PII_MODEL_LABELS)
    assert base_labels == expected, (
        "Bards EU-PII model label drift detected. "
        f"missing={sorted(expected - base_labels)}, "
        f"unexpected={sorted(base_labels - expected)}. "
        "Update EU_PII_ENTITY_MAPPING and EXPECTED_EU_PII_MODEL_LABELS."
    )


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
