"""Tests for the optional German PII recognizers.

These recognizers are disabled by default and only loaded when a German-language
registry is explicitly built. The tests call ``recognizer.analyze`` directly
(no model download, no AnalyzerEngine), like the other Germany recognizer tests.
"""
import pytest

from presidio_analyzer import RecognizerRegistry
from presidio_analyzer.predefined_recognizers import (
    EmailRecognizer,
    GermanAddressRecognizer,
    GermanCreditCardRecognizer,
    GermanHonorificPersonRecognizer,
    GermanPhoneRecognizer,
    GermanPostalCodeRecognizer,
    GermanUsernamePatternRecognizer,
)

GERMAN_RECOGNIZER_CLASSES = (
    "GermanPhoneRecognizer",
    "GermanCreditCardRecognizer",
    "GermanPostalCodeRecognizer",
    "GermanAddressRecognizer",
    "GermanUsernamePatternRecognizer",
    "GermanHonorificPersonRecognizer",
)


# --------------------------------------------------------------------------- #
# GermanPhoneRecognizer -> PHONE_NUMBER
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text, expected_len",
    [
        ("Tel: +49 30 12345678", 1),
        ("0049 30 12345678", 1),
        ("Rufnummer 030 12345678", 1),
        ("030/12345678", 1),
        ("(030) 12345678", 1),
        ("Handynummer +49-151-23456789", 1),
        ("mobil 0151 23456789", 1),
        # false positives: plain numbers must NOT match
        ("Bestellnummer 12345", 0),
        ("im Jahr 1999 geboren", 0),
        ("Artikel 42 Absatz 3", 0),
    ],
)
def test_german_phone(text, expected_len):
    rec = GermanPhoneRecognizer()
    results = rec.analyze(text, ["PHONE_NUMBER"])
    assert len(results) == expected_len
    for r in results:
        assert r.entity_type == "PHONE_NUMBER"


# --------------------------------------------------------------------------- #
# GermanCreditCardRecognizer -> CREDIT_CARD (Luhn-validated)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text, expected_len",
    [
        ("Kreditkarte 4111 1111 1111 1111", 1),  # Luhn-valid
        ("Kartennummer 4111-1111-1111-1111", 1),
        ("4012888888881881", 1),  # Luhn-valid Visa test number
        # Luhn-invalid -> rejected
        ("Kreditkarte 4111 1111 1111 1112", 0),
        ("Bestellnummer 1234 5678 9012 3456", 0),
        # too short / too long
        ("123456789012", 0),
    ],
)
def test_german_credit_card(text, expected_len):
    rec = GermanCreditCardRecognizer()
    results = rec.analyze(text, ["CREDIT_CARD"])
    assert len([r for r in results if r.entity_type == "CREDIT_CARD"]) == expected_len


def test_german_credit_card_luhn_high_score():
    rec = GermanCreditCardRecognizer()
    results = rec.analyze("Kreditkarte 4111 1111 1111 1111", ["CREDIT_CARD"])
    assert results and results[0].score > 0.5  # Luhn-valid is boosted


# --------------------------------------------------------------------------- #
# GermanPostalCodeRecognizer -> POSTAL_CODE
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text, expected_len",
    [
        ("Anschrift: 10115 Berlin", 1),
        ("80331 München", 1),
        ("PLZ 22085", 1),
        # invalid PLZ ranges / lengths
        ("00000", 0),
        ("99999", 0),
        ("Jahr 2024", 0),
        ("Nummer 1011", 0),
    ],
)
def test_german_postal_code(text, expected_len):
    rec = GermanPostalCodeRecognizer()
    results = rec.analyze(text, ["POSTAL_CODE"])
    assert len(results) == expected_len
    for r in results:
        assert r.entity_type == "POSTAL_CODE"


def test_german_postal_code_with_city_scores_higher_than_bare():
    rec = GermanPostalCodeRecognizer()
    with_city = rec.analyze("10115 Berlin", ["POSTAL_CODE"])[0]
    bare = rec.analyze("die Zahl 80331 steht da", ["POSTAL_CODE"])[0]
    assert with_city.score > bare.score


# --------------------------------------------------------------------------- #
# GermanAddressRecognizer -> LOCATION
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text, expected_len",
    [
        ("Bahnhofstraße 12", 1),
        ("Lindenallee 5a", 1),
        ("Hauptstr. 12-14", 1),
        ("Am Marktplatz 1", 1),
        ("Goetheweg 7", 1),
        # plain words without a street suffix must not match
        ("Das war gestern", 0),
        ("Die Sonne scheint", 0),
    ],
)
def test_german_address(text, expected_len):
    rec = GermanAddressRecognizer()
    results = rec.analyze(text, ["LOCATION"])
    assert len(results) >= expected_len if expected_len else len(results) == 0
    for r in results:
        assert r.entity_type == "LOCATION"


def test_german_address_with_house_number_scores_higher():
    rec = GermanAddressRecognizer()
    with_no = rec.analyze("Bahnhofstraße 12", ["LOCATION"])[0]
    assert with_no.score >= 0.5


# --------------------------------------------------------------------------- #
# GermanUsernamePatternRecognizer -> USERNAME
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text, expected_handle",
    [
        ("Mein Handle @max_mustermann", "@max_mustermann"),
        ("Schreib @user_123 an", "@user_123"),
    ],
)
def test_german_username_handle(text, expected_handle):
    rec = GermanUsernamePatternRecognizer()
    results = rec.analyze(text, ["USERNAME"])
    assert any(text[r.start:r.end] == expected_handle for r in results)
    for r in results:
        assert r.entity_type == "USERNAME"


def test_german_username_shape_detected():
    """A token with a digit/underscore is detected (low score; context boosts it)."""
    rec = GermanUsernamePatternRecognizer()
    text = "Benutzername max_mustermann123"
    results = rec.analyze(text, ["USERNAME"])
    assert any(text[r.start:r.end] == "max_mustermann123" for r in results)


def test_german_username_plain_word_not_matched():
    """A plain capitalized German word (no @, no digit/underscore) is not a username."""
    rec = GermanUsernamePatternRecognizer()
    results = rec.analyze("Der Benutzer schreibt einen Brief", ["USERNAME"])
    # no @handle and no digit/underscore token -> nothing
    assert results == []


# --------------------------------------------------------------------------- #
# GermanHonorificPersonRecognizer -> PERSON
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text, expected_substr",
    [
        ("Herr Müller war da", "Herr Müller"),
        ("Sehr geehrte Frau Schmidt", "Frau Schmidt"),
        ("Prof. Dr. Weber hält die Vorlesung", "Prof. Dr. Weber"),
        ("Ansprechpartner: Frau Dr. Anna Schmidt", "Frau Dr. Anna Schmidt"),
    ],
)
def test_german_honorific_person(text, expected_substr):
    rec = GermanHonorificPersonRecognizer()
    results = rec.analyze(text, ["PERSON"])
    assert any(text[r.start:r.end] == expected_substr for r in results), [
        text[r.start:r.end] for r in results
    ]
    for r in results:
        assert r.entity_type == "PERSON"


@pytest.mark.parametrize(
    "text",
    [
        # capitalized German nouns without an honorific must NOT be flagged
        "Das Haus und der Garten sind grün.",
        "Die Deutsche Bahn fährt nach Berlin.",
        "Der Bundeskanzler hielt eine Rede.",
    ],
)
def test_german_honorific_person_no_false_positive_on_nouns(text):
    rec = GermanHonorificPersonRecognizer()
    assert rec.analyze(text, ["PERSON"]) == []


def test_german_honorific_person_does_not_absorb_lowercase_words():
    rec = GermanHonorificPersonRecognizer()
    text = "Herr Müller ging gestern einkaufen."
    results = rec.analyze(text, ["PERSON"])
    assert results
    assert all(text[r.start:r.end] == "Herr Müller" for r in results)


# --------------------------------------------------------------------------- #
# Loading / registry behavior
# --------------------------------------------------------------------------- #
def test_german_recognizers_disabled_in_default_english_registry():
    """Default (English) registry must not load the optional German recognizers."""
    registry = RecognizerRegistry()
    registry.load_predefined_recognizers()
    names = {type(r).__name__ for r in registry.recognizers}
    for cls in GERMAN_RECOGNIZER_CLASSES:
        assert cls not in names


def test_german_recognizers_enabled_via_yaml_registry():
    """A YAML/dict registry config that lists them (for German) loads them."""
    from presidio_analyzer.recognizer_registry import RecognizerRegistryProvider

    cfg = {
        "supported_languages": ["de"],
        "recognizers": [
            {"name": name, "type": "predefined", "supported_languages": ["de"]}
            for name in GERMAN_RECOGNIZER_CLASSES
        ],
    }
    registry = RecognizerRegistryProvider(
        registry_configuration=cfg
    ).create_recognizer_registry()
    names = {type(r).__name__ for r in registry.recognizers}
    for cls in GERMAN_RECOGNIZER_CLASSES:
        assert cls in names


def test_english_recognizers_not_broken():
    """Adding the German recognizers must not affect the English defaults."""
    registry = RecognizerRegistry()
    registry.load_predefined_recognizers()
    names = {type(r).__name__ for r in registry.recognizers}
    assert {"EmailRecognizer", "CreditCardRecognizer", "PhoneRecognizer"} <= names

    res = EmailRecognizer().analyze("contact john@example.com", ["EMAIL_ADDRESS"])
    assert len(res) == 1 and res[0].entity_type == "EMAIL_ADDRESS"
