from typing import List, Optional

from presidio_analyzer import Pattern, PatternRecognizer


class GermanPostalCodeRecognizer(PatternRecognizer):
    """Recognize German postal codes (Postleitzahl, PLZ).

    A German PLZ is exactly 5 digits in the range 01001-99998. A bare 5-digit
    number is highly ambiguous (years, order ids, prices), so the base score is
    low and the recognizer is only actionable with context: it scores higher when
    the PLZ is immediately followed by a capitalized city name (e.g. ``10115
    Berlin``) or when address context words (PLZ, Postleitzahl, ...) are nearby.

    Returns ``POSTAL_CODE`` by default. Presidio has no canonical postal-code
    entity; set ``supported_entity="LOCATION"`` to fold it into the standard
    LOCATION entity instead. Distinct from the bundled :class:`DePlzRecognizer`,
    which returns the Germany-specific ``DE_PLZ`` entity.

    Optional and opt-in: disabled by default and only loaded when a German-language
    registry is built.

    :param patterns: List of patterns to be used by this recognizer.
    :param context: List of context words to increase confidence in detection.
    :param supported_language: Language this recognizer supports.
    :param supported_entity: The entity this recognizer can detect.
    :param name: Recognizer name.
    """

    COUNTRY_CODE = "de"

    # Valid PLZ range 01001-99998 (excludes 01000/99999 and 6-digit numbers).
    _PLZ = r"(?!01000\b|99999\b)(?:0[1-9]\d{3}|[1-9]\d{4})"

    PATTERNS = [
        Pattern(
            # ``(?-i:[A-ZÄÖÜ])`` forces an uppercase initial (a real city name) even
            # though the registry compiles patterns with re.IGNORECASE, so a trailing
            # lowercase word (e.g. "80331 steht") does not falsely boost the score.
            "German PLZ followed by a city",
            rf"\b{_PLZ}\b(?=\s+(?-i:[A-ZÄÖÜ])[a-zäöüß])",
            0.4,
        ),
        Pattern(
            "German PLZ (weak, context required)",
            rf"\b{_PLZ}\b",
            0.1,
        ),
    ]

    CONTEXT = [
        "plz",
        "postleitzahl",
        "postanschrift",
        "anschrift",
        "adresse",
        "wohnort",
        "ort",
        "stadt",
        "postfach",
        "lieferadresse",
        "rechnungsadresse",
    ]

    def __init__(
        self,
        patterns: Optional[List[Pattern]] = None,
        context: Optional[List[str]] = None,
        supported_language: str = "de",
        supported_entity: str = "POSTAL_CODE",
        name: Optional[str] = None,
    ):
        super().__init__(
            supported_entity=supported_entity,
            patterns=patterns if patterns else self.PATTERNS,
            context=context if context else self.CONTEXT,
            supported_language=supported_language,
            name=name,
        )
