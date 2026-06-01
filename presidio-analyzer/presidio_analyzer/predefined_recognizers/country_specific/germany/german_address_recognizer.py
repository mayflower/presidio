from typing import List, Optional

from presidio_analyzer import Pattern, PatternRecognizer


class GermanAddressRecognizer(PatternRecognizer):
    """Recognize German street addresses.

    Matches a capitalized name token ending in a common German street suffix
    (Straße/Str./Strasse, Weg, Allee, Platz, Gasse, Ring, Damm, Ufer, Chaussee,
    Promenade), optionally followed by a house number such as ``12``, ``12a`` or
    ``12-14``. A street name with a house number scores higher than a bare street
    name.

    Returns ``LOCATION`` (Presidio's standard entity for geographic/address data);
    set ``supported_entity="ADDRESS"`` if you prefer a dedicated address entity.

    Optional and opt-in: disabled by default and only loaded when a German-language
    registry is built.

    :param patterns: List of patterns to be used by this recognizer.
    :param context: List of context words to increase confidence in detection.
    :param supported_language: Language this recognizer supports.
    :param supported_entity: The entity this recognizer can detect.
    :param name: Recognizer name.
    """

    COUNTRY_CODE = "de"

    # Capitalized name token whose last component is a street suffix, e.g.
    # "Bahnhofstraße", "Lindenallee", "Hauptstr.", "Marktplatz".
    _STREET = (
        r"[A-ZÄÖÜ][A-Za-zÄÖÜäöüß.\-]*"
        r"(?:straße|strasse|str\.|weg|allee|platz|gasse|ring|damm|ufer|"
        r"chaussee|promenade)"
    )
    # House number: 12, 12a, 12-14, 12 a.
    _HOUSE_NO = r"\d{1,4}\s?[a-zA-Z]?(?:\s?[-/]\s?\d{1,4}\s?[a-zA-Z]?)?"

    PATTERNS = [
        Pattern(
            "German street with house number",
            rf"\b{_STREET}\s+{_HOUSE_NO}\b",
            0.5,
        ),
        Pattern(
            "German street name (weak)",
            rf"\b{_STREET}\b",
            0.3,
        ),
    ]

    CONTEXT = [
        "adresse",
        "anschrift",
        "straße",
        "strasse",
        "hausnummer",
        "wohnort",
        "wohnhaft",
        "wohnanschrift",
        "lieferadresse",
        "rechnungsadresse",
        "postanschrift",
    ]

    def __init__(
        self,
        patterns: Optional[List[Pattern]] = None,
        context: Optional[List[str]] = None,
        supported_language: str = "de",
        supported_entity: str = "LOCATION",
        name: Optional[str] = None,
    ):
        super().__init__(
            supported_entity=supported_entity,
            patterns=patterns if patterns else self.PATTERNS,
            context=context if context else self.CONTEXT,
            supported_language=supported_language,
            name=name,
        )
