from typing import List, Optional

from presidio_analyzer import Pattern, PatternRecognizer


class GermanHonorificPersonRecognizer(PatternRecognizer):
    """Recognize person names introduced by a German honorific or title.

    Matches a German honorific / academic title (Herr, Frau, Hr., Fr., Dr., Prof.)
    followed by one or more capitalized name tokens, e.g. ``Herr Müller``,
    ``Frau Dr. Schmidt``, ``Prof. Dr. Anna Weber``. The honorific gate keeps this
    from firing on arbitrary capitalized German nouns (German capitalizes all
    nouns). Returns ``PERSON`` with a moderate score; the honorific is included in
    the matched span.

    Optional and opt-in: disabled by default and only loaded when a German-language
    registry is built.

    :param patterns: List of patterns to be used by this recognizer.
    :param context: List of context words to increase confidence in detection.
    :param supported_language: Language this recognizer supports.
    :param supported_entity: The entity this recognizer can detect.
    :param name: Recognizer name.
    """

    COUNTRY_CODE = "de"

    # 1-3 capitalized name tokens (allowing hyphenated names like "Meyer-Schmidt").
    # ``(?-i:[A-ZÄÖÜ])`` forces the leading letter to be uppercase even though the
    # registry compiles patterns with re.IGNORECASE; this is what stops the match
    # from absorbing lowercase connector words (e.g. "und") or plain lowercase text.
    _CAP = r"(?-i:[A-ZÄÖÜ])"
    _NAMES = (
        rf"{_CAP}[a-zäöüß]+(?:-{_CAP}[a-zäöüß]+)?"
        rf"(?:\s+{_CAP}[a-zäöüß]+){{0,2}}"
    )

    PATTERNS = [
        Pattern(
            "Herr/Frau (+ optional title) + name",
            rf"\b(?:Herr|Frau|Hr\.|Fr\.)\s+(?:(?:Dr|Prof)\.?\s+){{0,2}}{_NAMES}",
            0.5,
        ),
        Pattern(
            "Dr./Prof. + name",
            rf"\b(?:Dr|Prof)\.?\s+(?:(?:Dr|Prof)\.?\s+)?{_NAMES}",
            0.45,
        ),
    ]

    CONTEXT = [
        "herr",
        "frau",
        "name",
        "vorname",
        "nachname",
        "ansprechpartner",
        "kontaktperson",
    ]

    def __init__(
        self,
        patterns: Optional[List[Pattern]] = None,
        context: Optional[List[str]] = None,
        supported_language: str = "de",
        supported_entity: str = "PERSON",
        name: Optional[str] = None,
    ):
        super().__init__(
            supported_entity=supported_entity,
            patterns=patterns if patterns else self.PATTERNS,
            context=context if context else self.CONTEXT,
            supported_language=supported_language,
            name=name,
        )
