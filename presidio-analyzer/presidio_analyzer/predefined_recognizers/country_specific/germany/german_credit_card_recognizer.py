from typing import List, Optional, Tuple

from presidio_analyzer import EntityRecognizer, Pattern, PatternRecognizer


class GermanCreditCardRecognizer(PatternRecognizer):
    """Recognize credit card numbers in German-language text.

    Matches a candidate of 13-19 digits with optional spaces or hyphens and
    validates it with the Luhn checksum, so random digit runs are rejected.
    German context words (kreditkarte, kartennummer, visa, ...) boost the score.

    Optional and opt-in: disabled by default and only loaded when a German-language
    registry is built. Returns ``CREDIT_CARD``.

    :param patterns: List of patterns to be used by this recognizer.
    :param context: List of context words to increase confidence in detection.
    :param supported_language: Language this recognizer supports.
    :param supported_entity: The entity this recognizer can detect.
    :param replacement_pairs: Tuples of substrings to strip before Luhn validation.
    :param name: Recognizer name.
    """

    COUNTRY_CODE = "de"

    PATTERNS = [
        Pattern(
            "Credit card (13-19 digits, spaces/hyphens, weak)",
            r"\b\d(?:[ -]?\d){12,18}\b",
            0.3,
        ),
    ]

    CONTEXT = [
        "kreditkarte",
        "kreditkartennummer",
        "kartennummer",
        "karte",
        "visa",
        "mastercard",
        "amex",
        "american express",
        "girocard",
        "kartenprüfnummer",
    ]

    def __init__(
        self,
        patterns: Optional[List[Pattern]] = None,
        context: Optional[List[str]] = None,
        supported_language: str = "de",
        supported_entity: str = "CREDIT_CARD",
        replacement_pairs: Optional[List[Tuple[str, str]]] = None,
        name: Optional[str] = None,
    ):
        self.replacement_pairs = (
            replacement_pairs if replacement_pairs else [("-", ""), (" ", "")]
        )
        super().__init__(
            supported_entity=supported_entity,
            patterns=patterns if patterns else self.PATTERNS,
            context=context if context else self.CONTEXT,
            supported_language=supported_language,
            name=name,
        )

    def validate_result(self, pattern_text: str) -> bool:
        """Validate the pattern by stripping separators and applying Luhn."""
        sanitized_value = EntityRecognizer.sanitize_value(
            pattern_text, self.replacement_pairs
        )
        if not sanitized_value.isdigit() or not (13 <= len(sanitized_value) <= 19):
            return False
        return self.__luhn_checksum(sanitized_value)

    @staticmethod
    def __luhn_checksum(sanitized_value: str) -> bool:
        def digits_of(n: str) -> List[int]:
            return [int(dig) for dig in str(n)]

        digits = digits_of(sanitized_value)
        odd_digits = digits[-1::-2]
        even_digits = digits[-2::-2]
        checksum = sum(odd_digits)
        for d in even_digits:
            checksum += sum(digits_of(str(d * 2)))
        return checksum % 10 == 0
