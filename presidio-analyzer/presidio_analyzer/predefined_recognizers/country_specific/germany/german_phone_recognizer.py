from typing import List, Optional

from presidio_analyzer.predefined_recognizers.generic.phone_recognizer import (
    PhoneRecognizer,
)


class GermanPhoneRecognizer(PhoneRecognizer):
    """Recognize German phone numbers using python-phonenumbers (region ``DE``).

    python-phonenumbers, at the default ``leniency=1`` (VALID), matches the common
    German formats — international (``+49`` / ``0049``), national leading-zero area
    and mobile numbers, and the usual separators (spaces, hyphens, parentheses, and
    the slash written between area code and number, e.g. ``030/12345678``) — while
    rejecting plain digit runs such as order numbers or years. German context words
    boost the confidence score.

    Optional and opt-in: like the other Germany recognizers it is disabled by
    default (``enabled: false`` in ``conf/default_recognizers.yaml``) and only loaded
    when a German-language registry is built. Returns ``PHONE_NUMBER``.

    :param context: Context words for enhancing the confidence score.
    :param supported_language: Language this recognizer supports.
    :param supported_entity: The entity this recognizer can detect.
    :param supported_regions: Phone number regions (defaults to ``("DE",)``).
    :param leniency: python-phonenumbers strictness (0=lenient .. 3=strict).
    :param name: Recognizer name.
    """

    COUNTRY_CODE = "de"

    CONTEXT = [
        "telefon",
        "telefonnummer",
        "tel",
        "mobil",
        "mobilnummer",
        "handy",
        "handynummer",
        "rufnummer",
        "fax",
        "faxnummer",
    ]

    def __init__(
        self,
        context: Optional[List[str]] = None,
        supported_language: str = "de",
        supported_entity: str = "PHONE_NUMBER",
        supported_regions=("DE",),
        leniency: Optional[int] = 1,
        name: Optional[str] = None,
    ):
        super().__init__(
            context=context if context else self.CONTEXT,
            supported_language=supported_language,
            supported_entity=supported_entity,
            supported_regions=supported_regions,
            leniency=leniency,
            name=name,
        )
