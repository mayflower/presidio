from typing import List, Optional

from presidio_analyzer import Pattern, PatternRecognizer


class GermanUsernamePatternRecognizer(PatternRecognizer):
    """Recognize usernames / handles in German-language text.

    Detects two shapes:

    - ``@handle`` mentions (e.g. ``@max_mustermann``) — the ``@`` is a strong
      signal, so these score on their own.
    - username-like tokens that contain a digit or underscore (e.g.
      ``max_mustermann``, ``user123``). These are ambiguous on their own and
      score low; German context words (Benutzername, Login, ...) boost them.

    Requiring a username-like shape (``@`` prefix, or a digit/underscore) keeps a
    plain capitalized German word from being flagged unless strong context exists.
    Returns ``USERNAME``.

    Optional and opt-in: disabled by default and only loaded when a German-language
    registry is built.

    :param patterns: List of patterns to be used by this recognizer.
    :param context: List of context words to increase confidence in detection.
    :param supported_language: Language this recognizer supports.
    :param supported_entity: The entity this recognizer can detect.
    :param name: Recognizer name.
    """

    COUNTRY_CODE = "de"

    PATTERNS = [
        Pattern(
            "Handle (@username)",
            r"(?<![\w@.])@[A-Za-z][A-Za-z0-9_.]{2,29}\b",
            0.5,
        ),
        Pattern(
            "Username-like token (digit/underscore, context required)",
            r"\b(?=[A-Za-z0-9_.\-]*[_0-9])[A-Za-z][A-Za-z0-9_.\-]{2,29}\b",
            0.15,
        ),
    ]

    CONTEXT = [
        "benutzername",
        "nutzername",
        "benutzer",
        "nutzer",
        "login",
        "handle",
        "konto",
        "account",
        "user",
        "username",
    ]

    def __init__(
        self,
        patterns: Optional[List[Pattern]] = None,
        context: Optional[List[str]] = None,
        supported_language: str = "de",
        supported_entity: str = "USERNAME",
        name: Optional[str] = None,
    ):
        super().__init__(
            supported_entity=supported_entity,
            patterns=patterns if patterns else self.PATTERNS,
            context=context if context else self.CONTEXT,
            supported_language=supported_language,
            name=name,
        )
