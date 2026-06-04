"""Recognizer for the ``bardsai/eu-pii-anonimization-multilang`` model.

This is a thin wrapper around :class:`HuggingFaceNerRecognizer`. The model is a
standard ``XLMRobertaForTokenClassification`` token-classification checkpoint
(BIO tags, 35 PII classes, 24 EU languages, Apache-2.0), so the existing
HuggingFace NER recognizer already handles loading, inference, BIO-tag merging,
chunking, scoring and device selection. This module only contributes the default
model id, a curated mapping from the model's labels to Presidio entities, and an
opt-in ``labels_to_ignore`` mechanism for hybrid (model + deterministic) setups.
"""

from types import MappingProxyType
from typing import Dict, Iterable, List, Mapping, Optional

from presidio_analyzer import RecognizerResult
from presidio_analyzer.nlp_engine import NlpArtifacts
from presidio_analyzer.predefined_recognizers.ner.huggingface_ner_recognizer import (
    HuggingFaceNerRecognizer,
)

DEFAULT_EU_PII_MODEL = "bardsai/eu-pii-anonimization-multilang"

# BIO-style prefixes used to normalize ``labels_to_ignore`` to base labels before
# the parent recognizer is constructed. Mirrors the parent's default
# ``label_prefixes`` so an ignored label written as ``"B-EMAIL_ADDRESS"`` matches
# the same base label the parent emits at inference time.
_DEFAULT_LABEL_PREFIXES = ("B-", "I-", "U-", "L-")

# Mapping from the model's 35 base labels (the ``B-``/``I-`` prefix stripped,
# matching ``config.json`` ``id2label``) to Presidio entity types.
#
# - Labels with a faithful Presidio standard entity use it (PERSON, LOCATION,
#   EMAIL_ADDRESS, ...).
# - The GDPR Art. 9 nationality/religion/politics trio collapses onto Presidio's
#   built-in ``NRP`` entity (nationality / religious / political group).
# - Every other label has no faithful Presidio standard, so it passes through as
#   its own descriptive entity type. ``HuggingFaceNerRecognizer`` already keeps
#   unmapped labels, but listing all 35 here makes ``supported_entities``
#   explicit and lets callers override individual entries.
EU_PII_ENTITY_MAPPING: Mapping[str, str] = MappingProxyType(
    {
        # --- faithful Presidio standard entities ---
        "PERSON_NAME": "PERSON",
        "PERSON_ALIAS": "PERSON",
        "EMAIL_ADDRESS": "EMAIL_ADDRESS",
        "PHONE_NUMBER": "PHONE_NUMBER",
        "IP_ADDRESS": "IP_ADDRESS",
        "LOCATION": "LOCATION",
        "GEO_LOCATION": "LOCATION",
        "POSTAL_ADDRESS": "LOCATION",
        "ORGANIZATION_NAME": "ORGANIZATION",
        "PAYMENT_CARD": "CREDIT_CARD",
        "DATE_OF_BIRTH": "DATE_TIME",
        "IDENTIFYING_LINK": "URL",
        # --- GDPR Art. 9 nationality / religion / politics -> NRP ---
        "ETHNIC_ORIGIN": "NRP",
        "RELIGION_OR_BELIEF": "NRP",
        "POLITICAL_OPINION": "NRP",
        # --- descriptive pass-through (no faithful Presidio standard) ---
        "HEALTH_DATA": "HEALTH_DATA",
        "BIOMETRIC_DATA": "BIOMETRIC_DATA",
        "CRIMINAL_OFFENCE_DATA": "CRIMINAL_OFFENCE_DATA",
        "SEXUAL_ORIENTATION": "SEXUAL_ORIENTATION",
        "TRADE_UNION_MEMBERSHIP": "TRADE_UNION_MEMBERSHIP",
        "ACCOUNT_IDENTIFIER": "ACCOUNT_IDENTIFIER",
        "AUTH_SECRET": "AUTH_SECRET",
        "BANK_ACCOUNT_IDENTIFIER": "BANK_ACCOUNT_IDENTIFIER",
        "CONTACT_HANDLE": "CONTACT_HANDLE",
        "DEVICE_IDENTIFIER": "DEVICE_IDENTIFIER",
        "DOCUMENT_IDENTIFIER": "DOCUMENT_IDENTIFIER",
        "DOCUMENT_REFERENCE": "DOCUMENT_REFERENCE",
        "FINANCIAL_AMOUNT": "FINANCIAL_AMOUNT",
        "ORGANIZATION_IDENTIFIER": "ORGANIZATION_IDENTIFIER",
        "PAYMENT_CARD_SECURITY": "PAYMENT_CARD_SECURITY",
        "PERSON_ATTRIBUTE": "PERSON_ATTRIBUTE",
        "PERSON_IDENTIFIER": "PERSON_IDENTIFIER",
        "PERSON_ROLE_OR_TITLE": "PERSON_ROLE_OR_TITLE",
        "PROPER_NAME": "PROPER_NAME",
        "VEHICLE_IDENTIFIER": "VEHICLE_IDENTIFIER",
    }
)

# Pinned reference set of the model's 35 base PII labels (BIO prefixes stripped,
# ``O`` excluded), independent of :data:`EU_PII_ENTITY_MAPPING`. It is the
# source of truth for what the ``bardsai/eu-pii-anonimization-multilang``
# checkpoint is expected to emit. The built-in mapping is checked against it at
# construction, and the opt-in integration ("drift") test checks the *live*
# model's ``id2label`` against it — so a remote label change surfaces early. If
# the model's labels legitimately change, update this set and
# :data:`EU_PII_ENTITY_MAPPING` together.
EXPECTED_EU_PII_MODEL_LABELS = frozenset(
    {
        "PERSON_NAME",
        "PERSON_ALIAS",
        "EMAIL_ADDRESS",
        "PHONE_NUMBER",
        "IP_ADDRESS",
        "LOCATION",
        "GEO_LOCATION",
        "POSTAL_ADDRESS",
        "ORGANIZATION_NAME",
        "PAYMENT_CARD",
        "DATE_OF_BIRTH",
        "IDENTIFYING_LINK",
        "ETHNIC_ORIGIN",
        "RELIGION_OR_BELIEF",
        "POLITICAL_OPINION",
        "HEALTH_DATA",
        "BIOMETRIC_DATA",
        "CRIMINAL_OFFENCE_DATA",
        "SEXUAL_ORIENTATION",
        "TRADE_UNION_MEMBERSHIP",
        "ACCOUNT_IDENTIFIER",
        "AUTH_SECRET",
        "BANK_ACCOUNT_IDENTIFIER",
        "CONTACT_HANDLE",
        "DEVICE_IDENTIFIER",
        "DOCUMENT_IDENTIFIER",
        "DOCUMENT_REFERENCE",
        "FINANCIAL_AMOUNT",
        "ORGANIZATION_IDENTIFIER",
        "PAYMENT_CARD_SECURITY",
        "PERSON_ATTRIBUTE",
        "PERSON_IDENTIFIER",
        "PERSON_ROLE_OR_TITLE",
        "PROPER_NAME",
        "VEHICLE_IDENTIFIER",
    }
)

# The GDPR Art. 9 trio that :data:`EU_PII_ENTITY_MAPPING` collapses onto ``NRP``.
# The ``gdpr_sensitive`` profile keeps them as their own descriptive entities.
_GDPR_SPECIAL_CATEGORY_NRP_LABELS = (
    "ETHNIC_ORIGIN",
    "RELIGION_OR_BELIEF",
    "POLITICAL_OPINION",
)

# Named mapping profiles. Pick one via the ``mapping_profile`` constructor
# argument (or the ``mapping_profile`` YAML field) instead of hand-building a
# full ``label_mapping``.
MAPPING_PROFILE_PRESIDIO_STANDARD = "presidio_standard"
MAPPING_PROFILE_GDPR_SENSITIVE = "gdpr_sensitive"
MAPPING_PROFILE_PRESERVE_MODEL_LABELS = "preserve_model_labels"
MAPPING_PROFILE_HIGH_RECALL = "high_recall"

MAPPING_PROFILES = frozenset(
    {
        MAPPING_PROFILE_PRESIDIO_STANDARD,
        MAPPING_PROFILE_GDPR_SENSITIVE,
        MAPPING_PROFILE_PRESERVE_MODEL_LABELS,
        MAPPING_PROFILE_HIGH_RECALL,
    }
)


def get_eu_pii_entity_mapping(
    profile: str = MAPPING_PROFILE_PRESIDIO_STANDARD,
) -> Dict[str, str]:
    """Return the model-label -> Presidio-entity mapping for a named profile.

    Profiles:

    - ``presidio_standard`` (default): :data:`EU_PII_ENTITY_MAPPING` exactly —
      faithful Presidio built-ins, the GDPR Art. 9 nationality/religion/politics
      trio collapsed onto ``NRP``, everything else descriptive. Backward
      compatible.
    - ``gdpr_sensitive``: like ``presidio_standard`` but keeps ``ETHNIC_ORIGIN``,
      ``RELIGION_OR_BELIEF`` and ``POLITICAL_OPINION`` as their own descriptive
      entities instead of collapsing them to ``NRP`` — so each special category
      can get its own anonymizer operator / retention rule.
    - ``preserve_model_labels``: every model label maps to itself (no Presidio
      remapping at all). Useful to inspect the raw model taxonomy.
    - ``high_recall``: like ``presidio_standard`` but also maps the ambiguous
      ``PROPER_NAME`` label to ``PERSON`` (other changes kept conservative).

    :param profile: One of :data:`MAPPING_PROFILES`.
    :return: A fresh, mutable mapping (safe for the caller to edit).
    :raises ValueError: If ``profile`` is not a known profile name.
    """
    if profile == MAPPING_PROFILE_PRESIDIO_STANDARD:
        return dict(EU_PII_ENTITY_MAPPING)
    if profile == MAPPING_PROFILE_GDPR_SENSITIVE:
        mapping = dict(EU_PII_ENTITY_MAPPING)
        for label in _GDPR_SPECIAL_CATEGORY_NRP_LABELS:
            mapping[label] = label
        return mapping
    if profile == MAPPING_PROFILE_PRESERVE_MODEL_LABELS:
        return {label: label for label in EU_PII_ENTITY_MAPPING}
    if profile == MAPPING_PROFILE_HIGH_RECALL:
        mapping = dict(EU_PII_ENTITY_MAPPING)
        mapping["PROPER_NAME"] = "PERSON"
        return mapping
    raise ValueError(
        f"Unknown mapping_profile {profile!r}. "
        f"Valid profiles are: {sorted(MAPPING_PROFILES)}."
    )


# Model labels for structured / checksummable PII that dedicated deterministic
# Presidio recognizers detect more reliably than a free-text NER model (regex
# plus validation: ``EmailRecognizer``, ``PhoneRecognizer``, ``IpRecognizer``,
# ``CreditCardRecognizer`` with Luhn, ``UrlRecognizer``). In a hybrid setup the
# deterministic layer owns these entities; :meth:`BardsEuPiiRecognizer.hybrid`
# passes this set as ``labels_to_ignore`` so the model side does not also emit
# them (avoiding double-detection and span disagreements).
STRUCTURED_LABELS_WITH_DETERMINISTIC_RECOGNIZERS = frozenset(
    {
        "EMAIL_ADDRESS",
        "PHONE_NUMBER",
        "IP_ADDRESS",
        "PAYMENT_CARD",
        "IDENTIFYING_LINK",
    }
)


def normalize_model_label(
    label: str, prefixes: Iterable[str] = _DEFAULT_LABEL_PREFIXES
) -> str:
    """Strip a leading BIO-style prefix (``B-``/``I-``/...) from a model label.

    Returns the model's *base* label (e.g. ``"B-PERSON_NAME"`` -> ``"PERSON_NAME"``,
    ``"O"`` -> ``"O"``). Used both to normalize ``labels_to_ignore`` before
    ``super().__init__`` runs (i.e. before the parent has set
    ``self.label_prefixes``) and to derive base labels from the live model's
    ``id2label`` in the drift check.

    :param label: A raw model label, possibly BIO-prefixed.
    :param prefixes: Prefixes to strip. Defaults to the standard BIO set.
    :return: The base label with any leading prefix removed.
    """
    if isinstance(label, str):
        for prefix in prefixes:
            if label.startswith(prefix):
                return label[len(prefix) :]
    return label


def validate_eu_pii_mapping_labels(mapping: Mapping[str, str]) -> None:
    """Validate a label mapping's keys against :data:`EXPECTED_EU_PII_MODEL_LABELS`.

    Raises if the mapping is missing any expected (pinned) model label or
    contains a key that is not an expected model label. This guards the built-in
    mapping / profiles against drift in the remote model's label set.

    :param mapping: A model-label -> Presidio-entity mapping. Keys are compared
        against the pinned expected label set as-is (they are base labels).
    :raises ValueError: If keys do not exactly match the expected label set.
    """
    keys = set(mapping.keys())
    expected = set(EXPECTED_EU_PII_MODEL_LABELS)
    missing = expected - keys
    unknown = keys - expected
    if not missing and not unknown:
        return

    details = []
    if missing:
        details.append(f"missing expected labels: {sorted(missing)}")
    if unknown:
        details.append(f"unknown labels: {sorted(unknown)}")
    raise ValueError(
        f"EU-PII label mapping does not match the pinned model label set of "
        f"{len(expected)} labels ({'; '.join(details)}). If the "
        f"{DEFAULT_EU_PII_MODEL!r} model's labels changed, update "
        f"EU_PII_ENTITY_MAPPING and EXPECTED_EU_PII_MODEL_LABELS together; for "
        f"an intentional custom mapping, pass validate_mapping=False."
    )


def _validate_threshold_value(value: float, where: str) -> None:
    """Raise ``ValueError`` unless ``value`` is a number in ``[0.0, 1.0]``.

    ``bool`` is rejected explicitly (it is a subclass of ``int``) so a stray
    ``True``/``False`` is not silently treated as ``1.0``/``0.0``.

    :param value: The candidate threshold.
    :param where: Human-readable location for the error message.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{where} must be a number between 0.0 and 1.0, got {value!r}"
        )
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(
            f"{where} must be between 0.0 and 1.0, got {value!r}"
        )


def _validate_threshold_maps(
    thresholds_by_entity: Optional[Dict[str, float]],
    thresholds_by_language: Optional[Dict[str, Dict[str, float]]],
) -> None:
    """Validate every value in both threshold maps is a float in ``[0.0, 1.0]``.

    Every value is checked (across all languages in ``thresholds_by_language``),
    not only those that apply to the instance's language, so a misconfiguration
    fails loudly at construction regardless of which language is registered.
    """
    if thresholds_by_entity:
        for entity, value in thresholds_by_entity.items():
            _validate_threshold_value(value, f"thresholds_by_entity[{entity!r}]")
    if thresholds_by_language:
        for language, entity_map in thresholds_by_language.items():
            if not isinstance(entity_map, dict):
                raise ValueError(
                    f"thresholds_by_language[{language!r}] must be a mapping of "
                    f"entity -> threshold, got {type(entity_map).__name__}"
                )
            for entity, value in entity_map.items():
                _validate_threshold_value(
                    value, f"thresholds_by_language[{language!r}][{entity!r}]"
                )


class BardsEuPiiRecognizer(HuggingFaceNerRecognizer):
    """Recognizer wrapping ``bardsai/eu-pii-anonimization-multilang``.

    An XLM-RoBERTa token-classification PII model covering 24 EU languages
    (Apache-2.0). This is a thin subclass of :class:`HuggingFaceNerRecognizer`
    that only bakes in the model id and the :data:`EU_PII_ENTITY_MAPPING`
    label mapping; all inference (loading, BIO-tag aggregation via the
    transformers pipeline, chunking, thresholding, device selection) is
    inherited from the parent.

    Optional and opt-in: it is not registered in
    ``conf/default_recognizers.yaml`` and is never part of the default registry.
    Instantiating it loads (and on first use downloads) the model, so it should
    only be created when explicitly added to an ``AnalyzerEngine`` or referenced
    in a recognizer-registry configuration. Requires the ``transformers`` extra::

        pip install 'presidio-analyzer[transformers]'

    The model is multilingual but a recognizer instance is registered for a
    single language. To cover several EU languages, register one instance per
    language (each pointing at the same model) or list multiple
    ``supported_languages`` in a YAML registry configuration.

    **Hybrid setups.** For structured/checksummable PII (e-mail, phone, IP,
    credit card, URL), dedicated deterministic recognizers are more reliable
    than this NER model. Pass ``labels_to_ignore`` (or use the
    :meth:`hybrid` constructor) so the model defers those labels to the
    deterministic layer: an ignored model label is dropped *before* it is
    mapped to a Presidio entity, so it is removed rather than emitted — and it
    does not appear in :attr:`supported_entities`. This is the safe way to
    suppress a label; mapping it to a sentinel entity is not, because when
    ``analyze`` is called with ``entities=None`` Presidio expands the request to
    every supported entity, which would let such a sentinel be returned.

    :param supported_entities: Explicit list of supported entities. If ``None``,
        derived from the values of the (post-``labels_to_ignore``) mapping.
    :param name: Recognizer name.
    :param supported_language: Language this recognizer instance supports.
    :param model_name: HuggingFace model id. Defaults to
        :data:`DEFAULT_EU_PII_MODEL`.
    :param label_mapping: Mapping from model labels (``B-``/``I-`` stripped) to
        Presidio entities. If given, it takes precedence over ``mapping_profile``.
        If ``None``, the mapping is resolved from ``mapping_profile``.
    :param mapping_profile: Named label-mapping profile (one of
        :data:`MAPPING_PROFILES`). Defaults to
        :data:`MAPPING_PROFILE_PRESIDIO_STANDARD` (backward compatible). See
        :func:`get_eu_pii_entity_mapping`. Ignored when ``label_mapping`` is set.
    :param threshold: Global minimum confidence score (0.0 - 1.0). Tunable:
        lower favors recall (compliance), higher favors precision (data utility).
        Used as the fallback when no finer threshold applies.
    :param aggregation_strategy: HuggingFace token aggregation strategy. Defaults
        to ``first``. This model's fast tokenizer does not expose real word ids,
        so the HuggingFace pipeline falls back to a heuristic; under ``simple``
        that heuristic fragments subword entities (e.g. an e-mail address is
        split into many pieces), whereas ``first`` keeps each entity as a single
        contiguous span.
    :param labels_to_ignore: Model labels (``B-``/``I-`` prefixes optional) to
        drop before entity mapping and thresholding. Use this in hybrid setups
        to hand structured PII to deterministic recognizers. See :meth:`hybrid`.
    :param thresholds_by_entity: Optional per-Presidio-entity thresholds, keyed
        by the *mapped* entity name (e.g. ``"PERSON"``, not ``"PERSON_NAME"``).
        Overrides ``threshold`` for those entities.
    :param thresholds_by_language: Optional per-language, per-entity thresholds,
        keyed by language code then mapped entity name (e.g.
        ``{"de": {"PERSON": 0.6}}``). For this instance only the sub-map for
        :attr:`supported_language` applies, and it takes precedence over
        ``thresholds_by_entity``. Resolution order per result is:
        ``thresholds_by_language[supported_language][entity]`` →
        ``thresholds_by_entity[entity]`` → ``threshold``.
    :param validate_mapping: Whether to check the mapping's keys against the
        pinned :data:`EXPECTED_EU_PII_MODEL_LABELS` (drift guard). ``None``
        (default) validates only the built-in mapping / profiles (i.e. when
        ``label_mapping`` is ``None``) and skips an explicit custom
        ``label_mapping``. ``True`` validates regardless (opt in for a custom
        mapping); ``False`` skips regardless (escape hatch). Validation runs on
        the full mapping, before ``labels_to_ignore`` is applied.

    All threshold values (global and in both maps) must be numbers in
    ``[0.0, 1.0]``; otherwise a ``ValueError`` is raised at construction.

    .. note::
        When threshold maps are supplied, the inherited ``threshold`` attribute
        holds the *lowest* configured threshold (the pre-filter floor); the
        original global value is kept as the resolution fallback. With no maps
        the two are identical, so ``threshold`` is unchanged for existing users.
    """

    def __init__(
        self,
        supported_entities: Optional[List[str]] = None,
        name: str = "BardsEuPiiRecognizer",
        supported_language: str = "en",
        model_name: str = DEFAULT_EU_PII_MODEL,
        label_mapping: Optional[Dict[str, str]] = None,
        threshold: float = 0.4,
        aggregation_strategy: str = "first",
        labels_to_ignore: Optional[Iterable[str]] = None,
        thresholds_by_entity: Optional[Dict[str, float]] = None,
        thresholds_by_language: Optional[Dict[str, Dict[str, float]]] = None,
        mapping_profile: str = MAPPING_PROFILE_PRESIDIO_STANDARD,
        validate_mapping: Optional[bool] = None,
        **kwargs,
    ):
        # An explicit label_mapping wins; otherwise resolve from the named
        # profile (which validates the name and raises on an unknown profile).
        using_builtin_mapping = label_mapping is None
        base_mapping = (
            dict(label_mapping)
            if label_mapping is not None
            else get_eu_pii_entity_mapping(mapping_profile)
        )

        # Guard the built-in mapping / profiles against remote label drift.
        # Default: validate only the built-in (a custom mapping is the user's
        # responsibility). ``validate_mapping`` can force it on/off. Validate the
        # full base mapping, before ``labels_to_ignore`` removes any keys.
        should_validate_mapping = (
            validate_mapping
            if validate_mapping is not None
            else using_builtin_mapping
        )
        if should_validate_mapping:
            validate_eu_pii_mapping_labels(base_mapping)

        # Normalize ignored labels to base labels (strip BIO prefixes) using the
        # same prefixes the parent strips at inference time. This must happen
        # before ``super().__init__`` because that call triggers ``load()`` (via
        # ``EntityRecognizer.__init__``), and the overridden ``load`` reads these
        # attributes.
        prefixes = kwargs.get("label_prefixes") or _DEFAULT_LABEL_PREFIXES
        self.labels_to_ignore = frozenset(
            normalize_model_label(label, prefixes)
            for label in (labels_to_ignore or ())
        )
        self._ignore_filter_installed = False

        # Drop ignored labels from the active mapping so they (a) never appear in
        # the derived ``supported_entities`` and (b) cannot be re-introduced by
        # the parent's unmapped-label discovery pass-through. Actual suppression
        # of predictions happens in ``_install_ignore_filter``.
        effective_mapping = {
            model_label: entity
            for model_label, entity in base_mapping.items()
            if model_label not in self.labels_to_ignore
        }

        # Per-entity / per-language thresholds. The global ``threshold`` is the
        # fallback; the maps refine it. Keyed by *mapped* Presidio entity, so
        # resolution in ``analyze`` happens on ``RecognizerResult.entity_type``.
        _validate_threshold_maps(thresholds_by_entity, thresholds_by_language)
        self._global_threshold = threshold
        self.thresholds_by_entity = (
            dict(thresholds_by_entity) if thresholds_by_entity else None
        )
        self.thresholds_by_language = (
            {
                lang: dict(entity_map)
                for lang, entity_map in thresholds_by_language.items()
            }
            if thresholds_by_language
            else None
        )
        self._has_custom_thresholds = bool(
            self.thresholds_by_entity or self.thresholds_by_language
        )

        # The parent pre-filters predictions at its single ``threshold`` inside
        # ``_predict_chunk`` — before we can apply a per-entity threshold. So pass
        # the *lowest* threshold that can apply for this language as the parent's
        # floor, then re-apply the precise per-entity/per-language threshold as a
        # post-filter in ``analyze``. With no custom maps the floor equals the
        # global threshold, so behavior is byte-for-byte unchanged.
        floor = threshold
        if self.thresholds_by_entity:
            floor = min(floor, min(self.thresholds_by_entity.values()))
        language_map = (self.thresholds_by_language or {}).get(supported_language)
        if language_map:
            floor = min(floor, min(language_map.values()))

        super().__init__(
            supported_entities=supported_entities,
            name=name,
            supported_language=supported_language,
            model_name=model_name,
            label_mapping=effective_mapping,
            threshold=floor,
            aggregation_strategy=aggregation_strategy,
            **kwargs,
        )

    @classmethod
    def hybrid(
        cls,
        labels_to_ignore: Optional[Iterable[str]] = None,
        **kwargs,
    ) -> "BardsEuPiiRecognizer":
        """Construct an instance tuned for a hybrid (model + deterministic) setup.

        In a hybrid pipeline, deterministic Presidio recognizers own the
        structured/checksummable entities (e-mail, phone, IP, credit card, URL)
        while this model owns the free-text NER entities (``PERSON``,
        ``LOCATION``, ...). This constructor ignores the structured model labels
        in :data:`STRUCTURED_LABELS_WITH_DETERMINISTIC_RECOGNIZERS` so the two
        layers do not double-detect or disagree on those spans.

        Any ``labels_to_ignore`` passed here is merged with (not a replacement
        for) the structured defaults::

            BardsEuPiiRecognizer.hybrid(labels_to_ignore=["FINANCIAL_AMOUNT"])

        :param labels_to_ignore: Additional model labels to ignore on top of
            :data:`STRUCTURED_LABELS_WITH_DETERMINISTIC_RECOGNIZERS`.
        :param kwargs: Forwarded to :class:`BardsEuPiiRecognizer`.
        :return: A configured :class:`BardsEuPiiRecognizer`.
        """
        merged = set(STRUCTURED_LABELS_WITH_DETERMINISTIC_RECOGNIZERS)
        merged.update(labels_to_ignore or ())
        return cls(labels_to_ignore=merged, **kwargs)

    def load(self) -> None:
        """Load the pipeline, then install the ignored-label filter once."""
        super().load()
        if self.labels_to_ignore and not self._ignore_filter_installed:
            self._install_ignore_filter()

    def _install_ignore_filter(self) -> None:
        """Wrap the loaded pipeline so ignored model labels are dropped at source.

        The drop happens on the raw pipeline predictions, before the parent's
        ``_predict_chunk`` maps labels to entities or applies the score
        threshold. This is the only place a prediction can be *removed* (rather
        than mapped): the parent keeps unmapped labels as descriptive
        pass-through entities, so omitting a label from ``label_mapping`` alone
        does not suppress it.
        """
        inner_pipeline = self.ner_pipeline
        ignored = self.labels_to_ignore
        normalize = self._normalize_label

        def pipeline_dropping_ignored_labels(*args, **kwargs):
            preds = inner_pipeline(*args, **kwargs)
            if not isinstance(preds, list):
                return preds
            return [
                pred
                for pred in preds
                if not (
                    isinstance(pred, dict)
                    and normalize(pred.get("entity_group") or pred.get("entity") or "")
                    in ignored
                )
            ]

        self.ner_pipeline = pipeline_dropping_ignored_labels
        self._ignore_filter_installed = True

    def _resolve_threshold(self, presidio_entity: str) -> float:
        """Resolve the confidence threshold for a mapped Presidio entity.

        Resolution order (most specific wins):

        1. ``thresholds_by_language[supported_language][presidio_entity]``
        2. ``thresholds_by_entity[presidio_entity]``
        3. the global ``threshold``

        :param presidio_entity: The mapped entity type (e.g. ``"PERSON"``).
        :return: The threshold to apply to that entity.
        """
        if self.thresholds_by_language:
            language_map = self.thresholds_by_language.get(self.supported_language)
            if language_map and presidio_entity in language_map:
                return language_map[presidio_entity]
        if self.thresholds_by_entity and presidio_entity in self.thresholds_by_entity:
            return self.thresholds_by_entity[presidio_entity]
        return self._global_threshold

    def analyze(
        self,
        text: str,
        entities: List[str],
        nlp_artifacts: Optional[NlpArtifacts] = None,
    ) -> List[RecognizerResult]:
        """Analyze ``text``, applying per-entity / per-language thresholds.

        The parent does inference, label mapping and the requested-entity
        filter, pre-filtering at the (floored) parent ``threshold``. When custom
        threshold maps are configured this method re-applies the precise
        per-entity / per-language threshold to the mapped results. With no maps
        it is a pass-through, so default behavior is unchanged.
        """
        results = super().analyze(text, entities, nlp_artifacts)
        if not self._has_custom_thresholds:
            return results
        return [
            result
            for result in results
            if result.score >= self._resolve_threshold(result.entity_type)
        ]
