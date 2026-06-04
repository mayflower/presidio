"""Recognizer for the ``bardsai/eu-pii-anonimization-multilang`` model.

This is a thin wrapper around :class:`HuggingFaceNerRecognizer`. The model is a
standard ``XLMRobertaForTokenClassification`` token-classification checkpoint
(BIO tags, 35 PII classes, 24 EU languages, Apache-2.0), so the existing
HuggingFace NER recognizer already handles loading, inference, BIO-tag merging,
chunking, scoring and device selection. This module only contributes the default
model id and a curated mapping from the model's labels to Presidio entities.
"""

from types import MappingProxyType
from typing import Dict, List, Mapping, Optional

from presidio_analyzer.predefined_recognizers.ner.huggingface_ner_recognizer import (
    HuggingFaceNerRecognizer,
)

DEFAULT_EU_PII_MODEL = "bardsai/eu-pii-anonimization-multilang"

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

    :param supported_entities: Explicit list of supported entities. If ``None``,
        derived from the values of ``label_mapping``.
    :param name: Recognizer name.
    :param supported_language: Language this recognizer instance supports.
    :param model_name: HuggingFace model id. Defaults to
        :data:`DEFAULT_EU_PII_MODEL`.
    :param label_mapping: Mapping from model labels (``B-``/``I-`` stripped) to
        Presidio entities. Defaults to a copy of :data:`EU_PII_ENTITY_MAPPING`.
    :param threshold: Minimum confidence score (0.0 - 1.0). Tunable: lower favors
        recall (compliance), higher favors precision (data utility).
    :param aggregation_strategy: HuggingFace token aggregation strategy. Defaults
        to ``first``. This model's fast tokenizer does not expose real word ids,
        so the HuggingFace pipeline falls back to a heuristic; under ``simple``
        that heuristic fragments subword entities (e.g. an e-mail address is
        split into many pieces), whereas ``first`` keeps each entity as a single
        contiguous span.
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
        **kwargs,
    ):
        super().__init__(
            supported_entities=supported_entities,
            name=name,
            supported_language=supported_language,
            model_name=model_name,
            label_mapping=(
                dict(label_mapping)
                if label_mapping is not None
                else dict(EU_PII_ENTITY_MAPPING)
            ),
            threshold=threshold,
            aggregation_strategy=aggregation_strategy,
            **kwargs,
        )
