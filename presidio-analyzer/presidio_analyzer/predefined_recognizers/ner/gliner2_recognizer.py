import json
import logging
from typing import Dict, List, Optional

from presidio_analyzer import (
    AnalysisExplanation,
    LocalRecognizer,
    RecognizerResult,
)
from presidio_analyzer.chunkers import BaseTextChunker
from presidio_analyzer.nlp_engine import (
    NlpArtifacts,
    device_detector,
)

try:
    from gliner2 import GLiNER2
except ImportError:
    GLiNER2 = None

logger = logging.getLogger("presidio-analyzer")

#: Default model id. ``fastino/gliner2-privacy-filter-PII-multi`` is a
#: multilingual PII detection model built on the GLiNER2 architecture.
#: It is *not* compatible with the ``gliner`` library used by
#: :class:`GLiNERRecognizer`; it requires the ``gliner2`` library.
DEFAULT_GLINER2_MODEL = "fastino/gliner2-privacy-filter-PII-multi"

#: Built-in mapping from the 42 PII labels emitted by
#: ``fastino/gliner2-privacy-filter-PII-multi`` to Presidio entity types.
#: Used as the default when neither ``entity_mapping`` nor
#: ``supported_entities`` is provided. Pass a custom ``entity_mapping`` to
#: override it (e.g. to use a different model or your own entity names).
GLINER2_PII_ENTITY_MAPPING: Dict[str, str] = {
    # Person / names
    "person": "PERSON",
    "full_name": "PERSON",
    "first_name": "PERSON",
    "middle_name": "PERSON",
    "last_name": "PERSON",
    "date_of_birth": "DATE_TIME",
    # Contact / address
    "email": "EMAIL_ADDRESS",
    "phone_number": "PHONE_NUMBER",
    "address": "LOCATION",
    "street_address": "LOCATION",
    "city": "LOCATION",
    "state_or_region": "LOCATION",
    "postal_code": "LOCATION",
    "country": "LOCATION",
    # Government / tax IDs
    "government_id": "GOVERNMENT_ID",
    "national_id_number": "NATIONAL_ID",
    "passport_number": "PASSPORT",
    "drivers_license_number": "DRIVER_LICENSE",
    "license_number": "LICENSE_NUMBER",
    "tax_id": "TAX_ID",
    "tax_number": "TAX_ID",
    # Banking / payment
    "bank_account": "BANK_ACCOUNT",
    "account_number": "BANK_ACCOUNT",
    "routing_number": "BANK_ROUTING_NUMBER",
    "iban": "IBAN_CODE",
    "payment_card": "CREDIT_CARD",
    "card_number": "CREDIT_CARD",
    "card_expiry": "CREDIT_CARD_EXPIRATION",
    "card_cvv": "CREDIT_CARD_CVV",
    # Digital identity
    "username": "USERNAME",
    "ip_address": "IP_ADDRESS",
    "account_id": "ACCOUNT_ID",
    "sensitive_account_id": "ACCOUNT_ID",
    # Secrets / credentials
    "password": "PASSWORD",
    "secret": "SECRET",
    "api_key": "API_KEY",
    "access_token": "ACCESS_TOKEN",
    "recovery_code": "RECOVERY_CODE",
    # Sensitive dates
    "sensitive_date": "DATE_TIME",
    "document_date": "DATE_TIME",
    "expiration_date": "DATE_TIME",
    "transaction_date": "DATE_TIME",
}


class GLiNER2Recognizer(LocalRecognizer):
    """GLiNER2 model based entity recognizer.

    This recognizer uses the `gliner2 <https://pypi.org/project/gliner2/>`_
    library (``GLiNER2.from_pretrained`` + ``GLiNER2.extract_entities``).

    It is a separate recognizer from :class:`GLiNERRecognizer` because the
    GLiNER2 architecture is *not* loadable by the ``gliner`` library:
    ``gliner.GLiNER.from_pretrained`` raises ``FileNotFoundError`` on a GLiNER2
    checkpoint (the repo ships a different config layout). The two libraries
    also expose different inference APIs (``predict_entities`` returns a flat
    list of dicts, while ``extract_entities`` returns a label-keyed mapping).

    The default model, ``fastino/gliner2-privacy-filter-PII-multi``, detects 42
    PII entity types. Like ``GLiNERRecognizer`` this recognizer is opt-in: it is
    not part of Presidio's default configuration and must be added to the
    registry explicitly.
    """

    def __init__(
        self,
        supported_entities: Optional[List[str]] = None,
        name: str = "GLiNER2Recognizer",
        supported_language: str = "en",
        version: str = "0.0.1",
        context: Optional[List[str]] = None,
        entity_mapping: Optional[Dict[str, str]] = None,
        model_name: str = DEFAULT_GLINER2_MODEL,
        threshold: float = 0.5,
        map_location: Optional[str] = None,
        label_descriptions: Optional[Dict[str, str]] = None,
        text_chunker: Optional[BaseTextChunker] = None,
        **model_kwargs,
    ):
        """GLiNER2 model based entity recognizer.

        :param supported_entities: List of supported entities for this
            recognizer. Cannot be combined with ``entity_mapping``. If provided,
            each entity is mapped to itself (the model is asked for that label
            verbatim).
        :param name: Name of the recognizer
        :param supported_language: Language code to use for the recognizer
        :param version: Version of the recognizer
        :param context: N/A for this recognizer
        :param entity_mapping: Mapping from the model's output labels to Presidio
            entity types (e.g. ``{"email": "EMAIL_ADDRESS"}``). If omitted and no
            ``supported_entities`` are given, the built-in
            :data:`GLINER2_PII_ENTITY_MAPPING` is used.
        :param model_name: The name of the GLiNER2 model to load
        :param threshold: The confidence threshold for the model's output
            (see GLiNER2's documentation)
        :param map_location: The device to use for the model
            (e.g. ``"cpu"`` or ``"cuda"``). If None, will auto-detect GPU or use
            CPU.
        :param label_descriptions: Optional mapping from model label to a natural
            language description (e.g. ``{"email": "an email address"}``). When
            provided, the descriptions are passed to GLiNER2's schema instead of
            bare labels, which can improve recall/precision for ambiguous labels.
            The keys should match the model labels (the keys of ``entity_mapping``).
        :param text_chunker: Custom text chunking strategy. If None, uses
            CharacterBasedTextChunker with default settings (chunk_size=250,
            chunk_overlap=50)
        :param model_kwargs: Additional keyword arguments to pass to
            ``GLiNER2.from_pretrained`` (e.g. ``quantize`` or ``compile``).
        """
        if entity_mapping:
            if supported_entities:
                raise ValueError(
                    "entity_mapping and supported_entities cannot be used together"
                )

            self.model_to_presidio_entity_mapping = entity_mapping
        elif supported_entities:
            self.model_to_presidio_entity_mapping = {
                entity: entity for entity in supported_entities
            }
        else:
            logger.info(
                "No entity mapping or supported entities provided, "
                "using the built-in GLiNER2 PII entity mapping"
            )
            self.model_to_presidio_entity_mapping = dict(GLINER2_PII_ENTITY_MAPPING)

        logger.info(
            "Using entity mapping "
            f"{json.dumps(self.model_to_presidio_entity_mapping, indent=2)}"
        )
        supported_entities = list(set(self.model_to_presidio_entity_mapping.values()))
        self.model_name = model_name

        self.map_location = (
            map_location if map_location is not None else device_detector.get_device()
        )

        self.threshold = threshold
        self.label_descriptions = label_descriptions
        self.model_kwargs = model_kwargs

        # Use provided chunker or default to in-house character-based chunker
        if text_chunker is not None:
            self.text_chunker = text_chunker
        else:
            from presidio_analyzer.chunkers import CharacterBasedTextChunker

            self.text_chunker = CharacterBasedTextChunker(
                chunk_size=250,
                chunk_overlap=50,
            )

        self.gliner2 = None

        super().__init__(
            supported_entities=supported_entities,
            name=name,
            supported_language=supported_language,
            version=version,
            context=context,
        )

        self.gliner2_labels = list(self.model_to_presidio_entity_mapping.keys())

    def load(self) -> None:
        """Load the GLiNER2 model."""
        if not GLiNER2:
            raise ImportError(
                "gliner2 is not installed. "
                "Please install it with `pip install 'presidio-analyzer[gliner2]'`."
            )

        self.gliner2 = GLiNER2.from_pretrained(
            self.model_name,
            map_location=self.map_location,
            **self.model_kwargs,
        )

    def analyze(
        self,
        text: str,
        entities: List[str],
        nlp_artifacts: Optional[NlpArtifacts] = None,
    ) -> List[RecognizerResult]:
        """Analyze text to identify entities using a GLiNER2 model.

        :param text: The text to be analyzed
        :param entities: The list of entities this recognizer is requested to return
        :param nlp_artifacts: N/A for this recognizer
        """

        # Combine the input labels as this model allows for ad-hoc labels.
        # When label descriptions are supplied, pass them through to GLiNER2's
        # schema instead of bare labels (the model accepts a {label: description}
        # mapping as its entity schema).
        labels = self.__create_input_labels(entities)
        labels_or_label_descriptions = self.label_descriptions or labels

        # Process text with automatic chunking
        def predict_func(text: str) -> List[RecognizerResult]:
            # GLiNER2 returns {"entities": {label: [{text, confidence, start, end}]}}
            raw_predictions = self.gliner2.extract_entities(
                text,
                labels_or_label_descriptions,
                threshold=self.threshold,
                include_confidence=True,
                include_spans=True,
            )
            entities_by_label = (
                raw_predictions.get("entities", raw_predictions)
                if isinstance(raw_predictions, dict)
                else {}
            )

            results = []
            for label, matches in entities_by_label.items():
                presidio_entity = str(
                    self.model_to_presidio_entity_mapping.get(label, label)
                )

                # Filter by requested entities
                if entities and presidio_entity not in entities:
                    continue

                for match in matches:
                    start = match.get("start")
                    end = match.get("end")
                    if start is None or end is None:
                        # Without spans we cannot place the entity in the text.
                        # Skip rather than guessing with a naive text.find().
                        logger.debug(
                            "Skipping GLiNER2 entity without start/end span: %s",
                            match,
                        )
                        continue

                    # Fall back to the recognizer threshold when the model does
                    # not return a confidence for this entity.
                    score = match.get("confidence")
                    if score is None:
                        score = self.threshold
                    score = float(score)

                    analysis_explanation = AnalysisExplanation(
                        recognizer=self.name,
                        original_score=score,
                        textual_explanation=(
                            f"Identified as {presidio_entity} by GLiNER2"
                        ),
                    )

                    results.append(
                        RecognizerResult(
                            entity_type=presidio_entity,
                            start=int(start),
                            end=int(end),
                            score=score,
                            analysis_explanation=analysis_explanation,
                        )
                    )
            return results

        predictions = self.text_chunker.predict_with_chunking(
            text=text,
            predict_func=predict_func,
        )

        return predictions

    def __create_input_labels(self, entities):
        """Append the entities requested by the user to the list of labels if it's not there."""  # noqa: E501
        labels = list(self.gliner2_labels)
        for entity in entities:
            if (
                entity not in self.model_to_presidio_entity_mapping.values()
                and entity not in self.gliner2_labels
            ):
                labels.append(entity)
        return labels
