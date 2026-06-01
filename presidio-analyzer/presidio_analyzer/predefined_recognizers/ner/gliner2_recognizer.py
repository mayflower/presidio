import json
import logging
from types import MappingProxyType
from typing import Dict, List, Literal, Mapping, Optional

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

#: Valid values for ``GLiNER2Recognizer(label_selection_strategy=...)``. They
#: control which model labels are sent to the model at ``analyze()`` time:
#: ``"all_configured"`` queries every configured label (the original behavior);
#: ``"requested_presidio_entities"`` queries only configured labels whose mapped
#: Presidio entity was requested (falling back to all configured when nothing is
#: requested); ``"configured_ner_only"`` queries exactly the configured labels
#: and never appends ad-hoc requested entities.
LABEL_SELECTION_STRATEGIES = (
    "all_configured",
    "requested_presidio_entities",
    "configured_ner_only",
)

#: Built-in mapping from the 42 PII labels emitted by
#: ``fastino/gliner2-privacy-filter-PII-multi`` to Presidio entity types.
#: Used as the default when neither ``entity_mapping`` nor
#: ``supported_entities`` is provided. Pass a custom ``entity_mapping`` to
#: override it (e.g. to use a different model or your own entity names).
#: Many values reuse Presidio's standard entities (PERSON, EMAIL_ADDRESS,
#: PHONE_NUMBER, LOCATION, CREDIT_CARD, IBAN_CODE, IP_ADDRESS, DATE_TIME), but
#: some are model-specific names with no built-in recognizer/operator
#: (e.g. USERNAME, PASSWORD, API_KEY, GOVERNMENT_ID, PASSPORT); override the
#: mapping if you need to align them with your own taxonomy.
#: Exposed as a read-only mapping; copy it (``dict(GLINER2_PII_ENTITY_MAPPING)``)
#: before mutating.
GLINER2_PII_ENTITY_MAPPING: Mapping[str, str] = MappingProxyType(
    {
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
)


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
        add_requested_entities: bool = True,
        label_thresholds: Optional[Dict[str, float]] = None,
        label_selection_strategy: Literal[
            "all_configured",
            "requested_presidio_entities",
            "configured_ner_only",
        ] = "requested_presidio_entities",
        model_threshold: Optional[float] = None,
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
            language description (e.g. ``{"email": "an email address"}``), which
            can improve recall/precision for ambiguous labels. Keys should match
            the model labels (the keys of ``entity_mapping``). Descriptions are
            overlaid onto the full label set at analysis time: labels without a
            description (and ad-hoc entities requested at ``analyze()`` time) are
            still queried as bare labels, so providing descriptions for a subset
            does not stop the other labels from being detected.
        :param add_requested_entities: When True (default), entity types requested
            at ``analyze()`` time that are not in ``entity_mapping`` are added to
            the model's query as ad-hoc labels (zero-shot). Set to False to keep
            the recognizer strictly within ``entity_mapping`` — useful in a mixed
            registry where other recognizers (e.g. regex/checksum ones) own those
            entity types, so GLiNER2 does not also try to detect them.
        :param label_thresholds: Optional per-model-label minimum confidence
            (e.g. ``{"person": 0.85, "username": 0.4}``). A match is kept only if
            its confidence is >= the label's threshold; labels without an entry
            fall back to ``threshold``. Lets you raise precision on noisy labels
            (person/name, phone) without lowering recall on clean ones.
        :param label_selection_strategy: Controls which model labels are queried
            for a given ``analyze()`` call (see :data:`LABEL_SELECTION_STRATEGIES`).
            ``"requested_presidio_entities"`` (default) queries only configured
            labels whose mapped Presidio entity is in the requested ``entities``
            (improving precision by not asking the model for unrequested types),
            falling back to all configured labels when nothing is requested.
            ``"all_configured"`` reproduces the original behavior (all configured
            labels). ``"configured_ner_only"`` queries exactly the configured
            labels and never appends ad-hoc requested entities.
        :param model_threshold: Optional confidence threshold passed to the model's
            ``extract_entities``. If None and ``label_thresholds`` are set, the
            model is queried at ``min(threshold, min(label_thresholds.values()))``
            so no candidate is pre-filtered below a label's cutoff (the per-label
            cutoffs are then applied as a post-filter); otherwise ``threshold``.
        :param text_chunker: Custom text chunking strategy. If None, uses
            CharacterBasedTextChunker with default settings (chunk_size=250,
            chunk_overlap=50)
        :param model_kwargs: Additional keyword arguments to pass to
            ``GLiNER2.from_pretrained`` (e.g. ``quantize`` or ``compile``).
        """
        if entity_mapping is not None and supported_entities is not None:
            # Identity-based check, matching GLiNER2RecognizerConfig's validator,
            # so the constraint behaves the same whether built directly or via YAML.
            raise ValueError(
                "entity_mapping and supported_entities cannot be used together"
            )

        if entity_mapping:
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

        # A YAML/no-code config passes unset fields as None; treat that as the
        # default (matching how the other optional config fields behave).
        if label_selection_strategy is None:
            label_selection_strategy = "requested_presidio_entities"
        if label_selection_strategy not in LABEL_SELECTION_STRATEGIES:
            raise ValueError(
                "label_selection_strategy must be one of "
                f"{LABEL_SELECTION_STRATEGIES}, got {label_selection_strategy!r}"
            )

        self.threshold = threshold
        self.label_descriptions = label_descriptions
        self.add_requested_entities = add_requested_entities
        self.label_thresholds = dict(label_thresholds) if label_thresholds else {}
        self.label_selection_strategy = label_selection_strategy
        self.model_threshold = model_threshold
        self.model_kwargs = model_kwargs

        # Threshold passed to the model's ``extract_entities``. When per-label
        # thresholds are used we query the model at the lowest of them (and the
        # base threshold) so no candidate is pre-filtered below a label's cutoff;
        # the per-label cutoffs are then re-applied as a post-filter in analyze().
        if model_threshold is not None:
            self._model_query_threshold = model_threshold
        elif self.label_thresholds:
            self._model_query_threshold = min(
                self.threshold, min(self.label_thresholds.values())
            )
        else:
            self._model_query_threshold = self.threshold

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
        labels = self.__create_input_labels(entities)

        # When label descriptions are supplied, build the GLiNER2 schema by
        # overlaying the descriptions onto the *full* label set (the model
        # accepts a {label: description} mapping). Labels without a description
        # fall back to the bare label, so configured and ad-hoc labels are still
        # queried instead of being dropped.
        if self.label_descriptions:
            schema = {
                label: self.label_descriptions.get(label, label) for label in labels
            }
        else:
            schema = labels

        # Process text with automatic chunking
        def predict_func(text: str) -> List[RecognizerResult]:
            # GLiNER2 returns {"entities": {label: [{text, confidence, start, end}]}};
            # only start/end/confidence are consumed below.
            raw_predictions = self.gliner2.extract_entities(
                text,
                schema,
                threshold=self._model_query_threshold,
                include_confidence=True,
                include_spans=True,
            )
            if isinstance(raw_predictions, dict):
                # Accept both the wrapped ({"entities": {...}}) and the
                # already-unwrapped ({label: [...]}) shapes.
                entities_by_label = raw_predictions.get("entities", raw_predictions)
            else:
                # An unexpected output shape would otherwise be silently treated
                # as "no PII found"; surface it instead.
                logger.warning(
                    "GLiNER2 returned unexpected output of type %s (expected dict); "
                    "treating as no detections. model=%s",
                    type(raw_predictions).__name__,
                    self.model_name,
                )
                entities_by_label = {}

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
                        # We request include_spans=True, so a missing span means
                        # the model dropped a real detection's offsets. We cannot
                        # place it in the text (guessing with text.find() would
                        # misplace redactions), so skip it -- but at warning level,
                        # because a detected PII entity is being discarded.
                        logger.warning(
                            "GLiNER2 returned entity for label '%s' without a "
                            "start/end span despite include_spans=True; dropping "
                            "this detection: %s",
                            label,
                            match,
                        )
                        continue

                    # Fall back to the recognizer threshold when the model does
                    # not return a confidence for this entity. Every returned
                    # match already passed the model's threshold filter, so the
                    # threshold is the correct conservative lower-bound score.
                    score = match.get("confidence")
                    if score is None:
                        logger.debug(
                            "GLiNER2 match for label '%s' missing confidence; "
                            "defaulting score to threshold %s",
                            label,
                            self.threshold,
                        )
                        score = self.threshold
                    score = float(score)

                    # Per-label precision control: drop matches below the label's
                    # own threshold (falling back to the base threshold).
                    effective_threshold = self.label_thresholds.get(
                        label, self.threshold
                    )
                    if score < effective_threshold:
                        continue

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
        """Build the model label list according to ``label_selection_strategy``.

        - ``"requested_presidio_entities"``: keep only configured model labels
          whose mapped Presidio entity is in ``entities``; if no entities are
          requested, keep all configured labels.
        - ``"all_configured"`` / ``"configured_ner_only"``: keep all configured
          labels.

        Then, unless the strategy is ``"configured_ner_only"`` or
        ``add_requested_entities`` is False, append each requested entity that is
        neither a known model label nor an already-mapped Presidio entity value
        (ad-hoc / zero-shot labels). An empty result falls back to all configured
        labels so the model is never queried with an empty schema.
        """
        if self.label_selection_strategy == "requested_presidio_entities" and entities:
            labels = [
                label
                for label in self.gliner2_labels
                if str(self.model_to_presidio_entity_mapping.get(label, label))
                in entities
            ]
        else:
            labels = list(self.gliner2_labels)

        if self.add_requested_entities and (
            self.label_selection_strategy != "configured_ner_only"
        ):
            for entity in entities:
                if (
                    entity not in self.model_to_presidio_entity_mapping.values()
                    and entity not in self.gliner2_labels
                ):
                    labels.append(entity)

        # Never query the model with an empty schema.
        if not labels:
            labels = list(self.gliner2_labels)
        return labels
