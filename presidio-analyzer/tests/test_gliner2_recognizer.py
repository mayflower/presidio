import ast
import os
import re
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from presidio_analyzer.chunkers import CharacterBasedTextChunker
from presidio_analyzer.predefined_recognizers import GLiNER2Recognizer
from presidio_analyzer.predefined_recognizers.ner.gliner2_recognizer import (
    GLINER2_PII_ENTITY_MAPPING,
)

GLINER2_MODULE = "presidio_analyzer.predefined_recognizers.ner.gliner2_recognizer"

# All unit tests in this module are CI-safe: they never download a model and
# never touch the network. The GLiNER2 class is replaced with a mock (see the
# ``mock_gliner2`` fixture / explicit ``patch`` calls) so the heavy ``gliner2``
# package does not even need to be installed. The single real-model test at the
# bottom is opt-in via PRESIDIO_RUN_FASTINO_GLINER2_INTEGRATION=1.


@pytest.fixture
def mock_gliner2():
    """Patch the module-level GLiNER2 class.

    ``EntityRecognizer.__init__`` calls ``load()``, which calls
    ``GLiNER2.from_pretrained``; patching the symbol means the recognizer's
    ``self.gliner2`` becomes the returned mock instance, so tests run without
    the (heavy) ``gliner2`` package installed.
    """
    mock_instance = MagicMock()
    with patch(f"{GLINER2_MODULE}.GLiNER2") as mock_cls:
        mock_cls.from_pretrained.return_value = mock_instance
        # expose the class mock for assertions on from_pretrained
        mock_instance.from_pretrained_cls = mock_cls
        yield mock_instance


def _entities_payload(by_label):
    """Wrap label->matches in the GLiNER2 extract_entities output shape."""
    return {"entities": by_label}


def test_analyze_maps_labels_and_preserves_indexes(mock_gliner2):
    text = "Email john.smith@acme.com or call +1 415 555 0199."
    mock_gliner2.extract_entities.return_value = _entities_payload(
        {
            "email": [
                {
                    "text": "john.smith@acme.com",
                    "confidence": 0.999,
                    "start": 6,
                    "end": 25,
                }
            ],
            "phone_number": [
                {
                    "text": "+1 415 555 0199",
                    "confidence": 0.998,
                    "start": 34,
                    "end": 49,
                }
            ],
            "person": [],
        }
    )

    recognizer = GLiNER2Recognizer(
        entity_mapping={
            "email": "EMAIL_ADDRESS",
            "phone_number": "PHONE_NUMBER",
            "person": "PERSON",
        },
    )

    results = recognizer.analyze(text, ["EMAIL_ADDRESS", "PHONE_NUMBER"])
    results = sorted(results, key=lambda r: r.start)

    assert len(results) == 2

    assert results[0].entity_type == "EMAIL_ADDRESS"
    assert (results[0].start, results[0].end) == (6, 25)
    assert text[results[0].start:results[0].end] == "john.smith@acme.com"
    assert results[0].score == pytest.approx(0.999, rel=1e-2)

    assert results[1].entity_type == "PHONE_NUMBER"
    assert (results[1].start, results[1].end) == (34, 49)
    assert text[results[1].start:results[1].end] == "+1 415 555 0199"


def test_default_pii_mapping_used_when_no_args(mock_gliner2):
    recognizer = GLiNER2Recognizer()

    # Built-in PII mapping is applied
    assert recognizer.model_to_presidio_entity_mapping == GLINER2_PII_ENTITY_MAPPING
    assert "email" in recognizer.gliner2_labels
    assert "phone_number" in recognizer.gliner2_labels

    supported = set(recognizer.get_supported_entities())
    assert {"EMAIL_ADDRESS", "PHONE_NUMBER", "PERSON", "IP_ADDRESS"} <= supported


def test_default_model_name(mock_gliner2):
    recognizer = GLiNER2Recognizer()
    assert recognizer.model_name == "fastino/gliner2-privacy-filter-PII-multi"


def test_entity_mapping_and_supported_entities_conflict(mock_gliner2):
    with pytest.raises(ValueError):
        GLiNER2Recognizer(
            entity_mapping={"email": "EMAIL_ADDRESS"},
            supported_entities=["EMAIL_ADDRESS"],
        )


def test_supported_entities_map_to_themselves(mock_gliner2):
    recognizer = GLiNER2Recognizer(supported_entities=["EMAIL_ADDRESS", "PHONE_NUMBER"])
    assert recognizer.model_to_presidio_entity_mapping == {
        "EMAIL_ADDRESS": "EMAIL_ADDRESS",
        "PHONE_NUMBER": "PHONE_NUMBER",
    }


def test_analyze_filters_unrequested_entities(mock_gliner2):
    mock_gliner2.extract_entities.return_value = _entities_payload(
        {
            "person": [
                {"text": "John Doe", "confidence": 0.9, "start": 0, "end": 8}
            ],
            "email": [
                {"text": "a@b.com", "confidence": 0.9, "start": 12, "end": 19}
            ],
        }
    )

    recognizer = GLiNER2Recognizer(
        entity_mapping={"person": "PERSON", "email": "EMAIL_ADDRESS"},
    )

    results = recognizer.analyze("John Doe at a@b.com", ["EMAIL_ADDRESS"])

    assert len(results) == 1
    assert results[0].entity_type == "EMAIL_ADDRESS"


def test_analyze_supports_unwrapped_output(mock_gliner2):
    """Defensive: accept a label-keyed dict without the 'entities' wrapper."""
    mock_gliner2.extract_entities.return_value = {
        "email": [{"text": "a@b.com", "confidence": 0.9, "start": 0, "end": 7}]
    }

    recognizer = GLiNER2Recognizer(entity_mapping={"email": "EMAIL_ADDRESS"})
    results = recognizer.analyze("a@b.com here", ["EMAIL_ADDRESS"])

    assert len(results) == 1
    assert results[0].entity_type == "EMAIL_ADDRESS"
    assert (results[0].start, results[0].end) == (0, 7)


def test_analyze_with_no_results(mock_gliner2):
    mock_gliner2.extract_entities.return_value = _entities_payload({"email": []})
    recognizer = GLiNER2Recognizer(entity_mapping={"email": "EMAIL_ADDRESS"})

    assert recognizer.analyze("nothing here", ["EMAIL_ADDRESS"]) == []


def test_extract_entities_called_with_confidence_and_spans(mock_gliner2):
    mock_gliner2.extract_entities.return_value = _entities_payload({"email": []})
    recognizer = GLiNER2Recognizer(
        entity_mapping={"email": "EMAIL_ADDRESS"}, threshold=0.7
    )

    recognizer.analyze("text", ["EMAIL_ADDRESS"])

    _, kwargs = mock_gliner2.extract_entities.call_args
    assert kwargs["include_confidence"] is True
    assert kwargs["include_spans"] is True
    assert kwargs["threshold"] == 0.7


def test_label_descriptions_passed_when_provided(mock_gliner2):
    mock_gliner2.extract_entities.return_value = _entities_payload({"email": []})
    descriptions = {"email": "an email address such as john@example.com"}
    recognizer = GLiNER2Recognizer(
        entity_mapping={"email": "EMAIL_ADDRESS"},
        label_descriptions=descriptions,
    )

    recognizer.analyze("text", ["EMAIL_ADDRESS"])

    # The label-descriptions mapping is passed as the schema (2nd positional arg)
    args, _ = mock_gliner2.extract_entities.call_args
    assert args[1] == descriptions


def test_labels_list_passed_when_no_descriptions(mock_gliner2):
    mock_gliner2.extract_entities.return_value = _entities_payload({"email": []})
    recognizer = GLiNER2Recognizer(entity_mapping={"email": "EMAIL_ADDRESS"})

    recognizer.analyze("text", ["EMAIL_ADDRESS"])

    args, _ = mock_gliner2.extract_entities.call_args
    # Without descriptions, a plain list of labels is passed
    assert isinstance(args[1], list)
    assert "email" in args[1]


def test_missing_confidence_falls_back_to_threshold(mock_gliner2):
    mock_gliner2.extract_entities.return_value = _entities_payload(
        {"email": [{"text": "a@b.com", "start": 0, "end": 7}]}  # no confidence
    )
    recognizer = GLiNER2Recognizer(
        entity_mapping={"email": "EMAIL_ADDRESS"}, threshold=0.66
    )

    results = recognizer.analyze("a@b.com", ["EMAIL_ADDRESS"])

    assert len(results) == 1
    assert results[0].score == pytest.approx(0.66)


def test_entities_without_spans_are_skipped(mock_gliner2):
    mock_gliner2.extract_entities.return_value = _entities_payload(
        {
            "email": [
                {"text": "a@b.com", "confidence": 0.9},  # missing start/end -> skip
                {"text": "c@d.com", "confidence": 0.9, "start": 10, "end": 17},
            ]
        }
    )
    recognizer = GLiNER2Recognizer(entity_mapping={"email": "EMAIL_ADDRESS"})

    results = recognizer.analyze("a@b.com or c@d.com", ["EMAIL_ADDRESS"])

    # Only the entity with a valid span survives
    assert len(results) == 1
    assert (results[0].start, results[0].end) == (10, 17)


def test_load_passes_map_location_and_model_kwargs():
    with patch(f"{GLINER2_MODULE}.GLiNER2") as mock_cls:
        mock_cls.from_pretrained.return_value = MagicMock()

        GLiNER2Recognizer(
            entity_mapping={"email": "EMAIL_ADDRESS"},
            map_location="cpu",
            quantize=True,
        )

        assert mock_cls.from_pretrained.called
        args, kwargs = mock_cls.from_pretrained.call_args
        assert args[0] == "fastino/gliner2-privacy-filter-PII-multi"
        assert kwargs["map_location"] == "cpu"
        assert kwargs["quantize"] is True


def test_import_error_when_gliner2_missing():
    with patch(f"{GLINER2_MODULE}.GLiNER2", None):
        with pytest.raises(ImportError):
            GLiNER2Recognizer(entity_mapping={"email": "EMAIL_ADDRESS"})


def test_chunking_adjusts_offsets(mock_gliner2):
    """Long text is chunked and entity offsets map back to the original text."""
    text = "John Smith lives here. " + ("x " * 120) + "Jane Doe works there."

    def fake_extract(chunk, entity_types, **kwargs):
        people = []
        if "John Smith" in chunk:
            start = chunk.find("John Smith")
            people.append(
                {"text": "John Smith", "confidence": 0.95,
                 "start": start, "end": start + 10}
            )
        if "Jane Doe" in chunk:
            start = chunk.find("Jane Doe")
            people.append(
                {"text": "Jane Doe", "confidence": 0.93,
                 "start": start, "end": start + 8}
            )
        return _entities_payload({"person": people})

    mock_gliner2.extract_entities.side_effect = fake_extract

    recognizer = GLiNER2Recognizer(
        entity_mapping={"person": "PERSON"},
        text_chunker=CharacterBasedTextChunker(chunk_size=250, chunk_overlap=50),
    )

    results = recognizer.analyze(text, ["PERSON"])

    assert mock_gliner2.extract_entities.call_count == 2
    assert len(results) == 2
    results = sorted(results, key=lambda r: r.start)
    assert text[results[0].start:results[0].end] == "John Smith"
    assert text[results[1].start:results[1].end] == "Jane Doe"


# ---------------------------------------------------------------------------
# Explicit coverage for the requested test matrix
# ---------------------------------------------------------------------------


def test_default_model_id_passed_to_from_pretrained():
    """Case 1: the model id is passed correctly to GLiNER2.from_pretrained."""
    with patch(f"{GLINER2_MODULE}.GLiNER2") as mock_cls:
        mock_cls.from_pretrained.return_value = MagicMock()

        recognizer = GLiNER2Recognizer(entity_mapping={"email": "EMAIL_ADDRESS"})

        assert recognizer.model_name == "fastino/gliner2-privacy-filter-PII-multi"
        args, _ = mock_cls.from_pretrained.call_args
        assert args[0] == "fastino/gliner2-privacy-filter-PII-multi"


@pytest.mark.parametrize(
    "label, expected_entity",
    [
        ("email", "EMAIL_ADDRESS"),
        ("phone_number", "PHONE_NUMBER"),
        ("person", "PERSON"),
        ("full_name", "PERSON"),
        ("iban", "IBAN_CODE"),
        ("card_number", "CREDIT_CARD"),
        ("payment_card", "CREDIT_CARD"),
        ("ip_address", "IP_ADDRESS"),
    ],
)
def test_builtin_entity_mapping_values(label, expected_entity):
    """Case 2: built-in PII mapping maps key labels to Presidio entities."""
    assert GLINER2_PII_ENTITY_MAPPING[label] == expected_entity


@pytest.mark.parametrize(
    "label, expected_entity",
    [
        ("email", "EMAIL_ADDRESS"),
        ("phone_number", "PHONE_NUMBER"),
        ("person", "PERSON"),
        ("full_name", "PERSON"),
        ("iban", "IBAN_CODE"),
        ("card_number", "CREDIT_CARD"),
        ("payment_card", "CREDIT_CARD"),
        ("ip_address", "IP_ADDRESS"),
    ],
)
def test_entity_mapping_applied_during_analyze(mock_gliner2, label, expected_entity):
    """Case 2: a model label is converted to its mapped Presidio entity."""
    mock_gliner2.extract_entities.return_value = _entities_payload(
        {label: [{"text": "x", "confidence": 0.9, "start": 0, "end": 1}]}
    )
    recognizer = GLiNER2Recognizer()  # built-in mapping

    results = recognizer.analyze("x", [expected_entity])

    assert len(results) == 1
    assert results[0].entity_type == expected_entity


def test_confidence_becomes_recognizer_result_score(mock_gliner2):
    """Case 4: the model confidence becomes RecognizerResult.score."""
    mock_gliner2.extract_entities.return_value = _entities_payload(
        {"email": [{"text": "a@b.com", "confidence": 0.873, "start": 0, "end": 7}]}
    )
    recognizer = GLiNER2Recognizer(entity_mapping={"email": "EMAIL_ADDRESS"})

    results = recognizer.analyze("a@b.com", ["EMAIL_ADDRESS"])

    assert len(results) == 1
    assert results[0].score == pytest.approx(0.873)
    # And the original score is surfaced in the explanation
    assert results[0].analysis_explanation.original_score == pytest.approx(0.873)


def test_import_error_message_is_actionable():
    """Case 7: a missing dependency raises a useful ImportError."""
    with patch(f"{GLINER2_MODULE}.GLiNER2", None):
        with pytest.raises(ImportError) as exc_info:
            GLiNER2Recognizer(entity_mapping={"email": "EMAIL_ADDRESS"})

    message = str(exc_info.value)
    assert "gliner2" in message
    # The message tells the user how to fix it
    assert "presidio-analyzer[gliner2]" in message


def test_yaml_registry_instantiates_recognizer_when_mocked():
    """Case 8: the YAML registry example builds the recognizer (dependency mocked)."""
    from presidio_analyzer.recognizer_registry import RecognizerRegistryProvider

    registry_configuration = {
        "supported_languages": ["en"],
        "recognizers": [
            {
                "name": "GLiNER2Recognizer",
                "type": "predefined",
                "supported_language": "en",
                "model_name": "fastino/gliner2-privacy-filter-PII-multi",
                "threshold": 0.4,
                "map_location": "cpu",
                "entity_mapping": {
                    "email": "EMAIL_ADDRESS",
                    "phone_number": "PHONE_NUMBER",
                },
            }
        ],
    }

    with patch(f"{GLINER2_MODULE}.GLiNER2") as mock_cls:
        mock_cls.from_pretrained.return_value = MagicMock()

        provider = RecognizerRegistryProvider(
            registry_configuration=registry_configuration
        )
        registry = provider.create_recognizer_registry()

    built = [
        rec
        for rec in registry.recognizers
        if rec.__class__.__name__ == "GLiNER2Recognizer"
    ]
    assert len(built) == 1
    recognizer = built[0]
    assert recognizer.model_name == "fastino/gliner2-privacy-filter-PII-multi"
    assert recognizer.threshold == 0.4
    assert recognizer.model_to_presidio_entity_mapping == {
        "email": "EMAIL_ADDRESS",
        "phone_number": "PHONE_NUMBER",
    }


def _gliner2_doc_path():
    """Locate docs/samples/python/gliner2.md from the repo root, if present."""
    # tests/ -> presidio-analyzer/ -> repo root
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "docs" / "samples" / "python" / "gliner2.md"


def test_docs_python_blocks_are_syntactically_valid():
    """Case 9: every ```python block in the docs sample compiles.

    The repo has no doctest/notebook-execution harness for markdown, so this is
    a lightweight, CI-safe guard: it compiles (does not execute) each Python
    fenced block to catch syntax errors in the documentation examples.
    """
    doc_path = _gliner2_doc_path()
    if not doc_path.exists():
        pytest.skip(f"docs sample not found at {doc_path}")

    text = doc_path.read_text(encoding="utf-8")
    # Match fenced python blocks, allowing the fence to be indented (e.g. when
    # the block sits inside a markdown list item).
    blocks = re.findall(r"```python\n(.*?)```", text, flags=re.DOTALL)
    assert blocks, "expected at least one ```python block in the docs sample"

    for i, block in enumerate(blocks):
        # Indented code blocks (inside list items) carry leading whitespace that
        # is valid markdown but not valid standalone Python; strip it first.
        snippet = textwrap.dedent(block)
        try:
            ast.parse(snippet)
        except SyntaxError as exc:  # pragma: no cover - failure path
            pytest.fail(f"python block #{i} in {doc_path.name} is invalid: {exc}")


# ---------------------------------------------------------------------------
# Regression tests for review findings
# ---------------------------------------------------------------------------


def test_adhoc_requested_label_is_added_to_schema(mock_gliner2):
    """__create_input_labels: a requested entity absent from the mapping is queried."""
    mock_gliner2.extract_entities.return_value = _entities_payload({"email": []})
    recognizer = GLiNER2Recognizer(entity_mapping={"email": "EMAIL_ADDRESS"})

    recognizer.analyze("text", ["CUSTOM_LABEL"])

    args, _ = mock_gliner2.extract_entities.call_args
    schema = args[1]  # list of labels (no descriptions configured)
    assert "email" in schema  # configured label
    assert "CUSTOM_LABEL" in schema  # ad-hoc requested label was appended


def test_add_requested_entities_false_keeps_recognizer_scoped(mock_gliner2):
    """With add_requested_entities=False, requested entities outside the mapping
    are not queried (so the recognizer stays scoped in a mixed registry)."""
    mock_gliner2.extract_entities.return_value = _entities_payload({"email": []})
    recognizer = GLiNER2Recognizer(
        entity_mapping={"email": "EMAIL_ADDRESS"},
        add_requested_entities=False,
    )

    recognizer.analyze("text", ["EMAIL_ADDRESS", "IP_ADDRESS", "CREDIT_CARD"])

    args, _ = mock_gliner2.extract_entities.call_args
    schema = args[1]
    assert "email" in schema  # configured label still queried
    assert "IP_ADDRESS" not in schema  # ad-hoc labels NOT appended
    assert "CREDIT_CARD" not in schema


def test_label_descriptions_overlay_covers_full_label_set(mock_gliner2):
    """label_descriptions for a subset must not drop the other configured labels."""
    mock_gliner2.extract_entities.return_value = _entities_payload({"email": []})
    recognizer = GLiNER2Recognizer(
        entity_mapping={"email": "EMAIL_ADDRESS", "phone_number": "PHONE_NUMBER"},
        label_descriptions={"email": "an email address"},  # description for email only
    )

    recognizer.analyze("text", ["EMAIL_ADDRESS", "PHONE_NUMBER"])

    args, _ = mock_gliner2.extract_entities.call_args
    schema = args[1]
    assert isinstance(schema, dict)
    # email keeps its description; phone_number is still queried as a bare label
    assert schema["email"] == "an email address"
    assert schema["phone_number"] == "phone_number"


def test_label_descriptions_overlay_keeps_adhoc_labels(mock_gliner2):
    """Ad-hoc requested labels survive even when label_descriptions is set."""
    mock_gliner2.extract_entities.return_value = _entities_payload({"email": []})
    recognizer = GLiNER2Recognizer(
        entity_mapping={"email": "EMAIL_ADDRESS"},
        label_descriptions={"email": "an email address"},
    )

    recognizer.analyze("text", ["CUSTOM_LABEL"])

    args, _ = mock_gliner2.extract_entities.call_args
    schema = args[1]
    assert isinstance(schema, dict)
    assert schema["email"] == "an email address"
    assert schema["CUSTOM_LABEL"] == "CUSTOM_LABEL"  # ad-hoc label, no description


def test_non_dict_model_output_warns_and_returns_empty(mock_gliner2, caplog):
    """A non-dict extract_entities result is surfaced via a warning, not silenced."""
    import logging

    mock_gliner2.extract_entities.return_value = ["unexpected", "list", "output"]
    recognizer = GLiNER2Recognizer(entity_mapping={"email": "EMAIL_ADDRESS"})

    with caplog.at_level(logging.WARNING, logger="presidio-analyzer"):
        results = recognizer.analyze("text", ["EMAIL_ADDRESS"])

    assert results == []
    assert any(
        "unexpected output" in r.getMessage() and r.levelno == logging.WARNING
        for r in caplog.records
    )


def test_entity_without_span_logs_warning(mock_gliner2, caplog):
    """Dropping a detected entity with no span is logged at WARNING (not debug)."""
    import logging

    mock_gliner2.extract_entities.return_value = _entities_payload(
        {"email": [{"text": "a@b.com", "confidence": 0.9}]}  # missing start/end
    )
    recognizer = GLiNER2Recognizer(entity_mapping={"email": "EMAIL_ADDRESS"})

    with caplog.at_level(logging.WARNING, logger="presidio-analyzer"):
        results = recognizer.analyze("a@b.com", ["EMAIL_ADDRESS"])

    assert results == []
    assert any(
        "without a start/end span" in r.getMessage() and r.levelno == logging.WARNING
        for r in caplog.records
    )


def test_empty_entity_mapping_with_supported_entities_raises(mock_gliner2):
    """Mutual-exclusion is identity-based: empty mapping + entities still conflicts."""
    with pytest.raises(ValueError):
        GLiNER2Recognizer(entity_mapping={}, supported_entities=["EMAIL_ADDRESS"])


def test_gliner2_pii_entity_mapping_is_read_only():
    """The exported default mapping is immutable (must be copied before mutation)."""
    with pytest.raises(TypeError):
        GLINER2_PII_ENTITY_MAPPING["email"] = "SOMETHING_ELSE"  # type: ignore[index]


def test_chunk_overlap_entity_is_deduplicated(mock_gliner2):
    """An entity in the overlap region of two chunks is emitted once, not twice."""
    # "Dr. Smith" sits in the 50-char overlap so both chunks report it.
    text = ("x " * 95) + "Dr. Smith" + (" x" * 100)

    call_count = 0

    def fake_extract(chunk, entity_types, **kwargs):
        nonlocal call_count
        call_count += 1
        people = []
        if "Dr. Smith" in chunk:
            start = chunk.find("Dr. Smith")
            score = 0.95 if call_count == 1 else 0.90
            people.append(
                {"text": "Dr. Smith", "confidence": score,
                 "start": start, "end": start + 9}
            )
        return _entities_payload({"person": people})

    mock_gliner2.extract_entities.side_effect = fake_extract

    recognizer = GLiNER2Recognizer(
        entity_mapping={"person": "PERSON"},
        text_chunker=CharacterBasedTextChunker(chunk_size=250, chunk_overlap=50),
    )

    results = recognizer.analyze(text, ["PERSON"])

    assert mock_gliner2.extract_entities.call_count >= 2  # multiple chunks processed
    assert len(results) == 1  # deduplicated to a single result
    assert text[results[0].start:results[0].end] == "Dr. Smith"
    assert results[0].score == 0.95  # highest-scoring duplicate kept


# ---------------------------------------------------------------------------
# Optional integration test against the real model (downloads weights).
# Opt-in only: PRESIDIO_RUN_FASTINO_GLINER2_INTEGRATION=1
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.getenv("PRESIDIO_RUN_FASTINO_GLINER2_INTEGRATION") != "1",
    reason=(
        "Set PRESIDIO_RUN_FASTINO_GLINER2_INTEGRATION=1 to run the real-model "
        "integration test (downloads the model and requires network access)."
    ),
)
def test_integration_real_model_detects_email_and_phone():
    """Real-model end-to-end check; skipped in CI by default."""
    pytest.importorskip("gliner2", reason="gliner2 package is not installed")

    text = "Email john.smith@acme.com or call +1 415 555 0199."
    recognizer = GLiNER2Recognizer(map_location="cpu")

    results = recognizer.analyze(text, ["EMAIL_ADDRESS", "PHONE_NUMBER"])
    detected = {r.entity_type: text[r.start:r.end] for r in results}

    assert detected.get("EMAIL_ADDRESS") == "john.smith@acme.com"
    assert detected.get("PHONE_NUMBER") == "+1 415 555 0199"
