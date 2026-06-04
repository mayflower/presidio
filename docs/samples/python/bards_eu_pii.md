# Using the Bards EU-PII model within Presidio

## What is the model

[`bardsai/eu-pii-anonimization-multilang`](https://huggingface.co/bardsai/eu-pii-anonimization-multilang)
*(Apache 2.0)* is a multilingual PII detector that covers the 24 official EU
languages. It is a standard token-classification model — an
`XLMRobertaForTokenClassification` checkpoint fine-tuned on top of
[`FacebookAI/xlm-roberta-base`](https://huggingface.co/FacebookAI/xlm-roberta-base)
— that emits BIO-tagged spans for 35 PII classes, organised into eight families
(personal identity, contact & location, official documents, financial, technical
identifiers, organization data, and the GDPR Art. 9 health/biometric and
special-category data).

Because it is an ordinary HuggingFace `token-classification` model, Presidio
needs no new inference code to run it: the bundled
[`HuggingFaceNerRecognizer`](https://github.com/microsoft/presidio/blob/main/presidio-analyzer/presidio_analyzer/predefined_recognizers/ner/huggingface_ner_recognizer.py)
already wraps the `transformers` pipeline (BIO-tag aggregation, chunking,
thresholding, device selection). `BardsEuPiiRecognizer` is a thin subclass that
only bakes in the model id and a curated mapping from the model's labels to
Presidio entities.

## Using it with Presidio

Presidio ships a built-in recognizer: `BardsEuPiiRecognizer`. It is **opt-in** —
it is not part of the default registry and is never loaded (or downloaded) unless
you add it explicitly.

### Installation

The model runs through the `transformers` extra (which pulls in
`transformers`, `accelerate`/`torch` and `huggingface_hub`):

```bash
pip install 'presidio-analyzer[transformers]'
```

### Example

```python
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_analyzer.predefined_recognizers import BardsEuPiiRecognizer
from presidio_anonymizer import AnonymizerEngine

# A small spaCy model is enough — we rely on the model for NER, not spaCy.
nlp_engine = NlpEngineProvider(
    nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
    }
).create_engine()

analyzer = AnalyzerEngine(nlp_engine=nlp_engine)

# Create and register the recognizer (model id + label mapping are baked in).
recognizer = BardsEuPiiRecognizer(threshold=0.4)
analyzer.registry.add_recognizer(recognizer)

# Optionally remove spaCy's NER recognizer to avoid duplicate person/location hits.
analyzer.registry.remove_recognizer("SpacyRecognizer")

text = "Contact John Smith at john.smith@example.com or +49 30 12345678."
results = analyzer.analyze(text=text, language="en")
print(results)

# Anonymize the detected entities.
anonymizer = AnonymizerEngine()
print(anonymizer.anonymize(text=text, analyzer_results=results).text)
```

## Entity mapping

The recognizer maps the model's 35 labels to Presidio entities via the
`EU_PII_ENTITY_MAPPING` constant. Labels with a faithful Presidio standard
entity use it; the GDPR Art. 9 nationality/religion/politics trio is collapsed
onto Presidio's built-in `NRP` entity; everything else passes through as its own
descriptive entity type (so you can attach a dedicated anonymizer operator per
category).

| Model label | Presidio entity |
|---|---|
| `PERSON_NAME`, `PERSON_ALIAS` | `PERSON` |
| `LOCATION`, `GEO_LOCATION`, `POSTAL_ADDRESS` | `LOCATION` |
| `ORGANIZATION_NAME` | `ORGANIZATION` |
| `EMAIL_ADDRESS` | `EMAIL_ADDRESS` |
| `PHONE_NUMBER` | `PHONE_NUMBER` |
| `IP_ADDRESS` | `IP_ADDRESS` |
| `PAYMENT_CARD` | `CREDIT_CARD` |
| `DATE_OF_BIRTH` | `DATE_TIME` |
| `IDENTIFYING_LINK` | `URL` |
| `ETHNIC_ORIGIN`, `RELIGION_OR_BELIEF`, `POLITICAL_OPINION` | `NRP` |
| `HEALTH_DATA`, `BIOMETRIC_DATA`, `CRIMINAL_OFFENCE_DATA`, `SEXUAL_ORIENTATION`, `TRADE_UNION_MEMBERSHIP` | *(same, descriptive)* |
| `ACCOUNT_IDENTIFIER`, `AUTH_SECRET`, `BANK_ACCOUNT_IDENTIFIER`, `CONTACT_HANDLE`, `DEVICE_IDENTIFIER`, `DOCUMENT_IDENTIFIER`, `DOCUMENT_REFERENCE`, `FINANCIAL_AMOUNT`, `ORGANIZATION_IDENTIFIER`, `PAYMENT_CARD_SECURITY`, `PERSON_ATTRIBUTE`, `PERSON_IDENTIFIER`, `PERSON_ROLE_OR_TITLE`, `PROPER_NAME`, `VEHICLE_IDENTIFIER` | *(same, descriptive)* |

To change how a label is mapped (for example, to send `PROPER_NAME` to `PERSON`,
or to keep the special categories descriptive instead of collapsing to `NRP`),
pass your own `label_mapping`:

```python
from presidio_analyzer.predefined_recognizers.ner.bards_eu_pii_recognizer import (
    EU_PII_ENTITY_MAPPING,
)

custom_mapping = dict(EU_PII_ENTITY_MAPPING)
custom_mapping["PROPER_NAME"] = "PERSON"
custom_mapping["ETHNIC_ORIGIN"] = "ETHNIC_ORIGIN"  # keep descriptive

recognizer = BardsEuPiiRecognizer(label_mapping=custom_mapping)
```

## Multilingual usage

The checkpoint is a single multilingual model, but a recognizer instance is
registered for one language. To cover several EU languages, register one
instance per language (each pointing at the same model):

```python
for lang in ("en", "de", "fr"):
    analyzer.registry.add_recognizer(BardsEuPiiRecognizer(supported_language=lang))
```

Make sure your `AnalyzerEngine` / NLP engine is configured for those languages.

## Hybrid setup (recommended)

The model is strong on free-form, contextual PII (names, locations, roles, the
GDPR special categories) but, like any NER model, it is noisier on rigid-format
identifiers than a dedicated validator. Presidio already ships precise
regex/checksum recognizers for those (`EmailRecognizer`, `PhoneRecognizer`,
`CreditCardRecognizer` with a Luhn check, `IbanRecognizer`, `IpRecognizer`,
`UrlRecognizer`). The best results come from a **hybrid**: let the deterministic
recognizers own the structured entities and scope the model to what it does best.

Scope the model by mapping the labels that overlap a deterministic recognizer to
a sentinel you never request — those entities then come only from the precise
recognizers:

```python
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_analyzer.predefined_recognizers import BardsEuPiiRecognizer
from presidio_analyzer.predefined_recognizers.ner.bards_eu_pii_recognizer import (
    EU_PII_ENTITY_MAPPING,
)

# Labels Presidio's deterministic recognizers already cover precisely.
DETERMINISTIC = {
    "EMAIL_ADDRESS", "IP_ADDRESS", "PAYMENT_CARD", "PHONE_NUMBER",
    "IDENTIFYING_LINK",
}
hybrid_mapping = {
    label: ("_MODEL_IGNORE_" if label in DETERMINISTIC else entity)
    for label, entity in EU_PII_ENTITY_MAPPING.items()
}

nlp_engine = NlpEngineProvider(nlp_configuration={
    "nlp_engine_name": "spacy",
    "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
}).create_engine()

# A default AnalyzerEngine already registers the deterministic recognizers.
analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
analyzer.registry.add_recognizer(BardsEuPiiRecognizer(label_mapping=hybrid_mapping))
# Let names/locations come from the model rather than spaCy's weaker NER.
analyzer.registry.remove_recognizer("SpacyRecognizer")

text = "Contact John Smith at john.smith@example.com or +49 30 12345678."
# `_MODEL_IGNORE_` is in the model's supported entities but never requested, so
# the model's email/phone/ip predictions are dropped — those come from regex.
results = analyzer.analyze(text=text, language="en")
```

### How the hybrid compares

On a German benchmark (ai4privacy `pii-masking-200k`, 2,000 examples), keeping the
**same regex/checksum layer** and swapping only the NER backend (span-overlap F1;
the structured buckets are identical by construction and omitted):

| NER bucket | spaCy `de_core_news_lg` | GLiNER2 | Bards EU-PII |
|---|---|---|---|
| PERSON   | 0.645 | 0.663 | **0.861** |
| LOCATION | 0.580 | 0.703 | **0.799** |
| USERNAME | 0.000 | 0.483 | **0.585** |
| **micro (NER)** | 0.590 | 0.661 | **0.815** |
| **micro (all 8 buckets)** | 0.665 | 0.705 | **0.820** |

Bards wins every NER bucket, and—unlike GLiNER2, which is recall-heavy but
imprecise—it does so with high precision (PERSON P=0.80, LOCATION P=0.85 vs
GLiNER2's 0.51 / 0.57), which matters when you anonymize by redaction.

## No-code (YAML) configuration

You can enable the recognizer from a registry configuration without writing
Python. Omitted fields fall back to the recognizer's defaults (model id, label
mapping, threshold):

```yaml
recognizers:
  - name: BardsEuPiiRecognizer
    type: predefined
    supported_languages:
      - en
      - de
    # optional overrides:
    # threshold: 0.4
    # device: cpu
```

```python
from presidio_analyzer.recognizer_registry import RecognizerRegistryProvider

registry = RecognizerRegistryProvider(
    registry_configuration={
        "supported_languages": ["en", "de"],
        "recognizers": [
            {
                "name": "BardsEuPiiRecognizer",
                "type": "predefined",
                "supported_languages": ["en", "de"],
            }
        ],
    }
).create_recognizer_registry()
```

## Tuning and limitations

- **Threshold.** `threshold` (default `0.4`) trades recall for precision: lower it
  to favour recall (compliance), raise it to favour precision (data utility).
- **Aggregation strategy.** The default is `aggregation_strategy="first"`. This
  model's fast tokenizer does not expose real word ids, so the HuggingFace
  pipeline uses a fallback heuristic; under `"simple"` that heuristic fragments
  subword entities (an e-mail address is split into many pieces), while `"first"`
  keeps each entity as one contiguous span.
- **Production / CPU.** The model card ships quantized ONNX weights
  (`onnx/model_quantized.onnx`) for faster CPU inference. `BardsEuPiiRecognizer`
  uses the PyTorch checkpoint; an ONNX path would require a custom recognizer.
- **Not a standalone compliance solution.** Like any model, recall degrades on
  OCR noise, code-switching and unusual formatting; combine it with Presidio's
  deterministic recognizers (regex/checksum for email, IBAN, credit card, IP…)
  for the structured entity types they cover precisely.
- **No dedicated "username" label.** Bare login-usernames are detected, but the
  model files them under `ACCOUNT_IDENTIFIER` (and name-shaped handles under
  `PERSON_NAME`); it never emits `CONTACT_HANDLE` for them (that label is for
  social `@handles`). If you want them in a `USERNAME` entity, add
  `ACCOUNT_IDENTIFIER → USERNAME` to your `label_mapping` — though name-shaped
  handles will still surface as `PERSON`, which the model cannot disambiguate.
