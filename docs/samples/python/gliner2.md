# Using GLiNER2 within Presidio

[GLiNER2](https://pypi.org/project/gliner2/) is the second-generation GLiNER
architecture for schema-based information extraction. Like GLiNER, it is a
compact, encoder-based model that performs zero-shot Named Entity Recognition
(NER) by taking both the text and the entity types as input. It ships as a
separate library (`gliner2`) with a different inference API
(`GLiNER2.extract_entities`).

This page describes how to use the multilingual PII model
[`fastino/gliner2-privacy-filter-PII-multi`](https://huggingface.co/fastino/gliner2-privacy-filter-PII-multi)
*(Apache 2.0)* with Presidio in production. The `gliner2` library is also
Apache 2.0.

## Two usage modes

| Mode | Library | Works with `fastino/gliner2-privacy-filter-PII-multi`? |
| --- | --- | --- |
| `GLiNERRecognizer` | `gliner` | **No.** `gliner.GLiNER.from_pretrained` raises `FileNotFoundError: No config file found` — the `gliner` library cannot load a GLiNER2 ("extractor") checkpoint. |
| `GLiNER2Recognizer` | `gliner2` | **Yes.** This is the supported path. |

Use **`GLiNER2Recognizer`**. The first-generation `GLiNERRecognizer`
(see [Using GLiNER within Presidio](gliner.md)) remains the right choice for
`gliner`-library models such as `urchade/gliner_multi_pii-v1`, but it is not
compatible with this model.

## Installation

```bash
pip install 'presidio-analyzer[gliner2]'
# For the anonymization example below:
pip install presidio-anonymizer
```

!!! warning "Opt-in and calibration"
    `GLiNER2Recognizer` is **opt-in**: it is not part of Presidio's default
    configuration and runs only after you add it to the registry. The default
    `threshold` (0.5) is a starting point — **calibrate it on your own data**.
    A lower threshold increases recall (more detections, more false positives);
    a higher threshold increases precision. For finer control, set
    `label_thresholds` (a per-label minimum confidence) to tighten noisy labels
    such as `person`/`phone_number` without lowering recall on clean ones, and
    `label_selection_strategy` to limit which labels the model is asked for.

## Entity mapping

The model emits 42 PII labels. `GLiNER2Recognizer` maps them to Presidio entity
types via `entity_mapping`. If you omit `entity_mapping` (and
`supported_entities`), the built-in `GLINER2_PII_ENTITY_MAPPING` below is used.
Copy and adjust it to fit your taxonomy:

```python
GLINER2_PII_ENTITY_MAPPING = {
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
```

## Python example: AnalyzerEngine + AnonymizerEngine

```python
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.predefined_recognizers import GLiNER2Recognizer
from presidio_anonymizer import AnonymizerEngine

# 1. Build the analyzer (default spaCy NLP engine for tokenization)
analyzer = AnalyzerEngine()

# 2. Register GLiNER2 with a custom entity_mapping
#    (omit entity_mapping entirely to use the built-in 42-label PII mapping)
analyzer.registry.add_recognizer(
    GLiNER2Recognizer(
        model_name="fastino/gliner2-privacy-filter-PII-multi",
        entity_mapping={
            "email": "EMAIL_ADDRESS",
            "phone_number": "PHONE_NUMBER",
            "person": "PERSON",
            "full_name": "PERSON",
            "address": "LOCATION",
        },
        threshold=0.5,
        map_location="cpu",  # use "cuda" on a GPU host
    )
)

# 3. Remove spaCy NER so its PERSON/LOCATION spans don't duplicate GLiNER2's
analyzer.registry.remove_recognizer("SpacyRecognizer")

text = "Email john.smith@acme.com or call +1 415 555 0199."
results = analyzer.analyze(text=text, language="en")
print(results)

# 4. Anonymize the detected spans
anonymized = AnonymizerEngine().anonymize(text=text, analyzer_results=results)
print(anonymized.text)
# -> "Email <EMAIL_ADDRESS> or call <PHONE_NUMBER>."
```

## No-code: YAML recognizer registry

`GLiNER2Recognizer` is a predefined recognizer, so it can be configured from a
registry YAML file. Create `recognizers-config.yml`:

```yaml
supported_languages:
  - en

recognizers:
  # Disable spaCy NER to avoid duplicate PERSON/LOCATION spans
  - name: "SpacyRecognizer"
    type: "predefined"
    class_name: "SpacyRecognizer"
    supported_languages: ["en"]
    enabled: false

  - name: "GLiNER2Recognizer"
    type: "predefined"
    class_name: "GLiNER2Recognizer"
    supported_languages: ["en"]
    model_name: "fastino/gliner2-privacy-filter-PII-multi"
    threshold: 0.5
    map_location: "cpu"
    # Only query the model for the labels mapped to the requested entities
    # (default). Use "all_configured" to query every label.
    label_selection_strategy: "requested_presidio_entities"
    # Per-label minimum confidence: raise precision on noisy labels without
    # lowering recall on clean ones. Labels without an entry use `threshold`.
    label_thresholds:
      person: 0.85
      phone_number: 0.7
    entity_mapping:
      email: "EMAIL_ADDRESS"
      phone_number: "PHONE_NUMBER"
      person: "PERSON"
      address: "LOCATION"
```

Load it and pass the registry to the analyzer:

```python
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.recognizer_registry import RecognizerRegistryProvider

provider = RecognizerRegistryProvider(conf_file="./recognizers-config.yml")
registry = provider.create_recognizer_registry()

analyzer = AnalyzerEngine(registry=registry, supported_languages=["en"])
print(analyzer.analyze(text="Email john.smith@acme.com", language="en"))
```

!!! note
    `entity_mapping` and `supported_entities` are mutually exclusive — provide at
    most one. Omit both to use the built-in 42-label PII mapping.

## Recommended: hybrid with Presidio's pattern recognizers

GLiNER2 is strongest on **free-form** entities — names, addresses/locations, and
identifiers that have no fixed syntax. For **rigid-format** PII (email, IP, IBAN,
credit card, and other checksum/regex-validated values), Presidio's built-in
deterministic recognizers are typically far more precise, since they validate
structure (Luhn, IBAN mod-97, well-formed IP/email) instead of inferring from
context.

The recommended setup therefore **routes each entity to the detector that is best
at it**: keep Presidio's pattern/checksum recognizers for the rigid-format PII
they validate, and scope `GLiNER2Recognizer` to the entities where it adds value
(person, location, and free-form IDs Presidio has no recognizer for). Give each
entity type a **single owner** — having two detectors emit the same type tends to
compound false positives rather than improve results.

```python
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.predefined_recognizers import GLiNER2Recognizer

# AnalyzerEngine ships the regex/checksum recognizers (Email, Phone, CreditCard,
# Ip, Iban, US_SSN, ...). They keep ownership of rigid-format PII.
analyzer = AnalyzerEngine()

# Scope GLiNER2 to free-form entities only; the deterministic recognizers own
# email / phone / ip / iban / credit card.
gliner2 = GLiNER2Recognizer(
    map_location="cpu",
    threshold=0.5,
    # Stay strictly within entity_mapping. Without this, GLiNER2 would also try
    # to detect the entity types the analyzer requests on behalf of the other
    # recognizers (email/phone/ip/iban/credit card) as ad-hoc labels, polluting
    # those high-precision results.
    add_requested_entities=False,
    entity_mapping={
        # names + locations/addresses (GLiNER2 >> spaCy NER, especially streets)
        "person": "PERSON", "full_name": "PERSON",
        "first_name": "PERSON", "last_name": "PERSON", "middle_name": "PERSON",
        "address": "LOCATION", "street_address": "LOCATION", "city": "LOCATION",
        "state_or_region": "LOCATION", "postal_code": "LOCATION",
        "country": "LOCATION",
        # free-form PII Presidio has no built-in recognizer for
        "username": "USERNAME", "password": "PASSWORD", "secret": "SECRET",
        "api_key": "API_KEY", "access_token": "ACCESS_TOKEN",
        "government_id": "GOVERNMENT_ID", "national_id_number": "NATIONAL_ID",
        "passport_number": "PASSPORT",
        "drivers_license_number": "DRIVER_LICENSE",
    },
)
analyzer.registry.add_recognizer(gliner2)

# GLiNER2 replaces spaCy as the NER source for person/location, so drop the
# spaCy recognizer to avoid lower-quality duplicate spans.
analyzer.registry.remove_recognizer("SpacyRecognizer")
```

!!! tip "Phone numbers: pick one owner"
    `PhoneRecognizer` (the `phonenumbers` library) is precise but misses formats
    it cannot parse; GLiNER2 has high recall but lower precision. **Unioning both
    measured worse than either alone** (the false positives compound). Keep phone
    with `PhoneRecognizer` by default; if you need the model's recall instead, add
    `"phone_number": "PHONE_NUMBER"` to the mapping *and remove* `PhoneRecognizer`,
    rather than running both.

!!! tip "Checksum recognizers reject malformed values"
    `CreditCardRecognizer` (Luhn), `IbanRecognizer` (mod-97), and similar
    recognizers intentionally reject values that fail validation. This is what
    makes them high-precision, but if your data contains non-standard or
    masked numbers you may also want GLiNER2 to detect them (add `card_number`
    /`payment_card` to the mapping and union the results), trading some precision
    for recall.

### GLiNER2 vs spaCy as the NER component (illustrative)

The hybrid above uses GLiNER2 for the NER-style entities (person, location,
free-form IDs). The alternative is to keep spaCy NER for those and pair it with
the same regex/checksum recognizers. The table below compares the two hybrids on
2,000 German examples from
[`ai4privacy/pii-masking-200k`](https://huggingface.co/datasets/ai4privacy/pii-masking-200k),
matched on character-overlap. Both hybrids use the **same** regex/checksum
recognizers, so the structured types (email, phone, credit card, IBAN, IP) score
identically — only the NER component differs:

| Entity | spaCy + regex (P / R / F1) | GLiNER2 + regex (P / R / F1) |
| --- | --- | --- |
| PERSON | 0.60 / 0.70 / 0.65 | 0.57 / 0.90 / **0.70** |
| LOCATION | 0.52 / 0.66 / 0.58 | 0.65 / 0.91 / **0.76** |
| USERNAME | 0.00 / 0.00 / 0.00 | 0.45 / 0.76 / **0.56** |
| EMAIL / PHONE / CREDIT_CARD / IBAN / IP | *identical (shared regex recognizers)* | *identical* |
| **Micro (all 8 types)** | 0.65 / 0.68 / 0.67 | 0.66 / 0.86 / **0.74** |

GLiNER2 wins the NER component — most strongly on **LOCATION** (it detects full
street addresses, which spaCy's `LOC` misses) and on free-form types like
**USERNAME** that spaCy has no label for, plus higher **PERSON** recall. spaCy's
person *precision* is marginally higher, and spaCy is much lighter/faster on CPU,
so it remains a reasonable choice when those matter more.

!!! note
    These numbers are illustrative: ai4privacy is synthetic (cleaner than real
    text), the per-type sample is modest, and results depend on the language,
    threshold, and entity mapping. Benchmark on your own data before relying on
    them — see [PII detection evaluation](https://microsoft.github.io/presidio/evaluation/).

## Multilingual notes

- The model's labels are multilingual: the model card lists support for
  **English (en), French (fr), Spanish (es), German (de), Italian (it),
  Portuguese (pt), and Dutch (nl)**.
- GLiNER2 runs independently of Presidio's NLP engine, but **Presidio Analyzer
  still needs a compatible NLP engine and language setup**: the analyzer
  tokenizes text and resolves the requested language through its configured NLP
  engine (spaCy by default). To analyze non-English text, configure the
  analyzer's NLP engine for that language (see
  [Customizing the NLP engine](../../analyzer/customizing_nlp_models.md)) and
  pass the matching `language=` code to `analyze()`.
- **Remove (or disable) `SpacyRecognizer`** in your configuration when you want
  GLiNER2 to be the source of NER-style entities (PERSON, LOCATION, …).
  Otherwise spaCy and GLiNER2 may both emit overlapping spans for the same text.

## Deployment notes

- **Optional dependency only.** `gliner2` is installed through the
  `presidio-analyzer[gliner2]` extra and is never part of Presidio's base
  dependencies.
- **No default enablement.** `GLiNER2Recognizer` is added to the registry only
  when you do so explicitly (in code or via registry YAML). It is never loaded
  by `load_predefined_recognizers()`.
- **Pin the model revision for reproducibility.** As of `gliner2` 1.x,
  `GLiNER2.from_pretrained` resolves the latest revision on the Hugging Face Hub
  and does not expose a `revision` argument. To pin an exact revision,
  pre-download it and point `model_name` at the local directory
  (`from_pretrained` loads local paths directly):

  ```python
  from huggingface_hub import snapshot_download
  from presidio_analyzer.predefined_recognizers import GLiNER2Recognizer

  model_dir = snapshot_download(
      "fastino/gliner2-privacy-filter-PII-multi",
      revision="<commit-sha>",        # pin an immutable commit
      local_dir="/models/gliner2-pii",
  )
  recognizer = GLiNER2Recognizer(model_name=model_dir, map_location="cpu")
  ```

- **Cache model files for offline deployments.** Bake the model into the
  container image or mount it from a volume (e.g. the `snapshot_download`
  directory above, or the `HF_HOME` / `HF_HUB_CACHE` directory). Setting
  `HF_HUB_OFFLINE=1` then prevents any network calls at runtime.
- **Tune thresholds per entity.** A single `threshold` controls global recall.
  For per-entity tuning, either post-filter `analyzer.analyze(...)` results by
  `entity_type` and `score`, or register multiple `GLiNER2Recognizer` instances
  each scoped to a subset of entities with its own threshold. Free-text labels
  such as **`person`/`full_name`** tend to over-trigger — raise their threshold
  (or post-filter) before relying on them.

## Limitations

- Automated PII detection is **probabilistic**: it can miss real PII (false
  negatives) and flag non-PII (false positives), and reported spans may be
  imprecise.
- **Calibrate on domain-specific data.** Validate precision/recall on a
  representative, labeled sample from your own data before relying on the output.
- **Do not treat this model as a sole compliance control.** Combine it with
  deterministic recognizers (regex/checksum-based), human review, and
  organizational controls; do not rely on it alone for regulatory obligations.

## Label descriptions (optional)

GLiNER2 supports natural-language descriptions for labels, which can help the
model disambiguate similar labels. Pass them via `label_descriptions` (keys must
match the model labels, i.e. the keys of `entity_mapping`):

```python
gliner2_recognizer = GLiNER2Recognizer(
    entity_mapping={"email": "EMAIL_ADDRESS", "phone_number": "PHONE_NUMBER"},
    label_descriptions={
        "email": "an email address such as john@example.com",
        "phone_number": "a telephone or mobile number",
    },
    map_location="cpu",
)
```

## GLiNERRecognizer vs. GLiNER2Recognizer

| | `GLiNERRecognizer` | `GLiNER2Recognizer` |
| --- | --- | --- |
| Library | `gliner` | `gliner2` |
| Extra | `presidio-analyzer[gliner]` | `presidio-analyzer[gliner2]` |
| Load | `GLiNER.from_pretrained` | `GLiNER2.from_pretrained` |
| Inference | `predict_entities` | `extract_entities` |
| Example model | `urchade/gliner_multi_pii-v1` | `fastino/gliner2-privacy-filter-PII-multi` |

See [Using GLiNER within Presidio](gliner.md) for the first-generation model.
