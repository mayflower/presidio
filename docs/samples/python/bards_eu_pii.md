# Using the Bards EU-PII model within Presidio

## What is the model

[`bardsai/eu-pii-anonimization-multilang`](https://huggingface.co/bardsai/eu-pii-anonimization-multilang)
*(Apache 2.0)* is a multilingual PII detector covering the 24 official EU
languages. It is a standard token-classification model — an
`XLMRobertaForTokenClassification` checkpoint fine-tuned on top of
[`FacebookAI/xlm-roberta-base`](https://huggingface.co/FacebookAI/xlm-roberta-base)
— that emits BIO-tagged spans for 35 PII classes across personal identity,
contact and location, official documents, financial data, technical identifiers,
organization data, and the GDPR Art. 9 health/biometric and special-category data.

`BardsEuPiiRecognizer` is the bundled recognizer that wraps it. It is **opt-in**:
it is not part of the default registry and is never loaded (or downloaded) unless
you add it explicitly. Because the model is an ordinary HuggingFace
`token-classification` checkpoint, Presidio needs no new inference code — the
recognizer is a thin subclass of the bundled
[`HuggingFaceNerRecognizer`](https://github.com/microsoft/presidio/blob/main/presidio-analyzer/presidio_analyzer/predefined_recognizers/ner/huggingface_ner_recognizer.py)
that bakes in the model id and a curated mapping from the model's labels to
Presidio entities.

## Installation

There are two interchangeable backends — pick the extra for the one you want
(see [CPU-optimized inference (ONNX Runtime)](#cpu-optimized-inference-onnx-runtime)
for the trade-off):

```bash
# PyTorch path — BardsEuPiiRecognizer (best on GPU; reference numerics).
# Pulls in transformers, accelerate/torch and huggingface_hub.
pip install 'presidio-analyzer[transformers]'

# CPU-optimized ONNX Runtime path — BardsEuPiiOnnxRecognizer.
# Pulls in Optimum + ONNX Runtime (torch is not required).
pip install 'presidio-analyzer[bards-onnx]'
```

## Quick start

The minimal way to see it work — the model running on its own:

```python
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_analyzer.predefined_recognizers import BardsEuPiiRecognizer

# A small spaCy model is enough: the recognizer does its own NER.
nlp_engine = NlpEngineProvider(nlp_configuration={
    "nlp_engine_name": "spacy",
    "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
}).create_engine()

analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
analyzer.registry.add_recognizer(BardsEuPiiRecognizer())

text = "Contact John Smith at john.smith@example.com."
print(analyzer.analyze(text=text, language="en"))
```

For anything beyond experimentation, use the hybrid setup below.

## Recommended production setup

The safest, highest-quality setup is **hybrid**: let deterministic recognizers
own the structured identifiers and let the model own the contextual, free-form
and sensitive PII.

- **Use `BardsEuPiiRecognizer.hybrid()`** (or `BardsEuPiiOnnxRecognizer.hybrid()`
  for the CPU-optimized ONNX backend — same behavior). The model is strong on
  free-form, contextual PII (names, locations, roles, the GDPR special categories)
  but, like any NER model, it is noisier on rigid-format identifiers than a
  dedicated validator. `hybrid()` drops the structured model labels (e-mail,
  phone, IP, credit card, URL) *before* they are emitted, so those entities come
  only from precise recognizers and the model is scoped to what it does best.
- **Keep the deterministic recognizers enabled.** A default `AnalyzerEngine`
  already registers Presidio's regex/checksum recognizers (`EmailRecognizer`,
  `PhoneRecognizer`, `CreditCardRecognizer` with a Luhn check, `IbanRecognizer`,
  `IpRecognizer`, `UrlRecognizer`). `hybrid()` is designed to complement them —
  do not remove them.
- **Remove weaker duplicate NER recognizers only when appropriate.** The bundled
  spaCy `SpacyRecognizer` also emits `PERSON`/`LOCATION` and can double-detect
  with weaker quality. Remove it **only** when the Bards recognizer covers every
  language you analyze; otherwise keep it as a fallback for the languages you have
  not registered a Bards instance for.

A runnable hybrid example is in [Examples](#examples).

## CPU-optimized inference (ONNX Runtime)

The recognizer ships in two interchangeable backends — same model, same labels,
same mapping / threshold / `hybrid()` behavior; only the inference runtime
differs:

- **`BardsEuPiiRecognizer` — the PyTorch path.** Runs the `safetensors`
  checkpoint through `transformers`. Best when you run on GPU, or want the
  reference PyTorch numerics. Needs the `transformers` extra.
- **`BardsEuPiiOnnxRecognizer` — the CPU-optimized ONNX Runtime path.** Loads the
  model card's upstream **quantized ONNX file**, `onnx/model_quantized.onnx`,
  through Optimum ONNX Runtime and runs it on CPU. Built for CPU-only
  deployments. Needs the `bards-onnx` extra; torch is not required to construct
  it.

`BardsEuPiiOnnxRecognizer` subclasses `BardsEuPiiRecognizer`, so everything else
in this guide — `hybrid()`, mapping profiles, per-entity / per-language
thresholds, `labels_to_ignore`, the entity mapping — applies unchanged. The
production recommendation is the same on either backend: **hybrid mode**,
**deterministic recognizers for the structured identifiers** (e-mail, phone,
IBAN, credit card, IP, URL), and **Bards/ONNX for the free-form contextual PII**.

```python
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_analyzer.predefined_recognizers import BardsEuPiiOnnxRecognizer

nlp_engine = NlpEngineProvider(nlp_configuration={
    "nlp_engine_name": "spacy",
    "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
}).create_engine()

analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
# Identical to the PyTorch hybrid setup, but on the quantized ONNX model.
analyzer.registry.add_recognizer(BardsEuPiiOnnxRecognizer.hybrid())
analyzer.registry.remove_recognizer("SpacyRecognizer")

text = "Contact John Smith at john.smith@example.com or +49 30 12345678."
results = analyzer.analyze(text=text, language="en")
```

### CPU deployment tuning

ONNX Runtime parallelism is controlled by environment variables (or the
`onnx_intra_op_num_threads` / `onnx_inter_op_num_threads` constructor kwargs).
For a CPU container:

- **Start with `WORKERS=1`.** One process loads the ONNX session once; the
  per-language recognizer instances share it, so extra workers mostly duplicate
  memory and threads.
- **Set `TOKENIZERS_PARALLELISM=false`** to silence the Hugging Face tokenizers
  fork/parallelism warning and avoid thread contention under a gunicorn server.
- **Set `ORT_INTRA_OP_THREADS` to the number of physical cores allocated to the
  container, then benchmark.** Leave `ORT_INTER_OP_THREADS=1` unless a benchmark
  says otherwise. Left unset, ONNX Runtime picks its own default.
- **Don't oversubscribe.** Total CPU pressure is roughly
  `WORKERS × ORT_INTRA_OP_THREADS`; keep that at or below the cores the container
  is given. Many workers each spawning many ORT threads contend and run *slower*,
  not faster.

### PyTorch fallback

Prefer PyTorch when you run on GPU or want the reference checkpoint:

- **Python:** use `BardsEuPiiRecognizer` (or `.hybrid()`) in place of the ONNX
  class — the constructor arguments are identical.
- **Container / YAML:** the default Bards hybrid image runs the ONNX backend;
  build the PyTorch variant with the `bards_hybrid_recognizers_pytorch.yaml`
  registry config — see
  [Deploy as a container](#deploy-as-a-container-bards--regex).

### Benchmarking PyTorch vs ONNX on your hardware

CPU throughput depends on the model version, hardware and thread settings, so
**measure both backends** rather than assuming a fixed speed-up. The evaluation
harness ships a throughput/latency benchmark,
[`benchmark_bards_cpu.py`](https://github.com/microsoft/presidio/blob/main/presidio-analyzer/evaluation/bards_eu_pii/benchmark_bards_cpu.py):

```bash
# PyTorch Bards hybrid
pip install 'presidio-analyzer[transformers]'
python presidio-analyzer/evaluation/bards_eu_pii/benchmark_bards_cpu.py \
  --input presidio-analyzer/evaluation/bards_eu_pii/sample_data.jsonl \
  --languages de,en,fr,it --backend pytorch --mode hybrid \
  --warmup 1 --repeat 20 --output bench_pytorch_hybrid.json

# ONNX Bards hybrid (pin ORT threads to the container's physical cores)
pip install 'presidio-analyzer[bards-onnx]'
ORT_INTRA_OP_THREADS=4 ORT_INTER_OP_THREADS=1 \
python presidio-analyzer/evaluation/bards_eu_pii/benchmark_bards_cpu.py \
  --input presidio-analyzer/evaluation/bards_eu_pii/sample_data.jsonl \
  --languages de,en,fr,it --backend onnx --mode hybrid \
  --warmup 1 --repeat 20 --output bench_onnx_hybrid.json
```

Compare `docs_per_second` and `p95_ms` between the two reports. The harness
[README](https://github.com/microsoft/presidio/blob/main/presidio-analyzer/evaluation/bards_eu_pii/README.md)
documents every option.

## Examples

### Python — standard mode

The model owns all PII it supports (no deterministic layer). Simple, but
structured identifiers rely on the model rather than validators:

```python
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_analyzer.predefined_recognizers import BardsEuPiiRecognizer
from presidio_anonymizer import AnonymizerEngine

nlp_engine = NlpEngineProvider(nlp_configuration={
    "nlp_engine_name": "spacy",
    "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
}).create_engine()

analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
analyzer.registry.add_recognizer(BardsEuPiiRecognizer(threshold=0.4))
# Avoid duplicate person/location hits from spaCy's weaker NER.
analyzer.registry.remove_recognizer("SpacyRecognizer")

text = "Contact John Smith at john.smith@example.com or +49 30 12345678."
results = analyzer.analyze(text=text, language="en")
print(AnonymizerEngine().anonymize(text=text, analyzer_results=results).text)
```

### Python — hybrid mode (recommended)

```python
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_analyzer.predefined_recognizers import BardsEuPiiRecognizer

nlp_engine = NlpEngineProvider(nlp_configuration={
    "nlp_engine_name": "spacy",
    "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
}).create_engine()

# A default AnalyzerEngine already registers the deterministic recognizers.
analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
# hybrid() defers EMAIL_ADDRESS, PHONE_NUMBER, IP_ADDRESS, PAYMENT_CARD and
# IDENTIFYING_LINK to the deterministic recognizers; the model keeps the rest.
analyzer.registry.add_recognizer(BardsEuPiiRecognizer.hybrid())
# Let names/locations come from the model rather than spaCy's weaker NER.
analyzer.registry.remove_recognizer("SpacyRecognizer")

text = "Contact John Smith at john.smith@example.com or +49 30 12345678."
results = analyzer.analyze(text=text, language="en")
```

To defer extra labels on top of the structured defaults (for example, hand
`FINANCIAL_AMOUNT` to a custom recognizer too), pass them to `hybrid()` — they
are merged with the defaults, not replaced:

```python
BardsEuPiiRecognizer.hybrid(labels_to_ignore=["FINANCIAL_AMOUNT"])
```

### YAML — hybrid mode

The same hybrid behavior with no Python. `labels_to_ignore` drops the structured
labels so the deterministic recognizers (already registered by default) own them.
Omitted fields fall back to the recognizer's defaults (model id, mapping,
threshold):

```yaml
recognizers:
  - name: BardsEuPiiRecognizer
    type: predefined
    supported_languages: [en, de]
    # Hybrid: defer structured identifiers to the deterministic recognizers.
    labels_to_ignore:
      - EMAIL_ADDRESS
      - PHONE_NUMBER
      - IP_ADDRESS
      - PAYMENT_CARD
      - IDENTIFYING_LINK
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
                "labels_to_ignore": [
                    "EMAIL_ADDRESS", "PHONE_NUMBER", "IP_ADDRESS",
                    "PAYMENT_CARD", "IDENTIFYING_LINK",
                ],
            }
        ],
    }
).create_recognizer_registry()
```

### Per-entity thresholds

Thresholds can be tuned per entity and per language; the most specific match
wins. See [Threshold calibration](#threshold-calibration) for the full rules.

```python
recognizer = BardsEuPiiRecognizer.hybrid(
    threshold=0.4,                                  # global fallback
    thresholds_by_entity={"PERSON": 0.3, "LOCATION": 0.3},
    thresholds_by_language={"de": {"PERSON": 0.35}},  # overrides per-entity for de
)
```

### Mapping profiles

Pick a named profile instead of building a full `label_mapping`. See
[Named mapping profiles](#named-mapping-profiles) for the full table.

```python
from presidio_analyzer.predefined_recognizers.ner.bards_eu_pii_recognizer import (
    MAPPING_PROFILE_GDPR_SENSITIVE,
)

# Keep the GDPR Art. 9 special categories as separate entities (not NRP).
recognizer = BardsEuPiiRecognizer.hybrid(mapping_profile=MAPPING_PROFILE_GDPR_SENSITIVE)
```

```yaml
recognizers:
  - name: BardsEuPiiRecognizer
    type: predefined
    supported_languages: [en, de]
    mapping_profile: gdpr_sensitive
```

## Entity mapping

The recognizer maps the model's 35 labels to Presidio entities via the
`EU_PII_ENTITY_MAPPING` constant. Labels with a faithful Presidio standard entity
use it; the GDPR Art. 9 nationality/religion/politics trio is collapsed onto
Presidio's built-in `NRP` entity; everything else passes through as its own
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

### Named mapping profiles

For the common cases you don't need to build a full `label_mapping` by hand —
pick a named **mapping profile**. Pass `mapping_profile=` (Python) or
`mapping_profile:` (YAML); an explicit `label_mapping` always takes precedence.

| Profile | What it does | When to use |
|---|---|---|
| `presidio_standard` *(default)* | The mapping above exactly: faithful built-ins, the Art. 9 nationality/religion/politics trio collapsed to `NRP`, the rest descriptive. | Default. Maximum compatibility with downstream Presidio tooling. |
| `gdpr_sensitive` | Like `presidio_standard`, but keeps `ETHNIC_ORIGIN`, `RELIGION_OR_BELIEF` and `POLITICAL_OPINION` as their **own** entities instead of `NRP`. | You must treat each GDPR Art. 9 special category separately (distinct anonymizer operator, retention or audit rule). |
| `preserve_model_labels` | Every model label maps to itself — no Presidio remapping at all. | Inspecting the raw 35-label model taxonomy, or doing your own downstream mapping. |
| `high_recall` | Like `presidio_standard`, but also maps the ambiguous `PROPER_NAME` to `PERSON`. | You'd rather over-capture person-like proper names than miss them. |

```python
from presidio_analyzer.predefined_recognizers.ner.bards_eu_pii_recognizer import (
    get_eu_pii_entity_mapping,
)

# Fetch a profile's mapping directly (e.g. to tweak it further):
mapping = get_eu_pii_entity_mapping("gdpr_sensitive")
```

### Model-label drift guard

The built-in mapping (and every profile) is pinned to the model's expected label
set, `EXPECTED_EU_PII_MODEL_LABELS` (the 35 base labels, BIO prefixes stripped
and `O` excluded). At construction the recognizer checks the built-in mapping's
keys against that pinned set and raises a `ValueError` if they don't match, so an
accidental edit to `EU_PII_ENTITY_MAPPING` fails fast. A **custom**
`label_mapping` is not validated by default; opt in with `validate_mapping=True`
(or force the check off for the built-in path with `validate_mapping=False`).

To catch drift in the **remote** model, run the opt-in integration check. It
downloads only the model `config.json` (not the weights), reads `id2label`,
strips prefixes, drops `O`, and compares against `EXPECTED_EU_PII_MODEL_LABELS`:

```bash
PRESIDIO_RUN_BARDS_EU_PII_INTEGRATION=1 \
  pytest presidio-analyzer/tests/test_bards_eu_pii_recognizer.py -k drift
```

If it fails, the upstream checkpoint's labels changed; update
`EU_PII_ENTITY_MAPPING` and `EXPECTED_EU_PII_MODEL_LABELS` together. Normal
(offline) unit tests skip this check and never touch the network.

## Multilingual usage

The checkpoint is a single multilingual model, but a recognizer instance is
registered for one language. To cover several EU languages, register one instance
per language (each pointing at the same model):

```python
for lang in ("en", "de", "fr"):
    analyzer.registry.add_recognizer(BardsEuPiiRecognizer.hybrid(supported_language=lang))
```

Make sure your `AnalyzerEngine` / NLP engine is configured for those languages.

### Language-specific context for deterministic recognizers

In hybrid mode the deterministic recognizers own the structured identifiers
(IBANs, documents, …). Presidio can **boost** a recognizer's confidence when
supportive context words appear near a match — but the words must be in the right
language. The bundled recognizers ship English context (for example
`IbanRecognizer`'s default is `["iban", "bank", "transaction"]`), so a German
*Konto* next to an IBAN won't help unless the recognizer also carries German
context.

> Context is a **confidence boost, not a detector.** It only raises the score of a
> span the recognizer already matched — and only when a context word matches the
> recognizer's own context words; it never creates a detection on its own. Use it
> to lift borderline structured identifiers, not as your only signal.

Define context groups per language:

```python
BANK_ACCOUNT_CONTEXT = {
    "en": ["iban", "bank", "account", "bic", "swift"],
    "de": ["iban", "bank", "konto", "bic", "swift"],
    "fr": ["iban", "banque", "compte", "bic", "swift"],
    "it": ["iban", "banca", "conto", "bic", "swift"],
}

DOCUMENT_CONTEXT = {
    "en": ["passport", "id", "identity", "document"],
    "de": ["pass", "ausweis", "personalausweis", "dokument"],
    "fr": ["passeport", "identité", "document"],
    "it": ["passaporto", "identità", "documento"],
}
```

**Recognizer-level context (durable).** Register the deterministic recognizers
once per language with that language's words, so words *in the text* near a match
boost the score:

```python
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_analyzer.predefined_recognizers import (
    BardsEuPiiRecognizer,
    IbanRecognizer,
)

nlp_engine = NlpEngineProvider(nlp_configuration={
    "nlp_engine_name": "spacy",
    "models": [
        {"lang_code": "en", "model_name": "en_core_web_sm"},
        {"lang_code": "de", "model_name": "de_core_news_sm"},
        {"lang_code": "fr", "model_name": "fr_core_news_sm"},
        {"lang_code": "it", "model_name": "it_core_news_sm"},
    ],
}).create_engine()

analyzer = AnalyzerEngine(
    nlp_engine=nlp_engine, supported_languages=["en", "de", "fr", "it"]
)

for lang in ("en", "de", "fr", "it"):
    # Bards owns free-form PII; the deterministic recognizer owns IBANs and
    # carries this language's context words so surrounding text can boost it.
    analyzer.registry.add_recognizer(
        BardsEuPiiRecognizer.hybrid(supported_language=lang)
    )
    analyzer.registry.add_recognizer(
        IbanRecognizer(supported_language=lang, context=BANK_ACCOUNT_CONTEXT[lang])
    )

# "Konto" near the IBAN boosts the deterministic recognizer's confidence.
text = "Bitte überweisen Sie den Betrag auf das Konto DE89370400440532013000."
results = analyzer.analyze(text=text, language="de")
```

**Request-level context (supplemental).** When the keyword isn't physically next
to the entity but you know the document's domain, pass the same language's words
to `analyze(..., context=[...])`. They are treated as if they appeared near every
match and boost recognizers whose context they match. Since `analyze` is called
per language, pass the list for that request's language:

```python
# German banking document:
results = analyzer.analyze(
    text="Konto: DE89370400440532013000",
    language="de",
    context=BANK_ACCOUNT_CONTEXT["de"],
)

# English document referencing an identity document:
results = analyzer.analyze(
    text="The number on the attached scan is X1234567.",
    language="en",
    context=DOCUMENT_CONTEXT["en"],
)
```

The universal tokens (`iban`, `bic`, `swift`) are language-agnostic and already
match the bundled English recognizers, but the language-specific words (`konto`,
`compte`, `conto`) only help once the recognizer carries them — which is why the
per-language registration above matters in multilingual setups.

## Threshold calibration

A single threshold is coarse: the model's confidence behaves differently per
entity type and per language, so one cut-off either lets noise through on the easy
entities or drops recall on the hard ones. Lower thresholds favour **recall**
(keep more, anonymize aggressively — good for compliance); higher thresholds
favour **precision** (keep only confident hits — good for preserving data
utility).

`BardsEuPiiRecognizer` resolves a threshold for each detected span in this order
(most specific first), keyed by the **mapped Presidio entity** (`PERSON`, not
`PERSON_NAME`):

1. `thresholds_by_language[supported_language][entity]`
2. `thresholds_by_entity[entity]`
3. the global `threshold`

When you pass neither map, the recognizer behaves exactly as a single global
threshold — there is no change for existing users.

**Recall-heavy** (compliance — catch as much as possible, tolerate some
over-redaction):

```python
recognizer = BardsEuPiiRecognizer.hybrid(
    threshold=0.35,
    thresholds_by_entity={
        "PERSON": 0.25,
        "LOCATION": 0.25,
        "NRP": 0.30,        # GDPR Art. 9 special categories
    },
)
```

**Precision-heavy** (data utility — only redact confident hits):

```python
recognizer = BardsEuPiiRecognizer.hybrid(
    threshold=0.6,
    thresholds_by_entity={"PERSON": 0.75, "ORGANIZATION": 0.8},
    thresholds_by_language={"de": {"PERSON": 0.65, "LOCATION": 0.85}},
)
```

The same maps work in YAML:

```yaml
recognizers:
  - name: BardsEuPiiRecognizer
    type: predefined
    supported_languages: [en, de]
    threshold: 0.6
    thresholds_by_entity:
      PERSON: 0.75
      ORGANIZATION: 0.8
    thresholds_by_language:
      de:
        PERSON: 0.65
        LOCATION: 0.85
```

All threshold values (global and in both maps) must be numbers between `0.0` and
`1.0`; an out-of-range or non-numeric value raises `ValueError` at construction.
Choose the values from data — see the evaluation harness below, which sweeps
thresholds and recommends balanced / high-recall / high-precision profiles.

> The default `aggregation_strategy="first"` keeps each entity as one contiguous
> span. This model's fast tokenizer does not expose real word ids, so under
> `"simple"` the pipeline's fallback heuristic fragments subword entities (an
> e-mail address splits into many pieces); prefer the default.

## Quality tuning checklist

- [ ] **Use hybrid mode** (`BardsEuPiiRecognizer.hybrid()`) so deterministic
  recognizers own structured identifiers and the model owns free-form PII.
- [ ] **Calibrate thresholds on your own data** — global, per-entity and
  per-language — rather than assuming the `0.4` default fits your domain.
- [ ] **Evaluate per language and per entity**, not just a single overall number;
  quality varies a lot across both.
- [ ] **Look at both overlap-span and exact-span metrics** — overlap answers "did
  we cover the PII for redaction", exact answers "are the boundaries right".
- [ ] **Test OCR-noisy, unusually formatted, and code-switched text**, which is
  where recall typically drops.
- [ ] **Add language-specific context words** to the deterministic recognizers to
  boost their confidence on structured identifiers in your domain — see
  [Language-specific context for deterministic recognizers](#language-specific-context-for-deterministic-recognizers).

## Evaluating quality

A reproducible, offline-friendly evaluation harness lives at
[`presidio-analyzer/evaluation/bards_eu_pii/`](https://github.com/microsoft/presidio/tree/main/presidio-analyzer/evaluation/bards_eu_pii)
— see its
[README](https://github.com/microsoft/presidio/blob/main/presidio-analyzer/evaluation/bards_eu_pii/README.md).
It computes exact-span and overlap-span precision/recall/F1 (and F2) per entity
and per language, sweeps thresholds to recommend profiles, and includes
robustness cases (OCR noise, spaced formatting, `Straße`/`Strasse`,
code-switching, usernames):

```bash
# Evaluate a hybrid setup on your own JSONL dataset.
python presidio-analyzer/evaluation/bards_eu_pii/evaluate_bards_eu_pii.py \
  --input your_data.jsonl --mode hybrid --threshold 0.4 --output results.json

# Sweep thresholds and get balanced / high-recall / high-precision recommendations.
python presidio-analyzer/evaluation/bards_eu_pii/evaluate_bards_eu_pii.py \
  --input your_data.jsonl --mode hybrid \
  --threshold-sweep 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8 --sweep-entity PERSON
```

### Illustrative benchmark

> **Illustrative example, not an authoritative benchmark.** The figures below are
> one run on one dataset / hardware / model-version combination; model versions,
> datasets and thresholds all move the numbers. **Regenerate them for your own
> data** with the harness above rather than relying on these values.

One run on a German benchmark (ai4privacy `pii-masking-200k`, 2,000 examples),
keeping the **same regex/checksum layer** and swapping only the NER backend
(span-overlap F1; the structured buckets are identical by construction and
omitted):

| NER bucket | spaCy `de_core_news_lg` | GLiNER2 | Bards EU-PII |
|---|---|---|---|
| PERSON   | 0.645 | 0.663 | **0.861** |
| LOCATION | 0.580 | 0.703 | **0.799** |
| USERNAME | 0.000 | 0.483 | **0.585** |
| **micro (NER)** | 0.590 | 0.661 | **0.815** |
| **micro (all 8 buckets)** | 0.665 | 0.705 | **0.820** |

In this run Bards led every NER bucket with high precision (PERSON P=0.80,
LOCATION P=0.85), which matters when you anonymize by redaction — but treat it as
a starting hypothesis to confirm or refute on your own data, not a guarantee.

## Deploy as a container (Bards + regex)

The repo ships a ready-to-run **hybrid** analyzer image — deterministic regex
recognizers own structured identifiers, Bards owns free-text PII — plus the stock
anonymizer:

- `presidio-analyzer/Dockerfile.bards` — the analyzer image. It defaults to the
  **CPU-optimized ONNX backend** (`BardsEuPiiOnnxRecognizer`, the quantized
  `onnx/model_quantized.onnx`). The model is **baked in** at build time and
  served offline (`HF_HUB_OFFLINE=1`), so the container never downloads a model
  at runtime. The image presets the CPU defaults from
  [CPU deployment tuning](#cpu-deployment-tuning): `WORKERS=1`,
  `TOKENIZERS_PARALLELISM=false`, and `ORT_INTRA_OP_THREADS` left unset for you
  to pin to the container's cores. Build the **PyTorch variant** with
  `--build-arg BARDS_BACKEND=pytorch` and
  `--build-arg RECOGNIZER_REGISTRY_CONF_FILE=presidio_analyzer/conf/bards_hybrid_recognizers_pytorch.yaml`.
- `presidio-analyzer/presidio_analyzer/conf/bards_hybrid*.yaml` — the configs
  that wire it up: a small per-language spaCy NLP engine (`bards_hybrid.yaml`),
  the language set (`bards_hybrid_analyzer.yaml`), and the hybrid recognizer
  registry. `bards_hybrid_recognizers.yaml` (default) selects
  `BardsEuPiiOnnxRecognizer`; `bards_hybrid_recognizers_pytorch.yaml` selects the
  PyTorch `BardsEuPiiRecognizer`. Both set `labels_to_ignore` and disable the
  spaCy NER recognizer.
- `docker-compose-bards.yml` — the analyzer + anonymizer pair.
- `.github/workflows/build-bards-hybrid.yml` — builds and pushes both images to
  GHCR (`ghcr.io/<owner>/presidio-analyzer-bards`, `…/presidio-anonymizer`) on
  pushes to the feature branch, `bards-v*` tags, or manual dispatch.

Run the published images, or build locally:

```bash
# pull + run the pair (analyzer on :5002, anonymizer on :5001)
docker compose -f docker-compose-bards.yml up -d

# or build locally
docker compose -f docker-compose-bards.yml build
```

The image ships with **en, de, fr, it**. Presidio registers a Bards instance per
language, but they **share one in-process model** — a single ONNX Runtime session
on the default ONNX backend, or mmap-shared `safetensors` weight pages on the
PyTorch variant — so the four languages do not cost 4× the model. On the PyTorch
variant the four languages measured about **1.1 GB of model RAM combined**.
Budget roughly **2 GB** for the analyzer (the compose file sets a 2 GB limit);
trim the language set in the `bards_hybrid*.yaml` configs (and rebuild) only if
you need to shave the remaining overhead.

## Limitations

- **Model quality varies by language and domain.** It is one multilingual model;
  recall and precision differ substantially across the 24 languages and across
  text domains. Evaluate on text representative of your use case.
- **Unusual formatting and OCR noise reduce recall.** Spaced-out or OCR-corrupted
  text (`J0hn Sm1th`, `john . smith @ example . com`) is often missed. Normalize
  such input **upstream** of Presidio; the evaluation harness includes optional,
  eval-only normalization hooks to measure the potential gain.
- **Structured identifiers should use validators/checksums when available.** For
  e-mail, phone, IBAN, credit card and IP, prefer Presidio's deterministic
  recognizers (regex + checksum) over the model — this is exactly what hybrid mode
  arranges.
- **Not a standalone compliance solution.** No PII model catches everything;
  combine it with deterministic recognizers, human review where the stakes are
  high, and evaluation on your own data. Do not rely on it alone for regulatory
  compliance.
- **No dedicated "username" label.** Bare login-usernames are detected, but the
  model files them under `ACCOUNT_IDENTIFIER` (and name-shaped handles under
  `PERSON_NAME`); it never emits `CONTACT_HANDLE` for them (that label is for
  social `@handles`). To collect them as a `USERNAME` entity, add
  `ACCOUNT_IDENTIFIER → USERNAME` to your `label_mapping` — though name-shaped
  handles still surface as `PERSON`, which the model cannot disambiguate.
- **Production / CPU.** For CPU-only deployments use `BardsEuPiiOnnxRecognizer`,
  which runs the model card's quantized ONNX weights (`onnx/model_quantized.onnx`)
  through ONNX Runtime; `BardsEuPiiRecognizer` runs the PyTorch checkpoint and is
  the better fit on GPU. The two are drop-in interchangeable — see
  [CPU-optimized inference (ONNX Runtime)](#cpu-optimized-inference-onnx-runtime)
  for the backend choice and CPU deployment tuning.
