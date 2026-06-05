# Bards EU-PII evaluation harness

A small, reproducible harness to measure the quality of
[`BardsEuPiiRecognizer`](../../presidio_analyzer/predefined_recognizers/ner/bards_eu_pii_recognizer.py)
on a local dataset. It computes **exact-span** and **overlap-span**
precision / recall / F1 (and F-beta), per entity type, per language, and
micro-averaged.

> The harness is **optional**. It is not part of the regular unit tests and never
> runs in normal CI. Running `evaluate_bards_eu_pii.py` downloads/loads the
> Hugging Face model the first time; the metrics module itself
> (`metrics.py`) is pure standard library and is unit-tested offline.

## Files

| File | Purpose |
|---|---|
| `evaluate_bards_eu_pii.py` | CLI: run the recognizer over a dataset and emit metrics JSON. |
| `benchmark_bards_cpu.py` | CLI: measure CPU throughput/latency (PyTorch vs ONNX). Speed only, not quality. |
| `metrics.py` | Pure-Python span matching + P/R/F1/F-beta. Offline, unit-tested. |
| `sweep.py` | Threshold-sweep recommendation logic. Offline, unit-tested. |
| `preprocess.py` | Experimental, eval-only text normalization (OCR/spaced email). Offline, unit-tested. |
| `ensemble.py` | Union / intersection span combination for comparing NER backends. Offline, unit-tested. |
| `schema.py` | JSONL record schema + loader/validator. |
| `label_maps.py` | Maps recognizer entity types onto evaluation buckets. |
| `sample_data.jsonl` | Tiny synthetic dataset (de/en/fr/it) for a fast smoke run. |
| `robustness_sample_data.jsonl` | Synthetic noisy cases (OCR, spacing, Straße/Strasse, code-switching, usernames). |
| `tests/` | Offline tests for `metrics.py`, `sweep.py`, `preprocess.py`, `ensemble.py`, and the benchmark/backend-routing CLIs (no model download). |

## Data format

One JSON object per line (JSONL):

```json
{
  "id": "example-1",
  "language": "de",
  "text": "Kontaktieren Sie Max Müller unter max@example.de",
  "spans": [
    {"start": 17, "end": 27, "entity_type": "PERSON"},
    {"start": 34, "end": 48, "entity_type": "EMAIL_ADDRESS"}
  ]
}
```

`start`/`end` are half-open character offsets into `text`. `entity_type` is an
**evaluation bucket** (e.g. `PERSON`, `EMAIL_ADDRESS`, `LOCATION`, `USERNAME`).

## Run against the sample data

```bash
pip install 'presidio-analyzer[transformers]'

python presidio-analyzer/evaluation/bards_eu_pii/evaluate_bards_eu_pii.py \
  --input presidio-analyzer/evaluation/bards_eu_pii/sample_data.jsonl \
  --mode hybrid \
  --threshold 0.4 \
  --output /tmp/bards_eval.json
```

CLI options:

- `--mode standard` uses a plain `BardsEuPiiRecognizer`; `--mode hybrid` uses
  `BardsEuPiiRecognizer.hybrid()` plus Presidio's deterministic recognizers
  (email, phone, IP, credit card, URL, IBAN) for the structured entities.
- `--languages de,en,fr,it` keeps only those languages (default: all in the file).
- `--entities PERSON,EMAIL_ADDRESS` restricts both gold and predictions to those
  evaluation buckets.
- `--threshold`, `--mapping-profile`, `--labels-to-ignore`,
  `--thresholds-by-entity` (JSON), `--thresholds-by-language` (JSON) are passed
  through to the recognizer.
- `--betas 1.0,2.0` chooses which F-beta values to report.
- `--output results.json` writes a deterministic, sorted JSON report (safe to
  commit or diff in CI). A short summary is printed to stdout regardless.

## Run against a converted benchmark dataset

The harness reads its own JSONL format, so convert a public benchmark to it
first. For example, [`ai4privacy/pii-masking-200k`](https://huggingface.co/datasets/ai4privacy/pii-masking-200k):

1. Filter to the language(s) you want.
2. For each row, emit `{"id", "language", "text", "spans"}` where each span has
   character offsets and an **evaluation bucket** `entity_type`. Map the
   dataset's own labels onto your buckets (e.g. `FIRSTNAME`/`LASTNAME` →
   `PERSON`, `USERNAME` → `USERNAME`).
3. Predictions are mapped to buckets via `label_maps.py`
   (`DEFAULT_PREDICTION_LABEL_MAP`, e.g. `ACCOUNT_IDENTIFIER` → `USERNAME`). Pass
   `--label-map '{"...": "..."}'` to override it.

Keeping gold and predicted labels in the **same bucket taxonomy** is what makes
the numbers comparable.

## Comparing NER backends (Bards, GLiNER, HuggingFace)

The harness can evaluate other optional NER recognizers next to Bards and combine
them. Only Bards is required; `gliner` and `huggingface` are **optional** — if the
package or model is missing the run fails with an actionable error (it is never a
hard dependency, and the offline unit tests never load any backend).

```bash
# Bards only (default) / Bards hybrid
python .../evaluate_bards_eu_pii.py --input data.jsonl --backend bards
python .../evaluate_bards_eu_pii.py --input data.jsonl --backend bards --mode hybrid

# GLiNER only / GLiNER hybrid  (pip install gliner)
python .../evaluate_bards_eu_pii.py --input data.jsonl --backend gliner
python .../evaluate_bards_eu_pii.py --input data.jsonl --backend gliner --mode hybrid

# A generic HuggingFace token-classification model
python .../evaluate_bards_eu_pii.py --input data.jsonl \
  --backend huggingface --huggingface-model dslim/bert-base-NER

# Union ensemble: keep every span either backend finds
python .../evaluate_bards_eu_pii.py --input data.jsonl \
  --backend bards --backend gliner --ensemble union --mode hybrid

# Intersection / agreement ensemble: keep only spans both backends agree on
python .../evaluate_bards_eu_pii.py --input data.jsonl \
  --backend bards --backend gliner --ensemble intersection --mode hybrid
```

- `--backend {bards,gliner,huggingface}` — repeat to compare/ensemble several;
  defaults to `bards`. `--gliner-model` / `--huggingface-model` pick the model id.
- `--ensemble {union,intersection}` — how to combine multiple backends. **Union**
  keeps all non-duplicate spans (higher recall); **intersection** keeps only spans
  the backends agree on — overlapping with the same entity type in every backend
  (higher precision). Defaults to `union` when more than one backend is given.
- In **hybrid** mode the deterministic recognizers own the structured identifiers
  for *all* backends, so the ensemble compares only free-text NER quality.
- The combination logic lives in `ensemble.py` and is unit-tested offline with
  synthetic `RecognizerResult` objects (no model). `--threshold-sweep` is not
  combined with multi-backend comparison — sweep the default Bards backend, then
  compare at the chosen threshold.

## Benchmarking CPU throughput (PyTorch vs ONNX)

`evaluate_bards_eu_pii.py` measures **quality**; `benchmark_bards_cpu.py`
measures **speed**. Use it to measure the ONNX CPU speed-up on your own hardware:
the same model runs through either the PyTorch checkpoint — the
[`BardsEuPiiRecognizer`](../../presidio_analyzer/predefined_recognizers/ner/bards_eu_pii_recognizer.py)
PyTorch path — or the upstream quantized ONNX file `onnx/model_quantized.onnx` —
the CPU-optimized
[`BardsEuPiiOnnxRecognizer`](../../presidio_analyzer/predefined_recognizers/ner/bards_eu_pii_onnx_recognizer.py)
ONNX Runtime path (needs the `bards-onnx` extra). The production Bards hybrid
container uses the ONNX path by default; see the
[deployment guide](../../../docs/samples/python/bards_eu_pii.md#cpu-optimized-inference-onnx-runtime).
Presidio and the model are imported lazily, so the script — and its offline
tests — stay light.

Input is either a harness JSONL file (the `text`/`language` fields are reused) or
a plain text file (one document per non-blank line, tagged with the first
`--languages` entry).

```bash
# 1. Compare PyTorch vs ONNX on the sample data (warm up once, time 20 passes)
pip install 'presidio-analyzer[transformers]'   # PyTorch backend
python presidio-analyzer/evaluation/bards_eu_pii/benchmark_bards_cpu.py \
  --input presidio-analyzer/evaluation/bards_eu_pii/sample_data.jsonl \
  --languages de,en,fr,it --backend pytorch --warmup 1 --repeat 20 \
  --output /tmp/bench_pytorch.json

pip install 'presidio-analyzer[bards-onnx]'      # ONNX Runtime backend
python presidio-analyzer/evaluation/bards_eu_pii/benchmark_bards_cpu.py \
  --input presidio-analyzer/evaluation/bards_eu_pii/sample_data.jsonl \
  --languages de,en,fr,it --backend onnx --warmup 1 --repeat 20 \
  --output /tmp/bench_onnx.json

# 2. Hybrid ONNX benchmark (NER + deterministic recognizers, as in production)
python presidio-analyzer/evaluation/bards_eu_pii/benchmark_bards_cpu.py \
  --input presidio-analyzer/evaluation/bards_eu_pii/sample_data.jsonl \
  --languages de,en,fr,it --backend onnx --mode hybrid --warmup 1 --repeat 20 \
  --output /tmp/bench_onnx_hybrid.json

# 3. Pin ONNX Runtime thread counts for a CPU container, then benchmark
ORT_INTRA_OP_THREADS=4 ORT_INTER_OP_THREADS=1 \
python presidio-analyzer/evaluation/bards_eu_pii/benchmark_bards_cpu.py \
  --input presidio-analyzer/evaluation/bards_eu_pii/sample_data.jsonl \
  --backend onnx --warmup 1 --repeat 50 --output /tmp/bench_onnx_4threads.json
```

- `--backend {pytorch,onnx}` — which recognizer to time (default `onnx`).
- `--mode {standard,hybrid}` — `standard` times the NER backend only; `hybrid`
  also runs the deterministic recognizers (email, phone, IP, credit card, URL,
  IBAN) per document, matching the production hybrid pipeline.
- `--languages de,en,fr,it` — one recognizer is built per language; plain-text
  inputs use the first language.
- `--warmup N` — untimed passes to load the model and let ONNX Runtime settle
  before timing (default 1).
- `--repeat N` — timed passes over the whole input (default 1). More passes give
  stabler percentiles; `p99_ms` is only reported once there are ≥ 100 samples.
- `--output results.json` — write the report; a one-line summary always prints.

The report records `docs_per_second`, `chars_per_second`, `total_seconds` and
`p50_ms`/`p95_ms`/`p99_ms` latencies (timed with `time.perf_counter`). For the
ONNX backend an `onnx` block echoes the provider, ONNX file and the resolved
`intra_op`/`inter_op` thread counts plus the `ORT_*` env vars, so a benchmark JSON
is self-describing. If `psutil` is installed an `rss_mb` field is added (it is
**not** a required dependency — the script omits the field when it is absent).

`ORT_INTRA_OP_THREADS` / `ORT_INTER_OP_THREADS` (or the
`onnx_intra_op_num_threads` / `onnx_inter_op_num_threads` constructor kwargs) tune
ONNX Runtime's parallelism. In a CPU container, pinning intra-op threads to the
container's CPU limit and leaving `WORKERS=1` is usually fastest; benchmark a few
values to find the best fit for your hardware.

## Threshold sweep: data-backed threshold profiles

Instead of guessing a threshold, sweep a range and let the harness recommend one:

```bash
python presidio-analyzer/evaluation/bards_eu_pii/evaluate_bards_eu_pii.py \
  --input your_data.jsonl \
  --mode hybrid \
  --threshold-sweep 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8 \
  --sweep-entity PERSON \
  --min-recall-for-high-precision 0.80 \
  --output sweep.json
```

The model runs **once** per example (at the lowest swept threshold) and each
threshold re-scores those cached predictions by confidence, so the sweep
evaluates every threshold without re-running inference. `--threshold-sweep`
sweeps the single **global** threshold and replaces `--threshold`
(per-entity/per-language threshold maps are not applied during a sweep).

For each threshold the report records micro precision / recall / F1 / F2 in both
`exact` and `overlap` modes, and then recommends three profiles (computed on
`--sweep-mode`, default `overlap`):

| Profile | Chosen threshold | Use it for |
|---|---|---|
| `balanced` | highest **F1** | A sensible default trade-off. |
| `high_recall` | highest **F2** | Compliance / redaction — catch as much PII as possible. |
| `high_precision` | highest **precision** with recall ≥ `--min-recall-for-high-precision` | Data utility — redact only confident hits without dropping below a recall floor. |

Ties are broken deterministically (higher recall, then lower threshold for the
recall-leaning profiles, higher threshold for `high_precision`). If no threshold
clears the recall floor, `high_precision` is reported as `null`.

`--sweep-entity PERSON` adds a second `sweep_entity` block that runs the same
analysis for one bucket, so you can tune a single entity independently.

### Turning recommendations into recognizer settings

- **Global threshold** — take the recommended profile's `threshold` and pass it
  as `threshold=` (or `--threshold`):

  ```python
  BardsEuPiiRecognizer(threshold=0.3)  # e.g. the high_recall pick
  ```

- **Per-entity threshold** — run with `--sweep-entity PERSON` (and again for
  other entities), then combine the per-entity picks into `thresholds_by_entity`:

  ```python
  BardsEuPiiRecognizer(
      threshold=0.4,                       # balanced micro pick
      thresholds_by_entity={"PERSON": 0.25, "LOCATION": 0.3},  # per-entity picks
  )
  ```

- **Per-language threshold** — sweep per language with `--languages de` (one run
  each), then assemble `thresholds_by_language`:

  ```python
  BardsEuPiiRecognizer(
      supported_language="de",
      thresholds_by_language={"de": {"PERSON": 0.35}},
  )
  ```

The sweep output JSON is deterministic (sorted keys, rounded), so you can commit
a baseline and diff it as the model or data changes.

## Robustness evaluation (OCR noise, formatting, code-switching)

`robustness_sample_data.jsonl` collects synthetic cases that PII detection
commonly degrades on: OCR/leet substitutions (`J0hn Sm1th`), spaced-out e-mails
(`john . smith @ example . com`), phone-spacing variants, German `Straße` vs
`Strasse`, code-switched English/German/French sentences, and
usernames/account identifiers.

Run it like any other dataset:

```bash
python presidio-analyzer/evaluation/bards_eu_pii/evaluate_bards_eu_pii.py \
  --input presidio-analyzer/evaluation/bards_eu_pii/robustness_sample_data.jsonl \
  --mode hybrid --threshold 0.4 --output robustness.json
```

### Compare standard vs hybrid mode

Run the same dataset twice and diff the two reports:

```bash
for MODE in standard hybrid; do
  python presidio-analyzer/evaluation/bards_eu_pii/evaluate_bards_eu_pii.py \
    --input presidio-analyzer/evaluation/bards_eu_pii/robustness_sample_data.jsonl \
    --mode "$MODE" --threshold 0.4 --output "robustness_$MODE.json"
done
```

`standard` uses only the model; `hybrid` adds the deterministic regex/checksum
recognizers. A large gap on structured buckets (email, phone, IP, credit card,
URL) means those entities should be owned by the deterministic layer — i.e. use
hybrid in production.

### Experimental preprocessing hooks (eval-only)

Two opt-in flags normalize the **input text before analysis**, to measure how
much light preprocessing would recover. They are **local to this harness and do
not change analyzer behavior** — production detection is unchanged unless a
later PR intentionally adds preprocessing:

- `--normalize-ocr-noise` — map leet/OCR digits back to letters when flanked by
  ASCII letters (`J0hn` → `John`, `Sm1th` → `Smith`); pure-numeric tokens
  (phones, account IDs) are left alone. Length-preserving.
- `--normalize-spaced-email` — collapse whitespace inside spaced e-mails
  (`john . smith @ example . com` → `john.smith@example.com`).

Predicted spans are mapped **back to the original offsets** before scoring, so
metrics stay comparable to the un-normalized gold. Example:

```bash
python presidio-analyzer/evaluation/bards_eu_pii/evaluate_bards_eu_pii.py \
  --input presidio-analyzer/evaluation/bards_eu_pii/robustness_sample_data.jsonl \
  --mode hybrid --threshold 0.4 \
  --normalize-ocr-noise --normalize-spaced-email \
  --output robustness_normalized.json
```

### Reading failures: preprocessing, regex, or threshold?

When a robustness case is missed, the cause is usually one of three, and the
harness helps you tell them apart:

- **Preprocessing issue** — the entity is recovered only after a
  `--normalize-*` flag (e.g. the spaced e-mail's recall jumps once whitespace is
  collapsed). The raw input is malformed for the detector; fix it by normalizing
  text **upstream** of Presidio (OCR cleanup, formatting), not in the analyzer.
- **Regex / deterministic-layer issue** — a structured entity is missed in
  `--mode standard` but caught in `--mode hybrid`. The model is the wrong tool
  for rigid formats; rely on the deterministic recognizers (or add/adjust a
  pattern recognizer).
- **Threshold issue** — recall climbs as the threshold drops in a
  `--threshold-sweep`. The detector *does* fire but below your cut-off; lower the
  global threshold or set a per-entity threshold (`thresholds_by_entity`).

If none of these recover the case (e.g. a code-switched name the model never
proposes at any threshold or after normalization), it is a genuine model-recall
gap rather than a preprocessing/regex/threshold one.

## Interpreting exact vs overlap metrics

- **Exact-span** counts a prediction correct only if the entity type *and* both
  offsets match the gold span. It is strict and punishes boundary disagreements
  (e.g. predicting `"Müller"` when the gold is `"Max Müller"`).
- **Overlap-span** counts a prediction correct if the entity type matches and the
  character ranges intersect at all. It measures "did we find and label the PII
  at roughly the right place", independent of exact tokenization.

For anonymization, overlap is usually the more meaningful signal: if any part of
the PII is detected and redacted, the sensitive token is covered. Exact-span is
useful when downstream consumers need precise boundaries. Report both.

## Why F2 is useful for PII recall

F-beta weights recall by `beta`. For PII/compliance you typically care more about
**not missing** sensitive data than about the occasional false positive, so
recall matters more than precision — `F2` (beta = 2) weights recall ~4× as
heavily as precision and is a better single number for "how safe is this for
redaction" than `F1`. `F1` (the default beta = 1) stays in the report for
balanced comparisons. Lower the recognizer's threshold (or set per-entity
thresholds) to trade precision for the recall that `F2` rewards.

## Regenerating the docs benchmark

The benchmark table in
[`docs/samples/python/bards_eu_pii.md`](../../../docs/samples/python/bards_eu_pii.md)
is an **example** produced with this harness. Regenerate it on your own data and
hardware rather than treating those figures as authoritative — model versions,
datasets and thresholds all move the numbers.

## Running the offline metrics tests

```bash
pytest presidio-analyzer/evaluation/bards_eu_pii
```

These import only `metrics.py` and never download a model or dataset.
