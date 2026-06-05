"""ONNX Runtime (INT8) variant of :class:`BardsEuPiiRecognizer` for CPU.

Runs the ``bardsai/eu-pii-anonimization-multilang`` model on CPU through the
upstream quantized ONNX file (``onnx/model_quantized.onnx``) with **ONNX Runtime
only** — no PyTorch and no Optimum. Inference is a plain
``onnxruntime.InferenceSession`` plus the model's fast tokenizer and a small
numpy BIO-aggregation step; the transformers ``token-classification`` pipeline
(which would pull in torch) is not used. It is a subclass of
:class:`BardsEuPiiRecognizer`: the default model id, entity mapping, mapping
profiles, thresholds, ``thresholds_by_entity`` / ``thresholds_by_language``,
``labels_to_ignore``, the :meth:`hybrid` constructor and the label/threshold
logic are all inherited. Only model loading and the raw inference differ.

The default PyTorch :class:`BardsEuPiiRecognizer` is unchanged; this recognizer
is opt-in (registered explicitly in code or in a YAML registry config). The
ONNX-specific dependencies (``onnxruntime``, ``transformers`` for the tokenizer)
are imported lazily, only in the load path, so importing this module — and the
normal unit tests — do not require them, and the recognizer can be constructed
without torch. Install them with ``pip install 'presidio-analyzer[bards-onnx]'``.
"""

import logging
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

from presidio_analyzer import RecognizerResult
from presidio_analyzer.nlp_engine import NlpArtifacts
from presidio_analyzer.predefined_recognizers.ner.bards_eu_pii_recognizer import (
    DEFAULT_EU_PII_MODEL,
    MAPPING_PROFILE_PRESIDIO_STANDARD,
    BardsEuPiiRecognizer,
)

logger = logging.getLogger("presidio-analyzer")

#: Default location of the upstream quantized ONNX file within the model repo.
DEFAULT_ONNX_MODEL_SUBFOLDER = "onnx"
DEFAULT_ONNX_MODEL_FILE = "model_quantized.onnx"
DEFAULT_ONNX_PROVIDER = "CPUExecutionProvider"
#: Hard cap on tokens fed to the model (XLM-RoBERTa positional limit). The text
#: chunker keeps chunks well under this; truncation here is only a safety net.
DEFAULT_ONNX_MAX_LENGTH = 512

# Module-level cache of loaded ORT sessions, keyed by everything that affects the
# loaded session (see ``_cache_key``). Multiple per-language recognizer instances
# that share a key reuse one heavy ORT session + tokenizer; each instance still
# builds its own lightweight inference pipeline (and ignored-label filter) around
# it, so no per-recognizer mutable state (labels_to_ignore, thresholds) is shared
# through the cache.
_ORT_SESSION_CACHE: Dict[Tuple[Any, ...], Tuple[Any, Any, Dict[int, str]]] = {}
_ORT_SESSION_CACHE_LOCK = threading.Lock()


def _resolve_threads(explicit: Optional[int], env_name: str) -> Optional[int]:
    """Resolve an ORT thread count: explicit value, else a positive-int env var.

    An explicit (non-``None``) value always wins. Otherwise ``env_name`` is read
    and accepted only if it parses to a positive integer; anything else (unset,
    non-numeric, zero or negative) yields ``None`` (ORT picks its own default).
    """
    if explicit is not None:
        return explicit
    raw = os.getenv(env_name)
    if raw is None:
        return None
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Ignoring %s=%r: expected an integer.", env_name, raw)
        return None
    if value <= 0:
        logger.warning(
            "Ignoring %s=%r: expected a positive integer.", env_name, raw
        )
        return None
    return value


class _OnnxTokenClassificationPipeline:
    """Torch-free token-classification pipeline over an ONNX Runtime session.

    Mimics the slice of the transformers token-classification pipeline that
    :class:`BardsEuPiiRecognizer` consumes: calling the instance with a string
    returns a list of ``{"entity_group", "score", "start", "end", "word"}`` dicts
    with the BIO tags already aggregated into character spans. Inference is ONNX
    Runtime + numpy only, so no torch / Optimum is required.

    Aggregation merges consecutive tokens that share an entity type (the model
    BIO-tags every sub-word, so ``B-``/``I-`` is not a reliable word boundary);
    an ``O`` token ends the current span. The span score is the first token's
    probability (``aggregation_strategy="first"`` semantics).
    """

    def __init__(
        self,
        session: Any,
        tokenizer: Any,
        id2label: Dict[int, str],
        max_length: int = DEFAULT_ONNX_MAX_LENGTH,
    ):
        self._session = session
        self._tokenizer = tokenizer
        self._id2label = id2label
        self._max_length = max_length
        self._input_names = {model_input.name for model_input in session.get_inputs()}
        self._output_name = session.get_outputs()[0].name

    def __call__(self, text: str) -> List[Dict[str, Any]]:
        """Run ONNX inference on ``text`` and return aggregated entity spans."""
        import numpy as np

        if not text:
            return []
        encoding = self._tokenizer(
            text,
            return_offsets_mapping=True,
            return_tensors="np",
            truncation=True,
            max_length=self._max_length,
        )
        offsets = encoding["offset_mapping"][0]
        feeds = {
            name: encoding[name].astype(np.int64)
            for name in self._input_names
            if name in encoding
        }
        logits = self._session.run([self._output_name], feeds)[0][0]
        shifted = logits - logits.max(axis=-1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / exp.sum(axis=-1, keepdims=True)
        label_ids = probs.argmax(axis=-1)

        spans: List[Dict[str, Any]] = []
        current: Optional[Dict[str, Any]] = None
        for index, (label_id, (start, end)) in enumerate(zip(label_ids, offsets)):
            if start == end:  # special / empty token (offset (0, 0))
                current = None
                continue
            label = self._id2label.get(int(label_id), "O")
            if label == "O":
                current = None
                continue
            entity = label.split("-", 1)[-1]  # strip the B-/I- prefix
            if current is not None and current["entity_group"] == entity:
                current["end"] = int(end)  # extend across sub-words / adjacent words
            else:
                current = {
                    "entity_group": entity,
                    "score": float(probs[index, label_id]),
                    "start": int(start),
                    "end": int(end),
                }
                spans.append(current)
        for span in spans:
            span["word"] = text[span["start"] : span["end"]]
        return spans


class BardsEuPiiOnnxRecognizer(BardsEuPiiRecognizer):
    """ONNX Runtime variant of :class:`BardsEuPiiRecognizer` for CPU inference.

    Loads the upstream quantized ONNX file into an ``onnxruntime.InferenceSession``
    and runs it through a small torch-free pipeline (tokenizer + numpy BIO
    aggregation), so all of the parent's label mapping, thresholding and
    ``labels_to_ignore`` behavior applies unchanged — but the image needs neither
    torch nor Optimum.

    Requires the ``bards-onnx`` extra (``pip install
    'presidio-analyzer[bards-onnx]'`` — ONNX Runtime + transformers for the
    tokenizer); torch is not required to construct or run this recognizer.

    :param onnx_model_subfolder: Subfolder in the model repo holding the ONNX
        file. Defaults to :data:`DEFAULT_ONNX_MODEL_SUBFOLDER` (``"onnx"``).
    :param onnx_model_file: ONNX file name. Defaults to
        :data:`DEFAULT_ONNX_MODEL_FILE` (``"model_quantized.onnx"``).
    :param onnx_provider: ONNX Runtime execution provider. Defaults to
        :data:`DEFAULT_ONNX_PROVIDER` (``"CPUExecutionProvider"``).
    :param onnx_intra_op_num_threads: ORT intra-op thread count. Defaults to the
        ``ORT_INTRA_OP_THREADS`` env var (if a positive integer), else ORT's
        default. An explicit value overrides the env var.
    :param onnx_inter_op_num_threads: ORT inter-op thread count. Defaults to the
        ``ORT_INTER_OP_THREADS`` env var (if a positive integer), else ORT's
        default. An explicit value overrides the env var.

    See :class:`BardsEuPiiRecognizer` for the inherited mapping / threshold /
    ``labels_to_ignore`` parameters. The graph optimization level is always
    ``ORT_ENABLE_ALL``.
    """

    # Inference runs on an ONNX Runtime session, so torch is not required to
    # construct this recognizer (the parent's torch import check is skipped).
    REQUIRES_TORCH = False

    def __init__(
        self,
        supported_entities: Optional[List[str]] = None,
        name: str = "BardsEuPiiOnnxRecognizer",
        supported_language: str = "en",
        model_name: str = DEFAULT_EU_PII_MODEL,
        label_mapping: Optional[Dict[str, str]] = None,
        threshold: float = 0.4,
        aggregation_strategy: str = "first",
        labels_to_ignore: Optional[List[str]] = None,
        thresholds_by_entity: Optional[Dict[str, float]] = None,
        thresholds_by_language: Optional[Dict[str, Dict[str, float]]] = None,
        mapping_profile: str = MAPPING_PROFILE_PRESIDIO_STANDARD,
        validate_mapping: Optional[bool] = None,
        onnx_model_subfolder: str = DEFAULT_ONNX_MODEL_SUBFOLDER,
        onnx_model_file: str = DEFAULT_ONNX_MODEL_FILE,
        onnx_provider: str = DEFAULT_ONNX_PROVIDER,
        onnx_intra_op_num_threads: Optional[int] = None,
        onnx_inter_op_num_threads: Optional[int] = None,
        device: str = "cpu",
        **kwargs,
    ):
        # ONNX / ORT configuration. Resolved before ``super().__init__`` because
        # that call triggers ``load()`` (via ``EntityRecognizer.__init__``), which
        # reads these attributes. Explicit kwargs win over env vars.
        self._onnx_model_subfolder = onnx_model_subfolder
        self._onnx_model_file = onnx_model_file
        self._onnx_provider = onnx_provider
        self._onnx_intra_op_num_threads = _resolve_threads(
            onnx_intra_op_num_threads, "ORT_INTRA_OP_THREADS"
        )
        self._onnx_inter_op_num_threads = _resolve_threads(
            onnx_inter_op_num_threads, "ORT_INTER_OP_THREADS"
        )

        super().__init__(
            supported_entities=supported_entities,
            name=name,
            supported_language=supported_language,
            model_name=model_name,
            label_mapping=label_mapping,
            threshold=threshold,
            aggregation_strategy=aggregation_strategy,
            labels_to_ignore=labels_to_ignore,
            thresholds_by_entity=thresholds_by_entity,
            thresholds_by_language=thresholds_by_language,
            mapping_profile=mapping_profile,
            validate_mapping=validate_mapping,
            device=device,
            **kwargs,
        )

    def _cache_key(self) -> Tuple[Any, ...]:
        """Return the ORT-session cache key (everything affecting the session)."""
        return (
            self.model_name,
            self.tokenizer_name or self.model_name,
            self._onnx_model_subfolder,
            self._onnx_model_file,
            self._onnx_provider,
            self._onnx_intra_op_num_threads,
            self._onnx_inter_op_num_threads,
        )

    def _resolve_model_dir(self) -> str:
        """Resolve the model repo to a local snapshot directory.

        Downloads only the config, tokenizer and the single quantized ONNX file
        into the Hugging Face cache and returns the snapshot directory. The load
        below reads files straight from this local path (not the repo id) and
        never calls the Hugging Face repo-tree API. That call has no offline cache
        fallback, so on a baked, offline image (``HF_HUB_OFFLINE=1``) it would
        otherwise abort the load; resolving a local dir makes the offline image
        load purely from the baked cache. Online (e.g. at image-build time) the
        files are fetched once and cached.
        """
        from huggingface_hub import snapshot_download

        return snapshot_download(
            self.model_name,
            allow_patterns=[
                "*.json",
                "*.txt",
                "*.model",
                "tokenizer*",
                "sentencepiece*",
                f"{self._onnx_model_subfolder}/{self._onnx_model_file}",
            ],
        )

    def _build_session_tokenizer_labels(
        self,
    ) -> Tuple[Any, Any, Dict[int, str]]:
        """Build the ORT session, tokenizer and id->label map (torch-free).

        Lazy-imports the ``bards-onnx`` extra (ONNX Runtime + transformers); no
        torch or Optimum is involved. The session is built straight from the local
        quantized ONNX file with ``ORT_ENABLE_ALL`` and the resolved thread
        counts; the tokenizer and ``id2label`` come from the snapshot root.
        """
        try:
            import onnxruntime as ort
            from transformers import AutoConfig, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "BardsEuPiiOnnxRecognizer requires ONNX Runtime and transformers. "
                "Install them with: "
                "pip install 'presidio-analyzer[bards-onnx]'. "
                f"Original error: {exc}"
            ) from exc

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        if self._onnx_intra_op_num_threads is not None:
            session_options.intra_op_num_threads = self._onnx_intra_op_num_threads
        if self._onnx_inter_op_num_threads is not None:
            session_options.inter_op_num_threads = self._onnx_inter_op_num_threads

        model_dir = self._resolve_model_dir()
        onnx_path = (
            os.path.join(model_dir, self._onnx_model_subfolder, self._onnx_model_file)
            if self._onnx_model_subfolder
            else os.path.join(model_dir, self._onnx_model_file)
        )
        session = ort.InferenceSession(
            onnx_path,
            sess_options=session_options,
            providers=[self._onnx_provider],
        )
        tokenizer_source = (
            self.tokenizer_name
            if self.tokenizer_name and self.tokenizer_name != self.model_name
            else model_dir
        )
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
        config = AutoConfig.from_pretrained(model_dir)
        id2label = {int(key): value for key, value in config.id2label.items()}
        return session, tokenizer, id2label

    def _load_or_get_cached_session(self) -> Tuple[Any, Any, Dict[int, str]]:
        """Return the cached ORT session + tokenizer + labels, building once."""
        key = self._cache_key()
        cached = _ORT_SESSION_CACHE.get(key)
        if cached is not None:
            return cached
        with _ORT_SESSION_CACHE_LOCK:
            cached = _ORT_SESSION_CACHE.get(key)
            if cached is None:
                cached = self._build_session_tokenizer_labels()
                _ORT_SESSION_CACHE[key] = cached
            return cached

    def load(self) -> None:
        """Load the ORT-backed pipeline, then install the ignored-label filter.

        Reuses a cached ORT session + tokenizer across instances with the same
        configuration and builds a fresh per-instance pipeline around it, so the
        per-recognizer ignored-label filter is never shared. Overrides the
        parent's torch pipeline loader entirely.
        """
        if self.ner_pipeline is not None:
            return
        if not self.model_name:
            raise ValueError(
                "model_name must be set before calling load(). "
                "Pass it to __init__() or set it directly."
            )
        session, tokenizer, id2label = self._load_or_get_cached_session()
        self.ner_pipeline = _OnnxTokenClassificationPipeline(
            session, tokenizer, id2label
        )
        if self.labels_to_ignore and not self._ignore_filter_installed:
            self._install_ignore_filter()

    def analyze(
        self,
        text: str,
        entities: List[str],
        nlp_artifacts: Optional[NlpArtifacts] = None,
    ) -> List[RecognizerResult]:
        """Analyze ``text``; skip ORT inference when no requested entity applies.

        Only adds an early return when ``entities`` is explicitly provided and
        has no overlap with :attr:`supported_entities`. For ``entities=None`` or
        an empty list, behavior is unchanged (delegated to the parent, which
        applies the per-entity / per-language thresholds).
        """
        if entities and not set(entities).intersection(self.supported_entities):
            return []
        return super().analyze(text, entities, nlp_artifacts)
