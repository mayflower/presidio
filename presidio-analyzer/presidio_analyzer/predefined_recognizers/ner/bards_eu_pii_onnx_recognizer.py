"""ONNX Runtime (INT8) variant of :class:`BardsEuPiiRecognizer` for CPU.

Runs the ``bardsai/eu-pii-anonimization-multilang`` model on CPU through the
upstream quantized ONNX file (``onnx/model_quantized.onnx``) via Optimum ONNX
Runtime, instead of the PyTorch checkpoint. It is a thin subclass of
:class:`BardsEuPiiRecognizer`: the default model id, entity mapping, mapping
profiles, thresholds, ``thresholds_by_entity`` / ``thresholds_by_language``,
``labels_to_ignore``, the :meth:`hybrid` constructor and the aggregation strategy
are all inherited. Only model loading differs.

The default PyTorch :class:`BardsEuPiiRecognizer` is unchanged; this recognizer
is opt-in (registered explicitly in code or in a YAML registry config). All
ONNX-specific dependencies (``optimum``, ``onnxruntime``) are imported lazily,
only in the ONNX load path, so importing this module — and the normal unit
tests — do not require them, and the recognizer can be constructed without torch.
Install them with ``pip install 'presidio-analyzer[bards-onnx]'``.
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

# Module-level cache of loaded ORT models, keyed by everything that affects the
# loaded model/session (see ``_cache_key``). Multiple per-language recognizer
# instances that share a key reuse one heavy ORT model + tokenizer; each instance
# still builds its own lightweight transformers pipeline (and ignored-label
# filter) around it, so no per-recognizer mutable state (labels_to_ignore,
# thresholds) is shared through the cache.
_ORT_MODEL_CACHE: Dict[Tuple[Any, ...], Tuple[Any, Any]] = {}
_ORT_MODEL_CACHE_LOCK = threading.Lock()


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


class BardsEuPiiOnnxRecognizer(BardsEuPiiRecognizer):
    """ONNX Runtime variant of :class:`BardsEuPiiRecognizer` for CPU inference.

    Loads the upstream quantized ONNX file with Optimum's
    ``ORTModelForTokenClassification`` and an ``AutoTokenizer``, then runs it
    through a standard transformers ``token-classification`` pipeline — so all of
    the parent's BIO aggregation, chunking, label mapping, thresholding and
    ``labels_to_ignore`` behavior applies unchanged.

    Requires the ``bards-onnx`` extra (``pip install
    'presidio-analyzer[bards-onnx]'`` — Optimum + ONNX Runtime); torch is not
    required to construct this recognizer.

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
        """Return the ORT-model cache key (everything affecting the session)."""
        return (
            self.model_name,
            self.tokenizer_name or self.model_name,
            self._onnx_model_subfolder,
            self._onnx_model_file,
            self._onnx_provider,
            self._onnx_intra_op_num_threads,
            self._onnx_inter_op_num_threads,
        )

    def _build_ort_model_and_tokenizer(self) -> Tuple[Any, Any]:
        """Build the ORT model and tokenizer (lazy-imports the bards-onnx extra)."""
        try:
            import onnxruntime as ort
            from optimum.onnxruntime import ORTModelForTokenClassification
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "BardsEuPiiOnnxRecognizer requires Optimum and ONNX Runtime. "
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

        model = ORTModelForTokenClassification.from_pretrained(
            self.model_name,
            subfolder=self._onnx_model_subfolder,
            file_name=self._onnx_model_file,
            provider=self._onnx_provider,
            session_options=session_options,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name or self.model_name
        )
        return model, tokenizer

    def _load_or_get_cached_ort(self) -> Tuple[Any, Any]:
        """Return the cached ORT model + tokenizer for this config, building once."""
        key = self._cache_key()
        cached = _ORT_MODEL_CACHE.get(key)
        if cached is not None:
            return cached
        with _ORT_MODEL_CACHE_LOCK:
            cached = _ORT_MODEL_CACHE.get(key)
            if cached is None:
                cached = self._build_ort_model_and_tokenizer()
                _ORT_MODEL_CACHE[key] = cached
            return cached

    def _build_token_pipeline(self, model: Any, tokenizer: Any) -> Any:
        """Build a transformers token-classification pipeline over the ORT model."""
        from transformers import pipeline as hf_pipeline

        return hf_pipeline(
            self.DEFAULT_HF_TASK,
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy=self.aggregation_strategy,
        )

    def load(self) -> None:
        """Load the ORT-backed pipeline, then install the ignored-label filter.

        Reuses a cached ORT model + tokenizer across instances with the same
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
        model, tokenizer = self._load_or_get_cached_ort()
        self.ner_pipeline = self._build_token_pipeline(model, tokenizer)
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
