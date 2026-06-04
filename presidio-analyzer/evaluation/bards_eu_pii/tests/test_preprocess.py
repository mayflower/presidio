"""Offline tests for the experimental eval-only preprocessing helpers."""
from preprocess import (
    map_span,
    normalize_ocr_noise,
    normalize_text,
)


# --------------------------------------------------------------------------- #
# OCR / leet normalization (length-preserving)
# --------------------------------------------------------------------------- #
def test_ocr_substitutes_digits_flanked_by_letters():
    assert normalize_ocr_noise("J0hn Sm1th") == "John Smith"


def test_ocr_is_length_preserving():
    noisy = "J0hn Sm1th"
    assert len(normalize_ocr_noise(noisy)) == len(noisy)


def test_ocr_leaves_pure_numeric_tokens_untouched():
    # Phone numbers / account numbers must not be corrupted.
    assert normalize_ocr_noise("+49 30 1234 5678") == "+49 30 1234 5678"
    assert normalize_ocr_noise("ACC-100245") == "ACC-100245"


def test_ocr_does_not_touch_digit_at_token_edge():
    # Digit not flanked by letters on BOTH sides is left alone.
    assert normalize_ocr_noise("user42") == "user42"
    assert normalize_ocr_noise("0scar") == "0scar"
    assert normalize_ocr_noise("abc123") == "abc123"


def test_ocr_noop_on_clean_text():
    assert normalize_ocr_noise("John Smith") == "John Smith"
    assert normalize_ocr_noise("") == ""


# --------------------------------------------------------------------------- #
# spaced-email normalization (length-changing) + offset mapping
# --------------------------------------------------------------------------- #
def test_spaced_email_collapses_whitespace():
    text = "Email john . smith @ example . com please."
    out, span_map = normalize_text(text, spaced_email=True)
    assert "john.smith@example.com" in out
    assert len(span_map) == len(out)


def test_spaced_email_offset_maps_back_to_original():
    text = "Email john . smith @ example . com please."
    out, span_map = normalize_text(text, spaced_email=True)
    start = out.index("john.smith@example.com")
    end = start + len("john.smith@example.com")
    orig_start, orig_end = map_span(start, end, span_map)
    # Mapped span covers the full spaced e-mail in the original text.
    assert text[orig_start:orig_end] == "john . smith @ example . com"


def test_spaced_email_does_not_collapse_normal_punctuation():
    text = "The meeting ended . Then we left ."
    out, _ = normalize_text(text, spaced_email=True)
    assert out == text  # no email pattern -> unchanged


def test_normal_email_is_unchanged_but_valid():
    text = "Email john.smith@example.com now."
    out, span_map = normalize_text(text, spaced_email=True)
    assert out == text
    start = out.index("john.smith@example.com")
    end = start + len("john.smith@example.com")
    assert map_span(start, end, span_map) == (start, end)


# --------------------------------------------------------------------------- #
# combined + map_span semantics
# --------------------------------------------------------------------------- #
def test_no_normalization_returns_identity_map():
    text = "John Smith"
    out, span_map = normalize_text(text)
    assert out == text
    assert span_map == [(i, i + 1) for i in range(len(text))]


def test_ocr_only_offsets_unchanged():
    text = "call J0hn Sm1th"
    out, span_map = normalize_text(text, ocr=True)
    assert out == "call John Smith"
    # Length-preserving: each output char maps to its own original index.
    assert map_span(5, 15, span_map) == (5, 15)


def test_map_span_none_returns_unchanged():
    assert map_span(3, 7, None) == (3, 7)


def test_map_span_handles_degenerate_span():
    span_map = [(0, 1), (1, 2)]
    assert map_span(5, 5, span_map) == (5, 5)  # start >= end -> unchanged
