"""Entity-type alignment between recognizer output and evaluation buckets.

Predictions from ``BardsEuPiiRecognizer`` (and the deterministic recognizers used
in hybrid mode) carry Presidio entity names, while gold data is authored in a
small set of evaluation "buckets". This module maps the former onto the latter so
the two are comparable. Pure standard library (no Presidio import) to keep the
harness core offline.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional

#: Default mapping from recognizer / Presidio entity types to evaluation buckets.
#: Identity for the standard Presidio types; a few model-specific labels are
#: folded onto common benchmark buckets (e.g. account-ish labels -> ``USERNAME``,
#: matching the methodology used for the figures in the recognizer's docs).
DEFAULT_PREDICTION_LABEL_MAP: Mapping[str, str] = {
    "PERSON": "PERSON",
    "LOCATION": "LOCATION",
    "ORGANIZATION": "ORGANIZATION",
    "EMAIL_ADDRESS": "EMAIL_ADDRESS",
    "PHONE_NUMBER": "PHONE_NUMBER",
    "IP_ADDRESS": "IP_ADDRESS",
    "CREDIT_CARD": "CREDIT_CARD",
    "URL": "URL",
    "DATE_TIME": "DATE_TIME",
    "NRP": "NRP",
    "IBAN_CODE": "IBAN_CODE",
    "ACCOUNT_IDENTIFIER": "USERNAME",
    "CONTACT_HANDLE": "USERNAME",
}


def to_eval_bucket(
    entity_type: str, label_map: Optional[Dict[str, str]] = None
) -> str:
    """Map a recognizer entity type onto an evaluation bucket.

    Entity types not present in the map pass through unchanged, so nothing is
    silently dropped; callers that want a closed label set should filter the
    result against their requested entities.
    """
    mapping = DEFAULT_PREDICTION_LABEL_MAP if label_map is None else label_map
    return mapping.get(entity_type, entity_type)
