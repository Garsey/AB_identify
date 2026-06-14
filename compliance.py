from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

try:
    from rapidfuzz import fuzz
except ImportError:  # Keeps the UI usable before optional speedups are installed.
    fuzz = None


@dataclass(frozen=True)
class ComplianceResult:
    label: str
    expected: str
    observed: str
    score: float
    passed: bool


def normalize_text(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9.%/ ]+", " ", value)
    return re.sub(r"\s+", " ", cleaned).strip().lower()


def compare_field(label: str, expected: str, observed_text: str, threshold: float = 85.0) -> ComplianceResult:
    expected_norm = normalize_text(expected)
    observed_norm = normalize_text(observed_text)
    score = fuzzy_score(expected_norm, observed_norm) if expected_norm else 0.0
    return ComplianceResult(
        label=label,
        expected=expected,
        observed=observed_text,
        score=score,
        passed=score >= threshold,
    )


def compare_abv(expected_abv: str, observed_text: str) -> ComplianceResult:
    expected_norm = normalize_text(expected_abv)
    observed_norm = normalize_text(observed_text)
    score = 100.0 if expected_norm and expected_norm in observed_norm else fuzzy_score(expected_norm, observed_norm)
    return ComplianceResult(
        label="ABV",
        expected=expected_abv,
        observed=observed_text,
        score=score,
        passed=score >= 90.0,
    )


def fuzzy_score(expected: str, observed: str) -> float:
    if not expected or not observed:
        return 0.0
    if fuzz is not None:
        return float(fuzz.partial_ratio(expected, observed))
    return SequenceMatcher(None, expected, observed).ratio() * 100.0
