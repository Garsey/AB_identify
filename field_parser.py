from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


MODEL_VERSION = 1
DEFAULT_MODEL_PATH = Path("weights/cola_field_parser.json")
FIELD_ORDER = [
    "brand_name",
    "class_type_designation",
    "alcohol_content",
    "net_contents",
    "bottler_producer_address",
    "country_of_origin",
    "government_warning",
]

FIELD_LABELS = {
    "brand_name": "Brand name",
    "class_type_designation": "Class/type designation",
    "alcohol_content": "Alcohol content",
    "net_contents": "Net contents",
    "bottler_producer_address": "Name and address of bottler/producer",
    "country_of_origin": "Country of origin for imports",
    "government_warning": "Government Health Warning Statement",
}


@dataclass(frozen=True)
class FieldPrediction:
    field: str
    value: str
    confidence: float
    source: str


def load_manifest(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_model(path: Path = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        model = json.load(handle)
    if model.get("version") != MODEL_VERSION:
        raise ValueError(f"Unsupported parser model version: {model.get('version')}")
    return model


def normalize_text(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9.%/ ]+", " ", value or "")
    return re.sub(r"\s+", " ", cleaned).strip().lower()


def image_id_from_filename(filename: str) -> str | None:
    stem = Path(filename).stem
    match = re.search(r"(\d{14}_\d+)", stem)
    return match.group(1) if match else None


def fuzzy_score(expected: str, observed: str) -> float:
    expected_norm = normalize_text(expected)
    observed_norm = normalize_text(observed)
    if not expected_norm or not observed_norm:
        return 0.0
    if expected_norm in observed_norm:
        return 1.0
    return SequenceMatcher(None, expected_norm, observed_norm).ratio()


def text_to_rows(text: str, chunk_size: int = 220) -> list[dict[str, str | int]]:
    compact = re.sub(r"\s+", " ", text or "").strip()
    if not compact:
        return [{"line": 1, "text": ""}]
    return [
        {"line": index + 1, "text": compact[index * chunk_size : (index + 1) * chunk_size]}
        for index in range((len(compact) + chunk_size - 1) // chunk_size)
    ]


def labels_to_rows(labels: dict[str, str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    legacy_map = {
        "brand_name": labels.get("brand_name", ""),
        "class_type_designation": labels.get("class_name") or labels.get("product_type", ""),
        "alcohol_content": labels.get("abv", ""),
        "net_contents": " ".join(part for part in [labels.get("volume", ""), labels.get("volume_unit", "")] if part),
        "bottler_producer_address": labels.get("bottler_producer_address", ""),
        "country_of_origin": labels.get("country_of_origin", ""),
        "government_warning": labels.get("government_warning", ""),
    }
    for key in FIELD_ORDER:
        value = labels.get(key, "")
        if not value:
            value = legacy_map.get(key, "")
        if value:
            rows.append({"field": display_field(key), "value": str(value)})
    return rows


def predictions_to_rows(predictions: list[FieldPrediction]) -> list[dict[str, str]]:
    return [
        {
            "field": display_field(prediction.field),
            "value": prediction.value,
            "confidence": f"{prediction.confidence:.0%}",
            "source": prediction.source,
        }
        for prediction in predictions
    ]


def display_field(field: str) -> str:
    return FIELD_LABELS.get(field, field.replace("_", " ").title().replace("Abv", "ABV"))


def build_phrase_index(records: list[dict[str, Any]], fields: list[str]) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, dict[str, dict[str, Any]]] = {field: {} for field in fields}
    for record in records:
        labels = record.get("labels", {})
        for field in fields:
            value = str(labels.get(field, "")).strip()
            value_norm = normalize_text(value)
            if is_low_specificity_phrase(field, value_norm):
                continue
            bucket = index[field].setdefault(value_norm, {"value": value, "value_norm": value_norm, "count": 0})
            bucket["count"] += 1

    return {
        field: sorted(values.values(), key=lambda item: (-item["count"], item["value_norm"]))
        for field, values in index.items()
    }


def is_low_specificity_phrase(field: str, value_norm: str) -> bool:
    generic_product_words = {
        "ale",
        "beer",
        "bourbon",
        "cabernet",
        "chardonnay",
        "gin",
        "honey",
        "lager",
        "malbec",
        "mead",
        "merlot",
        "peach",
        "pinot",
        "riesling",
        "rose",
        "rum",
        "sauvignon",
        "tequila",
        "vodka",
        "whiskey",
        "whisky",
        "wine",
    }
    if len(value_norm) < 8 and field in {"brand_name", "product_name"}:
        return True
    if field == "product_name" and value_norm in generic_product_words:
        return True
    return len(value_norm) < 4


def lookup_ground_truth(model: dict[str, Any], image_id: str | None) -> dict[str, Any] | None:
    if not image_id:
        return None
    return model.get("ground_truth", {}).get(image_id)


def parse_label_text(model: dict[str, Any], text: str, min_confidence: float = 0.75) -> list[FieldPrediction]:
    predictions: list[FieldPrediction] = []
    text_norm = normalize_text(text)

    for field in ["brand_name"]:
        match = best_phrase_match(field, model.get("phrase_index", {}).get(field, []), text_norm)
        if match and match.confidence >= min_confidence:
            predictions.append(match_for_field(field, match))

    class_type = parse_class_type_designation(model, text_norm)
    if class_type and class_type.confidence >= min_confidence and not has_prediction(predictions, "class_type_designation"):
        predictions.append(class_type)

    abv = parse_abv(text)
    if abv and abv.confidence >= min_confidence:
        predictions.append(abv)

    volume = parse_volume(text)
    if volume and volume.confidence >= min_confidence:
        predictions.append(volume)

    bottler = parse_bottler_producer_address(text)
    if bottler and bottler.confidence >= min_confidence:
        predictions.append(bottler)

    origin = parse_country_of_origin(text, model)
    if origin and origin.confidence >= min_confidence:
        predictions.append(origin)

    warning = parse_government_warning(text)
    if warning and warning.confidence >= min_confidence:
        predictions.append(warning)

    return sorted(predictions, key=lambda item: FIELD_ORDER.index(item.field) if item.field in FIELD_ORDER else 999)


def has_prediction(predictions: list[FieldPrediction], field: str) -> bool:
    return any(prediction.field == field for prediction in predictions)


@dataclass(frozen=True)
class PhraseMatch:
    value: str
    confidence: float
    source: str


def match_for_field(field: str, match: PhraseMatch) -> FieldPrediction:
    return FieldPrediction(field=field, value=match.value, confidence=match.confidence, source=match.source)


def best_phrase_match(field: str, entries: list[dict[str, Any]], text_norm: str) -> PhraseMatch | None:
    best: PhraseMatch | None = None
    for entry in entries:
        value_norm = entry["value_norm"]
        if value_norm in text_norm:
            position_ratio = text_norm.find(value_norm) / max(1, len(text_norm))
            if field == "brand_name":
                confidence = 0.98 if len(value_norm) >= 12 else 0.9
            elif position_ratio <= 0.45:
                confidence = 0.94 if len(value_norm) >= 12 else 0.88
            else:
                confidence = 0.78
            candidate = PhraseMatch(entry["value"], confidence, "trained phrase exact match")
        else:
            continue

        if best is None or candidate.confidence > best.confidence:
            best = candidate
    return best


def parse_product_type(text_norm: str) -> FieldPrediction | None:
    keyword_groups = [
        ("distilled spirits", ["whiskey", "whisky", "bourbon", "vodka", "gin", "rum", "tequila", "liqueur", "brandy", "spirits"]),
        ("malt beverage", ["beer", "ale", "lager", "stout", "porter", "ipa", "malt beverage", "hard cider"]),
        ("wine", ["wine", "pinot", "cabernet", "chardonnay", "sauvignon", "merlot", "riesling", "moscato", "rose"]),
    ]
    for value, keywords in keyword_groups:
        hits = sum(1 for keyword in keywords if re.search(rf"\b{re.escape(keyword)}\b", text_norm))
        if hits:
            confidence = min(0.95, 0.72 + hits * 0.08)
            return FieldPrediction("class_type_designation", value, confidence, "alcohol category keyword")
    return None


def parse_class_type_designation(model: dict[str, Any], text_norm: str) -> FieldPrediction | None:
    for field in ["class_name", "product_type"]:
        match = best_phrase_match(field, model.get("phrase_index", {}).get(field, []), text_norm)
        if match:
            return FieldPrediction("class_type_designation", match.value, match.confidence, match.source)
    return parse_product_type(text_norm)


def parse_abv(text: str) -> FieldPrediction | None:
    compact = re.sub(r"\s+", " ", text or "")
    patterns = [
        r"(?P<value>\d{1,2}(?:\.\d+)?)\s*%\s*(?:alc|alcohol|by volume|vol)?",
        r"(?:alc|alcohol)\.?\s*(?:/|by)?\s*(?:vol|volume)?\.?\s*(?P<value>\d{1,2}(?:\.\d+)?)\s*%",
        r"(?P<value>\d{1,2}(?:\.\d+)?)\s*(?:percent|per cent)\s*(?:alc|alcohol|by volume|vol)",
    ]
    candidates: list[tuple[float, str]] = []
    for pattern in patterns:
        for match in re.finditer(pattern, compact, flags=re.IGNORECASE):
            value = float(match.group("value"))
            if 0.1 <= value <= 80:
                context = compact[max(0, match.start() - 20) : match.end() + 30].lower()
                confidence = 0.96 if any(token in context for token in ["alc", "alcohol", "volume", "vol"]) else 0.86
                candidates.append((confidence, f"{value:g}%"))
    if not candidates:
        return None
    confidence, value = max(candidates, key=lambda item: item[0])
    return FieldPrediction("alcohol_content", value, confidence, "ABV regex")


def parse_volume(text: str) -> FieldPrediction | None:
    compact = re.sub(r"\s+", " ", text or "")
    unit_pattern = r"(?P<unit>ml|mL|milliliters?|liters?|litres?|l|cl|fl\.?\s*oz|oz)"
    pattern = rf"(?P<value>\d{{1,4}}(?:\.\d+)?)\s*{unit_pattern}\b"
    candidates: list[tuple[float, str]] = []
    for match in re.finditer(pattern, compact, flags=re.IGNORECASE):
        raw_value = float(match.group("value"))
        unit = match.group("unit").lower().replace(".", "").replace(" ", "")
        value = raw_value
        display_unit = unit
        if unit in {"l", "liter", "liters", "litre", "litres"}:
            value = raw_value * 1000
            display_unit = "ml"
        elif unit == "cl":
            value = raw_value * 10
            display_unit = "ml"
        elif unit in {"milliliter", "milliliters"}:
            display_unit = "ml"
        if 25 <= value <= 5000:
            confidence = 0.93 if display_unit == "ml" else 0.86
            candidates.append((confidence, f"{value:g} {display_unit}"))
    if not candidates:
        return None
    confidence, value = max(candidates, key=lambda item: item[0])
    return FieldPrediction("net_contents", value, confidence, "volume regex")


def parse_barcode(text: str) -> FieldPrediction | None:
    compact = re.sub(r"\D+", " ", text or "")
    for token in compact.split():
        if len(token) in {8, 12, 13, 14}:
            return FieldPrediction("barcode_value", token, 0.8, "barcode length regex")
    return None


def parse_bottler_producer_address(text: str) -> FieldPrediction | None:
    compact = re.sub(r"\s+", " ", text or "").strip()
    if not compact:
        return None
    patterns = [
        r"((?:produced|bottled|vinted|cellared|distilled|brewed|imported|selected)\s+(?:and\s+)?(?:bottled|produced|vinted|cellared|distilled|brewed|imported)?\s*(?:by|for)\s*[:\-]?\s*.{12,150})",
        r"((?:estate\s+bottled|grown\s+and\s+bottled)\s+.{12,150})",
    ]
    for pattern in patterns:
        match = re.search(pattern, compact, flags=re.IGNORECASE)
        if match:
            value = trim_at_compliance_boundary(match.group(1))
            return FieldPrediction("bottler_producer_address", value, 0.88, "producer/address phrase")
    return None


def trim_at_compliance_boundary(value: str) -> str:
    boundaries = [
        " government warning",
        " contains sulfites",
        " alc ",
        " alcohol ",
        " www.",
        " http",
    ]
    lowered = value.lower()
    end = len(value)
    for boundary in boundaries:
        index = lowered.find(boundary)
        if index > 20:
            end = min(end, index)
    return value[:end].strip(" .,;:-")


def parse_country_of_origin(text: str, model: dict[str, Any] | None = None) -> FieldPrediction | None:
    text_norm = normalize_text(text)
    if model:
        match = best_phrase_match("country_of_origin", model.get("phrase_index", {}).get("country_of_origin", []), text_norm)
        if match and match.confidence >= 0.82:
            return FieldPrediction("country_of_origin", match.value.title(), min(0.95, match.confidence), "trained country phrase")

    compact = re.sub(r"\s+", " ", text or "").strip()
    country = find_country_name(text_norm)
    if country:
        return FieldPrediction("country_of_origin", country, 0.88, "country name match")

    patterns = [
        r"(?:product|produce)\s+of\s+([A-Z][A-Za-z .'-]{3,40})",
        r"imported\s+from\s+([A-Z][A-Za-z .'-]{3,40})",
        r"imported\s+by\s+.{0,80}?\bfrom\s+([A-Z][A-Za-z .'-]{3,40})",
        r"([A-Z][A-Za-z .'-]{3,40})\s+(?:wine|whisky|whiskey|vodka|gin|rum|tequila|liqueur)",
    ]
    for pattern in patterns:
        match = re.search(pattern, compact, flags=re.IGNORECASE)
        if match:
            value = trim_at_compliance_boundary(match.group(1))
            if 3 <= len(value) <= 45:
                return FieldPrediction("country_of_origin", value.title(), 0.84, "origin phrase")
    return None


def parse_government_warning(text: str) -> FieldPrediction | None:
    if has_government_warning(text):
        return FieldPrediction("government_warning", "Present", 0.99, "government warning presence check")
    return None


def has_government_warning(text: str) -> bool:
    tokens = re.findall(r"[a-zA-Z]+", text or "")
    normalized = [token.lower().replace("0", "o").replace("1", "l") for token in tokens]
    compact = " ".join(normalized)
    if re.search(r"\bgovernment\s+war(?:n|r)?ing\b", compact, flags=re.IGNORECASE):
        return True
    for index, token in enumerate(normalized):
        if fuzzy_score("government", token) < 0.82:
            continue
        window = normalized[index + 1 : index + 7]
        if any(fuzzy_score("warning", candidate) >= 0.78 or fuzzy_score("waring", candidate) >= 0.82 for candidate in window):
            return True
    return False


COUNTRY_NAMES = {
    "argentina",
    "australia",
    "austria",
    "belgium",
    "brazil",
    "canada",
    "chile",
    "china",
    "france",
    "germany",
    "greece",
    "ireland",
    "israel",
    "italy",
    "japan",
    "mexico",
    "netherlands",
    "new zealand",
    "portugal",
    "south africa",
    "spain",
    "switzerland",
    "united kingdom",
}


def find_country_name(text_norm: str) -> str | None:
    for country in sorted(COUNTRY_NAMES, key=len, reverse=True):
        if re.search(rf"\b{re.escape(country)}\b", text_norm):
            return country.title()
    return None


def compare_prediction(field: str, predicted: str, actual: str) -> bool:
    if not predicted or not actual:
        return False
    if field in {"alcohol_content", "net_contents", "abv", "volume"}:
        pred_num = first_number(predicted)
        actual_num = first_number(actual)
        return pred_num is not None and actual_num is not None and math.isclose(pred_num, actual_num, rel_tol=0.02, abs_tol=0.05)
    return fuzzy_score(predicted, actual) >= 0.88


def first_number(value: str) -> float | None:
    match = re.search(r"\d+(?:\.\d+)?", value or "")
    return float(match.group(0)) if match else None
