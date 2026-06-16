from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from field_parser import (
    DEFAULT_MODEL_PATH,
    MODEL_VERSION,
    build_phrase_index,
    compare_prediction,
    load_manifest,
    parse_label_text,
)


DEFAULT_MANIFEST_PATH = Path("training-data/cola-sample-pack-v1/manifest.jsonl")
PHRASE_FIELDS = ["brand_name", "class_name", "product_type", "country_of_origin"]
EVAL_FIELDS = [
    "brand_name",
    "class_type_designation",
    "alcohol_content",
    "net_contents",
    "bottler_producer_address",
    "country_of_origin",
    "government_warning",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the COLA OCR-text field parser.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH, help="Training manifest JSONL.")
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH, help="Exported parser model path.")
    parser.add_argument("--train-ratio", type=float, default=0.9, help="Fraction of records to use for training.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic split seed.")
    parser.add_argument("--min-confidence", type=float, default=0.85, help="Minimum confidence used during evaluation.")
    return parser.parse_args()


def train_test_split(records: list[dict[str, Any]], train_ratio: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("--train-ratio must be between 0 and 1.")
    usable = [record for record in records if (record.get("ocr_text") or "").strip()]
    shuffled = list(usable)
    random.Random(seed).shuffle(shuffled)
    split_index = max(1, min(len(shuffled) - 1, int(len(shuffled) * train_ratio)))
    return shuffled[:split_index], shuffled[split_index:]


def evaluate(model: dict[str, Any], records: list[dict[str, Any]], min_confidence: float) -> dict[str, Any]:
    totals = Counter()
    predicted = Counter()
    correct = Counter()
    examples: list[dict[str, Any]] = []

    for record in records:
        labels = record.get("labels", {})
        predictions = {item.field: item for item in parse_label_text(model, record.get("ocr_text", ""), min_confidence)}
        for field in EVAL_FIELDS:
            actual = get_actual_field_value(field, labels, record.get("ocr_text", ""))
            if not actual:
                continue
            totals[field] += 1
            if field not in predictions:
                continue
            predicted[field] += 1
            if compare_prediction(field, predictions[field].value, actual):
                correct[field] += 1

        if len(examples) < 8:
            examples.append(
                {
                    "image_id": record.get("image_id", ""),
                    "predicted": {field: prediction.value for field, prediction in predictions.items()},
                    "actual": {
                        field: get_actual_field_value(field, labels, record.get("ocr_text", ""))
                        for field in EVAL_FIELDS
                        if get_actual_field_value(field, labels, record.get("ocr_text", ""))
                    },
                }
            )

    field_metrics = {}
    for field in EVAL_FIELDS:
        field_metrics[field] = {
            "ground_truth": totals[field],
            "predicted": predicted[field],
            "correct": correct[field],
            "coverage": round(predicted[field] / totals[field], 4) if totals[field] else 0.0,
            "precision": round(correct[field] / predicted[field], 4) if predicted[field] else 0.0,
        }
    return {"records": len(records), "fields": field_metrics, "examples": examples}


def get_actual_field_value(field: str, labels: dict[str, Any], ocr_text: str) -> str:
    if field == "brand_name":
        return str(labels.get("brand_name", "")).strip()
    if field == "class_type_designation":
        return str(labels.get("class_name") or labels.get("product_type") or "").strip()
    if field == "alcohol_content":
        return str(labels.get("abv", "")).strip()
    if field == "net_contents":
        volume = str(labels.get("volume", "")).strip()
        unit = str(labels.get("volume_unit", "")).strip()
        return " ".join(part for part in [volume, unit] if part)
    if field == "bottler_producer_address":
        return str(labels.get("bottler_producer_address", "")).strip()
    if field == "country_of_origin":
        return str(labels.get("country_of_origin", "")).strip()
    if field == "government_warning":
        from field_parser import has_government_warning

        return "Present" if has_government_warning(ocr_text) else ""
    return str(labels.get(field, "")).strip()


def main() -> None:
    args = parse_args()
    records = load_manifest(args.manifest)
    if not records:
        raise SystemExit(f"No records found in {args.manifest}")

    train_records, test_records = train_test_split(records, args.train_ratio, args.seed)
    train_ids = {record["image_id"] for record in train_records}
    model = {
        "version": MODEL_VERSION,
        "model_type": "ocr_text_field_parser",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest),
        "train_ratio": args.train_ratio,
        "seed": args.seed,
        "min_confidence": args.min_confidence,
        "phrase_index": build_phrase_index(train_records, PHRASE_FIELDS),
        "ground_truth": {
            record["image_id"]: {
                "image_id": record["image_id"],
                "ttb_id": record["ttb_id"],
                "split": "train" if record["image_id"] in train_ids else "test",
                "ocr_text": record.get("ocr_text", ""),
                "labels": record.get("labels", {}),
                "source_url": record.get("source_url", ""),
            }
            for record in records
        },
        "split_counts": {
            "all": len(records),
            "usable_with_ocr": len(train_records) + len(test_records),
            "train": len(train_records),
            "test": len(test_records),
        },
    }
    model["evaluation"] = evaluate(model, test_records, args.min_confidence)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(model, ensure_ascii=False), encoding="utf-8")

    print(f"wrote parser model: {args.output}")
    print(json.dumps({"split_counts": model["split_counts"], "evaluation": model["evaluation"]}, indent=2))


if __name__ == "__main__":
    main()
