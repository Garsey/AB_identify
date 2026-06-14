from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune/export placeholder for alcohol label OCR weights.")
    parser.add_argument("--dataset", type=Path, default=Path("data/alcohol-labels"), help="Dataset root in COCO/YOLO/docTR format.")
    parser.add_argument("--output", type=Path, default=Path("weights/alcohol_ocr.pt"), help="Output weights path.")
    parser.add_argument("--epochs", type=int, default=5, help="Fine-tuning epochs.")
    parser.add_argument(
        "--write-manifest-only",
        action="store_true",
        help="Create an output manifest without starting heavy model training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if not args.dataset.exists():
        raise SystemExit(
            f"Dataset folder not found: {args.dataset}. Add label data first, or pass --dataset to your local dataset path."
        )

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset),
        "epochs": args.epochs,
        "target_weights": str(args.output),
        "status": "manifest_only" if args.write_manifest_only else "training_not_implemented",
        "next_step": "Wire this script to docTR detection/recognition trainers, then export state dicts to target_weights.",
    }

    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if args.write_manifest_only:
        print(f"Wrote training manifest to {manifest_path}")
        return

    raise SystemExit(
        "Training scaffold is ready, but project-specific dataset parsing/trainer wiring is still required. "
        f"Manifest written to {manifest_path}."
    )


if __name__ == "__main__":
    main()
