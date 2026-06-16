from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


CLOUDFRONT_BASE_URL = "https://dyuie4zgfxmt6.cloudfront.net"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download COLA sample-pack images and write a training manifest."
    )
    parser.add_argument(
        "--pack",
        type=Path,
        default=Path("archive/cola-sample-pack-v1"),
        help="Folder containing cola.csv and cola_image.csv.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("training-data/cola-sample-pack-v1"),
        help="Output folder for images and manifest.jsonl.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional maximum images to process.")
    parser.add_argument("--sleep", type=float, default=0.05, help="Pause between downloads.")
    parser.add_argument("--overwrite", action="store_true", help="Re-download existing images.")
    return parser.parse_args()


def read_by_id(path: Path, id_field: str) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="", errors="replace") as handle:
        return {row[id_field]: row for row in csv.DictReader(handle)}


def download_image(url: str, output_path: Path, overwrite: bool) -> str:
    if output_path.exists() and not overwrite:
        return "exists"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "AB-identify-training/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        content_type = response.headers.get("Content-Type", "")
        data = response.read()

    if not data.startswith(b"RIFF") or data[8:12] != b"WEBP":
        raise ValueError(f"Unexpected image response for {url}: {content_type}")

    output_path.write_bytes(data)
    return "downloaded"


def main() -> None:
    args = parse_args()
    cola_path = args.pack / "cola.csv"
    image_csv_path = args.pack / "cola_image.csv"
    image_dir = args.out / "images"
    manifest_path = args.out / "manifest.jsonl"

    if not cola_path.exists() or not image_csv_path.exists():
        raise SystemExit(f"Expected cola.csv and cola_image.csv under {args.pack}")

    cola_by_id = read_by_id(cola_path, "TTB_ID")
    args.out.mkdir(parents=True, exist_ok=True)

    processed = 0
    downloaded = 0
    skipped = 0
    failed = 0

    with image_csv_path.open("r", encoding="utf-8-sig", newline="", errors="replace") as image_handle:
        image_rows = csv.DictReader(image_handle)
        with manifest_path.open("w", encoding="utf-8") as manifest:
            for row in image_rows:
                if args.limit and processed >= args.limit:
                    break

                image_id = row["TTB_IMAGE_ID"]
                cola = cola_by_id.get(row["TTB_ID"], {})
                image_path = image_dir / f"{image_id}.webp"
                url = f"{CLOUDFRONT_BASE_URL}/{image_id}.webp"

                try:
                    status = download_image(url, image_path, overwrite=args.overwrite)
                    if status == "downloaded":
                        downloaded += 1
                    else:
                        skipped += 1
                except (OSError, ValueError, urllib.error.URLError) as exc:
                    failed += 1
                    print(f"failed {image_id}: {exc}", flush=True)
                    continue

                record = {
                    "image_id": image_id,
                    "ttb_id": row["TTB_ID"],
                    "image_path": str(image_path.as_posix()),
                    "source_url": url,
                    "container_position": row.get("CONTAINER_POSITION", ""),
                    "width_pixels": row.get("WIDTH_PIXELS", ""),
                    "height_pixels": row.get("HEIGHT_PIXELS", ""),
                    "ocr_text": row.get("OCR_TEXT", ""),
                    "labels": {
                        "brand_name": cola.get("BRAND_NAME", ""),
                        "product_name": cola.get("PRODUCT_NAME", ""),
                        "product_type": cola.get("PRODUCT_TYPE", ""),
                        "class_name": cola.get("CLASS_NAME", ""),
                        "abv": cola.get("OCR_ABV", ""),
                        "volume": cola.get("OCR_VOLUME", ""),
                        "volume_unit": cola.get("OCR_VOLUME_UNIT", ""),
                        "barcode_type": cola.get("BARCODE_TYPE", ""),
                        "barcode_value": cola.get("BARCODE_VALUE", ""),
                    },
                }
                manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
                processed += 1

                if args.sleep:
                    time.sleep(args.sleep)

    print(
        f"processed={processed} downloaded={downloaded} existing={skipped} "
        f"failed={failed} manifest={manifest_path}"
    )


if __name__ == "__main__":
    main()
