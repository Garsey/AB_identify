from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

from PIL import Image, ImageOps


class OCRLoadError(RuntimeError):
    pass


@dataclass(frozen=True)
class OCRResult:
    text: str
    backend: str
    device: str
    line_count: int


@dataclass(frozen=True)
class OCRReader:
    backend: str
    device: str
    reader: Any | None = None


def build_ocr_reader(preferred_backend: str = "auto", preferred_device: str = "auto") -> OCRReader:
    backend = normalize_choice(preferred_backend, {"auto", "easyocr", "tesseract"}, "auto")
    device = normalize_choice(preferred_device, {"auto", "cuda", "cpu"}, "auto")

    if backend in {"auto", "easyocr"}:
        try:
            return build_easyocr_reader(device)
        except Exception as exc:
            if backend == "easyocr":
                raise OCRLoadError(f"EasyOCR could not be initialized: {exc}") from exc

    if backend in {"auto", "tesseract"}:
        try:
            import pytesseract  # noqa: F401
        except ImportError as exc:
            raise OCRLoadError("No OCR backend is available. Install requirements-ui.txt and requirements-ocr.txt.") from exc
        return OCRReader(backend="tesseract", device="cpu")

    raise OCRLoadError(f"Unsupported OCR backend: {preferred_backend}")


def normalize_choice(value: str, allowed: set[str], default: str) -> str:
    normalized = (value or default).strip().lower()
    return normalized if normalized in allowed else default


def build_easyocr_reader(device: str) -> OCRReader:
    import easyocr

    use_gpu = should_use_cuda(device)
    reader = easyocr.Reader(["en"], gpu=use_gpu, verbose=False)
    return OCRReader(backend="easyocr", device="cuda" if use_gpu else "cpu", reader=reader)


def should_use_cuda(device: str) -> bool:
    if device == "cpu":
        return False
    if device == "cuda":
        return torch_cuda_available()
    return torch_cuda_available()


def torch_cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def extract_text_from_image(image: Image.Image, ocr_reader: OCRReader) -> OCRResult:
    if ocr_reader.backend == "easyocr":
        return extract_with_easyocr(image, ocr_reader)
    if ocr_reader.backend == "tesseract":
        return extract_with_tesseract(image)
    raise OCRLoadError(f"Unsupported OCR backend: {ocr_reader.backend}")


def extract_with_easyocr(image: Image.Image, ocr_reader: OCRReader) -> OCRResult:
    import numpy as np

    if ocr_reader.reader is None:
        raise OCRLoadError("EasyOCR reader was not initialized.")

    prepared = prepare_for_ocr(image).convert("RGB")
    lines: list[str] = []
    for variant in rotated_variants(prepared):
        results = ocr_reader.reader.readtext(
            np.asarray(variant),
            detail=1,
            paragraph=False,
            decoder="beamsearch",
        )
        lines.extend(item[1].strip() for item in results if len(item) >= 2 and item[1].strip())
    lines = dedupe_lines(lines)
    text = normalize_ocr_text("\n".join(lines))
    return OCRResult(text=text, backend="easyocr", device=ocr_reader.device, line_count=len(lines))


def extract_with_tesseract(image: Image.Image) -> OCRResult:
    try:
        import pytesseract
    except ImportError as exc:
        raise OCRLoadError("pytesseract is not installed. Rebuild Docker or install requirements-ui.txt.") from exc

    prepared = prepare_for_ocr(image)
    try:
        text = pytesseract.image_to_string(prepared, config="--psm 6")
    except pytesseract.TesseractNotFoundError as exc:
        raise OCRLoadError("Tesseract OCR is not installed in this environment. Rebuild the Docker image.") from exc
    text = normalize_ocr_text(text)
    return OCRResult(text=text, backend="tesseract", device="cpu", line_count=len(text.splitlines()))


def prepare_for_ocr(image: Image.Image) -> Image.Image:
    grayscale = ImageOps.grayscale(image)
    width, height = grayscale.size
    max_side = max(width, height)
    if max_side < 1600:
        scale = 1600 / max_side
        grayscale = grayscale.resize((int(width * scale), int(height * scale)))
    return ImageOps.autocontrast(grayscale)


def rotated_variants(image: Image.Image) -> list[Image.Image]:
    return [
        image,
        image.rotate(90, expand=True),
        image.rotate(-90, expand=True),
    ]


def dedupe_lines(lines: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for line in lines:
        compact = re.sub(r"\W+", "", line).lower()
        if not compact or compact in seen:
            continue
        seen.add(compact)
        deduped.append(line)
    return deduped


def normalize_ocr_text(text: str) -> str:
    cleaned = re.sub(r"\n{3,}", "\n\n", text or "")
    return re.sub(r"[ \t]+", " ", cleaned).strip()


def env_backend() -> str:
    return os.getenv("OCR_BACKEND", "auto")


def env_device() -> str:
    return os.getenv("OCR_DEVICE", "auto")
