from __future__ import annotations

import re

from PIL import Image, ImageOps


class OCRLoadError(RuntimeError):
    pass


def extract_text_from_image(image: Image.Image) -> str:
    try:
        import pytesseract
    except ImportError as exc:
        raise OCRLoadError("pytesseract is not installed. Rebuild Docker or install requirements-ui.txt.") from exc

    prepared = prepare_for_ocr(image)
    try:
        text = pytesseract.image_to_string(prepared, config="--psm 6")
    except pytesseract.TesseractNotFoundError as exc:
        raise OCRLoadError("Tesseract OCR is not installed in this environment. Rebuild the Docker image.") from exc
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def prepare_for_ocr(image: Image.Image) -> Image.Image:
    grayscale = ImageOps.grayscale(image)
    width, height = grayscale.size
    max_side = max(width, height)
    if max_side < 1400:
        scale = 1400 / max_side
        grayscale = grayscale.resize((int(width * scale), int(height * scale)))
    return ImageOps.autocontrast(grayscale)
