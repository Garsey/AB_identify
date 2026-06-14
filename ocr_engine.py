from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image


DEFAULT_DEMO_TEXT = "OLD TOM DISTILLERY 45% ALC/VOL"


class OCRLoadError(RuntimeError):
    pass


def _load_state_dict(predictor: Any, weights_path: Path) -> None:
    import torch

    checkpoint = torch.load(weights_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "detection" in checkpoint and "recognition" in checkpoint:
        predictor.det_predictor.model.load_state_dict(checkpoint["detection"], strict=False)
        predictor.reco_predictor.model.load_state_dict(checkpoint["recognition"], strict=False)
        return

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    try:
        predictor.model.load_state_dict(checkpoint, strict=False)
    except AttributeError as exc:
        raise OCRLoadError(
            "Unsupported weights format. Expected a docTR predictor checkpoint or a dict with "
            "'detection' and 'recognition' state dicts."
        ) from exc


def build_predictor(weights_path: str | Path | None = None, use_pretrained: bool = True) -> Any:
    try:
        from doctr.models import ocr_predictor
    except Exception as exc:  # pragma: no cover - depends on optional native packages
        raise OCRLoadError("docTR is not installed. Install requirements.txt or run with Docker.") from exc

    predictor = ocr_predictor(pretrained=use_pretrained)
    if weights_path:
        path = Path(weights_path)
        if path.exists():
            _load_state_dict(predictor, path)
    return predictor


def extract_text_from_image(predictor: Any, image: Image.Image) -> str:
    import numpy as np

    image_array = np.asarray(image.convert("RGB"))
    result = predictor([image_array])
    exported = result.export()
    words: list[str] = []

    for page in exported.get("pages", []):
        for block in page.get("blocks", []):
            for line in block.get("lines", []):
                for word in line.get("words", []):
                    value = word.get("value", "").strip()
                    if value:
                        words.append(value)

    return " ".join(words).strip()
