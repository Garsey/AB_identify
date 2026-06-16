# AB Identify

Lightweight local prototype for reading alcohol label images and filling a confidence-filtered parsed table.

The project uses the standard training/deployment split:

1. Materialize labeled COLA sample data locally.
2. Train/export a parser artifact from OCR text and CSV ground truth.
3. Build Docker as an inference-only web app that reads new uploads with OCR and fills only confident table values.

## Local Setup

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-ui.txt
python -m streamlit run app.py
```

Then open http://localhost:8501.

## Docker Setup

```powershell
docker compose up --build
```

Then open http://localhost:8501.

The Docker image is inference-only. It installs Tesseract OCR plus the lightweight Python UI/parser dependencies, includes `weights/cola_field_parser.json`, and does not mount the full repo or training data at runtime.

## Project Layout

- `app.py` - Streamlit upload UI that OCRs a new image and fills the parsed table.
- `ocr_engine.py` - Tesseract-based OCR for uploaded label images.
- `field_parser.py` - trained OCR-text field parser and confidence filtering.
- `train.py` - trains the parser from the local manifest using a 90/10 split.
- `materialize_cola_sample.py` - downloads sample-pack image files and creates a training manifest.
- `weights/cola_field_parser.json` - exported parser artifact used by Docker.

## Training Flow

Download/materialize the sample-pack images first:

```powershell
python materialize_cola_sample.py
```

Train a 90/10 split parser:

```powershell
python train.py --train-ratio 0.9 --seed 42 --min-confidence 0.85
```

This writes:

```text
weights/cola_field_parser.json
```

Commit or package this exported artifact before building Docker. Training happens once in the development environment; Docker only loads the exported parser and serves inference.

## Current Model Behavior

The current app does not match uploads to training images. It reads text from the uploaded image with OCR, then fills fields only when the parser is confident. If OCR text is jumbled and only ABV or brand is clear, only those fields should appear. The raw OCR text and uploaded image remain visible so the user can manually inspect anything the parser leaves blank.

The current data does not include word-level bounding boxes, so this is not yet a fine-tuned OCR detector/recognizer. It is an OCR-plus-field-parser pipeline trained from COLA CSV ground truth.
