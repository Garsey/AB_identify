# AB Identify

Lightweight local prototype for reading alcohol label images, filling an editable seven-field table, and saving entries during the current app session.

The project uses the standard training/deployment split:

1. Materialize labeled COLA sample data locally.
2. Train/export a parser artifact from OCR text and CSV ground truth.
3. Build Docker as an inference-only web app that reads new uploads with pretrained OCR, fills only confident values, and lets users edit/save the final table entry.

## Local Setup

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m streamlit run app.py
```

Then open http://localhost:8501.

## Docker Setup

```powershell
docker compose up --build
```

Then open http://localhost:8501.

The Docker image is inference-only. It installs Tesseract OCR plus the lightweight Python UI/parser dependencies, includes `weights/cola_field_parser.json`, and does not mount the full repo or training data at runtime.

For CPU inference:

```powershell
docker compose up --build
```

For NVIDIA/CUDA inference, make sure Docker Desktop has NVIDIA Container Toolkit support enabled, then run:

```powershell
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

The app requests CUDA in that mode, but the OCR code still checks PyTorch CUDA availability before using it.

## Project Layout

- `app.py` - Streamlit upload UI that OCRs a new image and fills the parsed table.
- `ocr_engine.py` - pretrained OCR backend selection. EasyOCR is preferred, with Tesseract fallback.
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

The current app does not match uploads to training images. It reads text from each uploaded image with OCR, then fills fields only when the parser is confident. If OCR text is jumbled and only ABV or brand is clear, only those fields should appear. The raw OCR text and uploaded image remain visible so the user can manually inspect anything the parser leaves blank.

Users can upload multiple photos for the same product one at a time. Values already captured stay in the editable form while the next photo is processed. `Save entry` appends all seven values to an in-memory pandas table, which is intentionally cleared when the app reinitializes.

The seven tracked fields are:

- Brand name
- Class/type designation
- Alcohol content
- Net contents
- Name and address of bottler/producer
- Country of origin for imports
- Government Health Warning Statement

## OCR Design Decision

The project intentionally uses a pretrained OCR model rather than training our own OCR from scratch. EasyOCR is the preferred OCR backend because it is stronger than plain Tesseract for scene text and can evaluate rotated text, which helps with vertical label copy and alternate fonts. Tesseract remains available as a fallback.

Our exported training artifact currently improves the second layer: OCR text to structured alcohol-label fields. The current data does not include word-level bounding boxes, so it is not yet suitable for clean supervised OCR detector/recognizer fine-tuning. If we later add text-region annotations, we can fine-tune a pretrained OCR model while keeping the field parser layer.
