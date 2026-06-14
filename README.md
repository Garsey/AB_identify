# AB Identify

Lightweight local prototype for OCR-driven alcohol label compliance checks.

The app is designed for a hybrid workflow:

1. Develop and train locally in a Python virtual environment.
2. Save custom OCR weights to `weights/alcohol_ocr.pt`.
3. Run the Streamlit web UI locally or inside Docker without retraining.

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

## Project Layout

- `app.py` - Streamlit upload UI and compliance results.
- `ocr_engine.py` - docTR model initialization, optional custom weight loading, and OCR text extraction.
- `compliance.py` - fuzzy text matching for expected brand and ABV values.
- `train.py` - training/export scaffold for dataset-specific fine-tuning.
- `weights/` - local model artifacts. Large weight files are ignored by Git.

## Training Flow

Install the full OCR/training stack after the UI is working:

```powershell
pip install -r requirements-ocr.txt
```

Place your dataset under `data/alcohol-labels` or pass a custom path:

```powershell
python train.py --dataset data/alcohol-labels --epochs 5
```

The training scaffold currently writes a manifest and marks the point where docTR's detection and recognition trainers should be connected to your final dataset format. Once training is wired, export the resulting checkpoint to:

```text
weights/alcohol_ocr.pt
```

When that file exists, `app.py` will try to load it on startup.
