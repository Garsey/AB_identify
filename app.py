from __future__ import annotations

from pathlib import Path

import streamlit as st
from PIL import Image

from field_parser import (
    DEFAULT_MODEL_PATH,
    image_id_from_filename,
    labels_to_rows,
    load_model,
    lookup_ground_truth,
    parse_label_text,
    predictions_to_rows,
    text_to_rows,
)
from ocr_engine import OCRLoadError, extract_text_from_image


st.set_page_config(page_title="TTB Label Reader", page_icon="AB", layout="wide")
st.title("TTB Label Reader")
st.caption("Upload an alcohol label image. OCR reads the label, then the trained parser fills only high-confidence fields.")


@st.cache_resource(show_spinner=False)
def load_parser_model(model_path: str) -> dict:
    return load_model(Path(model_path))


with st.sidebar:
    st.header("Parser")
    model_path = st.text_input("Weights File", str(DEFAULT_MODEL_PATH))
    min_confidence = st.slider("Minimum Confidence", min_value=50, max_value=99, value=85, step=1) / 100
    st.divider()
    st.write("Known sample-pack filenames show CSV ground truth for testing. Unknown labels are parsed from OCR only.")


try:
    model = load_parser_model(model_path)
except FileNotFoundError:
    st.error(f"Parser model not found: {model_path}. Run `python train.py --train-ratio 0.9` first.")
    st.stop()
except ValueError as exc:
    st.error(str(exc))
    st.stop()


split_counts = model.get("split_counts", {})
evaluation = model.get("evaluation", {})
st.info(
    "Loaded "
    f"{Path(model_path).name}: {split_counts.get('train', 0)} train OCR rows, "
    f"{split_counts.get('test', 0)} held-out OCR rows. "
    "Blank fields mean the parser was not confident enough to fill them."
)


uploaded_file = st.file_uploader("Choose a label image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is None:
    st.empty()
else:
    image = Image.open(uploaded_file)
    uploaded_image_id = image_id_from_filename(uploaded_file.name)
    ground_truth = lookup_ground_truth(model, uploaded_image_id)

    preview_col, result_col = st.columns([0.85, 1.15])

    with preview_col:
        st.image(image, caption=uploaded_file.name, use_container_width=True)
        if uploaded_image_id:
            st.write(f"Uploaded Image ID: `{uploaded_image_id}`")
        else:
            st.write("No TTB image ID found in the uploaded filename.")

    with result_col:
        with st.spinner("Reading label text..."):
            try:
                extracted_text = extract_text_from_image(image)
            except OCRLoadError as exc:
                extracted_text = ""
                st.error(str(exc))

        predictions = parse_label_text(model, extracted_text, min_confidence=min_confidence)

        st.subheader("Parsed Table")
        if predictions:
            st.dataframe(predictions_to_rows(predictions), use_container_width=True, hide_index=True)
        else:
            st.warning("No fields cleared the confidence threshold. Use the OCR text and image to inspect manually.")

        st.subheader("OCR Text")
        st.dataframe(text_to_rows(extracted_text), use_container_width=True, hide_index=True)

    if ground_truth:
        st.subheader("CSV Ground Truth")
        st.write(f"Ground truth split: `{ground_truth.get('split', '')}`")
        st.dataframe(labels_to_rows(ground_truth.get("labels", {})), use_container_width=True, hide_index=True)

        with st.expander("Ground Truth OCR Text"):
            st.dataframe(text_to_rows(ground_truth.get("ocr_text", "")), use_container_width=True, hide_index=True)
    elif uploaded_image_id:
        st.warning(f"No CSV ground truth found for `{uploaded_image_id}` in the exported parser model.")
