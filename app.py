from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from field_parser import (
    DEFAULT_MODEL_PATH,
    FIELD_LABELS,
    FIELD_ORDER,
    FieldPrediction,
    load_model,
    parse_label_text,
    predictions_to_rows,
    text_to_rows,
)
from ocr_engine import OCRLoadError, build_ocr_reader, env_backend, env_device, extract_text_from_image


st.set_page_config(page_title="TTB Label Reader", page_icon="AB", layout="wide")
st.title("TTB Label Reader")


@st.cache_resource(show_spinner=False)
def load_parser_model() -> dict:
    return load_model(DEFAULT_MODEL_PATH)


@st.cache_resource(show_spinner=False)
def load_ocr_backend():
    return build_ocr_reader(env_backend(), env_device())


def empty_entry() -> dict[str, str]:
    return {field: "" for field in FIELD_ORDER}


def initialize_state() -> None:
    if "current_entry" not in st.session_state:
        st.session_state.current_entry = empty_entry()
    if "entries" not in st.session_state:
        st.session_state.entries = pd.DataFrame(columns=[FIELD_LABELS[field] for field in FIELD_ORDER])
    if "last_upload_key" not in st.session_state:
        st.session_state.last_upload_key = None
    if "last_ocr_text" not in st.session_state:
        st.session_state.last_ocr_text = ""
    if "last_predictions" not in st.session_state:
        st.session_state.last_predictions = []
    if "uploader_version" not in st.session_state:
        st.session_state.uploader_version = 0


def merge_predictions(predictions: list[FieldPrediction]) -> list[str]:
    filled: list[str] = []
    for prediction in predictions:
        if prediction.field not in st.session_state.current_entry:
            continue
        current_value = st.session_state.current_entry[prediction.field].strip()
        if not current_value and prediction.value.strip():
            st.session_state.current_entry[prediction.field] = prediction.value.strip()
            filled.append(prediction.field)
    return filled


def save_entry() -> None:
    row = {FIELD_LABELS[field]: st.session_state.current_entry.get(field, "") for field in FIELD_ORDER}
    st.session_state.entries = pd.concat([st.session_state.entries, pd.DataFrame([row])], ignore_index=True)
    st.session_state.current_entry = empty_entry()
    st.session_state.last_upload_key = None
    st.session_state.last_ocr_text = ""
    st.session_state.last_predictions = []
    st.session_state.uploader_version += 1


def clear_entry() -> None:
    st.session_state.current_entry = empty_entry()
    st.session_state.last_upload_key = None
    st.session_state.last_ocr_text = ""
    st.session_state.last_predictions = []
    st.session_state.uploader_version += 1


initialize_state()

try:
    model = load_parser_model()
except FileNotFoundError:
    st.error(f"Parser model not found: {DEFAULT_MODEL_PATH}. Run `python train.py --train-ratio 0.9` first.")
    st.stop()
except ValueError as exc:
    st.error(str(exc))
    st.stop()

try:
    ocr_reader = load_ocr_backend()
except OCRLoadError as exc:
    st.error(str(exc))
    st.stop()


st.markdown(
    """
Upload label photos one at a time for the same product. Each photo may fill only some fields; values already captured stay in the form while you upload the next photo. Review or edit every value before saving. When the entry is complete enough, select **Save entry** to append all seven fields to the table.
"""
)

uploaded_file = st.file_uploader(
    "Upload one label photo",
    type=["jpg", "jpeg", "png", "webp"],
    key=f"label_upload_{st.session_state.uploader_version}",
)

if uploaded_file is not None:
    upload_bytes = uploaded_file.getvalue()
    upload_key = f"{uploaded_file.name}:{len(upload_bytes)}:{hash(upload_bytes)}"
    if upload_key != st.session_state.last_upload_key:
        image = Image.open(uploaded_file)
        with st.spinner("Reading this label photo..."):
            try:
                ocr_result = extract_text_from_image(image, ocr_reader)
                predictions = parse_label_text(model, ocr_result.text, min_confidence=0.85)
                filled_fields = merge_predictions(predictions)
                st.session_state.last_ocr_text = ocr_result.text
                st.session_state.last_predictions = predictions
                st.session_state.last_upload_key = upload_key
                if filled_fields:
                    st.success("Filled: " + ", ".join(FIELD_LABELS[field] for field in filled_fields))
                else:
                    st.info("No new blank fields were filled from this photo.")
            except OCRLoadError as exc:
                st.error(str(exc))

    st.image(Image.open(uploaded_file), caption=uploaded_file.name, use_container_width=True)


form_col, text_col = st.columns([1.05, 0.95])

with form_col:
    st.subheader("Current Entry")
    with st.form("entry_form"):
        for field in FIELD_ORDER:
            label = FIELD_LABELS[field]
            if field in {"bottler_producer_address", "government_warning"}:
                st.session_state.current_entry[field] = st.text_area(
                    label,
                    value=st.session_state.current_entry.get(field, ""),
                    height=86,
                )
            else:
                st.session_state.current_entry[field] = st.text_input(
                    label,
                    value=st.session_state.current_entry.get(field, ""),
                )

        save_col, clear_col = st.columns(2)
        with save_col:
            saved = st.form_submit_button("Save entry", type="primary")
        with clear_col:
            cleared = st.form_submit_button("Clear current entry")

    if saved:
        save_entry()
        st.success("Saved entry to the table.")
        st.rerun()
    if cleared:
        clear_entry()
        st.rerun()

with text_col:
    st.subheader("Parsed Fields From Last Photo")
    predictions = st.session_state.last_predictions
    if predictions:
        st.dataframe(predictions_to_rows(predictions), use_container_width=True, hide_index=True)
    else:
        st.write("No parsed fields for the current photo yet.")

    st.subheader("OCR Text From Last Photo")
    st.dataframe(text_to_rows(st.session_state.last_ocr_text), use_container_width=True, hide_index=True)


st.subheader("Saved Entries")
st.dataframe(st.session_state.entries, use_container_width=True, hide_index=True)
