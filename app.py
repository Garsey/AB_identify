from __future__ import annotations

from pathlib import Path

import streamlit as st
from PIL import Image

from compliance import compare_abv, compare_field
from ocr_engine import DEFAULT_DEMO_TEXT, OCRLoadError, build_predictor, extract_text_from_image


WEIGHTS_PATH = Path("weights/alcohol_ocr.pt")


st.set_page_config(page_title="TTB Label Compliance Prototype", page_icon="AB", layout="wide")
st.title("TTB Label Compliance Prototype")
st.caption("Upload an alcohol label image and compare extracted OCR text against expected application data.")


@st.cache_resource(show_spinner=False)
def load_custom_model(weights_path: str) -> tuple[object | None, str]:
    try:
        predictor = build_predictor(weights_path=weights_path, use_pretrained=not Path(weights_path).exists())
    except OCRLoadError as exc:
        return None, str(exc)
    except Exception as exc:  # pragma: no cover - keeps UI usable when native deps fail at runtime
        return None, f"OCR model could not be initialized: {exc}"

    if Path(weights_path).exists():
        return predictor, f"Loaded custom weights from {weights_path}"
    return predictor, "Custom weights not found; using docTR pretrained weights."


with st.sidebar:
    st.header("Expected Application Data")
    expected_brand = st.text_input("Brand Name", "OLD TOM DISTILLERY")
    expected_abv = st.text_input("ABV %", "45%")
    match_threshold = st.slider("Brand Match Threshold", min_value=50, max_value=100, value=85, step=1)
    st.divider()
    st.write("Model artifact")
    st.code(str(WEIGHTS_PATH), language="text")


predictor, model_status = load_custom_model(str(WEIGHTS_PATH))
if predictor is None:
    st.warning(f"{model_status} Demo text will be used until dependencies and weights are available.")
else:
    st.info(model_status)


uploaded_file = st.file_uploader("Choose a label image", type=["jpg", "jpeg", "png"])

if uploaded_file is None:
    st.empty()
else:
    image = Image.open(uploaded_file)
    preview_col, result_col = st.columns([1, 1])

    with preview_col:
        st.image(image, caption="Uploaded Label", use_container_width=True)

    with result_col:
        with st.spinner("Extracting and analyzing text..."):
            if predictor is None:
                extracted_text = DEFAULT_DEMO_TEXT
            else:
                extracted_text = extract_text_from_image(predictor, image) or DEFAULT_DEMO_TEXT

            brand_result = compare_field("Brand Name", expected_brand, extracted_text, threshold=float(match_threshold))
            abv_result = compare_abv(expected_abv, extracted_text)

        st.subheader("Extracted Text")
        st.text_area("OCR output", extracted_text, height=120)

        st.subheader("Compliance Results")
        for result in (brand_result, abv_result):
            message = f"{result.label}: {result.score:.1f}% confidence"
            if result.passed:
                st.success(message)
            else:
                st.error(message)
