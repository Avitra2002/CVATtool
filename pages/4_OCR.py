import streamlit as st
from PIL import Image

st.title("Optical Character Recognition (OCR)")
st.write("Extract text from images using OCR models.")

# Model selection
ocr_models = ["Tesseract OCR", "EasyOCR", "Microsoft Read API"]
ocr_tooltips = {
    "Tesseract OCR": "An open-source OCR engine that recognizes over 100 languages.",
    "EasyOCR": "A ready-to-use OCR with 80+ supported languages.",
    "Microsoft Read API": "Cloud-based OCR service by Microsoft Azure.",
}

selected_ocr_model = st.selectbox("Select a model:", ocr_models)
st.info(ocr_tooltips[selected_ocr_model])

# Dynamic input fields
if selected_ocr_model == "Tesseract OCR":
    language = st.text_input("Enter language code (e.g., 'eng' for English):", value='eng')
elif selected_ocr_model == "EasyOCR":
    languages = st.multiselect("Select languages:", ["en", "fr", "de", "es", "it"])
elif selected_ocr_model == "Microsoft Read API":
    api_endpoint = st.text_input("Enter API Endpoint:")
    api_key = st.text_input("Enter API Key:", type="password")

# Image upload
uploaded_files = st.file_uploader("Upload Images", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])

if uploaded_files:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        st.image(image, caption=uploaded_file.name)
        st.write("Extracted text will be displayed here.")

# Download annotations
st.write("After processing, you can download the extracted text.")
if st.button("Download Text"):
    st.write("Downloading extracted text...")
