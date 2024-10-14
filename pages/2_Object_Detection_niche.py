import streamlit as st
from PIL import Image

st.title("Object Detection - Niche Objects")
st.write("Detect niche objects using specialized models.")

# Model selection
niche_model_options = ["Custom Model A", "Custom Model B"]
niche_model_tooltips = {
    "Custom Model A": "Custom Model A is trained specifically for niche objects.",
    "Custom Model B": "Custom Model B is optimized for small datasets.",
}

selected_niche_model = st.selectbox("Select a model:", niche_model_options)
st.info(niche_model_tooltips[selected_niche_model])

# Dynamic input fields based on selected model
if selected_niche_model == "Custom Model A":
    param_a = st.number_input("Parameter A:", min_value=0.0, max_value=1.0, value=0.5)
elif selected_niche_model == "Custom Model B":
    param_b = st.number_input("Parameter B:", min_value=0, max_value=100, value=50)

# Image upload
uploaded_files = st.file_uploader("Upload Images", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])

if uploaded_files:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        st.image(image, caption=uploaded_file.name)
        st.write("Niche object detection results will be displayed here.")

# Download annotations
st.write("After annotations, you can download the labels.")
download_format = st.selectbox("Select download format:", ["CSV", "YOLO"])
if st.button("Download Labels"):
    st.write(f"Downloading labels in {download_format} format...")
