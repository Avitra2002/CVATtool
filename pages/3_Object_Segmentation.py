import streamlit as st
from PIL import Image
# TODO: Plan model step by step 

st.title("Object Segmentation")
st.write("Segment objects within images using advanced models.")

# Model selection
segmentation_models = ["Mask R-CNN", "DeepLabV3", "SAM"]
segmentation_tooltips = {
    "Mask R-CNN": "Mask R-CNN extends Faster R-CNN to pixel-level image segmentation.",
    "DeepLabV3": "DeepLabV3 is a semantic segmentation model that uses atrous convolution.",
    "SAM": "Segment Anything Model (SAM) can segment any object given a prompt.",
}

selected_segmentation_model = st.selectbox("Select a model:", segmentation_models)
st.info(segmentation_tooltips[selected_segmentation_model])

# Dynamic input fields
if selected_segmentation_model == "Mask R-CNN":
    mask_threshold = st.slider("Mask Threshold:", 0.0, 1.0, 0.5)
elif selected_segmentation_model == "DeepLabV3":
    output_stride = st.selectbox("Output Stride:", [8, 16])
elif selected_segmentation_model == "SAM":
    prompt = st.text_input("Enter a prompt for segmentation:")

# Image upload
uploaded_files = st.file_uploader("Upload Images", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])

if uploaded_files:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        st.image(image, caption=uploaded_file.name)
        st.write("Segmentation results will be displayed here.")

# Download annotations
st.write("After annotations, you can download the labels.")
download_format = st.selectbox("Select download format:", ["CSV", "Mask Format"])
if st.button("Download Labels"):
    st.write(f"Downloading labels in {download_format} format...")
