import streamlit as st
from PIL import Image
# TODO: Plan model step by step 

st.title("Image Classification")
st.write("Classify images using pretrained models.")

# Model selection
classification_models = ["ResNet50", "InceptionV3", "Florence v2"]
classification_tooltips = {
    "ResNet50": "A 50-layer Residual Network for image classification.",
    "InceptionV3": "A deep convolutional neural network architecture for image recognition.",
    "Florence v2": "A powerful vision model by Microsoft for image classification.",
}

selected_classification_model = st.selectbox("Select a model:", classification_models)
st.info(classification_tooltips[selected_classification_model])

# Dynamic input fields
if selected_classification_model == "ResNet50":
    top_k = st.number_input("Show top K predictions:", min_value=1, max_value=10, value=5)
elif selected_classification_model == "InceptionV3":
    top_k = st.number_input("Show top K predictions:", min_value=1, max_value=10, value=5)
elif selected_classification_model == "Florence v2":
    # Assume Florence v2 requires an API key
    api_key = st.text_input("Enter Florence v2 API Key:", type="password")

# Image upload
uploaded_files = st.file_uploader("Upload Images", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])

if uploaded_files:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        st.image(image, caption=uploaded_file.name)
        st.write("Classification results will be displayed here.")

# Download annotations
st.write("After classification, you can download the labels.")
if st.button("Download Labels"):
    st.write("Downloading classification labels...")
