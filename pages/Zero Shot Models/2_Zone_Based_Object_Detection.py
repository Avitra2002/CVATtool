import streamlit as st
from PIL import Image
import os
import streamlit.components.v1 as components
import yaml
from utils.make_dir import save_and_extract_zip,save_images
from utils.object_detection_annotation import annotate_image_detection
from utils.create_yaml_file import create_yaml_file

# TODO: Plan model step by step 
def run():
    st.set_page_config(page_title='Zone Based Object Detection', layout='wide')
    st.title("Zone Based Object Detection ")
    st.write("Detect objects at specific areas")

    # Model selection
    zone_model_options = ["YOLOv11", "OWL-ViT","YOLO World"]
    zone_model_tooltips = {
        "YOLOv11": "Custom Model A is trained specifically for niche objects.",
        "OWL-ViT": "Custom Model B is optimized for small datasets.",
        "YOLO World": "Custom Model B is optimized for small datasets.",
    }

    #TODO: Custom the open source code under polygon-zone/main using component.html
    components.iframe("https://polygonzone.roboflow.com",height=800, scrolling=True)


    selected_zone_model = st.selectbox("Select a model:", zone_model_options)
    st.info(zone_model_tooltips[selected_zone_model])

    zone= st.text_input("Copy the numpy coordinates from the polygone zone above",placeholder='[np.array([[302, 255], [410, 493], [802, 286], [718, 138]])]')



    # Dynamic input fields based on selected model
    if selected_zone_model == "YOLOv11":
        conf_threshold = st.slider("Confidence Threshold:", 0.0, 1.0, 0.5)
        iou_threshold = st.slider("IOU Threshold:",0.0, 1.0, 0.5)
        def load_class_names(yaml_file):
            with open(yaml_file, 'r') as file:
                data = yaml.safe_load(file)
                return [data['names'][i] for i in range(len(data['names']))]

        # Load the class names from coco dataset
        class_names = load_class_names('/Users/phonavitra/Desktop/CVATtool/CVATtool/coco.yaml')

        # Create a multi-select dropdown
        custom_labels_list = st.multiselect(
            'Select the object classes:',
            class_names
        )
        custom_labels=None

    elif selected_zone_model == "OWL-ViT":
            conf_threshold = st.slider("Confidence Threshold:", 0.0, 1.0, 0.6)
            iou_threshold = st.slider("IOU Threshold", 0.0, 1.0, 0.05, 0.001) 
            custom_labels_input = st.text_input("Enter custom labels (comma-separated, e.g., cat,dog,car):", value="")
            custom_labels_list = [label.strip() for label in custom_labels_input.split(",")]

            if custom_labels_input.strip() == "":
                st.warning("Custom labels cannot be empty. Please enter labels separated by commas.")
            else:
                custom_labels = [[f"a photo of a {label}" for label in custom_labels_list]]


    elif selected_zone_model == "YOLO World":
            iou_threshold= st.slider("IOU Threshold", 0.000, 1.000, 0.050, 0.001)  
            conf_threshold = st.slider("Confidence Threshold:", 0.0, 1.0, 0.5)
            custom_labels_input= st.text_input("Enter custom labels (comma-separated, e.g., cat,dog,car):", value="")
            custom_labels_list=[label.strip() for label in custom_labels_input.split(",")]
            custom_labels=None

    task_name = st.text_input("Enter Task Name:", value="DefaultTask")
    task_folder_path= f"files/{task_name}"

    # Image upload
    uploaded_files = st.file_uploader("Upload Images (or Zip Folder)", accept_multiple_files=True, type=['png', 'jpg', 'jpeg','zip'])

    files_dir= 'files'
    folder_path =''



    if uploaded_files:
            for uploaded_file in uploaded_files:
                if uploaded_file.type == 'application/zip':
                    # Handle zip file
                    folder_path = save_and_extract_zip(files_dir,uploaded_file,task_name)
                    st.write(f"Extracted and saved files to: {folder_path}")
                else:
                    # Handle images
                    folder_path = save_images(files_dir,uploaded_files,task_name)
            st.write(f"Uploaded and saved images to: {folder_path}")

    if not folder_path:
            st.warning("Please upload images or zip file to proceed.")
            return

        # List files in the directory
    st.write(f"Folder path:{folder_path}")
    # image_files = [f for f in os.listdir(os.path.join(folder_path, "images")) if f.endswith(('png', 'jpg', 'jpeg'))]
    image_files = sorted(os.listdir(os.path.join(folder_path, "images")))


    if not image_files:
        st.warning("No images found in the uploaded folder.")
        return

    if "image_index" not in st.session_state:
        st.session_state["image_index"] = 0

    col1, col2, col3 = st.columns(3)

    # Display status for images
    if len(image_files) ==1:
        st.warning("There is only 1 image. Next and Previous Button is disabled")
    elif st.session_state.image_index==0:
        st.warning('This is the first image.')

    elif st.session_state.image_index == len(image_files)-1:
        st.warning('This is the last image.')
    else:
        st.info(f"Image {st.session_state.image_index + 1} of {len(image_files)}")

    with col1:
        st.button("Previous Image", on_click=lambda: st.session_state.update(image_index=max(0, st.session_state.image_index - 1)), 
                disabled=st.session_state.image_index == 0)

    with col2:
        st.button("Next Image", on_click=lambda: st.session_state.update(image_index=min(len(image_files) - 1, st.session_state.image_index + 1)), 
                disabled=st.session_state.image_index == len(image_files) - 1)

    with col3:
        if st.button("Annotate All"):
            # Step 1: Annotate all images
            for index, img_file in enumerate(image_files):
                img_path = os.path.join(folder_path, "images", img_file)
                annotation_filename = f"image_{index+1}.txt"  # Match with your annotation naming convention
                annotation_path = os.path.join(task_folder_path, "labels", annotation_filename)

                # Check if the annotation already exists
                if os.path.exists(annotation_path):
                    st.write(f"Skipping annotation for {img_file} (already annotated).")
                    continue  # Skip this image if it has already been annotated
                
                # Annotate the image
                annotate_image_detection(img_path, conf_threshold, iou_threshold, selected_zone_model, task_folder_path, index+1,task_prompt=None,custom_labels=custom_labels,custom_labels_list=custom_labels_list,zone=zone,All=True)

            st.success("All images have been annotated.")
            return

    # Show current image
    img_file = image_files[st.session_state.image_index]
    img_path = os.path.join(folder_path, "images", img_file)

    # Annotate the current image and display
    annotate_image_detection(img_path, conf_threshold, iou_threshold, selected_zone_model, task_folder_path, st.session_state.image_index + 1, task_prompt=None,custom_labels=custom_labels,custom_labels_list=custom_labels_list,zone=zone)

    create_yaml_file(task_folder_path, custom_labels_list)


# if __name__ == '__main__':
#     run()

run()