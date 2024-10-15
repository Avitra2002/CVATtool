import streamlit as st
from PIL import Image
from utils.make_dir import save_and_extract_zip,save_images
import cv2
import numpy
from model.object_detection.yolov8 import run_yolov8_model
from model.object_detection.fastercnn import run_fasterrcnn_model
from model.object_detection.OWL_vit import run_owlvit_model
import os
import glob

def annotate_image_detection(img_path, confidence_threshold, iou_threshold, selected_model, task_folder_path,image_index,custom_labels=None):
    """Run the selected model on the image and save YOLO format annotations."""
    image_cv = cv2.imread(img_path)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

    ## TODO: Check if all output of detection models are bouding boxes

    if selected_model == "YOLOv8":
        boxes, scores, labels = run_yolov8_model(image_cv, confidence_threshold, iou_threshold)
    elif selected_model == "Faster R-CNN":
        boxes, scores, labels = run_fasterrcnn_model(image_cv, confidence_threshold, iou_threshold)
    elif selected_model == "OWL-ViT":
        boxes, scores, labels = run_owlvit_model(image_cv, confidence_threshold, iou_threshold, custom_labels)

    labels_subfolder = os.path.join(task_folder_path, "labels")
    annotated_subfolder = os.path.join(task_folder_path, "annotated_images")

    # Ensure the subfolders exist
    os.makedirs(labels_subfolder, exist_ok=True)
    os.makedirs(annotated_subfolder, exist_ok=True)

    img_basename = f"image_{image_index}"
    img_extension = img_extension = os.path.splitext(img_path)[1] 

    # Create and save YOLO-style annotations in the 'labels' subfolder
    annotation_filename = img_basename + ".txt"
    annotation_path = os.path.join(labels_subfolder, annotation_filename)

    with open(annotation_path, 'w') as f:
        for box, score, label in zip(boxes, scores, labels):
            x_min, y_min, x_max, y_max = box

            # YOLO format: [class_id, x_center, y_center, width, height] (normalized)
            yolo_annotation = f"{label} {(x_min + x_max) / 2 / image_cv.shape[1]} {(y_min + y_max) / 2 / image_cv.shape[0]} {(x_max - x_min) / image_cv.shape[1]} {(y_max - y_min) / image_cv.shape[0]}"
            
            # Write annotation to file
            f.write(yolo_annotation + '\n')

            # Annotate the image with bounding boxes and labels
            cv2.rectangle(image_cv, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            cv2.putText(image_cv, f"Label: {label}, Score: {round(score, 2)}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Save the annotated image in the 'annotated_images' subfolder
    annotated_image_path = os.path.join(annotated_subfolder, img_basename + img_extension)
    cv2.imwrite(annotated_image_path, image_cv)

    # Display the annotated image
    st.image(image_cv, use_column_width=True)
    st.write(f"Annotations saved to: {annotation_path}")

model_paths = {
    "YOLOv8": "path_to_yolov8_model",
    "Faster R-CNN": "path_to_faster_rcnn_model",
    "OWL-ViT": "path_to_owlvit_model"
}
# Placeholder 
processor = None  
model_owlvit = None  
model_yolov8 = None  
model_fasterrcnn = None 

# Load models based on the selection
def load_model(selected_model):
    if selected_model == "YOLOv8":
        # Load YOLOv8 model here
        return model_yolov8
    elif selected_model == "Faster R-CNN":
        # Load Faster R-CNN model here
        return model_fasterrcnn
    elif selected_model == "OWL-ViT":
        # Load OWL-ViT model here
        return model_owlvit
    return None


def run():
    st.title("Object Detection And Annotation")
    st.write("Detect objects in images using state-of-the-art models and save the labels for .")

    # Model selection
    model_options = ["YOLOv8", "Faster R-CNN", "OWL-Vit"]
    model_tooltips = {
        "YOLOv8": "YOLOv5 is a family of object detection architectures and models pretrained on the COCO dataset.",
        "Faster R-CNN": "Faster R-CNN is a region-based convolutional neural network for object detection.",
        "SSD": "Single Shot MultiBox Detector (SSD) is a method for detecting objects in images using a single deep neural network.",
    }

    selected_model = st.selectbox("Select a model:", model_options)
    st.info(model_tooltips[selected_model])

    # Dynamic input fields based on selected model
    if selected_model == "YOLOv8":
        conf_threshold = st.slider("Confidence Threshold:", 0.0, 1.0, 0.5)
        nms_threshold = st.slider("NMS Threshold:", 0.0, 1.0, 0.4)
        iou_threshold = st.slider("IOU Threshold", 0.0, 1.0, 0.5, 0.05)
    elif selected_model == "Faster R-CNN":
        conf_threshold = st.slider("Confidence Threshold:", 0.0, 1.0, 0.7)
        iou_threshold = st.slider("IOU Threshold", 0.0, 1.0, 0.5, 0.05)
    elif selected_model == "OWL-Vit":
        conf_threshold = st.slider("Confidence Threshold:", 0.0, 1.0, 0.6)
        iou_threshold = st.slider("IOU Threshold", 0.0, 1.0, 0.5, 0.05)

    task_name = st.text_input("Enter Task Name:", value="DefaultTask")
    task_folder_path= f"files/{task_name}"

    # Image upload
    uploaded_files = st.file_uploader("Upload Images (or Zip  Folder)", accept_multiple_files=True, type=['png', 'jpg', 'jpeg','zip'])

    files_dir= 'files'
    folder_path = ""

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
    image_files = [f for f in os.listdir(os.path.join(folder_path, "images")) if f.endswith(('png', 'jpg', 'jpeg'))]


    if not image_files:
        st.warning("No images found in the uploaded folder.")
        return

    if "image_index" not in st.session_state:
        st.session_state["image_index"] = 0

    col1, col2, col3 = st.columns(3)

    # Display status for images
    if len(image_files) ==1:
        st.info("There is only 1 image. Next and Previous Button is disabled")
    elif st.session_state.image_index==0:
        st.info('This is the first image. Previous Button is disabled')

    elif st.session_state.image_index == len(image_files)-1:
        st.info('This is the last image. Next Button is disabled.')
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
                annotate_image_detection(img_path, conf_threshold, iou_threshold, selected_model, task_folder_path, index+1, custom_labels=None)

            st.success("All images have been annotated.")
            return

    # Show current image
    img_file = image_files[st.session_state.image_index]
    img_path = os.path.join(folder_path, "images", img_file)
    
    # Annotate the current image and display
    annotate_image_detection(img_path, conf_threshold, iou_threshold, selected_model, task_folder_path, st.session_state.image_index + 1, custom_labels=None)

# Run the app
if __name__ == '__main__':
    run()

    
