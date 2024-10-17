
import ast
from model.object_detection.yolov8 import run_yolov8_model
from model.object_detection.FlorenceV2 import run_florenceV2_model
from model.object_detection.OWL_vit import run_owlvit_model
from model.object_detection.yolo_world import run_yolov_world
from model.object_detection.Grounding_Dino import run_grounding_DINO
import cv2
import streamlit as st
import os 
import supervision as sv
import numpy as np
from utils.iou_zone_and_box import parse_string

def check_confidence_score(confidence_threshold, scores):
    if not scores:  # Check if any scores were returned
        st.warning("No detections were made.")
        return

    # Get the highest confidence score detected
    highest_confidence = max(scores)
    
    # Check if the confidence threshold is too high
    if confidence_threshold > highest_confidence:
        st.warning(f"The confidence threshold of {confidence_threshold:.2f} is too high. The highest confidence detected is {highest_confidence:.2f}.")
    
    else:
        st.info(f"The highest confidence level detected by the model for this image is {highest_confidence:.2f}.")

def annotate_image_detection(img_path, confidence_threshold, iou_threshold, selected_model, task_folder_path,image_index,task_prompt=None,custom_labels=None,custom_labels_list=None,zone=None):
    """Run the selected model on the image and save YOLO format annotations."""
    image_cv = cv2.imread(img_path)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)


    if selected_model == "YOLOv11":
        boxes, scores, labels = run_yolov8_model(img_path, confidence_threshold,iou_threshold, custom_labels_list,zone)
        check_confidence_score(confidence_threshold, scores)
    elif selected_model == "Florence V2":
        boxes, scores, labels = run_florenceV2_model(img_path, task_prompt, custom_labels,zone)
    elif selected_model == "OWL-ViT":
        boxes, scores, labels = run_owlvit_model(image_cv, confidence_threshold, iou_threshold, custom_labels,zone)
        check_confidence_score(confidence_threshold, scores)
    elif selected_model == "YOLO World":
        boxes, scores, labels = run_yolov_world(img_path, confidence_threshold,custom_labels_list,zone)
        check_confidence_score(confidence_threshold, scores)
    elif selected_model== "Grounding DINO":
        boxes, scores, labels = run_grounding_DINO(img_path, confidence_threshold, iou_threshold, custom_labels,custom_labels_list,zone)
        check_confidence_score(confidence_threshold, scores)


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
            x_min, y_min, x_max, y_max = map(int,box)

            # YOLO format: [class_id, x_center, y_center, width, height] (normalized)
            yolo_annotation = f"{label} {(x_min + x_max) / 2 / image_cv.shape[1]} {(y_min + y_max) / 2 / image_cv.shape[0]} {(x_max - x_min) / image_cv.shape[1]} {(y_max - y_min) / image_cv.shape[0]}"
            
            
            f.write(yolo_annotation + '\n')

            # Annotate the image with bounding boxes and labels
            cv2.rectangle(image_cv, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            if not isinstance(score, (int, float)):
                cv2.putText(image_cv, f"Label: {label}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            else:
                cv2.putText(image_cv, f"Label: {label}, Score: {round(score, 2)}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    if zone is not None:
        zones=parse_string(zone)
        # zones = ast.literal_eval(zone)
        
        if not isinstance(zones[0][0], list):
            zones = [zones]

        for zone in zones:
            zone_array = np.array(zone)
            zone_polygon=sv.PolygonZone(polygon=zone_array)
            zone_annotator= sv.PolygonZoneAnnotator(zone=zone_polygon,color= sv.Color.RED, thickness=3)

            image_cv= zone_annotator.annotate(scene=image_cv)

            

    # Save the annotated image in the 'annotated_images' subfolder
    annotated_image_path = os.path.join(annotated_subfolder, img_basename + img_extension)
    cv2.imwrite(annotated_image_path, image_cv)

    # Display the annotated image
    st.image(image_cv, use_column_width=True)
    st.write(f"Annotations saved to: {annotation_path}")