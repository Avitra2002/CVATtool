import streamlit as st
from ultralytics import YOLO

def run_yolov_world(img_path, confidence_threshold,custom_label_list):
    """Run YOLOv8 World and return results (boxes, scores, labels)."""
    model_yolo_world= YOLO("yolov8s-worldv2.pt")
    model_yolo_world.set_classes(custom_label_list)

    results= model_yolo_world.predict(img_path)

    boxes = results[0].boxes.xyxy
    scores = results[0].boxes.conf  
    labels = results[0].boxes.cls   

    filtered_boxes = []
    filtered_scores = []
    filtered_labels = []

    # Filter results based on confidence threshold
    for i in range(len(scores)):
        if scores[i] >= confidence_threshold:
            filtered_boxes.append(boxes[i].tolist()) 
            filtered_scores.append(scores[i].item())
            filtered_labels.append(int(labels[i])) 

    return filtered_boxes, filtered_scores, filtered_labels

##TODO: Add download option for custom .pt for these custom classes