import streamlit as st
from ultralytics import YOLO

def run_yolov8_model(img_path, confidence_threshold, custom_label_list):
    """Run YOLOv8 model and return results (boxes, scores, labels)."""
    
    model_yolov11= YOLO('YOLO11s.pt')

    results= model_yolov11(img_path)

    boxes = []
    scores = []
    labels = []

    for box, score, label in zip(results[0].boxes.xyxy, results[0].boxes.conf, results[0].boxes.cls):
        if score >= confidence_threshold and model_yolov11.names[int(label)] in custom_label_list:
            boxes.append(box.tolist()) 
            scores.append(score.item())  
            labels.append(int(label))  

    return boxes, scores, labels
