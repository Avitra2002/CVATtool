import streamlit as st
from ultralytics import YOLO
import supervision as sv
import numpy as np
import cv2
from PIL import Image
import yaml
import ast
from utils.iou_zone_and_box import calculate_iou, parse_string
import re

model_yolov11 = YOLO('YOLO11s.pt')

# def run_yolov8_model(img_path, confidence_threshold, iou_threshold, custom_label_list, zone=None):
#     """Run YOLOv8 model and return filtered results (boxes, scores, labels)."""
    
#     results = model_yolov11(img_path)

#     # Initialize lists for boxes, scores, and labels
#     boxes, scores, labels = [], [], []

#     if zone is None:
#         for box, score, label in zip(results[0].boxes.xyxy, results[0].boxes.conf, results[0].boxes.cls):
#             if score >= confidence_threshold and model_yolov11.names[int(label)] in custom_label_list:
#                 boxes.append(box.tolist())
#                 scores.append(score.item())
#                 labels.append(int(label))

#     else:
#         image = Image.open(img_path)
#         frame = np.array(image)

#         if isinstance(zone, str):
#             zone = ast.literal_eval(zone)  

#         zone = np.array(zone)
#         zone_polygon = sv.PolygonZone(polygon=zone)

#         detections = sv.Detections.from_ultralytics(results[0])
#         mask = zone_polygon.trigger(detections=detections)
#         detection_filtered = detections[mask]

#         # Filter based on custom label list and confidence threshold
#         for i in range(len(detection_filtered)):
#             class_name = detection_filtered.data['class_name'][i]
#             if class_name in custom_label_list and detection_filtered.confidence[i] >= confidence_threshold:
#                 box = detection_filtered.xyxy[i].tolist()
#                 iou_value = calculate_iou(box, zone_polygon.polygon)
#                 if iou_value >= iou_threshold:
#                     boxes.append(box)
#                     scores.append(detection_filtered.confidence[i].item())  # Convert to float
#                     labels.append(detection_filtered.class_id[i])

#     return boxes, scores, labels


def run_yolov8_model(img_path, confidence_threshold, iou_threshold, custom_label_list, zone=None):
    """Run YOLOv8 model and return filtered results (boxes, scores, labels)."""
    
    results = model_yolov11(img_path)
    boxes, scores, labels = [], [], []

    if zone is None:
        for box, score, label in zip(results[0].boxes.xyxy, results[0].boxes.conf, results[0].boxes.cls):
            if score >= confidence_threshold and model_yolov11.names[int(label)] in custom_label_list:
                boxes.append(box.tolist())
                scores.append(score.item())
                labels.append(int(label))
    else:


        image = Image.open(img_path)
        frame = np.array(image)
        zones=parse_string(zone)
        # zones = ast.literal_eval(zone)

        if not isinstance(zones[0][0], list):
            zones = [zones]  

        detections = sv.Detections.from_ultralytics(results[0])

        for zone_array in zones:
            zone = np.array(zone_array)
            zone_polygon = sv.PolygonZone(polygon=zone)

            mask = zone_polygon.trigger(detections=detections)
            detection_filtered = detections[mask]

            # Filter based on custom label list and confidence threshold
            for i in range(len(detection_filtered)):
                class_name = detection_filtered.data['class_name'][i]
                if class_name in custom_label_list and detection_filtered.confidence[i] >= confidence_threshold:
                    box = detection_filtered.xyxy[i].tolist()
                    iou_value = calculate_iou(box, zone_polygon.polygon)
                    if iou_value >= iou_threshold:
                        boxes.append(box)
                        scores.append(detection_filtered.confidence[i].item())
                        labels.append(detection_filtered.class_id[i])

    return boxes, scores, labels





