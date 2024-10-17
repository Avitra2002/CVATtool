import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
from utils.iou_zone_and_box import parse_string,calculate_iou
import supervision as sv

def run_yolov_world(img_path, confidence_threshold, iou_threshold, custom_label_list, zone):
    """Run YOLOv8 World and return results (boxes, scores, labels)."""
    
    
    model_yolo_world = YOLO("yolov8s-worldv2.pt")
    model_yolo_world.set_classes(custom_label_list)

    
    results = model_yolo_world.predict(img_path)
    detections = sv.Detections.from_ultralytics(results[0])
    
    
    boxes, scores, labels = [], [], []

    
    if zone is None:
    
        boxes = [box.tolist() for box, score, label in zip(results[0].boxes.xyxy, results[0].boxes.conf, results[0].boxes.cls) if score >= confidence_threshold]
        scores = [score.item() for score in results[0].boxes.conf if score >= confidence_threshold]
        labels = [int(label) for label in results[0].boxes.cls if results[0].boxes.conf >= confidence_threshold]

        return boxes, scores, labels

    else:
        # Handle zones
        zones = parse_string(zone)

        if not isinstance(zones[0][0], list):
            zones = [zones]  

        for zone_array in zones:
            zone_polygon = sv.PolygonZone(polygon=np.array(zone_array))
            mask = zone_polygon.trigger(detections=detections)
            detection_filtered = detections[mask]

            
            for box, confidence, class_id in zip(detection_filtered.xyxy, detection_filtered.confidence, detection_filtered.class_id):
                if confidence >= confidence_threshold:
                    box_list = box.tolist()
                    iou_value = calculate_iou(box_list, zone_polygon.polygon)
                    if iou_value >= iou_threshold:
                        boxes.append(box_list)
                        scores.append(confidence.item())
                        labels.append(class_id)

    return boxes, scores, labels


##TODO: Add download option for custom .pt for these custom classes