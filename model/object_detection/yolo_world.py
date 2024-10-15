import streamlit as st

def run_yolov_world(image, confidence_threshold, iou_threshold):
    """Run YOLOv8 World and return results (boxes, scores, labels)."""
    # Replace with YOLOv8 detection code
    st.write(f"Running YOLO World with confidence {confidence_threshold} and IOU {iou_threshold}")
    # Dummy results
    return [[10, 10, 100, 100]], [0.9], [0]