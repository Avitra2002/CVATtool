import streamlit as st
import os
from streamlit_img_label import st_img_label
from streamlit_img_label.manage import ImageManager, ImageDirManager
from transformers import Owlv2Processor, Owlv2ForObjectDetection
import cv2
import random
from PIL import Image
import torch
from utils.iou_zone_and_box import parse_string,calculate_iou

#TODO: add OWL vit model
from PIL import Image
import numpy as np
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection
import supervision as sv

processor_owl = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model_owl = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")


def run_owlvit_model(image_cv, confidence_threshold, iou_threshold, custom_labels,zone):
    """Run OWL-ViT model and return results (boxes, scores, labels)."""

    boxes, scores, labels = [], [], []
    pil_image = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))

    st.write(f"Running OWL-ViT with confidence {confidence_threshold} and IOU {iou_threshold}")
    
    inputs = processor_owl(text=custom_labels, images=pil_image, return_tensors="pt")
    outputs = model_owl(**inputs)
    target_sizes = torch.Tensor([pil_image.size[::-1]])
    
    results = processor_owl.post_process_object_detection(outputs=outputs, 
                                                      target_sizes=target_sizes, 
                                                      threshold=confidence_threshold)
    
    # Chnage Pytorch tensors into numpy
    boxes = results[0]['boxes'].detach().cpu().numpy()
    scores = results[0]['scores'].detach().cpu().numpy()
    labels = results[0]['labels'].detach().cpu().numpy()

    filtered_boxes, filtered_scores, filtered_labels = [], [], []

    # Handle no zones case (global object detection)
    if zone is None:
        for i, score in enumerate(scores):
            if score >= confidence_threshold:
                filtered_boxes.append(boxes[i].tolist())
                filtered_scores.append(score.item())
                filtered_labels.append(int(labels[i]))

        return filtered_boxes, filtered_scores, filtered_labels

    # Handle zone-based filtering
    else:

        zones = parse_string(zone)  # Parse the zone string into numpy arrays

        if not isinstance(zones[0][0], list):
            zones = [zones]

        for zone_array in zones:
            zone_polygon = sv.PolygonZone(polygon=np.array(zone_array))

            for i, box in enumerate(boxes):
                if scores[i] >= confidence_threshold:
                    iou_value = calculate_iou(box.tolist(), zone_polygon.polygon)
                    if iou_value >= iou_threshold:
                        filtered_boxes.append(box.tolist())
                        filtered_scores.append(scores[i].item())
                        filtered_labels.append(int(labels[i]))

        return filtered_boxes, filtered_scores, filtered_labels



