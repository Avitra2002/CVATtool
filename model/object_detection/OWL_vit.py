import streamlit as st
import os
from streamlit_img_label import st_img_label
from streamlit_img_label.manage import ImageManager, ImageDirManager
from transformers import Owlv2Processor, Owlv2ForObjectDetection
import cv2
import random
from PIL import Image
import torch

#TODO: add OWL vit model
from PIL import Image
import numpy as np
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection

processor_owl = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model_owl = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")


def run_owlvit_model(image_cv, confidence_threshold, iou_threshold, custom_labels,zone):
    """Run OWL-ViT model and return results (boxes, scores, labels)."""
    # Convert the OpenCV image (BGR format) to PIL image (RGB format)
    pil_image = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))

    st.write(f"Running OWL-ViT with confidence {confidence_threshold} and IOU {iou_threshold}")
    
    # Prepare inputs for the model
    inputs = processor_owl(text=custom_labels, images=pil_image, return_tensors="pt")
    outputs = model_owl(**inputs)

    # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
    target_sizes = torch.Tensor([pil_image.size[::-1]])
    
    # Convert outputs (bounding boxes and class logits) to Pascal VOC Format (xmin, ymin, xmax, ymax)
    results = processor_owl.post_process_object_detection(outputs=outputs, 
                                                      target_sizes=target_sizes, 
                                                      threshold=confidence_threshold)
    
    # Retrieve the first result (for the current image)
    if results:
        return results[0]["boxes"].tolist(), results[0]["scores"].tolist(), results[0]["labels"].tolist()
    else:
        return [], [], []
