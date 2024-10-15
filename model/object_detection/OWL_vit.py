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
def run_owlvit_model(image, confidence_threshold, iou_threshold, custom_labels):
    """Run OWL-ViT model and return results (boxes, scores, labels)."""
    # Example for running OWL-ViT model (replace with actual code)
    st.write(f"Running OWL-ViT with confidence {confidence_threshold} and IOU {iou_threshold}")
    # Dummy results
    return [[30, 30, 150, 150]], [0.95], [2]