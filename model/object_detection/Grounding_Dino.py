
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import streamlit as st

model_id = "IDEA-Research/grounding-dino-tiny"
device = "cuda" if torch.cuda.is_available() else "cpu"

processor_dino = AutoProcessor.from_pretrained(model_id)
model_dino = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

def run_grounding_DINO(img_path, confidence_threshold, iou_threshold, custom_labels):
    image= Image.open(img_path)
    """Run Grounding DINO model and return detection results."""
    st.write(f"Running Grounding DINO with confidence threshold {confidence_threshold} and IOU {iou_threshold}")
    
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Preprocess image and text
    inputs = processor_dino(images=image, text=custom_labels, return_tensors="pt").to(device)

    # Run the model and get outputs
    with torch.no_grad():
        outputs = model_dino(**inputs)

    # Post-process the results
    results = processor_dino.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=confidence_threshold, 
        text_threshold=iou_threshold,        
        target_sizes=[image.size[::-1]]     
    )
    print(results)

    # Extracting bounding boxes, labels, and confidence scores
    if results:
        boxes = results[0]["boxes"].tolist() if "boxes" in results[0] else []
        scores = results[0]["scores"].tolist() if "scores" in results[0] else []
        labels = results[0]["labels"] if "labels" in results[0] else []

        return boxes, scores, labels
    else:
        return [], [], []
