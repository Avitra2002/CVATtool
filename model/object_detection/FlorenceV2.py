import streamlit as st
import os
import cv2
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_florence = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base", torch_dtype=torch_dtype, trust_remote_code=True).to(device)
processor_florence = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)

def run_florenceV2_model(img_path, task_prompt, text_input):
    """Run Florence V2 model for object detection."""
    st.write(f"Running Florence V2")

    #Text input: custom labels
    custom_labels_list = [label.strip() for label in text_input.split(",")]

    image= Image.open(img_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    prompt = task_prompt+text_input

    try:
        inputs = processor_florence(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)
    except Exception as e:
        st.write(f"Error processing inputs for {img_path}: {e}")
        return [], [], []  # Return empty lists if processing fails
    
    generated_ids = model_florence.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        do_sample=False,
        num_beams=3,
    )
    
    generated_text = processor_florence.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor_florence.post_process_generation(generated_text, task=task_prompt, image_size=(image.width, image.height))
    print(f"Florence results: {parsed_answer}")

    bboxes = parsed_answer.get('<OPEN_VOCABULARY_DETECTION>', {}).get('bboxes', [])
    labels = parsed_answer.get('<OPEN_VOCABULARY_DETECTION>', {}).get('bboxes_labels', [])

    if not bboxes:
        return [], [], []
    
    label_to_id = {label: idx for idx, label in enumerate(custom_labels_list)}  # Map custom labels to indices

    # Convert detected labels to class IDs using the custom label mapping
    class_ids = [label_to_id[label] for label in labels if label in label_to_id]

    formatted_boxes = [[int(coord) for coord in box] for box in bboxes]
    
    scores = 'Not applicable' #Florence V2 does not give confidence level

    return formatted_boxes, scores, class_ids