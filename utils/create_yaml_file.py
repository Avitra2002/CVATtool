import streamlit as st
import yaml
import os

def create_yaml_file(task_folder_path, custom_labels_list):
    yaml_data = {
        'path': '../datasets/coco8',  # Update this to your actual dataset path
        'train': 'images/train',       # Update as needed
        'val': 'images/val',           # Update as needed
        'test': None,                  # Optional
        'names': {i: label for i, label in enumerate(custom_labels_list)}  # Create a mapping
    }

    yaml_file_path = os.path.join(task_folder_path, 'labels.yaml')
    with open(yaml_file_path, 'w') as yaml_file:
        yaml.dump(yaml_data, yaml_file)

    st.success(f"YAML file created at: {yaml_file_path}")