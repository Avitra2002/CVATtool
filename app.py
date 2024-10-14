import streamlit as st
from utils.homepage_chatbot import display_chatbot

# Set page configuration
st.set_page_config(page_title='Computer Vision Annotation Tool', layout='wide')


col1, col2 = st.columns([0.7, 0.3])


with col1:
    # Introduction Page
    st.title("Welcome to the Computer Vision Annotation Tool")
    st.write("""
        This tool helps you create labeled datasets for various computer vision tasks using zero-shot Large Language Models (LLMs) like phi 3.5, Florence v2, OWL-ViT v2, YOLO World, SAM, etc.
    """)
    st.write("Please select a task from the sidebar to get started.")


with col2:
    
    chat_container = st.container(height=700)  
    with chat_container:
        display_chatbot()



