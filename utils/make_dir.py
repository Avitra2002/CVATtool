import zipfile
import os

def create_task_folders(files_dir, task_name):
    """Create the main task folder with subfolders and return the path."""

    folder_name = f"{task_name.replace(' ', '_')}"
    folder_path = os.path.join(files_dir, folder_name)

    # Create the main folder and subfolders
    os.makedirs(os.path.join(folder_path, "images"), exist_ok=True)
    os.makedirs(os.path.join(folder_path, "labels"), exist_ok=True)
    os.makedirs(os.path.join(folder_path, "annotated_images"), exist_ok=True)

    return folder_path

def save_images(files_dir, uploaded_files, task_name):
    """Save uploaded images to the 'images' subfolder under the task folder and return the folder path."""
    folder_path = create_task_folders(files_dir, task_name)  
    images_subfolder = os.path.join(folder_path, "images")

    for i, uploaded_file in enumerate(uploaded_files):
        img_extension = uploaded_file.name.split('.')[-1]  
        img_path = os.path.join(images_subfolder, f"image_{i+1}.{img_extension}")  
        
        # Save image to the images subfolder
        with open(img_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    
    return folder_path

def save_and_extract_zip(files_dir, uploaded_file, task_name):
    """Save and extract a zip file to the 'images' subfolder under the task folder and return the folder path."""
    folder_path = create_task_folders(files_dir, task_name)  
    zip_path = os.path.join(folder_path, uploaded_file.name)

    # Save the uploaded zip file
    with open(zip_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Extract the zip file to the images subfolder
    images_subfolder = os.path.join(folder_path, "images")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(images_subfolder)

   
    return folder_path

#TODO: Process video as image frames for object detection