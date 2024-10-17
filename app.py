import streamlit as st

# Set page configuration


home= st.Page("pages/Home/home.py",title='Computer Vision Annotation Tool',icon=":material/home:",default=True)

#Zero Shot
zero_shot_object_detection= st.Page("pages/Zero Shot Models/1_Object_Detection.py",title='Object Detection',icon=":material/filter_center_focus:")
zero_shot_zone= st.Page("pages/Zero Shot Models/2_Zone_Based_Object_Detection.py",title='Zone Based Object Detection',icon=":material/detection_and_zone:")
zero_shot_tracking= st.Page("pages/Zero Shot Models/3_Object_Tracking.py",title='Object Tracking',icon=":material/videocam:")
zero_shot_clasification=st.Page("pages/Zero Shot Models/4_Image_Classification.py",title='Image Classification',icon=":material/label_important:")

#Pretrained
pretrained_license_plate= st.Page("pages/Pre-Trained Models/1_License_Plate_Text_Detection.py",title='License Plate Text Detection',icon=":material/credit_card:")
pre_traines_vehicle_detection= st.Page("pages/Pre-Trained Models/2_Vehicle_Detection.py",title='Vehicle Detection',icon=":material/garage:")


pg= st.navigation({
    'Home':[home],
    "Zero Shot Models":[zero_shot_object_detection,zero_shot_zone,zero_shot_tracking,zero_shot_clasification],
    "Pretrained Models":[pretrained_license_plate,pre_traines_vehicle_detection]
})

pg.run()


