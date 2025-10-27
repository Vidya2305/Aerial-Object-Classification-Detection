#📌 Streamlit Deployment
#● Create a simple UI with image upload
#● Display prediction (Bird / Drone) & confidence score
#● (Optional) Show YOLOv8 detection results with bounding boxes

import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import os
from ultralytics import YOLO
from pathlib import Path

@st.cache_resource
def load_classification_model():
    model = load_model("best_model_for_streamlit.h5")
    return model

@st.cache_resource
def load_yolo_model():
    return YOLO("runs/yolov8_cpu_training/weights/best.pt")

classifier = load_classification_model()
yolo_model = load_yolo_model()

st.set_page_config(page_title="Bird vs Drone Classifier and Detector", page_icon="🕊️", layout="centered")
st.set_page_config(page_title="Bird vs Drone Classifier and Detector", page_icon="🕊️", layout="centered")

with st.sidebar:
    with st.expander("ℹ️ About", expanded=False):
        st.markdown("""
        This app performs **Bird vs Drone** classification and detection using:

        1. 🧠 **MobileNetV2 Model** — classifies the uploaded image as a *Bird* or *Drone*.  
        2. 🎯 **YOLOv8 Model** — detects objects and draws **bounding boxes** around them.

        **Usage:**
        1. Upload an image (.jpg/.png/.jpeg).  
        2. Click **Predict 🕊️/🚁**.  
        3. View both **classification** and **detection** results!
        """)

st.title("🕊️ Bird vs Drone Image Classifier and Object Detector")
st.markdown("Upload an image to predict whether it's a **Bird** or a **Drone** and see YOLOv8 **object detection results** with bounding boxes.")

uploaded_file = st.file_uploader("📁 Upload an image (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict 🕊️/🚁"):
        img_resized = img.resize((224, 224))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        prediction = classifier.predict(img_array)

        pred = prediction[0][0]

        pred_class = "Drone" if pred > 0.5 else "Bird"
        confidence = round(pred * 100 if pred_class == "Drone" else (1 - pred) * 100, 2)

        st.subheader("🧠 Classification Result")
        st.success(f"**Class:** {pred_class}")
        st.info(f"**Confidence:** {confidence}%")

        if pred_class == "Bird":
            st.balloons()
        else:
            st.snow()

        st.subheader("🎯 YOLOv8 Object Detection Result")
        temp_path = "temp_uploaded_image.jpg"
        img.save(temp_path)

        results = yolo_model.predict(source=temp_path, conf=0.4, save=True, show=False, device="cpu")

        image_name = Path(results[0].path).name
        result_image_path = Path(results[0].save_dir) / image_name

        st.image(result_image_path, caption="YOLOv8 Detection with Bounding Boxes", use_container_width=True)

        os.remove(temp_path)

else:
    st.warning("Please upload an image to continue.")








