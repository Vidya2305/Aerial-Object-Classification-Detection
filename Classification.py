import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

@st.cache_resource
def load_classification_model():
    model = load_model("best_model_for_streamlit.h5")
    return model

classifier = load_classification_model()

st.set_page_config(page_title="Bird vs Drone Classifier", page_icon="🕊️", layout="centered")

st.title("🕊️ Bird vs Drone Image Classifier")
st.markdown("Upload an image to predict whether it's a **Bird** or a **Drone**.")

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

        st.subheader("🧠 Prediction Result")
        st.success(f"**Class:** {pred_class}")
        st.info(f"**Confidence:** {confidence}%")

        if pred_class == "Bird":
            st.balloons()
        else:
            st.snow()

else:
    st.warning("Please upload an image to continue.")






