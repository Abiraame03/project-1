import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pickle
import json
import requests
from io import BytesIO

# ==============================
# 1️⃣ Config: Model Paths
# ==============================
MODEL_URL = "https://github.com/Abiraame03/project-1/raw/main/models/mobilenetv2_bilstm_final.h5"
CLASS_MAP_URL = "https://github.com/Abiraame03/project-1/raw/main/models/class_indices.pkl"
THRESHOLD_URL = "https://github.com/Abiraame03/project-1/raw/main/models/best_threshold.json"

IMG_SIZE = (160, 160)

@st.cache_resource
def load_model_files():
    # Load model
    model_path = tf.keras.utils.get_file("mobilenetv2_bilstm_final.h5", MODEL_URL)
    model = tf.keras.models.load_model(model_path)
    
    # Load class map
    class_map_path = tf.keras.utils.get_file("class_indices.pkl", CLASS_MAP_URL)
    with open(class_map_path, "rb") as f:
        class_indices = pickle.load(f)
    inv_map = {v:k for k,v in class_indices.items()}
    
    # Load threshold
    threshold_path = tf.keras.utils.get_file("best_threshold.json", THRESHOLD_URL)
    with open(threshold_path, "r") as f:
        THRESHOLD = json.load(f)["threshold"]
    
    return model, inv_map, THRESHOLD

model, inv_map, THRESHOLD = load_model_files()

# ==============================
# 2️⃣ Streamlit UI
# ==============================
st.title("📝 Dyslexia Detection from Handwriting/Image")

uploaded_file = st.file_uploader("Upload handwriting image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess image
    img_resized = img.resize(IMG_SIZE)
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    prob = model.predict(img_array)[0][0]
    label = 1 if prob >= THRESHOLD else 0
    class_name = inv_map[label]
    confidence = float(prob) * 100
    
    # Severity levels
    if confidence < 5:
        severity = "Normal (0)"
    elif 5 <= confidence < 30:
        severity = "Mild (5-30)"
    elif 30 <= confidence <= 70:
        severity = "Moderate (30-70)"
    else:
        severity = "Severe (>70)"
    
    # Display results
    st.subheader("Prediction Results")
    st.write(f"**Class:** {class_name}")
    st.write(f"**Confidence:** {confidence:.2f}%")
    st.write(f"**Severity:** {severity}")
