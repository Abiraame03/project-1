import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import requests
import io
import json
import pickle

# --------------------------
# CONFIG
# --------------------------
st.set_page_config(page_title="Dyslexia Detection", page_icon="🧠", layout="centered")
st.title("🧠 Dyslexia Detection from Handwriting")

IMG_SIZE = (160, 160)

# --------------------------
# LOAD MODEL & THRESHOLD & CLASS MAP from GitHub
# --------------------------
MODEL_URL = "https://github.com/Abiraame03/project-1/raw/main/models/mobilenetv2_bilstm_final.h5"
THRESHOLD_URL = "https://github.com/Abiraame03/project-1/raw/main/models/best_threshold.json"
CLASS_MAP_URL = "https://github.com/Abiraame03/project-1/raw/main/models/class_indices_best.pkl"

@st.cache_resource
def load_model_from_github():
    # Load model
    model_path = tf.keras.utils.get_file("model.h5", MODEL_URL)
    model = tf.keras.models.load_model(model_path)
    
    # Load threshold
    threshold_path = tf.keras.utils.get_file("threshold.json", THRESHOLD_URL)
    with open(threshold_path, "r") as f:
        threshold = json.load(f)["threshold"]
    
    # Load class map
    class_map_path = tf.keras.utils.get_file("class_map.pkl", CLASS_MAP_URL)
    with open(class_map_path, "rb") as f:
        class_indices = pickle.load(f)
    inv_map = {v: k for k, v in class_indices.items()}
    
    return model, threshold, inv_map

model, THRESHOLD, inv_map = load_model_from_github()

# --------------------------
# PREDICTION FUNCTION
# --------------------------
def predict(img: Image.Image):
    img_resized = img.resize(IMG_SIZE)
    arr = np.array(img_resized) / 255.0
    arr = np.expand_dims(arr, axis=0).astype(np.float32)
    
    prob = float(model.predict(arr)[0][0])
    label = 1 if prob > THRESHOLD else 0
    class_name = inv_map[label]
    confidence = prob * 100
    
    # Severity based on confidence
    if confidence < 5:
        severity = "Normal (0)"
    elif 5 <= confidence < 30:
        severity = "Mild (5-30)"
    elif 30 <= confidence <= 70:
        severity = "Moderate (30-70)"
    else:
        severity = "Severe (>70)"
    
    return class_name, confidence, severity

# --------------------------
# STREAMLIT UI
# --------------------------
uploaded_file = st.file_uploader("Upload a child’s handwriting image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Predict"):
        with st.spinner("Analyzing..."):
            class_name, confidence, severity = predict(image)
        
        st.success("✅ Prediction Completed!")
        st.subheader("Prediction Results")
        st.write(f"**Prediction:** {class_name}")
        st.write(f"**Confidence:** {confidence:.2f}%")
        st.write(f"**Severity:** {severity}")
        st.progress(min(int(confidence), 100))
