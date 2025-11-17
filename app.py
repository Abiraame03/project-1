# ================================================================
# 1️⃣ IMPORTS
# ================================================================
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import pickle
import json

# ================================================================
# 2️⃣ APP TITLE
# ================================================================
st.set_page_config(page_title="Dyslexia Detection", layout="centered")
st.title("🧠 Dyslexia Detection & Severity Prediction")

# ================================================================
# 3️⃣ LOAD MODEL & THRESHOLD
# ================================================================
@st.cache_resource
def load_model_files():
    MODEL_PATH = "models/mobilenetv2_bilstm_final.h5"  # already in repo
    CLASS_MAP_PATH = "models/class_indices.pkl"        # already in repo
    THRESHOLD_PATH = "models/best_threshold.json"      # already in repo

    # Load model
    model = tf.keras.models.load_model(MODEL_PATH)

    # Load class mapping
    with open(CLASS_MAP_PATH, "rb") as f:
        class_indices = pickle.load(f)
    inv_map = {v: k for k, v in class_indices.items()}

    # Load threshold
    with open(THRESHOLD_PATH, "r") as f:
        threshold = json.load(f)["threshold"]

    return model, inv_map, threshold

model, inv_map, THRESHOLD = load_model_files()

IMG_SIZE = (160, 160)

# ================================================================
# 4️⃣ PREDICTION FUNCTION
# ================================================================
def predict_image(image):
    img = Image.open(image).convert("RGB")
    img_resized = img.resize(IMG_SIZE)

    arr = np.array(img_resized) / 255.0
    arr = np.expand_dims(arr, axis=0)

    prob = float(model.predict(arr)[0][0])
    label = 1 if prob > THRESHOLD else 0
    class_name = inv_map[label]

    # Severity calculation
    if prob <= 0.05:
        severity = "Normal (0)"
    elif 0.05 < prob <= 0.3:
        severity = "Mild (5-30)"
    elif 0.3 < prob <= 0.7:
        severity = "Moderate (30-70)"
    else:
        severity = "Severe (>70)"

    return class_name, prob, severity

# ================================================================
# 5️⃣ STREAMLIT UI
# ================================================================
uploaded_file = st.file_uploader("Upload an image of the child for prediction", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("Predicting...")

    class_name, prob, severity = predict_image(uploaded_file)

    st.subheader("Prediction Result")
    st.write(f"**Class:** {class_name}")
    st.write(f"**Confidence:** {prob*100:.2f}%")
    st.write(f"**Severity:** {severity}")
