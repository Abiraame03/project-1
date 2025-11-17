import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import pickle
import json
import pyttsx3

# Page setup
st.set_page_config(page_title="Dyslexia Detection", layout="centered")
st.title("Dyslexia Detection System 🧠")

# Initialize TTS engine
engine = pyttsx3.init()

# --- Load Models and resources ---
@st.cache_resource
def load_resources():
    model = load_model("models/mobilenetv2_bilstm_best_thr_044.h5")
    
    with open("models/class_indices_best.pkl", "rb") as f:
        class_indices = pickle.load(f)

    with open("models/best_threshold.json", "r") as f:
        thresholds = json.load(f)

    return model, class_indices, thresholds

model, class_indices, thresholds = load_resources()

# --- File uploader ---
uploaded_file = st.file_uploader("Upload sentence image (png/jpg)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img_array = img_to_array(image.resize((128,128))) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    pred_probs = model.predict(img_array)[0]
    pred_idx = int(np.argmax(pred_probs))
    pred_class = class_indices[pred_idx]
    confidence = float(pred_probs[pred_idx]) * 100

    # Determine severity based on threshold
    threshold = thresholds.get(pred_class, 0.5)
    severity = "Low"
    if confidence > threshold*100 + 20:
        severity = "High"
    elif confidence > threshold*100:
        severity = "Medium"

    st.success(f"Prediction: **{pred_class}** ({confidence:.2f}% confidence)")
    st.info(f"Dyslexia Severity: **{severity}**")

    # Speak result
    engine.say(f"The system predicts {pred_class} with {confidence:.2f} percent confidence. Severity: {severity}")
    engine.runAndWait()
