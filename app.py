import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import json
import pickle

# --------------------------
# Load model & config
# --------------------------
MODEL_PATH = "models/mobilenetv2_bilstm_best_thr_044.h5"
THRESH_PATH = "models/best_threshold.json"
CLASS_INDICES_PATH = "models/class_indices_best.pkl"

model = load_model(MODEL_PATH)

with open(THRESH_PATH, "r") as f:
    best_threshold = json.load(f).get("threshold", 0.5)

with open(CLASS_INDICES_PATH, "rb") as f:
    class_indices = pickle.load(f)

# Reverse mapping
idx_to_class = {v: k for k, v in class_indices.items()}

# --------------------------
# Streamlit UI
# --------------------------
st.title("🧠 Dyslexia Detection")
st.write("Upload an image of text reading or handwriting sample.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = load_img(uploaded_file, target_size=(128, 128))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    pred_prob = float(model.predict(img_array)[0][0])
    confidence = pred_prob if pred_prob >= 0 else 1 + pred_prob
    prediction = "Dyslexic" if pred_prob >= best_threshold else "Normal"

    # Severity
    confidence_percent = confidence * 100
    if prediction == "Normal":
        severity = "Normal (0)"
    elif confidence_percent <= 30:
        severity = "Mild (5-30)"
    elif confidence_percent <= 70:
        severity = "Moderate (30-70)"
    else:
        severity = "Severe (>70)"

    # Display
    st.markdown(f"**Prediction:** {prediction}")
    st.markdown(f"**Confidence:** {confidence_percent:.2f}%")
    st.markdown(f"**Severity:** {severity}")
