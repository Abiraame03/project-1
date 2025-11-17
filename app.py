import streamlit as st
from PIL import Image
import numpy as np
import cv2
import pyttsx3
import json
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# =========================
# 1️⃣ App Configuration
# =========================
st.set_page_config(page_title="Dyslexia Detection", page_icon="📝")
st.title("Dyslexia Detection in Children (6yr+)")

# =========================
# 2️⃣ Load Model & Thresholds
# =========================
model_path = "mobilenetv2_bilstm_best_thr_044.h5"
threshold_path = "best_threshold.json"
class_map_path = "class_indices_best.pkl"

model = load_model(model_path)
with open(threshold_path, "r") as f:
    thresholds = json.load(f)
with open(class_map_path, "rb") as f:
    class_map = pickle.load(f)

# Invert class_map for easy lookup
class_map = {v: k for k, v in class_map.items()}

# =========================
# 3️⃣ Text-to-Speech Setup
# =========================
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speech rate

def speak(text):
    engine.say(text)
    engine.runAndWait()

# =========================
# 4️⃣ Severity Mapping
# =========================
def get_severity(prob):
    if prob < thresholds['mild']:
        return "Normal"
    elif prob < thresholds['moderate']:
        return "Mild"
    elif prob < thresholds['severe']:
        return "Moderate"
    else:
        return "Severe"

# =========================
# 5️⃣ Prediction Function
# =========================
def predict_handwriting(image: Image.Image):
    image = image.convert('RGB').resize((128, 128))
    x = img_to_array(image) / 255.0
    x = np.expand_dims(x, axis=0)
    
    prob = model.predict(x)[0][0]  # Assuming binary output
    severity = get_severity(prob)
    
    # Generate voice message for abnormal patterns
    features_msg = ""
    if severity != "Normal":
        features_msg = (
            "Observed abnormal handwriting features: inconsistent strokes, "
            "irregular spacing, reversed letters, and spelling mistakes."
        )
        speak(f"Prediction: {severity}. {features_msg}")
    else:
        speak("Prediction: Normal handwriting")
    
    return prob, severity, features_msg

# =========================
# 6️⃣ Streamlit Interface
# =========================
st.header("Capture Handwriting Sample")
st.write("Use your webcam to take a handwriting sample.")

# Camera input
img_file_buffer = st.camera_input("Take a picture of handwriting")

if img_file_buffer is not None:
    image = Image.open(img_file_buffer)
    st.image(image, caption="Captured Image", use_column_width=True)
    
    if st.button("Predict Dyslexia Severity"):
        prob, severity, features_msg = predict_handwriting(image)
        st.subheader(f"Severity: {severity}")
        st.write(f"Probability Score: {prob:.2f}")
        if features_msg:
            st.write(f"Features: {features_msg}")
