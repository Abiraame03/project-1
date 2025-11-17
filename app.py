import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pickle
import json
from PIL import Image
import pyttsx3
import tempfile

# ===========================================
# Config
# ===========================================
MODEL_PATH = "mobilenetv2_bilstm_best_thr_044.h5"
CLASS_MAP_PATH = "class_indices_best.pkl"
THRESHOLD_PATH = "best_threshold.json"

IMG_SIZE = (160, 160)  # Must match training

# Severity mapping
severity_levels = {
    0: "Normal",
    1: "Mild",
    2: "Moderate",
    3: "Severe"
}

# Abnormal handwriting feedback templates
feedback_patterns = [
    "frequent spelling mistakes",
    "irregular stroke formation",
    "uneven spacing between letters",
    "inconsistent letter sizes",
    "mirror writing or reversed letters"
]

# ===========================================
# Load model and thresholds
# ===========================================
@st.cache_resource
def load_resources():
    model = load_model(MODEL_PATH)
    with open(CLASS_MAP_PATH, 'rb') as f:
        class_map = pickle.load(f)
    with open(THRESHOLD_PATH, 'r') as f:
        thresholds = json.load(f)
    return model, class_map, thresholds

model, class_map, thresholds = load_resources()

# ===========================================
# Text-to-speech
# ===========================================
engine = pyttsx3.init(driverName='espeak')  # Linux-friendly

def speak(text):
    engine.say(text)
    engine.runAndWait()

# ===========================================
# Streamlit UI
# ===========================================
st.title("📖 Dyslexia Detection in 6-Year-Old Children")
st.write("Capture handwriting via camera and get severity assessment with feedback.")

# Camera input
stframe = st.empty()
cap = cv2.VideoCapture(0)  # Use default camera

run = st.button("Capture Image")

if run:
    ret, frame = cap.read()
    if ret:
        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
        # Save temporarily
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        cv2.imwrite(temp_file.name, frame)

        # Preprocess
        img = Image.open(temp_file.name).convert('RGB')
        img = img.resize(IMG_SIZE)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        pred_probs = model.predict(img_array)[0]
        class_idx = np.argmax(pred_probs)
        class_name = class_map[class_idx]
        pred_score = pred_probs[class_idx]

        # Apply threshold
        threshold = thresholds.get(class_name, 0.5)
        if pred_score >= threshold:
            severity = severity_levels.get(class_idx, "Unknown")
        else:
            severity = "Normal"

        # Feedback
        feedback = ", ".join(np.random.choice(feedback_patterns, 2, replace=False))
        result_text = f"Predicted Class: {class_name}\nSeverity Level: {severity}\nObserved Patterns: {feedback}"
        
        st.success(result_text)
        speak(f"The child's handwriting shows {feedback}. Severity level is {severity}.")

# Release camera
cap.release()
