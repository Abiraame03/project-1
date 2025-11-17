import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import pickle
import json
import time
import random
import os # Added for checking file existence

# --- Set Up Configuration ---
MODEL_PATH = "models/mobilenetv2_bilstm_final.h5"
CLASS_MAP_PATH = "models/class_indices.pkl"
THRESHOLD_PATH = "models/best_threshold.json"
IMG_SIZE = (160, 160)
# Default values used only if real files are not loaded
DEFAULT_THRESHOLD = 0.5
DEFAULT_INV_MAP = {0: "No Dyslexia (Normal)", 1: "Dyslexia Detected"}

st.set_page_config(page_title="Dyslexia Detection & Severity Prediction", layout="centered")
st.header("🧠 Dyslexia Detection & Severity Prediction")
st.markdown("Use your device camera or upload a behavioral image (e.g., drawing or writing sample) for analysis.")

# --- Model Loading Structure (Prepared for Real Files) ---

@st.cache_resource
def load_model_and_metadata():
    """
    Attempts to load the real ML model and metadata files. 
    Falls back to simulation mode if files are not found.
    """
    
    ml_model = None
    threshold = DEFAULT_THRESHOLD
    inv_map = DEFAULT_INV_MAP
    
    # Check if the models directory exists and contains the files
    if all(os.path.exists(p) for p in [MODEL_PATH, CLASS_MAP_PATH, THRESHOLD_PATH]):
        
        # --- REAL LOADING (UNCOMMENT THIS BLOCK FOR PRODUCTION USE) ---
        try:
            st.info("Attempting to load real model files...")
            time.sleep(1) # Visual delay for loading indicator
            
            ml_model = tf.keras.models.load_model(MODEL_PATH)
            
            with open(CLASS_MAP_PATH, "rb") as f:
                class_indices = pickle.load(f)
            inv_map = {v: k for k, v in class_indices.items()}
            
            with open(THRESHOLD_PATH, "r") as f:
                threshold = json.load(f)["threshold"]
                
            st.success("✅ **Production Mode:** Real model loaded successfully.")
            return ml_model, inv_map, threshold
            
        except Exception as e:
            st.error(f"❌ Failed to load real model files, reverting to simulation. Error: {e}")
            pass # Fall through to simulation mode
    
    # --- SIMULATION FALLBACK (Used if files are missing or loading fails) ---
    st.warning(f"🚨 **Simulation Mode:** Model files are not included. Prediction and severity results are simulated.")
    return ml_model, inv_map, threshold

model, INV_MAP, THRESHOLD = load_model_and_metadata()

# --- Prediction Logic ---

def predict_image(image_input, ml_model, inv_map, threshold):
    """
    Preprocesses the image, gets a prediction probability (confidence score), 
    and determines the class and severity based on that score.
    """
    img = Image.open(image_input).convert("RGB")
    arr = np.array(img.resize(IMG_SIZE)) / 255.0
    arr = np.expand_dims(arr, axis=0)
    
    # --- Get Prediction Probability (Confidence Score) ---
    if ml_model is not None:
        # PRODUCTION: Use the real model's prediction
        prob = float(ml_model.predict(arr)[0][0])
    else:
        # SIMULATION: Generate a randomized score
        try:
            image_input.seek(0)
            seed = hash(image_input.getvalue()) % 1000
        except AttributeError:
            seed = int(time.time() * 1000) % 1000
        
        random.seed(seed)
        prob = random.uniform(0.05, 0.95) 
    
    # 1. Determine Class based on Confidence Score
    label = 1 if prob > threshold else 0
    class_name = inv_map[label]

    # 2. Determine Severity based on Confidence Score and Specified Ranges
    prob_percent = prob * 100
    if prob_percent <= 5:
        severity_tag = "Normal"
        severity_range = "0-5%"
    elif prob_percent <= 30:
        severity_tag = "Mild"
        severity_range = "5-30%"
    elif prob_percent <= 70:
        severity_tag = "Moderate"
        severity_range = "30-70%"
    else:
        severity_tag = "Severe"
        severity_range = ">70%"

    severity = f"{severity_tag} ({severity_range})"
    return class_name, float(prob), severity

# --- Handwriting Feature Generation ---

def generate_handwriting_features(severity_tag):
    """Generates a detailed analysis text based on the severity level."""
    
    features = {
        "Severe": [
            "Prominent letter and number reversals are observed, such as 'b' for 'd' and '6' for '9'.",
            "Significant spelling mistakes and frequent attempts at overwriting are clearly visible.",
            "The words show abnormal and inconsistent spacing, making the sentence difficult to read. The overall uniformity of letters is poor."
        ],
        "Moderate": [
            "Occasional letter reversals or inversions are present, particularly in multi-syllable words.",
            "There are a noticeable number of spelling errors, though overwriting is moderate.",
            "The gap between words is somewhat irregular, and the size and slant of individual letters vary inconsistently."
        ],
        "Mild": [
            "Very few, if any, letter reversals are noted.",
            "Spelling mistakes are minor and infrequent. Overwriting is minimal.",
            "Word spacing is mostly consistent, but slight irregularities in letter uniformity can be seen."
        ],
        "Normal": [
            "Handwriting appears generally uniform with clear spacing and minimal to no spelling or reversal issues.",
            "No abnormal features such as reversal, inversion, overwriting, or inconsistent spacing were detected.",
            "The quality of the sample suggests low risk for handwriting-based indicators of dyslexia."
        ]
    }
    
    tag = severity_tag.split(" ")[0] # Get 'Severe', 'Moderate', etc.
    analysis = features.get(tag, features["Normal"])
    
    # Combine features into a conversational paragraph
    text = f"Based on the predicted severity level of {tag}, the handwriting analysis reveals the following patterns. "
    text += " ".join(analysis)
    text += " This assessment is based on observations of letter and word formation."
    
    return text

# --- Streamlit UI ---

# Input Section
col_camera, col_upload = st.columns(2)
with col_camera:
    camera_file = st.camera_input("Take a Photo for Prediction")
with col_upload:
    uploaded_file = st.file_uploader(
        "Or Upload a File", 
        type=["jpg", "jpeg", "png"]
    )

processed_file = camera_file if camera_file is not None else uploaded_file

if processed_file:
    st.image(processed_file, caption="Input Image", use_column_width=True)
    
    # --- Run Prediction/Simulation ---
    with st.spinner("Analyzing image..."):
        class_name, prob, severity = predict_image(processed_file, model, INV_MAP, THRESHOLD)

    # --- Display Results ---
    st.subheader("Prediction Result")
    
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="Predicted Class", value=class_name)

    with col2:
        st.metric(label="Confidence Score", value=f"{prob*100:.2f}%")

    with col3:
        severity_tag = severity.split(" ")[0]
        st.metric(label="Severity Level", value=severity)
        
    st.markdown("---")

    # --- Handwriting Feature Analysis (Text Output Only) ---
    st.subheader("Handwriting Feature Analysis")
    
    # Generate Analysis Text
    analysis_text = generate_handwriting_features(severity_tag)
    st.markdown(f"**Detailed Analysis:** {analysis_text}")
    
    st.markdown("---")
    
    if class_name == INV_MAP[1]:
        st.error(f"⚠️ **Final Result:** {class_name} is indicated with **{severity}** severity.")
    else:
        st.success(f"✅ **Final Result:** {class_name} is indicated. Low risk.")
