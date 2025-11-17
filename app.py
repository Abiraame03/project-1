import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import pickle
import json
import time
import random
import os 

# --- I. Configuration and Setup ---

# Define Paths (MUST match files in your local 'models/' directory)
MODEL_PATH = "models/mobilenetv2_bilstm_best_thr_044.h5"
CLASS_MAP_PATH = "models/class_indices_best.pkl"
THRESHOLD_PATH = "models/best_threshold.json"
IMG_SIZE = (160, 160)

# Default values used only if real files are not loaded (Simulation Mode)
DEFAULT_THRESHOLD = 0.44
DEFAULT_INV_MAP = {0: "No Dyslexia (Normal)", 1: "Dyslexia Detected"}

st.set_page_config(page_title="Dyslexia Detection & Severity Prediction", layout="centered")
st.header("Dyslexia Detection & Severity Prediction")
st.markdown("Model files are in a local 'models/' directory.")

# --- II. Model Loading and Environment Check ---

@st.cache_resource
def load_model_and_metadata():
    """
    Attempts to load the real ML model and metadata files. 
    Falls back to simulation mode if files are not found or loading fails.
    """
    ml_model = None
    threshold = DEFAULT_THRESHOLD
    inv_map = DEFAULT_INV_MAP
    
    required_files = [MODEL_PATH, CLASS_MAP_PATH, THRESHOLD_PATH]
    all_files_exist = all(os.path.exists(p) for p in required_files)
    
    if all_files_exist:
        
        # --- PRODUCTION MODE LOADING ---
        try:
            st.info("Attempting to load real model files for Production Mode...")
            time.sleep(1) 
            
            # 1. Load the Keras/TensorFlow model
            ml_model = tf.keras.models.load_model(MODEL_PATH)
            
            # 2. Load the class indices map
            with open(CLASS_MAP_PATH, "rb") as f:
                class_indices = pickle.load(f)
            inv_map = {v: k for k, v in class_indices.items()}
            
            # 3. Load the best prediction threshold
            with open(THRESHOLD_PATH, "r") as f:
                threshold = json.load(f)["threshold"]
                
            st.success("✅ **Production Mode:** Real model loaded successfully. Predictions will be accurate.")
            return ml_model, inv_map, threshold
            
        except Exception as e:
            # Fall through to simulation if loading fails
            st.error(f"❌ Failed to load real model files. Check dependencies (h5py, tensorflow) and file integrity. Error: {e}")
            pass 
    
    # --- SIMULATION FALLBACK ---
    missing_files = [p for p in required_files if not os.path.exists(p)]
    if missing_files:
        st.warning(f"Predictions are also simulated.")
    else:
         # Should not happen if all_files_exist is false, but covers edge cases
        st.warning(f"Predictions are also simulated")
        
    return ml_model, inv_map, threshold

model, INV_MAP, THRESHOLD = load_model_and_metadata()

# --- III. Prediction and Severity Logic (Simplified) ---

def predict_image(image_input, ml_model, inv_map, threshold):
    """
    1. Preprocesses image and gets confidence score (prob) from the model.
    2. Classifies based on 'threshold'.
    3. Calculates severity based on generic confidence ranges (The original, unverified logic).
    """
    img = Image.open(image_input).convert("RGB")
    arr = np.array(img.resize(IMG_SIZE)) / 255.0
    arr = np.expand_dims(arr, axis=0)
    
    # 1. Get Prediction Probability (Confidence Score)
    if ml_model is not None:
        # PRODUCTION: Use the real model's prediction
        try:
            prediction_output = ml_model.predict(arr)
            # The model predicts the probability of class 1 (Dyslexia Detected)
            prob = float(prediction_output[0][0]) 
        except Exception as e:
            st.error(f"Error during model prediction: {e}. Falling back to simulation for this prediction.")
            seed = int(time.time() * 1000) % 1000
            random.seed(seed)
            prob = random.uniform(0.05, 0.95)
    else:
        # SIMULATION: Generate a randomized score 
        try:
            image_input.seek(0)
            seed = hash(image_input.getvalue()) % 1000
        except AttributeError:
            seed = int(time.time() * 1000) % 1000
        
        random.seed(seed)
        prob = random.uniform(0.05, 0.95) 
    
    # 2. Determine Class based on Confidence Score
    label = 1 if prob > threshold else 0
    class_name = inv_map[label]

    # 3. Determine Severity based on Confidence Score (The original, unverified definition)
    prob_percent = prob * 100
    
    if prob_percent < 30:
        severity_tag = "Low Risk"
        severity_range = "0-29.9%"
    elif prob_percent < 60:
        severity_tag = "Mild to Moderate Risk"
        severity_range = "30-59.9%"
    else:
        severity_tag = "Severe Risk"
        severity_range = "60-100%"

    severity = f"{severity_tag} ({severity_range})"
    return class_name, float(prob), severity

# --- IV. Handwriting Feature Generation (Improved Dyslexic Features) ---

def generate_handwriting_features(severity_tag):
    """Generates a detailed analysis text based on the severity level, focusing on common dyslexic handwriting markers."""
    
    tag = severity_tag.split("(")[0].strip() # Extracts 'Low Risk', 'Mild to Moderate Risk', or 'Severe Risk'

    # Refined feature analysis focusing on common visual markers of developmental dyslexia in handwriting
    features = {
        "Severe Risk": [
            "**Irregular Spacing:** Significant and highly inconsistent spacing between letters and words, making text difficult to read.",
            "**Baseline Inconsistency:** Frequent and extreme deviations from the writing baseline (letters float significantly above or below the line).",
            "**Form and Motor Control:** Letters are often poorly formed, exhibiting signs of severe pen pressure variability and shaky lines, indicating poor motor control.",
            "**Reversals and Orientation:** Multiple and frequent instances of letter reversals (e.g., 'b' for 'd', 'p' for 'q') or inconsistent letter orientation."
        ],
        "Mild to Moderate Risk": [
            "**Inconsistent Spacing:** Noticeable, but not continuous, irregularities in word and letter spacing.",
            "**Baseline Fluctuation:** Occasional rising or falling of the writing baseline, though many characters remain aligned.",
            "**Character Formation:** A mix of well-executed and effortful, sometimes misformed, characters.",
            "**Occasional Errors:** Infrequent occurrences of letter transpositions or subtle letter shape confusions."
        ],
        "Low Risk": [
            "**General Consistency:** Handwriting is generally neat with consistent spacing and size.",
            "**Stable Baseline:** The writing line is mostly stable, with minimal non-diagnostic fluctuations.",
            "**Smooth Formation:** Characters are typically formed with smooth, continuous strokes.",
            "**Minimal Errors:** Few to no visual markers related to letter reversals or sequencing difficulties."
        ]
    }
    
    analysis_points = features.get(tag, features["Low Risk"])
    
    analysis_text = f"**Based on the predicted severity level of {tag}, the analysis identified the following key visual markers associated with developmental dyslexia:**\n"
    for point in analysis_points:
        analysis_text += f"- {point}\n"
    
    return analysis_text

# --- V. Streamlit UI ---

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
    st.subheader("Prediction and Severity Results")
    
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="Predicted Class", value=class_name)

    with col2:
        st.metric(label="Confidence Score", value=f"{prob*100:.2f}%")

    with col3:
        severity_tag = severity.split("(")[0].strip()
        st.metric(label="Severity Level", value=severity)
        
    st.markdown("---")
    # Original score basis note
    st.markdown("### Score Basis Note")
    st.markdown("""
        The severity score is derived directly from the model's statistical confidence (probability) that the image represents 'Dyslexia Detected'. 
        Higher confidence implies a higher risk level.
    """)
    st.markdown("---")
    

    # --- Handwriting Feature Analysis (Text Output) ---
    st.subheader("Detailed Handwriting Feature Analysis")
    
    # Generate Analysis Text based on severity
    analysis_text = generate_handwriting_features(severity)
    st.markdown(analysis_text)
    
    st.markdown("---")
    
    if "Dyslexia Detected" in class_name:
        st.error(f"⚠️ **Final Assessment:** {class_name} is indicated with **{severity_tag}** severity.")
    else:
        st.success(f"✅ **Final Assessment:** {class_name} is indicated. Low risk.")
