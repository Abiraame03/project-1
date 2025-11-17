import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import pickle
import json
import time
import io

st.set_page_config(page_title="Dyslexia Detection & Severity Prediction", layout="centered")

# --- Configuration Constants ---
# NOTE: These paths reference where the real model files would be located.
MODEL_PATH = "models/mobilenetv2_bilstm_final.h5"
CLASS_MAP_PATH = "models/class_indices.pkl"
THRESHOLD_PATH = "models/best_threshold.json"
IMG_SIZE = (160, 160)

# --- Model Loading Structure (Simulated for Single-File Deployment) ---

@st.cache_resource
def load_model_and_metadata():
    """
    Simulated loading of the ML model and metadata files.
    
    FOR PRODUCTION: To enable real prediction, you must uncomment the 'REAL LOADING' 
    block below and ensure your 'models/' folder is deployed with the files.
    """
    
    # Display warning about simulation mode
    st.warning(f"🚨 **Simulation Mode Active:** The actual model files are not included. Prediction results are **simulated** using a consistent random score based on the image content.")

    # --- SIMULATION VALUES ---
    time.sleep(1) # Simulate load time
    ml_model = None 
    threshold = 0.5
    inv_map = {0: "No Dyslexia (Normal)", 1: "Dyslexia Detected"}
    
    # --- REAL LOADING (UNCOMMENT THIS BLOCK FOR PRODUCTION USE) ---
    # try:
    #     ml_model = tf.keras.models.load_model(MODEL_PATH)
    #     with open(CLASS_MAP_PATH, "rb") as f:
    #         class_indices = pickle.load(f)
    #     inv_map = {v: k for k, v in class_indices.items()}
    #     with open(THRESHOLD_PATH, "r") as f:
    #         threshold = json.load(f)["threshold"]
    #     st.success("Model structure loaded successfully for production use.")
    # except Exception:
    #     ml_model = None

    return ml_model, inv_map, threshold

# Load the model structure (either real or simulated)
model, INV_MAP, THRESHOLD = load_model_and_metadata()

# --- Prediction Logic ---

def predict_image(image_input, ml_model, inv_map, threshold):
    """
    Preprocesses the image, gets a prediction probability (confidence score), 
    and determines the class and severity based on that score.
    """
    # Image preprocessing
    img = Image.open(image_input).convert("RGB")
    img_resized = img.resize(IMG_SIZE)
    arr = np.array(img_resized) / 255.0
    arr = np.expand_dims(arr, axis=0)
    
    # --- Get Prediction Probability (Confidence Score) ---
    if ml_model is not None:
        # REAL PREDICTION: Use the actual model
        prob = float(ml_model.predict(arr)[0][0])
    else:
        # SIMULATION FALLBACK: Use deterministic random probability
        # Generate a seed based on image content for consistency when possible
        try:
            # For File Uploader, use hash of content
            image_input.seek(0)
            seed = hash(image_input.getvalue()) % 1000
        except AttributeError:
            # For Camera Input (or if file.getvalue() fails), use time
            seed = int(time.time() * 1000) % 1000
        
        np.random.seed(seed)
        prob = np.random.rand() * 0.9 + 0.05 # Range 5% to 95%
    
    # 1. Determine Class based on Confidence Score
    label = 1 if prob > threshold else 0
    class_name = inv_map[label]

    # 2. Determine Severity based on Confidence Score and Specified Ranges
    # This logic implements your requirement for severity calculation.
    if prob <= 0.05:
        severity = "Normal (0-5%)"
    elif 0.05 < prob <= 0.3:
        severity = "Mild (5-30%)"
    elif 0.3 < prob <= 0.7:
        severity = "Moderate (30-70%)"
    else:
        severity = "Severe (>70%)"

    return class_name, float(prob), severity

# --- Streamlit UI ---

st.header("🧠 Dyslexia Detection & Severity Prediction")
st.markdown("Use your device camera or upload a behavioral image (e.g., drawing or writing sample) for analysis.")

# Input Section
col_camera, col_upload = st.columns(2)
with col_camera:
    # 1. Camera Input
    camera_file = st.camera_input("Take a Photo for Prediction")
with col_upload:
    # 2. File Uploader Input
    uploaded_file = st.file_uploader(
        "Or Upload a File", 
        type=["jpg", "jpeg", "png"]
    )

# Determine the file to process
processed_file = camera_file if camera_file is not None else uploaded_file

if processed_file:
    st.image(processed_file, caption="Input Image", use_column_width=True)
    
    # Run prediction/simulation
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
        st.metric(label="Severity Level", value=severity)
        
    st.markdown("---")
    
    if class_name == INV_MAP[1]:
        st.error(f"⚠️ **Result:** {class_name} is indicated with **{severity}** severity.")
        st.info("The severity level is directly determined by the model's confidence score falling within the specified ranges.")
    else:
        st.success(f"✅ **Result:** {class_name} is indicated. Low risk.")
