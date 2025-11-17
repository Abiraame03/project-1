import streamlit as st
import numpy as np
from PIL import Image
# Removed direct imports for tensorflow, pickle, and json as we are simulating the model for portability.

st.set_page_config(page_title="Dyslexia Detection Simulator", layout="centered")

# --- Constants for Simulation (Replaces external files) ---

# Hardcoded metadata to simulate class_indices.pkl
INV_MAP = {
    0: "No Dyslexia (Normal)",
    1: "Dyslexia Detected"
}

# Hardcoded threshold to simulate best_threshold.json
THRESHOLD = 0.5 
IMG_SIZE = (160, 160) # Model input size

# --- Model Loading Simulation ---

@st.cache_resource
def load_metadata_and_simulate_model():
    """
    Simulates loading the model and metadata. 
    A warning is displayed because the actual ML model (.h5) is not included.
    """
    st.info("⚠️ **Note:** The actual machine learning model files are not included in this single-file deployment. Prediction results are **simulated** using random probability to demonstrate the app's functionality and logic.")
    return INV_MAP, THRESHOLD

INV_MAP, THRESHOLD = load_metadata_and_simulate_model()

# --- Prediction Logic Simulation ---

def predict_image(image):
    """
    Simulates the prediction process instead of calling model.predict().
    It generates a random probability based on the image's content hash for consistent results.
    """
    # Open and resize the image for context (though not used in the simulation)
    img = Image.open(image).convert("RGB")
    img_resized = img.resize(IMG_SIZE)
    
    # 1. Generate a consistent random probability (0.0 to 1.0)
    # Hashing the image content ensures the result is the same for the same uploaded image
    try:
        # Use a seed based on image content for consistency
        image.seek(0)
        seed = hash(image.getvalue()) % 1000
    except Exception:
        # Fallback if image operations fail
        seed = 42
        
    np.random.seed(seed)
    prob = np.random.rand()

    # 2. Determine class and severity using hardcoded THRESHOLD
    label = 1 if prob > THRESHOLD else 0
    class_name = INV_MAP[label]

    # 3. Determine severity based on probability (same logic as original code)
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

st.header("Dyslexia Detection & Severity Prediction")
st.markdown("Upload a behavioral image (e.g., drawing, writing sample) for a simulated prediction.")

uploaded_file = st.file_uploader(
    "Upload an image of the child for prediction", 
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    # Run simulation
    with st.spinner("Simulating prediction..."):
        class_name, prob, severity = predict_image(uploaded_file)

    # Display results
    st.subheader("Simulated Prediction Result")
    
    # Use st.metric for clear display
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="Predicted Class", value=class_name)

    with col2:
        st.metric(label="Simulated Confidence", value=f"{prob*100:.2f}%")

    with col3:
        st.metric(label="Severity Level", value=severity)
        
    st.markdown("---")
    
    if class_name == INV_MAP[1]:
        st.error(f"Prediction: **{class_name}** is indicated with {severity} severity.")
    else:
        st.success(f"Prediction: **{class_name}** is indicated. This is a very low-risk outcome.")
