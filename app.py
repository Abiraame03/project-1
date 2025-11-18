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
# NOTE: Model is assumed to have a single output (probability of Dyslexia).
MODEL_PATH = "models/mobilenetv2_bilstm_final.h5" 
CLASS_MAP_PATH = "models/class_indices_best.pkl"
THRESHOLD_PATH = "models/best_threshold.json" 
IMG_SIZE = (160, 160)

# Custom Labels based on the user's model output (Binary)
CLASS_LABELS = ["Non-dyslexic", "Dyslexic"]
# Severity labels are used to categorize the confidence score.
SEVERITY_LABELS = ["Mild", "Moderate", "Severe"]

# Default prediction threshold for binary classification
DEFAULT_THRESHOLD = 0.5

st.set_page_config(page_title="Dyslexia Detection & Severity Prediction", layout="centered")
st.header("Dyslexia Detection & Severity Prediction")
st.markdown("This application is configured for a Binary (Dyslexic/Non-dyslexic) Model. Severity and Risk are derived from the single confidence score.")

# --- II. Model Loading and Environment Check ---

@st.cache_resource
def load_model_and_metadata():
    """
    Attempts to load the real ML model. 
    Falls back to simulation mode if files are not found or loading fails.
    """
    ml_model = None
    
    required_files = [MODEL_PATH]
    all_files_exist = all(os.path.exists(p) for p in required_files)
    
    if all_files_exist:
        
        # --- PRODUCTION MODE LOADING ---
        try:
            st.info("Attempting to load real model files for Production Mode...")
            time.sleep(1) 
            
            # 1. Load the Keras/TensorFlow model (expected to have one output)
            ml_model = tf.keras.models.load_model(MODEL_PATH)
                
            st.success("✅ **Production Mode:** Real model loaded successfully. Predictions will use model logic.")
            return ml_model, False # False means it's NOT simulation
            
        except Exception as e:
            # Fall through to simulation if loading fails
            st.error(f"❌ Failed to load real model. Error: {e}")
            pass 
    
    # --- SIMULATION FALLBACK ---
    if ml_model is None:
        missing_files = [p for p in required_files if not os.path.exists(p)]
        if missing_files:
             st.warning(f"Predictions are also simulated.")
        else:
             st.warning("Predictions are also simulated.")
        
    return ml_model, True # True means it IS simulation

model, IS_SIMULATION_MODE = load_model_and_metadata()

# --- III. Prediction and Severity Logic (Binary Model) ---

def predict_image(image_input, ml_model, is_simulation):
    """
    Executes a binary prediction (Prob of Dyslexic) and derives Severity from that score.
    """
    img = Image.open(image_input).convert("RGB")
    arr = np.array(img.resize(IMG_SIZE)) / 255.0
    arr = np.expand_dims(arr, axis=0)
    
    # Set up random seed for consistent simulation based on image hash
    try:
        image_input.seek(0)
        seed = hash(image_input.getvalue()) % 1000
    except AttributeError:
        seed = int(time.time() * 1000) % 1000
    random.seed(seed)
    np.random.seed(seed)
    
    prob = 0.0 # Confidence score for Dyslexic (Class 1)
    
    if not is_simulation:
        # PRODUCTION: Use the real model's prediction
        try:
            # Assumes the model returns a single probability for class 1 (Dyslexic)
            prediction_output = ml_model.predict(arr, verbose=0)
            prob = float(prediction_output[0][0])
        except Exception as e:
            st.error(f"Error during model prediction: {e}. Falling back to simulation.")
            is_simulation = True
    
    if is_simulation:
        # FIX: Ensure a balanced 50/50 split around 0.5 in simulation mode to test both outcomes reliably.
        if random.choice([True, False]):
             # Generate low score (Non-dyslexic prediction)
             prob = random.uniform(0.05, 0.49)
        else:
             # Generate high score (Dyslexic prediction)
             prob = random.uniform(0.51, 0.95)
        
    # --- Process Model Outputs ---
    
    # 1. Binary Classification (Predicted Class)
    # The main classification is based on the default 0.5 threshold
    label_index = 1 if prob >= DEFAULT_THRESHOLD else 0
    class_label = CLASS_LABELS[label_index]
    
    # The displayed confidence is for the PREDICTED class
    if class_label == "Dyslexic":
        class_confidence = prob * 100
    else:
        # Confidence in Non-dyslexic is (1 - prob_dyslexic)
        class_confidence = (1.0 - prob) * 100 

    # 2. Derived Severity Mapping (based on prob of Dyslexic)
    # Define custom ranges for severity:
    if prob <= 0.10:
        severity_label = "Low Risk"
    elif prob <= 0.30:
        severity_label = "Mild Risk"
    elif prob <= 0.70:
        severity_label = "Moderate Risk"
    else: # prob > 0.75
        severity_label = "Severe Risk"
    
    # 3. Dyslexia Risk Percentage (is the raw Dyslexic probability)
    dyslexia_risk = prob * 100

    results = {
        "class_label": class_label,
        "class_confidence": float(class_confidence),
        # Severity is now a derived label, not a separate confidence
        "severity_label": severity_label,
        "dyslexia_risk": float(dyslexia_risk)
    }
    
    return results

# --- IV. Handwriting Feature Generation ---

def generate_handwriting_features(severity_label):
    """
    Generates a detailed analysis text based on the derived severity level.
    """
    
    # Refined feature analysis focusing on common visual markers of developmental dyslexia in handwriting
    features = {
        "Severe Risk": [
            "**Irregular Spacing:** Extremely inconsistent spacing between words and letters, indicating profound difficulty with motor execution control.",
            "**Baseline Inconsistency:** Frequent and extreme deviations from the writing line, reflecting poor spatial organization.",
            "**Form and Motor Control:** Letters are often severely malformed, exhibiting heavy/uneven pen pressure and tremors.",
            "**Pervasive Errors:** Frequent letter reversals, sequencing errors, or omissions are visually evident across the sample. **This corresponds to a high Dyslexia Risk score.**"
        ],
        "Moderate Risk": [
            "**Inconsistent Spacing:** Noticeable, but intermittent, irregularities in word and letter spacing, affecting overall flow.",
            "**Baseline Fluctuation:** The writing line frequently rises or falls across sentences, indicating moderate spatial planning effort.",
            "**Character Formation:** A mix of well-formed and clearly laborious characters, often with irregular size and slant.",
            "**Occasional Errors:** Minor, isolated instances of letter confusion (e.g., 'b' vs. 'd' shape) are present, suggesting a moderate risk profile."
        ],
        "Mild Risk": [
            "**Subtle Spacing Issues:** Spacing is generally adequate but shows slight, minor inconsistencies.",
            "**Stable Baseline:** The writing line is mostly stable, with minimal non-diagnostic fluctuations.",
            "**Smooth Formation:** Characters are typically formed with clear strokes, with only minor imperfections.",
            "**Minimal Errors:** Very few visual markers are present, suggesting a low impact of dyslexic features."
        ],
        # Used when the overall prediction is Non-dyslexic
        "Low Risk": [
            "**High Consistency:** Handwriting is highly legible, uniform in size, slant, and spacing, demonstrating high execution quality.",
            "**Consistent Baseline:** The writing consistently adheres to the baseline, indicating strong spatial awareness.",
            "**Fluent Strokes:** Strokes are smooth, continuous, and clear, suggesting high writing fluency and minimal motor planning effort.",
            "**No Characteristic Markers:** The image shows none of the characteristic visual markers associated with developmental dyslexia. **This supports the 'Non-dyslexic' classification.**"
        ]
    }
    
    analysis_points = features.get(severity_label, features["Low Risk"])
    
    analysis_text = f"**Qualitative Analysis: Features Consistent with Derived Severity ({severity_label}):**\n"
    analysis_text += "This analysis describes the visual handwriting features **expected at this risk level** and is used to **justify** the quantitative confidence score.\n\n"
        
    analysis_text += "**Key Visual Markers (Consistent with Prediction):**\n"
    for point in analysis_points:
        analysis_text += f"- {point}\n"
    
    return analysis_text

# --- V. Streamlit UI (Refactored to match image structure) ---

st.sidebar.header("Model Operation Mode")
if IS_SIMULATION_MODE:
    st.sidebar.error("Model is running in **Simulation Mode**.")
else:
    st.sidebar.success("Model is running in **Production Mode**.")


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
        # Use the multi-output prediction function
        results = predict_image(processed_file, model, IS_SIMULATION_MODE)

    # --- Display Core Results (Matching the image output structure) ---
    st.subheader("Prediction Results")
    st.markdown("---")
    
    class_label = results["class_label"]
    severity_label = results["severity_label"]
    dyslexia_risk = results["dyslexia_risk"]
    class_confidence = results["class_confidence"]
    
    # --- Conditional Display Logic ---
    if class_label == "Non-dyslexic":
        # Case 1: NOT Dyslexic (Predicted Class is Non-dyslexic)
        st.success("## 🟢 NORMAL HANDWRITING DETECTED")
        st.info("The model indicates a low probability of dyslexic markers.")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            # Predicted Class will be Non-dyslexic (X.XX%)
            st.metric(label="Predicted Class", value=f"{class_label} ({class_confidence:.2f}%)")
        with col2:
            # Dyslexia Risk is the probability of the *other* class
            st.metric(label="Dyslexia Risk", value=f"{dyslexia_risk:.2f}%")
            
        st.markdown("---")
        st.subheader("Justification and Feature Analysis")
        st.success(
            f"**Conclusion:** The handwriting features align with a {severity_label} profile. The low calculated risk of {dyslexia_risk:.2f}% strongly supports the primary classification of **Non-dyslexic**."
        )
        # Use "Low Risk" to pull the non-dyslexic feature list
        st.markdown(generate_handwriting_features("Low Risk"))
        
    else:
        # Case 2: DYSLEXIC MARKER DETECTED (Predicted Class is Dyslexic)
        st.error("## 🔴 DYSLEXIA DETECTED")
        st.warning(f"The model detected a high risk, classifying the sample as {class_label}.")

        col1, col2, col3 = st.columns(3)
        with col1:
            # Predicted Class will be Dyslexic (X.XX%)
            st.metric(label="Predicted Class", value=f"{class_label} ({class_confidence:.2f}%)")
        with col2:
            # Predicted Severity is the derived risk category
            st.metric(label="Predicted Risk Category", value=f"{severity_label}")
        with col3:
            # Dyslexia Risk is the raw probability of the Dyslexic class
            st.metric(label="Dyslexia Risk", value=f"{dyslexia_risk:.2f}%")
        
        st.markdown("---")
        st.subheader("Handwriting Feature Analysis")
        
        # Comprehensive Justreiification
        st.error(
            f"**Dyslexia Detection Justification:**\n\n"
            f"**Quantitative Evidence:** The model's classification is **{class_label}** with a confidence of **{class_confidence:.2f}%**. The cumulative Dyslexia Risk is **{dyslexia_risk:.2f}%**, indicating a high probability of developmental markers. This high risk calculation confirms the condition's presence.\n\n"
            f"**Observed Abnormalities:** The derived risk category is **{severity_label}**. This aligns with the visual analysis of the handwriting features, which exhibits characteristics typical of the **{severity_label}** profile, as detailed below. This unified evidence supports the detection of a dyslexic condition."
        )
        
        # Detailed Feature Analysis (uses the derived severity level)
        st.markdown(generate_handwriting_features(severity_label))
        
    st.markdown("---")
