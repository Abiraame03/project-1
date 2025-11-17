import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# --------------------------
# Config
# --------------------------
IMG_SIZE = (160, 160)  # your image input size
MODEL_PATH = "models/mobilenetv2_bilstm_best_thr_044.h5"  # your model

# Load model once
@st.cache_resource
def load_dyslexia_model():
    return load_model(MODEL_PATH)

model = load_dyslexia_model()

# --------------------------
# Streamlit UI
# --------------------------
st.title("Dyslexia Detection")

uploaded_file = st.file_uploader("Upload a child’s handwriting or image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess image
    img = load_img(uploaded_file, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0  # normalize
    img_array = np.expand_dims(img_array, axis=0)
    
    # Prediction
    prediction = model.predict(img_array)[0][0]
    confidence = float(prediction) * 100
    
    # Binary Prediction
    if prediction >= 0.5:
        pred_label = "Dyslexic"
    else:
        pred_label = "Normal"
    
    # Severity (based on range)
    if confidence < 5:
        severity = "Normal (0)"
    elif 5 <= confidence < 30:
        severity = "Mild (5-30)"
    elif 30 <= confidence <= 70:
        severity = "Moderate (30-70)"
    else:
        severity = "Severe (>70)"
    
    # Display results
    st.subheader("Prediction Results")
    st.write(f"**Prediction:** {pred_label}")
    st.write(f"**Confidence:** {confidence:.2f}%")
    st.write(f"**Severity:** {severity}")
