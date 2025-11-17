# =====================================
# 📦 Streamlit App: Dyslexia Detection
# =====================================
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# --------------------------
# Load model
# --------------------------
MODEL_PATH = "models/mobilenetv2_bilstm_best_thr_044.h5"
model = load_model(MODEL_PATH)

# --------------------------
# App UI
# --------------------------
st.title("🧠 Dyslexia Detection from Image")
st.write("Upload an image of handwriting or text sample for analysis.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # --------------------------
    # Preprocess image
    # --------------------------
    img = load_img(uploaded_file, target_size=(160, 160))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # --------------------------
    # Prediction
    # --------------------------
    pred_prob = float(model.predict(img_array)[0][0])  # binary output
    confidence = pred_prob if pred_prob >= 0 else 1 + pred_prob

    prediction = "Dyslexic" if pred_prob >= 0.5 else "Normal"

    # --------------------------
    # Severity calculation
    # --------------------------
    confidence_percent = confidence * 100
    if prediction == "Normal":
        severity = "Normal (0)"
    elif confidence_percent <= 30:
        severity = "Mild (5-30)"
    elif confidence_percent <= 70:
        severity = "Moderate (30-70)"
    else:
        severity = "Severe (>70)"

    # --------------------------
    # Display results
    # --------------------------
    st.markdown(f"**Prediction:** {prediction}")
    st.markdown(f"**Confidence:** {confidence_percent:.2f}%")
    st.markdown(f"**Severity:** {severity}")
