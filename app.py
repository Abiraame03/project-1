# app.py
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import pickle, json
import pyttsx3

# =========================
# App configuration
# =========================
st.set_page_config(page_title="Dyslexia Detection", layout="centered")
st.title("📝 Dyslexia Detection from Handwriting")
st.markdown("Capture handwriting to detect dyslexia and severity in children.")

# =========================
# Load model, threshold, class map
# =========================
@st.cache_resource
def load_model_and_configs():
    model_path = "/content/drive/MyDrive/Hybrid_BiLSTM_model/mobilenetv2_bilstm_best_thr_044.h5"
    threshold_path = "/content/drive/MyDrive/Hybrid_BiLSTM_model/best_threshold.json"
    class_map_path = "/content/drive/MyDrive/Hybrid_BiLSTM_model/class_indices_best.pkl"
    
    model = tf.keras.models.load_model(model_path)
    
    with open(threshold_path, "r") as f:
        threshold = json.load(f)
    
    with open(class_map_path, "rb") as f:
        class_map = pickle.load(f)
        
    return model, threshold, class_map

model, threshold, class_map = load_model_and_configs()

# =========================
# Camera input
# =========================
uploaded_file = st.camera_input("📸 Capture handwriting image")

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Captured Handwriting", use_column_width=True)
    
    # =========================
    # Preprocess image for model
    # =========================
    img_size = (128, 128)  # must match model input
    img_array = np.array(image.resize(img_size)) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # batch dimension

    # =========================
    # Predict class
    # =========================
    preds = model.predict(img_array)[0]
    predicted_index = np.argmax(preds)
    predicted_class = class_map[predicted_index]
    predicted_prob = preds[predicted_index]
    severity = "Low"
    
    # Apply threshold for severity
    if predicted_prob >= threshold["high"]:
        severity = "High"
    elif predicted_prob >= threshold["medium"]:
        severity = "Medium"

    st.subheader("Prediction Results")
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Probability:** {predicted_prob:.2f}")
    st.write(f"**Severity Level:** {severity}")
    st.write(f"**Thresholds (Proof):** {threshold}")

    # =========================
    # Voice output with general features
    # =========================
    features_text = (
        f"The handwriting shows features such as irregular strokes, "
        f"letter reversals, inconsistent spacing, or spelling mistakes. "
        f"These patterns indicate potential dyslexia symptoms for a child aged 6 years."
    )
    
    st.write("🗣 Voice Feedback:")
    st.write(features_text)
    
    # Text-to-speech
    engine = pyttsx3.init()
    engine.say(features_text)
    engine.runAndWait()
