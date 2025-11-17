import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# --------------------------
# Config
# --------------------------
IMG_SIZE = (160, 160)  # Input image size
MODEL_PATH = "models/mobilenetv2_bilstm_final.h5"

# --------------------------
# Load Keras model (cached)
# --------------------------
@st.cache_resource
def load_keras_model():
    return load_model(MODEL_PATH)

model = load_keras_model()

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="Dyslexia Detection", page_icon="🧠", layout="centered")
st.title("🧠 Dyslexia Detection from Handwriting")

uploaded_file = st.file_uploader("Upload a child’s handwriting or image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        with st.spinner("Analyzing..."):
            # Preprocess image
            img = image.resize(IMG_SIZE)
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

            # Prediction
            prediction = model.predict(img_array)[0][0]
            confidence = float(prediction) * 100
            pred_label = "Dyslexic" if prediction >= 0.5 else "Normal"

            # Severity
            if confidence < 5:
                severity = "Normal (0)"
            elif 5 <= confidence < 30:
                severity = "Mild (5-30)"
            elif 30 <= confidence <= 70:
                severity = "Moderate (30-70)"
            else:
                severity = "Severe (>70)"

            # Display results
            st.success("✅ Prediction Completed!")
            st.subheader("Prediction Results")
            st.write(f"**Prediction:** {pred_label}")
            st.write(f"**Confidence:** {confidence:.2f}%")
            st.write(f"**Severity:** {severity}")

        # Optional: Show confidence bar
        st.progress(min(int(confidence), 100))
