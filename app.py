import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# --------------------------
# Config
# --------------------------
IMG_SIZE = (160, 160)  # your image input size
TFLITE_MODEL_PATH = "models/mobilenetv2_bilstm_best_thr_044.tflite"  # your TFLite model

# Load TFLite model once
@st.cache_resource
def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model(TFLITE_MODEL_PATH)

# Get input & output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# --------------------------
# Streamlit UI
# --------------------------
st.title("Dyslexia Detection (TFLite)")

uploaded_file = st.file_uploader("Upload a child’s handwriting or image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess image
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Set the tensor
    interpreter.set_tensor(input_details[0]['index'], img_array)
    
    # Run inference
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]
    confidence = float(prediction) * 100
    
    # Binary Prediction
    if prediction >= 0.5:
        pred_label = "Dyslexic"
    else:
        pred_label = "Normal"
    
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
    st.subheader("Prediction Results")
    st.write(f"**Prediction:** {pred_label}")
    st.write(f"**Confidence:** {confidence:.2f}%")
    st.write(f"**Severity:** {severity}")
