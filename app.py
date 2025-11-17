import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# --------------------------
# Config
# --------------------------
IMG_SIZE = (160, 160)  # input image size
TFLITE_MODEL_PATH = "models/mobilenetv2_bilstm_best_thr_044.tflite"

# --------------------------
# Load TFLite model
# --------------------------
@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details

interpreter, input_details, output_details = load_tflite_model()

# --------------------------
# Streamlit UI
# --------------------------
st.title("Dyslexia Detection Using TFLite Model")
uploaded_file = st.file_uploader("Upload a child’s handwriting or image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    # Set TFLite input
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]

    # Calculate confidence
    confidence = float(prediction) * 100

    # Binary Prediction
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
    st.subheader("Prediction Results")
    st.write(f"**Prediction:** {pred_label}")
    st.write(f"**Confidence:** {confidence:.2f}%")
    st.write(f"**Severity:** {severity}")
