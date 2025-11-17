import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import tensorflow as tf

st.set_page_config(page_title="Dyslexia Detection", layout="centered")
st.title("📝 Dyslexia Handwriting Detection")
st.write("Capture handwriting using your webcam or upload an image for dyslexia analysis.")

# Load TFLite model
@st.cache_resource
def load_tflite_model(model_path="dyslexia_model.tflite"):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Webcam capture
st.subheader("Capture Handwriting")
run = st.button("Start Webcam")
FRAME_WINDOW = st.image([])

cap = None
if run:
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Unable to access webcam")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame_rgb)

        if st.button("Capture"):
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            Image.fromarray(frame_rgb).save(temp_file.name)
            st.success("Image captured!")
            break

    cap.release()
    cv2.destroyAllWindows()

# Prediction
st.subheader("Model Prediction")
uploaded_file = st.file_uploader("Or upload handwriting image", type=["png", "jpg", "jpeg"])

image_to_predict = None
if uploaded_file:
    image_to_predict = Image.open(uploaded_file)
elif 'temp_file' in locals():
    image_to_predict = Image.open(temp_file.name)

if image_to_predict:
    st.image(image_to_predict, caption="Input Image", use_column_width=True)
    
    # Preprocess for TFLite model
    img = image_to_predict.convert("L").resize((128, 128))
    img_array = np.array(img)/255.0
    img_array = img_array.reshape(1, 128, 128, 1).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])

    pred_class = np.argmax(prediction)
    confidence = np.max(prediction)

    st.write(f"**Predicted Class:** {pred_class}")
    st.write(f"**Confidence:** {confidence:.2f}")
