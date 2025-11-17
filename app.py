# app.py
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pickle, json, io
import cv2

# ---------------------------
# Config: update paths if needed
# ---------------------------
DYSLEXIA_MODEL_PATH = "mobilenetv2_bilstm_best_thr_044.h5"
THRESHOLD_PATH = "best_threshold.json"
DYSLEXIA_CLASS_MAP = "class_indices.pkl"

FEATURE_MODEL_PATH = "BEST_TUNED_MODEL.keras"
FEATURE_CLASS_MAP = "feature_class_indices.pkl"

IMG_SIZE = (160, 160)

# ---------------------------
# Load models once
# ---------------------------
@st.cache_resource
def load_models():
    # Binary Dyslexia model
    dys_model = tf.keras.models.load_model(DYSLEXIA_MODEL_PATH)
    with open(THRESHOLD_PATH, "r") as f:
        thr = json.load(f)["threshold"]
    with open(DYSLEXIA_CLASS_MAP, "rb") as f:
        dys_map = pickle.load(f)
    inv_map_dys = {v: k for k, v in dys_map.items()}

    # 3-class Handwriting feature model
    feat_model = tf.keras.models.load_model(FEATURE_MODEL_PATH)
    with open(FEATURE_CLASS_MAP, "rb") as f:
        feat_map = pickle.load(f)
    inv_map_feat = {v: k for k, v in feat_map.items()}

    return dys_model, thr, inv_map_dys, feat_model, inv_map_feat

dys_model, BEST_THRESHOLD, inv_map_dys, feat_model, inv_map_feat = load_models()

# ---------------------------
# Preprocessing
# ---------------------------
def preprocess_image(pil_img, size=IMG_SIZE):
    img = pil_img.convert("RGB").resize(size)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

# ---------------------------
# Heuristic analysis
# ---------------------------
def handwriting_heuristics(pil_img):
    img = np.array(pil_img.convert("L").resize(IMG_SIZE))
    th = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 11, 10)
    kernel = np.ones((2,2), np.uint8)
    opened = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
    stroke_density = opened.sum() / (255.0 * opened.shape[0] * opened.shape[1])
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    comp_count = len(contours)
    overwriting_score = min(1.0, comp_count / 50.0)
    freq_score = min(1.0, (comp_count/50.0)*0.5 + overwriting_score*0.5 + stroke_density*0.5)
    return {
        "stroke_density": float(stroke_density),
        "components": int(comp_count),
        "overwriting_score": float(overwriting_score),
        "frequency_score": float(freq_score)
    }

# ---------------------------
# Severity mapping
# ---------------------------
def severity_from_score(score):
    if score <= 0: return "Normal", 0
    if score <= 30: return "Mild", 1
    if score <= 70: return "Moderate", 2
    return "Severe", 3

# ---------------------------
# Dyslexia prediction with optional TTA
# ---------------------------
def predict_dyslexia(img, model, use_tta=True, tta_n=8):
    if not use_tta:
        arr = preprocess_image(img)
        return float(model.predict(arr)[0][0]), 0.0
    probs = []
    for _ in range(tta_n):
        arr = np.array(img.resize(IMG_SIZE), dtype=np.uint8)
        angle = np.random.uniform(-8, 8)
        M = cv2.getRotationMatrix2D((IMG_SIZE[0]//2, IMG_SIZE[1]//2), angle, 1.0)
        arr_r = cv2.warpAffine(arr, M, IMG_SIZE)
        arr_r = np.clip(arr_r*(1+np.random.uniform(-0.08,0.08)), 0, 255).astype(np.uint8)
        batch = np.expand_dims(arr_r/255.0, axis=0).astype(np.float32)
        probs.append(float(model.predict(batch)[0][0]))
    return float(np.mean(probs)), float(np.std(probs))

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Dyslexia Screening App", layout="wide")
st.title("📝 Dyslexia & Handwriting Screening")

st.markdown("""
Upload or capture handwriting of children (≤6 yrs).  
This app predicts **dyslexia risk** and **handwriting feature class**, with heuristic scores.
""")

# Columns for input and options
col1, col2 = st.columns([1,1])
with col1:
    st.header("1) Capture / Upload")
    cam_img = st.camera_input("Capture handwriting sample")
    file_img = st.file_uploader("Or upload image", type=["jpg","png","jpeg"])
    input_img = cam_img if cam_img is not None else file_img

with col2:
    st.header("2) Options")
    use_tta = st.checkbox("Use TTA (more stable prediction)", value=True)
    tta_n = st.slider("TTA samples", 4, 20, 8)
    st.caption(f"Model threshold: {BEST_THRESHOLD}")

st.write("---")

if input_img is not None:
    image = Image.open(io.BytesIO(input_img.read())).convert("RGB")
    st.image(image, caption="Input handwriting sample", use_column_width=True)

    # Handwriting feature model prediction
    arr_feat = preprocess_image(image)
    feat_preds = feat_model.predict(arr_feat)[0]
    feat_idx = int(np.argmax(feat_preds))
    feat_class = inv_map_feat.get(feat_idx, str(feat_idx))
    feat_conf = float(feat_preds[feat_idx])

    # Dyslexia prediction
    dys_prob, dys_std = predict_dyslexia(image, dys_model, use_tta=use_tta, tta_n=tta_n)
    dys_pct = dys_prob*100
    dys_label = "dyslexic" if dys_prob > BEST_THRESHOLD else "non_dyslexic"
    severity_text, severity_level = severity_from_score(dys_pct if dys_label=="dyslexic" else 0)

    # Heuristic analysis
    heur = handwriting_heuristics(image)

    # Display results
    st.subheader("Prediction Results")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Dyslexia risk (%)", f"{dys_pct:.2f}%")
        st.write(f"Decision: **{dys_label}**")
        st.write(f"Severity: **{severity_text}**")
        st.progress(min(100,int(dys_pct)))
    with c2:
        st.write(f"Handwriting feature: **{feat_class}** ({feat_conf*100:.2f}%)")
        st.write(f"Overwriting/Frequency score: {heur['frequency_score']:.2f}")

    # Save/export results
    result = {
        "dyslexia_prob": dys_prob,
        "dys_label": dys_label,
        "severity": severity_text,
        "feature_class": feat_class,
        "feature_conf": float(feat_conf),
        "heuristics": heur
    }
    if st.button("Save Result JSON"):
        fname = "result.json"
        with open(fname,"w") as f:
            json.dump(result, f, indent=2)
        st.download_button("Download JSON", data=open(fname,"rb").read(), file_name=fname)
else:
    st.info("Capture an image using the camera or upload a handwriting sample to start.")
