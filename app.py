# app.py
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pickle, json, os, io
import cv2
import matplotlib.pyplot as plt

# ---------------------------
# Configuration: update these paths if your Drive uses different names
# ---------------------------
DYSLEXIA_MODEL_PATH = "/content/drive/MyDrive/Hybrid_BiLSTM_model/mobilenetv2_bilstm_best_thr_044.h5"
THRESHOLD_PATH      = "/content/drive/MyDrive/Hybrid_BiLSTM_model/best_threshold.json"
DYSLEXIA_CLASS_MAP  = "/content/drive/MyDrive/Hybrid_BiLSTM_model/class_indices.pkl"

FEATURE_MODEL_PATH  = "/content/drive/MyDrive/DL-3 Model_TUNED_version/BEST_TUNED_MODEL.keras"
FEATURE_CLASS_MAP   = "/content/drive/MyDrive/DL-3 Model/class_indices.pkl"  # adjust if saved somewhere else

IMG_SIZE = (160,160)  # models expect 160x160

# ---------------------------
# Utilities: load models once
# ---------------------------
@st.cache_resource
def load_models():
    # Dyslexia binary model
    dys_model = tf.keras.models.load_model(DYSLEXIA_MODEL_PATH)
    with open(THRESHOLD_PATH, "r") as f:
        thr = json.load(f)["threshold"]
    with open(DYSLEXIA_CLASS_MAP, "rb") as f:
        dys_map = pickle.load(f)
    inv_map_dys = {v:k for k,v in dys_map.items()}

    # Handwriting-features (3-class) model
    feat_model = tf.keras.models.load_model(FEATURE_MODEL_PATH)
    with open(FEATURE_CLASS_MAP, "rb") as f:
        feat_map = pickle.load(f)
    inv_map_feat = {v:k for k,v in feat_map.items()}

    return dys_model, thr, inv_map_dys, feat_model, inv_map_feat

dys_model, BEST_THRESHOLD, inv_map_dys, feat_model, inv_map_feat = load_models()

# ---------------------------
# Image preprocessing helper
# ---------------------------
def preprocess_for_model(pil_img, size=IMG_SIZE):
    img = pil_img.convert("RGB").resize(size)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

# ---------------------------
# Heuristic analysis for overwriting / frequency of mistakes
# ---------------------------
def handwriting_heuristics(pil_img):
    """Return a small analysis: stroke_density, overwriting_score, components_count"""
    # convert to gray and binary
    img = np.array(pil_img.convert("L").resize(IMG_SIZE))
    # adaptive threshold to binarize (handwriting dark on lighter paper)
    th = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 11, 10)
    # morphological opening to remove speckle
    kernel = np.ones((2,2), np.uint8)
    opened = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
    # compute stroke density: ratio of ink pixels to image area
    stroke_density = opened.sum() / (255.0 * opened.shape[0] * opened.shape[1])
    # find contours (connected components)
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    comp_count = len(contours)

    # estimate overwriting: compute skeleton complexity approximated by intersections
    # skeletonize using thinning
    size = np.size(opened)
    skel = np.zeros(opened.shape, np.uint8)
    img_thin = opened.copy()
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    while True:
        eroded = cv2.erode(img_thin, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img_thin, temp)
        skel = cv2.bitwise_or(skel, temp)
        img_thin = eroded.copy()
        if cv2.countNonZero(img_thin) == 0:
            break
    # intersections approximate: convolve with 3x3 and count where > 3 neighbors in skeleton
    neigh = cv2.filter2D((skel>0).astype(np.uint8), -1, np.ones((3,3), np.uint8))
    intersections = np.sum((neigh > 3).astype(np.uint8))
    # normalize scores to 0-1 roughly
    overwriting_score = min(1.0, intersections / 50.0)
    # Frequency estimation: combine components & overwriting & stroke density
    freq_score = min(1.0, (comp_count/50.0) * 0.5 + overwriting_score*0.5 + stroke_density*0.5)
    return {
        "stroke_density": float(stroke_density),
        "components": int(comp_count),
        "intersections": int(intersections),
        "overwriting_score": float(overwriting_score),
        "frequency_score": float(freq_score)
    }

# ---------------------------
# Severity mapping helper
# ---------------------------
def severity_from_score(score_percent):
    # user mapping: normal - 0, mild (0-30), moderate (30-70), severe (>70)
    if score_percent <= 0:
        return "Normal", 0
    if score_percent <= 30:
        return "Mild", 1
    if score_percent <= 70:
        return "Moderate", 2
    return "Severe", 3

# ---------------------------
# Main Streamlit UI
# ---------------------------
st.set_page_config(page_title="Dyslexia Screening (Handwriting) — Demo", layout="wide")
st.title("📝 Dyslexia & Handwriting Feature Screening (children ≤ 6 yrs)")

st.markdown("""
This demo runs two models:
- **Dyslexia risk model (binary)** — MobileNetV2 + BiLSTM — outputs probability (0..1).
- **Handwriting features model (3 classes)** — outputs handwriting class: normal/reversal/corrected.

**Notes:** frequency/overwriting detection is heuristic (image-processing) and intended for additional signal — not clinical diagnosis.
""")

# Input: camera or upload
col1, col2 = st.columns([1,1])
with col1:
    st.header("1) Capture or upload a handwriting sample")
    img_file_buffer = st.camera_input("Capture handwriting sample (camera)")  # returns BytesIO or None
    uploaded_file = st.file_uploader("Or upload an image file (photo or scan)", type=["jpg","jpeg","png"])
    # prefer camera if provided
    input_blob = None
    if img_file_buffer is not None:
        input_blob = img_file_buffer.getvalue()
    elif uploaded_file is not None:
        input_blob = uploaded_file.read()
    else:
        st.info("Use camera or upload an image to start prediction.")

with col2:
    st.header("2) Options")
    use_tta = st.checkbox("Use Test-Time Augmentation (TTA) average (more stable)", value=True)
    tta_n = st.slider("TTA samples", min_value=4, max_value=20, value=8, step=2)
    st.write("Model threshold (loaded):", BEST_THRESHOLD)
    st.caption("Severity bins: Normal (0), Mild (0–30), Moderate (30–70), Severe (>70)")

st.write("---")

if input_blob is not None:
    # load image
    image = Image.open(io.BytesIO(input_blob)).convert("RGB")
    st.image(image, caption="Input image (resized for display)", use_column_width=True)

    # 1) Handwriting-feature prediction (3-class)
    arr_feat = preprocess_for_model(image, size=IMG_SIZE)
    preds_feat = feat_model.predict(arr_feat)[0]
    feat_idx = int(np.argmax(preds_feat))
    feat_class = inv_map_feat.get(feat_idx, str(feat_idx))
    feat_conf = float(preds_feat[feat_idx])

    # 2) Dyslexia binary prediction (prob)
    def predict_with_tta(img_pil, n=8):
        base = preprocess_for_model(img_pil, size=IMG_SIZE)
        if not use_tta:
            p = float(dys_model.predict(base)[0][0])
            return p, 0.0
        # simple augmentations for TTA
        aug_probs = []
        for i in range(n):
            # slight random transforms using cv2
            arr = np.array(img_pil.resize(IMG_SIZE)).astype(np.uint8)
            # random small rotation
            angle = np.random.uniform(-8,8)
            M = cv2.getRotationMatrix2D((IMG_SIZE[0]//2, IMG_SIZE[1]//2), angle, 1.0)
            arr_r = cv2.warpAffine(arr, M, IMG_SIZE)
            # random brightness
            arr_r = np.clip(arr_r * (1.0 + np.random.uniform(-0.08,0.08)), 0, 255).astype(np.uint8)
            batch = np.expand_dims(arr_r.astype("float32")/255.0, axis=0)
            p = float(dys_model.predict(batch)[0][0])
            aug_probs.append(p)
        return float(np.mean(aug_probs)), float(np.std(aug_probs))

    dys_prob, dys_std = predict_with_tta(image, n=tta_n if use_tta else 1)
    dys_pct = dys_prob * 100.0
    dys_label = "dyslexic" if dys_prob > BEST_THRESHOLD else "non_dyslexic"
    severity_text, severity_level = severity_from_score(dys_pct if dys_label=="dyslexic" else 0)

    # heuristics
    heur = handwriting_heuristics(image)
    freq_pct = heur["frequency_score"] * 100.0
    overwriting_flag = heur["overwriting_score"] > 0.25

    # Results display
    st.subheader("Prediction results")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Dyslexia (binary)**")
        st.metric("Risk probability", f"{dys_pct:.2f}%")
        st.write(f"Decision (threshold {BEST_THRESHOLD}): **{dys_label}**")
        st.write(f"Severity : **{severity_text}**")
        st.write(f"TTA stddev: {dys_std:.3f}" if use_tta else "")
        # show simple bar
        st.progress(min(100, int(dys_pct)))

    with c2:
        st.markdown("**Handwriting features (3-class)**")
        st.write(f"Predicted feature: **{feat_class}**")
        st.write(f"Confidence: **{feat_conf*100:.2f}%**")
        # abnormality summary
        abnormal = []
        if feat_class.lower() in ["reversal","inversion","mirror","mirror_reversal"]:
            abnormal.append("Reversal / mirror error")
        if feat_class.lower() in ["corrected","overwritten","overwrite"]:
            abnormal.append("Overwriting / corrections")
        # heuristics-based:
        if overwriting_flag:
            abnormal.append("High overwriting (heuristic)")
        if heur["components"] > 60 or heur["frequency_score"] > 0.5:
            abnormal.append("Frequent mistakes / scribbles (heuristic)")

        if abnormal:
            st.warning("Abnormal features detected: " + ", ".join(abnormal))
        else:
            st.success("No major abnormal handwriting features detected")

    st.write("---")
    st.subheader("Handwriting quality & heuristic details")
    st.write(f"Stroke density: {heur['stroke_density']:.4f}")
    st.write(f"Connected components: {heur['components']}")
    st.write(f"Skeleton intersections (proxy): {heur['intersections']}")
    st.write(f"Overwriting score (0-1): {heur['overwriting_score']:.3f}")
    st.write(f"Frequency score (0-1): {heur['frequency_score']:.3f}")
    st.info("Heuristic scores are approximations — use as complementary signals, not diagnosis.")

    st.write("---")
    st.subheader("Advice / Next steps")
    if dys_label == "dyslexic" or freq_pct > 40 or "Reversal / mirror error" in abnormal:
        st.error("Recommend further screening: consider teacher/clinician evaluation and more samples.")
    else:
        st.success("No urgent flags — continue monitoring and collect more writing samples.")

    # Export result button
    result = {
        "dyslexia_prob": dys_prob,
        "dys_label": dys_label,
        "severity": severity_text,
        "feature_class": feat_class,
        "feature_conf": float(feat_conf),
        "heuristics": heur
    }

    if st.button("Save result to local JSON"):
        fname = "prediction_result.json"
        with open(fname, "w") as f:
            json.dump(result, f, indent=2)
        st.download_button("Download result JSON", data=open(fname,"rb").read(), file_name=fname)
else:
    st.info("Waiting for image input (camera or upload).")

# ---------------------------
# References (papers)
# ---------------------------
st.write("---")
st.markdown("### References & reading (selected recent works)")
st.markdown("""
- Treiman R. *Statistical Learning, Letter Reversals, and Reading* — study of reversal errors in 5–6-year-olds (useful background on reversal frequency). (PMC). :contentReference[oaicite:0]{index=0}  
- Isa IS et al., *Automated Detection of Dyslexia Symptom Based on Handwriting* (2019) — early automated handwriting screening. :contentReference[oaicite:1]{index=1}  
- Suárez-Coalla P., *Dynamics of Sentence Handwriting in Dyslexia* (2020) — handwriting dynamics differ in dyslexia. :contentReference[oaicite:2]{index=2}  
- Review: Alkhurayyif Y., *A Review of AI-Based Dyslexia Detection* (2024) — overview of methods and limitations. :contentReference[oaicite:3]{index=3}
""")
st.caption("These papers explain why handwriting features (reversals, overwriting, stroke dynamics) are informative for dyslexia screening — use combined heuristics + model predictions for robust screening.")
