# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io, os, json, pickle, csv
import cv2
from gtts import gTTS

# ---------------------------
# CONFIG - paths inside repo
# ---------------------------
MODEL_PATH = "models/mobilenetv2_bilstm_best_thr_044.h5"   # dyslexia binary model
THRESHOLD_PATH = "models/best_threshold.json"
FEATURE_MODEL_PATH = "models/BEST_TUNED_MODEL.keras"      # 3-class handwriting features model
FEATURE_CLASS_MAP = "models/class_indices_best.pkl"
RESULTS_CSV = "screening_history.csv"
IMG_SIZE = (160, 160)

# ---------------------------
# UTIL: load resources
# ---------------------------
@st.cache_resource
def load_binary_model(path=MODEL_PATH):
    if not os.path.exists(path):
        return None
    return tf.keras.models.load_model(path, compile=False)

@st.cache_resource
def load_feature_model(path=FEATURE_MODEL_PATH):
    if not os.path.exists(path):
        return None
    return tf.keras.models.load_model(path, compile=False)

@st.cache_resource
def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)

@st.cache_resource
def load_pickle(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

binary_model = load_binary_model()
feature_model = load_feature_model()
best_thr_json = load_json(THRESHOLD_PATH)
feature_class_map = load_pickle(FEATURE_CLASS_MAP)

# Reverse maps
if feature_class_map:
    inv_feature_map = {v: k for k, v in feature_class_map.items()}
else:
    inv_feature_map = {0: "Corrected", 1: "Normal", 2: "Reversal"}  # fallback

best_threshold = best_thr_json.get("threshold") if best_thr_json else 0.5

# ---------------------------
# PREPROCESS
# ---------------------------
def preprocess_pil(img_pil, size=IMG_SIZE):
    img = img_pil.convert("RGB").resize(size)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

# ---------------------------
# HANDWRITING ABNORMALITY HEURISTICS
# ---------------------------
def detect_reversal_inversion_overwrites(pil_img):
    """Return dict with simple heuristics:
       reversal_detected, inversion_detected, overwrite_score (0-1)
    """
    img = np.array(pil_img.convert("L"))
    # Binarize (adaptive helps uneven lighting)
    th = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 11, 8)
    # remove tiny noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    cleaned = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)

    # find contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    small_cnt = 0
    large_cnt = 0
    reversal_flag = False
    inversion_flag = False
    suspicious_notes = []

    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        area = w*h
        if area < 100:  # tiny specks
            small_cnt += 1
            continue
        large_cnt += 1
        ratio = w / (h + 1e-6)
        # heuristic: wide shapes similar height may be b/d or p/q like flips
        if 0.8 <= ratio <= 1.2 and w*h > 400:
            reversal_flag = True
            suspicious_notes.append("possible letter flips (b/d or p/q)")
        # inversion: if shape is more similar vertically flipped
        roi = cleaned[y:y+h, x:x+w]
        if roi.size == 0: 
            continue
        vert_sym = np.sum(np.abs(roi - cv2.flip(roi, 0)))
        horiz_sym = np.sum(np.abs(roi - cv2.flip(roi, 1)))
        # if vertically similar less than horizontally -> likely upside-down
        if vert_sym < horiz_sym * 0.6:
            inversion_flag = True
            suspicious_notes.append("possible upside-down characters")

    # Overwrite score: many small contours => scratchy/overwriting
    total_cnt = small_cnt + large_cnt + 1e-6
    overwrite_score = min(1.0, small_cnt / total_cnt)

    return {
        "reversal_detected": reversal_flag,
        "inversion_detected": inversion_flag,
        "overwrite_score": float(overwrite_score),
        "suspicious_notes": suspicious_notes
    }

# ---------------------------
# SEVERITY MAPPING
# ---------------------------
def severity_from_pct(score_pct):
    """score_pct is probability*100 for dyslexic label"""
    if score_pct <= 0:
        return "Normal", 0
    if score_pct <= 30:
        return "Mild", 1
    if score_pct <= 70:
        return "Moderate", 2
    return "Severe", 3

# ---------------------------
# SAVE RESULTS
# ---------------------------
def append_result_to_csv(row_dict, csv_path=RESULTS_CSV):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row_dict.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)

# ---------------------------
# VOICE: text -> mp3 bytes using gTTS
# ---------------------------
def speak_text(text, lang="en"):
    tts = gTTS(text=text, lang=lang)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf.read()   # bytes (mp3) ready to play with st.audio

# ---------------------------
# UI
# ---------------------------
st.set_page_config(page_title="Dyslexia Screening (Camera)", layout="wide")
st.title("📝 Dyslexia screening & handwriting feature detection (camera)")

st.markdown(
    """
    **How to use:** ask the child to write/draw on a plain white paper, hold the paper steady,
    then use the camera button below to capture the image. The app will analyze handwriting features,
    estimate dyslexia risk and severity, and list abnormal handwriting patterns (reversals, inversions, overwriting).
    """
)

# show warnings if models missing
if binary_model is None:
    st.error(f"Binary dyslexia model not found at `{MODEL_PATH}`. Put it in `models/` folder.")
if feature_model is None:
    st.error(f"Handwriting feature model not found at `{FEATURE_MODEL_PATH}`. Put it in `models/` folder.")

col1, col2 = st.columns([2,1])

with col1:
    st.subheader("Capture handwriting (camera)")
    camera_file = st.camera_input("Use your webcam / phone camera to capture an image")

with col2:
    st.subheader("Quick settings")
    thr = st.slider("Dyslexia threshold (probability)", min_value=0.0, max_value=1.0, value=float(best_threshold), step=0.01)
    show_notes = st.checkbox("Show detection notes (heuristics)", value=True)
    play_voice = st.checkbox("Play a short voice summary", value=True)

if camera_file is not None:
    img = Image.open(camera_file).convert("RGB")
    st.image(img, caption="Captured image", use_column_width=True)

    # preprocess and predict
    x = preprocess_pil(img)

    # binary dyslexia model
    dys_prob = None
    dys_label = None
    if binary_model:
        try:
            p = binary_model.predict(x)[0].ravel()
            # p might be shape (1,) or (1,1)
            dys_prob = float(p[0]) if hasattr(p, "__len__") else float(p)
        except Exception as e:
            st.error(f"Error predicting dyslexia: {e}")
            dys_prob = None

    # feature model (3-class)
    feature_name = None
    feature_conf = None
    if feature_model:
        try:
            fp = feature_model.predict(x)[0]
            idx = int(np.argmax(fp))
            feature_conf = float(fp[idx])
            feature_name = inv_feature_map.get(idx, str(idx))
        except Exception as e:
            st.error(f"Error predicting handwriting feature: {e}")

    # heuristics
    heur = detect_reversal_inversion_overwrites(img)

    # thresholding (use UI slider thr, but default shows saved best_threshold)
    if dys_prob is not None:
        dys_label = "dyslexic" if dys_prob >= thr else "non_dyslexic"
        dys_pct = dys_prob * 100.0
    else:
        dys_label = "unknown"
        dys_pct = 0.0

    severity_txt, severity_id = severity_from_pct(dys_pct if dys_label=="dyslexic" else 0)

    # display outputs
    st.markdown("---")
    st.subheader("Prediction Summary")
    st.write(f"**Dyslexia probability:** {dys_prob:.4f}" if dys_prob is not None else "**Dyslexia probability:** N/A")
    st.write(f"**Prediction:** {dys_label}")
    st.write(f"**Severity:** {severity_txt}")
    st.write(f"**Handwriting feature:** {feature_name}  (conf: {feature_conf:.3f})" if feature_name else "Handwriting feature: N/A")

    if show_notes:
        st.markdown("**Heuristic feature checks:**")
        st.write(f"- Reversal-like patterns: {heur['reversal_detected']}")
        st.write(f"- Inversion (upside-down)-like patterns: {heur['inversion_detected']}")
        st.write(f"- Overwrite score (0-1): {heur['overwrite_score']:.3f}")
        if heur["suspicious_notes"]:
            st.write("- Notes:", ", ".join(heur["suspicious_notes"]))

    # Save record
    record = {
        "dys_prob": float(dys_prob) if dys_prob is not None else None,
        "dys_label": dys_label,
        "severity": severity_txt,
        "feature": feature_name,
        "feature_conf": feature_conf,
        "reversal": heur["reversal_detected"],
        "inversion": heur["inversion_detected"],
        "overwrite_score": heur["overwrite_score"]
    }
    append_result_to_csv(record)

    # voice summary
    if play_voice:
        # Build voice text
        if dys_prob is not None:
            pct = round(dys_prob*100,1)
            vtxt = f"Estimated dyslexia probability {pct} percent. Prediction {dys_label}. Severity {severity_txt}."
            if heur["reversal_detected"]:
                vtxt += " Reversal-like patterns detected."
            if heur["inversion_detected"]:
                vtxt += " Inversion-like patterns detected."
            if heur["overwrite_score"] > 0.3:
                vtxt += " Overwriting detected."
        else:
            vtxt = "Prediction not available."

        try:
            mp3_bytes = speak_text(vtxt)
            st.audio(mp3_bytes, format="audio/mp3")
        except Exception as e:
            st.warning("Voice output failed (gTTS). Reason: " + str(e))

st.markdown("---")
st.caption("Note: this is a screening tool, not a medical diagnosis. Use clinical assessment for decisions.")
