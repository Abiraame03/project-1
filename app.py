import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import pickle
import json
import time
import base64
import io
import wave
import struct
import random
import requests # Added for making HTTP API calls

# --- Set Up Gemini API Configuration ---
# Leave apiKey as an empty string; the environment will handle it.
API_KEY = ""
TTS_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-tts:generateContent"
TTS_MODEL = "gemini-2.5-flash-preview-tts"
TTS_VOICE = "Kore" # Clear, firm voice
TTS_SAMPLE_RATE = 24000 

st.set_page_config(page_title="Dyslexia Detection & TTS Analysis", layout="centered")

# --- Configuration Constants ---
MODEL_PATH = "models/mobilenetv2_bilstm_final.h5"
CLASS_MAP_PATH = "models/class_indices.pkl"
THRESHOLD_PATH = "models/best_threshold.json"
IMG_SIZE = (160, 160)
THRESHOLD = 0.5
INV_MAP = {0: "No Dyslexia (Normal)", 1: "Dyslexia Detected"}

# --- Model Loading Structure (Simulated for Single-File Deployment) ---

@st.cache_resource
def load_model_and_metadata():
    """
    Simulated loading of the ML model and metadata files.
    
    FOR PRODUCTION: To enable real prediction, you must uncomment the 'REAL LOADING' 
    block below and ensure your 'models/' folder is deployed.
    """
    
    st.warning(f"🚨 **Simulation Mode:** Model files are not included. Prediction and severity results are simulated. TTS functionality is configured for live API calls with a fallback.")

    # --- SIMULATION VALUES ---
    time.sleep(1) 
    ml_model = None 
    
    # --- REAL LOADING (UNCOMMENT THIS BLOCK FOR PRODUCTION USE) ---
    # global THRESHOLD, INV_MAP
    # try:
    #     ml_model = tf.keras.models.load_model(MODEL_PATH)
    #     with open(CLASS_MAP_PATH, "rb") as f:
    #         class_indices = pickle.load(f)
    #     INV_MAP = {v: k for k, v in class_indices.items()}
    #     with open(THRESHOLD_PATH, "r") as f:
    #         THRESHOLD = json.load(f)["threshold"]
    #     st.success("Model structure loaded successfully for production use.")
    # except Exception as e:
    #     # st.error(f"Failed to load real model files. {e}")
    #     ml_model = None

    return ml_model, INV_MAP, THRESHOLD

model, INV_MAP, THRESHOLD = load_model_and_metadata()

# --- Prediction Logic ---

def predict_image(image_input, ml_model, inv_map, threshold):
    """
    Preprocesses the image, gets a prediction probability (confidence score), 
    and determines the class and severity based on that score.
    """
    img = Image.open(image_input).convert("RGB")
    arr = np.array(img.resize(IMG_SIZE)) / 255.0
    arr = np.expand_dims(arr, axis=0)
    
    # --- Get Prediction Probability (Confidence Score) ---
    if ml_model is not None:
        prob = float(ml_model.predict(arr)[0][0])
    else:
        # SIMULATION FALLBACK: Deterministic random probability
        try:
            image_input.seek(0)
            seed = hash(image_input.getvalue()) % 1000
        except AttributeError:
            seed = int(time.time() * 1000) % 1000
        
        random.seed(seed)
        prob = random.uniform(0.05, 0.95) # Range 5% to 95%
    
    # 1. Determine Class based on Confidence Score
    label = 1 if prob > threshold else 0
    class_name = inv_map[label]

    # 2. Determine Severity based on Confidence Score and Specified Ranges
    prob_percent = prob * 100
    if prob_percent <= 5:
        severity_tag = "Normal"
        severity_range = "0-5%"
    elif prob_percent <= 30:
        severity_tag = "Mild"
        severity_range = "5-30%"
    elif prob_percent <= 70:
        severity_tag = "Moderate"
        severity_range = "30-70%"
    else:
        severity_tag = "Severe"
        severity_range = ">70%"

    severity = f"{severity_tag} ({severity_range})"
    return class_name, float(prob), severity

# --- Handwriting Feature Generation ---

def generate_handwriting_features(severity_tag):
    """Generates a detailed analysis text based on the severity level."""
    
    features = {
        "Severe": [
            "Prominent letter and number reversals are observed, such as 'b' for 'd' and '6' for '9'.",
            "Significant spelling mistakes and frequent attempts at overwriting are clearly visible.",
            "The words show abnormal and inconsistent spacing, making the sentence difficult to read. The overall uniformity of letters is poor."
        ],
        "Moderate": [
            "Occasional letter reversals or inversions are present, particularly in multi-syllable words.",
            "There are a noticeable number of spelling errors, though overwriting is moderate.",
            "The gap between words is somewhat irregular, and the size and slant of individual letters vary inconsistently."
        ],
        "Mild": [
            "Very few, if any, letter reversals are noted.",
            "Spelling mistakes are minor and infrequent. Overwriting is minimal.",
            "Word spacing is mostly consistent, but slight irregularities in letter uniformity can be seen."
        ],
        "Normal": [
            "Handwriting appears generally uniform with clear spacing and minimal to no spelling or reversal issues.",
            "No abnormal features such as reversal, inversion, overwriting, or inconsistent spacing were detected.",
            "The quality of the sample suggests low risk for handwriting-based indicators of dyslexia."
        ]
    }
    
    tag = severity_tag.split(" ")[0] # Get 'Severe', 'Moderate', etc.
    analysis = features.get(tag, features["Normal"])
    
    # Combine features into a conversational paragraph
    text = f"Based on the predicted severity level of {tag}, the handwriting analysis reveals the following patterns. "
    text += " ".join(analysis)
    text += " This assessment is based on observations of letter and word formation."
    
    return text

# --- TTS API Logic ---

def pcm_to_wav(pcm_data_base64, sample_rate):
    """Converts base64 PCM audio data to WAV format bytes."""
    pcm_data = base64.b64decode(pcm_data_base64)
    
    wav_io = io.BytesIO()
    with wave.open(wav_io, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit PCM (2 bytes)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    return wav_io.getvalue()

# Base64 encoded tiny silent WAV file (for fallback)
# 1 channel, 16bit, 24000Hz, 0.1 seconds of silence
SILENT_WAV_BASE64 = "UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAAABKAAACABAAZGF0YAAAAAAA"

def generate_tts_audio(text_to_speak):
    """Calls the Gemini TTS API using requests and returns the base64 audio data with exponential backoff."""
    headers = {'Content-Type': 'application/json'}
    payload = {
        "contents": [{
            "parts": [{ "text": text_to_speak }]
        }],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {
                    "prebuiltVoiceConfig": { "voiceName": TTS_VOICE }
                }
            }
        },
        "model": TTS_MODEL
    }
    
    url = f"{TTS_API_URL}?key={API_KEY}"
    
    # Implementing Exponential Backoff
    max_retries = 5
    base_delay = 1
    
    for attempt in range(max_retries):
        try:
            # Synchronous API call
            response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=10) # Added timeout
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            
            result = response.json()
            candidate = result.get('candidates', [{}])[0]
            part = candidate.get('content', {}).get('parts', [{}])[0]
            audio_data = part.get('inlineData', {}).get('data')
            mime_type = part.get('inlineData', {}).get('mimeType')

            # Check for correct audio format (audio/L16)
            if audio_data and mime_type and mime_type.startswith("audio/L16"):
                st.success("Voice output generated successfully via API.")
                return audio_data
            else:
                # Log error if structure is wrong
                if attempt == max_retries - 1: st.error("TTS API response missing expected audio data.")
                return None
            
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                # Retry only on transient errors (e.g., 429, 500, 503, timeout)
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                time.sleep(delay)
            else:
                st.error(f"TTS API request failed after {max_retries} attempts: {e}. Falling back to silent audio.")
                # Fallback on final failure
                break
        except Exception as e:
            # General error during processing (JSON parsing, etc.)
            st.error(f"An unexpected error occurred during TTS processing: {e}. Falling back to silent audio.")
            break
    
    # --- FALLBACK MECHANISM ---
    # Convert the silent WAV base64 string to raw bytes for st.audio
    return SILENT_WAV_BASE64

# --- Streamlit UI ---

st.header("🧠 Dyslexia Detection & Severity Prediction")
st.markdown("Use your device camera or upload a behavioral image (e.g., drawing or writing sample) for analysis.")

# Input Section
col_camera, col_upload = st.columns(2)
with col_camera:
    camera_file = st.camera_input("Take a Photo for Prediction")
with col_upload:
    uploaded_file = st.file_uploader(
        "Or Upload a File", 
        type=["jpg", "jpeg", "png"]
    )

processed_file = camera_file if camera_file is not None else uploaded_file

if processed_file:
    st.image(processed_file, caption="Input Image", use_column_width=True)
    
    # --- Run Prediction/Simulation ---
    with st.spinner("Analyzing image..."):
        class_name, prob, severity = predict_image(processed_file, model, INV_MAP, THRESHOLD)

    # --- Display Results ---
    st.subheader("Prediction Result")
    
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="Predicted Class", value=class_name)

    with col2:
        st.metric(label="Confidence Score", value=f"{prob*100:.2f}%")

    with col3:
        severity_tag = severity.split(" ")[0]
        st.metric(label="Severity Level", value=severity)
        
    st.markdown("---")

    # --- Handwriting Feature Analysis and TTS Output ---
    st.subheader("Handwriting Feature Analysis (Voice Output)")
    
    # 1. Generate Analysis Text
    analysis_text = generate_handwriting_features(severity_tag)
    st.markdown(f"**Text Analysis:** {analysis_text}")
    
    # 2. Generate and Play Audio
    with st.spinner("Attempting to generate voice output..."):
        tts_result_base64 = generate_tts_audio(analysis_text) 
    
    try:
        # Check if the result is the silent fallback
        is_fallback = tts_result_base64 == SILENT_WAV_BASE64
        
        if is_fallback:
            # Convert silent WAV base64 string directly to WAV bytes
            audio_bytes = base64.b64decode(tts_result_base64)
            st.warning("🎤 Voice output failed due to network error. Playing silent placeholder audio.")
        else:
            # Convert the returned base64 PCM data to a WAV byte stream
            audio_bytes = pcm_to_wav(tts_result_base64, TTS_SAMPLE_RATE)
            
        st.audio(audio_bytes, format='audio/wav')
        
    except Exception as e:
        st.error(f"Error converting or playing audio: {e}")
    
    st.markdown("---")
    
    if class_name == INV_MAP[1]:
        st.error(f"⚠️ **Result:** {class_name} is indicated with **{severity}** severity.")
    else:
        st.success(f"✅ **Result:** {class_name} is indicated. Low risk.")
