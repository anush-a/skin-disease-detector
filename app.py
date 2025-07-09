import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image, ImageOps
import pathlib

# Constants
MODEL_PATH = pathlib.Path("mobilenetv2_ham10000.h5")
CLASS_JSON = pathlib.Path("class_indices.json")

# Simple + medical-friendly labels
friendly = {
    "akiec": "Early skin cancer (Actinic Keratosis / IEC)",
    "bcc":   "Common skin cancer (Basal Cell Carcinoma)",
    "bkl":   "Non-cancerous growth (Benign Keratosis)",
    "df":    "Harmless skin bump (Dermatofibroma)",
    "mel":   "Dangerous skin cancer (Melanoma)",
    "nv":    "Mole (Melanocytic Nevus)",
    "vasc":  "Blood vessel lump (Vascular Lesion)",
}

@st.cache_resource
def load_assets():
    """Load model and label map once per session."""
    model = tf.keras.models.load_model(MODEL_PATH)
    with CLASS_JSON.open() as f:
        idx2class = {v: k for k, v in json.load(f).items()}
    return model, idx2class

model, idx2class = load_assets()

# UI
st.title("🩺 Skin Disease Detector")
st.caption("Detects 7 skin lesion types from dermatoscopic images")

uploaded = st.file_uploader("Upload a dermatoscopic image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    # Preprocess
    img = ImageOps.fit(image, (224, 224))
    arr = np.array(img) / 255.0
    pred = model.predict(np.expand_dims(arr, 0))[0]
    top = np.argmax(pred)
    
    raw_label = idx2class[top]
    pretty = friendly.get(raw_label, raw_label)

    st.success(f"🔍 Prediction: **{pretty}**  ({pred[top]*100:.2f}% confidence)")

    # Show probabilities
    st.subheader("Class probabilities:")
    for i, p in enumerate(pred):
        raw = idx2class[i]
        label = friendly.get(raw, raw)
        st.write(f"- **{label}**: {p*100:.2f}%")
