import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import base64
from io import BytesIO

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="Pneumonia Detector",
    page_icon="🫁",
    layout="centered"
)

# ---------------------------
# Custom Background & Button Colors
# ---------------------------
def set_custom_styles():
    st.markdown(
        """
        <style>
        /* Background color for the whole page */
        body {
            background-color: #ffff; /* Light blue background */
        }

        /* Title color */
        .title {
            color: #003366;
        }

        /* Style for primary buttons (Upload, etc) */
        div.stButton > button {
            background-color: #fff;  /* Dark blue button */
            color: white;
            border-radius: 8px;
            border: none;
            padding: 10px 24px;
            font-weight: bold;
            transition: background-color 0.3s ease;
        }
        div.stButton > button:hover {
            background-color: #003d66;  /* Darker blue on hover */
            cursor: pointer;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

set_custom_styles()

# ---------------------------
# Load model
# ---------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/best_model.h5")

model = load_model()

# ---------------------------
# Preprocessing
# ---------------------------
def preprocess_image(img):
    img = cv2.resize(img, (224, 224))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ---------------------------
# Save image as base64 for preview
# ---------------------------
def image_to_base64(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return base64.b64encode(byte_im).decode()

# ---------------------------
# Session state for history
# ---------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------------------
# Header
# ---------------------------
st.markdown("<h1 style='text-align: center;' class='title'>🫁 Pneumonia Detection from Chest X-ray</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload a chest X-ray to predict the likelihood of pneumonia.</p>", unsafe_allow_html=True)

# ---------------------------
# Upload Image
# ---------------------------
uploaded_file = st.file_uploader("📤 Upload a Chest X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert image
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    # Display image (smaller size)
    st.markdown("#### 📸 Uploaded Image")
    st.image(image, caption="X-ray Preview", width=300)

    # Preprocess and predict
    preprocessed = preprocess_image(img_array)
    prediction = model.predict(preprocessed)[0][0]
    confidence = prediction if prediction > 0.5 else 1 - prediction
    label = "Pneumonia" if prediction > 0.5 else "Normal"
    emoji = "⚠️" if prediction > 0.5 else "✅"

    # Display results
    st.markdown("### 🧪 Prediction Result")
    if prediction > 0.5:
        st.error(f"{emoji} Pneumonia Detected")
    else:
        st.success(f"{emoji} No Pneumonia Detected")

    st.progress(int(confidence * 100))
    st.write(f"**Confidence:** {confidence:.2%}")

    # Save to history
    encoded_img = image_to_base64(image)
    st.session_state.history.append({
        "img_base64": encoded_img,
        "label": label,
        "confidence": confidence,
        "emoji": emoji
    })

# ---------------------------
# History Section
# ---------------------------
if st.session_state.history:
    st.markdown("---")
    st.markdown("### 🕓 Prediction History")

    cols = st.columns(3)
    for i, record in enumerate(reversed(st.session_state.history[-6:])):  # Show last 6
        with cols[i % 3]:
            st.image(f"data:image/png;base64,{record['img_base64']}", width=120)
            st.markdown(f"**{record['emoji']} {record['label']}**")
            st.caption(f"Confidence: {record['confidence']:.2%}")
