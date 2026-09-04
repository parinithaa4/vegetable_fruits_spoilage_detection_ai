import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import matplotlib.pyplot as plt
import time

st.set_page_config(
    page_title="Vegetable & Fruit Spoilage Detection",
    layout="wide"
)

plt.style.use("ggplot")

st.markdown("""
<div style="background-color:#ffe6e6;padding:20px;border-radius:15px">
    <h1 style="color:#d32f2f;text-align:center">
    🍎 AI Freshness Detection System
    </h1>
    <p style="color:#d32f2f;text-align:center;font-size:18px">
    Upload an image or take a photo to detect Fresh or Rotten.
    </p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("vspoilage_detection.keras")

model = load_model()

st.sidebar.header("⚙ Settings")

THRESHOLD_HIGH = st.sidebar.slider("Rotten Threshold", 0.5, 0.9, 0.6)

fig_placeholder = st.empty()
label_placeholder = st.empty()
progress_placeholder = st.empty()

input_method = st.radio("Choose Input Method:", ["Upload Image", "Take Photo (Camera)"])

if input_method == "Upload Image":
    image = st.file_uploader("Upload fruit/vegetable image", type=["jpg","jpeg","png"])
else:
    image = st.camera_input("Take a photo of the fruit/vegetable")

if image is not None:

        file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)
        st.image(frame, caption="Image for Analysis", use_column_width=True)

        img = cv2.resize(frame, (224,224))
        img = preprocess_input(img)
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img, verbose=0)[0][0]

        label = "Rotten" if prediction > THRESHOLD_HIGH else "Fresh"
        confidence = prediction if label=="Rotten" else 1-prediction

        label_placeholder.markdown(
            f"<h2 style='color:{'#e74c3c' if label=='Rotten' else '#27ae60'}'>{label}</h2>",
            unsafe_allow_html=True
        )

        progress_placeholder.progress(int(confidence * 100))

        rotten_percent = prediction * 100
        fresh_percent = (1 - prediction) * 100

        fig, ax = plt.subplots(figsize=(6,4))
        bars = ax.bar(["Fresh", "Rotten"], [fresh_percent, rotten_percent])

        bars[0].set_color("#27ae60")
        bars[1].set_color("#e74c3c")

        ax.set_ylim(0,100)
        ax.set_ylabel("Confidence (%)")
        ax.set_title("Prediction Confidence", fontweight="bold")

        for i, v in enumerate([fresh_percent, rotten_percent]):
            ax.text(i, v + 2, f"{v:.1f}%", ha='center', fontweight='bold')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        fig_placeholder.pyplot(fig)
    
