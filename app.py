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
    Upload an image or use live camera to detect Fresh or Rotten.
    </p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("vspoilage_detection.keras")

model = load_model()

st.sidebar.header("⚙ Settings")

BUFFER_SIZE = st.sidebar.slider("Frames to Average", 5, 30, 10)
THRESHOLD_HIGH = st.sidebar.slider("Rotten Threshold", 0.5, 0.9, 0.6)
THRESHOLD_LOW = st.sidebar.slider("Fresh Threshold", 0.1, 0.5, 0.4)

stop_camera = st.sidebar.button("Stop Camera")

option = st.radio("Choose Input Method:", ["Upload Image", "Use Camera"])

frame_placeholder = st.empty()
fig_placeholder = st.empty()
label_placeholder = st.empty()
progress_placeholder = st.empty()

def decide_label(avg_pred, last_label):
    if avg_pred > THRESHOLD_HIGH:
        return "Rotten"
    elif avg_pred < THRESHOLD_LOW:
        return "Fresh"
    else:
        return last_label

if option == "Upload Image":

    image = st.file_uploader("Upload fruit/vegetable image", type=["jpg","jpeg","png"])

    if image is not None:

        file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)
        st.image(frame, caption="Uploaded Image", use_column_width=True)

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

else:

    cap = cv2.VideoCapture(0)
    last_label = "Fresh"
    prediction_buffer = []

    while True:

        if stop_camera:
            st.warning("Camera stopped.")
            break

        ret, frame = cap.read()
        if not ret:
            st.error("Camera not working.")
            break

        img = cv2.resize(frame,(224,224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_input = preprocess_input(img)
        img_input = np.expand_dims(img_input, axis=0)

        prediction = model.predict(img_input, verbose=0)[0][0]

        prediction_buffer.append(prediction)
        if len(prediction_buffer) > BUFFER_SIZE:
            prediction_buffer.pop(0)

        avg_prediction = np.mean(prediction_buffer)

        label = decide_label(avg_prediction, last_label)
        last_label = label

        text_color = (0,255,0) if label=="Fresh" else (0,0,255)

        cv2.putText(frame, f"{label}: {avg_prediction:.2f}",
                    (20,50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, text_color, 2)

        frame_placeholder.image(frame, channels="BGR")

        rotten_percent = avg_prediction * 100
        fresh_percent = (1 - avg_prediction) * 100

        fig, ax = plt.subplots(figsize=(6,4))
        bars = ax.bar(["Fresh", "Rotten"], [fresh_percent, rotten_percent])

        bars[0].set_color("#27ae60")
        bars[1].set_color("#e74c3c")

        ax.set_ylim(0,100)
        ax.set_ylabel("Confidence (%)")
        ax.set_title("Real-Time Freshness Confidence", fontweight="bold")

        for i, v in enumerate([fresh_percent, rotten_percent]):
            ax.text(i, v + 2, f"{v:.1f}%", ha='center', fontweight='bold')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        fig_placeholder.pyplot(fig)

        progress_placeholder.progress(
            int(rotten_percent) if label=="Rotten"
            else int(fresh_percent)
        )

        time.sleep(0.1)

    cap.release()