import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os
import av  # New requirement for WebRTC
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# ----------------------------
# SETTINGS
# ----------------------------
MODEL_PATH = "weights/best.pt"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


model = load_model()

# ----------------------------
# STREAMLIT PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="YOLO26n Detection App", layout="wide")
st.title("YOLO26n Object Detection App 🕵️‍♂️")

option = st.sidebar.radio("Select Input Type", ["Webcam", "Image Upload"])

# ----------------------------
# WEBCAM (Cloud Compatible)
# ----------------------------
if option == "Webcam":
    st.subheader("Webcam Live Prediction")


    class YOLOVideoProcessor(VideoProcessorBase):
        def recv(self, frame):
            # Convert WebRTC frame to numpy array (BGR)
            img = frame.to_ndarray(format="bgr24")

            # YOLO Prediction (Same logic as your original)
            results = model.predict(img, imgsz=416)
            annotated_frame = results[0].plot()

            # Return the annotated frame back to the browser
            return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")


    # This replaces the while loop and cv2.VideoCapture
    webrtc_streamer(
        key="yolo-detection",
        video_processor_factory=YOLOVideoProcessor,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        media_stream_constraints={"video": True, "audio": False},
    )

# ----------------------------
# IMAGE UPLOAD (Kept exactly the same)
# ----------------------------
elif option == "Image Upload":
    st.subheader("Upload Images for Prediction")
    uploaded_files = st.file_uploader(
        "Upload 1 or more images", type=["jpg", "jpeg", "png"], accept_multiple_files=True
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file).convert("RGB")
            img_array = np.array(image)

            results = model.predict(img_array, imgsz=416)
            annotated_img = results[0].plot()

            st.image(annotated_img, width=800)

            out_path = os.path.join(OUTPUT_DIR, f"pred_{uploaded_file.name}")
            cv2.imwrite(out_path, cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))