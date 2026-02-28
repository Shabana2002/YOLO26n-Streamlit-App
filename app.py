import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os
import av
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode

# ----------------------------
# SETTINGS & MODEL LOADING
# ----------------------------
MODEL_PATH = "weights/best.pt"

# Enhanced STUN settings to bypass firewalls
RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302", "stun:stun3.l.google.com:19302"]},
            {"urls": ["stun:stun4.l.google.com:19302"]},
        ],
        "iceTransportPolicy": "all",
    }
)

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
# WEBCAM (Cloud Optimized)
# ----------------------------
if option == "Webcam":
    st.subheader("Webcam Live Prediction")

    # The modern callback function
    def video_frame_callback(frame):
        img = frame.to_ndarray(format="bgr24")

        # Performance Tip: imgsz=256 is much faster on Cloud CPUs
        results = model.predict(img, imgsz=256, conf=0.25, verbose=False)
        annotated_frame = results[0].plot()

        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

    # Use the optimized streamer settings
    webrtc_streamer(
        key="yolo-detection-v2",  # Changed key to force a refresh
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_frame_callback=video_frame_callback,
        async_processing=True,
        media_stream_constraints={
            "video": {
                "width": {"max": 320},   # Force small to save CPU
                "height": {"max": 240},
                "frameRate": {"max": 10} # 10 FPS is enough for a cloud demo
            },
            "audio": False,
        },
        desired_playing_state=True, # Helps the play button respond faster
    )

# ----------------------------
# IMAGE UPLOAD
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
            st.image(results[0].plot(), use_container_width=True)