import streamlit as st
# THIS MUST BE THE VERY FIRST STREAMLIT COMMAND
st.set_page_config(page_title="YOLO26n Detection App", layout="wide")

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
# UI ELEMENTS
# ----------------------------
st.title("YOLO26n Object Detection App 🕵️‍♂️")
option = st.sidebar.radio("Select Input Type", ["Webcam", "Image Upload"])

# ----------------------------
# WEBCAM (Cloud Optimized)
# ----------------------------
if option == "Webcam":
    st.subheader("Webcam Live Prediction")

    def video_frame_callback(frame):
        img = frame.to_ndarray(format="bgr24")
        # Keep imgsz small for Cloud CPU speed
        results = model.predict(img, imgsz=256, conf=0.25, verbose=False)
        annotated_frame = results[0].plot()
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

    webrtc_streamer(
        key="yolo-detection-v3",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_frame_callback=video_frame_callback,
        async_processing=True,
        media_stream_constraints={
            "video": {
                "width": {"max": 320},
                "height": {"max": 240},
                "frameRate": {"max": 10}
            },
            "audio": False,
        },
        desired_playing_state=True,
    )

# ----------------------------
# IMAGE UPLOAD
# ----------------------------
elif option == "Image Upload":
    st.subheader("Upload Images for Prediction")
    uploaded_files = st.file_uploader(
        "Upload images", type=["jpg", "jpeg", "png"], accept_multiple_files=True
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file).convert("RGB")
            img_array = np.array(image)
            results = model.predict(img_array, imgsz=416)
            st.image(results[0].plot(), use_container_width=True)