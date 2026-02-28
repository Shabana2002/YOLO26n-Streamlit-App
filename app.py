import streamlit as st

# 1. MUST BE THE VERY FIRST COMMAND
st.set_page_config(page_title="YOLO26n Detection App", layout="wide")

import cv2
import numpy as np
from PIL import Image
import os
import av
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode

# ----------------------------
# SETTINGS & MODEL LOADING
# ----------------------------
MODEL_PATH = "weights/best.pt"

RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302", "stun:stun3.l.google.com:19302"]},
        ],
        "iceTransportPolicy": "all",
    }
)

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# ----------------------------
# UI ELEMENTS (Define 'option' here first!)
# ----------------------------
st.title("YOLO26n Object Detection App 🕵️‍♂️")
st.sidebar.warning("Cloud Version: Processing 1 frame per second for stability.")

# This creates the 'option' variable so the code below knows what it is
option = st.sidebar.radio("Select Input Type", ["Webcam", "Image Upload"])

# ----------------------------
# WEBCAM SECTION
# ----------------------------
if option == "Webcam":
    st.subheader("Webcam Live Prediction")

    # This class handles the video internally so the UI doesn't glitch/refresh
    class YOLOProcessor:
        def __init__(self):
            self.count = 0

        def recv(self, frame):
            self.count += 1
            img = frame.to_ndarray(format="bgr24")

            # Process 1 frame every 1.5 seconds to keep the Cloud CPU happy
            if self.count % 15 == 0:
                results = model.predict(img, imgsz=160, conf=0.3, verbose=False)
                annotated_frame = results[0].plot()
                return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

            # Return plain frame for other frames
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_streamer(
        key="yolo-final-stable-v1",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=YOLOProcessor,
        async_processing=True,
        media_stream_constraints={
            "video": {
                "width": 320,
                "height": 240,
                "frameRate": 15
            },
            "audio": False,
        },
    )

# ----------------------------
# IMAGE UPLOAD SECTION
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