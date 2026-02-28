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
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}]}
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


    # Define the processing function (Modern way)
    def video_frame_callback(frame):
        img = frame.to_ndarray(format="bgr24")

        # Performance Tip: Resize image before prediction to speed up CPU inference
        # YOLO will handle the resizing, but 320 is faster than 416/640
        results = model.predict(img, imgsz=320, conf=0.25, verbose=False)
        annotated_frame = results[0].plot()

        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")


    # Use the modern webrtc_streamer setup
    webrtc_streamer(
        key="yolo-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_frame_callback=video_frame_callback,  # Use callback instead of ProcessorClass
        media_stream_constraints={
            "video": {
                "width": {"ideal": 640},
                "height": {"ideal": 480},
                "frameRate": {"ideal": 15}  # Lower FPS = smoother processing on CPU
            },
            "audio": False
        },
        async_processing=True,  # FIXED: This stops the app from freezing while YOLO "thinks"
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
            st.image(results[0].plot(), width=800)