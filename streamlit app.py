import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os

# ----------------------------
# SETTINGS
# ----------------------------

MODEL_PATH = "weights/best.pt"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load YOLO model safely for Streamlit Cloud
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# ----------------------------
# STREAMLIT PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="YOLO26n Detection App", layout="wide")
st.title("YOLO26n Object Detection App 🕵️‍♂️")

# ----------------------------
# SIDEBAR OPTIONS
# ----------------------------
option = st.sidebar.radio("Select Input Type", ["Webcam", "Image Upload"])  # Webcam first

# ----------------------------
# WEBCAM
# ----------------------------
if option == "Webcam":
    st.subheader("Webcam Live Prediction")

    # Start/Stop buttons
    start_webcam = st.button("Start Webcam")
    stop_webcam = st.button("Stop Webcam")

    # Use session state to control webcam
    if "webcam_active" not in st.session_state:
        st.session_state.webcam_active = False

    if start_webcam:
        st.session_state.webcam_active = True
    if stop_webcam:
        st.session_state.webcam_active = False

    FRAME_WINDOW = st.image([])

    if st.session_state.webcam_active:
        cap = cv2.VideoCapture(0)

        while st.session_state.webcam_active:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to open webcam.")
                break

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # YOLO Prediction
            results = model.predict(frame_rgb, imgsz=416)
            annotated_frame = results[0].plot()

            # Show live frame
            FRAME_WINDOW.image(annotated_frame, width=800)

        cap.release()
        cv2.destroyAllWindows()
        st.session_state.webcam_active = False

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
            # Convert uploaded file to OpenCV format
            image = Image.open(uploaded_file).convert("RGB")
            img_array = np.array(image)

            # YOLO Prediction
            results = model.predict(img_array, imgsz=416)
            annotated_img = results[0].plot()

            # Show annotated image
            st.image(annotated_img, width=800)

            # Save output without printing messages
            out_path = os.path.join(OUTPUT_DIR, f"pred_{uploaded_file.name}")
            cv2.imwrite(out_path, cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))