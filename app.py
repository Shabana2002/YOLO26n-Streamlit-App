import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os
import pandas as pd

# ----------------------------
# 1. PAGE CONFIG & 2025 UI STYLING
# ----------------------------
st.set_page_config(page_title="YOLO26n Object Detection", page_icon="🕵️‍♂️", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@800&family=Inter:wght@700&display=swap');

    .stApp { background-color: #121212; color: #E0E0E0; }

    h3, .stButton>button, .metric-card b {
        font-family: 'Montserrat', sans-serif !important;
        text-transform: uppercase;
        letter-spacing: -0.5px;
    }

    .main-title { 
        color: #ff4b4b; 
        text-align: center; 
        font-weight: 800; 
        font-size: 3.8rem; 
        letter-spacing: -2.5px; 
        margin-bottom: 0px; 
        font-family: 'Montserrat', sans-serif;
    }

    .sub-heading { 
        color: #FFFFFF; 
        text-align: center; 
        font-weight: 700; 
        font-size: 1.4rem; 
        letter-spacing: 0.5px; 
        margin-top: -10px; 
        margin-bottom: 35px; 
        font-family: 'Inter', sans-serif;
    }

    .stButton>button {
        border-radius: 12px;
        background: linear-gradient(145deg, #ff4b4b, #d43f3f);
        color: white;
        transition: all 0.3s ease;
        font-size: 0.9rem;
        height: 3.5rem;
    }
    .stButton>button:hover { transform: scale(1.03); box-shadow: 0px 8px 25px rgba(255, 75, 75, 0.4); }

    [data-testid="stSidebar"], .st-emotion-cache-1cv0edp {
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        padding: 20px;
        border-radius: 15px;
        border-left: 6px solid #ff4b4b;
        margin-bottom: 20px;
        line-height: 1.6;
    }
    .summary-item {
        color: #ff4b4b;
        font-weight: bold;
        text-transform: capitalize;
    }
    </style>
    """, unsafe_allow_html=True)

# ----------------------------
# 2. MODEL LOADING
# ----------------------------
MODEL_PATH = "weights/best.pt"


@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


model = load_model()

# ----------------------------
# 3. SIDEBAR
# ----------------------------
with st.sidebar:
    st.markdown("### ⚙️ CONFIGURATION")
    is_local = os.path.exists("C:/") or os.path.exists("/Users/") or os.path.exists("/home/adminuser") == False
    option = st.radio("SELECT MODE", ["Webcam", "Image Upload"] if is_local else ["Image Upload"])
    st.markdown("---")
    conf_threshold = st.slider("CONFIDENCE", 0.0, 1.0, 0.25)

st.markdown("<h1 class='main-title'>YOLO26n Object Detection</h1>", unsafe_allow_html=True)


# ----------------------------
# 4. PROCESSING LOGIC
# ----------------------------
def process_results(results):
    detections = []
    counts = {}
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            conf = float(box.conf[0])
            detections.append({"Class": label, "Confidence": f"{conf:.2%}"})
            counts[label] = counts.get(label, 0) + 1
    return detections, counts


# ----------------------------
# 5. WEBCAM SECTION
# ----------------------------
if option == "Webcam":
    st.markdown("<p class='sub-heading'>Live Webcam Real-Time Stream</p>", unsafe_allow_html=True)
    col_view, col_stats = st.columns([2, 1])

    with col_stats:
        st.markdown("### 🎮 CONTROLS")
        start_btn = st.button("START LIVE FEED")
        stop_btn = st.button("TERMINATE STREAM")
        st.markdown("---")
        st.markdown("### 📊 LIVE STATS")
        counter_placeholder = st.empty()
        table_placeholder = st.empty()

    with col_view:
        FRAME_WINDOW = st.image([], use_container_width=True)

    if start_btn:
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or stop_btn: break
            results = model.predict(frame, imgsz=416, conf=conf_threshold, verbose=False)
            FRAME_WINDOW.image(cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB))

            det_list, det_counts = process_results(results)

            # Clean Summary Formatting
            summary_html = "<br>".join([f"<span class='summary-item'>{k}</span>: {v}" for k, v in det_counts.items()])
            counter_placeholder.markdown(
                f"<div class='metric-card'><b>OBJECT COUNTER</b><br>{summary_html if summary_html else 'Searching...'}</div>",
                unsafe_allow_html=True)

            if det_list:
                table_placeholder.table(pd.DataFrame(det_list).head(10))
        cap.release()

# ----------------------------
# 6. IMAGE UPLOAD SECTION
# ----------------------------
elif option == "Image Upload":
    st.markdown("<p class='sub-heading'>Image Batch Analysis & Processing</p>", unsafe_allow_html=True)
    uploaded_files = st.file_uploader("UPLOAD IMAGES", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            img = np.array(Image.open(uploaded_file).convert("RGB"))
            results = model.predict(img, imgsz=640, conf=conf_threshold, verbose=False)

            col_img, col_data = st.columns([2, 1])
            with col_img:
                st.image(results[0].plot(), use_container_width=True)

            with col_data:
                det_list, det_counts = process_results(results)

                # Clean Summary Formatting for Images
                summary_html = "<br>".join(
                    [f"<span class='summary-item'>{k}</span>: {v}" for k, v in det_counts.items()])
                st.markdown(
                    f"<div class='metric-card'><b>SUMMARY</b><br>{summary_html if summary_html else 'No detections'}</div>",
                    unsafe_allow_html=True)

                if det_list:
                    st.dataframe(pd.DataFrame(det_list), use_container_width=True, hide_index=True)
            st.markdown("---")