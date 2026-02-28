import gradio as gr
from ultralytics import YOLO
import numpy as np
from PIL import Image
import os
import requests

# ----------------------------
# SETTINGS
# ----------------------------
MODEL_URL = "https://huggingface.co/Ambatt/yolo26n-best/resolve/main/best.pt"  # Replace Ambatt with your HF username
MODEL_PATH = "weights/best.pt"
OUTPUT_DIR = "output"

os.makedirs("weights", exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# DOWNLOAD MODEL IF NOT EXISTS
# ----------------------------
if not os.path.exists(MODEL_PATH):
    print("Downloading YOLO model...")
    r = requests.get(MODEL_URL, stream=True)
    with open(MODEL_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Download complete!")

# ----------------------------
# LOAD YOLO MODEL
# ----------------------------
model = YOLO(MODEL_PATH)

# ----------------------------
# DETECTION FUNCTION
# ----------------------------
def detect(input_type, input_image):
    """
    Detect objects using YOLO.
    Supports webcam frames and uploaded images.
    """
    if input_type == "Image Upload" and input_image is not None:
        img_array = np.array(input_image.convert("RGB"))
    elif input_type == "Webcam" and input_image is not None:
        img_array = input_image  # already numpy array from Gradio webcam
    else:
        return None

    results = model.predict(img_array, imgsz=416)
    annotated_img = results[0].plot()

    # Save output
    out_path = os.path.join(OUTPUT_DIR, "pred_output.png")
    Image.fromarray(annotated_img).save(out_path)

    return annotated_img

# ----------------------------
# GRADIO INTERFACE
# ----------------------------
iface = gr.Interface(
    fn=detect,
    inputs=[
        gr.Radio(["Webcam", "Image Upload"], label="Select Input Type"),
        gr.Image(source="webcam", type="numpy", tool="editor", label="Webcam / Upload Image")
    ],
    outputs=gr.Image(type="numpy", label="Detected Output"),
    live=True
)

iface.launch()