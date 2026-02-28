import gradio as gr
from ultralytics import YOLO
import numpy as np
from PIL import Image
import os

# ----------------------------
# SETTINGS
# ----------------------------
MODEL_PATH = "weights/best.pt"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load YOLO model
model = YOLO(MODEL_PATH)

# ----------------------------
# DETECTION FUNCTION
# ----------------------------
def detect(input_data):
    """
    Detect objects using YOLO.
    Supports both webcam frames (video) and uploaded images.
    """
    # If input is PIL Image (from upload)
    if isinstance(input_data, Image.Image):
        img_array = np.array(input_data.convert("RGB"))
    else:
        # input_data is already an np.array from Gradio webcam capture
        img_array = input_data

    # YOLO prediction
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
    live=True,
)

iface.launch()