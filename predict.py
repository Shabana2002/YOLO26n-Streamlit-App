from ultralytics import YOLO
import os

# Load your trained model
model = YOLO("runs/detect/yolo26_cpu_50002/weights/best.pt")

# List of image paths
image_paths = [
    r"C:\Users\hp\Downloads\COCO_Dataset_old\subset\images\val\COCO_train2014_000000009960.jpg",
    r"C:\Users\hp\Downloads\COCO_Dataset_old\subset\images\val\COCO_train2014_000000063378.jpg",
    r"C:\Users\hp\Downloads\COCO_Dataset_old\subset\images\val\COCO_train2014_000000051571.jpg",
    r"C:\Users\hp\Downloads\COCO_Dataset_old\subset\images\val\COCO_train2014_000000577940.jpg",
    r"C:\Users\hp\Downloads\COCO_Dataset_old\subset\images\val\COCO_train2014_000000561635.jpg"
]

# Create output folder if it doesn't exist
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Loop through each image
for img_path in image_paths:
    results = model.predict(img_path, imgsz=416)  # run prediction
    results[0].show()  # display
    results[0].save(output_dir)  # save annotated image

print("All images processed and saved to 'output/' folder.")
# Run prediction using the default webcam (0)
results = model.predict(source=0,  # 0 = first webcam
                        imgsz=416,  # image size
                        show=True,  # display the video
                        save=True,  # save annotated video frames
                        project="webcam_output",  # folder to save results
                        name="yolo_webcam")  # subfolder name

print("Webcam prediction finished. Check the 'webcam_output/yolo_webcam' folder for results.")