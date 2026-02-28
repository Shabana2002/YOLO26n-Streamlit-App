from ultralytics import YOLO

# Load the latest trained model
model = YOLO("runs/detect/yolo26_cpu_50002/weights/best.pt")

# Validate on your dataset
metrics = model.val(data="data.yaml")

print("\nFinal Validation Metrics:")
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
print(f"Precision: {metrics.box.mp:.4f}")
print(f"Recall: {metrics.box.mr:.4f}")