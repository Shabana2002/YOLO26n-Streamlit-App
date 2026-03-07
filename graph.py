import pandas as pd
import matplotlib.pyplot as plt
import os

# Folder path
folder_path = r"C:\Users\hp\Downloads\COCO_Dataset_old\runs\detect\yolo26_cpu_50002"

# CSV file path
csv_path = os.path.join(folder_path, "results.csv")

# Read CSV
data = pd.read_csv(csv_path)

print("Columns found in CSV:")
print(data.columns)

# -----------------------------
# 1. Training Metrics Graph
# -----------------------------
plt.figure(figsize=(10,6))

plt.plot(data['epoch'], data['metrics/precision(B)'], label="Precision")
plt.plot(data['epoch'], data['metrics/recall(B)'], label="Recall")
plt.plot(data['epoch'], data['metrics/mAP50(B)'], label="mAP50")
plt.plot(data['epoch'], data['metrics/mAP50-95(B)'], label="mAP50-95")

plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("YOLO Training Metrics")
plt.legend()
plt.grid(True)

save_path = os.path.join(folder_path, "training_metrics.png")
plt.savefig(save_path)
plt.close()


# -----------------------------
# 2. Training Loss Graph
# -----------------------------
plt.figure(figsize=(10,6))

plt.plot(data['epoch'], data['train/box_loss'], label="Box Loss")
plt.plot(data['epoch'], data['train/cls_loss'], label="Class Loss")
plt.plot(data['epoch'], data['train/dfl_loss'], label="DFL Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss vs Epoch")
plt.legend()
plt.grid(True)

save_path = os.path.join(folder_path, "training_loss.png")
plt.savefig(save_path)
plt.close()


# -----------------------------
# 3. Validation Loss Graph
# -----------------------------
plt.figure(figsize=(10,6))

plt.plot(data['epoch'], data['val/box_loss'], label="Val Box Loss")
plt.plot(data['epoch'], data['val/cls_loss'], label="Val Class Loss")
plt.plot(data['epoch'], data['val/dfl_loss'], label="Val DFL Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Validation Loss vs Epoch")
plt.legend()
plt.grid(True)

save_path = os.path.join(folder_path, "validation_loss.png")
plt.savefig(save_path)
plt.close()

print("Graphs successfully saved in:", folder_path)