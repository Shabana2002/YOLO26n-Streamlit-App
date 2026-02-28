import os
import random
import shutil

# ==============================
# SETTINGS
# ==============================

TOTAL_IMAGES = 5000
TRAIN_SPLIT = 0.8
RANDOM_SEED = 42

# ==============================
# PATH SETUP (AUTO-DETECT BASE)
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SOURCE_IMG = os.path.join(BASE_DIR, "images", "train2014")
SOURCE_LBL = os.path.join(BASE_DIR, "labels", "train2014")

DEST_BASE = os.path.join(BASE_DIR, "subset")

DEST_IMG_TRAIN = os.path.join(DEST_BASE, "images", "train")
DEST_LBL_TRAIN = os.path.join(DEST_BASE, "labels", "train")
DEST_IMG_VAL = os.path.join(DEST_BASE, "images", "val")
DEST_LBL_VAL = os.path.join(DEST_BASE, "labels", "val")

# ==============================
# CREATE DESTINATION FOLDERS
# ==============================

for folder in [DEST_IMG_TRAIN, DEST_LBL_TRAIN, DEST_IMG_VAL, DEST_LBL_VAL]:
    os.makedirs(folder, exist_ok=True)

# ==============================
# VERIFY SOURCE PATHS
# ==============================

if not os.path.exists(SOURCE_IMG):
    raise FileNotFoundError(f"Image folder not found: {SOURCE_IMG}")

if not os.path.exists(SOURCE_LBL):
    raise FileNotFoundError(f"Label folder not found: {SOURCE_LBL}")

# ==============================
# LOAD IMAGES
# ==============================

all_images = [f for f in os.listdir(SOURCE_IMG)
              if f.lower().endswith(('.jpg', '.png'))]

print(f"Total images available: {len(all_images)}")

if len(all_images) == 0:
    raise ValueError("No images found in source folder!")

# Prevent sampling error
TOTAL_IMAGES = min(TOTAL_IMAGES, len(all_images))

# Reproducible shuffle
random.seed(RANDOM_SEED)
random.shuffle(all_images)

selected = all_images[:TOTAL_IMAGES]

split_index = int(TRAIN_SPLIT * TOTAL_IMAGES)
train_images = selected[:split_index]
val_images = selected[split_index:]

# ==============================
# COPY FUNCTION
# ==============================

def copy_files(image_list, dest_img, dest_lbl):
    copied = 0

    for img in image_list:
        lbl = os.path.splitext(img)[0] + ".txt"

        src_img_path = os.path.join(SOURCE_IMG, img)
        src_lbl_path = os.path.join(SOURCE_LBL, lbl)

        if os.path.exists(src_lbl_path):
            shutil.copy2(src_img_path, os.path.join(dest_img, img))
            shutil.copy2(src_lbl_path, os.path.join(dest_lbl, lbl))
            copied += 1

    return copied


# ==============================
# COPY TRAIN & VAL
# ==============================

train_copied = copy_files(train_images, DEST_IMG_TRAIN, DEST_LBL_TRAIN)
val_copied = copy_files(val_images, DEST_IMG_VAL, DEST_LBL_VAL)

# ==============================
# SUMMARY
# ==============================

print("\n========== SUMMARY ==========")
print(f"Requested images: {TOTAL_IMAGES}")
print(f"Training images copied: {train_copied}")
print(f"Validation images copied: {val_copied}")
print("Subset created successfully!")
print("=============================")


print("\n=== VERIFICATION ===")

def count_files(folder):
    return len([
        f for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f))
    ])

print("Train images:", count_files(DEST_IMG_TRAIN))
print("Train labels:", count_files(DEST_LBL_TRAIN))
print("Val images:", count_files(DEST_IMG_VAL))
print("Val labels:", count_files(DEST_LBL_VAL))