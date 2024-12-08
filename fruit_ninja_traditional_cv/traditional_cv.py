import os
import glob
import cv2
import numpy as np

# Directory structure:
# test_images_dir: contains test images (e.g., .jpg, .png)
# test_labels_dir: contains YOLOv8 label files with the same base name as the images but .txt extension
# train_images_dir, train_labels_dir: similarly for training (to learn color ranges)

train_images_dir = "../fruit_ninja_yolov8/FruitSalad-1/train/images"
train_labels_dir = "../fruit_ninja_yolov8/FruitSalad-1/train/labels" 
test_images_dir = "../fruit_ninja_yolov8/FruitSalad-1/test/images"
test_labels_dir = "../fruit_ninja_yolov8/FruitSalad-1/test/labels"

image_ext = "*.jpg"  # Adjust if needed

def get_image_label_pairs(images_dir, labels_dir, image_ext="*.jpg"):
    image_files = sorted(glob.glob(os.path.join(images_dir, image_ext)))
    pairs = []
    for img_path in image_files:
        filename = os.path.basename(img_path)
        base, _ = os.path.splitext(filename)
        label_path = os.path.join(labels_dir, base + ".txt")
        if os.path.exists(label_path):
            pairs.append((img_path, label_path))
    return pairs

# Function to create a binary mask from YOLOv8-style labels
def create_mask_from_yolo_labels(img, label_path):
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    if not os.path.exists(label_path):
        return mask  # no annotations

    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) == 5:
            # YOLO format: class x_center y_center width height (all normalized)
            # Convert back to pixel coordinates
            cls, x_center, y_center, width, height = parts
            x_center, y_center, width, height = float(x_center), float(y_center), float(width), float(height)

            # Convert normalized coords to pixel coords
            x_center_pix = int(x_center * w)
            y_center_pix = int(y_center * h)
            w_pix = int(width * w)
            h_pix = int(height * h)

            x1 = max(0, x_center_pix - w_pix // 2)
            y1 = max(0, y_center_pix - h_pix // 2)
            x2 = min(w-1, x_center_pix + w_pix // 2)
            y2 = min(h-1, y_center_pix + h_pix // 2)

            # Fill in the bounding box region with 1s
            mask[y1:y2+1, x1:x2+1] = 255

    return mask

###########################################
# Step 1: Learn the fruit color from training images and labels

train_pairs = get_image_label_pairs(train_images_dir, train_labels_dir)

h_values, s_values, v_values = [], [], []

for img_path, label_path in train_pairs:
    img = cv2.imread(img_path)
    if img is None:
        continue

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gt_mask = create_mask_from_yolo_labels(img, label_path)

    fruit_pixels = hsv[gt_mask > 128]  # pixels inside bounding boxes
    if len(fruit_pixels) > 0:
        h_values.extend(fruit_pixels[:, 0])
        s_values.extend(fruit_pixels[:, 1])
        v_values.extend(fruit_pixels[:, 2])

if len(h_values) == 0:
    raise ValueError("No fruit pixels found in training data. Check your training labels or paths.")

# Compute mean and std of H, S, V
h_mean, s_mean, v_mean = np.mean(h_values), np.mean(s_values), np.mean(v_values)
h_std, s_std, v_std = np.std(h_values), np.std(s_values), np.std(v_values)

# Define a color range based on mean ± 2*std (Adjust as necessary)
h_lower = max(0, h_mean - 2*h_std)
h_upper = min(179, h_mean + 2*h_std)
s_lower = max(0, s_mean - 2*s_std)
s_upper = min(255, s_mean + 2*s_std)
v_lower = max(0, v_mean - 2*v_std)
v_upper = min(255, v_mean + 2*v_std)

lower_bound = (int(h_lower), int(s_lower), int(v_lower))
upper_bound = (int(h_upper), int(s_upper), int(v_upper))

###########################################
# Step 2: Evaluate on test images
test_pairs = get_image_label_pairs(test_images_dir, test_labels_dir)

TP, FP, FN = 0, 0, 0

for img_path, label_path in test_pairs:
    img = cv2.imread(img_path)
    if img is None:
        continue

    gt_mask = create_mask_from_yolo_labels(img, label_path)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    pred_mask = cv2.inRange(hsv, lower_bound, upper_bound)

    gt_binary = (gt_mask > 128).astype(np.uint8)
    pred_binary = (pred_mask > 128).astype(np.uint8)

    # Compute TP, FP, FN
    tp = np.sum((pred_binary == 1) & (gt_binary == 1))
    fp = np.sum((pred_binary == 1) & (gt_binary == 0))
    fn = np.sum((pred_binary == 0) & (gt_binary == 1))

    TP += tp
    FP += fp
    FN += fn

precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

print("Precision:", precision)
print("Recall:", recall)
