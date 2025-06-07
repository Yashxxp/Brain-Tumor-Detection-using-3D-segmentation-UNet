import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import tensorflow as tf
from tqdm import tqdm

def load_nifti_image(filepath):
    return nib.load(filepath).get_fdata()

def preprocess_data(image_paths, label_paths):
    X, y = [], []
    for img_path, lbl_path in zip(image_paths, label_paths):
        img = load_nifti_image(img_path)
        lbl = load_nifti_image(lbl_path)

        # Ensure shape compatibility
        if img.shape != lbl.shape:
            print(f"Skipping file due to shape mismatch: {img_path}")
            continue

        X.append(img)
        y.append(lbl)

    X = np.expand_dims(np.array(X), -1)  # Add channel dimension
    y = np.array(y).astype(np.uint8)
    return X, y

def dice_score_per_class(y_true, y_pred, smooth=1e-6):
    scores = []
    for cls in range(4):
        y_true_cls = (y_true == cls).astype(np.float32)
        y_pred_cls = (y_pred == cls).astype(np.float32)
        intersection = np.sum(y_true_cls * y_pred_cls)
        union = np.sum(y_true_cls) + np.sum(y_pred_cls)
        score = (2. * intersection + smooth) / (union + smooth)
        scores.append(score)
    return scores

# Set correct directories
image_dir = 'images'
label_dir = 'masks'

image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.nii')])
label_paths = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith('.nii')])

# Load and preprocess data
X_data, y_data = preprocess_data(image_paths, label_paths)

# Make sure there's enough data
if len(X_data) < 2:
    raise ValueError("Not enough data. Add more image/label pairs in .nii format.")

# Normalize input images
X_data = X_data / np.max(X_data)

# Split data
X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.3, random_state=42)

# Load your trained model
model = load_model('finalvalaug.h5', compile=False)

# Run prediction
y_pred = model.predict(X_val)
y_pred_labels = np.argmax(y_pred, axis=-1)

# Compute Dice scores
dice_scores = dice_score_per_class(y_val, y_pred_labels)
mean_dice = np.mean(dice_scores)

print("\n=== Evaluation Results ===")
print(f"Mean Dice Score: {mean_dice:.4f}")
for i, score in enumerate(dice_scores):
    print(f"Dice Score - Class {i}: {score:.4f}")

# Visualization
idx = 0  # Visualize the first sample
mid_slice = y_val.shape[1] // 2  # Middle slice
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("Ground Truth")
plt.imshow(y_val[idx, mid_slice, :, :], cmap='viridis')

plt.subplot(1, 3, 2)
plt.title("Prediction")
plt.imshow(y_pred_labels[idx, mid_slice, :, :], cmap='viridis')

plt.subplot(1, 3, 3)
plt.title("Input Image")
plt.imshow(X_val[idx, mid_slice, :, :, 0], cmap='gray')

plt.tight_layout()
plt.show()
