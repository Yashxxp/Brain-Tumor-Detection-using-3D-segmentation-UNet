import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv3DTranspose
from sklearn.metrics import accuracy_score, f1_score
import nibabel as nib
from tqdm import tqdm
from scipy.ndimage import zoom  # for resizing 3D volumes

# ========== 1. Patch Conv3DTranspose ==========

class PatchedConv3DTranspose(Conv3DTranspose):
    def __init__(self, *args, **kwargs):
        kwargs.pop("groups", None)
        super().__init__(*args, **kwargs)

custom_objects = {'Conv3DTranspose': PatchedConv3DTranspose}

# ========== 2. Load model ==========

model = tf.keras.models.load_model('finalvalaug.h5', custom_objects=custom_objects)
model.summary()

# ========== 3. Load NIfTI volumes from folder ==========

def load_nifti_folder(image_dir):
    X = []
    filenames = sorted(os.listdir(image_dir))
    print(f"Found {len(filenames)} files in {image_dir}")

    target_shape = (160, 192, 128)

    for fname in tqdm(filenames, desc="Loading NIfTI data"):
        img_path = os.path.join(image_dir, fname)

        try:
            img = nib.load(img_path).get_fdata()
        except Exception as e:
            print(f"Error loading {fname}: {e}")
            continue

        # Handle 3D and 4D images
        if img.ndim == 3:
            zoom_factors = (
                target_shape[0] / img.shape[0],
                target_shape[1] / img.shape[1],
                target_shape[2] / img.shape[2]
            )
            img_resized = zoom(img, zoom_factors, order=1)
            img_resized = img_resized[..., np.newaxis]  # add channel dim
        elif img.ndim == 4:
            zoom_factors = (
                target_shape[0] / img.shape[0],
                target_shape[1] / img.shape[1],
                target_shape[2] / img.shape[2],
                1  # don't change number of channels
            )
            img_resized = zoom(img, zoom_factors, order=1)
        else:
            print(f"Unsupported image shape {img.shape} in file {fname}")
            continue

        X.append(img_resized)

    if not X:  # Check if no data was loaded
        print("No data was loaded. Please check your image directory.")
    return np.array(X)

image_dir = 'images'  # Folder with image .nii.gz files

# Load the validation data
X_val = load_nifti_folder(image_dir)

# Check if data is loaded properly
if X_val.shape[0] == 0:
    print("No validation data found. Exiting.")
    exit()

print(f"Loaded validation data: {X_val.shape[0]} samples")

# ========== 4. Prediction ==========

batch_size = 2  # Adjust based on your system
y_pred_logits = model.predict(X_val, batch_size=batch_size)
y_pred = np.argmax(y_pred_logits, axis=-1)  # Convert softmax output to class predictions

# Since labels are not provided, you may need to create a method for calculating metrics.
# Here's a placeholder for your label data:
y_val = np.zeros_like(y_pred)  # Placeholder: create dummy labels matching prediction shape You will need real labels.

# ========== 5. Flatten for metric calculation ==========

y_pred_flat = y_pred.flatten()
y_true_flat = y_val.flatten()

# ========== 6. Metrics ==========

accuracy = accuracy_score(y_true_flat, y_pred_flat)
f1 = f1_score(y_true_flat, y_pred_flat, average='weighted')

# Dice score for each class
def dice_per_class(y_true, y_pred, num_classes):
    dice_scores = []
    for i in range(num_classes):
        y_true_i = (y_true == i).astype(np.uint8)
        y_pred_i = (y_pred == i).astype(np.uint8)
        intersection = np.sum(y_true_i * y_pred_i)
        union = np.sum(y_true_i) + np.sum(y_pred_i)
        dice = (2. * intersection) / (union + 1e-6)
        dice_scores.append(dice)
    return dice_scores

num_classes = y_pred_logits.shape[-1]
dice_scores = dice_per_class(y_true_flat, y_pred_flat, num_classes)
mean_dice = np.mean(dice_scores)

# ========== 7. Print Results ==========

# ========== Tumor Type Classification ==========

tumor_class_map = {
    1: 'Edema',
    2: 'Non-enhancing Tumor',
    3: 'Enhancing Tumor'
}

print("\nDetected Tumor Types per Sample:")
print("\nTumor Volume (in mm³) per Sample:")
print("\nTumor Volume Per Class (in mm³) per Sample:")
print("\nEvaluation Results:")
for i in range(y_pred.shape[0]):
    unique_classes = np.unique(y_pred[i])
    detected_tumors = [tumor_class_map[c] for c in unique_classes if c in tumor_class_map]
    print(f"Sample {i+1}: {', '.join(detected_tumors) if detected_tumors else 'No tumor detected'}")

# ========== Tumor Volume Estimation ==========

voxel_volume_mm3 = 1.0  # Assumes isotropic voxel size 1x1x1 mm (adjust if needed)
print("\nTumor Volume (in mm³) per Sample:")
for i in range(y_pred.shape[0]):
    tumor_voxels = np.sum(y_pred[i] > 0)  # excludes background (class 0)
    tumor_volume = tumor_voxels * voxel_volume_mm3
    print(f"Sample {i+1}: {tumor_volume:.2f} mm³")

print("\nTumor Volume Per Class (in mm³) per Sample:")
for i in range(y_pred.shape[0]):
    print(f"Sample {i+1}:")
    for class_id, class_name in tumor_class_map.items():
        class_voxels = np.sum(y_pred[i] == class_id)
        class_volume = class_voxels * voxel_volume_mm3
        print(f"  {class_name}: {class_volume:.2f} mm³")

print("\nEvaluation Results:")
print(f"Accuracy       : {accuracy:.4f}")
print(f"F1 Score       : {f1:.4f}")
print(f"Mean Dice Score: {mean_dice:.4f}")
for i, dice in enumerate(dice_scores):
    print(f"Dice Score (Class {i}): {dice:.4f}")
