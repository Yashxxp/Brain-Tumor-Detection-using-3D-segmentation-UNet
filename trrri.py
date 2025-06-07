import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import os
import nibabel as nib
from tqdm import tqdm
import config  # config.py must have correct paths
import Model   # Model.py must have Unet_3d, dice_coef, dice_coef_loss

# --- Model setup ---
input_img = Input((128, 128, 128, 1))  # Adjusted for single-channel input
model = Model.Unet_3d(input_img, 8, 0.1, True)

learning_rate = 0.001
epochs = 60

model.compile(
    optimizer=Adam(learning_rate=learning_rate),  # Fixed deprecated 'lr' and 'decay'
    loss=Model.dice_coef_loss,
    metrics=[Model.dice_coef]
)
model.summary()

# --- Data paths ---
all_images = sorted(os.listdir(config.IMAGES_DATA_DIR))
all_masks = sorted(os.listdir(config.LABELS_DATA_DIR))

# --- Training loop ---
for epoch in tqdm(range(epochs), desc="Epochs"):
    for num in tqdm(range(min(len(all_images), len(all_masks))), desc="Samples"):

        # Load image
        image_path = os.path.join(config.IMAGES_DATA_DIR, all_images[num])
        img = nib.load(image_path).get_fdata()
        if img.ndim == 3:
            img = np.expand_dims(img, axis=-1)  # shape becomes (240, 240, 155, 1)

        # Load mask
        mask_path = os.path.join(config.LABELS_DATA_DIR, all_masks[num])
        msk = nib.load(mask_path).get_fdata()

        # Crop
        cropped_img = img[56:184, 80:208, 13:141, :]
        cropped_msk = msk[56:184, 80:208, 13:141]

        # Reshape for training
        X = cropped_img.reshape(1, 128, 128, 128, 1)
        y = cropped_msk.reshape(1, 128, 128, 128)

        y[y == 4] = 3  # Normalize class label
        y = to_categorical(y, num_classes=4)

        # Train
        checkpoint_cb = ModelCheckpoint("braintumor_model.h5", save_best_only=True, verbose=1)
        model.fit(x=X, y=y, epochs=1, callbacks=[checkpoint_cb])

# Save final model
model.save('3d_braintumor_model_final.h5')
