"""
U-Net Brain Tumor Segmentation

Implements a U-Net based convolutional neural network
for pixel-level brain tumor segmentation on MRI images.
"""


import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

# --- CONFIG ---
IMG_SIZE = (256, 256)
IMAGE_DIR = "C:/Users/navjo/Videos/TumorRecognitionAI/images/brain"
MASK_DIR = "C:/Users/navjo/Videos/TumorRecognitionAI/masks/brain"
MODEL_PATH = "models/unet_brain_segmentation.keras"

# --- LOAD DATA ---
def load_data(image_dir, mask_dir):
    images, masks = [], []
    for filename in os.listdir(image_dir):
        img_path = os.path.join(image_dir, filename)
        mask_path = os.path.join(mask_dir, filename)
        if not os.path.exists(mask_path):
            continue

        img = load_img(img_path, color_mode="grayscale", target_size=IMG_SIZE)
        mask = load_img(mask_path, color_mode="grayscale", target_size=IMG_SIZE)

        img = img_to_array(img) / 255.0
        mask = img_to_array(mask) / 255.0
        mask = np.where(mask > 0.5, 1.0, 0.0)  # binarize

        images.append(img)
        masks.append(mask)

    return np.array(images), np.array(masks)

print("📦 Loading data...")
X, y = load_data(IMAGE_DIR, MASK_DIR)
print(f"✅ Loaded {len(X)} image-mask pairs.")

# --- SPLIT ---
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- BUILD U-NET ---
def build_unet(input_shape):
    inputs = layers.Input(input_shape)

    # Encoder
    c1 = layers.Conv2D(16, 3, activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(16, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D()(c1)

    c2 = layers.Conv2D(32, 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(32, 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D()(c2)

    c3 = layers.Conv2D(64, 3, activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(64, 3, activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D()(c3)

    # Bottleneck
    c4 = layers.Conv2D(128, 3, activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(128, 3, activation='relu', padding='same')(c4)

    # Decoder
    u5 = layers.UpSampling2D()(c4)
    u5 = layers.concatenate([u5, c3])
    c5 = layers.Conv2D(64, 3, activation='relu', padding='same')(u5)
    c5 = layers.Conv2D(64, 3, activation='relu', padding='same')(c5)

    u6 = layers.UpSampling2D()(c5)
    u6 = layers.concatenate([u6, c2])
    c6 = layers.Conv2D(32, 3, activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(32, 3, activation='relu', padding='same')(c6)

    u7 = layers.UpSampling2D()(c6)
    u7 = layers.concatenate([u7, c1])
    c7 = layers.Conv2D(16, 3, activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(16, 3, activation='relu', padding='same')(c7)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c7)

    return models.Model(inputs, outputs)

model = build_unet((*IMG_SIZE, 1))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# --- TRAIN ---
print("🚀 Training U-Net...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=8
)

# --- SAVE ---
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
model.save(MODEL_PATH)
print(f"Model saved to: {MODEL_PATH}")
