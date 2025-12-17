"""
CNN Training Module

Defines and trains a Convolutional Neural Network
for brain tumor classification using MRI images.
"""

# train_model_fixed.py

import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import shuffle

# Paths
TRAIN_DIR = "data/train"
TEST_DIR = "data/test"
IMG_SIZE = 128

# Function to load data
def load_data(data_dir):
    images = []
    labels = []
    for label, category in enumerate(["no", "yes"]):
        folder = os.path.join(data_dir, category)
        for file in os.listdir(folder):
            img_path = os.path.join(folder, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                images.append(img)
                labels.append(label)
    images = np.array(images).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
    labels = np.array(labels)
    return shuffle(images, labels, random_state=42)

# Load train and test data
X_train, y_train = load_data(TRAIN_DIR)
X_test, y_test = load_data(TEST_DIR)

print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(X_train)

# Build improved CNN
model = Sequential([
    Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),   # increased dropout to reduce overfitting
    Dense(1, activation='sigmoid')
])

# Compile
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train with augmentation
history = model.fit(datagen.flow(X_train, y_train, batch_size=16),
                    epochs=15,
                    validation_data=(X_test, y_test))

# Save in modern Keras format
os.makedirs("models", exist_ok=True)
model.save("models/brain_tumor_model.keras")
print("✅ Improved model saved as models/brain_tumor_model.keras")
