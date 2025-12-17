"""
VGG16-Based Tumor Classification

Implements transfer learning using a pretrained VGG16
network for brain tumor classification from MRI images.
"""


import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, Conv2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight

# --- CONFIG ---
IMG_SIZE = (150, 150)
BATCH_SIZE = 16
EPOCHS = 50
LR = 0.0005
DATA_TRAIN = "data/train"
DATA_VAL = "data/test"
MODEL_PATH = "models/vgg16_brain_tumor.keras"

# --- DATA PREPROCESSING ---
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=25,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest"
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    DATA_TRAIN,
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

val_generator = val_datagen.flow_from_directory(
    DATA_VAL,
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

# --- CLASS WEIGHTS ---
labels = train_generator.classes
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(labels),
    y=labels
)
class_weights = dict(enumerate(class_weights))
print("[INFO] Class Weights:", class_weights)

# --- VGG16 BASE (adapted for grayscale) ---
input_tensor = Input(shape=(*IMG_SIZE, 1))
x = Conv2D(3, (3, 3), padding="same")(input_tensor)  # Convert grayscale to 3 channels
vgg_base = VGG16(include_top=False, weights="imagenet", input_shape=(*IMG_SIZE, 3))
vgg_output = vgg_base(x)

# --- CUSTOM HEAD ---
x = Flatten()(vgg_output)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=input_tensor, outputs=output)
model.compile(optimizer=Adam(learning_rate=LR), loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

# --- CALLBACKS ---
checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=7,
    restore_best_weights=True,
    verbose=1
)

# --- TRAIN ---
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    class_weight=class_weights,
    callbacks=[checkpoint, early_stop]
)

print("Training complete. Best model saved as:", MODEL_PATH)

