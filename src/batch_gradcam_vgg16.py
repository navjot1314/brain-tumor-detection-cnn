import numpy as np
import tensorflow as tf
import cv2
import os
import csv
from glob import glob

# --- CONFIG ---
MODEL_PATH = "models/vgg16_brain_tumor.keras"
TEST_DIR = "data/test"
RESULT_DIR = "results/brain"
IMG_SIZE = (150, 150)

# --- LOAD TRAINED MODEL ---
orig_model = tf.keras.models.load_model(MODEL_PATH)
print(" Model loaded.")

# Reuse layer instances
rgb_layer = orig_model.get_layer("conv2d")
vgg = orig_model.get_layer("vgg16")
flatten = orig_model.get_layer("flatten")
dense = orig_model.get_layer("dense")
dropout = orig_model.get_layer("dropout")
dense_1 = orig_model.get_layer("dense_1")

# --- REBUILD UNIFIED GRAPH ---
input_tensor = tf.keras.layers.Input(shape=(*IMG_SIZE, 1), name="input_gray")
x = rgb_layer(input_tensor)

block5_conv3_output = None
for layer in vgg.layers[1:]:
    x = layer(x)
    if layer.name == "block5_conv3":
        block5_conv3_output = x

y = flatten(x)
y = dense(y)
y = dropout(y)
output = dense_1(y)

model = tf.keras.models.Model(inputs=input_tensor, outputs=output)
grad_model = tf.keras.models.Model(inputs=model.input, outputs=[block5_conv3_output, model.output])
print(" Unified graph rebuilt.")

# --- PREP OUTPUT FOLDERS ---
os.makedirs(f"{RESULT_DIR}/yes", exist_ok=True)
os.makedirs(f"{RESULT_DIR}/no", exist_ok=True)
csv_path = f"{RESULT_DIR}/brain_predictions.csv"

# --- PROCESS IMAGES ---
with open(csv_path, mode="w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["filename", "class", "prediction"])

    for label in ["yes", "no"]:
        image_paths = glob(f"{TEST_DIR}/{label}/*.JPG")
        for img_path in image_paths:
            filename = os.path.basename(img_path)
            print(f" Processing: {filename}")

            # Load and preprocess image
            img = tf.keras.preprocessing.image.load_img(img_path, color_mode="grayscale", target_size=IMG_SIZE)
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0).astype("float32") / 255.0
            img_tensor = tf.convert_to_tensor(img_array)

            # Grad-CAM
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(img_tensor)
                loss = predictions[:, 0]

            grads = tape.gradient(loss, conv_outputs)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            conv_outputs = conv_outputs[0]
            heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
            heatmap = tf.maximum(heatmap, 0)
            heatmap /= (tf.reduce_max(heatmap) + 1e-8)

            # Overlay
            heatmap_np = heatmap.numpy()
            heatmap_np = cv2.resize(heatmap_np, IMG_SIZE)
            heatmap_np = np.uint8(255 * heatmap_np)
            heatmap_color = cv2.applyColorMap(heatmap_np, cv2.COLORMAP_JET)

            original = cv2.imread(img_path)
            original = cv2.resize(original, IMG_SIZE)
            overlay = cv2.addWeighted(original, 0.6, heatmap_color, 0.4, 0)

            # Save
            out_path = f"{RESULT_DIR}/{label}/{filename}"
            cv2.imwrite(out_path, overlay)

            # Log
            pred_score = predictions.numpy()[0, 0]
            writer.writerow([filename, label, f"{pred_score:.4f}"])

print(f"\n All images processed. Overlays saved to `{RESULT_DIR}/yes/` and `{RESULT_DIR}/no/`.")
print(f" Prediction log saved to `{csv_path}`.")
