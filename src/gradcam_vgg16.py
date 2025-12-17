import numpy as np
import tensorflow as tf
import cv2
import os

# --- CONFIG ---
IMAGE_PATH = "C:/Users/navjo/Videos/TumorRecognitionAI/data/test/yes/Y146.JPG"
RESULT_PATH = "results/gradcam_output.png"
MODEL_PATH = "models/vgg16_brain_tumor.keras"
IMG_SIZE = (150, 150)

# --- LOAD TRAINED MODEL ---
orig_model = tf.keras.models.load_model(MODEL_PATH)
print(" Model loaded directly from file.")

# Grab layer instances from the original model
rgb_layer = orig_model.get_layer("conv2d")          # grayscale -> RGB converter
vgg = orig_model.get_layer("vgg16")                 # VGG16 submodel
flatten = orig_model.get_layer("flatten")
dense = orig_model.get_layer("dense")
dropout = orig_model.get_layer("dropout")
dense_1 = orig_model.get_layer("dense_1")

# --- REBUILD A FRESH GRAPH USING THE SAME LAYER INSTANCES ---
# Create a new input (grayscale)
input_tensor = tf.keras.layers.Input(shape=(*IMG_SIZE, 1), name="input_gray_rebuild")

# Apply rgb converter
x = rgb_layer(input_tensor)

# Apply VGG16 layer-by-layer, skipping its InputLayer
block5_conv3_output = None
for layer in vgg.layers[1:]:  # skip vgg.layers[0] which is InputLayer
    x = layer(x)
    if layer.name == "block5_conv3":
        block5_conv3_output = x

# Classifier head
y = flatten(x)
y = dense(y)
y = dropout(y)
y = dense_1(y)

# Fresh unified model (same weights, same layers, new connected graph)
model = tf.keras.Model(inputs=input_tensor, outputs=y)
print(" Unified graph rebuilt with original weights.")

# --- LOAD IMAGE ---
print(f" Loading image from: {IMAGE_PATH}")
img = tf.keras.preprocessing.image.load_img(IMAGE_PATH, color_mode="grayscale", target_size=IMG_SIZE)
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0).astype("float32") / 255.0
img_tensor = tf.convert_to_tensor(img_array)

# Sanity check: forward pass
pred = model(img_tensor).numpy()[0, 0]
print(f"🔮 Prediction: {pred:.4f}")

# --- GRAD-CAM ---
# Build a grad_model that returns (block5_conv3 activations, prediction)
grad_model = tf.keras.Model(inputs=model.input, outputs=[block5_conv3_output, model.output])

with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(img_tensor)
    loss = predictions[:, 0]

# Gradients of the prediction w.r.t. block5_conv3
grads = tape.gradient(loss, conv_outputs)
if grads is None:
    raise ValueError("Gradients could not be computed. Check model connectivity.")

# Global-average-pool gradients over spatial dims
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # (C,)

# Weight conv outputs by pooled grads
conv_outputs = conv_outputs[0]                        # (H, W, C)
heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)  # (H, W)

# ReLU and normalize
heatmap = tf.maximum(heatmap, 0)
heatmap /= (tf.reduce_max(heatmap) + 1e-8)

# --- OVERLAY HEATMAP ---
heatmap_np = heatmap.numpy()
heatmap_np = cv2.resize(heatmap_np, IMG_SIZE)
heatmap_np = np.uint8(255 * heatmap_np)
heatmap_color = cv2.applyColorMap(heatmap_np, cv2.COLORMAP_JET)

original = cv2.imread(IMAGE_PATH)
original = cv2.resize(original, IMG_SIZE)
overlay = cv2.addWeighted(original, 0.6, heatmap_color, 0.4, 0)

os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
cv2.imwrite(RESULT_PATH, overlay)
print(f"Grad-CAM saved to: {RESULT_PATH}")

