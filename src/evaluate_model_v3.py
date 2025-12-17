import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- SETTINGS ---
MODEL_PATH = "models/vgg16_brain_tumor.keras"
TEST_DIR = "data/test"
IMG_SIZE = (150, 150)  #  match training size
BATCH_SIZE = 16        #  match training batch size

print("[INFO] Loading trained model...")
model = load_model(MODEL_PATH)
print(" Model loaded successfully!")

# --- Load Test Dataset ---
print("[INFO] Loading test dataset...")

test_datagen = ImageDataGenerator(rescale=1.0/255)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    color_mode="grayscale",  #  match training input shape
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

# --- Evaluate Model ---
print("[INFO] Evaluating model on test dataset...")
loss, accuracy = model.evaluate(test_generator, verbose=1)
print(f"\n📊 Test Accuracy: {accuracy * 100:.2f}%")
print(f"📉 Test Loss: {loss:.4f}")

# --- Predictions ---
print("[INFO] Generating predictions...")
predictions = model.predict(test_generator)
predicted_classes = (predictions > 0.5).astype("int32").flatten()

true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

print("\n🧾 Classification Report:")
print(classification_report(true_classes, predicted_classes, target_names=class_labels, zero_division=0))

print("\n Confusion Matrix:")
cm = confusion_matrix(true_classes, predicted_classes)
print(cm)

# --- Save Predictions to CSV ---
print("[INFO] Saving predictions to CSV...")
filenames = test_generator.filenames
confidences = predictions.flatten()
df = pd.DataFrame({
    "Filename": filenames,
    "True Label": true_classes,
    "Predicted Label": predicted_classes,
    "Confidence": confidences
})
df.to_csv("evaluation_results.csv", index=False)
print("Saved predictions to evaluation_results.csv")

# --- Visualize Confusion Matrix ---
print("[INFO] Plotting confusion matrix...")
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# --- ROC Curve ---
print("[INFO] Plotting ROC curve...")
fpr, tpr, _ = roc_curve(true_classes, confidences)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
