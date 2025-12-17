# scripts/predict.py

import sys
import numpy as np
from tensorflow.keras.models import load_model
from preprocess import load_image

MODEL_PATH = "models/brain_tumor_model.keras"

def predict_tumor(img_path):
    model = load_model(MODEL_PATH)
    img = load_image(img_path)
    pred = model.predict(img)[0][0]
    if pred > 0.5:
        print(" Tumor Detected!")
    else:
        print(" No Tumor Detected!")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)
    image_path = sys.argv[1]
    predict_tumor(image_path)
