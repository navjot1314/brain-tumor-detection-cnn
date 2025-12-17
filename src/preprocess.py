# scripts/preprocess.py

import os
import cv2
import numpy as np

IMG_SIZE = 128

def load_image(img_path):
    """Load a single image and preprocess it"""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Image not found: {img_path}")
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
    return img

def load_dataset(data_dir):
    """Load all images from a dataset folder"""
    images = []
    labels = []
    for label, category in enumerate(["no", "yes"]):
        folder = f"{data_dir}/{category}"
        for file in os.listdir(folder):
            img_path = f"{folder}/{file}"
            try:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                images.append(img)
                labels.append(label)
            except:
                continue
    images = np.array(images).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
    labels = np.array(labels)
    return images, labels
