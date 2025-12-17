import os
import shutil
import random

# Paths
RAW_DATA_DIR = "data/raw"
TRAIN_DIR = "data/train"
TEST_DIR = "data/test"

# Train-test split ratio
SPLIT_RATIO = 0.8  # 80% train, 20% test

# Ensure directories exist
for folder in [TRAIN_DIR, TEST_DIR]:
    for category in ["yes", "no"]:
        os.makedirs(os.path.join(folder, category), exist_ok=True)

# Loop through each category
for category in ["yes", "no"]:
    category_path = os.path.join(RAW_DATA_DIR, category)
    images = os.listdir(category_path)
    random.shuffle(images)

    split_point = int(len(images) * SPLIT_RATIO)
    train_images = images[:split_point]
    test_images = images[split_point:]

    # Copy images to train folder
    for img in train_images:
        src = os.path.join(category_path, img)
        dst = os.path.join(TRAIN_DIR, category, img)
        shutil.copy(src, dst)

    # Copy images to test folder
    for img in test_images:
        src = os.path.join(category_path, img)
        dst = os.path.join(TEST_DIR, category, img)
        shutil.copy(src, dst)

print(" Dataset successfully split into train/ and test/ folders!")
