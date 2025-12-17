import os

train_path = "data/train"
yes_count = len(os.listdir(os.path.join(train_path, "yes")))
no_count = len(os.listdir(os.path.join(train_path, "no")))
print(f"Yes: {yes_count}, No: {no_count}")
