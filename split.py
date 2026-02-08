import os
import shutil
import random

random.seed(42)

SOURCE_DIR = "."
OUTPUT_DIR = "dataset"

SPLITS = {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15
}

classes = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(d)]

# create output folders
for split in SPLITS:
    for cls in classes:
        os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)

for cls in classes:
    files = os.listdir(cls)
    random.shuffle(files)

    n = len(files)
    n_train = int(n * SPLITS["train"])
    n_val = int(n * SPLITS["val"])

    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]

    for f in train_files:
        shutil.copy(os.path.join(cls, f), os.path.join(OUTPUT_DIR, "train", cls, f))
    for f in val_files:
        shutil.copy(os.path.join(cls, f), os.path.join(OUTPUT_DIR, "val", cls, f))
    for f in test_files:
        shutil.copy(os.path.join(cls, f), os.path.join(OUTPUT_DIR, "test", cls, f))

print("✅ Dataset split completed.")
