from pathlib import Path
import shutil
import random

import cv2
import pandas as pd
from tqdm import tqdm

from config import (
    RAW_DIR,
    PROCESSED_DIR,
    OUTPUT_DIR,
    CLASS_NAMES,
    IMG_SIZE,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    RANDOM_SEED,
    SUPPORTED_EXTENSIONS,
)

random.seed(RANDOM_SEED)

# ==========================================================
# Reset processed dataset
# ==========================================================

if PROCESSED_DIR.exists():
    print("Removing old processed dataset...")
    shutil.rmtree(PROCESSED_DIR)

for split in ["train", "val", "test"]:
    for cls in CLASS_NAMES:
        (PROCESSED_DIR / split / cls).mkdir(
            parents=True,
            exist_ok=True,
        )

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# Dataset report
# ==========================================================

report = []

# ==========================================================
# Process each class
# ==========================================================

for cls in CLASS_NAMES:

    print(f"\n{'=' * 60}")
    print(f"Processing class: {cls}")
    print(f"{'=' * 60}")

    class_dir = RAW_DIR / cls

    if not class_dir.exists():
        print(f"[WARNING] Missing folder: {class_dir}")
        continue

    images = [
        img
        for img in class_dir.iterdir()
        if img.is_file()
        and img.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    random.shuffle(images)

    total = len(images)

    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    splits = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:],
    }

    image_counter = 1
    failed = 0

    for split_name, image_list in splits.items():

        print(f"\n{split_name.upper()} ({len(image_list)} images)")

        for image_path in tqdm(image_list):

            img = cv2.imread(str(image_path))

            if img is None:
                print(f"Cannot read: {image_path.name}")
                failed += 1
                continue

            img = cv2.resize(img, IMG_SIZE)

            save_path = (
                PROCESSED_DIR
                / split_name
                / cls
                / f"{cls}_{image_counter:04d}.jpg"
            )

            cv2.imwrite(str(save_path), img)

            image_counter += 1

    report.append(
        {
            "Class": cls,
            "Total Images": total,
            "Train": len(splits["train"]),
            "Validation": len(splits["val"]),
            "Test": len(splits["test"]),
            "Failed": failed,
        }
    )

# ==========================================================
# Save report
# ==========================================================

df = pd.DataFrame(report)

csv_path = OUTPUT_DIR / "dataset_report.csv"

df.to_csv(csv_path, index=False)

print("\n")
print("=" * 60)
print("DATASET REPORT")
print("=" * 60)
print(df)

print("\nDataset preprocessing completed!")

print(f"\nProcessed dataset:")
print(PROCESSED_DIR)

print(f"\nDataset report:")
print(csv_path)