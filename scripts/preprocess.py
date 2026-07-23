from pathlib import Path
import shutil
import random

import cv2
import numpy as np
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
# Auto Crop Object
# ==========================================================

def _segment_foreground(gray_blur, invert):
    """
    Threshold ảnh grayscale đã blur, trả về mask nhị phân (255 = vật thể).
    invert=True  : giả định vật thể TỐI hơn nền (dùng THRESH_BINARY_INV)
    invert=False : giả định vật thể SÁNG hơn nền (dùng THRESH_BINARY)
    """
    flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, thresh = cv2.threshold(gray_blur, 0, 255, flag + cv2.THRESH_OTSU)
    return thresh


def _border_leak_ratio(mask):
    """
    Tỉ lệ pixel "vật thể" (mask=255) nằm dọc theo viền ngoài cùng của
    ảnh. Nếu vật thể thật sự nằm giữa khung hình (như cách thường chụp),
    tỉ lệ này phải THẤP. Nếu threshold chọn sai chiều sáng/tối, nó sẽ
    khoanh nhầm cả vùng nền lớn xung quanh - vùng đó luôn chạm viền ảnh
    - nên tỉ lệ này sẽ CAO. Dùng chỉ số này để chọn đúng chiều threshold.
    """
    h, w = mask.shape
    border = np.concatenate([
        mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1],
    ])
    return float((border == 255).mean())


def auto_crop(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Thử cả 2 chiều threshold (vật thể tối / vật thể sáng), chọn chiều
    # nào có ít pixel "vật thể" dính vào viền ảnh hơn - đó là dấu hiệu
    # đáng tin cậy nhất cho biết thuật toán đang khoanh đúng vật thể ở
    # giữa khung hình, chứ không phải khoanh nhầm cả vùng nền.
    thresh_dark_object = _segment_foreground(blur, invert=True)
    thresh_bright_object = _segment_foreground(blur, invert=False)

    leak_dark = _border_leak_ratio(thresh_dark_object)
    leak_bright = _border_leak_ratio(thresh_bright_object)

    thresh = thresh_dark_object if leak_dark <= leak_bright else thresh_bright_object

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if len(contours) == 0:
        return img

    img_h, img_w = img.shape[:2]

    MIN_AREA = img_h * img_w * 0.005
    # Loại luôn những contour chiếm gần hết khung hình (nhiều khả năng là
    # nền/background bị khoanh nhầm thành vật thể, kể cả sau khi đã chọn
    # chiều threshold tốt hơn ở trên).
    MAX_AREA = img_h * img_w * 0.90

    valid_contours = [
        c for c in contours
        if MIN_AREA < cv2.contourArea(c) < MAX_AREA
    ]

    if len(valid_contours) == 0:
        return img

    x_min = img_w
    y_min = img_h
    x_max = 0
    y_max = 0

    for cnt in valid_contours:

        x, y, w, h = cv2.boundingRect(cnt)

        x_min = min(x_min, x)
        y_min = min(y_min, y)

        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)

    crop_w = x_max - x_min
    crop_h = y_max - y_min

    pad_x = int(crop_w * 0.15)
    pad_y = int(crop_h * 0.15)

    x_min = max(0, x_min - pad_x)
    y_min = max(0, y_min - pad_y)

    x_max = min(img_w, x_max + pad_x)
    y_max = min(img_h, y_max + pad_y)

    crop = img[y_min:y_max, x_min:x_max]

    crop_h, crop_w = crop.shape[:2]

    # Không crop nếu vùng crop quá nhỏ
    if min(crop_h, crop_w) < 120:
        return img

    return crop


def pad_to_square(img):
    """
    Thêm viền (padding) để ảnh thành hình vuông TRƯỚC khi resize, thay vì
    resize thẳng crop_w x crop_h (thường không vuông) về 224x224 - resize
    trực tiếp sẽ kéo méo hình dạng vật thể (vd vật thể tròn bị kéo thành
    oval). Màu viền lấy trung bình màu 10px viền ngoài ảnh gốc, gần với
    nền thật hơn là màu đen/trắng cứng.
    """
    h, w = img.shape[:2]
    size = max(h, w)

    border_pixels = np.concatenate([
        img[:10, :].reshape(-1, 3),
        img[-10:, :].reshape(-1, 3),
        img[:, :10].reshape(-1, 3),
        img[:, -10:].reshape(-1, 3),
    ])
    fill_color = tuple(int(v) for v in border_pixels.mean(axis=0))

    top = (size - h) // 2
    bottom = size - h - top
    left = (size - w) // 2
    right = size - w - left

    return cv2.copyMakeBorder(
        img, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=fill_color,
    )


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

            img = auto_crop(img)

            img = pad_to_square(img)

            img = cv2.resize(
                img,
                IMG_SIZE,
                interpolation=cv2.INTER_CUBIC,
            )

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