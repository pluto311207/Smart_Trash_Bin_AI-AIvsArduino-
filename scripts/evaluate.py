"""
evaluate.py

Đánh giá model đã train trên tập test:
    - Accuracy
    - Precision / Recall / F1-score (per class + macro avg)
    - Confusion Matrix (số lượng + phần trăm), xuất cả ảnh và CSV

Chạy:
    python scripts/evaluate.py --experiment augmentation --phase finetune
    python scripts/evaluate.py --experiment no_augmentation --phase transfer
"""

import argparse
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
)

from config import (
    TEST_DIR,
    IMG_SIZE,
    BATCH_SIZE,
    CLASS_NAMES,
    EXPERIMENTS,
    transfer_model_path,
    finetune_model_path,
    OUTPUT_DIR,
)

# QUAN TRỌNG: phải import custom_layers TRƯỚC khi gọi load_model(), để
# Keras biết cách deserialize các custom layer (RandomHue) đã lưu trong
# model .keras. Nếu thiếu dòng import này sẽ bị lỗi:
#   TypeError: <class 'keras.src.models.sequential.Sequential'> could not
#   be deserialized properly...
from custom_layers import RandomHue


def load_test_dataset():
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False,
    )
    return test_ds


def get_predictions(model, test_ds):
    y_true = []
    y_pred = []

    for images, labels in test_ds:
        probs = model.predict(images, verbose=0)
        preds = np.argmax(probs, axis=1)

        y_true.extend(labels.numpy().tolist())
        y_pred.extend(preds.tolist())

    return np.array(y_true), np.array(y_pred)


def save_confusion_matrix(cm, filename, percent=False):
    fig, ax = plt.subplots(figsize=(6, 6))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASS_NAMES,
    )

    disp.plot(
        ax=ax,
        cmap="Blues",
        colorbar=False,
        values_format=".2f" if percent else "d",
    )

    title = "Confusion Matrix (%)" if percent else "Confusion Matrix"
    plt.title(title)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {OUTPUT_DIR / filename}")


def save_confusion_csv(cm, filename):
    df = pd.DataFrame(
        cm,
        index=CLASS_NAMES,
        columns=CLASS_NAMES,
    )

    df.to_csv(OUTPUT_DIR / filename)

    print(f"Saved: {OUTPUT_DIR / filename}")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment",
        choices=EXPERIMENTS,
        default="augmentation",
        help="Choose which experiment to evaluate",
    )

    parser.add_argument(
        "--phase",
        choices=["transfer", "finetune"],
        default="finetune",
        help="Choose which phase to evaluate",
    )

    args = parser.parse_args()

    model_path = (
        transfer_model_path(args.experiment)
        if args.phase == "transfer"
        else finetune_model_path(args.experiment)
    )

    print(f"\nLoading model:")
    print(model_path)

    # custom_objects đưa vào tường minh để chắc chắn Keras deserialize
    # đúng RandomHue, kể cả trong trường hợp registry tự động không nhận
    # ra (ví dụ chạy từ working directory khác).
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={"RandomHue": RandomHue},
    )

    test_ds = load_test_dataset()

    print("\nClass order:")
    print(test_ds.class_names)

    y_true, y_pred = get_predictions(model, test_ds)

    # ==========================
    # Test dataset statistics
    # ==========================

    print("\n===== Test Dataset =====")

    unique, counts = np.unique(y_true, return_counts=True)

    for u, c in zip(unique, counts):
        print(f"{CLASS_NAMES[u]:10s}: {c}")

    # ==========================
    # Classification Report
    # ==========================

    print("\n===== Classification Report =====")

    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        digits=4,
    )

    print(report)

    accuracy = accuracy_score(y_true, y_pred)

    print(f"Overall Accuracy: {accuracy:.4f}")

    tag = f"{args.experiment}_{args.phase}"

    # ==========================
    # Confusion Matrix
    # ==========================

    cm = confusion_matrix(y_true, y_pred)

    save_confusion_matrix(
        cm,
        filename=f"confusion_matrix_{tag}.png",
        percent=False,
    )

    save_confusion_csv(
        cm,
        filename=f"confusion_matrix_{tag}.csv",
    )

    # ==========================
    # Normalized Confusion Matrix
    # ==========================

    cm_percent = confusion_matrix(
        y_true,
        y_pred,
        normalize="true",
    )

    save_confusion_matrix(
        cm_percent,
        filename=f"confusion_matrix_percent_{tag}.png",
        percent=True,
    )

    # ==========================
    # Save report
    # ==========================

    report_path = OUTPUT_DIR / f"classification_report_{tag}.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Model: {model_path}\n")
        f.write(f"Overall Accuracy: {accuracy:.4f}\n\n")
        f.write(report)

    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()