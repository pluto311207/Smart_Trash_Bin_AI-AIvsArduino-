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
import shutil
from pathlib import Path

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

# Thư mục gốc chứa ảnh bị đoán sai, tổ chức theo cấu trúc
# outputs/misclassified/<TrueClass>_to_<PredictedClass>/
MISCLASSIFIED_DIR = OUTPUT_DIR / "misclassified"

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
    y_probs = []

    for images, labels in test_ds:
        probs = model.predict(images, verbose=0)
        preds = np.argmax(probs, axis=1)

        y_true.extend(labels.numpy().tolist())
        y_pred.extend(preds.tolist())
        y_probs.extend(probs.tolist())

    return np.array(y_true), np.array(y_pred), np.array(y_probs)


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


def run_error_analysis(y_true, y_pred, y_probs, file_paths, tag):
    """
    Tìm toàn bộ ảnh test bị đoán sai, copy vào đúng thư mục
    outputs/misclassified/<TrueClass>_to_<PredictedClass>/, đồng thời
    xuất 1 file CSV liệt kê chi tiết từng ảnh sai (nhãn thật, nhãn đoán,
    độ tự tin, xác suất từng lớp).
    """

    # Reset thư mục misclassified mỗi lần chạy, tránh lẫn ảnh của lần
    # evaluate trước (model/experiment khác) với lần này.
    if MISCLASSIFIED_DIR.exists():
        shutil.rmtree(MISCLASSIFIED_DIR)

    for true_c in CLASS_NAMES:
        for pred_c in CLASS_NAMES:
            if true_c != pred_c:
                (MISCLASSIFIED_DIR / f"{true_c}_to_{pred_c}").mkdir(
                    parents=True, exist_ok=True
                )

    records = []

    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            continue

        true_label = CLASS_NAMES[y_true[i]]
        pred_label = CLASS_NAMES[y_pred[i]]
        confidence = float(np.max(y_probs[i]))

        src_path = Path(file_paths[i])
        dest_dir = MISCLASSIFIED_DIR / f"{true_label}_to_{pred_label}"
        dest_path = dest_dir / src_path.name

        shutil.copy2(src_path, dest_path)

        record = {
            "filename": src_path.name,
            "original_path": str(src_path),
            "true_label": true_label,
            "predicted_label": pred_label,
            "confidence": confidence,
        }
        for idx, class_name in enumerate(CLASS_NAMES):
            record[f"prob_{class_name}"] = float(y_probs[i][idx])

        records.append(record)

    df = pd.DataFrame(records)

    csv_path = OUTPUT_DIR / f"misclassified_{tag}.csv"
    df.to_csv(csv_path, index=False)

    print(f"\n===== Error Analysis =====")
    print(f"Total misclassified: {len(records)} / {len(y_true)}")

    if len(records) > 0:
        print("\nBreakdown by error type:")
        breakdown = df.groupby(["true_label", "predicted_label"]).size()
        for (true_l, pred_l), count in breakdown.items():
            print(f"  {true_l:10s} -> {pred_l:10s}: {count}")

    print(f"\nSaved misclassified images to: {MISCLASSIFIED_DIR}")
    print(f"Saved: {csv_path}")

    return df


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

    # .file_paths chỉ đáng tin cậy khi shuffle=False (đã set trong
    # load_test_dataset) - đây là danh sách đường dẫn ảnh đúng theo thứ
    # tự mà test_ds sẽ duyệt qua, dùng để đối chiếu ngược lại ảnh gốc
    # cho từng dự đoán.
    file_paths = test_ds.file_paths

    print("\nClass order:")
    print(test_ds.class_names)

    y_true, y_pred, y_probs = get_predictions(model, test_ds)

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

    # ==========================
    # Error Analysis
    # ==========================

    run_error_analysis(
        y_true,
        y_pred,
        y_probs,
        file_paths,
        tag,
    )


if __name__ == "__main__":
    main()