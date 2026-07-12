
import csv
import random

import matplotlib
matplotlib.use("Agg")  # để chạy được cả khi không có màn hình (server/RPi)
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

# Cố định seed để các lần chạy có thể so sánh công bằng với nhau - nếu
# không, sự khác biệt giữa các lần train một phần đến từ random init/
# shuffle chứ không phải do thay đổi hyperparameter.
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

from config import (
    TRAIN_DIR,
    VAL_DIR,
    IMG_SIZE,
    IMG_SHAPE,
    BATCH_SIZE,
    NUM_CLASSES,
    CLASS_NAMES,
    EXPERIMENTS,
    INITIAL_EPOCHS,
    FINETUNE_EPOCHS,
    FINETUNE_AT_LAYER,
    LEARNING_RATE,
    FINETUNE_LEARNING_RATE,
    transfer_model_path,
    finetune_model_path,
    history_csv_path,
    OUTPUT_DIR,
)
from custom_layers import RandomHue


# ----------------------------------------------------------------------
# Load dataset (dùng chung cho cả 2 experiment)
# ----------------------------------------------------------------------
def load_datasets():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=True,
        seed=42,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False,
    )

    print("Class order phát hiện từ thư mục:", train_ds.class_names)
    print("Class order khai báo trong config.py:", CLASS_NAMES)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds


# ----------------------------------------------------------------------
# Data Augmentation (chỉ dùng khi use_augmentation=True)
# ----------------------------------------------------------------------
def build_augmentation():
    return models.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.15),
            layers.RandomZoom(0.15),
            # Thêm 2 layer này để model học phân biệt tốt hơn giữa bề mặt
            # phản chiếu (metal) và bề mặt mờ (paper) - đây là nguồn nhầm
            # lẫn Paper<->Metal phổ biến. 0.15 là mức trung bình: đủ mạnh để
            # có tác dụng chống overfitting, không quá mạnh làm khó học.
            layers.RandomContrast(0.15),
            layers.RandomBrightness(0.15),
            # RandomHue: xoay nhẹ tông màu (hue) - giúp model không dựa dẫm
            # vào màu sắc cụ thể (giấy vàng/nâu vs kim loại xám) mà học các
            # đặc trưng khác (độ phản chiếu, texture). Factor nhỏ (0.03) vì
            # hue rất nhạy cảm, chỉnh mạnh dễ làm ảnh trông giả.
            RandomHue(factor=0.03),
        ],
        name="data_augmentation",
    )


# ----------------------------------------------------------------------
# Load MobileNetV2 + thêm classifier
# ----------------------------------------------------------------------
def build_model(base_trainable=False, use_augmentation=False):
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=IMG_SHAPE,
    )
    base_model.trainable = base_trainable

    # MobileNetV2 cần input được scale về [-1, 1]
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

    inputs = tf.keras.Input(shape=IMG_SHAPE)
    x = inputs

    if use_augmentation:
        x = build_augmentation()(x)

    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    return model, base_model


# ----------------------------------------------------------------------
# Compile
# ----------------------------------------------------------------------
def compile_model(model, learning_rate):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )


# ----------------------------------------------------------------------
# Callbacks: EarlyStopping + ReduceLROnPlateau
# ----------------------------------------------------------------------
def build_callbacks():
    return [
        # Dừng training khi val_accuracy không cải thiện sau `patience` epoch,
        # tự động khôi phục lại weight tốt nhất (tránh lưu model đã overfit
        # ở epoch cuối cùng).
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        # Giảm learning rate khi val_loss chững lại, giúp model học tinh hơn
        # ở giai đoạn cuối thay vì bị "nhảy" qua điểm tối ưu.
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
    ]


# ----------------------------------------------------------------------
# Vẽ graph accuracy / loss cho 1 experiment
# ----------------------------------------------------------------------
def plot_history(full_history, experiment_name):
    acc = full_history["accuracy"]
    val_acc = full_history["val_accuracy"]
    loss = full_history["loss"]
    val_loss = full_history["val_loss"]
    epochs_range = range(len(acc))

    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, acc, label="Train Accuracy")
    plt.plot(epochs_range, val_acc, label="Val Accuracy")
    plt.legend(loc="lower right")
    plt.title(f"Accuracy - {experiment_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig(OUTPUT_DIR / f"{experiment_name}_accuracy.png")
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, loss, label="Train Loss")
    plt.plot(epochs_range, val_loss, label="Val Loss")
    plt.legend(loc="upper right")
    plt.title(f"Loss - {experiment_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(OUTPUT_DIR / f"{experiment_name}_loss.png")
    plt.close()

    print(f"Đã lưu graph: {experiment_name}_accuracy.png / {experiment_name}_loss.png")


# ----------------------------------------------------------------------
# Lưu accuracy/loss từng epoch ra CSV
# ----------------------------------------------------------------------
def save_history_csv(full_history, experiment_name):
    path = history_csv_path(experiment_name)
    epochs = range(len(full_history["accuracy"]))

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "accuracy", "val_accuracy", "loss", "val_loss"])
        for e in epochs:
            writer.writerow(
                [
                    e + 1,
                    full_history["accuracy"][e],
                    full_history["val_accuracy"][e],
                    full_history["loss"][e],
                    full_history["val_loss"][e],
                ]
            )

    print(f"Đã lưu accuracy/loss từng epoch tại: {path}")


# ----------------------------------------------------------------------
# Tính class_weight từ số lượng ảnh thực tế mỗi lớp trong train/
# ----------------------------------------------------------------------
def compute_class_weights():
    counts = {}
    for idx, class_name in enumerate(CLASS_NAMES):
        class_dir = TRAIN_DIR / class_name
        n_images = len(list(class_dir.glob("*")))
        counts[class_name] = n_images

    total = sum(counts.values())
    n_classes = len(CLASS_NAMES)

    class_weights = {}
    for idx, class_name in enumerate(CLASS_NAMES):
        # Công thức chuẩn (linear) là: total / (n_classes * count).
        # Nhưng công thức linear dễ đẩy quá tay (lớp ít ảnh bị boost quá
        # mạnh, "nuốt" luôn recall của lớp khác - đã thấy ở lần chạy trước:
        # Paper được cải thiện nhưng Plastic bị nhầm thành Paper nhiều hơn).
        # Lấy căn bậc 2 để làm dịu mức độ chênh lệch weight giữa các lớp.
        raw_weight = total / (n_classes * counts[class_name])
        class_weights[idx] = raw_weight ** 0.5

    print("\nSố lượng ảnh mỗi lớp (train):", counts)
    print("Class weight tương ứng (đã làm dịu bằng sqrt):", class_weights)

    return class_weights


# ----------------------------------------------------------------------
# Chạy 1 experiment đầy đủ: Transfer Learning + Fine-tuning
# ----------------------------------------------------------------------
def run_experiment(experiment_name, use_augmentation, train_ds, val_ds, class_weights):
    print(f"\n{'=' * 60}")
    print(f"EXPERIMENT: {experiment_name} (augmentation={use_augmentation})")
    print(f"{'=' * 60}")

    # ---------- PHASE 1: Transfer Learning (freeze base model) ----------
    print(f"\n----- [{experiment_name}] PHASE 1: Transfer Learning -----")
    model, base_model = build_model(base_trainable=False, use_augmentation=use_augmentation)
    compile_model(model, learning_rate=LEARNING_RATE)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=INITIAL_EPOCHS,
        class_weight=class_weights,
        callbacks=build_callbacks(),
    )

    model.save(transfer_model_path(experiment_name))
    print(f"Đã lưu model: {transfer_model_path(experiment_name)}")

    # Gộp history vào 1 dict để cộng dồn cả phase 2 sau này
    full_history = {
        "accuracy": list(history.history["accuracy"]),
        "val_accuracy": list(history.history["val_accuracy"]),
        "loss": list(history.history["loss"]),
        "val_loss": list(history.history["val_loss"]),
    }

    # ---------- PHASE 2: Fine-tuning ----------
    print(f"\n----- [{experiment_name}] PHASE 2: Fine-tuning -----")
    base_model.trainable = True
    for layer in base_model.layers[:FINETUNE_AT_LAYER]:
        layer.trainable = False

    compile_model(model, learning_rate=FINETUNE_LEARNING_RATE)

    # Dùng epoch thực tế mà phase 1 đã dừng lại (có thể sớm hơn
    # INITIAL_EPOCHS nếu EarlyStopping đã kích hoạt), rồi cộng thêm ngân
    # sách epoch của phase 2.
    start_epoch = history.epoch[-1] + 1
    total_epochs = start_epoch + FINETUNE_EPOCHS
    history_fine = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=total_epochs,
        initial_epoch=start_epoch,
        class_weight=class_weights,
        callbacks=build_callbacks(),
    )

    model.save(finetune_model_path(experiment_name))
    print(f"Đã lưu model: {finetune_model_path(experiment_name)}")

    full_history["accuracy"].extend(history_fine.history["accuracy"])
    full_history["val_accuracy"].extend(history_fine.history["val_accuracy"])
    full_history["loss"].extend(history_fine.history["loss"])
    full_history["val_loss"].extend(history_fine.history["val_loss"])

    plot_history(full_history, experiment_name)
    save_history_csv(full_history, experiment_name)

    # QUAN TRỌNG: model.fit() với EarlyStopping(restore_best_weights=True)
    # đã khôi phục model về epoch có val_accuracy tốt nhất - nhưng
    # history.history["val_accuracy"][-1] chỉ là epoch cuối trước khi dừng
    # (sau khi đã "chờ" thêm patience epoch), KHÔNG phải giá trị thực tế
    # của model đã khôi phục. Phải evaluate lại trực tiếp để lấy đúng số.
    train_loss, train_accuracy = model.evaluate(train_ds, verbose=0)
    val_loss, val_accuracy = model.evaluate(val_ds, verbose=0)

    print(f"[{experiment_name}] Final train accuracy (best-weights): {train_accuracy:.4f}")
    print(f"[{experiment_name}] Final val accuracy   (best-weights): {val_accuracy:.4f}")

    full_history["final_train_accuracy"] = train_accuracy
    full_history["final_val_accuracy"] = val_accuracy

    return full_history


# ----------------------------------------------------------------------
# So sánh accuracy giữa các experiment
# ----------------------------------------------------------------------
def plot_comparison(all_histories):
    plt.figure(figsize=(9, 6))
    for experiment_name, full_history in all_histories.items():
        epochs_range = range(len(full_history["val_accuracy"]))
        plt.plot(epochs_range, full_history["val_accuracy"], label=f"{experiment_name} (val)")

    plt.legend(loc="lower right")
    plt.title("So sánh Val Accuracy: có vs không Augmentation")
    plt.xlabel("Epoch")
    plt.ylabel("Val Accuracy")
    comparison_path = OUTPUT_DIR / "comparison_accuracy.png"
    plt.savefig(comparison_path)
    plt.close()
    print(f"\nĐã lưu graph so sánh tại: {comparison_path}")


def save_comparison_summary(all_histories):
    summary_path = OUTPUT_DIR / "comparison_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["experiment", "final_train_accuracy", "final_val_accuracy"])
        for experiment_name, full_history in all_histories.items():
            writer.writerow(
                [
                    experiment_name,
                    full_history["final_train_accuracy"],
                    full_history["final_val_accuracy"],
                ]
            )

    print(f"Đã lưu bảng tổng kết so sánh tại: {summary_path}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    train_ds, val_ds = load_datasets()
    class_weights = compute_class_weights()

    all_histories = {}

    for experiment_name in EXPERIMENTS:
        use_augmentation = experiment_name == "augmentation"
        full_history = run_experiment(
            experiment_name, use_augmentation, train_ds, val_ds, class_weights
        )
        all_histories[experiment_name] = full_history

    plot_comparison(all_histories)
    save_comparison_summary(all_histories)

    print("\nHoàn tất tất cả experiment. Kết quả so sánh nằm trong thư mục outputs/:")
    print("  - comparison_accuracy.png  : graph overlay accuracy 2 experiment")
    print("  - comparison_summary.csv   : accuracy cuối cùng của từng experiment")
    print("  - history_<experiment>.csv : accuracy/loss từng epoch chi tiết")


if __name__ == "__main__":
    main()