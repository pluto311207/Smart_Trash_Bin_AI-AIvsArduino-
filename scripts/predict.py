import argparse
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from config import (
    IMG_SIZE,
    CLASS_NAMES,
    EXPERIMENTS,
    DEMO_DIR,
    PREDICTION_DIR,
    SUPPORTED_EXTENSIONS,
    transfer_model_path,
    finetune_model_path,
)

from custom_layers import RandomHue


def load_image(image_path):
    img = tf.keras.utils.load_img(
        image_path,
        target_size=IMG_SIZE
    )

    img_array = tf.keras.utils.img_to_array(img)

    return img_array


def predict_image(model, img_array):

    batch = np.expand_dims(img_array, axis=0)

    probs = model.predict(batch, verbose=0)[0]

    predicted_index = int(np.argmax(probs))

    predicted_label = CLASS_NAMES[predicted_index]

    confidence = float(np.max(probs))

    prob_dict = {
        CLASS_NAMES[i]: float(probs[i])
        for i in range(len(CLASS_NAMES))
    }

    return predicted_label, confidence, prob_dict


def save_prediction_image(
    image_path,
    predicted_label,
    confidence,
):

    image = cv2.imread(str(image_path))

    text1 = f"{predicted_label}"
    text2 = f"{confidence * 100:.2f}%"

    cv2.putText(
        image,
        text1,
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    cv2.putText(
        image,
        text2,
        (10, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )

    save_path = (
        PREDICTION_DIR
        / f"{Path(image_path).stem}_result.jpg"
    )

    cv2.imwrite(str(save_path), image)

    return save_path


def predict_single_file(model, image_path):

    img_array = load_image(image_path)

    label, confidence, probs = predict_image(
        model,
        img_array,
    )

    print("\n" + "=" * 60)
    print(f"Image      : {image_path}")
    print(f"Prediction : {label}")
    print(f"Confidence : {confidence * 100:.2f}%")

    print("\nClass Probabilities:")

    for class_name, p in sorted(
        probs.items(),
        key=lambda x: -x[1]
    ):
        print(f"{class_name:10s}: {p * 100:.2f}%")

    saved_path = save_prediction_image(
        image_path,
        label,
        confidence,
    )

    print(f"\nSaved result: {saved_path}")


def predict_demo_folder(model):

    if not DEMO_DIR.exists():
        print(f"Demo folder not found: {DEMO_DIR}")
        return

    images = [
        img
        for img in DEMO_DIR.iterdir()
        if img.is_file()
        and img.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if len(images) == 0:
        print("No images found in demo folder.")
        return

    print(f"\nFound {len(images)} image(s) in demo folder.")

    for image_path in images:
        predict_single_file(
            model,
            image_path,
        )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        default=None,
        help="Image path for prediction",
    )

    parser.add_argument(
        "--experiment",
        choices=EXPERIMENTS,
        default="no_augmentation",
    )

    parser.add_argument(
        "--phase",
        choices=["transfer", "finetune"],
        default="finetune",
    )

    args = parser.parse_args()

    model_path = (
        transfer_model_path(args.experiment)
        if args.phase == "transfer"
        else finetune_model_path(args.experiment)
    )

    print(f"Loading model: {model_path}")

    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            "RandomHue": RandomHue
        },
    )

    if args.image is not None:

        predict_single_file(
            model,
            args.image,
        )

    else:

        predict_demo_folder(model)


if __name__ == "__main__":
    main()