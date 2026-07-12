# ♻️ Smart Trash Bin AI

An AI-powered Smart Trash Bin that automatically classifies waste into different categories using Computer Vision and Deep Learning.

> Current Status: Prototype v1.0

---

# Project Overview

This project aims to develop an intelligent waste sorting system capable of recognizing different types of recyclable waste.

Current supported classes:

- Plastic
- Paper
- Metal

The long-term goal is to deploy the trained model on Raspberry Pi for real-time Edge AI inference.

---

# Features

- Image preprocessing pipeline
- Automatic train / validation / test split
- MobileNetV2 Transfer Learning
- Fine-tuning
- Data Augmentation experiment
- Evaluation using:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion Matrix
- Training history visualization

---

# Project Structure

```
Smart-Trash-Bin-AI/
│
├── dataset/
│   ├── raw/
│   └── processed/
│       ├── train/
│       ├── val/
│       └── test/
│
├── models/
│
├── outputs/
│
├── src/
│   ├── config.py
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── ...
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Dataset

Current dataset:

| Class | Images |
|--------|-------:|
| Plastic | ~300 |
| Metal | ~200 |
| Paper | ~200 |

Total:

Approximately **700 images**.

The dataset was manually collected using a smartphone under different lighting conditions and viewing angles.

---

# Model

Current model:

- MobileNetV2
- Image Size: 224 × 224
- Transfer Learning
- Fine-tuning
- TensorFlow / Keras

---

# Experiments

Two training strategies were compared.

### 1. Without Data Augmentation

- Original images only

### 2. With Data Augmentation

Augmentation includes:

- Random Flip
- Random Rotation
- Random Zoom
- RandomBrightness
- RandomContrast
- RandomHue

The two approaches are evaluated and compared using the same test dataset.

---

# Current Results

Current prototype performance:

- Test Accuracy: ~65%
- Evaluation metrics:
  - Precision
  - Recall
  - F1-score
  - Confusion Matrix

This result is based on a relatively small dataset and serves as the first working prototype.

---

# Results

## Training Accuracy & Loss

### Without Data Augmentation

![Training Curves](outputs/no_augmentation_loss.png)
![Training Curves](outputs/no_augmentation_accuracy.png)

### With Data Augmentation

![Training Curves](outputs/augmentation_loss.png)
![Training Curves](outputs/augmentation_accuracy.png)

---

## Confusion Matrix

### Best Model

![Confusion Matrix](outputs/confusion_matrix_no_augmentation_transfer.png)

---

## Classification Report

===== Classification Report =====
              precision    recall  f1-score   support

       Metal     0.6296    0.8095    0.7083        21
       Paper     0.7222    0.6190    0.6667        21
     Plastic     0.6429    0.5806    0.6102        31

    accuracy                         0.6575        73
   macro avg     0.6649    0.6697    0.6617        73
weighted avg     0.6619    0.6575    0.6547        73

Overall Accuracy: 0.6575

---

## Demo Predictions

![](outputs/predictions/Metal_01_result.jpg)

```
Image      : Metal_01.jpg
Prediction : Paper
Confidence : 52.69%

Class Probabilities:
Paper     : 52.69%
Metal     : 41.21%
Plastic   : 6.10%
```

![](outputs/predictions/Paper_01_result.jpg)

```
Image      : Paper_01.jpg
Prediction : Paper
Confidence : 88.42%

Class Probabilities:
Paper     : 88.42%
Plastic   : 10.84%
Metal     : 0.74%
```

![](outputs/predictions/Paper_02_result.jpg)

```
Image      : Paper_02.jpg
Prediction : Paper
Confidence : 62.10%

Class Probabilities:
Paper     : 62.10%
Plastic   : 29.24%
Metal     : 8.66%
```

![](outputs/predictions/Paper_03_result.jpg)

```
Image      : Paper_03.jpg
Prediction : Paper
Confidence : 57.34%

Class Probabilities:
Paper     : 57.34%
Plastic   : 22.56%
Metal     : 20.10%
```

![](outputs/predictions/Paper_04_result.jpg)

```
Image      : Paper_04.jpg
Prediction : Paper
Confidence : 70.16%

Class Probabilities:
Paper     : 70.16%
Plastic   : 28.56%
Metal     : 1.27%
```

![](outputs/predictions/Paper_05_result.jpg)

```
Image      : Paper_05.jpg
Prediction : Paper
Confidence : 89.58%

Class Probabilities:
Paper     : 89.58%
Plastic   : 5.42%
Metal     : 4.99%
```

![](outputs/predictions/Paper_06_result.jpg)

```
Image      : Paper_06.jpg
Prediction : Paper
Confidence : 51.00%

Class Probabilities:
Paper     : 51.00%
Plastic   : 30.06%
Metal     : 18.94%
```

![](outputs/predictions/Paper_07_result.jpg)

```
Image      : Paper_07.jpg
Prediction : Paper
Confidence : 76.31%

Class Probabilities:
Paper     : 76.31%
Plastic   : 17.74%
Metal     : 5.94%
```

![](outputs/predictions/Plastic_01_result.jpg)

```
Image      : Plastic_01.jpg
Prediction : Plastic
Confidence : 67.99%

Class Probabilities:
Plastic   : 67.99%
Metal     : 20.45%
Paper     : 11.56%
```

![](outputs/predictions/Plastic_02_result.jpg)

```
Image      : Plastic_02.jpg
Prediction : Paper
Confidence : 52.99%

Class Probabilities:
Paper     : 52.99%
Plastic   : 43.27%
Metal     : 3.74%
```

![](outputs/predictions/Plastic_03_result.jpg)

```
Image      : Plastic_03.jpg
Prediction : Paper
Confidence : 85.40%

Class Probabilities:
Paper     : 85.40%
Plastic   : 11.30%
Metal     : 3.30%
```

![](outputs/predictions/Plastic_04_result.jpg)

```
Image      : Plastic_04.jpg
Prediction : Metal
Confidence : 49.35%

Class Probabilities:
Metal     : 49.35%
Plastic   : 48.53%
Paper     : 2.12%
```

![](outputs/predictions/Plastic_05_result.jpg)

```
Image      : Plastic_05.jpg
Prediction : Metal
Confidence : 86.26%

Class Probabilities:
Metal     : 86.26%
Paper     : 7.44%
Plastic   : 6.30%
```

![](outputs/predictions/Plastic_06_result.jpg)

```
Image      : Plastic_06.jpg
Prediction : Metal
Confidence : 79.80%

Class Probabilities:
Metal     : 79.80%
Plastic   : 12.85%
Paper     : 7.35%
```

---

# How to Run

## 1. Clone repository

```bash
git clone https://github.com/pluto311207/Smart_Trash_Bin_AI-AIvsArduino-.git
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Preprocess dataset

```bash
python src/preprocess.py
```

## 4. Train model

```bash
python src/train.py
```

## 5. Evaluate model

```bash
python src/evaluate.py
```

---

## Future Improvements (Version 2)

- [ ] Increase dataset diversity with more object categories.
- [ ] Improve image quality by removing blurry samples.
- [ ] Automatically crop objects before resizing.
- [ ] Increase object size inside images.
- [ ] Collect more Metal and Plastic samples.
- [ ] Evaluate EfficientNetB0/B1.
- [ ] Deploy model on Raspberry Pi.
- [ ] Optimize inference speed using TensorFlow Lite.
- [ ] Improve Edge AI performance.

---

# Technologies Used

- Python
- TensorFlow
- Keras
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

---

## Conclusion
The prototype successfully demonstrates that MobileNetV2 with transfer learning can classify waste into three categories (Metal, Plastic, and Paper) using a relatively small custom dataset.

The model achieved an overall test accuracy of approximately 65%, showing that it learned meaningful visual features and was able to correctly classify many unseen images. Paper objects were generally recognized with higher confidence, while Metal and Plastic were more frequently confused due to similar colors, shapes, and reflective surfaces.

Prediction results also indicate that the model performs well on common objects included in the training dataset, but struggles with uncommon plastic items such as cosmetic containers or rubber accessories. This suggests that the current limitation is primarily related to dataset diversity rather than the model architecture itself.

Overall, the project successfully validates the feasibility of using deep learning for a Smart Trash Bin prototype. Future improvements will focus on expanding the dataset, improving image quality, automatically cropping objects before training, and deploying an optimized model on Raspberry Pi for real-time edge AI inference.

---

# License

This project is developed for educational and research purposes.