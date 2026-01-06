# Chest X-Ray Pathology Classification using Deep Learning

## 📌 Overview
This project presents a deep learning-based classifier for chest X-ray images,
capable of distinguishing between **Normal**, **Pneumonia**, and **Tuberculosis**
cases. The model is designed as a **computer-aided diagnosis (CAD)** support tool,
prioritizing high sensitivity for pathological cases.

## 🧠 Model
- Architecture: **DenseNet169**
- Technique: **Transfer Learning**
- Framework: **PyTorch**
- Input: Grayscale chest X-ray images (512x512)

## 📊 Dataset
The [dataset](https://www.kaggle.com/datasets/muhammadrehan00/chest-xray-dataset/data) was obtained from a public Kaggle repository containing chest X-ray
images organized into training, validation, and test splits.

| Class         | Train | Validation | Test |
|---------------|-------|------------|------|
| Normal        | 7,263 | 900        | 925  |
| Pneumonia    | 4,674 | 570        | 580  |
| Tuberculosis | 8,513 | 1,064      | 1,064|

⚠️ Dataset is **not included** in this repository due to size and licensing restrictions.

## 🔄 Preprocessing & Data Augmentation
- Resize to 512×512
- Grayscale conversion
- Histogram equalization
- Small rotations (±5°)
- Brightness and contrast variations

## ⚙️ Training Strategy
- Loss Function: CrossEntropyLoss
- Optimizer: Adam
- Early Stopping (patience = 5)
- Full fine-tuning (no frozen layers)

## 📈 Results
- **Overall Accuracy:** 77%
- High recall for pathological classes:
  - Pneumonia: 95%
  - Tuberculosis: 100%

This behavior makes the model suitable for **screening scenarios**, where false negatives are more critical than false positives.

## 🧪 Experiments
Several experiments were conducted, including:
- ResNet18 baseline
- Training without data augmentation
- Layer freezing/unfreezing strategies
- Hyperparameter fine-tuning

Details are available in the final report.

## 📁 Repository Structure

```text
chest-xray-pathology-classifier/
│
├── notebooks/          # Exploratory analysis and experiments
│   └── chest_xray_classifier.ipynb
│
├── src/                # Notebook code organized by responsibility
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
│
├── reports/            # Project documentation
│   └── final_report.pdf
│
├── results/            # Generated results and figures
│   ├── confusion_matrix.png
│   └── training_curves.png
│
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## 📄 Report
A complete academic-style report is available in the `reports/` folder.

## 🚀 Future Work
- Class imbalance mitigation (weighted loss, oversampling)
- Model explainability (Grad-CAM)
- Performance improvement for the Normal class

## 👤 Authors
- Gustavo Nascimento Pavanelli
- Gabriel Campello Dalbuquerque Lima
- Marcel Capistrano Almeida Rodrigues
