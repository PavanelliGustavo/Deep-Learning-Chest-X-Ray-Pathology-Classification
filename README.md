# Chest X-Ray Pathology Classification using Deep Learning

## 📌 Overview
This project presents a deep learning-based classifier for chest X-ray images,
capable of distinguishing between **Normal**, **Pneumonia**, and **Tuberculosis**
cases. The model is designed as a **computer-aided diagnosis (CAD)** support tool,
prioritizing high sensitivity for pathological cases, making it suitable for
screening and triage scenarios.

Beyond model performance, this repository emphasizes **clean project organization,
code modularity, and reusability**, following best practices commonly adopted in
applied machine learning projects.

---

## 🧠 Model
- Architecture: **DenseNet169**
- Technique: **Transfer Learning**
- Framework: **PyTorch**
- Input: Grayscale chest X-ray images (512×512)

The first convolutional layer was adapted to handle single-channel (grayscale)
images, and the final classification layer was replaced to support the three
target classes.

---

## 📊 Dataset
The dataset was obtained from a public Kaggle repository:
[Chest X-Ray Dataset](https://www.kaggle.com/datasets/muhammadrehan00/chest-xray-dataset/data)

It contains chest X-ray images organized into **training, validation, and test**
splits with a natural class imbalance, common in medical imaging datasets.

| Class         | Train | Validation | Test |
|---------------|-------|------------|------|
| Normal        | 7,263 | 900        | 925  |
| Pneumonia    | 4,674 | 570        | 580  |
| Tuberculosis | 8,513 | 1,064      | 1,064|

> ⚠️ The dataset is **not included** in this repository due to size and licensing
restrictions.

---

## 🔄 Preprocessing & Data Augmentation
- Resize to 512×512
- Grayscale conversion
- Histogram equalization
- Small rotations (±5°)
- Brightness and contrast variations

Data augmentation was applied only during training to reduce overfitting and
encourage the model to focus on lung structure rather than pixel memorization.

---

## ⚙️ Training Strategy
- Loss Function: **CrossEntropyLoss**
- Optimizer: **Adam**
- Learning Rate Scheduling: **ReduceLROnPlateau**
- Early Stopping (patience = 5)
- Full fine-tuning (no frozen layers)

Mixed Precision Training (AMP) was used when supported by the available hardware.

---

## 📈 Results
- **Overall Accuracy:** 77%
- High recall for pathological classes:
  - Pneumonia: 95%
  - Tuberculosis: 100%

This behavior reflects a **high-sensitivity model**, prioritizing the detection
of pathological cases over false positives — a desirable property in medical
screening contexts.

---

## 🧪 Experimental Analysis
Multiple experimental configurations were evaluated during development, including:
- ResNet18 as a baseline architecture
- Training without data augmentation
- Layer freezing and unfreezing strategies
- Hyperparameter fine-tuning

A detailed discussion of these experiments and their outcomes is available in the
final project report.

---

## 📁 Repository Structure

The repository is organized to clearly separate **experimentation**, **reusable
source code**, and **results**, following common practices in professional machine
learning projects.

```text
chest-xray-pathology-classifier/
│
├── notebooks/          # Orchestrates experiments and result visualization
│   ├── original_chest_xray_classifier.ipynb
│   └── modeluar_chest_xray_classifier.ipynb
│
├── src/                # Reusable, modular source code
│   ├── dataset.py      # Dataset and DataLoader definitions
│   ├── model.py        # Model architecture and adaptations
│   ├── train.py        # Training loop and optimization logic
│   ├── evaluate.py     # Model evaluation and metric computation
│   └── utils.py        # Utility functions (device, early stopping, plotting)
│
├── reports/            # Project documentation
│   └── final_report.pdf
│
├── results/            # Generated figures and visual results
│   ├── confusion_matrix.png
│   └── training_curves.png
│
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```
This structure ensures that:
- Core logic is reusable outside the notebook
- Experiments remain readable and reproducible
- The project can scale beyond a single notebook

---

## 📓 Notebooks

This repository contains two notebooks with distinct purposes:

- `original_chest_xray_classifier.ipynb`  
  Original experimental notebook used during development. It contains full
  training execution and rendered outputs, demonstrating that the model and
  methodology work as expected.

- `modular_chest_xray_classifier.ipynb`  
  Refactored version showcasing a clean and reusable code structure. Core logic
  has been moved to the `src/` directory, following best practices for
  maintainable machine learning projects.

### ⚠️ Training Note

The modularized notebook is provided **without executed training outputs** due to
the high computational and financial cost of training DenseNet169 on high-resolution
chest X-ray images.

Model training requires access to a GPU-enabled environment (e.g., Google Colab Pro
or equivalent). The original notebook includes executed outputs demonstrating the
full training and evaluation process, while the modular version focuses on code
organization, reusability, and best practices for scalable machine learning
projects.

---

## 📄 Report
A complete academic-style report describing the methodology, experiments, and
results is available in the `reports/` folder.
> ⚠️ Note: The report is written in **Portuguese**, as it was originally produced for an academic course.
---

## 🚀 Future Work
- Class imbalance mitigation (weighted loss, oversampling)
- Model explainability using Grad-CAM
- Performance improvements for the Normal class
- External validation on additional datasets

---

## 👤 Authors
- Gustavo Nascimento Pavanelli  
- Gabriel Campello Dalbuquerque Lima  
- Marcel Capistrano Almeida Rodrigues
