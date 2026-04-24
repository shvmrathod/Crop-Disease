# Crop Disease Classification (Syngenta Task)

## Problem Statement
Build a machine learning model to classify plant diseases from leaf images to support early detection and timely intervention for farmers.

---

## Dataset
- PlantVillage Dataset (Kaggle)
- Subset of 10 classes selected for faster experimentation and training

---

## Approach
- Image preprocessing and augmentation (rotation, flip, zoom)
- Transfer Learning using MobileNetV2
- Train-validation split (80-20)

---

## Model Performance

| Metric | Value |
|--------|-------|
| Training Accuracy | ~91% |
| Validation Accuracy | ~92% |
| Epochs | 10 |

---

## Key Insights
- Dataset is clean and well-labeled  
- Slight class imbalance observed across categories  
- Model generalizes well with minimal overfitting  

---

## Business Recommendation
MobileNetV2 is recommended for deployment in Syngenta’s mobile application due to its strong balance of accuracy, speed, and lightweight architecture. The model performs well on unseen data and is suitable for real-time disease detection in field conditions. Its compact size makes it ideal for mobile deployment, enabling farmers to quickly identify crop diseases and take corrective actions.

---

## Setup Instructions

```bash
pip install tensorflow-macos tensorflow-metal
pip install numpy pandas matplotlib seaborn scikit-learn pillow kaggle jupyter