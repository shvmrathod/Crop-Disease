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

## Evaluation
- Confusion matrix shows strong diagonal dominance indicating correct predictions  
- Correct predictions demonstrate the model's ability to capture disease patterns  
- Incorrect predictions mainly occur between visually similar disease classes  

---

## Business Recommendation
MobileNetV2 is recommended for deployment in Syngenta’s mobile application due to its strong balance of accuracy, speed, and lightweight architecture. The model performs well on unseen data and is suitable for real-time disease detection in field conditions. Its compact size makes it ideal for mobile deployment, enabling farmers to quickly identify crop diseases and take corrective actions.

---

## Application (Gradio Demo)
A simple Gradio-based web application is included for real-time disease prediction.

### Run the app
```bash
python app.py