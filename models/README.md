# Models Directory

This directory stores the machine learning and computer vision models used in the **Explainable AI-Based Food and Nutrition Analysis System**.

---

## 1. Computer Vision: Pretrained MobileNetV2 (ONNX)
- **Model File**: `mobilenetv2-7.onnx` (automatically cached or loaded upon first execution)
- **Framework**: ONNX Runtime / OpenCV DNN
- **Architecture**: MobileNetV2 (Sandler et al., CVPR 2018)
- **Input Dimensions**: 3 x 224 x 224 (RGB, normalized with ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`)
- **Output**: 1,000 ImageNet classification logits mapped to culinary food entities.
- **Why MobileNetV2?**: Ultra-lightweight (~14 MB), low latency for real-time mobile/web inference, and zero reliance on heavy multi-gigabyte frameworks.

---

## 2. Machine Learning: Nutrition Health Tier Classifier
- **Model File**: `nutrition_classifier.joblib`
- **Algorithm**: Random Forest Classifier / Decision Tree Classifier (`scikit-learn`)
- **Features Used**:
  - `calories` (kcal)
  - `protein` (g)
  - `carbohydrates` (g)
  - `fat` (g)
  - `fiber` (g)
  - `sugar` (g)
- **Target Classes**:
  1. `Good / Nutrient-Rich` (Score >= 70)
  2. `Moderate / Balanced` (Score 45 - 69.9)
  3. `Needs Improvement / Calorie-Dense` (Score < 45)
- **Evaluation Metrics**: Precision, Recall, F1-Score, Confusion Matrix.

---

## 3. Explainable AI: SHAP Explainer
- **Framework**: `shap` (TreeExplainer)
- **Purpose**: Computes Shapley values for each nutrient feature to explain local and global feature importance.
- **Transparency Complement**: Paired with deterministic rule-based point attribution for complete educational auditability.
