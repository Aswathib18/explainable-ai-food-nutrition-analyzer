"""
Explainable AI (XAI) and Machine Learning Module.
Implements:
1. Transparent feature-contribution breakdown for the 0-100 Nutrition Score.
2. Supervised Machine Learning Classifier (Random Forest / Decision Tree) for Health Tier classification.
3. SHAP (SHapley Additive exPlanations) TreeExplainer for local and global model interpretability.
4. Data-driven educational nutritional recommendations.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import joblib
import logging

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import config
from src.scoring import calculate_nutrition_score

logger = logging.getLogger(__name__)

FEATURE_NAMES = ["calories", "protein", "carbohydrates", "fat", "fiber", "sugar"]


class NutritionXAIModel:
    """
    Manages the machine learning classification model, evaluation metrics,
    and SHAP explainer for nutritional health tier predictions.
    """

    def __init__(self, data_path: Path = config.NUTRITION_DATA_PATH):
        self.data_path = data_path
        self.model: Optional[RandomForestClassifier] = None
        self.explainer = None
        self.classes_: List[str] = []
        self.metrics: Dict[str, Any] = {}
        self.is_trained = False
        self._load_or_train()

    def _load_or_train(self) -> None:
        """Load pre-existing ML model or train a new one from the dataset."""
        if config.ML_MODEL_PATH.exists():
            try:
                saved_payload = joblib.load(config.ML_MODEL_PATH)
                self.model = saved_payload["model"]
                self.classes_ = list(saved_payload["classes"])
                self.metrics = saved_payload["metrics"]
                self.is_trained = True
                self._init_shap()
                logger.info("Loaded trained ML model from disk.")
                return
            except Exception as exc:
                logger.warning(f"Could not load saved model, re-training: {exc}")

        # Train model if not present or load failed
        self.train_model()

    def train_model(self) -> Dict[str, Any]:
        """
        Train a Random Forest Classifier on the nutrition dataset.
        Evaluates performance using train/test split (80/20) and computes
        accuracy, precision, recall, F1-score, and confusion matrix.
        """
        try:
            if not self.data_path.exists():
                return {"error": "Nutrition dataset file not found."}

            df = pd.read_csv(self.data_path)
            if df.empty or len(df) < 15:
                return {"error": "Dataset too small for meaningful ML training."}

            # Generate labels using the transparent scoring algorithm
            labels = []
            for _, row in df.iterrows():
                score_dict = calculate_nutrition_score(row.to_dict())
                labels.append(score_dict["health_tier"])

            df["health_tier"] = labels
            X = df[FEATURE_NAMES].fillna(0.0)
            y = df["health_tier"]

            # 80/20 Train-Test Split with stratification
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.20, random_state=42, stratify=y
            )

            # Train interpretable Random Forest Classifier
            clf = RandomForestClassifier(
                n_estimators=60,
                max_depth=4,
                min_samples_split=3,
                random_state=42
            )
            clf.fit(X_train, y_train)

            # Evaluation
            y_pred = clf.predict(X_test)
            classes = sorted(y.unique().tolist())

            acc = float(accuracy_score(y_test, y_pred))
            prec = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
            rec = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
            f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
            cm = confusion_matrix(y_test, y_pred, labels=classes).tolist()
            report = classification_report(y_test, y_pred, target_names=classes, output_dict=True, zero_division=0)

            self.model = clf
            self.classes_ = classes
            self.metrics = {
                "accuracy": round(acc * 100, 2),
                "precision": round(prec * 100, 2),
                "recall": round(rec * 100, 2),
                "f1_score": round(f1 * 100, 2),
                "confusion_matrix": cm,
                "classes": classes,
                "classification_report": report,
                "train_samples": len(X_train),
                "test_samples": len(X_test)
            }
            self.is_trained = True

            # Save model to disk
            config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
            joblib.dump(
                {
                    "model": clf,
                    "classes": classes,
                    "metrics": self.metrics
                },
                config.ML_MODEL_PATH
            )

            self._init_shap()
            return self.metrics

        except Exception as exc:
            logger.error(f"Error training ML model: {exc}")
            return {"error": str(exc)}

    def _init_shap(self) -> None:
        """Initialize SHAP TreeExplainer for the trained Random Forest model."""
        try:
            import shap
            if self.model is not None:
                self.explainer = shap.TreeExplainer(self.model)
        except Exception as exc:
            logger.warning(f"Could not initialize SHAP explainer: {exc}")
            self.explainer = None

    def explain_with_shap(self, nutrition_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Compute local SHAP values for the given food item's nutrient vector.
        Shows how each nutrient pushes the model toward or away from its predicted class.
        """
        if not self.is_trained or self.model is None or self.explainer is None:
            return None

        try:
            # Construct feature vector
            vector = np.array([[
                float(nutrition_info.get("calories", 0.0)),
                float(nutrition_info.get("protein", 0.0)),
                float(nutrition_info.get("carbohydrates", 0.0)),
                float(nutrition_info.get("fat", 0.0)),
                float(nutrition_info.get("fiber", 0.0)),
                float(nutrition_info.get("sugar", 0.0))
            ]])

            # Predict class and probability
            predicted_class = self.model.predict(vector)[0]
            probabilities = self.model.predict_proba(vector)[0]
            class_idx = list(self.model.classes_).index(predicted_class)
            confidence = float(probabilities[class_idx])

            # Compute SHAP values
            shap_values = self.explainer.shap_values(vector)

            # Handle multidimensional SHAP output for multi-class classifier
            if isinstance(shap_values, list):
                # Class-specific SHAP values
                class_shap = shap_values[class_idx][0]
            elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
                class_shap = shap_values[0, :, class_idx]
            else:
                class_shap = np.array(shap_values)[0]

            contributions = []
            for name, val, shap_val in zip(FEATURE_NAMES, vector[0], class_shap):
                contributions.append({
                    "feature": name.capitalize(),
                    "value": float(val),
                    "shap_value": round(float(shap_val), 4),
                    "direction": "Positive Support" if shap_val >= 0 else "Negative Opposition"
                })

            # Sort by absolute SHAP impact
            contributions = sorted(contributions, key=lambda x: abs(x["shap_value"]), reverse=True)

            return {
                "predicted_class": predicted_class,
                "confidence": round(confidence * 100, 1),
                "shap_contributions": contributions,
                "base_value": float(self.explainer.expected_value[class_idx])
                if isinstance(self.explainer.expected_value, (list, np.ndarray))
                else float(self.explainer.expected_value)
            }

        except Exception as exc:
            logger.error(f"Error computing SHAP values: {exc}")
            return None


def generate_dietary_recommendations(
    nutrition_info: Dict[str, Any],
    score_result: Dict[str, Any]
) -> List[str]:
    """
    Generate transparent, evidence-aligned educational dietary recommendations.
    Provides practical suggestions without making medical claims.
    """
    recommendations: List[str] = []

    calories = float(nutrition_info.get("calories", 0.0))
    protein = float(nutrition_info.get("protein", 0.0))
    fiber = float(nutrition_info.get("fiber", 0.0))
    sugar = float(nutrition_info.get("sugar", 0.0))
    fat = float(nutrition_info.get("fat", 0.0))
    score = score_result.get("score", 50.0)

    # 1. Fiber recommendations
    if fiber < 2.0:
        recommendations.append(
            "**Boost Dietary Fiber**: This item contains low fiber (< 2g). "
            "Consider pairing it with fiber-rich foods like leafy greens, legumes, chia seeds, or whole grains."
        )
    elif fiber >= 5.0:
        recommendations.append(
            "**Excellent Fiber Source**: Provides over 5g of dietary fiber, supporting smooth digestion and prolonged fullness."
        )

    # 2. Protein recommendations
    if protein < 4.0:
        recommendations.append(
            "**Pair with Protein**: This food is low in protein (< 4g). "
            "To support muscle repair and sustained energy, combine it with eggs, tofu, Greek yogurt, or lean poultry."
        )
    elif protein >= 15.0:
        recommendations.append(
            "**High-Protein Profile**: Delivers a substantial protein serving (≥ 15g), ideal for satiety and active lifestyles."
        )

    # 3. Sugar alerts
    if sugar > 15.0:
        recommendations.append(
            "**Mind Sugar Intake**: Contains elevated sugar (> 15g). "
            "Consider managing portion sizes to avoid rapid post-meal glucose spikes."
        )

    # 4. Fat & Calorie density
    if fat > 14.0 and calories > 300:
        recommendations.append(
            "**Energy-Dense Food**: Contains higher fat and calorie levels. "
            "Best enjoyed in mindful portions balanced with lighter, nutrient-dense side dishes."
        )

    # 5. Overall score praise
    if score >= 75.0:
        recommendations.append(
            "**Nutrient-Dense Choice**: This food achieved an outstanding reference score due to its high fiber/protein density and low sugar content."
        )

    if not recommendations:
        recommendations.append(
            "**Balanced Nutrient Profile**: This food provides a steady balance of macronutrients suitable for varied daily meals."
        )

    return recommendations


# Global singleton instance
_XAI_MODEL_INSTANCE: Optional[NutritionXAIModel] = None


def get_xai_model() -> NutritionXAIModel:
    """Retrieve singleton NutritionXAIModel instance."""
    global _XAI_MODEL_INSTANCE
    if _XAI_MODEL_INSTANCE is None:
        _XAI_MODEL_INSTANCE = NutritionXAIModel()
    return _XAI_MODEL_INSTANCE
