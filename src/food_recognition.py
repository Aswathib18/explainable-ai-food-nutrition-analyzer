"""
Food Recognition Pipeline using Pretrained Computer Vision Models.
Implements deep learning inference via ONNX Runtime and maps detected classes
to canonical food items in the nutritional database.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
from PIL import Image
import logging

import config
from src.image_processing import preprocess_for_model
from src.utils import ensure_model_files

logger = logging.getLogger(__name__)

# ==============================================================================
# IMAGENET FOOD CLASS MAPPING DICTIONARY
# Maps raw ImageNet synsets/classes to canonical items in nutrition_data.csv
# ==============================================================================
IMAGENET_TO_FOOD_MAP: Dict[str, str] = {
    # Fruits
    "granny smith": "Apple",
    "apple": "Apple",
    "custard apple": "Apple",
    "pomegranate": "Apple",
    "banana": "Banana",
    "orange": "Orange",
    "strawberry": "Strawberry",
    "lemon": "Lemon",
    "pineapple": "Pineapple",
    "fig": "Peach",
    "jackfruit": "Mango",

    # Vegetables & Starches
    "broccoli": "Broccoli",
    "cauliflower": "Cauliflower",
    "zucchini": "Cucumber",
    "cucumber": "Cucumber",
    "bell pepper": "Bell Pepper",
    "sweet pepper": "Bell Pepper",
    "head cabbage": "Lettuce",
    "mushroom": "Mushroom",
    "mashed potato": "Potato",
    "acorn squash": "Sweet Potato",
    "butternut squash": "Sweet Potato",
    "artichoke": "Broccoli",
    "cardoon": "Broccoli",
    "corn": "Corn",

    # Fast Food, Bread & Prepared Meals
    "pizza": "Pizza",
    "cheeseburger": "Burger",
    "hamburger": "Burger",
    "hotdog": "Hot Dog",
    "bagel": "Bagel",
    "pretzel": "Pretzel",
    "burrito": "Burrito",
    "french loaf": "White Bread",
    "bakery": "White Bread",
    "dough": "White Bread",
    "guacamole": "Avocado",
    "ice cream": "Ice Cream",
    "ice lolly": "Ice Cream",
    "chocolate sauce": "Chocolate Bar",
    "trifle": "Chocolate Bar",
    "carbonara": "Pasta",
    "spaghetti squash": "Pasta",
    "meat loaf": "Beef Steak",
    "potpie": "Chicken Noodle Soup",
    "consomme": "Chicken Noodle Soup",
    "hot pot": "Chicken Noodle Soup",
    "soup bowl": "Chicken Noodle Soup",
    "espresso": "Espresso",
    "eggnog": "Milk",
    "plate": "Green Salad"
}


def softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax probabilities with numerical stability."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


class FoodClassifier:
    """
    Deep Learning Food Classifier using local ONNX Runtime inference.
    Loads MobileNetV2 and runs local image classification without external APIs.
    """

    def __init__(self):
        self.session = None
        self.classes: List[str] = []
        self.is_ready = False
        self.fallback_mode = False
        self.status_message = ""
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Load ONNX runtime session and ImageNet classes."""
        try:
            # Ensure model files exist or download them
            success, msg = ensure_model_files()
            if not success:
                self.fallback_mode = True
                self.status_message = f"Demo/Fallback mode — {msg}"
                logger.warning(self.status_message)
                return

            # Read class labels
            if config.IMAGENET_CLASSES_PATH.exists():
                with open(config.IMAGENET_CLASSES_PATH, "r", encoding="utf-8") as f:
                    self.classes = [line.strip() for line in f if line.strip()]

            # Load ONNX Inference Session
            import onnxruntime as ort
            # Use CPU execution provider for maximum universal compatibility
            self.session = ort.InferenceSession(
                str(config.MOBILENET_MODEL_PATH),
                providers=["CPUExecutionProvider"]
            )
            self.input_name = self.session.get_inputs()[0].name
            self.is_ready = True
            self.fallback_mode = False
            self.status_message = "MobileNetV2 ONNX Model loaded successfully (Local Inference)."
            logger.info(self.status_message)

        except Exception as exc:
            self.is_ready = False
            self.fallback_mode = True
            self.status_message = f"Demo/Fallback mode — food recognition model is not currently available: {exc}"
            logger.error(self.status_message)

    def classify_image(self, image: Image.Image) -> Dict[str, Any]:
        """
        Run inference on the provided PIL Image.
        Returns:
            dict containing food_name, raw_label, confidence, top_predictions,
            is_fallback_mode, and mode_description.
        """
        if not self.is_ready or self.session is None or self.fallback_mode:
            return {
                "food_name": "Apple",
                "raw_label": "Demo / Fallback Mode",
                "confidence": 0.0,
                "top_predictions": [
                    {"food_name": "Apple", "confidence": 0.0, "raw_label": "Fallback"},
                    {"food_name": "Banana", "confidence": 0.0, "raw_label": "Fallback"},
                    {"food_name": "Salad", "confidence": 0.0, "raw_label": "Fallback"}
                ],
                "is_fallback_mode": True,
                "mode_description": "Demo/Fallback mode — food recognition model is not currently available."
            }

        try:
            # 1. Preprocess image into batch tensor (1, 3, 224, 224)
            tensor = preprocess_for_model(image)

            # 2. Run inference
            raw_outputs = self.session.run(None, {self.input_name: tensor})
            logits = raw_outputs[0][0]

            # 3. Softmax probabilities
            probabilities = softmax(logits)

            # 4. Extract top candidate indices
            top_k_indices = np.argsort(probabilities)[::-1][:15]

            food_candidates: List[Dict[str, Any]] = []
            seen_canonical = set()

            for idx in top_k_indices:
                class_label = self.classes[idx] if idx < len(self.classes) else f"class_{idx}"
                class_label_lower = class_label.lower()
                prob = float(probabilities[idx])

                # Check if class label matches any known food in mapping
                matched_canonical = None
                for map_key, food_val in IMAGENET_TO_FOOD_MAP.items():
                    if map_key in class_label_lower:
                        matched_canonical = food_val
                        break

                if matched_canonical and matched_canonical not in seen_canonical:
                    seen_canonical.add(matched_canonical)
                    food_candidates.append({
                        "food_name": matched_canonical,
                        "raw_label": class_label.title(),
                        "confidence": round(prob * 100, 1)
                    })

                if len(food_candidates) >= 5:
                    break

            # If no direct food match found in top 15, inspect top 1 general class
            if not food_candidates:
                top_idx = top_k_indices[0]
                top_label = self.classes[top_idx] if top_idx < len(self.classes) else "Unknown Item"
                top_prob = float(probabilities[top_idx])
                return {
                    "food_name": "Unknown Food",
                    "raw_label": top_label.title(),
                    "confidence": round(top_prob * 100, 1),
                    "top_predictions": [
                        {"food_name": "Unknown", "confidence": round(top_prob * 100, 1), "raw_label": top_label}
                    ],
                    "is_fallback_mode": False,
                    "mode_description": "Pre-trained MobileNetV2 Deep Learning Computer Vision Model"
                }

            top_pred = food_candidates[0]
            return {
                "food_name": top_pred["food_name"],
                "raw_label": top_pred["raw_label"],
                "confidence": top_pred["confidence"],
                "top_predictions": food_candidates[:3],
                "is_fallback_mode": False,
                "mode_description": "Pre-trained MobileNetV2 Deep Learning Computer Vision Model"
            }

        except Exception as exc:
            logger.error(f"Inference error in food recognition: {exc}")
            return {
                "food_name": "Apple",
                "raw_label": "Error Recovery",
                "confidence": 0.0,
                "top_predictions": [],
                "is_fallback_mode": True,
                "mode_description": f"Demo/Fallback mode — Error occurred during inference: {exc}"
            }


# Singleton instance
_CLASSIFIER_INSTANCE: Optional[FoodClassifier] = None


def get_food_classifier() -> FoodClassifier:
    """Retrieve singleton FoodClassifier instance."""
    global _CLASSIFIER_INSTANCE
    if _CLASSIFIER_INSTANCE is None:
        _CLASSIFIER_INSTANCE = FoodClassifier()
    return _CLASSIFIER_INSTANCE
