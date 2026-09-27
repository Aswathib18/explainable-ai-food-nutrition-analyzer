"""
Configuration Settings for Explainable AI-Based Food and Nutrition Analysis System.

Defines project paths, model configurations, nutritional scoring weights,
and daily recommended values based on standard dietary guidelines (e.g., USDA/FDA).
"""

from pathlib import Path

# ==============================================================================
# BASE DIRECTORIES
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
SRC_DIR = BASE_DIR / "src"
ASSETS_DIR = BASE_DIR / "assets"
TESTS_DIR = BASE_DIR / "tests"

# ==============================================================================
# DATA FILES
# ==============================================================================
NUTRITION_DATA_PATH = DATA_DIR / "nutrition_data.csv"
FOOD_ALIASES_PATH = DATA_DIR / "food_aliases.csv"
DB_PATH = DATA_DIR / "analysis_history.db"

# ==============================================================================
# MODEL CONFIGURATIONS
# ==============================================================================
# ONNX MobileNetV2 pretrained vision model
MOBILENET_MODEL_FILENAME = "mobilenetv2-7.onnx"
MOBILENET_MODEL_PATH = MODELS_DIR / MOBILENET_MODEL_FILENAME
MOBILENET_MODEL_URL = (
    "https://media.githubusercontent.com/media/onnx/models/main/validated/vision/"
    "classification/mobilenet/model/mobilenetv2-7.onnx"
)
IMAGENET_CLASSES_PATH = MODELS_DIR / "imagenet_classes.txt"
IMAGENET_CLASSES_URL = (
    "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
)

# ML Nutrition Health Classification Model
ML_MODEL_PATH = MODELS_DIR / "nutrition_classifier.joblib"

# Computer Vision Preprocessing Parameters
IMAGE_INPUT_SIZE = (224, 224)
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]
CONFIDENCE_THRESHOLD = 0.30  # Threshold below which manual fallback is advised

# Supported Image Formats
ALLOWED_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp"]
MAX_IMAGE_SIZE_MB = 10

# ==============================================================================
# DAILY RECOMMENDED REFERENCE VALUES (Adult 2000 kcal Diet - FDA Reference)
# ==============================================================================
DAILY_REFERENCE_VALUES = {
    "calories": 2000.0,      # kcal
    "protein": 50.0,         # grams
    "carbohydrates": 275.0,  # grams
    "fat": 78.0,             # grams
    "fiber": 28.0,           # grams
    "sugar": 50.0            # grams (max recommended free/added sugar benchmark)
}

# ==============================================================================
# NUTRITION SCORING WEIGHTS & THRESHOLDS (0 - 100 Reference Score)
# ==============================================================================
# Baseline starting score is 50.
# Points are awarded for beneficial nutrients (fiber, protein) and balanced calories,
# while penalties are applied for excessive sugar, saturated/total fat, and extreme calorie density.
SCORING_CONFIG = {
    "baseline_score": 50.0,
    "weights": {
        "fiber_bonus": 2.5,        # Up to +20 points for high dietary fiber
        "protein_bonus": 1.0,      # Up to +15 points for protein content
        "sugar_penalty": 1.2,      # Up to -20 points for high sugar
        "fat_penalty": 0.8,        # Up to -15 points for high total fat
        "calorie_density_penalty": 0.05,  # Penalty for excessive calories per serving
        "nutrient_balance_bonus": 5.0     # Bonus for well-balanced macro profile
    },
    "caps": {
        "min_score": 0.0,
        "max_score": 100.0
    }
}

# Health Category Tier Boundaries
HEALTH_TIERS = {
    "Good / Nutrient-Rich": (70.0, 100.0),
    "Moderate / Balanced": (45.0, 69.9),
    "Needs Improvement / Calorie-Dense": (0.0, 44.9)
}

# UI Theme & Colors
UI_THEME = {
    "primary": "#2E7D32",     # Forest Green
    "secondary": "#1565C0",   # Deep Blue
    "accent": "#F57C00",      # Amber / Orange
    "danger": "#D32F2F",      # Crimson
    "background_card": "#F8F9FA",
    "text_dark": "#212121"
}
