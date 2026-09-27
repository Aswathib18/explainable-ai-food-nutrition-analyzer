"""
Utility functions for database management, image validation, file downloads,
and formatting in the Food and Nutrition Analysis System.
"""

import sqlite3
import urllib.request
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd
from PIL import Image

import config

logger = logging.getLogger(__name__)

# ==============================================================================
# SQLITE DATABASE UTILITIES (ANALYSIS HISTORY)
# ==============================================================================

def init_db(db_path: Path = config.DB_PATH) -> None:
    """
    Initialize SQLite database for storing food analysis history.
    Includes backward-compatible column upgrades for packaged food scanning.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS analysis_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                food_name TEXT NOT NULL,
                confidence REAL NOT NULL,
                calories REAL NOT NULL,
                protein REAL NOT NULL,
                carbohydrates REAL NOT NULL,
                fat REAL NOT NULL,
                fiber REAL NOT NULL,
                sugar REAL NOT NULL,
                nutrition_score REAL NOT NULL,
                health_tier TEXT NOT NULL,
                source_type TEXT NOT NULL,
                product_name TEXT,
                analysis_mode TEXT,
                serving_size TEXT,
                saturated_fat REAL,
                trans_fat REAL,
                added_sugar REAL,
                sodium REAL,
                ocr_confidence REAL,
                detected_allergens TEXT
            )
            """
        )

        # Ensure any existing databases have the new packaged food columns
        cursor.execute("PRAGMA table_info(analysis_history)")
        existing_cols = [row[1] for row in cursor.fetchall()]

        new_columns = [
            ("product_name", "TEXT"),
            ("analysis_mode", "TEXT"),
            ("serving_size", "TEXT"),
            ("saturated_fat", "REAL"),
            ("trans_fat", "REAL"),
            ("added_sugar", "REAL"),
            ("sodium", "REAL"),
            ("ocr_confidence", "REAL"),
            ("detected_allergens", "TEXT")
        ]

        for col_name, col_type in new_columns:
            if col_name not in existing_cols:
                try:
                    cursor.execute(f"ALTER TABLE analysis_history ADD COLUMN {col_name} {col_type}")
                except Exception as exc:
                    logger.debug(f"Column {col_name} might already exist: {exc}")

        conn.commit()


def save_analysis(
    food_name: str,
    confidence: float,
    nutrition_info: Dict[str, Any],
    nutrition_score: float,
    health_tier: str,
    source_type: str = "AI Detected",
    analysis_mode: str = "Food Image Analysis",
    serving_size: Optional[str] = None,
    detected_allergens: Optional[str] = None,
    ocr_confidence: Optional[float] = None,
    db_path: Path = config.DB_PATH
) -> bool:
    """
    Save a completed analysis result to SQLite database.
    Does NOT store raw image files to respect privacy and maintain light storage.
    Supports both food recognition and packaged food scanning fields.
    """
    try:
        init_db(db_path)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Extract values safely
        cal = float(nutrition_info.get("calories") or 0.0)
        prot = float(nutrition_info.get("protein") or 0.0)
        carb = float(nutrition_info.get("carbohydrates") or 0.0)
        fat_val = float(nutrition_info.get("fat") or nutrition_info.get("total_fat") or 0.0)
        fib = float(nutrition_info.get("fiber") or 0.0)
        sug = float(nutrition_info.get("sugar") or nutrition_info.get("total_sugar") or 0.0)

        sat_fat = float(nutrition_info["saturated_fat"]) if nutrition_info.get("saturated_fat") is not None else None
        trans_fat = float(nutrition_info["trans_fat"]) if nutrition_info.get("trans_fat") is not None else None
        added_sug = float(nutrition_info["added_sugar"]) if nutrition_info.get("added_sugar") is not None else None
        sod = float(nutrition_info["sodium"]) if nutrition_info.get("sodium") is not None else None

        serving = serving_size or str(nutrition_info.get("serving_size") or "")
        prod_name = nutrition_info.get("product_name") or food_name

        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO analysis_history (
                    timestamp, food_name, confidence, calories, protein,
                    carbohydrates, fat, fiber, sugar, nutrition_score,
                    health_tier, source_type, product_name, analysis_mode,
                    serving_size, saturated_fat, trans_fat, added_sugar,
                    sodium, ocr_confidence, detected_allergens
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp,
                    food_name,
                    round(float(confidence), 3),
                    round(cal, 1),
                    round(prot, 1),
                    round(carb, 1),
                    round(fat_val, 1),
                    round(fib, 1),
                    round(sug, 1),
                    round(float(nutrition_score), 1),
                    health_tier,
                    source_type,
                    prod_name,
                    analysis_mode,
                    serving,
                    sat_fat,
                    trans_fat,
                    added_sug,
                    sod,
                    round(float(ocr_confidence), 3) if ocr_confidence is not None else None,
                    detected_allergens
                )
            )
            conn.commit()
        return True
    except Exception as exc:
        logger.error(f"Error saving analysis to SQLite: {exc}")
        return False


def get_analysis_history(limit: int = 50, db_path: Path = config.DB_PATH) -> pd.DataFrame:
    """Retrieve recent analysis records as a pandas DataFrame."""
    try:
        init_db(db_path)
        with sqlite3.connect(db_path) as conn:
            query = f"""
                SELECT id, timestamp, food_name, product_name, analysis_mode,
                       serving_size, confidence, ocr_confidence, calories,
                       protein, carbohydrates, fat, saturated_fat, trans_fat,
                       fiber, sugar, added_sugar, sodium,
                       nutrition_score, health_tier, detected_allergens, source_type
                FROM analysis_history
                ORDER BY id DESC
                LIMIT {limit}
            """
            df = pd.read_sql_query(query, conn)
            return df
    except Exception as exc:
        logger.error(f"Error retrieving analysis history: {exc}")
        return pd.DataFrame()


def clear_analysis_history(db_path: Path = config.DB_PATH) -> bool:
    """Clear all records from the analysis history database."""
    try:
        init_db(db_path)
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM analysis_history")
            conn.commit()
        return True
    except Exception as exc:
        logger.error(f"Error clearing history: {exc}")
        return False


# ==============================================================================
# IMAGE & FILE VALIDATION UTILITIES
# ==============================================================================

def validate_uploaded_image(file_obj) -> tuple[bool, str]:
    """
    Validate uploaded image object.
    Checks:
    - Object presence
    - Extension
    - File size
    - Corrupted image check via PIL
    Returns (is_valid, error_message).
    """
    if file_obj is None:
        return False, "No file uploaded. Please upload a food image."

    file_name = getattr(file_obj, "name", "").lower()
    suffix = Path(file_name).suffix.lower()

    if suffix not in config.ALLOWED_IMAGE_EXTENSIONS:
        return False, f"Unsupported file type '{suffix}'. Allowed formats: {', '.join(config.ALLOWED_IMAGE_EXTENSIONS)}"

    size_bytes = getattr(file_obj, "size", None)
    if size_bytes and (size_bytes > config.MAX_IMAGE_SIZE_MB * 1024 * 1024):
        return False, f"File size exceeds maximum allowed limit of {config.MAX_IMAGE_SIZE_MB} MB."

    try:
        image = Image.open(file_obj)
        image.verify()
        if hasattr(file_obj, "seek"):
            file_obj.seek(0)
        return True, ""
    except Exception as exc:
        return False, f"Invalid or corrupted image file: {str(exc)}"


# ==============================================================================
# MODEL DOWNLOAD HELPER
# ==============================================================================

def ensure_model_files() -> tuple[bool, str]:
    """
    Ensures pre-trained ONNX model and ImageNet classes file exist in models/ directory.
    Downloads them automatically if missing.
    Returns (success, message).
    """
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. ImageNet classes
    if not config.IMAGENET_CLASSES_PATH.exists():
        try:
            req = urllib.request.Request(
                config.IMAGENET_CLASSES_URL,
                headers={"User-Agent": "Mozilla/5.0"}
            )
            data = urllib.request.urlopen(req, timeout=15).read()
            config.IMAGENET_CLASSES_PATH.write_bytes(data)
        except Exception as exc:
            return False, f"Could not download ImageNet labels: {exc}"

    # 2. MobileNetV2 ONNX model
    if not config.MOBILENET_MODEL_PATH.exists() or config.MOBILENET_MODEL_PATH.stat().st_size < 1000000:
        try:
            req = urllib.request.Request(
                config.MOBILENET_MODEL_URL,
                headers={"User-Agent": "Mozilla/5.0"}
            )
            data = urllib.request.urlopen(req, timeout=30).read()
            config.MOBILENET_MODEL_PATH.write_bytes(data)
        except Exception as exc:
            return False, f"Could not download MobileNetV2 ONNX model: {exc}"

    return True, "Models verified."
