"""
Unit tests for database logging and image processing helpers.
"""

from pathlib import Path
import numpy as np
from PIL import Image
import pytest

from src.utils import init_db, save_analysis, get_analysis_history, clear_analysis_history
from src.image_processing import preprocess_for_model, compute_image_metrics


@pytest.fixture
def temp_db(tmp_path):
    """Fixture providing a temporary SQLite database path."""
    db_file = tmp_path / "test_history.db"
    init_db(db_file)
    return db_file


def test_sqlite_history_lifecycle(temp_db):
    """Test inserting, reading, and clearing history records."""
    # Initially empty
    df_empty = get_analysis_history(db_path=temp_db)
    assert df_empty.empty

    # Save a record
    dummy_nutrition = {
        "calories": 95.0,
        "protein": 0.5,
        "carbohydrates": 25.0,
        "fat": 0.3,
        "fiber": 4.4,
        "sugar": 19.0
    }
    saved = save_analysis(
        food_name="Apple",
        confidence=92.5,
        nutrition_info=dummy_nutrition,
        nutrition_score=78.2,
        health_tier="Good / Nutrient-Rich",
        source_type="AI Detected",
        db_path=temp_db
    )
    assert saved is True

    # Retrieve and verify
    df_history = get_analysis_history(db_path=temp_db)
    assert len(df_history) == 1
    assert df_history.iloc[0]["food_name"] == "Apple"
    assert df_history.iloc[0]["nutrition_score"] == 78.2

    # Clear history
    cleared = clear_analysis_history(db_path=temp_db)
    assert cleared is True
    assert get_analysis_history(db_path=temp_db).empty


def test_image_preprocessing_pipeline():
    """Test that image preprocessing produces correct batch shape and normalization."""
    # Create synthetic test image (100x150 RGB)
    synth_img = Image.new("RGB", (100, 150), color=(128, 64, 32))

    tensor = preprocess_for_model(synth_img, target_size=(224, 224))

    # Verify dimensions: (batch_size=1, channels=3, height=224, width=224)
    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == np.float32

    # Verify metrics
    metrics = compute_image_metrics(synth_img)
    assert metrics["width"] == 100
    assert metrics["height"] == 150
    assert "mean_brightness" in metrics
