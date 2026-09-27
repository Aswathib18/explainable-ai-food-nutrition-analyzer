"""
Unit tests for nutrition dataset loading, alias resolution, and food lookup.
"""

import pytest
import config
from src.nutrition import NutritionDatabase, normalize_food_name


@pytest.fixture
def nutrition_db():
    """Fixture providing an initialized NutritionDatabase instance."""
    return NutritionDatabase(
        nutrition_path=config.NUTRITION_DATA_PATH,
        aliases_path=config.FOOD_ALIASES_PATH
    )


def test_dataset_loading(nutrition_db):
    """Test that nutrition dataset loads properly with required columns and entries."""
    assert not nutrition_db.df_nutrition.empty, "Nutrition DataFrame should not be empty"
    assert len(nutrition_db.df_nutrition) >= 50, "Dataset should contain at least 50 food items"

    required_columns = [
        "food", "calories", "protein", "carbohydrates",
        "fat", "fiber", "sugar", "serving_size", "serving_unit", "category"
    ]
    for col in required_columns:
        assert col in nutrition_db.df_nutrition.columns, f"Missing required column: {col}"


def test_normalization():
    """Test string normalization logic."""
    assert normalize_food_name("  Apple!  ") == "apple"
    assert normalize_food_name("Chicken_Breast??") == "chicken_breast"
    assert normalize_food_name("Red-Delicious   Apple") == "red delicious apple"
    assert normalize_food_name("") == ""


def test_canonical_food_lookup(nutrition_db):
    """Test retrieval of canonical foods."""
    apple_info = nutrition_db.get_nutrition_info("Apple")
    assert apple_info is not None, "Apple should be found in dataset"
    assert apple_info["food"] == "Apple"
    assert apple_info["calories"] > 0
    assert "daily_value_percentages" in apple_info


def test_alias_matching(nutrition_db):
    """Test that common aliases resolve correctly to canonical foods."""
    # "granny smith" -> Apple
    res1 = nutrition_db.resolve_food_name("granny smith apple")
    assert res1 == "Apple"

    # "cheeseburger" -> Burger
    res2 = nutrition_db.resolve_food_name("cheeseburger")
    assert res2 == "Burger"

    # "white rice" -> White Rice
    res3 = nutrition_db.resolve_food_name("cooked rice")
    assert res3 == "White Rice"


def test_unknown_food_handling(nutrition_db):
    """Test that non-existent foods return None rather than fabricating values."""
    info = nutrition_db.get_nutrition_info("UnobtainiumFruitXYZ999")
    assert info is None, "Non-existent food should return None"
