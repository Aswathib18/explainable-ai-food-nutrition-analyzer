"""
Unit tests for transparent nutrition scoring algorithm and boundary checks.
"""

import pytest
from src.scoring import calculate_nutrition_score


def test_score_boundary_guarantee():
    """Ensure nutrition score is always clamped strictly between 0 and 100."""
    # Extreme high-calorie, ultra-processed profile
    worst_case = {
        "calories": 2000.0,
        "protein": 0.0,
        "carbohydrates": 300.0,
        "fat": 150.0,
        "fiber": 0.0,
        "sugar": 250.0
    }
    result_worst = calculate_nutrition_score(worst_case)
    assert 0.0 <= result_worst["score"] <= 100.0, "Score must never drop below 0"
    assert result_worst["score"] == 0.0, "Extreme unhealthy food should clamp at 0"

    # Extreme superfood profile
    best_case = {
        "calories": 100.0,
        "protein": 40.0,
        "carbohydrates": 20.0,
        "fat": 1.0,
        "fiber": 25.0,
        "sugar": 1.0
    }
    result_best = calculate_nutrition_score(best_case)
    assert 0.0 <= result_best["score"] <= 100.0, "Score must never exceed 100"
    assert result_best["score"] >= 80.0, "High fiber & protein food should achieve high score"


def test_healthy_food_scoring():
    """Test realistic healthy food (e.g., Broccoli)."""
    broccoli = {
        "calories": 55.0,
        "protein": 3.7,
        "carbohydrates": 11.2,
        "fat": 0.6,
        "fiber": 5.1,
        "sugar": 2.6
    }
    result = calculate_nutrition_score(broccoli)
    assert result["score"] >= 70.0, "Broccoli should score in the Good tier (>= 70)"
    assert result["health_tier"] == "Good / Nutrient-Rich"

    # Check factors
    features = [f["feature"] for f in result["factors"]]
    assert "Dietary Fiber" in features, "Fiber should contribute to broccoli score"


def test_processed_food_scoring():
    """Test realistic calorie-dense food (e.g., French Fries / Donut)."""
    donut = {
        "calories": 300.0,
        "protein": 3.0,
        "carbohydrates": 40.0,
        "fat": 15.0,
        "fiber": 1.0,
        "sugar": 25.0
    }
    result = calculate_nutrition_score(donut)
    assert result["score"] < 50.0, "Donut should score below 50 due to high sugar and fat"
    assert any(f["status"] == "negative" and "Sugar" in f["feature"] for f in result["factors"])
    assert any(f["status"] == "negative" and "Fat" in f["feature"] for f in result["factors"])


def test_zero_values_handling():
    """Test handling of foods with zero values (e.g. Water or Black Tea)."""
    zero_food = {
        "calories": 0.0,
        "protein": 0.0,
        "carbohydrates": 0.0,
        "fat": 0.0,
        "fiber": 0.0,
        "sugar": 0.0
    }
    result = calculate_nutrition_score(zero_food)
    assert 0.0 <= result["score"] <= 100.0
    assert result["score"] == 50.0, "Neutral food with zero attributes should match baseline score (50.0)"
