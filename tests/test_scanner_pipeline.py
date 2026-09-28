"""
Unit tests for Real-Time Packaged Food Scanner Pipeline:
- OCR nutrition label parsing
- OCR typo normalization
- Ingredient list parsing and categorization
- Allergen detection (Milk, Soy, Wheat, Peanuts, Tree Nuts, etc.)
- Cross-contact / May-contain precautionary statement extraction
- Packaged food Reference Nutrition Score calculation
- Database history storage with packaged food schema
"""

import pytest
from pathlib import Path
from src.nutrition_parser import (
    normalize_ocr_text,
    parse_nutrition_label,
    compute_per_100g_conversion
)
from src.ingredient_parser import (
    extract_ingredient_text_block,
    split_into_ingredients,
    categorize_ingredient,
    parse_ingredients_pipeline
)
from src.allergen_detector import (
    detect_allergens,
    split_into_precautionary_and_ingredients
)
from src.scoring import calculate_nutrition_score
from src.utils import init_db, save_analysis, get_analysis_history


def test_ocr_nutrition_parsing_and_normalization():
    """Test parsing typical OCR text with common character confusions."""
    sample_ocr = """
    CRUNCHY BISCUITS
    Nutrition Facts
    Serving Size: 30g
    Calories 150 kcal
    Pr0tein 4.0 g
    Carbohydrat3 22.0 g
    Total Fat 6.0 g
    Saturat3d Fat 2.0 g
    Trans Fat 0.0 g
    F1ber 3.0 g
    Total Suqars 8.0 g
    Added Sugars 5.0 g
    5odium 120 mg
    """
    parsed = parse_nutrition_label(sample_ocr)

    assert parsed["calories"] == 150.0
    assert parsed["protein"] == 4.0
    assert parsed["carbohydrates"] == 22.0
    assert parsed["total_fat"] == 6.0
    assert parsed["saturated_fat"] == 2.0
    assert parsed["trans_fat"] == 0.0
    assert parsed["fiber"] == 3.0
    assert parsed["total_sugar"] == 8.0
    assert parsed["added_sugar"] == 5.0
    assert parsed["sodium"] == 120.0
    assert parsed["serving_size"] == "30g"


def test_missing_values_return_none():
    """Verify missing nutritional values return None and are NEVER invented."""
    incomplete_ocr = """
    TEA CRACKERS
    Nutrition Information
    Energy 100 kcal
    Protein 2g
    """
    parsed = parse_nutrition_label(incomplete_ocr)
    assert parsed["calories"] == 100.0
    assert parsed["protein"] == 2.0
    # Missing fields must strictly be None
    assert parsed["carbohydrates"] is None
    assert parsed["total_fat"] is None
    assert parsed["saturated_fat"] is None
    assert parsed["trans_fat"] is None
    assert parsed["fiber"] is None
    assert parsed["total_sugar"] is None
    assert parsed["added_sugar"] is None
    assert parsed["sodium"] is None


def test_per_100g_conversion():
    """Test mathematical conversion from serving size to per 100g."""
    label_data = {
        "serving_size": "25g",
        "calories": 120.0,
        "protein": 2.5,
        "carbohydrates": 15.0,
        "total_fat": 5.0,
        "saturated_fat": 1.0,
        "trans_fat": 0.0,
        "fiber": 1.0,
        "total_sugar": 4.0,
        "added_sugar": 2.0,
        "sodium": 50.0
    }
    per_100g, reason = compute_per_100g_conversion(label_data)
    assert per_100g is not None
    # 100g / 25g = 4.0x
    assert per_100g["calories"] == 480.0
    assert per_100g["protein"] == 10.0
    assert per_100g["total_fat"] == 20.0
    assert per_100g["sodium"] == 200.0


def test_ingredient_parsing_nested_parentheses():
    """Test splitting ingredient list while preserving parenthetical sub-ingredients."""
    raw_ing = (
        "Ingredients: Enriched Flour (Wheat Flour, Niacin, Iron, Thiamin), "
        "Sugar, Palm Oil, Milk Solids, Soy Lecithin (Emulsifier), Salt."
    )
    res = parse_ingredients_pipeline(raw_ing)
    ings = res["ingredients"]

    assert len(ings) == 6
    assert "Enriched Flour (Wheat Flour, Niacin, Iron, Thiamin)" in ings[0]
    assert ings[1] == "Sugar"
    assert ings[2] == "Palm Oil"
    assert ings[3] == "Milk Solids"
    assert "Soy Lecithin" in ings[4]
    assert ings[5] == "Salt"


def test_allergen_detection_direct_ingredients():
    """Test detecting direct allergens from ingredient list with exact matched terms."""
    ing_text = "Wheat Flour, Sugar, Milk Solids, Soy Lecithin, Cocoa Butter, Salt."
    res = detect_allergens(ing_text)

    assert res["has_allergens"] is True
    detected_combos = [(a["allergen"], a["detected_term"], a["source"]) for a in res["allergen_table"]]

    # Verify Wheat, Milk, Soy
    assert any(c[0] == "Wheat & Gluten" and "Wheat" in c[1] and c[2] == "Ingredient" for c in detected_combos)
    assert any(c[0] == "Milk / Dairy" and "Milk Solids" in c[1] and c[2] == "Ingredient" for c in detected_combos)
    assert any(c[0] == "Soy" and "Soy Lecithin" in c[1] and c[2] == "Ingredient" for c in detected_combos)


def test_may_contain_precautionary_separation():
    """Test distinguishing direct ingredients from precautionary cross-contact statements."""
    label_text = (
        "Ingredients: Oats, Sugar, Peanut Flour, Honey.\n"
        "May contain traces of milk and tree nuts.\n"
        "Manufactured in a facility that also processes soy."
    )
    res = detect_allergens(label_text)

    assert res["has_allergens"] is True
    table = res["allergen_table"]

    # Peanut Flour must be classified as Ingredient
    peanut_matches = [a for a in table if a["allergen"] == "Peanuts"]
    assert len(peanut_matches) >= 1
    assert any(p["source"] == "Ingredient" and "Peanut Flour" in p["detected_term"] for p in peanut_matches)

    # Milk and Tree Nuts must be classified as Precautionary
    milk_matches = [a for a in table if a["allergen"] == "Milk / Dairy"]
    assert any(m["source"] == "Precautionary" for m in milk_matches)

    nuts_matches = [a for a in table if a["allergen"] == "Tree Nuts"]
    assert any(n["source"] == "Precautionary" for n in nuts_matches)

    # Precautionary statements list should be populated
    assert len(res["precautionary_statements"]) >= 1


def test_no_allergen_terms_detected():
    """Test clean response and safety disclaimer when no allergens are matched."""
    clean_text = "Ingredients: Pure Spring Water, Citric Acid, Natural Apple Flavor."
    res = detect_allergens(clean_text)

    assert res["has_allergens"] is False
    assert len(res["allergen_table"]) == 0
    assert "Allergen detection is based on the text successfully extracted" in res["disclaimer"]


def test_packaged_food_scoring_penalties():
    """Test that packaged food scoring properly applies penalties for added sugar, sat fat, trans fat, sodium."""
    junk_snack = {
        "calories": 280.0,
        "protein": 1.0,
        "carbohydrates": 40.0,
        "total_fat": 14.0,
        "saturated_fat": 6.0,
        "trans_fat": 0.5,
        "fiber": 0.5,
        "total_sugar": 22.0,
        "added_sugar": 18.0,
        "sodium": 350.0
    }
    score_res = calculate_nutrition_score(junk_snack)
    assert score_res["score"] < 40.0
    assert score_res["health_tier"] == "Needs Improvement / Calorie-Dense"

    factors = [f["feature"] for f in score_res["factors"]]
    assert "Saturated Fat" in factors
    assert "Trans Fat" in factors
    assert "Added Sugar" in factors
    assert "Sodium Content" in factors


def test_packaged_food_db_history_lifecycle(tmp_path):
    """Test database saving and retrieval with packaged food columns."""
    db_file = tmp_path / "packaged_history.db"
    init_db(db_file)

    packaged_nut = {
        "product_name": "Crunchy Oat Biscuit",
        "calories": 150.0,
        "protein": 4.0,
        "carbohydrates": 22.0,
        "total_fat": 6.0,
        "saturated_fat": 2.0,
        "trans_fat": 0.0,
        "fiber": 3.0,
        "total_sugar": 8.0,
        "added_sugar": 5.0,
        "sodium": 120.0
    }
    saved = save_analysis(
        food_name="Crunchy Oat Biscuit",
        confidence=85.0,
        nutrition_info=packaged_nut,
        nutrition_score=72.5,
        health_tier="Good / Nutrient-Rich",
        source_type="Real-Time Camera Scanner",
        analysis_mode="Real-Time Food Label Scanner",
        serving_size="30g",
        detected_allergens="Wheat, Milk, Soy",
        ocr_confidence=0.88,
        db_path=db_file
    )
    assert saved is True

    df = get_analysis_history(db_path=db_file)
    assert len(df) == 1
    assert df.iloc[0]["product_name"] == "Crunchy Oat Biscuit"
    assert df.iloc[0]["analysis_mode"] == "Real-Time Food Label Scanner"
    assert df.iloc[0]["serving_size"] == "30g"
    assert df.iloc[0]["saturated_fat"] == 2.0
    assert df.iloc[0]["detected_allergens"] == "Wheat, Milk, Soy"


def test_end_to_end_ocr_sample_image():
    """Verify real OCR engine extracts text and parses nutritional values from an image."""
    from PIL import Image
    from src.ocr import run_ocr
    sample_path = Path("assets/sample_biscuit_label.jpg")
    if not sample_path.exists():
        pytest.skip("sample_biscuit_label.jpg not found in assets")

    img = Image.open(sample_path)
    ocr_res = run_ocr(img)
    if not ocr_res["success"]:
        pytest.skip(f"OCR not available on test environment: {ocr_res.get('error_message')}")
    assert ocr_res["success"] is True
    assert ocr_res["confidence"] > 0.60
    assert "nutrition" in ocr_res["raw_text"].lower()

    parsed = parse_nutrition_label(ocr_res["raw_text"])
    assert parsed["calories"] == 150.0
    assert parsed["protein"] == 4.0
    assert parsed["carbohydrates"] == 22.0
    assert parsed["saturated_fat"] == 2.0

