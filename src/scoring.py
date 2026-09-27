"""
Transparent Nutrition Scoring Algorithm.
Calculates an interpretable 0–100 Reference Nutrition Score based on documented
macronutrient thresholds, dietary fiber density, protein quality, sugar content,
and caloric load, extended to packaged food metrics (saturated fat, trans fat, added sugar, sodium).
"""

from typing import Dict, Any, List, Tuple, Optional
import config


def calculate_nutrition_score(nutrition_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate a transparent 0-100 Reference Nutrition Score.

    Methodology:
    - Baseline Score: Starts at 50.0 (Neutral benchmark).
    - Positive Contributions:
      * Dietary Fiber: Essential for glycemic control, digestive health, and microbiome (+ up to 18 pts).
      * Protein: Essential amino acids, tissue repair, and satiety (+ up to 14 pts).
      * Macro Balance Bonus: Awarded when protein, fiber, and low sugar coexist (+ up to 6 pts).
    - Negative Deductions:
      * Excess Total Sugar: Penalty applied when sugar exceeds 5g per serving (- up to 22 pts).
      * Added Sugar: Additional penalty when specifically present on packaged food label (- up to 15 pts).
      * High Total Fat: Penalty applied when fat exceeds 3g per serving (- up to 18 pts).
      * Saturated Fat: Penalty for saturated fat exceeding 1.5g per serving (- up to 12 pts).
      * Trans Fat: Severe penalty for artificial trans fats > 0g (- up to 10 pts).
      * Sodium: Penalty for elevated sodium exceeding 140 mg per serving (- up to 12 pts).
      * Calorie Density: Penalty for calorie load exceeding 250 kcal/serving (- up to 15 pts).
    - Boundary Guarantee: Clamped strictly within [0.0, 100.0].

    Returns a detailed breakdown dictionary with points, factor details, and health tier.
    """
    calories = float(nutrition_info.get("calories") or 0.0)
    protein = float(nutrition_info.get("protein") or 0.0)
    carbs = float(nutrition_info.get("carbohydrates") or 0.0)
    fat = float(nutrition_info.get("fat") or nutrition_info.get("total_fat") or 0.0)
    fiber = float(nutrition_info.get("fiber") or 0.0)
    sugar = float(nutrition_info.get("sugar") or nutrition_info.get("total_sugar") or 0.0)

    # Packaged food specific fields (may be None if from basic whole food or not detected)
    sat_fat = nutrition_info.get("saturated_fat")
    trans_fat = nutrition_info.get("trans_fat")
    added_sugar = nutrition_info.get("added_sugar")
    sodium = nutrition_info.get("sodium")

    baseline = config.SCORING_CONFIG["baseline_score"]
    factors: List[Dict[str, Any]] = []

    # 1. Dietary Fiber Contribution (+ up to 18 points)
    fiber_points = min(18.0, fiber * config.SCORING_CONFIG["weights"]["fiber_bonus"])
    if fiber_points > 0:
        factors.append({
            "feature": "Dietary Fiber",
            "impact": round(fiber_points, 1),
            "status": "positive",
            "reason": f"High fiber ({fiber}g) promotes digestive health, satiety, and blood sugar regulation.",
            "value": f"{fiber} g"
        })

    # 2. Protein Contribution (+ up to 14 points)
    protein_points = min(14.0, protein * config.SCORING_CONFIG["weights"]["protein_bonus"])
    if protein_points > 0:
        factors.append({
            "feature": "Protein Content",
            "impact": round(protein_points, 1),
            "status": "positive",
            "reason": f"Provides essential amino acids ({protein}g) supporting lean muscle and metabolism.",
            "value": f"{protein} g"
        })

    # 3. Macro Balance Bonus (+ up to 6 points)
    if protein >= 3.0 and fiber >= 2.0 and sugar <= 10.0:
        balance_pts = config.SCORING_CONFIG["weights"]["nutrient_balance_bonus"]
        factors.append({
            "feature": "Nutrient Balance",
            "impact": round(balance_pts, 1),
            "status": "positive",
            "reason": "Harmonious balance of protein, dietary fiber, and controlled sugar levels.",
            "value": "Balanced"
        })
    else:
        balance_pts = 0.0

    # 4. Total Sugar Deduction (- up to 22 points)
    if sugar > 5.0:
        sugar_penalty = min(22.0, (sugar - 5.0) * config.SCORING_CONFIG["weights"]["sugar_penalty"])
        factors.append({
            "feature": "Sugar Content",
            "impact": round(-sugar_penalty, 1),
            "status": "negative",
            "reason": f"Elevated sugar ({sugar}g) can trigger rapid insulin spikes and surplus caloric load.",
            "value": f"{sugar} g"
        })
    else:
        sugar_penalty = 0.0
        if sugar <= 5.0 and sugar > 0:
            factors.append({
                "feature": "Low Sugar",
                "impact": 0.0,
                "status": "neutral",
                "reason": f"Low sugar content ({sugar}g) supports glycemic stability.",
                "value": f"{sugar} g"
            })

    # 5. Packaged Food: Added Sugar Deduction (- up to 15 points)
    added_sugar_penalty = 0.0
    if added_sugar is not None and float(added_sugar) > 0.0:
        as_val = float(added_sugar)
        added_sugar_penalty = min(15.0, as_val * 1.0)
        factors.append({
            "feature": "Added Sugar",
            "impact": round(-added_sugar_penalty, 1),
            "status": "negative",
            "reason": f"Contains {as_val}g refined added sugars beyond natural food matrix.",
            "value": f"{as_val} g"
        })

    # 6. Total Fat Deduction (- up to 18 points)
    if fat > 3.0:
        fat_penalty = min(18.0, (fat - 3.0) * config.SCORING_CONFIG["weights"]["fat_penalty"])
        factors.append({
            "feature": "Total Fat",
            "impact": round(-fat_penalty, 1),
            "status": "negative",
            "reason": f"Total fat ({fat}g) contributes high energy density requiring portion moderation.",
            "value": f"{fat} g"
        })
    else:
        fat_penalty = 0.0

    # 7. Packaged Food: Saturated Fat Deduction (- up to 12 points)
    sat_fat_penalty = 0.0
    if sat_fat is not None and float(sat_fat) > 1.5:
        sf_val = float(sat_fat)
        sat_fat_penalty = min(12.0, (sf_val - 1.5) * 1.5)
        factors.append({
            "feature": "Saturated Fat",
            "impact": round(-sat_fat_penalty, 1),
            "status": "negative",
            "reason": f"Saturated fat ({sf_val}g) exceeds low-fat benchmark (1.5g/serving).",
            "value": f"{sf_val} g"
        })

    # 8. Packaged Food: Trans Fat Deduction (- up to 10 points)
    trans_fat_penalty = 0.0
    if trans_fat is not None and float(trans_fat) > 0.0:
        tf_val = float(trans_fat)
        trans_fat_penalty = min(10.0, tf_val * 5.0)
        factors.append({
            "feature": "Trans Fat",
            "impact": round(-trans_fat_penalty, 1),
            "status": "negative",
            "reason": f"Contains industrial trans fat ({tf_val}g); WHO recommends minimizing intake.",
            "value": f"{tf_val} g"
        })

    # 9. Packaged Food: Sodium Deduction (- up to 12 points)
    sodium_penalty = 0.0
    if sodium is not None and float(sodium) > 140.0:
        sod_val = float(sodium)
        sodium_penalty = min(12.0, (sod_val - 140.0) * 0.02)
        factors.append({
            "feature": "Sodium Content",
            "impact": round(-sodium_penalty, 1),
            "status": "negative",
            "reason": f"Sodium ({sod_val}mg) exceeds low-sodium benchmark (140mg/serving).",
            "value": f"{sod_val} mg"
        })

    # 10. Calorie Density Deduction (- up to 15 points)
    if calories > 250.0:
        calorie_penalty = min(
            15.0,
            (calories - 250.0) * config.SCORING_CONFIG["weights"]["calorie_density_penalty"]
        )
        factors.append({
            "feature": "Calorie Density",
            "impact": round(-calorie_penalty, 1),
            "status": "negative",
            "reason": f"High energy per serving ({calories} kcal) relative to reference intake.",
            "value": f"{calories} kcal"
        })
    else:
        calorie_penalty = 0.0

    # Aggregate Total Score
    positive_total = fiber_points + protein_points + balance_pts
    negative_total = (
        sugar_penalty
        + added_sugar_penalty
        + fat_penalty
        + sat_fat_penalty
        + trans_fat_penalty
        + sodium_penalty
        + calorie_penalty
    )

    raw_score = baseline + positive_total - negative_total

    # Strictly clamp between 0.0 and 100.0
    final_score = max(
        config.SCORING_CONFIG["caps"]["min_score"],
        min(config.SCORING_CONFIG["caps"]["max_score"], raw_score)
    )
    final_score = round(final_score, 1)

    # Determine Health Tier
    health_tier = "Moderate / Balanced"
    for tier_name, (low, high) in config.HEALTH_TIERS.items():
        if low <= final_score <= high:
            health_tier = tier_name
            break

    return {
        "score": final_score,
        "baseline": baseline,
        "health_tier": health_tier,
        "factors": factors,
        "positive_impact_total": round(positive_total, 1),
        "negative_impact_total": round(negative_total, 1)
    }
