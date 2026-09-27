"""
Nutrition Label OCR Parser and Structured Data Extractor.

Parses unstructured OCR text from packaged food nutrition labels,
applies controlled OCR error normalization, extracts nutritional metrics,
handles serving sizes, and provides optional per-serving / per-100g conversions.
"""

import re
from typing import Dict, Any, Optional, Tuple, List

# Controlled common OCR character substitutions for nutrition labels
OCR_NORMALIZATION_MAP = [
    (r"\bcal0ries\b", "calories"),
    (r"\bcalorles\b", "calories"),
    (r"\benerg[yv]\b", "energy"),
    (r"\bpr0tein\b", "protein"),
    (r"\bpr0teins\b", "proteins"),
    (r"\bcarbohydrat3\b", "carbohydrate"),
    (r"\bcarbohydra1es\b", "carbohydrates"),
    (r"\bcarbs\b", "carbohydrates"),
    (r"\bsuqar\b", "sugar"),
    (r"\bsug4r\b", "sugar"),
    (r"\bsuqars\b", "sugars"),
    (r"\bf1ber\b", "fiber"),
    (r"\bf1bre\b", "fibre"),
    (r"\b5odium\b", "sodium"),
    (r"\bs0dium\b", "sodium"),
    (r"\b5at\b", "fat"),
    (r"\bsaturat3d\b", "saturated"),
    (r"\bsat\.?\s*fat\b", "saturated fat"),
    (r"\btrans\.?\s*fat\b", "trans fat"),
    (r"\badd3d\s*sugar\b", "added sugar"),
    (r"\bserv1ng\b", "serving"),
]


def normalize_ocr_text(text: str) -> str:
    """
    Applies controlled typo corrections for typical OCR character confusion
    without altering arbitrary words.
    """
    if not text:
        return ""
    normalized = text
    for pattern, replacement in OCR_NORMALIZATION_MAP:
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
    return normalized


def _clean_numeric_string(num_str: str) -> Optional[float]:
    """
    Convert OCR string containing numbers into float.
    Handles 'O'/'o' for 0, 'l'/'I' for 1, and comma as decimal separator.
    """
    if not num_str:
        return None
    s = num_str.strip()
    s = re.sub(r"[Oo]", "0", s)
    s = re.sub(r"[lI|]", "1", s)
    s = s.replace(",", ".")

    # Match numeric floating point or integer pattern
    match = re.search(r"[-+]?\d*\.?\d+", s)
    if match:
        try:
            val = float(match.group(0))
            return val
        except ValueError:
            return None
    return None


def extract_field_value(
    text: str,
    field_keywords: List[str],
    unit_hints: List[str] = ["g", "mg", "kcal", "kj", "cal", "%"],
    exclude_phrases: Optional[List[str]] = None
) -> Optional[float]:
    """
    Locates line or pattern containing any field keyword and extracts
    the adjacent numeric value and optional unit.
    Ignores lines matching exclude_phrases (e.g., 'calories from fat').
    """
    lines = text.split("\n")
    unit_pattern = "|".join([re.escape(u) for u in unit_hints])
    excludes = [e.lower() for e in (exclude_phrases or [])]

    # Sort keywords by length descending so specific phrases like 'total fat' match before 'fat'
    sorted_keywords = sorted(field_keywords, key=len, reverse=True)

    for line in lines:
        line_clean = line.strip()
        line_lower = line_clean.lower()

        # Check if line contains forbidden phrases
        if any(exc in line_lower for exc in excludes):
            continue

        for kw in sorted_keywords:
            # Pattern: Keyword followed by separator then number and optional unit
            pattern = (
                rf"\b{re.escape(kw)}\b\s*[:\-\s]*"
                rf"([Oo\dlI|.,]+)\s*(?:{unit_pattern})?"
            )
            match = re.search(pattern, line_clean, re.IGNORECASE)
            if match:
                val = _clean_numeric_string(match.group(1))
                if val is not None:
                    return val

            # Also pattern where number comes directly after keyword with symbols
            pattern_rev = (
                rf"\b{re.escape(kw)}\b[^\d]*?(\d+[\.,]?\d*)\s*(?:{unit_pattern})?"
            )
            match_rev = re.search(pattern_rev, line_clean, re.IGNORECASE)
            if match_rev:
                val = _clean_numeric_string(match_rev.group(1))
                if val is not None:
                    return val

    # If not found per line, check multiline window (keyword on one line, number on next)
    for i in range(len(lines) - 1):
        line1 = lines[i].strip()
        line2 = lines[i + 1].strip()
        line1_lower = line1.lower()
        if any(exc in line1_lower for exc in excludes):
            continue
        for kw in sorted_keywords:
            if re.search(rf"\b{re.escape(kw)}\b", line1, re.IGNORECASE):
                val = _clean_numeric_string(line2)
                if val is not None:
                    return val

    return None


def extract_serving_size(text: str) -> Optional[str]:
    """
    Extract serving size description such as '30g', '1 pack (50g)', '2 biscuits (30g)'.
    """
    patterns = [
        r"(?:serving size|serving|per serving|portie|portion)\s*[:\-\s]*([^\n,;]+(?:\(\s*\d+\s*[a-zA-Z]+\s*\)|\d+\s*[a-zA-Z]+))",
        r"(?:serving size|serving)\s*[:\-\s]*([0-9]+\s*(?:g|ml|oz|piece|pieces|biscuit|biscuits))",
        r"(?:per\s+([0-9]+\s*(?:g|ml|oz)))",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            serving_text = match.group(1).strip()
            # Clean up trailing punctuation, balance parentheses
            serving_text = re.sub(r"[,;:]+$", "", serving_text).strip()
            if "(" in serving_text and ")" not in serving_text:
                serving_text = serving_text + ")"
            if len(serving_text) < 40:
                return serving_text

    # Search for standalone "100g" or "100 g"
    if re.search(r"\b100\s*g\b", text, re.IGNORECASE):
        return "100g"

    return None


def extract_product_name(text: str) -> Optional[str]:
    """
    Attempt to extract product name from the beginning of OCR text.
    Looks for the first substantial non-nutrition header line.
    """
    ignore_headers = [
        "nutrition", "facts", "information", "per", "serving", "ingredients",
        "energy", "calories", "table", "typical", "values", "amount", "daily",
        "value", "contains", "manufactured", "pack", "batch", "exp", "mfg"
    ]
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    for line in lines[:6]:  # Look in top 6 lines
        line_lower = line.lower()
        if any(h in line_lower for h in ignore_headers):
            continue
        if not re.search(r"[a-zA-Z]{3,}", line):
            continue
        if 3 <= len(line) <= 45:
            return line.title()

    return None


def parse_nutrition_label(raw_ocr_text: str) -> Dict[str, Any]:
    """
    Convert raw OCR label text into structured nutrition dictionary.

    Returns:
    {
        "product_name": str or None,
        "serving_size": str or None,
        "calories": float or None,
        "protein": float or None,
        "carbohydrates": float or None,
        "total_fat": float or None,
        "saturated_fat": float or None,
        "trans_fat": float or None,
        "fiber": float or None,
        "total_sugar": float or None,
        "added_sugar": float or None,
        "sodium": float or None,
        "normalized_text": str
    }
    Missing values are explicitly set to None ('Not detected').
    """
    normalized_text = normalize_ocr_text(raw_ocr_text)

    # 1. Product Name
    product_name = extract_product_name(normalized_text)

    # 2. Serving Size
    serving_size = extract_serving_size(normalized_text)

    # 3. Calories / Energy (kcal)
    calories = extract_field_value(
        normalized_text,
        ["calories", "energy", "energy value", "kcal"],
        unit_hints=["kcal", "cal"],
        exclude_phrases=["calories from fat", "energy from fat"]
    )
    if calories is None:
        kj_val = extract_field_value(normalized_text, ["kj"], unit_hints=["kj"])
        if kj_val is not None:
            calories = round(kj_val / 4.184, 1)

    # 4. Protein
    protein = extract_field_value(
        normalized_text,
        ["protein", "proteins", "total protein"],
        unit_hints=["g", "gm"]
    )

    # 5. Carbohydrates
    carbohydrates = extract_field_value(
        normalized_text,
        ["total carbohydrate", "total carbohydrates", "carbohydrate", "carbohydrates", "carbs"],
        unit_hints=["g", "gm"]
    )

    # 6. Total Fat (excludes 'calories from fat' and 'saturated fat')
    total_fat = extract_field_value(
        normalized_text,
        ["total fat", "fat", "lipids"],
        unit_hints=["g", "gm"],
        exclude_phrases=["calories from fat", "energy from fat", "saturated fat", "trans fat"]
    )

    # 7. Saturated Fat
    saturated_fat = extract_field_value(
        normalized_text,
        ["saturated fat", "saturated fats", "saturated", "sat fat"],
        unit_hints=["g", "gm"]
    )

    # 8. Trans Fat
    trans_fat = extract_field_value(
        normalized_text,
        ["trans fat", "trans fats", "transfat"],
        unit_hints=["g", "gm"]
    )

    # 9. Dietary Fiber
    fiber = extract_field_value(
        normalized_text,
        ["dietary fiber", "dietary fibre", "fiber", "fibre"],
        unit_hints=["g", "gm"]
    )

    # 10. Total Sugar
    total_sugar = extract_field_value(
        normalized_text,
        ["total sugars", "total sugar", "sugars", "sugar"],
        unit_hints=["g", "gm"],
        exclude_phrases=["includes added sugars", "added sugars"]
    )

    # 11. Added Sugar
    added_sugar = extract_field_value(
        normalized_text,
        ["includes added sugars", "added sugars", "added sugar"],
        unit_hints=["g", "gm"]
    )

    # 12. Sodium / Salt (mg or g)
    sodium = extract_field_value(
        normalized_text,
        ["sodium"],
        unit_hints=["mg", "g"]
    )
    if sodium is None:
        salt_val = extract_field_value(normalized_text, ["salt"], unit_hints=["g", "mg"])
        if salt_val is not None:
            if salt_val < 25.0:  # In grams
                sodium = round(salt_val * 400.0, 1)
            else:  # In mg
                sodium = round(salt_val * 0.4, 1)

    return {
        "product_name": product_name,
        "serving_size": serving_size,
        "calories": calories,
        "protein": protein,
        "carbohydrates": carbohydrates,
        "total_fat": total_fat,
        "saturated_fat": saturated_fat,
        "trans_fat": trans_fat,
        "fiber": fiber,
        "total_sugar": total_sugar,
        "added_sugar": added_sugar,
        "sodium": sodium,
        "normalized_text": normalized_text
    }


def compute_per_100g_conversion(
    nutrition_dict: Dict[str, Any]
) -> Tuple[Optional[Dict[str, float]], Optional[str]]:
    """
    If serving size is known (e.g., '30g', '50 g'), compute per-100g values.
    Returns (per_100g_dict, serving_g_or_reason).
    """
    serving_str = nutrition_dict.get("serving_size")
    if not serving_str:
        return None, "Serving size not specified"

    # Match grams in serving size string
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:g|gm|grams)\b", str(serving_str), re.IGNORECASE)
    if not match:
        return None, "Serving size does not have clear gram unit"

    try:
        serving_grams = float(match.group(1))
        if serving_grams <= 0 or serving_grams > 1000:
            return None, "Invalid serving weight"

        factor = 100.0 / serving_grams
        per_100g: Dict[str, float] = {}

        keys_to_convert = [
            "calories", "protein", "carbohydrates", "total_fat",
            "saturated_fat", "trans_fat", "fiber", "total_sugar",
            "added_sugar", "sodium"
        ]

        for k in keys_to_convert:
            val = nutrition_dict.get(k)
            if val is not None:
                per_100g[k] = round(float(val) * factor, 1)
            else:
                per_100g[k] = None

        return per_100g, f"Converted from {serving_grams}g serving ({factor:.2f}x multiplier)"

    except Exception as exc:
        return None, f"Conversion error: {str(exc)}"
