"""
Ingredient List Parsing and Categorization Module.

Extracts ingredient sections from raw OCR text, isolates individual ingredients
(handling nested parenthetical sub-ingredients), normalizes OCR artifacts,
and tags ingredients into neutral food science categories.
"""

import re
from typing import Dict, List, Any, Tuple, Optional

# Food Science Category Mapping for Neutral Ingredient Classification
CATEGORY_KEYWORDS: Dict[str, Tuple[str, List[str]]] = {
    "Grains & Cereals": (
        "",
        ["flour", "wheat", "oat", "oats", "rice", "corn", "maize", "barley",
         "rye", "malt", "starch", "semolina", "atta", "maida", "grain", "cereal"]
    ),
    "Sugars & Sweeteners": (
        "",
        ["sugar", "glucose", "fructose", "sucrose", "dextrose", "syrup",
         "corn syrup", "honey", "invert sugar", "molasses", "maltodextrin",
         "caramel", "sweetener", "maltitol", "sorbitol", "stevia"]
    ),
    "Oils & Fats": (
        "",
        ["oil", "fat", "shortening", "margarine", "butter", "cocoa butter",
         "palm oil", "sunflower oil", "canola oil", "vegetable oil", "ghee",
         "lard", "tallow", "olein"]
    ),
    "Dairy & Milk Derivatives": (
        "",
        ["milk", "whey", "casein", "caseinate", "lactose", "cheese",
         "cream", "curd", "dairy", "yogurt"]
    ),
    "Soy Components": (
        "",
        ["soy", "soya", "soybean", "edamame", "tofu"]
    ),
    "Nuts & Legumes": (
        "",
        ["peanut", "almond", "cashew", "walnut", "hazelnut", "pistachio",
         "pecan", "bean", "lentil", "chickpea"]
    ),
    "Seeds": (
        "",
        ["sesame", "chia", "flax", "flaxseed", "sunflower seed", "pumpkin seed"]
    ),
    "Salt & Leavening Agents": (
        "",
        ["salt", "sodium chloride", "baking powder", "baking soda",
         "sodium bicarbonate", "ammonium bicarbonate", "leavening"]
    ),
    "Emulsifiers & Stabilizers": (
        "",
        ["lecithin", "emulsifier", "e322", "e471", "e472", "mono- and diglycerides",
         "polysorbate", "xanthan", "guar gum", "carrageenan", "pectin"]
    ),
    "Preservatives & Antioxidants": (
        "",
        ["preservative", "sodium benzoate", "potassium sorbate", "citric acid",
         "ascorbic acid", "tocopherol", "bht", "bha", "e202", "e211", "sulfite"]
    ),
    "Flavouring & Spices": (
        "",
        ["flavor", "flavour", "flavoring", "flavouring", "vanilla", "vanillin",
         "cocoa", "cacao", "spice", "cinnamon", "cardamom", "pepper", "extract"]
    )
}


def extract_ingredient_text_block(raw_text: str) -> str:
    """
    Extract the ingredient section from the raw OCR text.
    Looks for headers like 'INGREDIENTS:', 'Ingredients', 'Contains:'
    """
    lines = raw_text.split("\n")
    start_collecting = False
    collected_lines = []

    header_pattern = re.compile(
        r"^(?:ingredients|ingrédients|ingrediente|ingredients\s*[:\-])",
        re.IGNORECASE
    )

    stop_pattern = re.compile(
        r"^(?:nutrition facts|nutrition information|best before|mfg|mfd|batch no|marketed by|manufactured by)",
        re.IGNORECASE
    )

    for line in lines:
        line_strip = line.strip()
        if not line_strip:
            continue

        if header_pattern.search(line_strip):
            start_collecting = True
            # Remove the header word itself from the line
            cleaned_first_line = header_pattern.sub("", line_strip).strip(" :.-")
            if cleaned_first_line:
                collected_lines.append(cleaned_first_line)
            continue

        if start_collecting:
            if stop_pattern.search(line_strip):
                break
            collected_lines.append(line_strip)

    if collected_lines:
        return " ".join(collected_lines)

    # Fallback: if no explicit header found, search for line containing 'ingredients:' anywhere
    match = re.search(r"ingredients\s*[:\-]\s*([^\n\r]+)", raw_text, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # If completely absent, return the whole raw text
    return raw_text.strip()


def split_into_ingredients(text: str) -> List[str]:
    """
    Splits text on commas and semicolons, while preserving parenthetical clauses
    such as 'Enriched Flour (Wheat Flour, Niacin, Iron, Thiamin)'.
    """
    ingredients: List[str] = []
    current_item: List[str] = []
    paren_depth = 0

    for char in text:
        if char in "([":
            paren_depth += 1
            current_item.append(char)
        elif char in ")]":
            if paren_depth > 0:
                paren_depth -= 1
            current_item.append(char)
        elif char in ",;" and paren_depth == 0:
            item_str = "".join(current_item).strip()
            if item_str:
                ingredients.append(item_str)
            current_item = []
        else:
            current_item.append(char)

    if current_item:
        last_str = "".join(current_item).strip()
        if last_str:
            ingredients.append(last_str)

    # Clean items (strip trailing dots, excessive symbols)
    cleaned = []
    for item in ingredients:
        clean = re.sub(r"^[^\w]+|[^\w\)]+$", "", item).strip()
        if len(clean) >= 2 and not clean.isdigit():
            cleaned.append(clean)

    return cleaned


def categorize_ingredient(ingredient: str) -> Dict[str, str]:
    """
    Classify an individual ingredient into a neutral food category.
    """
    ing_lower = ingredient.lower()
    for cat_name, (icon, keywords) in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            # Word boundary search
            if re.search(rf"\b{re.escape(kw)}\b", ing_lower):
                return {
                    "category": cat_name,
                    "icon": icon,
                    "ingredient": ingredient
                }

    return {
        "category": "Other Ingredients",
        "icon": "",
        "ingredient": ingredient
    }


def parse_ingredients_pipeline(raw_text: str) -> Dict[str, Any]:
    """
    Full pipeline to parse, normalize, and categorize ingredients.
    """
    extracted_block = extract_ingredient_text_block(raw_text)
    ingredient_list = split_into_ingredients(extracted_block)

    categorized_items = []
    category_counts: Dict[str, int] = {}

    for ing in ingredient_list:
        info = categorize_ingredient(ing)
        categorized_items.append(info)
        cat = info["category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1

    return {
        "raw_extracted_block": extracted_block,
        "ingredients": ingredient_list,
        "categorized_items": categorized_items,
        "category_counts": category_counts,
        "total_ingredients_count": len(ingredient_list)
    }
