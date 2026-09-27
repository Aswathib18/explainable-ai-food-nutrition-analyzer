"""
Allergen Detection and Precautionary Cross-Contact Warning System.

Detects common food allergens and cross-contact / may-contain statements
using controlled, word-boundary regex matching.
Extracts exact detected terms and strictly distinguishes between:
1. Direct Ingredients
2. Precautionary / Cross-Contact Warnings ("May contain...", "Manufactured in...")
"""

import re
from typing import Dict, List, Any, Tuple, Optional

# ==============================================================================
# CONFIGURABLE ALLERGEN DICTIONARY
# Keys: Canonical Allergen Category
# Values: List of specific matched terms / ingredients to look for
# ==============================================================================
ALLERGEN_TAXONOMY: Dict[str, Dict[str, Any]] = {
    "Milk / Dairy": {
        "icon": "🥛",
        "terms": [
            "milk", "milk powder", "milk solids", "skimmed milk", "whole milk",
            "skim milk", "milk fat", "dairy", "whey", "whey protein",
            "casein", "caseinate", "caseinates", "sodium caseinate",
            "calcium caseinate", "lactose", "butter", "buttermilk", "cheese",
            "cream", "sour cream", "yogurt", "yoghurt", "curd", "ghee",
            "milk protein", "condensed milk", "milk derivative"
        ]
    },
    "Soy": {
        "icon": "🌱",
        "terms": [
            "soy", "soya", "soybean", "soybeans", "soy protein", "soy lecithin",
            "soya lecithin", "soy flour", "soya flour", "soy sauce", "tofu",
            "edamame", "textured vegetable protein", "isolated soy protein",
            "hydrolyzed soy protein", "soy oil"
        ]
    },
    "Wheat & Gluten": {
        "icon": "🌾",
        "terms": [
            "wheat", "wheat flour", "refined wheat flour", "whole wheat",
            "wheat protein", "wheat germ", "wheat gluten", "gluten",
            "maida", "atta", "semolina", "durum", "spelt", "barley",
            "rye", "oats", "malt", "malt extract", "triticale"
        ]
    },
    "Peanuts": {
        "icon": "🥜",
        "terms": [
            "peanut", "peanuts", "peanut flour", "peanut oil", "peanut butter",
            "groundnut", "groundnuts", "arachis oil"
        ]
    },
    "Tree Nuts": {
        "icon": "🌰",
        "terms": [
            "tree nut", "tree nuts", "nuts",
            "almond", "almonds", "cashew", "cashews", "walnut", "walnuts",
            "hazelnut", "hazelnuts", "pistachio", "pistachios", "pecan", "pecans",
            "macadamia", "macadamia nuts", "brazil nut", "brazil nuts",
            "pine nut", "pine nuts", "chestnut", "chestnuts"
        ]
    },
    "Eggs": {
        "icon": "🥚",
        "terms": [
            "egg", "eggs", "egg powder", "egg white", "egg yolk", "albumin",
            "ovalbumin", "ovomucin", "ovomucoid", "lysozyme", "meringue"
        ]
    },
    "Fish": {
        "icon": "🐟",
        "terms": [
            "fish", "salmon", "tuna", "cod", "anchovy", "anchovies",
            "fish gelatin", "fish oil", "fish sauce", "tilapia", "haddock",
            "mackerel", "sardine", "sardines"
        ]
    },
    "Crustacean & Shellfish": {
        "icon": "🦐",
        "terms": [
            "shellfish", "crustacean", "crustaceans", "shrimp", "shrimps",
            "prawn", "prawns", "crab", "crabs", "lobster", "lobsters",
            "crayfish", "clam", "clams", "mussel", "mussels", "oyster",
            "oysters", "scallop", "scallops", "squid"
        ]
    },
    "Sesame": {
        "icon": "🥯",
        "terms": [
            "sesame", "sesame seed", "sesame seeds", "sesame oil",
            "sesame paste", "tahini", "til"
        ]
    },
    "Mustard": {
        "icon": "🌿",
        "terms": [
            "mustard", "mustard seed", "mustard seeds", "mustard flour",
            "mustard oil", "dijon mustard"
        ]
    },
    "Sulphites": {
        "icon": "🍷",
        "terms": [
            "sulphite", "sulphites", "sulfite", "sulfites", "sulfur dioxide",
            "sulphur dioxide", "sodium metabisulfite", "sodium metabisulphite",
            "potassium metabisulfite", "sodium sulfite", "e220", "e221",
            "e222", "e223", "e224", "e228"
        ]
    }
}

# Precautionary Cross-Contact Statement Regex Patterns
PRECAUTIONARY_PATTERNS = [
    r"(?:may contain(?:\s+traces of)?\s*[:\s]*)([^.]+)",
    r"(?:may contain\s+)([^.]+)",
    r"(?:contains traces of\s+)([^.]+)",
    r"(?:manufactured in a facility that(?: also)? processes\s+)([^.]+)",
    r"(?:made in a facility that(?: also)? handles\s+)([^.]+)",
    r"(?:processed on equipment that(?: also)? processes\s+)([^.]+)",
    r"(?:produced in an environment where\s+)([^.]+)",
    r"(?:packed in a plant that handles\s+)([^.]+)",
    r"(?:facility handles\s+)([^.]+)"
]

ALLERGEN_SAFETY_DISCLAIMER = (
    "Allergen detection is based on the text successfully extracted from the package label. "
    "It may miss allergens because of OCR errors, incomplete labels, cross-contact statements, "
    "or ingredients not recognized by the system. Always verify the original packaging and "
    "manufacturer allergen information. This system does not provide medical advice or guarantees."
)


def split_into_precautionary_and_ingredients(raw_text: str) -> Tuple[str, str]:
    """
    Split the OCR text into main ingredient text and precautionary / may-contain text.
    Removes precautionary clauses from main ingredients so they are not misclassified.
    """
    precautionary_snippets: List[str] = []
    cleaned_ingredients = raw_text

    for pattern in PRECAUTIONARY_PATTERNS:
        matches = list(re.finditer(pattern, cleaned_ingredients, re.IGNORECASE))
        for m in matches:
            full_match = m.group(0).strip()
            if full_match:
                precautionary_snippets.append(full_match)
        # Remove precautionary text from ingredient stream
        cleaned_ingredients = re.sub(pattern, " ", cleaned_ingredients, flags=re.IGNORECASE)

    # Combine precautionary text
    precautionary_text = " . ".join(precautionary_snippets)
    return cleaned_ingredients, precautionary_text


def _match_terms_in_text(text: str, terms: List[str]) -> List[str]:
    """
    Perform controlled word-boundary matching to find exact allergen terms.
    Sort terms by length descending to match longest specific term first.
    Replaces matched spans to prevent shorter substring duplication.
    """
    detected_terms = []
    sorted_terms = sorted(terms, key=len, reverse=True)
    text_lower = f" {text.lower()} "

    # Normalize punctuation into spaces for clean token boundaries
    normalized_text = re.sub(r"[^\w\s-]", " ", text_lower)

    for term in sorted_terms:
        # Regex using word boundaries
        pattern = rf"\b{re.escape(term.lower())}\b"
        if re.search(pattern, normalized_text):
            detected_terms.append(term)
            # Mask out matched term so generic sub-terms don't double match
            normalized_text = re.sub(pattern, " [MATCHED] ", normalized_text)

    return detected_terms



def detect_allergens(
    ingredient_text: str,
    precautionary_text: Optional[str] = None
) -> Dict[str, Any]:
    """
    Detect allergens and cross-contact warnings from package label text.

    Distinguishes:
    - Direct ingredients (Source: 'Ingredient')
    - Precautionary warnings (Source: 'Precautionary')

    Returns:
        {
            "has_allergens": bool,
            "detected_list": [...],
            "allergen_table": [...],
            "summary_icons": [...],
            "precautionary_statements": [...],
            "disclaimer": str
        }
    """
    if precautionary_text is None:
        ing_clean, prec_clean = split_into_precautionary_and_ingredients(ingredient_text)
    else:
        ing_clean = ingredient_text
        prec_clean = precautionary_text

    allergen_table: List[Dict[str, str]] = []
    seen_combinations = set()
    summary_icons = []

    # 1. Search in Direct Ingredients
    for allergen_cat, data in ALLERGEN_TAXONOMY.items():
        matched_terms = _match_terms_in_text(ing_clean, data["terms"])
        for term in matched_terms:
            combo_key = (allergen_cat, term, "Ingredient")
            if combo_key not in seen_combinations:
                seen_combinations.add(combo_key)
                allergen_table.append({
                    "allergen": allergen_cat,
                    "detected_term": term.title(),
                    "source": "Ingredient",
                    "icon": data["icon"]
                })
                if data["icon"] not in summary_icons:
                    summary_icons.append(data["icon"])

    # 2. Search in Precautionary statements
    if prec_clean:
        for allergen_cat, data in ALLERGEN_TAXONOMY.items():
            matched_terms = _match_terms_in_text(prec_clean, data["terms"])
            for term in matched_terms:
                combo_key = (allergen_cat, term, "Precautionary")
                if combo_key not in seen_combinations:
                    seen_combinations.add(combo_key)
                    allergen_table.append({
                        "allergen": allergen_cat,
                        "detected_term": term.title(),
                        "source": "Precautionary",
                        "icon": data["icon"]
                    })
                    if data["icon"] not in summary_icons:
                        summary_icons.append(data["icon"])

    # Extract distinct precautionary clauses for user visibility
    precautionary_clauses = []
    if prec_clean:
        for p in prec_clean.split("."):
            p_strip = p.strip()
            if p_strip and len(p_strip) > 5 and p_strip not in precautionary_clauses:
                precautionary_clauses.append(p_strip)

    has_allergens = len(allergen_table) > 0

    return {
        "has_allergens": has_allergens,
        "allergen_table": allergen_table,
        "summary_icons": summary_icons,
        "precautionary_statements": precautionary_clauses,
        "disclaimer": ALLERGEN_SAFETY_DISCLAIMER
    }
