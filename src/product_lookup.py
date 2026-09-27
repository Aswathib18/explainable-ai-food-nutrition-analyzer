"""
Optional Open Food Facts Online Product Lookup Module.

Provides non-blocking, optional reference lookup from Open Food Facts.
The primary mode of this application remains local OCR and manual verification;
this module gracefully handles timeouts and network absence without blocking execution.
"""

import urllib.request
import urllib.parse
import json
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

OPEN_FOOD_FACTS_SEARCH_URL = "https://world.openfoodfacts.org/cgi/search.pl"


def lookup_open_food_facts(
    query: str,
    timeout_seconds: float = 3.0
) -> Optional[Dict[str, Any]]:
    """
    Search Open Food Facts database by product name or barcode.
    Strictly optional and non-blocking with 3.0s timeout.
    Returns None if offline, timed out, or not found.
    """
    if not query or len(query.strip()) < 2:
        return None

    try:
        params = {
            "search_terms": query.strip(),
            "search_simple": 1,
            "action": "process",
            "json": 1,
            "page_size": 1
        }
        url = f"{OPEN_FOOD_FACTS_SEARCH_URL}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "ExplainableAINutritionScanner/1.0 (academic-project)"}
        )

        with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
            if response.status != 200:
                return None
            data = json.loads(response.read().decode("utf-8"))

        products = data.get("products", [])
        if not products:
            return None

        p = products[0]
        nutriments = p.get("nutriments", {})

        return {
            "source": "Online product database (Open Food Facts)",
            "product_name": p.get("product_name") or query,
            "brands": p.get("brands", "Unknown Brand"),
            "serving_size": p.get("serving_size"),
            "calories": nutriments.get("energy-kcal_100g") or nutriments.get("energy-kcal"),
            "protein": nutriments.get("proteins_100g") or nutriments.get("proteins"),
            "carbohydrates": nutriments.get("carbohydrates_100g") or nutriments.get("carbohydrates"),
            "total_fat": nutriments.get("fat_100g") or nutriments.get("fat"),
            "saturated_fat": nutriments.get("saturated-fat_100g") or nutriments.get("saturated-fat"),
            "trans_fat": nutriments.get("trans-fat_100g") or nutriments.get("trans-fat"),
            "fiber": nutriments.get("fiber_100g") or nutriments.get("fiber"),
            "total_sugar": nutriments.get("sugars_100g") or nutriments.get("sugars"),
            "added_sugar": nutriments.get("added-sugars_100g") or nutriments.get("added-sugars"),
            "sodium": (nutriments.get("sodium_100g") * 1000) if nutriments.get("sodium_100g") is not None else None,
            "ingredients_text": p.get("ingredients_text_en") or p.get("ingredients_text")
        }

    except Exception as exc:
        logger.debug(f"Open Food Facts lookup unavailable ({exc}). Continuing in offline mode.")
        return None
