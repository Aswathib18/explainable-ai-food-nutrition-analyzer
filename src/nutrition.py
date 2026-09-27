"""
Nutrition data access and food matching module.
Handles loading datasets, alias resolution, fuzzy matching, and nutritional metric retrieval.
"""

import re
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd
import logging

import config

logger = logging.getLogger(__name__)


def normalize_food_name(name: str) -> str:
    """Normalize input text by lowercasing, stripping punctuation, and extra whitespace."""
    if not isinstance(name, str):
        return ""
    text = name.lower()
    # Replace non-alphanumeric characters with spaces
    text = re.sub(r"[^\w\s]", " ", text)
    # Collapse multiple spaces into single space
    text = re.sub(r"\s+", " ", text).strip()
    return text


class NutritionDatabase:
    """
    In-memory representation of the nutrition dataset and alias lookup dictionary.
    """

    def __init__(
        self,
        nutrition_path: Path = config.NUTRITION_DATA_PATH,
        aliases_path: Path = config.FOOD_ALIASES_PATH
    ):
        self.nutrition_path = nutrition_path
        self.aliases_path = aliases_path
        self.df_nutrition: pd.DataFrame = pd.DataFrame()
        self.alias_map: Dict[str, str] = {}
        self._load_data()

    def _load_data(self) -> None:
        """Load CSV datasets into memory and index food aliases."""
        try:
            if self.nutrition_path.exists():
                self.df_nutrition = pd.read_csv(self.nutrition_path)
                # Ensure correct numeric types
                numeric_cols = ["calories", "protein", "carbohydrates", "fat", "fiber", "sugar", "serving_size"]
                for col in numeric_cols:
                    if col in self.df_nutrition.columns:
                        self.df_nutrition[col] = pd.to_numeric(self.df_nutrition[col], errors="coerce").fillna(0.0)
            else:
                logger.error(f"Nutrition dataset not found at {self.nutrition_path}")

            if self.aliases_path.exists():
                df_aliases = pd.read_csv(self.aliases_path)
                for _, row in df_aliases.iterrows():
                    alias = normalize_food_name(str(row["alias"]))
                    canonical = str(row["canonical_food"]).strip()
                    if alias and canonical:
                        self.alias_map[alias] = canonical
        except Exception as exc:
            logger.error(f"Error loading nutrition database: {exc}")

    def get_all_food_names(self) -> List[str]:
        """Return sorted list of all available canonical food names."""
        if not self.df_nutrition.empty and "food" in self.df_nutrition.columns:
            return sorted(self.df_nutrition["food"].dropna().unique().tolist())
        return []

    def get_all_categories(self) -> List[str]:
        """Return list of distinct food categories."""
        if not self.df_nutrition.empty and "category" in self.df_nutrition.columns:
            return sorted(self.df_nutrition["category"].dropna().unique().tolist())
        return []

    def resolve_food_name(self, raw_name: str) -> Optional[str]:
        """
        Find canonical food name using exact match, alias mapping, or word token matching.
        """
        if not raw_name or self.df_nutrition.empty:
            return None

        clean_query = normalize_food_name(raw_name)

        # 1. Direct canonical case-insensitive match
        for food in self.df_nutrition["food"]:
            if normalize_food_name(food) == clean_query:
                return food

        # 2. Check alias mapping
        if clean_query in self.alias_map:
            canonical = self.alias_map[clean_query]
            if not self.df_nutrition[self.df_nutrition["food"] == canonical].empty:
                return canonical

        # 3. Substring matching in aliases
        for alias, canonical in self.alias_map.items():
            if alias in clean_query or clean_query in alias:
                if not self.df_nutrition[self.df_nutrition["food"] == canonical].empty:
                    return canonical

        # 4. Token overlap matching with canonical names
        query_tokens = set(clean_query.split())
        best_match = None
        max_overlap = 0

        for food in self.df_nutrition["food"]:
            food_tokens = set(normalize_food_name(food).split())
            overlap = len(query_tokens.intersection(food_tokens))
            if overlap > max_overlap:
                max_overlap = overlap
                best_match = food

        if max_overlap > 0:
            return best_match

        return None

    def get_nutrition_info(self, food_name: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve nutritional data for a food item and compute Daily Value percentages (%DV).
        Returns None if food is not available in the database.
        """
        canonical_name = self.resolve_food_name(food_name)
        if not canonical_name or self.df_nutrition.empty:
            return None

        matches = self.df_nutrition[self.df_nutrition["food"] == canonical_name]
        if matches.empty:
            return None

        row = matches.iloc[0].to_dict()

        # Compute % Daily Values (DV) based on FDA adult 2,000 calorie reference
        dv_reference = config.DAILY_REFERENCE_VALUES
        pct_dv = {
            "calories_pct": round((float(row.get("calories", 0.0)) / dv_reference["calories"]) * 100, 1),
            "protein_pct": round((float(row.get("protein", 0.0)) / dv_reference["protein"]) * 100, 1),
            "carbs_pct": round((float(row.get("carbohydrates", 0.0)) / dv_reference["carbohydrates"]) * 100, 1),
            "fat_pct": round((float(row.get("fat", 0.0)) / dv_reference["fat"]) * 100, 1),
            "fiber_pct": round((float(row.get("fiber", 0.0)) / dv_reference["fiber"]) * 100, 1),
            "sugar_pct": round((float(row.get("sugar", 0.0)) / dv_reference["sugar"]) * 100, 1),
        }

        # Combine with raw nutritional metrics
        nutrition_result = {
            "food": row.get("food", canonical_name),
            "calories": float(row.get("calories", 0.0)),
            "protein": float(row.get("protein", 0.0)),
            "carbohydrates": float(row.get("carbohydrates", 0.0)),
            "fat": float(row.get("fat", 0.0)),
            "fiber": float(row.get("fiber", 0.0)),
            "sugar": float(row.get("sugar", 0.0)),
            "serving_size": float(row.get("serving_size", 100.0)),
            "serving_unit": str(row.get("serving_unit", "g")),
            "category": str(row.get("category", "General")),
            "daily_value_percentages": pct_dv
        }

        return nutrition_result

    def get_category_averages(self, category: str) -> Optional[Dict[str, float]]:
        """Return average nutrient values for foods in a specified category."""
        if self.df_nutrition.empty:
            return None
        cat_df = self.df_nutrition[self.df_nutrition["category"] == category]
        if cat_df.empty:
            return None
        return {
            "calories": round(float(cat_df["calories"].mean()), 1),
            "protein": round(float(cat_df["protein"].mean()), 1),
            "carbohydrates": round(float(cat_df["carbohydrates"].mean()), 1),
            "fat": round(float(cat_df["fat"].mean()), 1),
            "fiber": round(float(cat_df["fiber"].mean()), 1),
            "sugar": round(float(cat_df["sugar"].mean()), 1)
        }


# Global singleton instance
_NUTRITION_DB_INSTANCE: Optional[NutritionDatabase] = None


def get_nutrition_db() -> NutritionDatabase:
    """Retrieve singleton NutritionDatabase instance."""
    global _NUTRITION_DB_INSTANCE
    if _NUTRITION_DB_INSTANCE is None:
        _NUTRITION_DB_INSTANCE = NutritionDatabase()
    return _NUTRITION_DB_INSTANCE
