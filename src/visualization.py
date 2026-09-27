"""
Data Visualization Module for Food and Nutrition Analysis System.
Generates interactive, publication-quality Plotly charts for:
1. Macronutrient distribution (gram weights & calorie share).
2. Daily Value (%DV) benchmarks.
3. Transparent feature contribution breakdown (XAI).
4. SHAP feature importance plot.
5. Machine Learning Confusion Matrix.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import config


def create_macro_donut_chart(nutrition_info: Dict[str, Any]) -> go.Figure:
    """
    Create an interactive donut chart showing the macronutrient breakdown
    by weight and estimated calorie contribution.
    - Protein: 4 kcal/g
    - Carbohydrates: 4 kcal/g
    - Fat: 9 kcal/g
    """
    protein = max(0.0, float(nutrition_info.get("protein", 0.0)))
    carbs = max(0.0, float(nutrition_info.get("carbohydrates", 0.0)))
    fat = max(0.0, float(nutrition_info.get("fat", 0.0)))

    # Handle edge case where all macros are 0 (e.g., black coffee / water)
    total_g = protein + carbs + fat
    if total_g == 0:
        labels = ["Negligible Macros"]
        values = [1.0]
        colors = ["#9E9E9E"]
        custom_data = ["0 kcal"]
    else:
        labels = ["Protein", "Carbohydrates", "Fat"]
        values = [protein, carbs, fat]
        colors = ["#1E88E5", "#FB8C00", "#E53935"]
        cal_protein = round(protein * 4, 1)
        cal_carbs = round(carbs * 4, 1)
        cal_fat = round(fat * 9, 1)
        custom_data = [f"{cal_protein} kcal", f"{cal_carbs} kcal", f"{cal_fat} kcal"]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.55,
                marker=dict(colors=colors, line=dict(color="#FFFFFF", width=2)),
                textinfo="label+percent",
                hoverinfo="label+value+percent",
                customdata=custom_data,
                hovertemplate="<b>%{label}</b><br>Weight: %{value:.1f} g<br>Energy: %{customdata}<br>Share: %{percent}<extra></extra>"
            )
        ]
    )

    fig.update_layout(
        title=dict(text="<b>Macronutrient Distribution (by Weight)</b>", font=dict(size=15)),
        margin=dict(l=20, r=20, t=40, b=20),
        height=320,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def create_daily_value_bar_chart(nutrition_info: Dict[str, Any]) -> go.Figure:
    """
    Create a horizontal bar chart displaying % Daily Value (%DV)
    for key nutritional components against standard 2,000 kcal dietary guidelines.
    """
    pct = nutrition_info.get("daily_value_percentages", {})
    nutrients = ["Calories", "Protein", "Carbohydrates", "Fat", "Fiber", "Sugar"]
    keys = ["calories_pct", "protein_pct", "carbs_pct", "fat_pct", "fiber_pct", "sugar_pct"]
    values = [float(pct.get(k, 0.0)) for k in keys]

    # Color code: green for good/fiber/protein, amber for moderate, red for high sugar/fat
    bar_colors = []
    for nut, val in zip(nutrients, values):
        if nut == "Sugar" and val > 40:
            bar_colors.append("#E53935")
        elif nut == "Fiber" and val >= 20:
            bar_colors.append("#2E7D32")
        elif nut == "Protein" and val >= 25:
            bar_colors.append("#1976D2")
        else:
            bar_colors.append("#43A047" if val <= 30 else "#FB8C00")

    fig = go.Figure(
        data=[
            go.Bar(
                x=values,
                y=nutrients,
                orientation="h",
                marker=dict(color=bar_colors, line=dict(color="#424242", width=1)),
                text=[f"{v:.1f}%" for v in values],
                textposition="auto",
                hovertemplate="<b>%{y}</b>: %{x:.1f}% of Daily Reference<extra></extra>"
            )
        ]
    )

    # Reference 100% line
    fig.add_vline(x=100, line_dash="dash", line_color="#D32F2F", annotation_text="100% DV", annotation_position="top right")

    fig.update_layout(
        title=dict(text="<b>% Daily Value Contribution (%DV per Serving)</b>", font=dict(size=15)),
        xaxis=dict(title="% of Reference Intake (2000 kcal Diet)", zeroline=True),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=20, r=30, t=40, b=20),
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def create_feature_contribution_chart(score_result: Dict[str, Any]) -> go.Figure:
    """
    Create a waterfall or diverging horizontal bar chart illustrating
    the exact points awarded (positive) and deducted (negative)
    to calculate the 0-100 Reference Nutrition Score.
    """
    factors = score_result.get("factors", [])
    if not factors:
        fig = go.Figure()
        fig.update_layout(title="No factor adjustments applied.")
        return fig

    feature_names = [f["feature"] for f in factors]
    impacts = [float(f["impact"]) for f in factors]
    reasons = [f["reason"] for f in factors]
    values = [f["value"] for f in factors]

    # Colors: Green for positive, Red/Crimson for negative
    colors = ["#2E7D32" if val >= 0 else "#D32F2F" for val in impacts]

    fig = go.Figure(
        data=[
            go.Bar(
                x=impacts,
                y=feature_names,
                orientation="h",
                marker=dict(color=colors, line=dict(color="#424242", width=1)),
                text=[f"{'+' if v>0 else ''}{v:.1f} pts" for v in impacts],
                textposition="auto",
                customdata=list(zip(values, reasons)),
                hovertemplate="<b>%{y}</b> (%{customdata[0]})<br>Impact: %{x:+.1f} points<br>Reason: %{customdata[1]}<extra></extra>"
            )
        ]
    )

    fig.add_vline(x=0, line_color="#000000", line_width=1.5)

    fig.update_layout(
        title=dict(text="<b>Transparent Feature Contributions (Points Attribution)</b>", font=dict(size=15)),
        xaxis=dict(title="Score Impact (Points Added or Deducted)", zeroline=True),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=20, r=20, t=40, b=20),
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def create_shap_waterfall_chart(shap_result: Dict[str, Any]) -> go.Figure:
    """
    Create a SHAP feature importance plot showing how each nutrient
    influences the Machine Learning classifier's output.
    """
    contributions = shap_result.get("shap_contributions", [])
    predicted_class = shap_result.get("predicted_class", "Class")
    confidence = shap_result.get("confidence", 0.0)

    if not contributions:
        fig = go.Figure()
        fig.update_layout(title="SHAP explanations unavailable.")
        return fig

    features = [c["feature"] for c in contributions]
    shap_vals = [float(c["shap_value"]) for c in contributions]
    raw_vals = [c["value"] for c in contributions]

    colors = ["#1976D2" if v >= 0 else "#E64A19" for v in shap_vals]

    fig = go.Figure(
        data=[
            go.Bar(
                x=shap_vals,
                y=features,
                orientation="h",
                marker=dict(color=colors),
                text=[f"{v:+.3f}" for v in shap_vals],
                textposition="auto",
                customdata=raw_vals,
                hovertemplate="<b>%{y}</b> (Amount: %{customdata})<br>SHAP Value: %{x:+.4f}<extra></extra>"
            )
        ]
    )

    fig.add_vline(x=0, line_color="#424242", line_width=1.5)

    fig.update_layout(
        title=dict(
            text=f"<b>SHAP Feature Influence for Predicted: '{predicted_class}' ({confidence}%)</b>",
            font=dict(size=14)
        ),
        xaxis=dict(title="SHAP Value (Contribution to Log-Odds / Class Probability)"),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=20, r=20, t=45, b=20),
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def create_category_comparison_chart(
    food_name: str,
    nutrition_info: Dict[str, Any],
    category_averages: Dict[str, float]
) -> go.Figure:
    """
    Create a grouped bar chart comparing the selected food item
    against its category average.
    """
    metrics = ["Calories", "Protein", "Carbs", "Fat", "Fiber", "Sugar"]
    food_keys = ["calories", "protein", "carbohydrates", "fat", "fiber", "sugar"]

    food_vals = [float(nutrition_info.get(k, 0.0)) for k in food_keys]
    cat_vals = [float(category_averages.get(k, 0.0)) for k in food_keys]

    fig = go.Figure(
        data=[
            go.Bar(name=food_name, x=metrics, y=food_vals, marker_color="#2E7D32"),
            go.Bar(name=f"Category Avg ({nutrition_info.get('category')})", x=metrics, y=cat_vals, marker_color="#9E9E9E")
        ]
    )

    fig.update_layout(
        title=dict(text=f"<b>Comparison: {food_name} vs. Category Benchmark</b>", font=dict(size=15)),
        barmode="group",
        xaxis=dict(title="Nutritional Metric"),
        yaxis=dict(title="Amount (kcal / grams)"),
        margin=dict(l=20, r=20, t=40, b=20),
        height=320,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def create_confusion_matrix_chart(cm: List[List[int]], classes: List[str]) -> go.Figure:
    """
    Create a publication-quality confusion matrix heatmap for ML evaluation.
    """
    cm_array = np.array(cm)
    fig = go.Figure(
        data=go.Heatmap(
            z=cm_array,
            x=classes,
            y=classes,
            colorscale="Greens",
            text=cm_array,
            texttemplate="%{text}",
            textfont={"size": 16},
            hoverongaps=False
        )
    )

    fig.update_layout(
        title=dict(text="<b>ML Model Evaluation: Confusion Matrix</b>", font=dict(size=15)),
        xaxis=dict(title="Predicted Health Tier"),
        yaxis=dict(title="Actual Health Tier", autorange="reversed"),
        margin=dict(l=40, r=40, t=40, b=40),
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def create_packaged_macro_bar_chart(nutrition_info: Dict[str, Any]) -> go.Figure:
    """
    Create a clean macronutrient bar chart for packaged foods showing
    Protein, Carbohydrates, and Total Fat (with Saturated Fat breakout if present).
    """
    categories = ["Protein", "Carbohydrates", "Total Fat"]
    values = [
        float(nutrition_info.get("protein") or 0.0),
        float(nutrition_info.get("carbohydrates") or 0.0),
        float(nutrition_info.get("total_fat") or nutrition_info.get("fat") or 0.0)
    ]
    colors = ["#1E88E5", "#FB8C00", "#E53935"]

    sat_fat = nutrition_info.get("saturated_fat")
    if sat_fat is not None and float(sat_fat) > 0.0:
        categories.append("Saturated Fat")
        values.append(float(sat_fat))
        colors.append("#D32F2F")

    fig = go.Figure(
        data=[
            go.Bar(
                x=categories,
                y=values,
                marker=dict(color=colors, line=dict(color="#424242", width=1)),
                text=[f"{v:.1f} g" for v in values],
                textposition="auto",
                hovertemplate="<b>%{x}</b>: %{y:.1f} g<extra></extra>"
            )
        ]
    )

    fig.update_layout(
        title=dict(text="<b>Macronutrient Profile (grams per serving)</b>", font=dict(size=14)),
        xaxis=dict(title="Macronutrient"),
        yaxis=dict(title="Grams (g)", rangemode="tozero"),
        margin=dict(l=20, r=20, t=40, b=20),
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def create_sugar_fiber_chart(nutrition_info: Dict[str, Any]) -> go.Figure:
    """
    Comparison chart between Dietary Fiber (beneficial) and Total / Added Sugars (deductive).
    """
    labels = []
    values = []
    colors = []

    fiber = nutrition_info.get("fiber")
    if fiber is not None:
        labels.append("Dietary Fiber")
        values.append(float(fiber))
        colors.append("#2E7D32")

    tot_sugar = nutrition_info.get("total_sugar") or nutrition_info.get("sugar")
    if tot_sugar is not None:
        labels.append("Total Sugar")
        values.append(float(tot_sugar))
        colors.append("#FB8C00")

    add_sugar = nutrition_info.get("added_sugar")
    if add_sugar is not None:
        labels.append("Added Sugar")
        values.append(float(add_sugar))
        colors.append("#D32F2F")

    if not labels:
        labels = ["No Data Detected"]
        values = [0.0]
        colors = ["#9E9E9E"]

    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=values,
                marker=dict(color=colors, line=dict(color="#424242", width=1)),
                text=[f"{v:.1f} g" for v in values],
                textposition="auto",
                hovertemplate="<b>%{x}</b>: %{y:.1f} g<extra></extra>"
            )
        ]
    )

    fig.update_layout(
        title=dict(text="<b>Sugar vs. Fiber Quality Breakdown</b>", font=dict(size=14)),
        xaxis=dict(title="Nutrient Type"),
        yaxis=dict(title="Grams (g)", rangemode="tozero"),
        margin=dict(l=20, r=20, t=40, b=20),
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def create_allergen_summary_chart(allergen_table: List[Dict[str, str]]) -> go.Figure:
    """
    Create a summary bar chart showing detected allergens categorized by source.
    """
    if not allergen_table:
        fig = go.Figure()
        fig.update_layout(title="No allergens detected.")
        return fig

    df_al = pd.DataFrame(allergen_table)
    counts = df_al.groupby(["allergen", "source"]).size().reset_index(name="count")

    fig = px.bar(
        counts,
        x="allergen",
        y="count",
        color="source",
        barmode="group",
        color_discrete_map={"Ingredient": "#D32F2F", "Precautionary": "#FFA000"},
        labels={"allergen": "Allergen Category", "count": "Detected Count", "source": "Detection Source"},
        title="<b>Detected Allergen Distribution by Source</b>"
    )

    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        height=280,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig

