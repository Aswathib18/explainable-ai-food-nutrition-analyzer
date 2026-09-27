"""
Streamlit Web Application: Explainable AI-Based Real-Time Food & Nutrition Analysis System.
Author: BSc Data Science Academic Project

Supported Modes:
1. 🍎 Food Image Analysis (MobileNetV2 Computer Vision + USDA lookup + XAI)
2. 📦 Real-Time Food Label Scanner (Camera Capture + OCR + Nutrition Parser + Allergen Warning)
3. 🧾 Ingredient Scanner (Camera Capture + OCR + Ingredient Categorization + Cross-Contact Alerts)
4. ✍ Manual Nutrition Entry (Manual Data Entry + Transparent Scoring + Recommendations)
"""

import sys
from pathlib import Path
import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np

# Add project root to system path for reliable imports
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config
from src.utils import (
    validate_uploaded_image,
    save_analysis,
    get_analysis_history,
    clear_analysis_history
)
from src.image_processing import (
    load_image,
    compute_image_metrics,
    preprocess_for_ocr,
    assess_image_quality
)
from src.food_recognition import get_food_classifier
from src.nutrition import get_nutrition_db
from src.scoring import calculate_nutrition_score
from src.explainability import (
    get_xai_model,
    generate_dietary_recommendations
)
from src.visualization import (
    create_macro_donut_chart,
    create_daily_value_bar_chart,
    create_feature_contribution_chart,
    create_shap_waterfall_chart,
    create_category_comparison_chart,
    create_confusion_matrix_chart,
    create_packaged_macro_bar_chart,
    create_sugar_fiber_chart,
    create_allergen_summary_chart
)
from src.ocr import run_ocr
from src.nutrition_parser import (
    parse_nutrition_label,
    compute_per_100g_conversion
)
from src.ingredient_parser import (
    parse_ingredients_pipeline
)
from src.allergen_detector import (
    detect_allergens,
    ALLERGEN_SAFETY_DISCLAIMER
)
from src.product_lookup import lookup_open_food_facts

# ==============================================================================
# PAGE CONFIGURATION & STYLING
# ==============================================================================
st.set_page_config(
    page_title="Explainable AI Food & Nutrition System",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for academic aesthetic, metrics cards, and badges
st.markdown(
    """
    <style>
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #F8F9FA 0%, #FFFFFF 100%);
        border: 1px solid #E0E0E0;
        border-radius: 10px;
        padding: 14px 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
        margin-bottom: 10px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: #1E88E5;
        margin-top: 4px;
    }
    .metric-label {
        font-size: 12px;
        font-weight: 600;
        color: #616161;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-sub {
        font-size: 11px;
        color: #9E9E9E;
    }

    /* Score Badges */
    .score-badge {
        display: inline-block;
        font-size: 18px;
        font-weight: 700;
        padding: 6px 14px;
        border-radius: 20px;
        margin-top: 6px;
    }
    .badge-good { background-color: #E8F5E9; color: #2E7D32; border: 1px solid #A5D6A7; }
    .badge-moderate { background-color: #FFF8E1; color: #F57F17; border: 1px solid #FFE082; }
    .badge-poor { background-color: #FFEBEE; color: #C62828; border: 1px solid #EF9A9A; }

    /* Factor tags */
    .tag-positive {
        color: #2E7D32;
        background-color: #E8F5E9;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    .tag-negative {
        color: #C62828;
        background-color: #FFEBEE;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
    }

    /* Academic banner */
    .academic-banner {
        background: linear-gradient(90deg, #1565C0 0%, #2E7D32 100%);
        color: white;
        padding: 16px 22px;
        border-radius: 10px;
        margin-bottom: 18px;
    }
    .academic-banner h2 {
        color: white;
        margin: 0;
        font-size: 24px;
    }
    .academic-banner p {
        color: #E0E0E0;
        margin: 4px 0 0 0;
        font-size: 13.5px;
    }

    /* Pipeline Step Badge */
    .pipeline-step {
        display: inline-block;
        background-color: #ECEFF1;
        color: #37474F;
        padding: 5px 12px;
        border-radius: 16px;
        font-size: 12px;
        font-weight: 600;
        margin: 2px 4px;
        border: 1px solid #CFD8DC;
    }
    .pipeline-step-active {
        background-color: #E8F5E9;
        color: #2E7D32;
        border-color: #81C784;
    }

    /* Allergen alert box */
    .allergen-box {
        border-left: 5px solid #D32F2F;
        background-color: #FFEBEE;
        padding: 12px 16px;
        border-radius: 4px;
        margin: 10px 0;
    }
    .precautionary-box {
        border-left: 5px solid #FFA000;
        background-color: #FFF8E1;
        padding: 12px 16px;
        border-radius: 4px;
        margin: 10px 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ==============================================================================
# INITIALIZE SERVICES
# ==============================================================================
classifier = get_food_classifier()
nutrition_db = get_nutrition_db()
xai_model = get_xai_model()
all_foods = nutrition_db.get_all_food_names()

# ==============================================================================
# SIDEBAR NAVIGATION & SETTINGS
# ==============================================================================
with st.sidebar:
    st.markdown("### 🎓 Academic Project Info")
    st.info(
        "**Title**: Explainable AI-Based Real-Time Food and Nutrition Analysis System\n\n"
        "**Domain**: Computer Vision, OCR, Explainable AI, Nutrition Informatics"
    )

    st.markdown("---")
    st.markdown("### 🧭 Main Navigation Modes")

    selected_mode = st.radio(
        "Select Application Mode:",
        [
            "🍎 Food Image Analysis",
            "📦 Real-Time Food Label Scanner",
            "🧾 Ingredient Scanner",
            "✍ Manual Nutrition Entry"
        ],
        index=0
    )

    st.markdown("---")
    st.markdown("### ⚙️ Mode Settings")

    if selected_mode == "🍎 Food Image Analysis":
        food_input_mode = st.radio(
            "Food Source:",
            ["📸 Upload Food Photo", "🔍 Manual Food Search"],
            index=0
        )
        confidence_threshold = st.slider(
            "AI Confidence Threshold (%):",
            min_value=10,
            max_value=90,
            value=30,
            step=5,
            help="If AI vision confidence falls below this threshold, manual confirmation is advised."
        )
        st.markdown("---")
        st.markdown("### 🤖 Vision Model Status")
        if classifier.fallback_mode:
            st.warning(f"⚠️ {classifier.status_message}")
        else:
            st.success("✅ MobileNetV2 ONNX Active (Local CPU Inference)")

    elif selected_mode in ["📦 Real-Time Food Label Scanner", "🧾 Ingredient Scanner"]:
        st.caption("📷 Camera Capture Mode: Uses Streamlit Real-Time Camera input.")
        st.caption("🔒 Privacy: Images processed in-memory and not stored by default.")
        st.success("✅ Local EasyOCR Deep Learning Engine Active")

    st.markdown("---")
    if st.button("🔄 Reset Application Session", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# ==============================================================================
# MAIN PAGE HEADER & PIPELINE BANNER
# ==============================================================================
st.markdown(
    """
    <div class="academic-banner">
        <h2>🥗 Explainable AI-Based Real-Time Food & Nutrition Analysis System</h2>
        <p>An end-to-end multi-modal framework integrating deep learning computer vision, real-time packaged food label OCR, transparent rule-based scoring, SHAP explainability, and allergen detection.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Navigation tabs for comprehensive demonstration
main_tabs = st.tabs(["📊 Interactive Analysis", "🧠 ML Model & SHAP Lab", "📜 Analysis History", "ℹ️ System Architecture & Viva"])


# ==============================================================================
# HELPER: DISPLAY NUTRITION METRICS CARDS
# ==============================================================================
def display_nutrition_metrics_grid(nut: Dict[str, Any]):
    """Renders 10 clean metric cards for packaged or whole food nutrition."""
    def fmt_g(v):
        return f"{float(v):.1f} g" if v is not None else "Not detected"

    def fmt_mg(v):
        return f"{float(v):.0f} mg" if v is not None else "Not detected"

    c1, c2, c3, c4 = st.columns(4)
    c5, c6, c7, c8 = st.columns(4)
    c9, c10 = st.columns(2)

    cal = nut.get("calories")
    cal_str = f"{float(cal):.0f} kcal" if cal is not None else "Not detected"
    c1.markdown(f"""<div class="metric-card"><div class="metric-label">Calories</div><div class="metric-value">{cal_str}</div></div>""", unsafe_allow_html=True)
    c2.markdown(f"""<div class="metric-card"><div class="metric-label">Protein</div><div class="metric-value">{fmt_g(nut.get('protein'))}</div></div>""", unsafe_allow_html=True)
    c3.markdown(f"""<div class="metric-card"><div class="metric-label">Carbohydrates</div><div class="metric-value">{fmt_g(nut.get('carbohydrates'))}</div></div>""", unsafe_allow_html=True)
    c4.markdown(f"""<div class="metric-card"><div class="metric-label">Total Fat</div><div class="metric-value">{fmt_g(nut.get('total_fat') or nut.get('fat'))}</div></div>""", unsafe_allow_html=True)

    c5.markdown(f"""<div class="metric-card"><div class="metric-label">Saturated Fat</div><div class="metric-value">{fmt_g(nut.get('saturated_fat'))}</div></div>""", unsafe_allow_html=True)
    c6.markdown(f"""<div class="metric-card"><div class="metric-label">Trans Fat</div><div class="metric-value">{fmt_g(nut.get('trans_fat'))}</div></div>""", unsafe_allow_html=True)
    c7.markdown(f"""<div class="metric-card"><div class="metric-label">Dietary Fiber</div><div class="metric-value">{fmt_g(nut.get('fiber'))}</div></div>""", unsafe_allow_html=True)
    c8.markdown(f"""<div class="metric-card"><div class="metric-label">Total Sugar</div><div class="metric-value">{fmt_g(nut.get('total_sugar') or nut.get('sugar'))}</div></div>""", unsafe_allow_html=True)

    c9.markdown(f"""<div class="metric-card"><div class="metric-label">Added Sugar</div><div class="metric-value">{fmt_g(nut.get('added_sugar'))}</div></div>""", unsafe_allow_html=True)
    c10.markdown(f"""<div class="metric-card"><div class="metric-label">Sodium</div><div class="metric-value">{fmt_mg(nut.get('sodium'))}</div></div>""", unsafe_allow_html=True)


# ==============================================================================
# HELPER: DISPLAY ALLERGEN SECTION
# ==============================================================================
def display_allergen_section(allergen_res: Dict[str, Any]):
    """Renders prominent allergen warning section, distinction, and table."""
    st.markdown("### 🚨 ALLERGEN INFORMATION")

    table = allergen_res.get("allergen_table", [])
    has_allergens = allergen_res.get("has_allergens", False)
    precautionary = allergen_res.get("precautionary_statements", [])

    if has_allergens:
        # Separate direct ingredient allergens and precautionary allergens
        ing_allergens = [a for a in table if a["source"] == "Ingredient"]
        prec_allergens = [a for a in table if a["source"] == "Precautionary"]

        if ing_allergens:
            st.markdown("#### ⚠️ Potential Allergens Detected in Ingredients:")
            for a in ing_allergens:
                st.markdown(
                    f"""
                    <div class="allergen-box">
                        <b>{a['icon']} {a['allergen']} Detected</b><br>
                        Matched Ingredient Term: <code>{a['detected_term']}</code>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        if prec_allergens or precautionary:
            st.markdown("#### ⚠️ Cross-Contact / Precautionary Statements Detected:")
            st.caption("Note: These are facility cross-contact warnings, not direct recipe ingredients.")
            for p in precautionary:
                st.markdown(
                    f"""
                    <div class="precautionary-box">
                        <b>Facility Cross-Contact Warning:</b> "{p}"
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # Allergen category table
        st.markdown("##### 📋 Allergen Summary Table")
        df_al = pd.DataFrame(table)[["allergen", "detected_term", "source"]]
        df_al.columns = ["Allergen Category", "Detected Term", "Source"]
        st.dataframe(df_al, use_container_width=True)

    else:
        st.success("✅ No supported allergen terms detected in the scanned ingredient text.")

    # Mandatory safety disclaimer
    st.info(f"ℹ️ **Safety Note**: {ALLERGEN_SAFETY_DISCLAIMER}")


# ==============================================================================
# TAB 1: INTERACTIVE ANALYSIS (MODES 1, 2, 3, 4)
# ==============================================================================
with main_tabs[0]:

    # --------------------------------------------------------------------------
    # MODE 1: 🍎 FOOD IMAGE ANALYSIS
    # --------------------------------------------------------------------------
    if selected_mode == "🍎 Food Image Analysis":
        col_input, col_results = st.columns([1, 1.4], gap="large")

        active_food: str = ""
        active_source: str = "AI Detected"
        active_confidence: float = 100.0
        image_for_display = None
        cv_prediction_meta = None

        with col_input:
            st.subheader("1. Input Food Source")

            if food_input_mode == "📸 Upload Food Photo":
                uploaded_file = st.file_uploader(
                    "Upload a clear food photo (JPG, JPEG, PNG, WEBP):",
                    type=["jpg", "jpeg", "png", "webp"],
                    help="Maximum size: 10 MB",
                    key="food_uploader"
                )

                st.caption("Or test with pre-packaged sample foods:")
                c1, c2, c3, c4 = st.columns(4)
                sample_choice = None
                if c1.button("🍎 Apple"):
                    sample_choice = "Apple"
                if c2.button("🍕 Pizza"):
                    sample_choice = "Pizza"
                if c3.button("🥦 Broccoli"):
                    sample_choice = "Broccoli"
                if c4.button("🍔 Burger"):
                    sample_choice = "Burger"

                if sample_choice:
                    active_food = sample_choice
                    active_source = "User Selected"
                    active_confidence = 100.0
                    st.success(f"Selected sample: {sample_choice}")

                if uploaded_file is not None:
                    is_valid, err_msg = validate_uploaded_image(uploaded_file)
                    if not is_valid:
                        st.error(f"❌ {err_msg}")
                    else:
                        try:
                            pil_img = load_image(uploaded_file)
                            image_for_display = pil_img
                            st.image(pil_img, caption="Uploaded Food Image", use_container_width=True)

                            img_metrics = compute_image_metrics(pil_img)
                            with st.expander("🖼️ Computer Vision Image Diagnostics", expanded=False):
                                m_c1, m_c2, m_c3 = st.columns(3)
                                m_c1.metric("Resolution", f"{img_metrics['width']}x{img_metrics['height']}")
                                m_c2.metric("Brightness", f"{img_metrics['mean_brightness']}")
                                m_c3.metric("Sharpness", f"{img_metrics['sharpness_score']}")

                            with st.spinner("🤖 Running MobileNetV2 Deep Learning Inference..."):
                                cv_prediction_meta = classifier.classify_image(pil_img)

                            detected_food = cv_prediction_meta["food_name"]
                            conf = cv_prediction_meta["confidence"]

                            if cv_prediction_meta.get("is_fallback_mode"):
                                st.warning(f"⚠️ {cv_prediction_meta.get('mode_description')}")

                            st.markdown("#### 🎯 AI Recognition Output")
                            st.info(f"**Predicted Food**: **{detected_food}**\n\n**Confidence**: **{conf:.1f}%**")

                            if cv_prediction_meta.get("top_predictions"):
                                st.caption("Alternative Candidate Predictions:")
                                for rank, alt in enumerate(cv_prediction_meta["top_predictions"], 1):
                                    st.write(f"{rank}. **{alt['food_name']}** ({alt['raw_label']}) — {alt['confidence']:.1f}%")
                                    st.progress(min(1.0, alt["confidence"] / 100.0))

                            if conf < confidence_threshold or detected_food == "Unknown Food":
                                st.warning(
                                    f"⚠️ Prediction confidence ({conf:.1f}%) is below your threshold ({confidence_threshold}%). "
                                    "You can confirm or manually pick the correct food below."
                                )
                                manual_override = st.selectbox(
                                    "Select Food Item Manually:",
                                    options=["(Use AI Prediction)"] + all_foods,
                                    index=0,
                                    key="manual_override_select"
                                )
                                if manual_override != "(Use AI Prediction)":
                                    active_food = manual_override
                                    active_source = "User Selected (Manual Override)"
                                    active_confidence = 100.0
                                else:
                                    active_food = detected_food
                                    active_source = "AI Detected"
                                    active_confidence = conf
                            else:
                                active_food = detected_food
                                active_source = "AI Detected"
                                active_confidence = conf

                        except Exception as exc:
                            st.error(f"Image processing error: {exc}")

            else:
                st.markdown("#### 🔍 Direct Food Selection")
                manual_pick = st.selectbox(
                    "Choose food item from database:",
                    options=all_foods,
                    index=all_foods.index("Apple") if "Apple" in all_foods else 0,
                    key="direct_food_pick"
                )
                if manual_pick:
                    active_food = manual_pick
                    active_source = "User Selected"
                    active_confidence = 100.0

        with col_results:
            st.subheader("2. Nutritional Profiling & Explainable AI")

            if not active_food or active_food == "Unknown Food":
                st.info("👈 Upload an image or select a food item on the left to begin analysis.")
            else:
                nutrition_info = nutrition_db.get_nutrition_info(active_food)
                if not nutrition_info:
                    st.error(f"⚠️ Nutrition information for '**{active_food}**' is not available in current dataset.")
                else:
                    st.markdown(
                        f"### 🍽️ {nutrition_info['food']} "
                        f"<span style='font-size:14px; color:#616161;'>({active_source} • {nutrition_info['serving_size']} {nutrition_info['serving_unit']} serving)</span>",
                        unsafe_allow_html=True
                    )

                    # Display macro metric cards
                    display_nutrition_metrics_grid(nutrition_info)

                    # Score & XAI
                    score_dict = calculate_nutrition_score(nutrition_info)
                    score_val = score_dict["score"]
                    tier_val = score_dict["health_tier"]
                    badge_class = "badge-good" if score_val >= 70 else ("badge-moderate" if score_val >= 45 else "badge-poor")

                    st.markdown(
                        f"""
                        <div style="background:#FFFFFF; border:1px solid #E0E0E0; border-radius:10px; padding:18px; margin: 15px 0;">
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <div>
                                    <span style="font-size:14px; font-weight:600; color:#616161;">REFERENCE NUTRITION SCORE</span>
                                    <div style="font-size:36px; font-weight:800; color:#2E7D32;">{score_val:.1f} <span style="font-size:20px; color:#9E9E9E;">/ 100</span></div>
                                </div>
                                <div class="score-badge {badge_class}">{tier_val}</div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    save_analysis(
                        food_name=nutrition_info["food"],
                        confidence=active_confidence,
                        nutrition_info=nutrition_info,
                        nutrition_score=score_val,
                        health_tier=tier_val,
                        source_type=active_source,
                        analysis_mode="Food Image Analysis"
                    )

                    st.markdown("### 🔍 Explainable AI: Why did this food receive this score?")
                    xai_col1, xai_col2 = st.columns(2)
                    pos_factors = [f for f in score_dict["factors"] if f["status"] == "positive"]
                    neg_factors = [f for f in score_dict["factors"] if f["status"] == "negative"]

                    with xai_col1:
                        st.markdown("**Positive Contributing Factors (Score Boosters):**")
                        if pos_factors:
                            for pf in pos_factors:
                                st.markdown(f"✓ <span class='tag-positive'>+{pf['impact']} pts</span> **{pf['feature']}** ({pf['value']})<br><small>{pf['reason']}</small>", unsafe_allow_html=True)
                        else:
                            st.caption("No notable positive score boosts identified.")

                    with xai_col2:
                        st.markdown("**Negative Deductions (Score Reducers):**")
                        if neg_factors:
                            for nf in neg_factors:
                                st.markdown(f"⚠️ <span class='tag-negative'>{nf['impact']} pts</span> **{nf['feature']}** ({nf['value']})<br><small>{nf['reason']}</small>", unsafe_allow_html=True)
                        else:
                            st.caption("No penalizing deductions for excess sugar, fat, or calorie density.")

                    st.plotly_chart(create_feature_contribution_chart(score_dict), use_container_width=True)

                    st.markdown("### 📈 Nutrient Visualizations")
                    ch_col1, ch_col2 = st.columns(2)
                    with ch_col1:
                        st.plotly_chart(create_macro_donut_chart(nutrition_info), use_container_width=True)
                    with ch_col2:
                        st.plotly_chart(create_daily_value_bar_chart(nutrition_info), use_container_width=True)

                    cat_avg = nutrition_db.get_category_averages(nutrition_info["category"])
                    if cat_avg:
                        st.plotly_chart(
                            create_category_comparison_chart(nutrition_info["food"], nutrition_info, cat_avg),
                            use_container_width=True
                        )

                    st.markdown("### 💡 Data-Driven Educational Recommendations")
                    recs = generate_dietary_recommendations(nutrition_info, score_dict)
                    for r in recs:
                        st.info(r)

    # --------------------------------------------------------------------------
    # MODE 2: 📦 REAL-TIME FOOD LABEL SCANNER
    # --------------------------------------------------------------------------
    elif selected_mode == "📦 Real-Time Food Label Scanner":
        st.markdown(
            """
            <div style="background:#E3F2FD; border:1px solid #90CAF9; border-radius:8px; padding:12px; margin-bottom:14px;">
                <b>📦 Real-Time Food Label Scanner</b><br>
                Point your camera at a packaged food nutrition label to extract real nutritional metrics via OCR,
                calculate a transparent Reference Nutrition Score, explain contributions, and detect allergen ingredients.
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <span class="pipeline-step">📷 Camera Capture</span> →
            <span class="pipeline-step">🖼 Image Preprocessing</span> →
            <span class="pipeline-step">🔍 OCR</span> →
            <span class="pipeline-step">🧾 Text Extraction</span> →
            <span class="pipeline-step">🥗 Nutrition Parsing</span> →
            <span class="pipeline-step">⚠️ Allergen Detection</span> →
            <span class="pipeline-step">📊 Nutrition Analysis</span> →
            <span class="pipeline-step">🧠 Explainable AI</span>
            """,
            unsafe_allow_html=True
        )

        st.caption("🔒 *Camera Privacy Note: Captured images are processed locally in-memory and are not stored by default.*")

        scan_col_left, scan_col_right = st.columns([1, 1.3], gap="large")

        captured_label_img = None
        label_source_type = "Real-Time Camera Capture"

        with scan_col_left:
            st.subheader("📷 Scan Food Label")

            input_tab1, input_tab2, input_tab3 = st.tabs(["📷 Real-Time Camera", "📁 Upload Image", "🧪 Test Sample Label"])

            with input_tab1:
                st.write("Position the nutrition label clearly inside the camera.")
                cam_image = st.camera_input("Position the nutrition label clearly inside the camera.", key="label_cam_input")
                if cam_image:
                    captured_label_img = load_image(cam_image)
                    label_source_type = "Real-Time Camera Capture"

            with input_tab2:
                uploaded_label = st.file_uploader(
                    "Upload a clear photo of nutrition facts label:",
                    type=["jpg", "jpeg", "png", "webp"],
                    key="label_file_uploader"
                )
                if uploaded_label:
                    captured_label_img = load_image(uploaded_label)
                    label_source_type = "Uploaded Label Photo"

            with input_tab3:
                st.write("Quickly test with our verified packaged food sample:")
                if st.button("🧪 Load Sample: Oat & Honey Biscuit Label", use_container_width=True):
                    sample_path = Path("assets/sample_biscuit_label.jpg")
                    if sample_path.exists():
                        captured_label_img = Image.open(sample_path)
                        label_source_type = "Verified Sample Label"
                        st.session_state["active_sample_label"] = captured_label_img

                if "active_sample_label" in st.session_state and captured_label_img is None:
                    captured_label_img = st.session_state["active_sample_label"]
                    label_source_type = "Verified Sample Label"

            # Image Preprocessing & Diagnostics display
            if captured_label_img is not None:
                st.markdown("---")
                st.markdown("#### 🖼️ Image Preprocessing & Diagnostics")

                with st.spinner("Processing image for OCR..."):
                    preprocessed_dict = preprocess_for_ocr(captured_label_img)

                diag = preprocessed_dict["diagnostics"]
                metrics = diag["metrics"]

                # Quality indicator badge
                if diag["is_acceptable"]:
                    st.success(f"✅ Quality: **{diag['quality_status']}** (Sharpness: {metrics['sharpness_score']}, Brightness: {metrics['mean_brightness']})")
                else:
                    st.warning(f"⚠️ Quality: **{diag['quality_status']}** — {diag['quality_desc']}")
                    for sugg in diag["suggestions"]:
                        st.write(f"- {sugg}")

                # Show Original vs Processed side-by-side
                img_c1, img_c2 = st.columns(2)
                with img_c1:
                    st.image(preprocessed_dict["original"], caption="Original Image", use_container_width=True)
                with img_c2:
                    st.image(preprocessed_dict["processed_primary"], caption="Processed for OCR (CLAHE)", use_container_width=True)

                with st.expander("🔬 View Computer Vision Preprocessing Stages", expanded=False):
                    stage_tabs = st.tabs(["Grayscale", "CLAHE Contrast", "Sharpened", "Otsu Threshold", "Adaptive Gaussian"])
                    with stage_tabs[0]:
                        st.image(preprocessed_dict["grayscale"], use_container_width=True)
                    with stage_tabs[1]:
                        st.image(preprocessed_dict["contrast_enhanced"], use_container_width=True)
                    with stage_tabs[2]:
                        st.image(preprocessed_dict["sharpened"], use_container_width=True)
                    with stage_tabs[3]:
                        st.image(preprocessed_dict["otsu_thresh"], use_container_width=True)
                    with stage_tabs[4]:
                        st.image(preprocessed_dict["adaptive_thresh"], use_container_width=True)

                # Execute OCR
                with st.spinner("🔍 Running Local EasyOCR Detection & Recognition..."):
                    # Run on contrast enhanced processed image for maximum text accuracy
                    ocr_res = run_ocr(preprocessed_dict["contrast_enhanced"])

                if not ocr_res["success"]:
                    st.error("⚠️ Unable to reliably read the label. Please capture the image again with better lighting.")
                    st.caption("You can also manually enter the values using the Manual Nutrition Entry mode.")
                else:
                    raw_ocr_text = ocr_res["raw_text"]
                    ocr_conf = ocr_res["confidence"]

                    st.markdown(f"**OCR Status**: ✅ Detected {ocr_res['line_count']} text regions *(Confidence: {ocr_conf * 100:.1f}%)*")
                    with st.expander("📄 View Raw Extracted OCR Text", expanded=False):
                        st.text_area("Raw Text:", value=raw_ocr_text, height=180, disabled=True)

                    # Initial parsing
                    parsed_initial = parse_nutrition_label(raw_ocr_text)
                    allergens_initial = detect_allergens(raw_ocr_text)

                    # Store in session state for verification form
                    st.session_state["ocr_parsed_data"] = parsed_initial
                    st.session_state["ocr_allergens_data"] = allergens_initial
                    st.session_state["ocr_confidence_score"] = ocr_conf
                    st.session_state["label_source_type"] = label_source_type

        with scan_col_right:
            st.subheader("2. OCR Extraction & User Verification")

            if "ocr_parsed_data" not in st.session_state:
                st.info("👈 Capture a food label using the camera or upload an image on the left to begin OCR scanning.")
            else:
                p_data = st.session_state["ocr_parsed_data"]

                st.markdown("### 🔍 Verify Extracted Nutrition Information")
                st.write("*These values were extracted from the package label using OCR. Please verify them before analysis.*")

                with st.form("verify_nutrition_form"):
                    f_c1, f_c2 = st.columns(2)
                    with f_c1:
                        v_product = st.text_input("Product Name:", value=p_data.get("product_name") or "Packaged Food Product")
                        v_serving = st.text_input("Serving Size:", value=p_data.get("serving_size") or "30g")
                        v_calories = st.number_input("Calories (kcal):", min_value=0.0, max_value=2500.0, value=float(p_data.get("calories") or 0.0), step=5.0)
                        v_protein = st.number_input("Protein (g):", min_value=0.0, max_value=150.0, value=float(p_data.get("protein") or 0.0), step=0.5)
                        v_carbs = st.number_input("Carbohydrates (g):", min_value=0.0, max_value=300.0, value=float(p_data.get("carbohydrates") or 0.0), step=1.0)
                        v_fat = st.number_input("Total Fat (g):", min_value=0.0, max_value=150.0, value=float(p_data.get("total_fat") or 0.0), step=0.5)

                    with f_c2:
                        v_sat_fat = st.number_input("Saturated Fat (g):", min_value=0.0, max_value=100.0, value=float(p_data.get("saturated_fat") or 0.0), step=0.5)
                        v_trans_fat = st.number_input("Trans Fat (g):", min_value=0.0, max_value=50.0, value=float(p_data.get("trans_fat") or 0.0), step=0.1)
                        v_fiber = st.number_input("Dietary Fiber (g):", min_value=0.0, max_value=80.0, value=float(p_data.get("fiber") or 0.0), step=0.5)
                        v_total_sugar = st.number_input("Total Sugar (g):", min_value=0.0, max_value=200.0, value=float(p_data.get("total_sugar") or 0.0), step=0.5)
                        v_added_sugar = st.number_input("Added Sugar (g):", min_value=0.0, max_value=200.0, value=float(p_data.get("added_sugar") or 0.0), step=0.5)
                        v_sodium = st.number_input("Sodium (mg):", min_value=0.0, max_value=5000.0, value=float(p_data.get("sodium") or 0.0), step=10.0)

                    v_raw_text = st.text_area(
                        "Scanned Ingredients & Statements for Allergen Detection:",
                        value=p_data.get("normalized_text") or "",
                        height=100
                    )

                    confirm_button = st.form_submit_button("✅ Confirm & Analyze Label", use_container_width=True)

                if confirm_button or "confirmed_label_data" in st.session_state:
                    if confirm_button:
                        confirmed_nut = {
                            "product_name": v_product,
                            "serving_size": v_serving,
                            "calories": v_calories,
                            "protein": v_protein,
                            "carbohydrates": v_carbs,
                            "total_fat": v_fat,
                            "saturated_fat": v_sat_fat,
                            "trans_fat": v_trans_fat,
                            "fiber": v_fiber,
                            "total_sugar": v_total_sugar,
                            "added_sugar": v_added_sugar,
                            "sodium": v_sodium,
                            "raw_text": v_raw_text
                        }
                        st.session_state["confirmed_label_data"] = confirmed_nut

                    c_nut = st.session_state["confirmed_label_data"]

                    st.markdown("---")
                    st.markdown(f"### 📦 Confirmed Nutrition Profile: {c_nut['product_name']}")
                    st.caption(f"Serving Size: **{c_nut['serving_size']}** | Source: **{st.session_state.get('label_source_type', 'OCR Scanner')}**")

                    # Display confirmed metrics cards
                    display_nutrition_metrics_grid(c_nut)

                    # Per-Serving vs Per-100g table (Requirement 14)
                    per_100g_data, conv_reason = compute_per_100g_conversion(c_nut)
                    if per_100g_data:
                        with st.expander("⚖️ View Per-Serving vs. Per-100g Comparison", expanded=False):
                            st.caption(f"Conversion basis: {conv_reason}")
                            df_compare = pd.DataFrame({
                                "Metric": [
                                    "Calories (kcal)", "Protein (g)", "Carbohydrates (g)",
                                    "Total Fat (g)", "Saturated Fat (g)", "Dietary Fiber (g)",
                                    "Total Sugar (g)", "Added Sugar (g)", "Sodium (mg)"
                                ],
                                f"Per Serving ({c_nut['serving_size']})": [
                                    f"{c_nut['calories']:.0f}", f"{c_nut['protein']:.1f}", f"{c_nut['carbohydrates']:.1f}",
                                    f"{c_nut['total_fat']:.1f}", f"{c_nut['saturated_fat']:.1f}", f"{c_nut['fiber']:.1f}",
                                    f"{c_nut['total_sugar']:.1f}", f"{c_nut['added_sugar']:.1f}", f"{c_nut['sodium']:.0f}"
                                ],
                                "Per 100g": [
                                    f"{per_100g_data['calories']:.0f}", f"{per_100g_data['protein']:.1f}", f"{per_100g_data['carbohydrates']:.1f}",
                                    f"{per_100g_data['total_fat']:.1f}", f"{per_100g_data['saturated_fat']:.1f}", f"{per_100g_data['fiber']:.1f}",
                                    f"{per_100g_data['total_sugar']:.1f}", f"{per_100g_data['added_sugar']:.1f}", f"{per_100g_data['sodium']:.0f}"
                                ]
                            })
                            st.dataframe(df_compare, use_container_width=True)

                    # Calculate Reference Nutrition Score
                    pkg_score_res = calculate_nutrition_score(c_nut)
                    pkg_score_val = pkg_score_res["score"]
                    pkg_tier_val = pkg_score_res["health_tier"]
                    pkg_badge_class = "badge-good" if pkg_score_val >= 70 else ("badge-moderate" if pkg_score_val >= 45 else "badge-poor")

                    st.markdown(
                        f"""
                        <div style="background:#FFFFFF; border:1px solid #E0E0E0; border-radius:10px; padding:18px; margin: 15px 0;">
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <div>
                                    <span style="font-size:14px; font-weight:600; color:#616161;">REFERENCE NUTRITION SCORE</span>
                                    <div style="font-size:36px; font-weight:800; color:#2E7D32;">{pkg_score_val:.1f} <span style="font-size:20px; color:#9E9E9E;">/ 100</span></div>
                                </div>
                                <div class="score-badge {pkg_badge_class}">{pkg_tier_val}</div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # Allergen Detection on verified text
                    allergen_analysis = detect_allergens(c_nut.get("raw_text", ""))
                    detected_allergens_summary = ", ".join([f"{a['allergen']} ({a['detected_term']})" for a in allergen_analysis.get("allergen_table", [])]) or "None Detected"

                    # Save to SQLite history
                    save_analysis(
                        food_name=c_nut["product_name"],
                        confidence=round(float(st.session_state.get("ocr_confidence_score", 0.90)) * 100, 1),
                        nutrition_info=c_nut,
                        nutrition_score=pkg_score_val,
                        health_tier=pkg_tier_val,
                        source_type="Real-Time Food Label Scanner",
                        analysis_mode="Real-Time Food Label Scanner",
                        serving_size=c_nut["serving_size"],
                        detected_allergens=detected_allergens_summary,
                        ocr_confidence=st.session_state.get("ocr_confidence_score", 0.90)
                    )

                    # Explainable AI section (Requirement 16)
                    st.markdown("### 🧠 Why did this food receive this score?")
                    st.caption("Feature contribution explanation based on transparent, auditable points attribution:")
                    sc_col1, sc_col2 = st.columns(2)
                    pos_f = [f for f in pkg_score_res["factors"] if f["status"] == "positive"]
                    neg_f = [f for f in pkg_score_res["factors"] if f["status"] == "negative"]

                    with sc_col1:
                        st.markdown("**Positive Contributors (Score Boosters):**")
                        if pos_f:
                            for pf in pos_f:
                                st.markdown(f"✓ <span class='tag-positive'>+{pf['impact']} pts</span> **{pf['feature']}** ({pf['value']})<br><small>{pf['reason']}</small>", unsafe_allow_html=True)
                        else:
                            st.caption("No significant positive bonuses.")

                    with sc_col2:
                        st.markdown("**Negative Deductions (Score Reducers):**")
                        if neg_f:
                            for nf in neg_f:
                                st.markdown(f"⚠️ <span class='tag-negative'>{nf['impact']} pts</span> **{nf['feature']}** ({nf['value']})<br><small>{nf['reason']}</small>", unsafe_allow_html=True)
                        else:
                            st.caption("No penalizing deductions.")

                    st.plotly_chart(create_feature_contribution_chart(pkg_score_res), use_container_width=True)

                    # Allergen warning display
                    display_allergen_section(allergen_analysis)

                    # Ingredient parsing & categorization (Requirement 18, 19)
                    ing_parsed = parse_ingredients_pipeline(c_nut.get("raw_text", ""))
                    if ing_parsed["ingredients"]:
                        st.markdown("### 🧾 Detected Ingredients Breakdown")
                        ing_col1, ing_col2 = st.columns(2)
                        with ing_col1:
                            st.markdown("**Parsed Ingredients List:**")
                            for item in ing_parsed["ingredients"]:
                                st.markdown(f"• {item}")
                        with ing_col2:
                            st.markdown("**Food Science Category Tags:**")
                            for item in ing_parsed["categorized_items"]:
                                st.markdown(f"{item['icon']} **{item['ingredient']}** — *{item['category']}*")

                    # Visualizations (Requirement 26)
                    st.markdown("### 📈 Packaged Food Nutrient Charts")
                    p_ch1, p_ch2 = st.columns(2)
                    with p_ch1:
                        st.plotly_chart(create_packaged_macro_bar_chart(c_nut), use_container_width=True)
                    with p_ch2:
                        st.plotly_chart(create_sugar_fiber_chart(c_nut), use_container_width=True)

                    if allergen_analysis.get("allergen_table"):
                        st.plotly_chart(create_allergen_summary_chart(allergen_analysis["allergen_table"]), use_container_width=True)

                    # Optional Open Food Facts Lookup (Requirement 29)
                    with st.expander("🌐 Optional Product Database Lookup (Open Food Facts)", expanded=False):
                        st.caption("Cross-reference with open global database (strictly non-blocking with 3s timeout).")
                        off_search = st.text_input("Search Open Food Facts for comparison:", value=c_nut["product_name"], key="off_lookup_input")
                        if st.button("🔍 Search Online Database", key="btn_off_search"):
                            with st.spinner("Querying Open Food Facts..."):
                                off_data = lookup_open_food_facts(off_search)
                            if off_data:
                                st.success(f"Found: **{off_data['product_name']}** ({off_data.get('brands')})")
                                st.json(off_data)
                            else:
                                st.info("Product not found in Open Food Facts or offline. Local OCR remains primary.")

    # --------------------------------------------------------------------------
    # MODE 3: 🧾 INGREDIENT SCANNER
    # --------------------------------------------------------------------------
    elif selected_mode == "🧾 Ingredient Scanner":
        st.markdown(
            """
            <div style="background:#FFF3E0; border:1px solid #FFE082; border-radius:8px; padding:12px; margin-bottom:14px;">
                <b>🧾 Ingredient List Scanner</b><br>
                Dedicated scanner for package ingredient sections. Captures text, normalizes OCR typos,
                classifies ingredients into neutral food science categories, detects allergens, and isolates cross-contact statements.
            </div>
            """,
            unsafe_allow_html=True
        )

        ing_col_left, ing_col_right = st.columns([1, 1.3], gap="large")

        captured_ing_img = None

        with ing_col_left:
            st.subheader("📷 Capture Ingredients List")
            ing_tab1, ing_tab2 = st.tabs(["📷 Real-Time Camera", "📁 Upload Image / Sample"])

            with ing_tab1:
                st.write("Position the ingredient list clearly inside the camera.")
                ing_cam = st.camera_input("Position the ingredient list inside the camera.", key="ing_cam_input")
                if ing_cam:
                    captured_ing_img = load_image(ing_cam)

            with ing_tab2:
                ing_up = st.file_uploader("Upload an ingredient list image:", type=["jpg", "jpeg", "png", "webp"], key="ing_uploader")
                if ing_up:
                    captured_ing_img = load_image(ing_up)
                elif st.button("🧪 Load Sample Ingredient Image"):
                    sample_path = Path("assets/sample_biscuit_label.jpg")
                    if sample_path.exists():
                        captured_ing_img = Image.open(sample_path)

            if captured_ing_img is not None:
                st.image(captured_ing_img, caption="Captured Ingredient Image", use_container_width=True)
                with st.spinner("Processing image and running OCR..."):
                    pre_ocr = preprocess_for_ocr(captured_ing_img)
                    ocr_res = run_ocr(pre_ocr["contrast_enhanced"])

                if not ocr_res["success"]:
                    st.error("⚠️ Unable to reliably read the ingredient list. Please try again with better focus and lighting.")
                else:
                    st.success(f"✅ OCR Extracted {ocr_res['line_count']} text lines ({ocr_res['confidence']*100:.1f}% confidence)")
                    st.session_state["ing_ocr_raw_text"] = ocr_res["raw_text"]

        with ing_col_right:
            st.subheader("2. Ingredient & Allergen Analysis")

            if "ing_ocr_raw_text" not in st.session_state:
                st.info("👈 Capture or upload an ingredient label on the left to extract ingredients.")
            else:
                raw_ing_ocr = st.session_state["ing_ocr_raw_text"]

                st.markdown("#### ✏️ Verify Extracted Ingredient Text")
                user_ing_text = st.text_area("OCR Ingredient Text (Edit if needed):", value=raw_ing_ocr, height=140)

                if st.button("🔬 Analyze Ingredients & Allergens", key="btn_analyze_ingredients", use_container_width=True):
                    st.session_state["verified_ing_text"] = user_ing_text

                if "verified_ing_text" in st.session_state:
                    active_ing_text = st.session_state["verified_ing_text"]

                    # 1. Parse and categorize ingredients
                    ing_parsed_res = parse_ingredients_pipeline(active_ing_text)
                    st.markdown("### 🧾 Parsed Ingredients List")
                    st.write(f"Total Ingredients Identified: **{ing_parsed_res['total_ingredients_count']}**")

                    ing_list_col, ing_cat_col = st.columns(2)
                    with ing_list_col:
                        st.markdown("**Individual Ingredients:**")
                        for item in ing_parsed_res["ingredients"]:
                            st.markdown(f"• {item}")

                    with ing_cat_col:
                        st.markdown("**Category Classification:**")
                        for item in ing_parsed_res["categorized_items"]:
                            st.markdown(f"{item['icon']} **{item['ingredient']}** — *{item['category']}*")

                    # 2. Detect allergens and cross-contact statements
                    allergen_res = detect_allergens(active_ing_text)
                    st.markdown("---")
                    display_allergen_section(allergen_res)

                    if allergen_res.get("allergen_table"):
                        st.plotly_chart(create_allergen_summary_chart(allergen_res["allergen_table"]), use_container_width=True)

    # --------------------------------------------------------------------------
    # MODE 4: ✍ MANUAL NUTRITION ENTRY
    # --------------------------------------------------------------------------
    elif selected_mode == "✍ Manual Nutrition Entry":
        st.markdown(
            """
            <div style="background:#F3E5F5; border:1px solid #CE93D8; border-radius:8px; padding:12px; margin-bottom:14px;">
                <b>✍ Manual Nutrition Entry Mode</b><br>
                Directly enter nutritional facts from any food package or recipe to calculate the Reference Nutrition Score,
                explain feature contributions, verify % Daily Values, and assess allergen risks.
            </div>
            """,
            unsafe_allow_html=True
        )

        with st.form("manual_entry_form"):
            st.subheader("Nutritional Information Entry")
            m_c1, m_c2, m_c3 = st.columns(3)
            with m_c1:
                man_name = st.text_input("Product / Meal Name:", value="Whole Wheat Bread")
                man_serving = st.text_input("Serving Size:", value="2 slices (56g)")
                man_cal = st.number_input("Calories (kcal):", min_value=0.0, max_value=2500.0, value=140.0, step=5.0)
                man_prot = st.number_input("Protein (g):", min_value=0.0, max_value=150.0, value=6.0, step=0.5)

            with m_c2:
                man_carb = st.number_input("Carbohydrates (g):", min_value=0.0, max_value=300.0, value=24.0, step=1.0)
                man_fat = st.number_input("Total Fat (g):", min_value=0.0, max_value=150.0, value=2.0, step=0.5)
                man_sat_fat = st.number_input("Saturated Fat (g):", min_value=0.0, max_value=100.0, value=0.5, step=0.5)
                man_trans_fat = st.number_input("Trans Fat (g):", min_value=0.0, max_value=50.0, value=0.0, step=0.1)

            with m_c3:
                man_fiber = st.number_input("Dietary Fiber (g):", min_value=0.0, max_value=80.0, value=4.0, step=0.5)
                man_sugar = st.number_input("Total Sugar (g):", min_value=0.0, max_value=200.0, value=3.0, step=0.5)
                man_add_sugar = st.number_input("Added Sugar (g):", min_value=0.0, max_value=200.0, value=1.5, step=0.5)
                man_sodium = st.number_input("Sodium (mg):", min_value=0.0, max_value=5000.0, value=210.0, step=10.0)

            man_ingredients = st.text_area(
                "Optional Ingredient List (for Allergen Detection):",
                value="Whole wheat flour, water, yeast, wheat gluten, brown sugar, soybean oil, salt."
            )

            submit_manual = st.form_submit_button("📊 Calculate Nutrition Score & Explainability", use_container_width=True)

        if submit_manual:
            man_data = {
                "product_name": man_name,
                "serving_size": man_serving,
                "calories": man_cal,
                "protein": man_prot,
                "carbohydrates": man_carb,
                "total_fat": man_fat,
                "saturated_fat": man_sat_fat,
                "trans_fat": man_trans_fat,
                "fiber": man_fiber,
                "total_sugar": man_sugar,
                "added_sugar": man_add_sugar,
                "sodium": man_sodium
            }

            st.markdown(f"### 🍽️ Results for: **{man_name}**")
            display_nutrition_metrics_grid(man_data)

            man_score_res = calculate_nutrition_score(man_data)
            m_score = man_score_res["score"]
            m_tier = man_score_res["health_tier"]
            m_badge = "badge-good" if m_score >= 70 else ("badge-moderate" if m_score >= 45 else "badge-poor")

            st.markdown(
                f"""
                <div style="background:#FFFFFF; border:1px solid #E0E0E0; border-radius:10px; padding:18px; margin: 15px 0;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            <span style="font-size:14px; font-weight:600; color:#616161;">REFERENCE NUTRITION SCORE</span>
                            <div style="font-size:36px; font-weight:800; color:#2E7D32;">{m_score:.1f} <span style="font-size:20px; color:#9E9E9E;">/ 100</span></div>
                        </div>
                        <div class="score-badge {m_badge}">{m_tier}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Explainable AI
            st.markdown("### 🧠 Why did this food receive this score?")
            st.caption("Feature contribution explanation:")
            p_c1, p_c2 = st.columns(2)
            with p_c1:
                st.markdown("**Positive Contributing Factors:**")
                for pf in [f for f in man_score_res["factors"] if f["status"] == "positive"]:
                    st.markdown(f"✓ <span class='tag-positive'>+{pf['impact']} pts</span> **{pf['feature']}** ({pf['value']})<br><small>{pf['reason']}</small>", unsafe_allow_html=True)
            with p_c2:
                st.markdown("**Negative Deductions:**")
                for nf in [f for f in man_score_res["factors"] if f["status"] == "negative"]:
                    st.markdown(f"⚠️ <span class='tag-negative'>{nf['impact']} pts</span> **{nf['feature']}** ({nf['value']})<br><small>{nf['reason']}</small>", unsafe_allow_html=True)

            st.plotly_chart(create_feature_contribution_chart(man_score_res), use_container_width=True)

            # Allergen check
            if man_ingredients.strip():
                man_al = detect_allergens(man_ingredients)
                display_allergen_section(man_al)
                man_al_summary = ", ".join([a["allergen"] for a in man_al.get("allergen_table", [])])
            else:
                man_al_summary = "None entered"

            # Save to SQLite history
            save_analysis(
                food_name=man_name,
                confidence=100.0,
                nutrition_info=man_data,
                nutrition_score=m_score,
                health_tier=m_tier,
                source_type="Manual Nutrition Entry",
                analysis_mode="Manual Nutrition Entry",
                serving_size=man_serving,
                detected_allergens=man_al_summary
            )


# ==============================================================================
# TAB 2: ML MODEL & SHAP LAB
# ==============================================================================
with main_tabs[1]:
    st.subheader("🧠 Machine Learning Model & SHAP Explainer Lab")
    st.markdown(
        """
        This section demonstrates the **Supervised Machine Learning** component of our project.
        A **Random Forest Classifier** is trained on nutritional feature vectors to classify
        foods into 3 health tiers. **SHAP (SHapley Additive exPlanations)** is then applied to compute
        local and global game-theoretic Shapley values.
        """
    )

    ml_col1, ml_col2 = st.columns([1, 1.2])

    with ml_col1:
        st.markdown("#### Model Performance Metrics")
        if xai_model.is_trained:
            m = xai_model.metrics
            pm1, pm2 = st.columns(2)
            pm1.metric("Overall Accuracy", f"{m.get('accuracy', 0)}%")
            pm2.metric("Weighted F1-Score", f"{m.get('f1_score', 0)}%")

            pm3, pm4 = st.columns(2)
            pm3.metric("Weighted Precision", f"{m.get('precision', 0)}%")
            pm4.metric("Weighted Recall", f"{m.get('recall', 0)}%")

            st.caption(f"Evaluated on {m.get('test_samples', 0)} test samples (80/20 train/test split).")

            if st.button("🔄 Retrain Random Forest Model"):
                with st.spinner("Retraining Random Forest Classifier..."):
                    new_metrics = xai_model.train_model()
                    st.success("Model successfully retrained and saved!")
                    st.rerun()

            if "confusion_matrix" in m and "classes" in m:
                st.plotly_chart(
                    create_confusion_matrix_chart(m["confusion_matrix"], m["classes"]),
                    use_container_width=True
                )
        else:
            st.error("ML Model is not currently trained.")

    with ml_col2:
        st.markdown("#### Interactive SHAP Local Feature Explanation")
        st.write("Select a food item to compute real-time SHAP values on its nutrient vector:")
        shap_food_pick = st.selectbox("Select food for SHAP analysis:", options=all_foods, key="shap_food_pick_box")

        if shap_food_pick:
            shap_nut_info = nutrition_db.get_nutrition_info(shap_food_pick)
            if shap_nut_info:
                shap_result = xai_model.explain_with_shap(shap_nut_info)
                if shap_result:
                    st.markdown(
                        f"**Model Prediction**: `{shap_result['predicted_class']}` "
                        f"*(Confidence: {shap_result['confidence']}%)*"
                    )
                    st.plotly_chart(create_shap_waterfall_chart(shap_result), use_container_width=True)

                    st.markdown("##### SHAP Feature Contribution Table")
                    df_shap = pd.DataFrame(shap_result["shap_contributions"])
                    st.dataframe(df_shap, use_container_width=True)
                else:
                    st.warning("SHAP explanation could not be computed for this item.")


# ==============================================================================
# TAB 3: ANALYSIS HISTORY & CSV EXPORT
# ==============================================================================
with main_tabs[2]:
    st.subheader("📜 Recent Analysis History (SQLite)")
    st.markdown("Log of recent nutritional assessments and packaged food scans saved locally in `data/analysis_history.db`:")

    history_df = get_analysis_history(limit=50)

    if history_df.empty:
        st.info("No analysis history recorded yet. Run a scan or food analysis to log entries!")
    else:
        hm1, hm2, hm3, hm4 = st.columns(4)
        hm1.metric("Total Meals / Items Logged", len(history_df))
        hm2.metric("Average Calories", f"{history_df['calories'].mean():.0f} kcal")
        hm3.metric("Average Nutrition Score", f"{history_df['nutrition_score'].mean():.1f} / 100")
        scan_count = len(history_df[history_df["analysis_mode"].str.contains("Scanner", na=False)])
        hm4.metric("Packaged Scans", scan_count)

        st.dataframe(history_df, use_container_width=True)

        h_col1, h_col2 = st.columns([1, 1])
        with h_col1:
            csv_data = history_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Export History to CSV",
                data=csv_data,
                file_name="food_analysis_history.csv",
                mime="text/csv",
                use_container_width=True
            )
        with h_col2:
            if st.button("🗑️ Clear History Database", use_container_width=True):
                clear_analysis_history()
                st.success("History database cleared.")
                st.rerun()


# ==============================================================================
# TAB 4: ARCHITECTURE & VIVA PREPARATION
# ==============================================================================
with main_tabs[3]:
    st.subheader("ℹ️ System Architecture & BSc Project Viva Reference")

    st.markdown(
        """
        ### 🎯 Comprehensive Project Overview
        - **Problem**: Consumers and students struggle to understand packaged food nutritional labels, calculate health trade-offs, and spot allergen risks.
        - **Proposed Multi-Modal Solution**: An end-to-end Explainable AI system that unites:
          1. **Computer Vision**: Real-time camera capture and MobileNetV2 image recognition.
          2. **Image Preprocessing**: CLAHE contrast enhancement, sharpening, noise reduction, and adaptive thresholding to optimize OCR read rates.
          3. **Real OCR**: Local, deep learning Optical Character Recognition (EasyOCR CRAFT + CRNN) running on CPU.
          4. **Controlled Normalization**: Typo correction for character confusion without arbitrary substitution.
          5. **Nutrition & Allergen Parsing**: Structured regex extraction with parenthetical ingredient parsing and precautionary statement isolation.
          6. **Explainable AI (XAI)**:
             - *Deterministic Mathematical Points Attribution*: Auditable factor explanations for individual packaged products.
             - *Statistical Game-Theoretic SHAP*: TreeExplainer on Random Forest Classifier for global and local feature attribution.
          7. **User Verification**: Mandatory verification loop so users inspect and correct OCR output before final scoring.

        ---

        ### ⚙️ Multi-Modal System Pipeline Flow
        1. **Camera Capture**: `st.camera_input()` captures current packaging frame locally in-memory.
        2. **Preprocessing**: Grayscale conversion, CLAHE contrast enhancement, sharpening, and image quality assessment (resolution, blur via Laplacian variance, brightness, contrast).
        3. **OCR Text Extraction**: Local EasyOCR neural network extracts bounding boxes, text lines, and per-line confidence scores.
        4. **Structured Parsing**:
           - Nutrition facts extraction with unit validation and per-100g conversion.
           - Ingredient list splitting preserving nested parentheses.
           - Controlled allergen dictionary matching with exact detected terms and cross-contact statement separation.
        5. **User Verification**: Editable fields allow the user to review and correct any OCR artifacts.
        6. **Scoring & Explainable AI**: Reference Nutrition Score (0–100) calculated with positive boosters (protein, fiber) and penalties (added sugar, saturated fat, trans fat, sodium, calorie density).
        7. **Storage**: SQLite history logging with CSV export.

        ---

        ### 🎓 Viva Voce Key Questions & Answers
        - **Q1: Why is EasyOCR preferred over cloud OCR APIs like Google Vision?**
          *Answer*: EasyOCR runs 100% locally on CPU without sending consumer images over the internet. This preserves user privacy, eliminates recurring API costs, and allows the system to operate offline.
        - **Q2: Why must allergen detection separate ingredients from precautionary statements?**
          *Answer*: A product containing "Soy lecithin" has soy as an active recipe ingredient, whereas "May contain traces of peanuts" indicates potential equipment cross-contact. Conflating the two misleads users about actual recipe formulation.
        - **Q3: Why provide both rule-based scoring and SHAP explainability?**
          *Answer*: Transparent rule-based point attribution provides 100% deterministic auditability for consumer nutrition labels (e.g. exactly -6 points for 4g saturated fat), while SHAP explains the probabilistic machine learning classifier's feature importance across the entire food dataset.
        - **Q4: How are OCR typos safely corrected?**
          *Answer*: We apply controlled regular expressions specifically targeting common digit/letter confusions in nutritional terms (e.g., `Cal0ries -> Calories`, `Pr0tein -> Protein`, `5odium -> Sodium`) without replacing arbitrary words.
        """
    )


# ==============================================================================
# FOOTER & SAFETY DISCLAIMER (Requirement 30)
# ==============================================================================
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #757575; font-size: 12px; padding: 10px 0;">
        <b>Safety & Educational Project Disclaimer:</b> This application is an educational Data Science project. Nutrition values are extracted from product labels and may contain OCR or labeling errors. Allergen detection is based on the text successfully read from the package and is not a guarantee that a product is safe for a person with an allergy. Always verify the original packaging and manufacturer allergen information. This system does not provide medical advice.
    </div>
    """,
    unsafe_allow_html=True
)
