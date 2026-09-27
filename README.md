# 🥗 Explainable AI-Based Real-Time Food and Nutrition Analysis System

**Academic Degree**: Bachelor of Science in Data Science (BSc Data Science) — 3rd-Year Project  
**Author**: Academic Project Portfolio  
**Technology Domain**: Computer Vision, Machine Learning, Explainable AI (XAI), Nutrition Informatics  
**Frameworks**: Python 3.12+ (Python 3.14.x Tested & Verified), Streamlit, ONNX Runtime, Scikit-Learn, SHAP, OpenCV, Plotly  

---

## 1. Project Title
**"Explainable AI-Based Real-Time Food and Nutrition Analysis System"**

---

## 2. Project Overview
This project presents an end-to-end, interpretable data science solution designed to assist users in understanding the nutritional composition of meals directly from food photographs. Traditional nutrition applications often operate as "black boxes" — either providing unverified calorie counts or relying on opaque algorithms without justification. 

This system integrates:
1. **Computer Vision Inference** (Pretrained MobileNetV2 in ONNX format) to identify food categories locally.
2. **Nutritional Data Mining** (USDA FoodData Central-aligned reference profiles) to retrieve macro- and micronutrients.
3. **Transparent Mathematical Nutrition Scoring** (0–100 Reference Score) with documented feature weights.
4. **Explainable AI (XAI)** utilizing both **deterministic factor attribution** and **Shapley Additive exPlanations (SHAP)** applied to a trained supervised machine learning classifier.
5. **Interactive Academic Dashboard** built with Streamlit and Plotly for real-time visualization, dietary recommendations, and persistent SQLite analysis history.

---

## 3. Problem Statement
Diet-related health conditions (such as obesity, cardiovascular disease, and type 2 diabetes) are closely tied to daily nutritional intake. However, individuals routinely face two major hurdles:
- **Nutritional Illiteracy**: Evaluating calories, macronutrient balance, and hidden sugars from visual meal portions is cognitively difficult.
- **Black-Box AI Skepticism**: Modern deep learning apps rarely justify why an item was classified as healthy or unhealthy, leading to mistrust among health professionals and end-users.

**Solution**: A lightweight, locally runnable, and explainable AI system that not only detects food and estimates nutritional metrics, but also transparently explains the exact positive and negative factors driving its health assessment.

---

## 4. Objectives
- **Computer Vision**: Implement a real-time, local food classification pipeline without reliance on external paid APIs or cloud keys.
- **Data Engineering**: Compile and curate an 86-item nutritional reference database with alias resolution and token-matching fallbacks.
- **Explainable AI (XAI)**: Demystify the nutrition scoring mechanism through dual-level interpretability (rule-based point attribution and game-theoretic SHAP values).
- **Machine Learning**: Train and evaluate a multi-class Random Forest model for dietary quality tiering with classification metrics (Precision, Recall, F1, Confusion Matrix).
- **Usability & Fallback**: Provide human-in-the-loop manual overrides when image confidence is ambiguous.
- **Academic Rigor**: Create modular, unit-tested Python code suitable for undergraduate viva presentation.

---

## 5. Key Features & Operational Modes

The system operates across **4 distinct, fully interactive application modes**:

1. 🍎 **Food Image Analysis**:
   - Deep learning computer vision inference using MobileNetV2 ONNX.
   - USDA FoodData Central-aligned reference profiles for 86 culinary entities.
   - Top-3 candidate ranking with configurable confidence thresholds and manual fallback.
   - Transparent 0–100 Reference Nutrition Score with deterministic point attribution and SHAP TreeExplainer.

2. 📦 **Real-Time Food Label Scanner**:
   - Camera input (`st.camera_input`) capturing packaged food nutrition tables in real-time.
   - Computer vision preprocessing pipeline: RGB conversion, CLAHE contrast enhancement, sharpening, noise reduction, and Otsu/Adaptive Gaussian thresholding.
   - Image quality diagnostics: automated resolution, blur (Laplacian variance), brightness, and contrast checks with real-time feedback.
   - Local deep learning OCR (EasyOCR CRAFT + CRNN) running 100% on CPU.
   - Controlled OCR typo normalization (`Cal0ries -> Calories`, `Pr0tein -> Protein`, `5odium -> Sodium`).
   - Structured nutrition extraction: Calories, Protein, Carbohydrates, Total Fat, Saturated Fat, Trans Fat, Dietary Fiber, Total Sugar, Added Sugar, Sodium, and Serving Size.
   - Human-in-the-loop verification form allowing user inspection and correction of scanned fields.
   - Automatic Per-Serving vs. Per-100g conversion calculation.
   - Packaged food scoring deductions for saturated fat, trans fat, added sugar, and sodium.
   - Allergen detection and isolation of cross-contact / precautionary statements.

3. 🧾 **Ingredient Scanner**:
   - Camera or upload capture of ingredient panels.
   - Parenthetical clause parsing (handling sub-ingredients cleanly).
   - Neutral classification into food science categories (Grains, Added sugars, Oils/fats, Dairy, Soy, Nuts, Emulsifiers, Preservatives, Flavouring).
   - Configurable allergen detection matching 11 major allergen families with exact detected terms.
   - Clear separation between actual recipe ingredients and precautionary ("May contain traces of...") facility warnings.

4. ✍ **Manual Nutrition Entry**:
   - Direct numerical input for any packaged product or custom meal.
   - Immediate mathematical scoring, XAI factor breakdowns, and allergen risk checking.

---

## 6. Multi-Modal System Architecture

```text
                                  ┌──────────────────────────────┐
                                  │      Streamlit Dashboard     │
                                  └──────────────┬───────────────┘
                                                 │
                   ┌─────────────────────────────┴─────────────────────────────┐
                   ▼                                                           ▼
       [ 🍎 Food Image Mode ]                                      [ 📦 Label / Ingredient Scanner ]
                   │                                                           │
                   ▼                                                           ▼
      [ MobileNetV2 ONNX Model ]                                  [ CV Preprocessing: CLAHE/Otsu ]
                   │                                                           │
                   ▼                                                           ▼
      [ USDA Database Mapping ]                                   [ Local EasyOCR (CRAFT + CRNN) ]
                   │                                                           │
                   │                                                           ▼
                   │                                              [ Controlled OCR Normalizer ]
                   │                                                           │
                   │                                                           ▼
                   │                                              [ Nutrition & Allergen Parser ]
                   │                                                           │
                   │                                                           ▼
                   │                                              [ Mandatory User Verification ]
                   │                                                           │
                   └─────────────────────────────┬─────────────────────────────┘
                                                 │
                                                 ▼
                             [ Transparent 0–100 Reference Score ]
                             (Fiber, Protein, Sugar, Sat Fat, Sodium)
                                                 │
                                                 ▼
                             [ Explainable AI (XAI) Attribution ]
                             (Deterministic Factor Points + SHAP)
                                                 │
                                                 ▼
                                     [ Allergen Warning System ]
                                  (Ingredients vs Precautionary)
                                                 │
                                                 ▼
                                [ Plotly Interactive Visualizations ]
                                                 │
                                                 ▼
                               [ SQLite Storage (analysis_history.db) ]
```

---

## 7. Technology Stack
- **Language**: Python 3.12+ (Python 3.14.4 fully verified)
- **Frontend / Dashboard**: Streamlit
- **Optical Character Recognition (OCR)**: EasyOCR (Local CRAFT + CRNN neural networks)
- **Computer Vision Inference**: ONNX Runtime (CPU Provider, MobileNetV2)
- **Image Processing**: Pillow, OpenCV (`cv2`)
- **Data Manipulation**: Pandas, NumPy
- **Machine Learning**: Scikit-Learn (`RandomForestClassifier`)
- **Explainable AI**: SHAP (`TreeExplainer`)
- **Visualizations**: Plotly Express & Plotly Graph Objects
- **Testing**: Pytest (21 automated unit tests)
- **Database**: SQLite (Local embedded storage with CSV export)

- **Interactive Visualizations**: Plotly Express & Graph Objects, Matplotlib
- **Database**: SQLite3 (Standard Library)
- **Unit Testing**: Pytest

---

## 8. Project Structure

```text
Nutrition Project/
│
├── app.py                      # Main Streamlit academic dashboard application
├── config.py                   # Central settings, paths, daily values, and weights
├── requirements.txt            # Minimal verified Python dependencies
├── README.md                   # Comprehensive project documentation & viva guide
├── .gitignore                  # Git ignore rules for caches, venvs, and databases
│
├── data/
│   ├── nutrition_data.csv      # 86-item USDA reference nutritional dataset
│   ├── food_aliases.csv        # 160+ aliases and culinary variation mappings
│   └── analysis_history.db     # SQLite database for storing past analyses (auto-created)
│
├── models/
│   ├── README.md               # Model architecture and download notes
│   ├── mobilenetv2-7.onnx      # 14MB pretrained computer vision classification model
│   ├── imagenet_classes.txt    # 1,000 ImageNet category labels
│   └── nutrition_classifier.joblib  # Trained Random Forest ML model & metrics
│
├── src/
│   ├── __init__.py             # Package marker
│   ├── image_processing.py     # Image loading, EXIF correction, resizing, normalization
│   ├── food_recognition.py     # ONNX vision inference, food mapping, fallback mode
│   ├── nutrition.py            # Dataset querying, alias matching, Daily Value calculation
│   ├── scoring.py              # Transparent 0-100 scoring algorithm & factor impacts
│   ├── explainability.py       # ML classifier training, SHAP explainer, recommendations
│   ├── visualization.py        # Publication-grade interactive Plotly charts
│   └── utils.py                # SQLite history CRUD, image validation, download helpers
│
├── assets/
│   ├── README.md               # Assets and sample test image documentation
│   ├── sample_apple.jpg        # Test image: Fresh red apple
│   ├── sample_pizza.jpg        # Test image: Cheese & tomato pizza slice
│   └── sample_broccoli.jpg     # Test image: Fresh broccoli florets
│
└── tests/
    ├── test_nutrition.py       # Unit tests for dataset loading, lookup & aliases
    ├── test_scoring.py         # Unit tests for scoring bounds [0, 100] & factors
    └── test_utils.py           # Unit tests for SQLite history & image preprocessing
```

---

## 9. Installation & Setup Instructions (Windows PowerShell)

### Step 1: Check Python Version
Open Windows PowerShell inside VS Code (`Ctrl + ~`) and verify your Python version:
```powershell
python --version
```
*(This project is fully verified on Python 3.12, 3.13, and **Python 3.14.4**).*

### Step 2: Create a Virtual Environment (Recommended)
```powershell
python -m venv venv
```

### Step 3: Install Dependencies
If PowerShell execution policies restrict script activation, install directly using the virtual environment's Python executable:
```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```
*Or, if using your system Python:*
```powershell
python -m pip install -r requirements.txt
```

---

## 10. How to Run the Application

Execute Streamlit using the virtual environment Python:
```powershell
.\venv\Scripts\python.exe -m streamlit run app.py
```
*Or with system Python:*
```powershell
python -m streamlit run app.py
```

### Accessing the Dashboard:
Once started, open your web browser at:
👉 **`http://localhost:8501`**

---

## 11. Testing the Application with Food Images

### Method A: Use Included Sample Images
Three high-resolution sample images are pre-packaged in the `assets/` directory:
1. In the sidebar, select **"📸 Upload Food Image (AI Vision)"**.
2. Click **"Browse files"** and select:
   - `assets/sample_apple.jpg` ➔ AI identifies **Apple** (93.2% confidence).
   - `assets/sample_pizza.jpg` ➔ AI identifies **Pizza** (18.7% confidence).
   - `assets/sample_broccoli.jpg` ➔ AI identifies **Broccoli** (86.4% confidence).
3. The dashboard will automatically extract the nutritional breakdown, compute the Reference Nutrition Score, and render interactive SHAP and feature contribution charts.

### Method B: Use Quick Test Buttons
If you do not have an image handy, simply click any of the quick-test buttons in the UI:
`🍎 Apple`, `🍕 Pizza`, `🥦 Broccoli`, or `🍔 Burger`.

### Method C: Manual Search Mode
Select **"🔍 Manual Food Search"** in the sidebar to directly look up any of the 86 items in the database.

---

## 12. Nutrition Scoring Methodology

The Reference Nutrition Score is a transparent mathematical index ranging strictly between **0.0 and 100.0**:

$$\text{Score} = \text{clamp}_{[0, 100]}\Big(\text{Baseline} + \text{Bonus}_{\text{fiber}} + \text{Bonus}_{\text{protein}} + \text{Bonus}_{\text{balance}} - \text{Penalty}_{\text{sugar}} - \text{Penalty}_{\text{fat}} - \text{Penalty}_{\text{calories}}\Big)$$

### Mathematical Parameter Breakdown:
- **Baseline Score**: Starts at a neutral benchmark of **50.0 points**.
- **Dietary Fiber Bonus**: Up to **+18.0 points** ($\text{Fiber} \times 2.5$), promoting digestive health and glycemic control.
- **Protein Bonus**: Up to **+14.0 points** ($\text{Protein} \times 1.0$), rewarding muscle repair and satiety.
- **Macro Balance Bonus**: **+5.0 points** if protein $\ge 3\text{g}$, fiber $\ge 2\text{g}$, and sugar $\le 10\text{g}$.
- **Sugar Penalty**: Up to **-22.0 points** applied if sugar exceeds 5g per serving.
- **Total Fat Penalty**: Up to **-18.0 points** applied if fat exceeds 3g per serving.
- **Calorie Density Penalty**: Up to **-15.0 points** applied if calories exceed 250 kcal per serving.

### Health Tiers:
- 🟢 **Good / Nutrient-Rich**: Score $\ge 70.0$
- 🟡 **Moderate / Balanced**: Score $45.0 - 69.9$
- 🔴 **Needs Improvement / Calorie-Dense**: Score $< 45.0$

---

## 13. Explainable AI (XAI) Methodology

Explainability is a core pillar of this project:
1. **Deterministic Attribution**: Each factor displayed to the user indicates the exact mathematical adjustment (e.g. `+12.8 pts for Dietary Fiber`, `-15.2 pts for Elevated Sugar`).
2. **Game-Theoretic SHAP (TreeExplainer)**:
   - A multi-class **Random Forest Classifier** is trained on the USDA dataset features (`calories`, `protein`, `carbohydrates`, `fat`, `fiber`, `sugar`).
   - The SHAP explainer calculates **Shapley values** for any given food vector:
     $$\phi_i(v) = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(|N|-|S|-1)!}{|N|!} \big(v(S \cup \{i\}) - v(S)\big)$$
   - This measures the marginal contribution of each nutrient to the log-odds of assigning the food to its predicted health tier.
   - The UI displays an interactive SHAP waterfall and horizontal bar chart.

---

## 14. Running Unit Tests

Run the full automated test suite using `pytest`:
```powershell
python -m pytest tests/ -v
```
**Expected Output**: `11 passed in <1.0s` (covering dataset integrity, alias matching, scoring boundaries, boundary clamps, and SQLite history).

---

## 15. Limitations
- **2D Image Estimation**: A standard photograph cannot determine hidden ingredients (e.g., butter, salt, dressings) or deep-frying oil uptake.
- **Volumetric Approximation**: Without depth cameras (LiDAR) or 3D bounding boxes, precise serving sizes cannot be inferred from a single photograph; reference standard servings are utilized.
- **Culinary Variation**: Nutritional data represents USDA baseline averages; actual nutrient profiles vary by brand, preparation method, and recipe.

---

## 16. Future Scope
- Integration with depth sensors or stereo cameras for automated volumetric portion calculation.
- Barcode and QR code scanning for packaged branded products.
- User profile personalization (e.g., custom daily calorie and macronutrient targets for diabetic or renal diets).
- Mobile application deployment using Flutter or React Native with ONNX mobile runtimes.

---

## 17. Disclaimer
> **Academic Project Disclaimer**: This software is an educational Data Science project. Nutritional values are reference estimates derived from public USDA datasets and may vary significantly by preparation, portion size, and brand. The system does not provide medical advice, diagnosis, or personalized dietary treatment.
