# 📦 Olist Delivery Risk Intelligence

An end-to-end e-commerce machine learning and logistics intelligence system built from the Brazilian Olist dataset. It predicts whether a delivered order will arrive past its promised delivery date, uncovers critical logistics bottlenecks via comprehensive Exploratory Data Analysis (EDA), extracts native model risk drivers, and prioritizes orders via an interactive Streamlit operational dashboard.

---

## 🚀 Key Features & Capabilities

1. **Data Ingestion & Hygiene Audit**: Automated validation of 8 relational tables (row counts, duplicates, missingness rates).
2. **Deep-Dive Exploratory Data Analysis (EDA)**:
   - **Interstate Logistics Disparity**: Pinpoints that cross-state shipments incur a **+53% late delivery risk premium** (9.27% vs 6.06% for intrastate).
   - **Customer Experience Penalty**: Proves that late deliveries crash average review ratings from **4.29 ★ to 2.57 ★**, with 54.0% yielding negative 1- or 2-star reviews.
   - **Cargo & Pricing Dynamics**: Analyzes product weight, cubic volume, freight-to-price ratio, and category-specific lateness.
   - **Temporal & Seasonal Bottlenecks**: Evaluates monthly surge patterns (e.g., Black Friday & courier disruptions) and day-of-week trends.
3. **Advanced Feature Engineering**:
   - Order-level timeline milestones (`approval_delay_hours`, `carrier_handoff_hours`, `estimated_transit_days`, `shipping_buffer_days`).
   - Cross-border route classification (`is_interstate = customer_state != seller_state`).
   - Cargo dimensions (`avg_product_volume_cm3`, `avg_product_weight_g`, `freight_ratio`).
   - Strict target leakage safeguards: zero post-delivery outcomes used as predictors.
4. **Machine Learning Model Benchmark**:
   - Compares **XGBoost Classifier**, **Random Forest**, and **Logistic Regression**.
   - Evaluated out-of-sample using **ROC-AUC (0.840)**, **PR-AUC (0.449)**, Precision, Recall, and F1 scores.
5. **Native Interpretability (Zero-SHAP)**:
   - Fast, deterministic model interpretability leveraging native tree Gini/Gain and regression coefficients.
   - No heavy, slow, or brittle SHAP dependencies.
   - Outputs ranked risk drivers (`reports/feature_importance.csv` and `feature_importance.png`).
6. **Modern, Simple Streamlit Dashboard (`app.py`)**:
   - **Executive Overview**: High-level KPI cards, delivery variance distribution, and milestone waterfall.
   - **Deep-Dive EDA**: Interactive visualizations of geographic routes, review score impact, cargo characteristics, and temporal trends.
   - **Operational Risk Queue**: Interactive threshold slider, workload capacity tuning, search filters, and one-click CSV export.
   - **Live Risk Simulator**: Interactive "What-If" order scorer for operations and dispatch managers.
   - **Model & Drivers**: Out-of-sample benchmark comparison and native risk driver rankings.
   - **Data Quality**: Ingestion hygiene and leakage safeguard audit.
7. **Production SQL Suite**:
   - PostgreSQL schema (`sql/schema.sql`) and analytical queries (`sql/analysis.sql`).

---

## 📊 Quick Performance Summary

| Algorithm | ROC-AUC | PR-AUC | Precision | Recall | F1 Score | Test Samples |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost Classifier** (Champion) | **0.8404** | **0.4494** | 0.2418 | **0.7201** | **0.3621** | 19,296 |
| **Random Forest** | 0.8273 | 0.4380 | **0.5728** | 0.2690 | 0.3661 | 19,296 |
| **Logistic Regression** | 0.7711 | 0.3269 | 0.1777 | 0.6799 | 0.2817 | 19,296 |

> **Key Findings:**
> - Destination State (`customer_state`) and Cross-State Route (`is_interstate`) are the two strongest predictors of delivery delay.
> - Warehouse Handoff (`carrier_handoff_hours`) is the primary lever under merchant control — reducing dispatch time directly cuts delivery risk.

---

## 🛠️ Project Structure

```text
olist_delivery_risk_project/
├── app.py                      # Modern, Simple Streamlit Operational App
├── requirements.txt            # Python dependencies (Zero SHAP)
├── README.md                   # Project documentation & runbook
├── data/
│   ├── raw/                    # Raw Olist CSV datasets
│   └── processed_orders.csv    # Engineered order-level dataset
├── src/
│   ├── __init__.py
│   ├── pipeline.py             # Feature engineering & ML training pipeline
│   ├── eda.py                  # Comprehensive Exploratory Data Analysis suite
│   └── explain.py              # Native feature importance & interpretability
├── sql/
│   ├── schema.sql              # PostgreSQL DDL schema
│   └── analysis.sql            # Analytical & operational SQL queries
├── artifacts/                  # Persisted models & metadata
│   ├── model_metadata.json
│   ├── xgboost.joblib
│   ├── random_forest.joblib
│   └── logistic_regression.joblib
└── reports/                    # Generated EDA charts, metrics, and risk queue
    ├── eda_summary.json
    ├── eda_state_risk.png
    ├── eda_review_impact.png
    ├── eda_category_risk.png
    ├── feature_importance.png
    ├── feature_importance.csv
    ├── model_metrics.csv
    ├── threshold_analysis.csv
    ├── test_predictions.csv
    └── data_quality_summary.csv
```

---

## ⚡ Setup & Execution

### 1. Environment Setup

```bash
# Create and activate virtual environment
python -m venv .venv

# Windows:
.venv\Scripts\activate

# macOS / Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Feature Engineering & Model Training

```bash
python -m src.pipeline
```
Outputs:
- Ingestion data quality audit (`reports/data_quality_summary.csv`)
- Engineered dataset (`data/processed_orders.csv`)
- Model artifacts (`artifacts/*.joblib` and `artifacts/model_metadata.json`)
- Benchmark comparison (`reports/model_metrics.csv`)
- Threshold operational sensitivity (`reports/threshold_analysis.csv`)
- Ranked out-of-sample test predictions (`reports/test_predictions.csv`)

### 3. Run Deep-Dive EDA

```bash
python -m src.eda
```
Generates comprehensive analysis reports and charts in `reports/`:
- `reports/eda_summary.json`
- `reports/eda_state_risk.png`
- `reports/eda_review_impact.png`
- `reports/eda_category_risk.png`
- `reports/late_rate_by_month.png`

### 4. Extract Native Risk Drivers & Interpretability

```bash
python -m src.explain
```
Extracts native Gini/Gain and regression coefficients without any SHAP dependency:
- `reports/feature_importance.csv`
- `reports/feature_importance.png`

### 5. Launch the Streamlit Operational Dashboard

```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🔒 Target Definition & Anti-Leakage Safeguards

- **Supervised Target**: `is_late = 1` if `order_delivered_customer_date > order_estimated_delivery_date`.
- **Exclusion of Undelivered Orders**: Training is restricted to completed deliveries with validated timestamps.
- **Leakage Prevention**:
  - Delivery delay days and final delivery timestamps are strictly omitted from model features.
  - Customer review scores/comments are post-delivery outcomes and are used strictly in EDA, never during model training.
- **Decision Support**: Predictions are probabilistic risk scores designed to empower logistics managers to prioritize warehouse dispatch and customer notifications.
