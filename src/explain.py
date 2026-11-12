"""
Model Interpretability and Feature Importance Engine.
Replaces heavy and brittle SHAP dependencies with fast, deterministic,
native model interpretability:
- Tree-based Gini & Gain Importance for XGBoost and Random Forest
- Standardized Coefficients for Logistic Regression
- High-level category grouping and business impact annotations
"""

import os
import tempfile
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
REP = ROOT / "reports"

FEATURE_DESCRIPTIONS = {
    "carrier_handoff_hours": ("Logistics Bottleneck", "Hours between order approval and carrier pickup"),
    "estimated_transit_days": ("Logistics Schedule", "Days scheduled for carrier to deliver"),
    "shipping_buffer_days": ("Seller Buffer", "Days between estimated delivery and seller limit"),
    "approval_delay_hours": ("Payment Processing", "Hours to verify payment and approve order"),
    "is_interstate": ("Geography & Route", "Customer and seller are in different states (1=Yes, 0=No)"),
    "customer_state": ("Geography & Route", "Destination Brazilian state"),
    "freight_ratio": ("Cargo & Pricing", "Ratio of shipping freight cost to product price"),
    "total_freight": ("Cargo & Pricing", "Total freight fee in Brazilian Reais (R$)"),
    "total_price": ("Order Value", "Total merchandise value in Brazilian Reais (R$)"),
    "avg_product_weight_g": ("Cargo & Packaging", "Average item weight in grams"),
    "avg_product_volume_cm3": ("Cargo & Packaging", "Average item cubic volume in cm³"),
    "item_count": ("Order Complexity", "Total number of items in the order"),
    "unique_sellers": ("Order Complexity", "Number of distinct dispatching sellers"),
    "purchase_month": ("Seasonality", "Month of purchase (peaks during holidays / strikes)"),
    "purchase_dayofweek": ("Temporal", "Day of week order was placed (0=Mon, 6=Sun)"),
    "purchase_hour": ("Temporal", "Hour of day order was placed (0-23)"),
    "is_weekend_purchase": ("Temporal", "Weekend order placed on Saturday or Sunday"),
    "payment_installments_max": ("Payment", "Maximum credit installments used"),
    "payment_total": ("Order Value", "Total amount charged to customer"),
    "payment_methods": ("Payment", "Distinct payment instruments used"),
    "category_count": ("Order Complexity", "Distinct product categories in order")
}

def get_feature_group(col_name: str) -> str:
    """Classify feature into a cohesive operational domain."""
    for key, (group, _) in FEATURE_DESCRIPTIONS.items():
        if col_name.startswith(key):
            return group
    if "state" in col_name.lower():
        return "Geography & Route"
    return "Other / Metadata"

def extract_feature_names(preprocessor, numeric_cols, categorical_cols):
    """Retrieve full decoded column names after one-hot encoding."""
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_names = list(cat_encoder.get_feature_names_out(categorical_cols))
    return numeric_cols + cat_names

def generate_feature_importance():
    meta_path = ART / "model_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError("model_metadata.json not found. Run pipeline first.")

    meta = json.load(open(meta_path))
    model_name = meta["best_model"]
    model_path = ART / f"{model_name}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model {model_path} not found.")

    pipe = joblib.load(model_path)
    pre = pipe.named_steps["preprocess"]
    clf = pipe.named_steps["model"]

    all_features = extract_feature_names(pre, meta["numeric_features"], meta["categorical_features"])

    if hasattr(clf, "feature_importances_"):
        raw_importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        raw_importances = np.abs(clf.coef_[0])
    else:
        raise AttributeError("Classifier has neither feature_importances_ nor coef_")

    df_imp = pd.DataFrame({
        "feature": all_features,
        "importance_score": raw_importances
    })

    # Normalize to 100%
    total_score = df_imp["importance_score"].sum()
    df_imp["relative_importance_pct"] = np.round(df_imp["importance_score"] / total_score * 100, 2)
    df_imp["domain_group"] = df_imp["feature"].apply(get_feature_group)
    df_imp = df_imp.sort_values("importance_score", ascending=False).reset_index(drop=True)

    # Save CSV
    df_imp.to_csv(REP / "feature_importance.csv", index=False)

    # Generate Matplotlib Chart
    top15 = df_imp.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=150)
    colors = ["#2b5c8f" if "customer_state" not in f else "#e07a5f" for f in top15["feature"]]
    bars = ax.barh(top15["feature"], top15["relative_importance_pct"], color=colors, height=0.65, edgecolor="none")

    ax.set_title(f"Top 15 Late Delivery Risk Drivers ({model_name.replace('_', ' ').title()})",
                 fontsize=13, fontweight="bold", pad=14, color="#1e293b")
    ax.set_xlabel("Relative Importance Contribution (%)", fontsize=11, fontweight="medium", labelpad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.grid(axis="x", linestyle="--", alpha=0.35)

    # Annotate percentage labels
    for bar in bars:
        w = bar.get_width()
        ax.annotate(f" {w:.1f}%",
                    xy=(w, bar.get_y() + bar.get_height() / 2),
                    xytext=(3, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=9, color="#334155", fontweight="bold")

    plt.tight_layout()
    plot_path = REP / "feature_importance.png"
    plt.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close()

    print(f"Feature importance saved to: {REP / 'feature_importance.csv'}")
    print(f"Feature importance plot saved to: {plot_path}")
    print("\nTop 10 Risk Drivers:")
    print(df_imp[["feature", "relative_importance_pct", "domain_group"]].head(10).to_string(index=False))

    return df_imp

if __name__ == "__main__":
    generate_feature_importance()
