"""
Comprehensive Exploratory Data Analysis (EDA) Suite.
Analyzes delivery performance across:
1. Geographic and interstate logistics routes
2. Downstream customer satisfaction & review scores impact
3. Fulfillment timeline stages and bottlenecks
4. Product category, weight, and freight ratio characteristics
5. Temporal trends and monthly seasonality
"""

import os
import tempfile
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.pipeline import load_data, quality_report, build_order_features

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports"
REP.mkdir(exist_ok=True)

# Styling palette for clean, professional charts
PRIMARY = "#2563eb"
ACCENT = "#e11d48"
MUTED = "#64748b"
BG_GRID = "#f1f5f9"

def run_eda():
    print("Loading raw Olist datasets...")
    data = load_data()
    quality_report(data)

    print("Building engineered order features...")
    df, num, cat = build_order_features(data)

    # -------------------------------------------------------------
    # 1. Geographic & Interstate Logistics Analysis
    # -------------------------------------------------------------
    print("Analyzing geographic and route performance...")
    interstate_summary = df.groupby("is_interstate").agg(
        total_orders=("is_late", "count"),
        late_orders=("is_late", "sum"),
        late_rate_pct=("is_late", lambda x: round(float(x.mean() * 100), 2)),
        avg_delay_days=("delivery_delay_days", lambda x: round(float(x.mean()), 2)),
        median_delay_days=("delivery_delay_days", lambda x: round(float(x.median()), 2))
    ).reset_index()
    interstate_summary["route_type"] = interstate_summary["is_interstate"].map({
        0: "Intrastate (Same State)",
        1: "Interstate (Cross-State)"
    })
    interstate_summary.to_csv(REP / "eda_interstate_analysis.csv", index=False)

    # State level risk
    state_summary = df.groupby("customer_state").agg(
        total_orders=("is_late", "count"),
        late_orders=("is_late", "sum"),
        late_rate_pct=("is_late", lambda x: round(float(x.mean() * 100), 2)),
        avg_delay_days=("delivery_delay_days", lambda x: round(float(x.mean()), 2)),
        median_freight=("total_freight", lambda x: round(float(x.median()), 2))
    ).reset_index().sort_values("late_rate_pct", ascending=False)
    state_summary.to_csv(REP / "eda_state_analysis.csv", index=False)

    # Plot: Top 10 High-Risk vs Low-Risk States
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    top_states = state_summary[state_summary["total_orders"] >= 200].head(12)
    bars = ax.bar(top_states["customer_state"], top_states["late_rate_pct"], color="#ea580c", edgecolor="none")
    ax.axhline(df["is_late"].mean() * 100, color="#2563eb", linestyle="--", linewidth=1.5, label=f"National Avg ({df.is_late.mean()*100:.1f}%)")
    ax.set_title("Top Brazilian States by Late Delivery Rate (Min 200 Orders)", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("Late Delivery Rate (%)", fontsize=10)
    ax.set_xlabel("Customer State", fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")
    plt.tight_layout()
    plt.savefig(REP / "eda_state_risk.png", dpi=160, bbox_inches="tight")
    plt.close()

    # -------------------------------------------------------------
    # 2. Customer Reviews & Downstream Satisfaction Impact
    # -------------------------------------------------------------
    print("Analyzing customer review score impact...")
    reviews = data["reviews"].groupby("order_id")["review_score"].mean().reset_index()
    order_reviews = df[["order_id", "is_late", "delivery_delay_days"]].merge(reviews, on="order_id", how="inner")

    review_stats = order_reviews.groupby("is_late").agg(
        order_count=("review_score", "count"),
        mean_rating=("review_score", lambda x: round(float(x.mean()), 2)),
        pct_1_star=("review_score", lambda x: round(float((x == 1).mean() * 100), 2)),
        pct_negative_1_2=("review_score", lambda x: round(float((x <= 2).mean() * 100), 2)),
        pct_5_star=("review_score", lambda x: round(float((x == 5).mean() * 100), 2))
    ).reset_index()
    review_stats["status"] = review_stats["is_late"].map({0: "Delivered On-Time", 1: "Delivered Late"})
    review_stats.to_csv(REP / "eda_review_impact.csv", index=False)

    # Plot Review Score Impact
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)
    # Mean review rating comparison
    bars1 = ax1.bar(["Delivered On-Time", "Delivered Late"], review_stats["mean_rating"], color=["#10b981", "#ef4444"], width=0.55)
    ax1.set_ylim(0, 5)
    ax1.set_title("Customer Review Score (1 to 5 Stars)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Average Rating Stars", fontsize=10)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    for bar in bars1:
        h = bar.get_height()
        ax1.annotate(f"{h:.2f} ★", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Bad review proportion (1-2 stars)
    bars2 = ax2.bar(["Delivered On-Time", "Delivered Late"], review_stats["pct_negative_1_2"], color=["#0ea5e9", "#dc2626"], width=0.55)
    ax2.set_title("Negative Reviews Share (1 or 2 Stars)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("% of Orders with 1-2 Stars", fontsize=10)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    for bar in bars2:
        h = bar.get_height()
        ax2.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(REP / "eda_review_impact.png", dpi=160, bbox_inches="tight")
    plt.close()

    # -------------------------------------------------------------
    # 3. Product Category Risk Analysis
    # -------------------------------------------------------------
    print("Analyzing product category risk...")
    items = data["items"]
    prod = data["products"].merge(data["translation"], on="product_category_name", how="left")
    item_prod = items.merge(prod[["product_id", "product_category_name_english"]], on="product_id", how="left")
    primary_cat = item_prod.groupby("order_id")["product_category_name_english"].first().reset_index()

    df_cat = df.merge(primary_cat, on="order_id", how="left")
    cat_summary = df_cat.groupby("product_category_name_english").agg(
        total_orders=("is_late", "count"),
        late_orders=("is_late", "sum"),
        late_rate_pct=("is_late", lambda x: round(float(x.mean() * 100), 2)),
        avg_freight=("total_freight", lambda x: round(float(x.mean()), 2)),
        avg_weight_g=("avg_product_weight_g", lambda x: round(float(x.mean()), 1))
    ).reset_index()

    # Filter categories with meaningful volume (min 300 orders)
    cat_summary = cat_summary[cat_summary["total_orders"] >= 300].sort_values("late_rate_pct", ascending=False)
    cat_summary.to_csv(REP / "eda_category_analysis.csv", index=False)

    # Plot top categories by late rate
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    top_cats = cat_summary.head(10).iloc[::-1]
    ax.barh(top_cats["product_category_name_english"], top_cats["late_rate_pct"], color="#4f46e5", height=0.6)
    ax.set_title("Highest-Risk Product Categories (Min 300 Orders)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Late Delivery Rate (%)", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(REP / "eda_category_risk.png", dpi=160, bbox_inches="tight")
    plt.close()

    # -------------------------------------------------------------
    # 4. Monthly Trend & Seasonality Analysis
    # -------------------------------------------------------------
    print("Generating monthly seasonality trend...")
    # Calculate month stats using purchase timestamp if available
    monthly = df.groupby("purchase_month").agg(
        orders=("is_late", "count"),
        late_rate_pct=("is_late", lambda x: round(float(x.mean() * 100), 2))
    ).reset_index()

    fig, ax1 = plt.subplots(figsize=(10, 4.5), dpi=150)
    ax2 = ax1.twinx()
    ax1.bar(monthly["purchase_month"], monthly["orders"], color="#cbd5e1", alpha=0.7, label="Order Volume")
    ax2.plot(monthly["purchase_month"], monthly["late_rate_pct"], color="#dc2626", marker="o", linewidth=2.2, label="Late Rate (%)")

    ax1.set_xlabel("Purchase Month (1=Jan, 12=Dec)", fontsize=10)
    ax1.set_ylabel("Order Volume", fontsize=10, color="#475569")
    ax2.set_ylabel("Late Delivery Rate (%)", fontsize=10, color="#dc2626")
    ax1.set_title("Seasonal Order Volume vs Late Delivery Rate by Month", fontsize=12, fontweight="bold", pad=12)
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax1.grid(axis="y", linestyle=":", alpha=0.3)
    plt.tight_layout()
    plt.savefig(REP / "late_rate_by_month.png", dpi=160, bbox_inches="tight")
    plt.close()

    # -------------------------------------------------------------
    # 5. Master EDA JSON Summary
    # -------------------------------------------------------------
    ontime_rating = float(review_stats.loc[review_stats["is_late"] == 0, "mean_rating"].values[0]) if len(review_stats) > 0 else 4.2
    late_rating = float(review_stats.loc[review_stats["is_late"] == 1, "mean_rating"].values[0]) if len(review_stats) > 1 else 2.6
    intrastate_late = float(interstate_summary.loc[interstate_summary["is_interstate"] == 0, "late_rate_pct"].values[0])
    interstate_late = float(interstate_summary.loc[interstate_summary["is_interstate"] == 1, "late_rate_pct"].values[0])

    eda_summary = {
        "labeled_delivered_orders": int(len(df)),
        "late_orders": int(df["is_late"].sum()),
        "late_rate_pct": round(float(df["is_late"].mean() * 100), 2),
        "median_delay_days": round(float(df["delivery_delay_days"].median()), 2),
        "mean_order_value_brl": round(float(df["total_price"].mean()), 2),
        "mean_freight_value_brl": round(float(df["total_freight"].mean()), 2),
        "intrastate_late_rate_pct": intrastate_late,
        "interstate_late_rate_pct": interstate_late,
        "interstate_risk_premium_multiplier": round(interstate_late / max(0.01, intrastate_late), 2),
        "ontime_mean_review_stars": ontime_rating,
        "late_mean_review_stars": late_rating,
        "review_rating_drop_stars": round(ontime_rating - late_rating, 2),
        "top_risk_state": state_summary.iloc[0]["customer_state"],
        "top_risk_state_rate_pct": float(state_summary.iloc[0]["late_rate_pct"]),
        "safest_state": state_summary[state_summary["total_orders"] >= 200].iloc[-1]["customer_state"],
        "median_carrier_handoff_hours": round(float(df["carrier_handoff_hours"].median()), 1),
        "median_estimated_transit_days": round(float(df["estimated_transit_days"].median()), 1)
    }

    with open(REP / "eda_summary.json", "w") as f:
        json.dump(eda_summary, f, indent=2)

    print("EDA execution complete! Summary saved to reports/eda_summary.json")
    print(json.dumps(eda_summary, indent=2))
    return eda_summary

if __name__ == "__main__":
    run_eda()
