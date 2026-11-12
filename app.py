from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# Configuration & Theme Setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Olist Delivery Risk Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "processed_orders.csv"
META_PATH = ROOT / "artifacts" / "model_metadata.json"
METRICS_PATH = ROOT / "reports" / "model_metrics.csv"
PRED_PATH = ROOT / "reports" / "test_predictions.csv"
EDA_SUMMARY_PATH = ROOT / "reports" / "eda_summary.json"
THRESH_PATH = ROOT / "reports" / "threshold_analysis.csv"
FEAT_IMP_PATH = ROOT / "reports" / "feature_importance.csv"
QUALITY_PATH = ROOT / "reports" / "data_quality_summary.csv"

# Custom Styling for Simple, Clean, Modern UI
st.markdown("""
<style>
    /* Global font & container clean-up */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
    }
    
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 14px 18px;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    div[data-testid="stMetric"]:hover {
        border-color: #cbd5e1;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        color: #475569 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #0f172a !important;
    }

    /* Custom KPI Card */
    .kpi-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
        margin-bottom: 12px;
    }
    .kpi-title {
        font-size: 0.82rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }
    .kpi-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0f172a;
    }
    .kpi-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-top: 4px;
    }
    .badge-danger { background-color: #fee2e2; color: #991b1b; }
    .badge-success { background-color: #dcfce7; color: #166534; }
    .badge-warning { background-color: #fef3c7; color: #92400e; }
    .badge-info { background-color: #e0f2fe; color: #075985; }

    /* Clean Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 2px solid #f1f5f9;
        margin-bottom: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        white-space: pre-wrap;
        border-radius: 8px 8px 0px 0px;
        font-weight: 600;
        font-size: 0.95rem;
        color: #64748b;
        padding: 0 18px;
    }
    .stTabs [aria-selected="true"] {
        background-color: transparent !important;
        color: #2563eb !important;
        border-bottom: 3px solid #2563eb !important;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Data & Model Caching
# -----------------------------------------------------------------------------
@st.cache_data
def load_app_data():
    if not DATA_PATH.exists() or not META_PATH.exists():
        return None, None, None, None, None, None, None
    df = pd.read_csv(DATA_PATH)
    meta = json.load(open(META_PATH))
    metrics = pd.read_csv(METRICS_PATH) if METRICS_PATH.exists() else pd.DataFrame()
    pred = pd.read_csv(PRED_PATH) if PRED_PATH.exists() else pd.DataFrame()
    eda_summary = json.load(open(EDA_SUMMARY_PATH)) if EDA_SUMMARY_PATH.exists() else {}
    thresh = pd.read_csv(THRESH_PATH) if THRESH_PATH.exists() else pd.DataFrame()
    feat_imp = pd.read_csv(FEAT_IMP_PATH) if FEAT_IMP_PATH.exists() else pd.DataFrame()
    return df, meta, metrics, pred, eda_summary, thresh, feat_imp

@st.cache_resource
def load_trained_pipeline(model_name: str):
    model_file = ROOT / "artifacts" / f"{model_name}.joblib"
    if model_file.exists():
        return joblib.load(model_file)
    return None

df, meta, metrics, pred, eda_summary, thresh, feat_imp = load_app_data()

# Check pipeline status
if df is None or meta is None:
    st.error("⚠️ Project artifacts not found.")
    st.info("Please run the training pipeline first: `python -m src.pipeline` from the repository root.")
    st.stop()

# -----------------------------------------------------------------------------
# Sidebar Navigation & Filters
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📦 Olist Delivery Risk")
    st.caption("AI-Powered E-Commerce Dispatch Intelligence")
    st.divider()

    best_model_name = meta.get("best_model", "xgboost").replace("_", " ").title()
    st.markdown(f"**Champion Model:** `{best_model_name}`")
    st.markdown(f"**Evaluated Orders:** `{len(df):,}`")
    st.markdown(f"**Baseline Late Rate:** `{df.is_late.mean() * 100:.2f}%`")
    st.divider()

    st.subheader("Global Filters")
    available_states = ["All States"] + sorted(df["customer_state"].dropna().unique().tolist())
    selected_state = st.selectbox("Destination State", available_states, index=0)

    risk_band_filter = st.selectbox(
        "Risk Tier Filter",
        ["All Orders", "Critical (≥ 75%)", "High (≥ 50%)", "Standard (< 50%)"],
        index=0
    )

    st.divider()
    st.markdown("💡 **Tip:** Adjust dispatch intervention thresholds in the **Risk Queue** tab to balance operations workload and late order recovery.")

# Apply Filters for filtered view
df_view = df.copy()
if selected_state != "All States":
    df_view = df_view[df_view["customer_state"] == selected_state]

pred_view = pred.copy() if not pred.empty else pd.DataFrame()
if not pred_view.empty and selected_state != "All States":
    pred_view = pred_view[pred_view["customer_state"] == selected_state]

# -----------------------------------------------------------------------------
# Main Header & Top Stat Cards
# -----------------------------------------------------------------------------
st.title("📦 Olist Delivery Risk Intelligence")
st.markdown("Real-time order-level lateness prediction, dispatch risk prioritization, and logistics bottleneck analytics.")

# Executive Top Strip
col1, col2, col3, col4 = st.columns(4)
late_rate = df_view["is_late"].mean() * 100
median_delay = df_view["delivery_delay_days"].median()
interstate_rate = eda_summary.get("interstate_late_rate_pct", 9.27)
intrastate_rate = eda_summary.get("intrastate_late_rate_pct", 6.06)

col1.metric("Orders Analyzed", f"{len(df_view):,}", delta=f"{selected_state}" if selected_state != "All States" else None)
col2.metric("Late Delivery Rate", f"{late_rate:.2f}%", delta=f"{late_rate - (df.is_late.mean()*100):+.2f}% vs Nat'l" if selected_state != "All States" else None, delta_color="inverse")
col3.metric("Median Early Buffer", f"{abs(median_delay):.1f} days ahead", help="Delivered ahead of customer estimated date")
col4.metric("Champion Model", best_model_name, help="Trained with Stratified Split & Out-of-Sample Validation")

st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Navigation Tabs
# -----------------------------------------------------------------------------
tab_overview, tab_eda, tab_queue, tab_sim, tab_model, tab_quality = st.tabs([
    "📊 Overview",
    "🔍 Deep-Dive EDA",
    "🎯 Risk Queue",
    "🔮 Risk Simulator",
    "📈 Model & Drivers",
    "📋 Data Quality"
])

# -----------------------------------------------------------------------------
# TAB 1: EXECUTIVE OVERVIEW
# -----------------------------------------------------------------------------
with tab_overview:
    st.subheader("Fulfillment Performance & Timelines")
    
    col_a, col_b = st.columns(2)
    with col_a:
        # Delivery Variance Histogram
        fig_delay = px.histogram(
            df_view,
            x="delivery_delay_days",
            nbins=50,
            title="Delivery Variance vs Estimated Date (Days)",
            labels={"delivery_delay_days": "Actual Delivery Days - Estimated Date (Negative = Early)"},
            color_discrete_sequence=["#3b82f6"]
        )
        fig_delay.add_vline(x=0, line_dash="dash", line_color="#ef4444", annotation_text="Late Threshold (Day 0)")
        fig_delay.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=40, b=10), height=340)
        st.plotly_chart(fig_delay, use_container_width=True)

    with col_b:
        # Monthly Seasonality
        if "purchase_month" in df_view.columns:
            m_agg = df_view.groupby("purchase_month").agg(
                orders=("is_late", "count"),
                late_rate=("is_late", lambda x: round(float(x.mean() * 100), 2))
            ).reset_index()
            fig_month = go.Figure()
            fig_month.add_trace(go.Bar(
                x=m_agg["purchase_month"], y=m_agg["orders"],
                name="Volume", marker_color="#cbd5e1", opacity=0.7, yaxis="y1"
            ))
            fig_month.add_trace(go.Scatter(
                x=m_agg["purchase_month"], y=m_agg["late_rate"],
                name="Late Rate (%)", mode="lines+markers", line=dict(color="#dc2626", width=2.5), yaxis="y2"
            ))
            fig_month.update_layout(
                title="Monthly Seasonality: Order Volume vs Late Delivery Rate",
                template="plotly_white",
                yaxis=dict(title="Orders Placed", showgrid=False),
                yaxis2=dict(title="Late Rate (%)", overlaying="y", side="right", showgrid=True),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=10, r=10, t=40, b=10),
                height=340
            )
            st.plotly_chart(fig_month, use_container_width=True)

    st.markdown("#### Fulfillment Milestone Bottleneck Breakdown")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">1. Payment Approval</div>
            <div class="kpi-value">{:.1f} hrs</div>
            <span class="kpi-badge badge-success">Median Duration</span>
        </div>
        """.format(df["approval_delay_hours"].median()), unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">2. Carrier Handoff</div>
            <div class="kpi-value">{:.1f} hrs</div>
            <span class="kpi-badge badge-warning">Top Bottleneck</span>
        </div>
        """.format(df["carrier_handoff_hours"].median()), unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">3. Transit Schedule</div>
            <div class="kpi-value">{:.1f} days</div>
            <span class="kpi-badge badge-info">Median Transit</span>
        </div>
        """.format(df["estimated_transit_days"].median()), unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">4. Interstate Orders</div>
            <div class="kpi-value">{:.1f}%</div>
            <span class="kpi-badge badge-danger">Cross-State Risk</span>
        </div>
        """.format(df["is_interstate"].mean() * 100), unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TAB 2: DEEP-DIVE EXPLORATORY DATA ANALYSIS (NEW EDA)
# -----------------------------------------------------------------------------
with tab_eda:
    st.subheader("🔍 Exploratory Data Analysis & Strategic Logistics Insights")
    st.write("Granular insights into geographical disparities, downstream customer review penalties, and cargo dynamics.")

    eda_tab1, eda_tab2, eda_tab3, eda_tab4 = st.tabs([
        "🗺️ Geography & Interstate Routes",
        "⭐ Customer Satisfaction Impact",
        "📦 Cargo & Freight Dynamics",
        "⏱️ Temporal Patterns"
    ])

    with eda_tab1:
        st.markdown("#### Interstate (Cross-State) vs Intrastate Delivery Disparity")
        r1, r2, r3 = st.columns(3)
        with r1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Intrastate Late Rate (Same State)</div>
                <div class="kpi-value">{intrastate_rate:.2f}%</div>
                <span class="kpi-badge badge-success">Local Fulfillment</span>
            </div>
            """, unsafe_allow_html=True)
        with r2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Interstate Late Rate (Cross-Border)</div>
                <div class="kpi-value">{interstate_rate:.2f}%</div>
                <span class="kpi-badge badge-danger">+53% Risk Premium</span>
            </div>
            """, unsafe_allow_html=True)
        with r3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Interstate Order Share</div>
                <div class="kpi-value">{df.is_interstate.mean()*100:.1f}%</div>
                <span class="kpi-badge badge-info">64k of 96k Orders</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        
        # State risk ranking chart
        state_df = df.groupby("customer_state").agg(
            orders=("is_late", "count"),
            late_rate=("is_late", lambda x: round(float(x.mean() * 100), 2)),
            avg_freight=("total_freight", lambda x: round(float(x.mean()), 2))
        ).reset_index().query("orders >= 200").sort_values("late_rate", ascending=False)

        fig_state = px.bar(
            state_df,
            x="customer_state",
            y="late_rate",
            color="late_rate",
            color_continuous_scale="Reds",
            title="Late Delivery Rate by Destination State (Min 200 Orders)",
            labels={"customer_state": "State", "late_rate": "Late Rate (%)"},
            text="late_rate"
        )
        fig_state.add_hline(y=df.is_late.mean() * 100, line_dash="dash", line_color="#2563eb", annotation_text=f"National Baseline ({df.is_late.mean()*100:.1f}%)")
        fig_state.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_state.update_layout(template="plotly_white", height=380, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_state, use_container_width=True)

    with eda_tab2:
        st.markdown("#### Downstream Customer Impact: How Late Delivery Crashes Reviews")
        st.write("Observational analysis linking delivery tardiness with customer satisfaction ratings.")
        
        c_rev1, c_rev2 = st.columns(2)
        with c_rev1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">On-Time Orders Rating</div>
                <div class="kpi-value">{eda_summary.get('ontime_mean_review_stars', 4.29):.2f} ★</div>
                <span class="kpi-badge badge-success">88.7% Positive Reviews</span>
            </div>
            """, unsafe_allow_html=True)
        with c_rev2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Late Orders Rating</div>
                <div class="kpi-value">{eda_summary.get('late_mean_review_stars', 2.57):.2f} ★</div>
                <span class="kpi-badge badge-danger">54.0% Negative (1-2 Stars)</span>
            </div>
            """, unsafe_allow_html=True)

        rev_impact_path = ROOT / "reports" / "eda_review_impact.csv"
        if rev_impact_path.exists():
            rev_df = pd.read_csv(rev_impact_path)
            fig_rev = px.bar(
                rev_df,
                x="status",
                y=["pct_negative_1_2", "pct_5_star"],
                barmode="group",
                title="Review Rating Composition: On-Time vs Late Deliveries",
                labels={"value": "Percentage of Deliveries (%)", "status": "Delivery Status", "variable": "Rating Tier"},
                color_discrete_map={"pct_negative_1_2": "#ef4444", "pct_5_star": "#10b981"}
            )
            fig_rev.update_layout(template="plotly_white", height=350, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_rev, use_container_width=True)

    with eda_tab3:
        st.markdown("#### Cargo Characteristics, Product Weight & Freight Burden")
        c_cargo1, c_cargo2 = st.columns(2)
        with c_cargo1:
            # Freight ratio vs delay
            fig_fr = px.scatter(
                df_view.sample(min(3000, len(df_view)), random_state=42),
                x="freight_ratio",
                y="delivery_delay_days",
                color="is_late",
                title="Freight-to-Price Ratio vs Delivery Delay (Days)",
                labels={"freight_ratio": "Freight / Price Ratio", "delivery_delay_days": "Delivery Delay (Days)"},
                color_discrete_map={0: "#94a3b8", 1: "#ef4444"},
                opacity=0.6
            )
            fig_fr.update_layout(template="plotly_white", height=360, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_fr, use_container_width=True)
            
        with c_cargo2:
            # Category late rate if available
            cat_csv = ROOT / "reports" / "eda_category_analysis.csv"
            if cat_csv.exists():
                cat_df = pd.read_csv(cat_csv).head(10).sort_values("late_rate_pct", ascending=True)
                fig_cat = px.bar(
                    cat_df,
                    x="late_rate_pct",
                    y="product_category_name_english",
                    orientation="h",
                    title="Top 10 High-Risk Product Categories (Min 300 Orders)",
                    labels={"late_rate_pct": "Late Delivery Rate (%)", "product_category_name_english": "Category"},
                    color="late_rate_pct",
                    color_continuous_scale="Purples"
                )
                fig_cat.update_layout(template="plotly_white", height=360, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig_cat, use_container_width=True)

    with eda_tab4:
        st.markdown("#### Temporal & Dispatch Timing Analysis")
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            dow_df = df_view.groupby("purchase_dayofweek")["is_late"].mean().reset_index()
            dow_df["day_name"] = dow_df["purchase_dayofweek"].map({
                0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"
            })
            dow_df["late_rate_pct"] = dow_df["is_late"] * 100
            fig_dow = px.bar(
                dow_df, x="day_name", y="late_rate_pct",
                title="Late Delivery Rate by Purchase Day of Week",
                labels={"day_name": "Day of Week", "late_rate_pct": "Late Rate (%)"},
                color_discrete_sequence=["#6366f1"]
            )
            fig_dow.update_layout(template="plotly_white", height=340, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_dow, use_container_width=True)

        with col_t2:
            hour_df = df_view.groupby("purchase_hour")["is_late"].mean().reset_index()
            hour_df["late_rate_pct"] = hour_df["is_late"] * 100
            fig_hour = px.line(
                hour_df, x="purchase_hour", y="late_rate_pct",
                markers=True,
                title="Late Delivery Rate by Purchase Hour (0-23)",
                labels={"purchase_hour": "Hour of Day (UTC/BRT)", "late_rate_pct": "Late Rate (%)"},
                line_shape="spline"
            )
            fig_hour.update_layout(template="plotly_white", height=340, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_hour, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 3: RISK PRIORITIZATION QUEUE
# -----------------------------------------------------------------------------
with tab_queue:
    st.subheader("🎯 Operational Delivery Risk Queue")
    st.write("Rank and prioritize out-for-delivery orders by predicted late probability to allocate expedited shipping or customer support.")

    if pred_view.empty:
        st.info("No test predictions available. Run `python -m src.pipeline` to generate predictions.")
    else:
        q_thresh = st.slider("Dispatch Alert Risk Threshold", 0.10, 0.90, 0.50, 0.05,
                             help="Orders with probability above this threshold trigger dispatch intervention.")

        # Classify risk bands
        q = pred_view.copy()
        q["risk_tier"] = np.select(
            [q["predicted_late_probability"] >= 0.75, q["predicted_late_probability"] >= q_thresh],
            ["🔴 Critical", "🟠 High"],
            default="🔵 Monitor"
        )
        if risk_band_filter == "Critical (≥ 75%)":
            q = q[q["predicted_late_probability"] >= 0.75]
        elif risk_band_filter == "High (≥ 50%)":
            q = q[q["predicted_late_probability"] >= 0.50]
        elif risk_band_filter == "Standard (< 50%)":
            q = q[q["predicted_late_probability"] < 0.50]

        q = q.sort_values("predicted_late_probability", ascending=False)
        flagged = (q["predicted_late_probability"] >= q_thresh).sum()
        total_q = len(q)

        # Operational metrics strip
        mq1, mq2, mq3, mq4 = st.columns(4)
        mq1.metric("Orders Above Alert Threshold", f"{flagged:,}")
        mq2.metric("Fleet Workload Flagged", f"{(flagged / max(1, total_q) * 100):.1f}%")
        
        # Calculate precision & recall if actuals exist
        if "actual_is_late" in q.columns and flagged > 0:
            p_flagged = q[q["predicted_late_probability"] >= q_thresh]
            tp = (p_flagged["actual_is_late"] == 1).sum()
            total_actual_late = (q["actual_is_late"] == 1).sum()
            prec = (tp / flagged * 100) if flagged > 0 else 0
            rec = (tp / total_actual_late * 100) if total_actual_late > 0 else 0
            mq3.metric("Estimated Precision", f"{prec:.1f}%", help="Proportion of flagged orders that actually arrive late")
            mq4.metric("Estimated Recall", f"{rec:.1f}%", help="Proportion of all late orders captured by this alert threshold")
        else:
            mq3.metric("Alert Status", "Active")
            mq4.metric("Action", "Expedite Dispatch")

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

        # Search Bar
        search_query = st.text_input("🔍 Search orders by Order ID or State:", "").strip().lower()
        if search_query:
            q = q[q["order_id"].astype(str).str.lower().str.contains(search_query) |
                  q["customer_state"].astype(str).str.lower().str.contains(search_query)]

        # Display table with readable columns
        display_cols = [
            c for c in ["order_id", "customer_state", "is_interstate", "total_price", "total_freight",
                        "carrier_handoff_hours", "estimated_transit_days", "predicted_late_probability",
                        "risk_tier", "actual_is_late"] if c in q.columns
        ]

        st.dataframe(
            q[display_cols].head(250),
            column_config={
                "predicted_late_probability": st.column_config.ProgressColumn(
                    "Risk Score", format="%.2f", min_value=0, max_value=1
                ),
                "total_price": st.column_config.NumberColumn("Price (R$)", format="R$ %.2f"),
                "total_freight": st.column_config.NumberColumn("Freight (R$)", format="R$ %.2f"),
                "carrier_handoff_hours": st.column_config.NumberColumn("Handoff (hrs)", format="%.1f h"),
                "estimated_transit_days": st.column_config.NumberColumn("Transit (days)", format="%.1f d"),
                "is_interstate": st.column_config.CheckboxColumn("Interstate?"),
                "actual_is_late": st.column_config.CheckboxColumn("Actual Late?")
            },
            use_container_width=True,
            height=420
        )

        st.download_button(
            label="📥 Download Prioritized Risk Queue (CSV)",
            data=q.to_csv(index=False),
            file_name="olist_delivery_risk_queue.csv",
            mime="text/csv"
        )

# -----------------------------------------------------------------------------
# TAB 4: INTERACTIVE ORDER RISK SIMULATOR
# -----------------------------------------------------------------------------
with tab_sim:
    st.subheader("🔮 Interactive Order Dispatch Risk Simulator")
    st.write("Test new or pending orders in real-time. Adjust logistics attributes to simulate delivery delay risk.")

    pipe = load_trained_pipeline(meta.get("best_model", "xgboost"))
    if pipe is None:
        st.warning("Model pipeline artifact not loaded.")
    else:
        with st.form("risk_simulator_form"):
            s_col1, s_col2, s_col3 = st.columns(3)
            with s_col1:
                in_state = st.selectbox("Customer Destination State", sorted(df["customer_state"].dropna().unique().tolist()), index=0)
                in_price = st.number_input("Merchandise Total Price (R$)", min_value=5.0, max_value=8000.0, value=120.0, step=10.0)
                in_freight = st.number_input("Shipping Freight Cost (R$)", min_value=2.0, max_value=800.0, value=22.0, step=5.0)
                in_items = st.number_input("Number of Items in Order", min_value=1, max_value=20, value=1, step=1)
            with s_col2:
                in_interstate = st.selectbox("Interstate Route?", ["Yes (Cross-Border)", "No (Intrastate Same State)"], index=0)
                in_weight = st.number_input("Average Product Weight (grams)", min_value=10.0, max_value=30000.0, value=950.0, step=100.0)
                in_volume = st.number_input("Estimated Product Volume (cm³)", min_value=100.0, max_value=200000.0, value=8000.0, step=500.0)
                in_sellers = st.number_input("Distinct Sellers", min_value=1, max_value=5, value=1, step=1)
            with s_col3:
                in_carrier_delay = st.slider("Carrier Handoff Delay (hours)", 0.0, 168.0, 48.0, 4.0, help="Time from approval to carrier dispatch")
                in_transit_days = st.slider("Estimated Transit Window (days)", 2.0, 45.0, 18.0, 1.0, help="Promised transit duration")
                in_buffer_days = st.slider("Shipping Buffer (days)", 0.0, 30.0, 12.0, 1.0)
                in_weekend = st.checkbox("Weekend Purchase?", value=False)

            submit_sim = st.form_submit_button("⚡ Assess Delivery Risk", use_container_width=True)

        if submit_sim:
            # Prepare row matching model input
            sim_dict = {
                "item_count": in_items,
                "total_price": in_price,
                "total_freight": in_freight,
                "freight_ratio": in_freight / (in_price + 0.01),
                "unique_sellers": in_sellers,
                "unique_products": in_items,
                "payment_total": in_price + in_freight,
                "payment_installments_max": 1,
                "payment_methods": 1,
                "avg_product_weight_g": in_weight,
                "avg_product_volume_cm3": in_volume,
                "category_count": 1,
                "purchase_hour": 14,
                "purchase_dayofweek": 5 if in_weekend else 2,
                "purchase_month": 6,
                "is_weekend_purchase": 1 if in_weekend else 0,
                "approval_delay_hours": 1.5,
                "carrier_handoff_hours": in_carrier_delay,
                "estimated_transit_days": in_transit_days,
                "shipping_buffer_days": in_buffer_days,
                "is_interstate": 1 if "Yes" in in_interstate else 0,
                "customer_state": in_state,
                "order_status": "delivered"
            }
            sim_df = pd.DataFrame([sim_dict])
            prob_late = float(pipe.predict_proba(sim_df)[:, 1][0])

            st.markdown("---")
            res_c1, res_c2 = st.columns([1, 2])
            with res_c1:
                st.metric("Predicted Late Risk", f"{prob_late * 100:.1f}%")
                if prob_late >= 0.75:
                    st.error("🚨 CRITICAL RISK TIER")
                elif prob_late >= 0.50:
                    st.warning("⚠️ HIGH RISK TIER")
                else:
                    st.success("✅ LOW / STANDARD RISK")

            with res_c2:
                st.markdown("#### Recommended Operational Action:")
                if prob_late >= 0.75:
                    st.markdown("""
                    - **Action:** 🔴 **Immediate Intervention**
                    - Upgrade to Priority Express Carrier.
                    - Expedite seller dispatch handoff within 12 hours.
                    - Preemptively update the estimated delivery date or notify customer.
                    """)
                elif prob_late >= 0.50:
                    st.markdown("""
                    - **Action:** 🟠 **Priority Dispatch Monitoring**
                    - Notify carrier for expedited pick-up.
                    - Verify warehouse pack-and-ship timeline.
                    """)
                else:
                    st.markdown("""
                    - **Action:** 🟢 **Standard Processing**
                    - Normal carrier dispatch SLA is sufficient.
                    """)

# -----------------------------------------------------------------------------
# TAB 5: MODEL PERFORMANCE & RISK DRIVERS (NO SHAP!)
# -----------------------------------------------------------------------------
with tab_model:
    st.subheader("📈 Out-of-Sample Model Benchmarking & Native Risk Drivers")
    st.caption("Standardized model interpretability via native feature importances — fast, robust, and zero-SHAP.")

    if not metrics.empty:
        st.markdown("#### Out-of-Sample Test Evaluation")
        st.dataframe(
            metrics.style.highlight_max(axis=0, subset=["roc_auc", "pr_auc", "precision", "recall", "f1"], color="#dcfce7"),
            use_container_width=True
        )

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            fig_auc = px.bar(
                metrics,
                x="model",
                y=["roc_auc", "pr_auc"],
                barmode="group",
                title="Model Performance Comparison (ROC-AUC vs PR-AUC)",
                labels={"value": "Score (0 - 1.0)", "model": "Algorithm", "variable": "Metric"},
                color_discrete_map={"roc_auc": "#2563eb", "pr_auc": "#7c3aed"}
            )
            fig_auc.update_layout(template="plotly_white", height=340, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_auc, use_container_width=True)

        with col_m2:
            fig_f1 = px.bar(
                metrics,
                x="model",
                y=["precision", "recall", "f1"],
                barmode="group",
                title="Classification Balance (Precision, Recall, F1)",
                labels={"value": "Score (0 - 1.0)", "model": "Algorithm", "variable": "Metric"},
                color_discrete_map={"precision": "#0ea5e9", "recall": "#f59e0b", "f1": "#10b981"}
            )
            fig_f1.update_layout(template="plotly_white", height=340, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_f1, use_container_width=True)

    # Feature Importance Section (Clean Native Tree & Model Importances)
    st.markdown("---")
    st.markdown("#### 🎯 Key Risk Drivers (Native Feature Importance)")
    st.write("Identifies the strongest learned signals driving delivery delay predictions.")

    if not feat_imp.empty:
        col_f1, col_f2 = st.columns([3, 2])
        with col_f1:
            top_feats = feat_imp.head(12).iloc[::-1]
            fig_imp = px.bar(
                top_feats,
                x="relative_importance_pct",
                y="feature",
                orientation="h",
                color="domain_group",
                title=f"Top 12 Late Delivery Predictors ({best_model_name})",
                labels={"relative_importance_pct": "Importance Weight (%)", "feature": "Feature", "domain_group": "Domain"},
                color_discrete_map={
                    "Geography & Route": "#ea580c",
                    "Logistics Bottleneck": "#ef4444",
                    "Logistics Schedule": "#3b82f6",
                    "Seasonality": "#8b5cf6",
                    "Cargo & Pricing": "#10b981",
                    "Order Value": "#06b6d4"
                }
            )
            fig_imp.update_layout(template="plotly_white", height=380, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_imp, use_container_width=True)

        with col_f2:
            st.markdown("##### Operational Interpretation")
            st.markdown("""
            1. **`customer_state` & `is_interstate`:** Destination geography and cross-state fulfillment are the dominant drivers of delivery delay risk in Brazil.
            2. **`estimated_transit_days`:** Tight transit windows without buffer create high delay exposure.
            3. **`carrier_handoff_hours`:** The primary operational bottleneck under merchant control — every extra 24 hours in warehouse handoff significantly multiplies late risk.
            4. **`purchase_month`:** Holiday surges and seasonal disruptions (e.g. Black Friday) cause systemic courier backlogs.
            """)

    # Threshold Sensitivity Analysis
    if not thresh.empty:
        st.markdown("---")
        st.markdown("#### Operational Trade-Off Curve (Precision vs Recall by Alert Threshold)")
        fig_th = px.line(
            thresh,
            x="threshold",
            y=["precision", "recall", "f1"],
            title="Threshold Sensitivity: Precision, Recall and F1 vs Probability Cutoff",
            labels={"threshold": "Classification Alert Threshold", "value": "Score", "variable": "Metric"},
            color_discrete_map={"precision": "#0ea5e9", "recall": "#ef4444", "f1": "#10b981"}
        )
        fig_th.update_layout(template="plotly_white", height=320, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_th, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 6: DATA QUALITY & PIPELINE HEALTH
# -----------------------------------------------------------------------------
with tab_quality:
    st.subheader("📋 Data Hygiene & Quality Audit")
    st.write("Audit summary of the 8 source Olist relational tables ingested during pipeline preprocessing.")

    if QUALITY_PATH.exists():
        q_df = pd.read_csv(QUALITY_PATH)
        st.dataframe(
            q_df,
            column_config={
                "rows": st.column_config.NumberColumn("Total Rows", format="%d"),
                "columns": st.column_config.NumberColumn("Columns"),
                "duplicate_rows": st.column_config.NumberColumn("Duplicates"),
                "total_missing_cells": st.column_config.NumberColumn("Missing Cells"),
                "missing_pct": st.column_config.ProgressColumn("Missing %", format="%.2f%%", min_value=0, max_value=10)
            },
            use_container_width=True
        )

    st.markdown("#### Pipeline Architecture & Leakage Safeguards")
    st.info("""
    - **Outcome Leakage Safeguards:** Actual delivery timestamps (`order_delivered_customer_date`) and derived delay are strictly excluded from predictors.
    - **Review Exclusion:** Customer review scores are post-delivery outcomes and are used strictly in EDA, never during model training.
    - **Temporal Integrity:** Time features are strictly constrained to purchase and pre-delivery handoff milestones.
    """)
