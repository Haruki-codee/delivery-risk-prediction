from pathlib import Path
import json, joblib
import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np

ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title="Olist Delivery Risk",page_icon="📦",layout="wide")
st.title("📦 Olist Delivery Risk Intelligence")
st.caption("Order-level late-delivery prediction, prioritization and model monitoring")

metrics_path=ROOT/"reports"/"model_metrics.csv"
pred_path=ROOT/"reports"/"test_predictions.csv"
data_path=ROOT/"data"/"processed_orders.csv"
meta_path=ROOT/"artifacts"/"model_metadata.json"

if not data_path.exists() or not meta_path.exists():
    st.warning("Project artifacts are not built yet. Run `python -m src.pipeline` from the project folder.")
    st.stop()

df=pd.read_csv(data_path)
metrics=pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
pred=pd.read_csv(pred_path) if pred_path.exists() else pd.DataFrame()
meta=json.load(open(meta_path))
late_rate=df.is_late.mean()*100
c1,c2,c3,c4=st.columns(4)
c1.metric("Delivered orders analyzed",f"{len(df):,}")
c2.metric("Late delivery rate",f"{late_rate:.1f}%")
c3.metric("Median delivery variance",f"{df.delivery_delay_days.median():.1f} days")
c4.metric("Selected model",meta["best_model"].replace("_"," ").title())

tabs=st.tabs(["Overview","Risk queue","Model performance","Explainability","Data quality"])
with tabs[0]:
    a,b=st.columns(2)
    with a:
        st.plotly_chart(px.histogram(df,x="delivery_delay_days",nbins=45,
            title="Delivery variance (days)",labels={"delivery_delay_days":"Days vs estimated date"}),use_container_width=True)
    with b:
        state=df.groupby("customer_state",as_index=False).is_late.mean().sort_values("is_late",ascending=False).head(15)
        st.plotly_chart(px.bar(state,x="customer_state",y="is_late",title="Late rate by customer state",
            labels={"is_late":"Late rate","customer_state":"State"}),use_container_width=True)
    month=df.groupby("purchase_month",as_index=False).is_late.mean()
    st.plotly_chart(px.line(month,x="purchase_month",y="is_late",markers=True,title="Late rate by purchase month"),use_container_width=True)

with tabs[1]:
    st.subheader("Delivery-risk prioritization")
    st.write("Orders are ranked by predicted probability of late delivery. Use the threshold to tune operational workload.")
    threshold=st.slider("High-risk probability threshold",0.1,0.9,0.5,0.05)
    if pred.empty:
        st.info("Run the training pipeline to create test predictions.")
    else:
        q=pred.copy()
        q["risk_band"] = np.select(
    [
        q["predicted_late_probability"] >= 0.75,
        q["predicted_late_probability"] >= threshold
    ],
    [
        "Critical",
        "High"
    ],
    default="Monitor"
)
        q=q.sort_values("predicted_late_probability",ascending=False)
        st.metric("Orders above threshold",f"{(q.predicted_late_probability>=threshold).sum():,}")
        st.dataframe(q[["predicted_late_probability","risk_band","actual_is_late"]].head(300),
                     use_container_width=True)
        st.download_button("Download prioritized risk queue",q.to_csv(index=False),
                           file_name="delivery_risk_queue.csv",mime="text/csv")

with tabs[2]:
    st.subheader("Out-of-sample model comparison")
    if metrics.empty: st.info("Metrics will appear after training.")
    else:
        st.dataframe(metrics,use_container_width=True)
        st.plotly_chart(px.bar(metrics,x="model",y="roc_auc",title="ROC-AUC by model"),use_container_width=True)
        st.caption("Use PR-AUC, recall and operational capacity alongside ROC-AUC; the default 0.50 threshold is not automatically optimal.")

with tabs[3]:
    st.subheader("Explainability")
    shap_path=ROOT/"reports"/"shap_summary.png"
    if shap_path.exists(): st.image(str(shap_path),caption="SHAP global feature impact")
    else:
        st.info("Optional SHAP output not found. Run `python -m src.explain` after model training.")
    st.write("Interpretation note: model explanations describe learned associations, not causal effects.")

with tabs[4]:
    quality=ROOT/"reports"/"data_quality_summary.csv"
    if quality.exists(): st.dataframe(pd.read_csv(quality),use_container_width=True)
    else: st.info("Run the pipeline to generate data quality checks.")
