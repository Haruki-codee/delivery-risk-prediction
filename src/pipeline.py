from pathlib import Path
import json, warnings
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             average_precision_score, precision_score, recall_score, f1_score)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
ART = ROOT / "artifacts"
REP = ROOT / "reports"
ART.mkdir(exist_ok=True); REP.mkdir(exist_ok=True)

FILES = {
    "orders": "olist_orders_dataset.csv", "items": "olist_order_items_dataset.csv",
    "payments": "olist_order_payments_dataset.csv", "customers": "olist_customers_dataset.csv",
    "products": "olist_products_dataset.csv", "sellers": "olist_sellers_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "translation": "product_category_name_translation.csv",
}

def load_data():
    data = {}
    for key, name in FILES.items():
        path = RAW / name
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Place all Olist CSVs in data/raw/.")
        data[key] = pd.read_csv(path)
    return data

def quality_report(data):
    rows = []
    for name, df in data.items():
        rows.append({
            "table": name, "rows": len(df), "columns": len(df.columns),
            "duplicate_rows": int(df.duplicated().sum()),
            "total_missing_cells": int(df.isna().sum().sum()),
            "missing_pct": round(float(df.isna().mean().mean()*100), 3)
        })
    report = pd.DataFrame(rows)
    report.to_csv(REP/"data_quality_summary.csv", index=False)
    return report

def build_order_features(data):
    orders = data["orders"].copy()
    date_cols = ["order_purchase_timestamp", "order_approved_at",
                 "order_delivered_carrier_date", "order_delivered_customer_date",
                 "order_estimated_delivery_date"]
    for c in date_cols:
        orders[c] = pd.to_datetime(orders[c], errors="coerce")

    items = data["items"].copy()
    items_agg = items.groupby("order_id").agg(
        item_count=("order_item_id", "count"),
        total_price=("price", "sum"),
        total_freight=("freight_value", "sum"),
        unique_sellers=("seller_id", "nunique"),
        unique_products=("product_id", "nunique"),
        shipping_limit=("shipping_limit_date", "min")
    ).reset_index()
    items_agg["shipping_limit"] = pd.to_datetime(items_agg["shipping_limit"], errors="coerce")

    pay = data["payments"].groupby("order_id").agg(
        payment_total=("payment_value", "sum"),
        payment_installments_max=("payment_installments", "max"),
        payment_methods=("payment_type", "nunique")
    ).reset_index()

    cust = data["customers"][["customer_id","customer_state","customer_city"]].drop_duplicates("customer_id")
    feat = orders.merge(items_agg, on="order_id", how="left").merge(pay, on="order_id", how="left")
    feat = feat.merge(cust, on="customer_id", how="left")

    # Product mix: order-level average product weight and category diversity.
    product = data["products"][["product_id","product_category_name","product_weight_g"]]
    item_prod = items.merge(product, on="product_id", how="left")
    prod_agg = item_prod.groupby("order_id").agg(
        avg_product_weight_g=("product_weight_g","mean"),
        category_count=("product_category_name","nunique")
    ).reset_index()
    feat = feat.merge(prod_agg, on="order_id", how="left")

    # Label is only defined for completed deliveries with both timestamps.
    delivered = feat["order_delivered_customer_date"].notna() & feat["order_estimated_delivery_date"].notna()
    feat = feat.loc[delivered].copy()
    feat["delivery_delay_days"] = (
        feat["order_delivered_customer_date"] - feat["order_estimated_delivery_date"]
    ).dt.total_seconds()/86400
    feat["is_late"] = (feat["delivery_delay_days"] > 0).astype(int)

    feat["purchase_hour"] = feat["order_purchase_timestamp"].dt.hour
    feat["purchase_dayofweek"] = feat["order_purchase_timestamp"].dt.dayofweek
    feat["purchase_month"] = feat["order_purchase_timestamp"].dt.month
    feat["approval_delay_hours"] = (
        feat["order_approved_at"] - feat["order_purchase_timestamp"]
    ).dt.total_seconds()/3600
    feat["carrier_handoff_hours"] = (
        feat["order_delivered_carrier_date"] - feat["order_approved_at"]
    ).dt.total_seconds()/3600
    feat["estimated_transit_days"] = (
        feat["order_estimated_delivery_date"] - feat["order_delivered_carrier_date"]
    ).dt.total_seconds()/86400
    feat["shipping_buffer_days"] = (
        feat["order_estimated_delivery_date"] - feat["shipping_limit"]
    ).dt.total_seconds()/86400

    # Do not use actual delivery timestamp or derived delay as predictors (target leakage).
    numeric = [
        "item_count","total_price","total_freight","unique_sellers","unique_products",
        "payment_total","payment_installments_max","payment_methods",
        "avg_product_weight_g","category_count","purchase_hour","purchase_dayofweek",
        "purchase_month","approval_delay_hours","carrier_handoff_hours",
        "estimated_transit_days","shipping_buffer_days"
    ]
    categorical = ["customer_state","order_status"]
    keep = ["order_id","is_late","delivery_delay_days"] + numeric + categorical
    out = feat[keep].replace([np.inf,-np.inf], np.nan)
    out.to_csv(ROOT/"data"/"processed_orders.csv", index=False)
    return out, numeric, categorical

def train_models(df, numeric, categorical):
    X = df[numeric + categorical].copy()
    y = df["is_late"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pre = ColumnTransformer([
        ("num", Pipeline([("imputer",SimpleImputer(strategy="median")),
                          ("scaler",StandardScaler())]), numeric),
        ("cat", Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),
                          ("onehot",OneHotEncoder(handle_unknown="ignore"))]), categorical)
    ])
    models = {
        "logistic_regression": LogisticRegression(max_iter=1500, class_weight="balanced"),
        "random_forest": RandomForestClassifier(n_estimators=250, min_samples_leaf=3,
                         class_weight="balanced_subsample", random_state=42, n_jobs=-1)
    }
    try:
        from xgboost import XGBClassifier
        scale = max(1.0, (y_train == 0).sum()/max(1,(y_train == 1).sum()))
        models["xgboost"] = XGBClassifier(
            n_estimators=350, max_depth=5, learning_rate=0.04,
            subsample=0.85, colsample_bytree=0.85, reg_lambda=2,
            eval_metric="logloss", scale_pos_weight=scale, random_state=42,
            n_jobs=-1
        )
    except Exception as e:
        warnings.warn(f"XGBoost unavailable; skipping: {e}")

    metrics, fitted = [], {}
    for name, model in models.items():
        pipe = Pipeline([("preprocess",pre),("model",model)])
        pipe.fit(X_train,y_train)
        prob = pipe.predict_proba(X_test)[:,1]
        pred = (prob >= 0.5).astype(int)
        metrics.append({
            "model":name, "roc_auc":roc_auc_score(y_test,prob),
            "pr_auc":average_precision_score(y_test,prob),
            "precision":precision_score(y_test,pred,zero_division=0),
            "recall":recall_score(y_test,pred,zero_division=0),
            "f1":f1_score(y_test,pred,zero_division=0),
            "test_rows":len(y_test)
        })
        fitted[name] = pipe
        joblib.dump(pipe, ART/f"{name}.joblib")
        pd.DataFrame(classification_report(y_test,pred,output_dict=True)).T.to_csv(REP/f"{name}_classification_report.csv")
        pd.DataFrame(confusion_matrix(y_test,pred),index=["actual_on_time","actual_late"],
                     columns=["predicted_on_time","predicted_late"]).to_csv(REP/f"{name}_confusion_matrix.csv")
    m = pd.DataFrame(metrics).sort_values("roc_auc",ascending=False)
    m.to_csv(REP/"model_metrics.csv",index=False)
    best_name = m.iloc[0]["model"]
    with open(ART/"model_metadata.json","w") as f:
        json.dump({"best_model":best_name,"numeric_features":numeric,
                   "categorical_features":categorical,"threshold":0.5},f,indent=2)
    return m, fitted[best_name], X_test, y_test

def run():
    data=load_data()
    quality_report(data)
    df,num,cat=build_order_features(data)
    metrics,best,X_test,y_test=train_models(df,num,cat)
    # Persist a compact scoring sample for the dashboard.
    scored=X_test.copy()
    scored["actual_is_late"]=y_test.values
    scored["predicted_late_probability"]=best.predict_proba(X_test)[:,1]
    scored.to_csv(REP/"test_predictions.csv",index=False)
    print(f"Completed. Labeled orders: {len(df):,}; late rate: {df.is_late.mean():.1%}")
    print(metrics.to_string(index=False))

if __name__=="__main__":
    run()
