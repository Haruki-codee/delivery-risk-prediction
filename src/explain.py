"""Optional SHAP explainability report for the best saved tree model."""
from pathlib import Path
import json, joblib, pandas as pd
ROOT=Path(__file__).resolve().parents[1]
def main():
    meta=json.load(open(ROOT/"artifacts"/"model_metadata.json"))
    model_name=meta["best_model"]
    pipe=joblib.load(ROOT/"artifacts"/f"{model_name}.joblib")
    if model_name=="logistic_regression":
        print("For Logistic Regression, inspect coefficients using the app's feature importance view.")
        return
    try:
        import shap
        from src.pipeline import load_data, build_order_features
        df,num,cat=build_order_features(load_data())
        X=df[num+cat].sample(min(1500,len(df)),random_state=42)
        Xt=pipe.named_steps["preprocess"].transform(X)
        estimator=pipe.named_steps["model"]
        explainer=shap.TreeExplainer(estimator)
        values=explainer.shap_values(Xt)
        if isinstance(values,list): values=values[-1]
        import matplotlib.pyplot as plt
        shap.summary_plot(values,Xt,show=False)
        plt.tight_layout()
        plt.savefig(ROOT/"reports"/"shap_summary.png",dpi=160,bbox_inches="tight")
        print("Saved reports/shap_summary.png")
    except Exception as e:
        print(f"SHAP report could not be generated: {e}")
if __name__=="__main__": main()
