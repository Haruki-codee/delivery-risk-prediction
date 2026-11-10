from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from src.pipeline import load_data, quality_report, build_order_features

ROOT=Path(__file__).resolve().parents[1]
def main():
    data=load_data()
    quality_report(data)
    df,_,_=build_order_features(data)
    summary={
        "labeled_delivered_orders":int(len(df)),
        "late_orders":int(df.is_late.sum()),
        "late_rate_pct":round(float(df.is_late.mean()*100),2),
        "median_delay_days":round(float(df.delivery_delay_days.median()),2),
        "mean_order_value":round(float(df.total_price.mean()),2)
    }
    pd.Series(summary).to_json(ROOT/"reports"/"eda_summary.json",indent=2)
    ax=df.groupby("purchase_month").is_late.mean().plot(kind="bar",title="Late delivery rate by purchase month")
    ax.set_ylabel("Late rate")
    plt.tight_layout()
    plt.savefig(ROOT/"reports"/"late_rate_by_month.png",dpi=150)
    print(summary)
if __name__=="__main__": main()
