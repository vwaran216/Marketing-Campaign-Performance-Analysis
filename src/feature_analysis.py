import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 70)
print("MARKETING CAMPAIGN FEATURE ANALYSIS")
print("=" * 70)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "processed" / "model_ready_dataset.csv"
REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_FILE)

target = "ROI"

categorical_features = [
    "Campaign_Type",
    "Target_Audience",
    "Channel_Used",
    "Language",
    "Customer_Segment",
    "Brand",
    "Campaign_DayOfWeek"
]

numerical_features = [
    "Duration",
    "Campaign_Year",
    "Campaign_Month"
]

print("\n1. NUMERICAL FEATURE RELATIONSHIPS")

correlations = df[numerical_features + [target]].corr(numeric_only=True)[target]
correlations = correlations.drop(target).sort_values(
    key=lambda x: x.abs(),
    ascending=False
)

print(correlations.round(4))

correlations.to_csv(REPORT_DIR / "numerical_feature_correlations.csv")

print("\n2. CATEGORICAL FEATURE ANALYSIS")

for feature in categorical_features:

    summary = (
        df.groupby(feature, dropna=False)
        .agg(
            Campaign_Count=("ROI", "count"),
            Average_ROI=("ROI", "mean"),
            Median_ROI=("ROI", "median")
        )
        .sort_values("Average_ROI", ascending=False)
        .round(4)
    )

    print(f"\n{feature}")
    print(summary.to_string())

    filename = f"{feature.lower()}_roi_analysis.csv"

    summary.to_csv(REPORT_DIR / filename)

print("\n3. FEATURE ANALYSIS COMPLETED")
print("Reports saved in:", REPORT_DIR)
print("=" * 70)