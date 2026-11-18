import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    ROOT / "data" / "processed"
    / "combined_dataset_unmodified.csv"
)

df = pd.read_csv(DATA_PATH)

print("=" * 70)
print("ROI PATTERN ANALYSIS")
print("=" * 70)

# 1. Column information
print("\n1. DATASET COLUMNS")

print(df.columns.tolist())

# 2. ROI relationship with numeric columns
print("\n2. NUMERIC CORRELATION WITH ROI")

numeric = df.select_dtypes(include=np.number)

correlations = (
    numeric.corr()["ROI"]
    .drop("ROI")
    .sort_values(ascending=False)
)

print(correlations.to_string())

# 3. ROI by campaign type
print("\n3. ROI BY CAMPAIGN TYPE")

if "Campaign_Type" in df.columns:
    print(
        df.groupby("Campaign_Type")["ROI"]
        .agg(["count", "mean", "median", "min", "max"])
        .round(3)
    )

# 4. ROI by channel
print("\n4. ROI BY CHANNEL")

if "Channel_Used" in df.columns:
    print(
        df.groupby("Channel_Used")["ROI"]
        .agg(["count", "mean", "median"])
        .sort_values("mean", ascending=False)
        .head(20)
        .round(3)
    )

# 5. ROI by brand
print("\n5. ROI BY BRAND")

print(
    df.groupby("Brand")["ROI"]
    .agg(["count", "mean", "median", "std"])
    .round(3)
)

# 6. Unique ROI values
print("\n6. ROI VALUE PATTERN")

print("Unique ROI values:", df["ROI"].nunique())

print("\nMost frequent ROI values:")

print(df["ROI"].value_counts().head(20))

# 7. Missing values
print("\n7. MISSING VALUE ANALYSIS")

print(
    df.isnull().sum()
    .sort_values(ascending=False)
    .to_string()
)

print("\n" + "=" * 70)
print("ROI PATTERN ANALYSIS COMPLETED")
print("=" * 70)