
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    ROOT / "data" / "processed"
    / "combined_dataset_unmodified.csv"
)

OUTPUT_DIR = ROOT / "data" / "processed"

df = pd.read_csv(DATA_PATH)

print("=" * 70)
print("MARKETING CAMPAIGN DATA QUALITY ANALYSIS")
print("=" * 70)

# --------------------------------------------------
# 1. DATASET OVERVIEW
# --------------------------------------------------

print("\n1. DATASET OVERVIEW")

print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Duplicate complete rows:", df.duplicated().sum())

# Campaign ID investigation
if "Campaign_ID" in df.columns:
    print(
        "\nDuplicate non-null Campaign IDs:",
        df["Campaign_ID"].dropna().duplicated().sum()
    )

# --------------------------------------------------
# 2. NUMERICAL DATA QUALITY
# --------------------------------------------------

print("\n2. NUMERICAL DATA QUALITY")

numeric_columns = [
    "Duration",
    "Impressions",
    "Clicks",
    "Leads",
    "Conversions",
    "Revenue",
    "Acquisition_Cost",
    "ROI",
    "Engagement_Score"
]

numeric_columns = [
    col for col in numeric_columns if col in df.columns
]

quality_rows = []

for col in numeric_columns:

    values = pd.to_numeric(df[col], errors="coerce")

    quality_rows.append({
        "Column": col,
        "Missing": values.isna().sum(),
        "Zero": (values == 0).sum(),
        "Negative": (values < 0).sum(),
        "Minimum": values.min(),
        "Median": values.median(),
        "Maximum": values.max()
    })

quality_df = pd.DataFrame(quality_rows)

print(quality_df.round(3).to_string(index=False))

quality_df.to_csv(
    OUTPUT_DIR / "numeric_quality_report.csv",
    index=False
)

# --------------------------------------------------
# 3. CAMPAIGN METRIC CONSISTENCY
# --------------------------------------------------

print("\n3. CAMPAIGN METRIC CONSISTENCY")

checks = {
    "Clicks greater than Impressions":
        ("Clicks", "Impressions"),

    "Leads greater than Clicks":
        ("Leads", "Clicks"),

    "Conversions greater than Leads":
        ("Conversions", "Leads")
}

for description, (first, second) in checks.items():

    first_values = pd.to_numeric(df[first], errors="coerce")
    second_values = pd.to_numeric(df[second], errors="coerce")

    count = (
        (first_values > second_values)
        & first_values.notna()
        & second_values.notna()
    ).sum()

    print(f"{description}: {count}")

# --------------------------------------------------
# 4. CATEGORICAL QUALITY
# --------------------------------------------------

print("\n4. CATEGORICAL QUALITY")

categorical_columns = [
    "Brand",
    "Campaign_Type",
    "Target_Audience",
    "Channel_Used",
    "Language",
    "Customer_Segment"
]

for col in categorical_columns:

    if col in df.columns:

        print(f"\n{col}")
        print("Unique values:", df[col].nunique())
        print("Missing values:", df[col].isna().sum())

        print(
            df[col].value_counts(dropna=False).head(10)
        )

# --------------------------------------------------
# 5. DATE QUALITY
# --------------------------------------------------

print("\n5. DATE QUALITY")

dates = pd.to_datetime(
    df["Date"],
    format="%d-%m-%Y",
    errors="coerce"
)

print("Valid dates:", dates.notna().sum())
print("Invalid dates:", dates.isna().sum())

# --------------------------------------------------
# 6. SAVE SUMMARY
# --------------------------------------------------

print("\n6. REPORT SAVING")

print(
    "Numeric report:",
    OUTPUT_DIR / "numeric_quality_report.csv"
)

print("\n" + "=" * 70)
print("DATA QUALITY ANALYSIS COMPLETED")
print("=" * 70)
