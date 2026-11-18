import pandas as pd
from pathlib import Path

# Project paths
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

PROCESSED.mkdir(parents=True, exist_ok=True)

FILES = {
    "Nykaa": "nykaa_campaign_data_with_nulls.csv",
    "Purplle": "purplle_campaign_data_with_nulls.csv",
    "Tira": "tira_campaign_data_with_nulls.csv"
}

print("=" * 65)
print("MARKETING CAMPAIGN - ADVANCED DATA QUALITY ANALYSIS")
print("=" * 65)

# Load datasets
datasets = []

for brand, filename in FILES.items():

    path = RAW / filename

    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")

    df = pd.read_csv(path)

    df["Brand"] = brand

    datasets.append(df)

    print(f"\n{brand}: {len(df):,} records loaded")

# Combine datasets
combined_df = pd.concat(datasets, ignore_index=True)

print("\n1. COMBINED DATASET")
print("Shape:", combined_df.shape)

# Preserve original combined data without cleaning
output_path = PROCESSED / "combined_dataset_unmodified.csv"
combined_df.to_csv(output_path, index=False)

print("Saved:", output_path)

# Missing value percentages
print("\n2. MISSING VALUE ANALYSIS")

missing_report = pd.DataFrame({
    "Missing_Count": combined_df.isnull().sum(),
    "Missing_Percentage": (
        combined_df.isnull().mean() * 100
    ).round(2)
})

print(missing_report.to_string())

missing_report.to_csv(
    PROCESSED / "missing_value_report.csv"
)

# Numeric statistics
print("\n3. NUMERIC DATA SUMMARY")

print(
    combined_df.select_dtypes(include="number")
    .describe()
    .round(2)
    .to_string()
)

# Brand-wise distribution
print("\n4. BRAND DISTRIBUTION")

print(combined_df["Brand"].value_counts())

# Campaign type analysis
print("\n5. CAMPAIGN TYPE DISTRIBUTION")

print(combined_df["Campaign_Type"].value_counts())

# Channel analysis
print("\n6. CHANNEL ANALYSIS")

print("Unique channel combinations:",
      combined_df["Channel_Used"].nunique())

print("\nTop 20 channel values:")

print(
    combined_df["Channel_Used"]
    .value_counts()
    .head(20)
    .to_string()
)

print("\nSample channel values:")

print(
    combined_df["Channel_Used"]
    .dropna()
    .unique()[:30]
)

# ROI analysis
print("\n7. ROI ANALYSIS")

print(combined_df["ROI"].describe().round(3))

print("\nROI sample values:")

print(combined_df["ROI"].dropna().head(20).tolist())

# Date analysis
print("\n8. DATE ANALYSIS")

print("Sample dates:")
print(combined_df["Date"].dropna().head(10).tolist())

print("Unique dates:",
      combined_df["Date"].nunique())

# Business logic validation
print("\n9. BUSINESS LOGIC CHECKS")

checks = {
    "Negative Impressions": combined_df["Impressions"] < 0,
    "Negative Clicks": combined_df["Clicks"] < 0,
    "Negative Leads": combined_df["Leads"] < 0,
    "Negative Conversions": combined_df["Conversions"] < 0,
    "Negative Revenue": combined_df["Revenue"] < 0,
    "Negative Acquisition Cost": combined_df["Acquisition_Cost"] < 0,
    "Clicks Greater Than Impressions": (
        combined_df["Clicks"] > combined_df["Impressions"]
    ),
    "Conversions Greater Than Clicks": (
        combined_df["Conversions"] > combined_df["Clicks"]
    )
}

for name, condition in checks.items():
    print(f"{name}: {int(condition.fillna(False).sum()):,}")

# Duplicate check
print("\n10. DUPLICATE CHECK")

print("Exact duplicate rows:",
      combined_df.duplicated().sum())

print("\n" + "=" * 65)
print("ADVANCED DATA QUALITY ANALYSIS COMPLETED")
print("=" * 65)