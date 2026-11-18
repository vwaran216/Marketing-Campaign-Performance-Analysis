
import pandas as pd
from pathlib import Path

print("=" * 75)
print("MARKETING CAMPAIGN DATASET VALIDATION")
print("=" * 75)

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"

ORIGINAL_FILE = DATA_DIR / "combined_dataset_unmodified.csv"
CLEANED_FILE = DATA_DIR / "cleaned_dataset.csv"
MODEL_FILE = DATA_DIR / "model_ready_dataset.csv"

# --------------------------------------------------
# 2. LOAD DATASETS
# --------------------------------------------------

original = pd.read_csv(ORIGINAL_FILE)
cleaned = pd.read_csv(CLEANED_FILE)
model_df = pd.read_csv(MODEL_FILE)

print("\n1. DATASET OVERVIEW")
print("-" * 50)

for name, data in [
    ("Original", original),
    ("Cleaned", cleaned),
    ("Model-ready", model_df)
]:
    print(f"{name}: {data.shape[0]} rows, {data.shape[1]} columns")

# --------------------------------------------------
# 3. ROW COUNT VALIDATION
# --------------------------------------------------

print("\n2. ROW COUNT VALIDATION")
print("-" * 50)

print("Original rows:", len(original))
print("Cleaned rows:", len(cleaned))
print("Model-ready rows:", len(model_df))

print("Rows removed during cleaning:",
      len(original) - len(cleaned))

print("Rows excluded due to missing ROI:",
      len(cleaned) - len(model_df))

print("Cleaned row count matches original:",
      len(original) == len(cleaned))

# --------------------------------------------------
# 4. MISSING VALUE ANALYSIS
# --------------------------------------------------

print("\n3. MISSING VALUE ANALYSIS")
print("-" * 50)

missing_report = pd.DataFrame({
    "Column": cleaned.columns,
    "Missing_Values": cleaned.isna().sum().values,
    "Missing_Percentage": (
        cleaned.isna().mean().values * 100
    ).round(2)
})

missing_report = missing_report.sort_values(
    by="Missing_Values",
    ascending=False
)

print(missing_report.to_string(index=False))

missing_report.to_csv(
    DATA_DIR / "dataset_missing_value_validation.csv",
    index=False
)

# --------------------------------------------------
# 5. ROI VALIDATION
# --------------------------------------------------

print("\n4. ROI VALIDATION")
print("-" * 50)

print("Original missing ROI:", original["ROI"].isna().sum())
print("Cleaned missing ROI:", cleaned["ROI"].isna().sum())
print("Model-ready missing ROI:", model_df["ROI"].isna().sum())

print("\nOriginal ROI distribution:")
print(original["ROI"].describe())

print("\nCleaned ROI distribution:")
print(cleaned["ROI"].describe())

print("\nModel-ready ROI distribution:")
print(model_df["ROI"].describe())

print("\nROI categories in model-ready dataset:")
print("Negative ROI:", (model_df["ROI"] < 0).sum())
print("Zero ROI:", (model_df["ROI"] == 0).sum())
print("Positive ROI:", (model_df["ROI"] > 0).sum())

# --------------------------------------------------
# 6. BRAND-WISE VALIDATION
# --------------------------------------------------

print("\n5. BRAND-WISE ROI ANALYSIS")
print("-" * 50)

brand_report = model_df.groupby("Brand").agg(
    Campaign_Count=("Campaign_ID", "count"),
    Average_ROI=("ROI", "mean"),
    Median_ROI=("ROI", "median"),
    Minimum_ROI=("ROI", "min"),
    Maximum_ROI=("ROI", "max")
).round(3)

print(brand_report)

brand_report.to_csv(
    DATA_DIR / "brand_validation_report.csv"
)

# --------------------------------------------------
# 7. NUMERICAL DATA VALIDATION
# --------------------------------------------------

print("\n6. NUMERICAL DATA VALIDATION")
print("-" * 50)

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

for column in numeric_columns:
    if column in cleaned.columns:
        print(
            f"{column}: "
            f"Missing={cleaned[column].isna().sum()}, "
            f"Negative={cleaned[column].lt(0).sum()}, "
            f"Zero={cleaned[column].eq(0).sum()}"
        )

# --------------------------------------------------
# 8. CAMPAIGN METRIC CONSISTENCY
# --------------------------------------------------

print("\n7. CAMPAIGN METRIC CONSISTENCY")
print("-" * 50)

checks = {
    "Clicks greater than Impressions":
        cleaned["Clicks"] > cleaned["Impressions"],

    "Leads greater than Clicks":
        cleaned["Leads"] > cleaned["Clicks"],

    "Conversions greater than Leads":
        cleaned["Conversions"] > cleaned["Leads"]
}

for label, condition in checks.items():
    print(f"{label}: {condition.sum()}")

# --------------------------------------------------
# 9. BRAND AND CHANNEL VALIDATION
# --------------------------------------------------

print("\n8. CATEGORICAL VALIDATION")
print("-" * 50)

print("\nBrand counts:")
print(cleaned["Brand"].value_counts())

print("\nCampaign type counts:")
print(cleaned["Campaign_Type"].value_counts())

print("\nUnique standardized channel combinations:",
      cleaned["Channel_Used"].nunique())

print("\nUnknown channel records:",
      (cleaned["Channel_Used"] == "Unknown").sum())

# --------------------------------------------------
# 10. DATE VALIDATION
# --------------------------------------------------

print("\n9. DATE VALIDATION")
print("-" * 50)

date_values = pd.to_datetime(
    cleaned["Date"],
    errors="coerce"
)

print("Valid dates:", date_values.notna().sum())
print("Invalid/missing dates:", date_values.isna().sum())

if date_values.notna().any():
    print("Earliest date:", date_values.min())
    print("Latest date:", date_values.max())

# --------------------------------------------------
# 11. DUPLICATE VALIDATION
# --------------------------------------------------

print("\n10. DUPLICATE VALIDATION")
print("-" * 50)

print("Cleaned duplicate rows:",
      cleaned.duplicated().sum())

print("Model-ready duplicate rows:",
      model_df.duplicated().sum())

print("Duplicate non-null Campaign IDs:",
      cleaned["Campaign_ID"].dropna().duplicated().sum())

# --------------------------------------------------
# 12. FINAL VALIDATION SUMMARY
# --------------------------------------------------

print("\n11. FINAL VALIDATION SUMMARY")
print("=" * 75)

print("Original dataset:", original.shape)
print("Cleaned dataset:", cleaned.shape)
print("Model-ready dataset:", model_df.shape)

print("Missing ROI records:", cleaned["ROI"].isna().sum())
print("Remaining missing numerical values:",
      cleaned[numeric_columns].isna().sum().sum())

print("\nValidation reports saved:")
print(DATA_DIR / "dataset_missing_value_validation.csv")
print(DATA_DIR / "brand_validation_report.csv")

print("\n" + "=" * 75)
print("DATASET VALIDATION COMPLETED")
print("=" * 75)
