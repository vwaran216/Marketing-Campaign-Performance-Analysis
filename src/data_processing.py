import pandas as pd
from pathlib import Path

print("=" * 70)
print("MARKETING CAMPAIGN DATA PREPROCESSING")
print("=" * 70)

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "processed" / "combined_dataset_unmodified.csv"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

CLEANED_FILE = OUTPUT_DIR / "cleaned_dataset.csv"
MODEL_FILE = OUTPUT_DIR / "model_ready_dataset.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 1. LOAD ORIGINAL DATA
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print("\n1. ORIGINAL DATASET")
print("Rows:", len(df))
print("Columns:", len(df.columns))

# Preserve original column names but remove accidental spaces
df.columns = df.columns.str.strip()

# Clean whitespace in text columns
for column in df.select_dtypes(include="object").columns:
    df[column] = df[column].str.strip()

# Convert empty strings into missing values
df.replace(r"^\s*$", pd.NA, regex=True, inplace=True)

# --------------------------------------------------
# 2. REMOVE DUPLICATES
# --------------------------------------------------

before = len(df)

df.drop_duplicates(inplace=True)

print("\n2. DUPLICATE CHECK")
print("Duplicate rows removed:", before - len(df))

# --------------------------------------------------
# 3. STANDARDIZE CHANNEL COMBINATIONS
# --------------------------------------------------

def standardize_channels(value):
    if pd.isna(value):
        return "Unknown"

    channels = [
        item.strip()
        for item in str(value).split(",")
        if item.strip()
    ]

    channels = sorted(set(channels))

    return ", ".join(channels) if channels else "Unknown"


df["Channel_Used"] = df["Channel_Used"].apply(
    standardize_channels
)

# --------------------------------------------------
# 4. HANDLE MISSING CATEGORICAL VALUES
# --------------------------------------------------

categorical_columns = [
    "Campaign_Type",
    "Target_Audience",
    "Channel_Used",
    "Language",
    "Customer_Segment"
]

for column in categorical_columns:
    df[column] = df[column].fillna("Unknown")

# --------------------------------------------------
# 5. HANDLE NUMERICAL VALUES
# --------------------------------------------------

# Duration is known before a campaign starts.
# Use median imputation rather than zero.

duration_median = df["Duration"].median()

df["Duration"] = df["Duration"].fillna(duration_median)

# Do not impute ROI, revenue, acquisition cost,
# impressions, clicks, leads or conversions.

# --------------------------------------------------
# 6. PROCESS CAMPAIGN DATES
# --------------------------------------------------

df["Date"] = pd.to_datetime(
    df["Date"],
    format="%d-%m-%Y",
    errors="coerce"
)

df["Campaign_Year"] = df["Date"].dt.year
df["Campaign_Month"] = df["Date"].dt.month
df["Campaign_Day"] = df["Date"].dt.day
df["Campaign_DayOfWeek"] = df["Date"].dt.dayofweek

# --------------------------------------------------
# 7. CHECK CLEANED DATA
# --------------------------------------------------

print("\n3. CLEANED DATA QUALITY")

print("Total rows:", len(df))
print("Missing ROI:", df["ROI"].isna().sum())
print("Missing Duration:", df["Duration"].isna().sum())

print("\nMissing values in categorical columns:")

for column in categorical_columns:
    print(column, ":", df[column].isna().sum())

print("\nDate records:")
print("Valid dates:", df["Date"].notna().sum())
print("Invalid/missing dates:", df["Date"].isna().sum())

# --------------------------------------------------
# 8. SAVE CLEANED DATASET
# --------------------------------------------------

df.to_csv(CLEANED_FILE, index=False)

print("\n4. CLEANED DATASET SAVED")
print(CLEANED_FILE)

# --------------------------------------------------
# 9. CREATE MODEL-READY DATASET
# --------------------------------------------------

model_df = df.dropna(subset=["ROI"]).copy()

print("\n5. MODEL-READY DATASET")
print("Rows with valid ROI:", len(model_df))
print("Rows excluded due to missing ROI:", len(df) - len(model_df))

model_df.to_csv(MODEL_FILE, index=False)

print("\nModel-ready dataset saved:")
print(MODEL_FILE)

# --------------------------------------------------
# 10. FINAL SUMMARY
# --------------------------------------------------

print("\n6. FINAL SUMMARY")
print("Original rows:", len(pd.read_csv(INPUT_FILE)))
print("Cleaned rows:", len(df))
print("Model-ready rows:", len(model_df))
print("Brands:", df["Brand"].nunique())
print("Campaign types:", df["Campaign_Type"].nunique())
print("Channel combinations:", df["Channel_Used"].nunique())

print("\n" + "=" * 70)
print("DATA PREPROCESSING COMPLETED")
print("=" * 70)