import pandas as pd
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "data" / "processed" / "model_ready_dataset.csv"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "feature_engineered_dataset.csv"

# Load dataset
df = pd.read_csv(INPUT_FILE)

print("Original dataset shape:", df.shape)

# 1. Create campaign date features
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

df["Campaign_Quarter"] = df["Date"].dt.quarter
df["Campaign_Week"] = df["Date"].dt.isocalendar().week.astype("Int64")

# 2. Create campaign performance features
df["Click_Through_Rate"] = (
    df["Clicks"] / df["Impressions"].replace(0, pd.NA)
)

df["Conversion_Rate"] = (
    df["Conversions"] / df["Clicks"].replace(0, pd.NA)
)

df["Lead_Conversion_Rate"] = (
    df["Conversions"] / df["Leads"].replace(0, pd.NA)
)

# 3. Create cost and revenue features
df["Cost_Per_Click"] = (
    df["Acquisition_Cost"] / df["Clicks"].replace(0, pd.NA)
)

df["Cost_Per_Lead"] = (
    df["Acquisition_Cost"] / df["Leads"].replace(0, pd.NA)
)

df["Revenue_Per_Conversion"] = (
    df["Revenue"] / df["Conversions"].replace(0, pd.NA)
)

# 4. Create profitability category using the supplied ROI
df["Profit_Loss"] = df["ROI"].apply(
    lambda value: "Profit" if value > 0 else "Loss_or_BreakEven"
)

# Save feature-engineered dataset
df.to_csv(OUTPUT_FILE, index=False)

print("\nFeature engineering completed.")
print("New dataset shape:", df.shape)
print("Output saved to:", OUTPUT_FILE)

print("\nNew features:")
print([
    "Campaign_Quarter",
    "Campaign_Week",
    "Click_Through_Rate",
    "Conversion_Rate",
    "Lead_Conversion_Rate",
    "Cost_Per_Click",
    "Cost_Per_Lead",
    "Revenue_Per_Conversion",
    "Profit_Loss"
])

print("\nProfit/Loss distribution:")
print(df["Profit_Loss"].value_counts())