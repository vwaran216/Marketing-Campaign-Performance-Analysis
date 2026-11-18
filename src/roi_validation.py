import pandas as pd
import numpy as np
from pathlib import Path

# Project paths
ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    ROOT / "data" / "processed"
    / "combined_dataset_unmodified.csv"
)

OUTPUT_PATH = (
    ROOT / "data" / "processed"
    / "roi_validation_report.csv"
)

print("=" * 65)
print("ROI VALIDATION AND CHANNEL INVESTIGATION")
print("=" * 65)

# Load original combined dataset
df = pd.read_csv(DATA_PATH)

# --------------------------------------------------
# 1. ROI VALIDATION
# --------------------------------------------------

print("\n1. ROI VALIDATION")

valid = df.dropna(
    subset=["Revenue", "Acquisition_Cost", "ROI"]
).copy()

valid = valid[valid["Acquisition_Cost"] > 0]

# Common ROI formula candidates
valid["ROI_Candidate_1"] = (
    (valid["Revenue"] - valid["Acquisition_Cost"])
    / valid["Acquisition_Cost"]
)

valid["ROI_Candidate_2"] = (
    valid["Revenue"] / valid["Acquisition_Cost"]
)

valid["ROI_Candidate_3"] = (
    (valid["Revenue"] - valid["Acquisition_Cost"])
    / valid["Revenue"]
)

print("Records available for ROI validation:", len(valid))

for candidate in [
    "ROI_Candidate_1",
    "ROI_Candidate_2",
    "ROI_Candidate_3"
]:
    actual = valid["ROI"]
    predicted = valid[candidate]

    mae = np.mean(np.abs(actual - predicted))

    correlation = actual.corr(predicted)

    within_tolerance = np.isclose(
        actual,
        predicted,
        rtol=0.01,
        atol=0.01
    ).mean() * 100

    print(f"\n{candidate}")
    print("Mean Absolute Error:", round(mae, 4))
    print("Correlation:", round(correlation, 4))
    print("Matches within tolerance (%):",
          round(within_tolerance, 2))

# ROI distribution
print("\n2. ROI DISTRIBUTION")

print(df["ROI"].describe().round(4))

print("\nROI by brand:")

print(
    df.groupby("Brand")["ROI"]
    .agg(["count", "mean", "median", "min", "max"])
    .round(3)
)

print("\nZero ROI records:",
      (df["ROI"] == 0).sum())

print("Negative ROI records:",
      (df["ROI"] < 0).sum())

print("Positive ROI records:",
      (df["ROI"] > 0).sum())

# --------------------------------------------------
# 2. CHANNEL INVESTIGATION
# --------------------------------------------------

print("\n3. CHANNEL INVESTIGATION")

channel_data = df["Channel_Used"].dropna().astype(str)

channel_sets = channel_data.apply(
    lambda x: sorted(
        set(
            item.strip()
            for item in x.split(",")
            if item.strip()
        )
    )
)

all_channels = sorted({
    channel
    for channels in channel_sets
    for channel in channels
})

print("\nIndividual marketing channels:")
print(all_channels)

print("\nTotal individual channels:", len(all_channels))

print("\nMost frequent channel combinations:")

print(
    channel_data.value_counts().head(20)
)

# Compare unique raw combinations with canonical combinations
canonical_channels = channel_sets.apply(
    lambda channels: ", ".join(channels)
)

print("\nOriginal unique combinations:",
      channel_data.nunique())

print("Canonical unique combinations:",
      canonical_channels.nunique())

# --------------------------------------------------
# 3. DATE INVESTIGATION
# --------------------------------------------------

print("\n4. DATE INVESTIGATION")

dates = pd.to_datetime(
    df["Date"],
    format="%d-%m-%Y",
    errors="coerce"
)

print("Valid dates:", dates.notna().sum())
print("Invalid or missing dates:", dates.isna().sum())

print("Earliest date:", dates.min())
print("Latest date:", dates.max())

# --------------------------------------------------
# 4. SAVE ROI AUDIT
# --------------------------------------------------

valid[
    [
        "Brand",
        "Revenue",
        "Acquisition_Cost",
        "ROI",
        "ROI_Candidate_1",
        "ROI_Candidate_2",
        "ROI_Candidate_3"
    ]
].head(1000).to_csv(OUTPUT_PATH, index=False)

print("\nROI audit sample saved to:")
print(OUTPUT_PATH)

print("\n" + "=" * 65)
print("INVESTIGATION COMPLETED")
print("=" * 65)