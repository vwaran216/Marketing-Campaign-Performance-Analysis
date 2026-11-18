import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "processed" / "combined_dataset_unmodified.csv"

OUTPUT_PATH = ROOT / "data" / "processed" / "roi_diagnostic_report.csv"

print("=" * 70)
print("DEEP ROI FORMULA DIAGNOSTIC")
print("=" * 70)

# Load original dataset
df = pd.read_csv(DATA_PATH)

# Convert relevant columns to numeric
for col in ["Revenue", "Acquisition_Cost", "ROI"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

valid = df.dropna(
    subset=["Revenue", "Acquisition_Cost", "ROI"]
).copy()

valid = valid[valid["Acquisition_Cost"] > 0].copy()

# --------------------------------------------------
# 1. SOURCE DATA DISTRIBUTION
# --------------------------------------------------

print("\n1. SOURCE DATA DISTRIBUTION")

for col in ["Revenue", "Acquisition_Cost", "ROI"]:
    print(f"\n{col}")
    print(valid[col].describe(
        percentiles=[0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99]
    ).round(4))

# --------------------------------------------------
# 2. FORMULA INVESTIGATION
# --------------------------------------------------

print("\n2. ROI FORMULA INVESTIGATION")

valid["Candidate_1"] = (
    (valid["Revenue"] - valid["Acquisition_Cost"])
    / valid["Acquisition_Cost"]
)

valid["Candidate_2"] = (
    valid["Revenue"] / valid["Acquisition_Cost"]
)

valid["Candidate_3"] = (
    (valid["Revenue"] - valid["Acquisition_Cost"])
    / valid["Revenue"].replace(0, np.nan)
)

candidates = [
    "Candidate_1",
    "Candidate_2",
    "Candidate_3"
]

for candidate in candidates:

    print(f"\n{candidate}")

    predicted = valid[candidate]
    actual = valid["ROI"]

    for scale in [0.01, 1, 100]:

        scaled = predicted * scale

        comparison = pd.DataFrame({
            "actual": actual,
            "predicted": scaled
        }).dropna()

        mae = np.mean(
            np.abs(
                comparison["actual"] - comparison["predicted"]
            )
        )

        correlation = comparison["actual"].corr(
            comparison["predicted"]
        )

        matches = np.isclose(
            comparison["actual"],
            comparison["predicted"],
            rtol=0.01,
            atol=0.01
        ).mean() * 100

        print(
            f"Scale {scale}: "
            f"MAE={mae:.4f}, "
            f"Correlation={correlation:.4f}, "
            f"Matches={matches:.2f}%"
        )

# --------------------------------------------------
# 3. ROI AND REVENUE RELATIONSHIP
# --------------------------------------------------

print("\n3. ROI RELATIONSHIP ANALYSIS")

print(
    valid[
        ["ROI", "Revenue", "Acquisition_Cost"]
    ].corr().round(4)
)

# --------------------------------------------------
# 4. SAMPLE RECORD INVESTIGATION
# --------------------------------------------------

print("\n4. SAMPLE RECORD INVESTIGATION")

sample_columns = [
    col for col in [
        "Brand",
        "Campaign_ID",
        "Campaign_Type",
        "Channel_Used",
        "Revenue",
        "Acquisition_Cost",
        "ROI",
        "Candidate_1",
        "Candidate_2",
        "Candidate_3"
    ]
    if col in valid.columns
]

print("\nFirst 15 records:")
print(valid[sample_columns].head(15).to_string(index=False))

# --------------------------------------------------
# 5. LARGEST ROI DISCREPANCIES
# --------------------------------------------------

print("\n5. LARGEST ROI DISCREPANCIES")

valid["Candidate_1_Error"] = (
    valid["ROI"] - valid["Candidate_1"]
).abs()

valid["Candidate_3_Error"] = (
    valid["ROI"] - valid["Candidate_3"]
).abs()

print("\nTop 15 Candidate 1 discrepancies:")

print(
    valid.nlargest(15, "Candidate_1_Error")[
        sample_columns + ["Candidate_1_Error"]
    ].to_string(index=False)
)

# --------------------------------------------------
# 6. SAVE DIAGNOSTIC REPORT
# --------------------------------------------------

valid.to_csv(OUTPUT_PATH, index=False)

print("\nDiagnostic report saved to:")
print(OUTPUT_PATH)

print("\n" + "=" * 70)
print("DEEP ROI DIAGNOSTIC COMPLETED")
print("=" * 70)