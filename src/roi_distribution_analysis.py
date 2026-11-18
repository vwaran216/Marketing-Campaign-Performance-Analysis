import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "data" / "processed" / "model_ready_dataset.csv"
REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_FILE)
roi = df["ROI"].dropna()

print("=" * 70)
print("ROI DISTRIBUTION ANALYSIS")
print("=" * 70)

print("\n1. ROI STATISTICS")

print(roi.describe().round(4))

print("\n2. ROI PERCENTILES")

print(
    roi.quantile(
        [0, 0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 1]
    ).round(4)
)

print("\n3. ROI DISTRIBUTION CATEGORIES")

distribution = pd.Series({
    "Negative ROI": (roi < 0).sum(),
    "Zero ROI": (roi == 0).sum(),
    "ROI between 0 and 1": ((roi > 0) & (roi < 1)).sum(),
    "ROI between 1 and 5": ((roi >= 1) & (roi < 5)).sum(),
    "ROI between 5 and 10": ((roi >= 5) & (roi < 10)).sum(),
    "ROI above 10": (roi >= 10).sum()
})

print(distribution)

distribution.to_csv(
    REPORT_DIR / "roi_distribution_categories.csv"
)

print("\n4. OUTLIER ANALYSIS")

q1 = roi.quantile(0.25)
q3 = roi.quantile(0.75)
iqr = q3 - q1

lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

outliers = roi[(roi < lower) | (roi > upper)]

print("Lower boundary:", round(lower, 4))
print("Upper boundary:", round(upper, 4))
print("Potential outliers:", len(outliers))
print("Outlier percentage:", round(len(outliers) / len(roi) * 100, 2), "%")

print("\n5. SAVING VISUALIZATIONS")

plt.figure(figsize=(10, 5))
sns.histplot(roi, bins=80, kde=True)
plt.title("ROI Distribution")
plt.xlabel("ROI")
plt.ylabel("Campaign Count")
plt.tight_layout()
plt.savefig(REPORT_DIR / "roi_distribution.png", dpi=300)
plt.savefig(REPORT_DIR / "roi_distribution.png", dpi=300)
plt.close()

plt.figure(figsize=(10, 4))
sns.boxplot(x=roi)
plt.title("ROI Boxplot")
plt.xlabel("ROI")
plt.tight_layout()
plt.savefig(REPORT_DIR / "roi_boxplot.png", dpi=300)
plt.savefig(REPORT_DIR / "roi_boxplot.png", dpi=300)
plt.close()

print("\nROI DISTRIBUTION ANALYSIS COMPLETED")
print("Reports saved in:", REPORT_DIR)