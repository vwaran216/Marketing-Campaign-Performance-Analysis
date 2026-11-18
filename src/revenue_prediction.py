import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "processed" / "feature_engineered_dataset.csv"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)


# Load dataset
df = pd.read_csv(INPUT_FILE)

print("Original dataset shape:", df.shape)

# Remove rows where target Revenue is missing
df = df.dropna(subset=["Revenue"]).copy()

print("Dataset after removing missing Revenue:", df.shape)


# Define target
X = df[
    [
        "Campaign_Type",
        "Target_Audience",
        "Duration",
        "Channel_Used",
        "Impressions",
        "Clicks",
        "Leads",
        "Conversions",
        "Acquisition_Cost",
        "Language",
        "Engagement_Score",
        "Customer_Segment",
        "Brand",
        "Campaign_Year",
        "Campaign_Month",
        "Campaign_DayOfWeek",
        "Campaign_Quarter",
        "Campaign_Week",
        "Click_Through_Rate",
        "Conversion_Rate",
        "Lead_Conversion_Rate",
        "Cost_Per_Click",
        "Cost_Per_Lead"
    ]
]

y = df["Revenue"]


# Identify feature types
categorical_features = X.select_dtypes(include=["object", "str"]).columns.tolist()
numerical_features = X.select_dtypes(include=["number"]).columns.tolist()


# Preprocessing
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# Regression models
models = {
    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=30,
        max_depth=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
}


# Train and evaluate models
results = []
trained_models = {}

for name, model in models.items():

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    print(f"\nTraining {name}...")

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    })

    trained_models[name] = pipeline

    print("MAE:", round(mae, 4))
    print("RMSE:", round(rmse, 4))
    print("R2 Score:", round(r2, 4))


# Compare models
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="RMSE")

print("\nRevenue Prediction Model Comparison:")
print(results_df.to_string(index=False))

# Save comparison report
comparison_file = REPORT_DIR / "revenue_model_comparison.csv"
results_df.to_csv(comparison_file, index=False)


# Save best model
best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]

model_file = MODEL_DIR / "best_revenue_model.pkl"
joblib.dump(best_model, model_file)

print("\nBest Model:", best_model_name)
print("Model saved to:", model_file)
print("Comparison report saved to:", comparison_file)

print("\nRevenue Prediction completed successfully.")