
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

print("=" * 70)
print("MARKETING CAMPAIGN ROI MODEL TRAINING")
print("=" * 70)

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "data" / "processed" / "model_ready_dataset.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load dataset
df = pd.read_csv(DATA_FILE)

print("\n1. DATASET")
print("Shape:", df.shape)

# 2. Select prediction target
TARGET = "ROI"

# Use only features available before campaign launch.
# Exclude post-campaign performance and identifiers.
features = [
    "Campaign_Type",
    "Target_Audience",
    "Duration",
    "Channel_Used",
    "Language",
    "Customer_Segment",
    "Brand",
    "Campaign_Year",
    "Campaign_Month",
    "Campaign_DayOfWeek"
]

X = df[features].copy()
y = df[TARGET].copy()

# 3. Identify feature types
categorical_features = X.select_dtypes(
    include=["object", "str"]
).columns.tolist()

numerical_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

# 4. Preprocessing
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numerical_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42
)

print("\n2. DATA SPLIT")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# 6. Models
models = {
    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
}

results = []
best_model = None
best_score = float("inf")
best_name = None

# 7. Train and evaluate
print("\n3. MODEL TRAINING")

for name, algorithm in models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", algorithm)
    ])

    print(f"\nTraining {name}...")

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print("MAE:", round(mae, 4))
    print("RMSE:", round(rmse, 4))
    print("R2 Score:", round(r2, 4))

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    if mae < best_score:
        best_score = mae
        best_model = pipeline
        best_name = name

# 8. Save results
results_df = pd.DataFrame(results)

results_df = results_df.sort_values("MAE")

results_df.to_csv(
    BASE_DIR / "data" / "processed" / "model_comparison.csv",
    index=False
)

# 9. Save best model
joblib.dump(
    best_model,
    MODEL_DIR / "best_roi_model.pkl"
)

print("\n4. MODEL COMPARISON")
print(results_df.to_string(index=False))

print("\nBest model:", best_name)

print("\nSaved model:")
print(MODEL_DIR / "best_roi_model.pkl")

print("\n" + "=" * 70)
print("MODEL TRAINING COMPLETED")
print("=" * 70)
