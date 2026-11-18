import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "processed" / "feature_engineered_dataset.csv"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)


# Load dataset
df = pd.read_csv(INPUT_FILE)

print("Dataset shape:", df.shape)

# Define target and remove target-related leakage
X = df[
    [
        "Campaign_Type",
        "Target_Audience",
        "Duration",
        "Channel_Used",
        "Language",
        "Customer_Segment",
        "Brand",
        "Campaign_Year",
        "Campaign_Month",
        "Campaign_DayOfWeek",
        "Campaign_Quarter",
        "Campaign_Week"
    ]
].copy()

y = df["Profit_Loss"].map({
    "Profit": 1,
    "Loss_or_BreakEven": 0
})

# Remove rows with invalid target labels, if any
valid_rows = y.notna()
X = X.loc[valid_rows]
y = y.loc[valid_rows].astype(int)

print("\nTarget distribution:")
print(y.value_counts().rename(index={1: "Profit", 0: "Loss_or_BreakEven"}))


# Identify feature types
categorical_features = X.select_dtypes(include=["object", "str"]).columns.tolist()
numerical_features = X.select_dtypes(include=["number"]).columns.tolist()


# Preprocessing
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
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
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# Lightweight classification models
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=500,
        class_weight="balanced"
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=8,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42
    )
}


# Train and evaluate
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

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    trained_models[name] = pipeline

    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=["Loss_or_BreakEven", "Profit"],
        zero_division=0
    ))


# Compare models
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="F1 Score", ascending=False)

print("\nClassification Model Comparison:")
print(results_df.to_string(index=False))


# Save comparison report
comparison_file = REPORT_DIR / "profitability_model_comparison.csv"
results_df.to_csv(comparison_file, index=False)


# Save best model
best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]

model_file = MODEL_DIR / "best_profitability_model.pkl"
joblib.dump(best_model, model_file)

print("\nBest Model:", best_model_name)
print("Model saved to:", model_file)
print("Comparison report saved to:", comparison_file)

print("\nProfit/Loss Classification completed successfully.")