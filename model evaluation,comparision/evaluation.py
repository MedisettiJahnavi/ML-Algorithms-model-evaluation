"""
Model Evaluation & Comparison
- Classification: Accuracy, Precision, Recall, Confusion Matrix
- Regression: MAE, MSE, RMSE, R2
Run: python evaluate_models.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, confusion_matrix,
    ConfusionMatrixDisplay, mean_absolute_error, mean_squared_error, r2_score,
)
CLASSIFICATION_CSV = None        # e.g. "task2_data.csv"
CLASSIFICATION_TARGET = None     # e.g. "Survived"
REGRESSION_CSV = None            # e.g. "task3_data.csv"
REGRESSION_TARGET = None         # e.g. "Price"
TEST_SIZE = 0.2
SEED = 42
def load_xy(csv_path, target, fallback):
    """Load X, y from a CSV (with simple cleaning) or from a sklearn dataset."""
    if csv_path:
        df = pd.read_csv(csv_path).dropna()
        y = df[target]
        X = pd.get_dummies(df.drop(columns=[target]), drop_first=True)
        return X, y
    data = fallback()
    return pd.DataFrame(data.data, columns=data.feature_names), pd.Series(data.target)
# PART 1: CLASSIFICATION
Xc, yc = load_xy(CLASSIFICATION_CSV, CLASSIFICATION_TARGET, load_breast_cancer)
Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    Xc, yc, test_size=TEST_SIZE, random_state=SEED, stratify=yc
)
clf_models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "Decision Tree": DecisionTreeClassifier(random_state=SEED),
    "Random Forest": RandomForestClassifier(random_state=SEED),
}
avg = "binary" if yc.nunique() == 2 else "weighted"
clf_rows = []
fig, axes = plt.subplots(1, len(clf_models), figsize=(5 * len(clf_models), 4))
for ax, (name, model) in zip(np.atleast_1d(axes), clf_models.items()):
    model.fit(Xc_train, yc_train)
    pred = model.predict(Xc_test)
    acc = accuracy_score(yc_test, pred)
    prec = precision_score(yc_test, pred, average=avg, zero_division=0)
    rec = recall_score(yc_test, pred, average=avg, zero_division=0)
    cm = confusion_matrix(yc_test, pred)

    print(f"\n=== {name} ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print("Confusion Matrix:\n", cm)

    ConfusionMatrixDisplay(cm).plot(ax=ax, colorbar=False)
    ax.set_title(name)

    clf_rows.append({
        "Model": name,
        "Accuracy (%)": round(acc * 100, 2),
        "Precision (%)": round(prec * 100, 2),
        "Recall (%)": round(rec * 100, 2),
    })

plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150)
clf_table = pd.DataFrame(clf_rows).sort_values("Accuracy (%)", ascending=False)
print("\n\n##### CLASSIFICATION MODEL COMPARISON #####")
print(clf_table.to_string(index=False))
clf_table.to_csv("classification_comparison.csv", index=False)
# PART 2: REGRESSION
Xr, yr = load_xy(REGRESSION_CSV, REGRESSION_TARGET, load_diabetes)
Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    Xr, yr, test_size=TEST_SIZE, random_state=SEED
)
reg_models = {
    "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
    "Decision Tree Regressor": DecisionTreeRegressor(random_state=SEED),
    "Random Forest Regressor": RandomForestRegressor(random_state=SEED),
}
reg_rows = []
for name, model in reg_models.items():
    model.fit(Xr_train, yr_train)
    pred = model.predict(Xr_test)

    mae = mean_absolute_error(yr_test, pred)
    mse = mean_squared_error(yr_test, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(yr_test, pred)

    print(f"\n=== {name} ===")
    print(f"MAE : {mae:.4f}")
    print(f"MSE : {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2  : {r2:.4f}")

    reg_rows.append({
        "Model": name, "MAE": round(mae, 4), "MSE": round(mse, 4),
        "RMSE": round(rmse, 4), "R2 Score": round(r2, 4),
    })
reg_table = pd.DataFrame(reg_rows).sort_values("R2 Score", ascending=False)
print("\n\n##### REGRESSION MODEL COMPARISON #####")
print(reg_table.to_string(index=False))
reg_table.to_csv("regression_comparison.csv", index=False)
# BEST MODELS
print("\nBest classification model:", clf_table.iloc[0]["Model"])
print("Best regression model    :", reg_table.iloc[0]["Model"])
plt.show()