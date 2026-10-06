# MACHINE LEARNING MODEL COMPARISON
# Project: Iris Flower Classification
# 1. IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)
# 2. PROBLEM DEFINITION
print("=" * 60)
print("MACHINE LEARNING MODEL COMPARISON")
print("Iris Flower Classification")
print("=" * 60)
print("\nProblem Statement:")
print("The objective is to classify iris flowers into three species")
print("using their sepal and petal measurements.")
# 3. LOAD DATASET
iris = load_iris()
# Create DataFrame
df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)
# Add target column
df["target"] = iris.target

# Add flower species names
df["species"] = df["target"].map({
    0: "setosa",
    1: "versicolor",
    2: "virginica"
})
# 4. DATASET DESCRIPTION
print("\nDataset Shape:")
print(df.shape)
print("\nFirst 5 Rows:")
print(df.head())
print("\nDataset Information:")
print(df.info())
print("\nStatistical Description:")
print(df.describe())
print("\nClass Distribution:")
print(df["species"].value_counts())
# 5. DATA CLEANING
print("\nMissing Values:")
print(df.isnull().sum())
print("\nDuplicate Rows:")
print(df.duplicated().sum())
# Remove duplicate rows if present
df = df.drop_duplicates()
print("\nData cleaning completed.")
# 6. DATA ANALYSIS
# Visualization 1: Species Distribution
plt.figure(figsize=(7, 5))
sns.countplot(
    data=df,
    x="species"
)
plt.title("Iris Species Distribution")
plt.xlabel("Species")
plt.ylabel("Number of Samples")
plt.tight_layout()
plt.show()
# Visualization 2: Scatter Plot
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df,
    x="petal length (cm)",
    y="petal width (cm)",
    hue="species",
    s=80
)
plt.title("Petal Length vs Petal Width")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.tight_layout()
plt.show()
# Visualization 3: Correlation Heatmap
plt.figure(figsize=(8, 6))

correlation = df[
    [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)"
    ]
].corr()

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm"
)

plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()
# 7. FEATURE SELECTION
features = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)"
]
X = df[features]
y = df["target"]
print("\nSelected Features:")
print(features)
print("\nTarget:")
print("Iris flower species")
# 8. TRAIN / TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))
# 9. FEATURE SCALING
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# 10. MODEL 1 - LOGISTIC REGRESSION
model1 = LogisticRegression(
    max_iter=200
)

model1.fit(
    X_train_scaled,
    y_train
)

# Prediction
y_pred1 = model1.predict(X_test_scaled)
# 11. MODEL 2 - DECISION TREE CLASSIFIER
model2 = DecisionTreeClassifier(
    random_state=42,
    max_depth=4
)

model2.fit(
    X_train,
    y_train
)

# Prediction
y_pred2 = model2.predict(X_test)
# 12. MODEL EVALUATION
# Model 1 metrics

accuracy1 = accuracy_score(y_test, y_pred1)

precision1 = precision_score(
    y_test,
    y_pred1,
    average="weighted"
)

recall1 = recall_score(
    y_test,
    y_pred1,
    average="weighted"
)
# Model 2 metrics

accuracy2 = accuracy_score(y_test, y_pred2)
precision2 = precision_score(
    y_test,
    y_pred2,
    average="weighted"
)

recall2 = recall_score(
    y_test,
    y_pred2,
    average="weighted"
)
# 13. DISPLAY RESULTS
print("\n" + "=" * 60)
print("MODEL 1 - LOGISTIC REGRESSION")
print("=" * 60)

print("Accuracy :", round(accuracy1 * 100, 2), "%")
print("Precision:", round(precision1 * 100, 2), "%")
print("Recall   :", round(recall1 * 100, 2), "%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred1,
        target_names=iris.target_names
    )
)


print("\n" + "=" * 60)
print("MODEL 2 - DECISION TREE")
print("=" * 60)

print("Accuracy :", round(accuracy2 * 100, 2), "%")
print("Precision:", round(precision2 * 100, 2), "%")
print("Recall   :", round(recall2 * 100, 2), "%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred2,
        target_names=iris.target_names
    )
)
# 14. CONFUSION MATRICES
cm1 = confusion_matrix(y_test, y_pred1)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm1,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=iris.target_names,
    yticklabels=iris.target_names
)

plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()


cm2 = confusion_matrix(y_test, y_pred2)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm2,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=iris.target_names,
    yticklabels=iris.target_names
)

plt.title("Confusion Matrix - Decision Tree")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()
# 15. MODEL COMPARISON TABLE

comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree"
    ],
    "Accuracy": [
        accuracy1,
        accuracy2
    ],
    "Precision": [
        precision1,
        precision2
    ],
    "Recall": [
        recall1,
        recall2
    ]
})

comparison["Accuracy"] = comparison["Accuracy"] * 100
comparison["Precision"] = comparison["Precision"] * 100
comparison["Recall"] = comparison["Recall"] * 100

comparison = comparison.round(2)

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(comparison)


# ============================================================
# 16. MODEL COMPARISON VISUALIZATION
# ============================================================

comparison_plot = comparison.set_index("Model")

comparison_plot.plot(
    kind="bar",
    figsize=(9, 6)
)

plt.title("Comparison of Machine Learning Models")
plt.xlabel("Models")
plt.ylabel("Score (%)")
plt.ylim(0, 110)
plt.xticks(rotation=0)
plt.legend(title="Metrics")
plt.tight_layout()
plt.show()
# 17. FINAL CONCLUSION
if accuracy1 > accuracy2:
    print("\nFinal Conclusion:")
    print("Logistic Regression performed better based on accuracy.")
elif accuracy2 > accuracy1:
    print("\nFinal Conclusion:")
    print("Decision Tree performed better based on accuracy.")
else:
    print("\nFinal Conclusion:")
    print("Both models achieved the same accuracy.")
print("\nProject completed successfully.")