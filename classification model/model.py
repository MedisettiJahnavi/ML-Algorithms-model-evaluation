import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix
df = pd.read_csv("Iris.csv")
print(df)
print(df.head())
print(df.tail())
print(df.info())
print(df.describe())
print("Dataset shape:", df.shape)
print("Columns:", df.columns)
print(df.isnull().sum())
print("Duplicate rows before cleaning:", df.duplicated().sum())
df = df.drop_duplicates()
print("Duplicate rows after cleaning:", df.duplicated().sum())
print(df["Species"].value_counts())
sns.pairplot(df, hue="Species")
plt.show()
X = df.drop(columns=["Id", "Species"])
y = df["Species"]
print("Features:")
print(X.head())
print("\nTarget:")
print(y.head())
le = LabelEncoder()
y = le.fit_transform(y)
print("Encoded target:")
print(y)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)
logistic_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
logistic_model.fit(X_train, y_train)
y_pred_lr = logistic_model.predict(X_test)
print("Logistic Regression Predictions:")
print(y_pred_lr)
lr_accuracy = accuracy_score(y_test, y_pred_lr)
print("Logistic Regression Accuracy:",lr_accuracy)
cm_lr = confusion_matrix(y_test, y_pred_lr)
print("Confusion Matrix:")
print(cm_lr)
sns.heatmap(
    cm_lr,
    annot=True,
    fmt="d",
    xticklabels=le.classes_,
    yticklabels=le.classes_
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Logistic Regression Confusion Matrix")
plt.show()
print("\n========== MODEL COMPARISON ==========")
print("Logistic Regression Accuracy:",lr_accuracy)
print("Decision Tree Accuracy:", dt_accuracy)
models = ["Logistic Regression", "Decision Tree"]
accuracies = [lr_accuracy, dt_accuracy]
plt.bar(models, accuracies)
plt.xlabel("Classification Models")
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.ylim(0, 1.1)
plt.show()