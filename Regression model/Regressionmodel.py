import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
# STEP 1: Load Dataset
dataset_path = Path(__file__).resolve().with_name("Student_Marks.csv")
df = pd.read_csv(dataset_path)
print("DATASET ")
print(df.head())
# STEP 2: Explore Dataset
print("\nDATASET INFORMATION")
df.info()
print("\nSTATISTICAL SUMMARY ")
print(df.describe())
print("\n DATASET SHAPE" )
print(df.shape)
# STEP 3: Check Missing Values
print("\n MISSING VALUES ")
print(df.isnull().sum())
# STEP 4: Clean Dataset
df = df.dropna()
df = df.drop_duplicates()
print("\nDataset after cleaning:")
print(df)
# STEP 5: Select Features and Target
X = df[["time_study"]]
y = df["Marks"]
print("\nFEATURES")
print(X.head())
print("\nTARGET")
print(y.head())
# STEP 6: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))
# STEP 7: Create Linear Regression Model
model = LinearRegression()
# STEP 8: Train Model
model.fit(X_train, y_train)
print("\nMODEL TRAINING")
print("Model training completed.")
# STEP 9: Regression Equation
print("\nREGRESSION EQUATION")
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("Marks =", model.coef_[0], "* time_study +", model.intercept_)
# STEP 10: Make Predictions
y_pred = model.predict(X_test)
print("\nPREDICTIONS")
print(y_pred)
# STEP 11: Compare Actual and Predicted Values
comparison = pd.DataFrame({
    "Actual Marks": y_test.values,
    "Predicted Marks": y_pred
})
print("\nACTUAL VS PREDICTED")
print(comparison)
# STEP 12: Evaluate Model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("\nMODEL PERFORMANCE")
print("Mean Squared Error:", mse)
print("R² Score:", r2)
# STEP 13: Regression Visualization
plt.scatter(
    X_test["time_study"],
    y_test,
    label="Actual Values"
)
X_test_sorted = X_test.sort_values("time_study")
plt.plot(
    X_test_sorted["time_study"],
    model.predict(X_test_sorted),
    label="Regression Line"
)
plt.xlabel("Study Time")
plt.ylabel("Marks")
plt.title("Linear Regression: Study Time vs Marks")
plt.legend()
plt.show()
# STEP 14: Actual vs Predicted Visualization
plt.scatter(
    y_test,
    y_pred
)
plt.xlabel("Actual Marks")
plt.ylabel("Predicted Marks")
plt.title("Actual vs Predicted Marks")
plt.show()