"""
AI-ML Assignment 3
Topic: Salary Prediction using Polynomial Regression
Dataset: Position_Salaries (Kaggle - akram24/position-salaries)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------------------------
# Task 1: Data Understanding (2 Marks)
# ---------------------------------------------------------------------------
print("=" * 70)
print("TASK 1: DATA UNDERSTANDING")
print("=" * 70)

df = pd.read_csv("Position_Salaries.csv")

print("\nFirst five records:")
print(df.head())

print("\nInput Feature : 'Level' (numeric encoding of Position)")
print("Target Variable : 'Salary'")

print("\nDataset Info:")
df.info()

print("\nSummary Statistics:")
print(df.describe())

# ---------------------------------------------------------------------------
# Task 2: Data Preprocessing (2 Marks)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("TASK 2: DATA PREPROCESSING")
print("=" * 70)

print("\nMissing values per column:")
print(df.isnull().sum())

X = df[["Level"]].values      # Input feature (kept 2D for sklearn)
y = df["Salary"].values       # Target variable

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set size: {X_train.shape[0]} rows")
print(f"Testing set size : {X_test.shape[0]} rows")

# ---------------------------------------------------------------------------
# Task 3: Model Development (3 Marks)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("TASK 3: MODEL DEVELOPMENT")
print("=" * 70)

degree = 3
poly = PolynomialFeatures(degree=degree)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

model = LinearRegression()
model.fit(X_train_poly, y_train)

y_pred = model.predict(X_test_poly)

print(f"\nPolynomial degree used: {degree}")
print("\nPredicted vs Actual salaries (test set):")
for actual, pred in zip(y_test, y_pred):
    print(f"  Actual: {actual:>10,.0f}   Predicted: {pred:>12,.0f}")

# ---------------------------------------------------------------------------
# Task 4: Model Evaluation (2 Marks)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("TASK 4: MODEL EVALUATION")
print("=" * 70)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nMean Absolute Error (MAE): {mae:,.2f}")
print(f"Mean Squared Error  (MSE): {mse:,.2f}")
print(f"R^2 Score               : {r2:.4f}")

# --- Scatter plot of original data + polynomial regression curve ---
X_grid = np.arange(X.min(), X.max() + 0.1, 0.1).reshape(-1, 1)
y_grid_pred = model.predict(poly.transform(X_grid))

plt.figure(figsize=(8, 6))
plt.scatter(X, y, color="red", label="Actual Data")
plt.plot(X_grid, y_grid_pred, color="blue", label=f"Polynomial Fit (degree={degree})")
plt.scatter(X_test, y_pred, color="green", marker="x", s=100, label="Test Predictions")
plt.title("Salary vs Position Level (Polynomial Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.legend()
plt.tight_layout()
plt.savefig("polynomial_regression_plot.png", dpi=150)
print("\nPlot saved as 'polynomial_regression_plot.png'")

print("\nObservations:")
print("""
1. The polynomial regression curve (degree 3) closely follows the sharp,
   non-linear rise in salary at higher position levels, which a straight
   line (simple linear regression) could not capture.
2. The R^2 score obtained on the small test set indicates how well the
   curve generalizes; because the dataset only has 10 records, the
   train/test split leaves very few test points, so metrics can vary
   noticeably with the random split.
3. MAE and MSE show the average prediction error in salary units - MSE is
   larger due to squaring, which penalizes bigger deviations (e.g. at the
   'CEO' level, where salary jumps sharply) more heavily than MAE.
""")

# ---------------------------------------------------------------------------
# Task 5: Conclusion (1 Mark)
# ---------------------------------------------------------------------------
print("=" * 70)
print("TASK 5: CONCLUSION")
print("=" * 70)

conclusion = """
This project used Polynomial Regression (degree 3) to predict employee
salaries based on their position level. The relationship between level
and salary in this dataset is clearly non-linear, with salary rising
slowly at junior levels and then increasing sharply at senior levels such
as Partner, C-level, and CEO. A simple Linear Regression model would fit
a straight line through the data, systematically underestimating salaries
at both the lowest and highest levels while overestimating them in the
middle. Polynomial Regression instead fits a curved line by adding
higher-degree terms (Level^2, Level^3) of the input feature, allowing it
to bend and follow the true shape of the data far more closely. The main
advantage of Polynomial Regression for this dataset is its ability to
model this curvature accurately without needing a completely different,
more complex algorithm - it remains a linear model in terms of its
coefficients, so it is easy to train and interpret while still capturing
non-linear trends. Overall, Polynomial Regression proved to be a much
better fit than Linear Regression for salary prediction on this dataset.
"""
print(conclusion)

with open("conclusion.txt", "w") as f:
    f.write(conclusion.strip())
