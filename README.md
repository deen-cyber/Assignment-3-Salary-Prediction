# Assignment 3 – Salary Prediction using Polynomial Regression

## Objective
Predict employee salaries based on their position level. Since the
relationship between position level and salary is non-linear, a
**Polynomial Regression** model (degree = 3) is built, trained, and
evaluated on the Position Salaries dataset.

## Dataset Link
[Position Salaries Dataset – Kaggle](https://www.kaggle.com/datasets/akram24/position-salaries)

> The dataset is not redistributed in this repository per the assignment
> instructions. Download it from the Kaggle link above and place it as
> `Position_Salaries.csv` in the project root before running the script.

## Libraries Used
- `pandas` – data loading and exploration
- `numpy` – numerical operations
- `matplotlib` – plotting the scatter plot and regression curve
- `scikit-learn` – `train_test_split`, `PolynomialFeatures`,
  `LinearRegression`, and evaluation metrics (`mean_absolute_error`,
  `mean_squared_error`, `r2_score`)

## Methodology
1. **Data Understanding** – Loaded the dataset with Pandas, inspected the
   first five rows, identified `Level` as the input feature and `Salary`
   as the target variable, and reviewed `.info()` / `.describe()` output.
2. **Data Preprocessing** – Checked for missing values (none found),
   selected `Level` as `X` and `Salary` as `y`, and split the data into
   80% training / 20% testing sets.
3. **Model Development** – Transformed `Level` into polynomial features
   of degree 3 using `PolynomialFeatures`, then trained a
   `LinearRegression` model on the transformed features. Used it to
   predict salaries for the test set.
4. **Model Evaluation** – Computed MAE, MSE, and R² Score on the test
   predictions, and plotted the original data against the fitted
   polynomial regression curve.

## Results
| Metric | Value |
|---|---|
| MAE | ≈ 70,635 |
| MSE | ≈ 6.26 × 10⁹ |
| R² Score | ≈ 0.88 |

*(Exact values may vary slightly depending on the random train/test split
seed.)*

**Observations:**
1. The degree-3 polynomial curve closely follows the sharp, non-linear
   rise in salary at higher position levels, which a straight-line model
   could not capture.
2. Because the dataset has only 10 records, the 80/20 split leaves very
   few test points, so evaluation metrics can vary noticeably between
   runs/seeds.
3. MSE is much larger than MAE because squaring penalizes large errors
   (e.g., at the CEO level, where salary jumps sharply) more heavily.

## Conclusion
Polynomial Regression (degree 3) captures the non-linear jump in salary
across position levels far better than Linear Regression, which would fit
only a straight line and systematically misestimate salaries at both the
lowest and highest levels. By adding higher-degree terms of the input
feature (Level², Level³), the model bends to match the true curvature of
the data while remaining linear in its coefficients — keeping it simple
to train and interpret. For this dataset, that curve-fitting ability is
Polynomial Regression's key advantage, making it a much better fit than
Linear Regression for salary prediction here.

## Files
- `Assignment-3.py` – full solution script (Tasks 1–5)
- `Position_Salaries.csv` – dataset (add locally; not committed per
  instructions unless license permits redistribution)
- `polynomial_regression_plot.png` – generated scatter plot + regression
  curve
# Assignment-3-Salary-Prediction
