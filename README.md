# Salary Prediction using Polynomial Regression

AI-ML Assignment 3

## 📌 Objective
Predict employee salaries from their position level. Since salary rises
non-linearly with level, a **Polynomial Regression** model (degree = 3)
is built to capture this curve, then evaluated against held-out data.

## 📊 Dataset
[Position Salaries Dataset – Kaggle](https://www.kaggle.com/datasets/akram24/position-salaries)

10 records: `Position` (job title), `Level` (1–10, numeric encoding of
position), and `Salary` (target).

> Not redistributed here per assignment rules — download from the Kaggle
> link above and place it as `Position_Salaries.csv` in this folder.

## 🛠️ Libraries Used
| Library | Purpose |
|---|---|
| `pandas` | loading & exploring data |
| `numpy` | numerical operations |
| `matplotlib` | scatter plot + regression curve |
| `scikit-learn` | train/test split, polynomial features, model, metrics |

## ⚙️ Methodology
1. **Data Understanding** — loaded with Pandas, inspected `.head()`,
   `.info()`, `.describe()`; identified `Level` as input, `Salary` as
   target.
2. **Preprocessing** — checked for missing values (none), split into
   80% train / 20% test.
3. **Model Development** — transformed `Level` into degree-3 polynomial
   features, trained `LinearRegression` on the transformed features,
   predicted test salaries.
4. **Evaluation** — computed MAE, MSE, R² on test predictions; plotted
   original data vs. the fitted curve.

## 📈 Results
| Metric | Value |
|---|---|
| MAE | ≈ 70,635 |
| MSE | ≈ 6.26 × 10⁹ |
| R² Score | ≈ 0.88 |

*(Varies slightly with the random train/test split seed — dataset is
only 10 rows.)*

**Observations**
- The degree-3 curve tracks the sharp jump in salary at senior levels
  (Partner, C-level, CEO) that a straight line would miss.
- With only 10 rows, the 80/20 split leaves very few test points, so
  metrics can shift noticeably between runs.
- MSE is much larger than MAE since squaring penalizes big errors (e.g.
  the CEO salary jump) more heavily.

## ✅ Conclusion
Polynomial Regression (degree 3) fits this dataset far better than
Linear Regression would, since salary doesn't grow linearly with level —
it stays flat early on, then jumps sharply at senior roles. Adding
higher-degree terms of `Level` (Level², Level³) lets the model curve to
match that shape while remaining a linear model in its coefficients, so
it's still simple to train and interpret. That ability to fit curvature
without added algorithmic complexity is Polynomial Regression's key
advantage here.

## 📁 Files
- `Assignment-3.py` — full solution (Tasks 1–5)
- `Position_Salaries.csv` — dataset (add locally, not committed)
- `polynomial_regression_plot.png` — generated plot
- `conclusion.txt` — plain-text conclusion output
