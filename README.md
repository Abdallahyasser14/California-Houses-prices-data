Here’s a polished and professional **GitHub README.md** version of your project, formatted with Markdown and emojis for visual appeal and readability:

---

# 🧠 Machine Learning Algorithms from Scratch: Linear Regression

This project implements several **core machine learning algorithms from scratch** using only standard Python libraries like **NumPy**.
The main focus is on **Linear Regression** and its **regularized variants (Lasso & Ridge)**.

The goal is to demonstrate a **deep mathematical understanding** of how these algorithms work — and to **verify our implementations** by directly comparing them against **Scikit-learn’s** optimized versions.

---

## ✨ Key Features

* 🧩 **From-Scratch Implementation** — Custom-built algorithms, including:

  * Manual `train/test` split
  * Custom **Gradient Descent**
  * **Normal Equation** for closed-form solutions

* 📊 **In-Depth EDA** — Comprehensive data visualization and exploration.

* 🔗 **Multicollinearity Analysis** — Studied the effects of correlated features on model performance.

* ⚙️ **Manual Hyperparameter Tuning** — Experimented with different `λ` (lambda) values for **Lasso** and **Ridge** regression.

* ⚖️ **Sklearn Comparison** — Benchmarked every custom model against its `scikit-learn` counterpart.

* ✅ **Result Verification** — Confirmed our implementations produce **virtually identical metrics** (e.g., **R²**, **MSE**, **MAE**) to those from Scikit-learn.

---

## 🤖 Algorithms Implemented

| Algorithm                                | Regularization | Approach         | File                        |
| ---------------------------------------- | -------------- | ---------------- | --------------------------- |
| **Linear Regression (Closed Form)**      | None           | Normal Equation  | `closed_form.py`            |
| **Linear Regression (Gradient Descent)** | None           | Iterative        | `gradient_descent_base.py`  |
| **Lasso Regression (L1)**                | L1             | Gradient Descent | `lasso_gradient_descent.py` |
| **Ridge Regression (L2)**                | L2             | Gradient Descent | `ridge_gradient_descent.py` |

---

## 📁 Project Structure

```
.
├── data/raw/                      # Raw dataset (e.g., California Housing)
├── notebooks/                     # Jupyter notebooks for EDA & comparisons
├── pdfs/                          # Papers or related documentation
├── src/                           # Source code for algorithms
│   ├── __init__.py
│   ├── closed_form.py
│   ├── gradient_descent_base.py
│   ├── lasso_gradient_descent.py
│   └── ridge_gradient_descent.py
├── .gitignore
├── README.md                      # This file
└── requirements.txt               # Project dependencies
```

---

## 📊 Results & Verification

The **notebooks/** directory includes:

* **EDA** (Exploratory Data Analysis) with visualizations
* **Multicollinearity tests** — Removing correlated features
* **Manual λ tuning** for Ridge and Lasso
* **Performance comparison** between our models and Scikit-learn equivalents

---

### 🔍 Model Performance Comparison

| #  | Model Name                               | Test MAE     | Test MSE     |
| -- | ---------------------------------------- | ------------ | ------------ |
| 1  | My Closed Form                           | **0.462865** | **0.387338** |
| 2  | My Closed Form (Dropped Corr)            | 0.480839     | 0.415712     |
| 3  | Sklearn Closed Form                      | 0.462865     | 0.387338     |
| 4  | Sklearn Closed Form (Dropped Corr)       | 0.480839     | 0.415712     |
| 5  | My GD                                    | 0.470651     | 0.395920     |
| 6  | My GD (Dropped Corr)                     | 0.482726     | 0.419421     |
| 7  | Sklearn GD                               | 0.463217     | 0.389512     |
| 8  | Sklearn GD (Dropped Corr)                | 0.480315     | 0.416847     |
| 9  | My Lasso GD                              | 0.470652     | 0.395922     |
| 10 | My Lasso GD (Dropped Corr)               | 0.482727     | 0.419424     |
| 11 | Sklearn Lasso GD                         | 0.529689     | 0.493118     |
| 12 | Sklearn Lasso GD (Dropped Corr)          | 0.513742     | 0.517789     |
| 13 | My Ridge Closed Form                     | 0.462865     | 0.387335     |
| 14 | My Ridge Closed Form (Dropped Corr)      | 0.480838     | 0.415711     |
| 15 | Sklearn Ridge Closed Form                | 0.462865     | 0.387335     |
| 16 | Sklearn Ridge Closed Form (Dropped Corr) | 0.480838     | 0.415711     |
| 17 | My Ridge GD                              | 0.470651     | 0.395920     |
| 18 | My Ridge GD (Dropped Corr)               | 0.482726     | 0.419422     |
| 19 | Sklearn Ridge GD                         | 0.471611     | 0.405191     |
| 20 | Sklearn Ridge GD (Dropped Corr)          | 0.483501     | 0.427438     |

---

