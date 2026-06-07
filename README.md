# ScratchML: From-Scratch Optimization Engine

A purely foundational implementation of Linear Regression optimization algorithms built completely from scratch using only NumPy and Python. This project utilizes the classic Kaggle **House Prices: Advanced Regression Techniques** dataset to validate custom mathematical formulations against closed-form analytical solutions.

## 📌 Project Overview
Most machine learning pipelines rely blindly on high-level libraries like Scikit-Learn (`model.fit()`). This repository peels back the layers of abstraction to implement the core mathematical concepts found in optimization theory, including matrix calculus, vectorization, feature standardization, and early-stopping regularization loops.

### Key Features
* **Analytical Solver:** Implemented the exact closed-form **Normal Equation**.
* **Batch Gradient Descent:** Developed the complete full-batch vector optimization loop tracking Mean Squared Error (MSE) convergence.
* **Mini-Batch Gradient Descent:** Engineered an incremental, randomized chunk-based learning loop to mimic production scale constraints.
* **Early Stopping Regularization:** Implemented a non-parametric validation-tracking loop with automated parameter restoration to prevent overfitting.
* **Kaggle-Compliant Pipeline:** Built an end-to-end processing script that normalizes unseen test features to output valid competition submission sets.

---

## 📐 Mathematical Formulations Implemented

### 1. The Normal Equation
To find the absolute global minimum of the cost function analytically without hyperparameter tuning:
$$\theta = (X^T X)^{-1} X^T y$$

### 2. Batch Gradient Descent Weight Updates
To iteratively update parameters by calculating the partial derivatives of the cost function across all $m$ training rows simultaneously:
$$\theta^{(\text{next step})} = \theta - \alpha \frac{2}{m} X_b^T (X_b\theta - y)$$

### 3. Mean Squared Error (MSE) Cost Function
$$J(\theta) = \frac{1}{m} \sum_{i=1}^{m} \left( \hat{y}^{(i)} - y^{(i)} \right)^2$$

---

## 🛠️ Tech Stack & Constraints
* **Language:** Python 3
* **Libraries:** NumPy (Matrix algebra), Pandas (Data loading), Matplotlib (Convergence visualization)
* **Strict Constraints:** No `scikit-learn`, `statsmodels`, or other high-level ML framework dependencies were permitted.

---

## 📂 Repository Structure
```text
├── House Prices.ipynb     # Main Jupyter notebook containing implementation steps 1-7
├── train.csv              # Kaggle training dataset (1,460 rows)
├── test.csv               # Kaggle test set for final inference (1,459 rows)
├── data_description.txt   # Complete feature metadata documentation
└── submission.csv         # Exported custom model predictions ready for Kaggle grading
