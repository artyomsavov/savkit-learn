# svlearn 🚀

A lightweight, clean, and modern machine learning library implemented from scratch using NumPy. This project serves as an educational ecosystem to understand the mathematical foundations and inner workings of popular ML algorithms, structured in a `scikit-learn`-like architecture.

## 🛠️ Project Structure

```text
savkit-learn/
├── notebooks/
│   ├── algorithms.ipynb    # From-scratch implementations of 10 core ML models using NumPy
│   └── metrics.ipynb       # Math & explanations for all regression, classification, and clustering metrics
├── svlearn/                # Core production-ready package
│   ├── __init__.py         # Package initialization
│   ├── base.py             # Abstract base classes and strict type aliases (Features, Target, Prediction)
│   ├── linear_model/       # Linear & Logistic Regressions with vectorized gradient descent
│   │   ├── __init__.py
│   │   ├── linear.py
│   │   └── logistic.py
│   ├── neighbors/          # K-Nearest Neighbors (KNN) with vectorized distance calculation
│   │   ├── __init__.py
│   │   └── knn.py
│   ├── tree/               # Decision Trees based on Entropy and Information Gain
│   │   ├── __init__.py
│   │   └── decision_tree.py
│   └── ensemble/           # Random Forest Classifier built on custom Decision Trees
│       ├── __init__.py
│       └── random_forest.py
├── pyproject.toml          # Modern project configuration (uv, ruff, pyright)
├── README.md               # Main project documentation and quick start guide
└── THEORY.md               # Comprehensive markdown guide with math behind each algorithm

```

## ✨ Key Features

* **Strict Type Checking & Annotations:** Leverages Python 3.12 syntax, `jaxtyping`, and `numpy.typing` for shape and type safety.
* **Optimized NumPy Vectorization:** Avoids slow Python loops wherever possible (e.g., fully vectorized matrix distance calculations in KNN).
* **Clean Architecture:** Strict object-oriented design using unified `fit` and `predict` interfaces via standard inheritance.
* **Modern Tooling:** Managed with `uv`, formatted using `ruff`, and statically typed with `pyright`.

## 📦 Installation

To install the package locally in editable mode for development, clone the repository and use `uv` or `pip`:

```bash
# Using uv (recommended)
uv pip install -e .

# Or using standard pip
pip install -e .

```

## 🚀 Quick Start

Here is a quick example of training the custom **Logistic Regression** and **Random Forest** models:

```python
import numpy as np
from svlearn.linear_model import LogisticRegression
from svlearn.ensemble import RandomForest

# Generate synthetic dataset
X = np.random.randn(100, 5)
y = np.random.randint(0, 2, size=100)

# 1. Logistic Regression
log_reg = LogisticRegression(lr=0.01, n_iters=500)
log_reg.fit(X, y)
predictions = log_reg.predict(X)

# 2. Random Forest
rf = RandomForest(n_trees=10, max_depth=10)
rf.fit(X, y)
rf_predictions = rf.predict(X)

```

## 📝 Notebooks Overview

1. **`algorithms.ipynb`**: Contains step-by-step mathematical translation into code for 10 foundational ML algorithms using pure NumPy.
2. **`metrics.ipynb`**: A comprehensive mathematical guide detailing all key classification (Accuracy, Precision, Recall, F1, ROC-AUC), regression (MSE, MAE, R²), and clustering (Silhouette, Adjusted Rand Index) metrics found in `scikit-learn`.

## 🔧 Code Quality & Linting

You can run static analysis and formatting checks using the tools configured in `pyproject.toml`:

```bash
# Check for linting errors and style violations
ruff check .

# Automatically format the code according to PEP 8
ruff format .

# Run static type checking across the package
pyright

```

