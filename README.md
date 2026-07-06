A lightweight, clean, and modern machine learning library implemented from scratch using NumPy. This project serves as an educational ecosystem to understand the mathematical foundations and inner workings of popular ML algorithms, structured in a `scikit-learn`-like architecture.

## Project Structure

```text
savkit-learn/
├── notebooks/
│   ├── algorithms.ipynb    # From-scratch implementations of 10 core classic ML models using NumPy
│   └── metrics.ipynb       # Math & explanations for all regression, classification, and clustering metrics
├── svlearn/                # Core production-ready package
│   ├── __init__.py         # Package initialization
│   ├── base.py             # Abstract base classes and strict type aliases (Features, Target, Prediction)
│   ├── linear_model/       # Linear & Logistic Regressions 
│   │   ├── __init__.py
│   │   ├── linear.py
│   │   └── logistic.py
│   ├── neighbors/          # K-Nearest Neighbors (KNN) 
│   │   ├── __init__.py
│   │   └── knn.py
│   ├── tree/               # Decision Trees based on Entropy and Information Gain
│   │   ├── __init__.py
│   │   └── decision_tree.py
│   └── ensemble/           # Random Forest Classifier built on custom Decision Trees
│       ├── __init__.py
│       └── random_forest.py
├── pyproject.toml          # Project configuration (uv, ruff, pyright)
├── README.md               # Main project documentation and quick start guide
└── THEORY.md               # Comprehensive markdown guide with math behind each algorithm

```

## Key Features

* **Sufficient Theory, Practical Implementation: Mathematical background is covered adequately, and every model's code is explicitly structured according to its formulas — making the connection clear without over‑theorising.
* **Strict Type Checking & Annotations:** Leverages Python 3.12 syntax, `jaxtyping`, and `numpy.typing` for shape and type safety.
* **Clean Architecture:** Strict object-oriented design using unified `fit` and `predict` interfaces via standard inheritance.
* **Modern Tooling:** Managed with `uv`, formatted using `ruff`, and statically typed with `pyright`.

## Installation & Setup

Clone the repository and install the library using `uv`:

```bash
# Clone the repository
git clone [https://github.com/artyomsavov/savkit-learn.git](https://github.com/artyomsavov/savkit-learn.git)
cd savkit-learn

# Build and install the package
uv pip install .

```

> **Note for Development:** If you want to modify the source code and see changes instantly without re-installing, use editable mode instead: `uv pip install -e .`

## Quick Start

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

## Documentation & Notebooks Overview

1. **`THEORY.md`**: A comprehensive mathematical guide detailing the core concepts, loss functions, and optimization techniques behind all 10 implemented algorithms.
2. **`notebooks/algorithms.ipynb`**: Contains step-by-step mathematical translation into clean, from-scratch code for 10 foundational machine learning algorithms using NumPy.
3. **`notebooks/metrics.ipynb`**: A deep dive into the mathematical formulations of all major regression, classification, and clustering metrics found in `scikit-learn`.

## Code Quality & Linting

You can run static analysis and formatting checks using the tools configured in `pyproject.toml`:

```bash
# Check for linting errors and style violations
ruff check .

# Automatically format the code according to PEP 8
ruff format .

# Run static type checking across the package
pyright

```

