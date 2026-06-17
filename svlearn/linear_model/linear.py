from typing import Self
import numpy as np

from ..base import BaseEstimator, Features, Target, Prediction


class LinearRegression(BaseEstimator):
    def __init__(self, lr: float = 0.001, n_iters: int = 1000) -> None:
        self.lr = lr
        self.n_iters = n_iters
        self.weights: np.ndarray | None = None
        self.bias: float | None = None

    def fit(self, X: Features, y: Target) -> Self:
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features, dtype=np.float64)
        self.bias = 0 

        for _ in range(self.n_iters):
            y_pred = X @ self.weights + self.bias

            dw = (X.T @ (y_pred - y)) / n_samples 
            db = np.sum(y_pred - y) / n_samples 

            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db

        return self

    def predict(self, X: Features) -> Prediction:
        if self.weights is None or self.bias is None:
            raise RuntimeError('Before calling predict, you must fit the model.')

        return X @ self.weights + self.bias 

