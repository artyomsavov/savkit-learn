from typing import Self
import numpy as np

from ..base import BaseEstimator, Features, Target, Prediction


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

class LogisticRegression(BaseEstimator):
    def __init__(self, lr: float = 0.001, n_iters: int = 1000) -> None:
        self.lr = lr
        self.n_iters = n_iters
        self.weights: np.ndarray | None = None 
        self.bias: float | None = None 

    def fit(self, X: Features, y: Target) -> Self:
        n_samples, n_features = X.shape
        
        self.weights = np.zeros(n_features, dtype=np.float64)
        self.bias = 0.0

        for _ in range(self.n_iters):
            y_pred = sigmoid(X @ self.weights + self.bias)

            dw = X.T @ (y_pred - y) / n_samples
            db = np.sum(y_pred - y) / n_samples

            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db
        
        return self

    def predict_proba(self, X: Features) -> Prediction:
        if self.weights is None or self.bias is None:
            raise RuntimeError('Before calling predict, you must fit the model.')
            
        linear_pred = X @ self.weights + self.bias
        return sigmoid(linear_pred)

    def predict(self, X: Features) -> Prediction:
        probabilities = self.predict_proba(X)
        
        return (probabilities > 0.5).astype(np.int64)

