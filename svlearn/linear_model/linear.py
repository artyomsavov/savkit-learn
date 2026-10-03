
import numpy as np
from beartype import beartype
from jaxtyping import jaxtyped  # pyright: ignore[reportUnknownVariableType]
from numpy.typing import NDArray

from ..base import BaseEstimator, Features, Prediction, Target


class LinearRegression(BaseEstimator):
    def __init__(self, lr: float = 0.001, n_iters: int = 1000) -> None:
        self.lr = lr
        self.n_iters = n_iters
        self.weights: NDArray[np.float64] | None = None
        self.bias: float | None = None

    @jaxtyped(typechecker=beartype)
    def fit(self, X: Features, y: Target) -> "LinearRegression":
        n_samples, n_features = X.shape
        w = np.zeros(n_features, dtype=np.float64)
        b = 0.0

        for _ in range(self.n_iters):
            err = X @ w + b - y
            dw = (X.T @ err) / n_samples
            db = float(np.sum(err)) / n_samples
            w = w - self.lr * dw
            b = b - self.lr * db

        self.weights, self.bias = w, b
        return self

    @jaxtyped(typechecker=beartype)
    def predict(self, X: Features) -> Prediction:
        if self.weights is None or self.bias is None:
            raise RuntimeError("Before calling predict, you must fit the model.")

        return X @ self.weights + self.bias
