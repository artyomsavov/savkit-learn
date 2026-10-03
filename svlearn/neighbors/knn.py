from collections import Counter

import numpy as np
from beartype import beartype
from jaxtyping import jaxtyped  # pyright: ignore[reportUnknownVariableType]

from ..base import BaseEstimator, Features, Prediction, Target


class KNN(BaseEstimator):
    def __init__(self, k: int = 3) -> None:
        self.k = k
        self.X_train: Features | None = None
        self.y_train: Target | None = None

    @jaxtyped(typechecker=beartype)
    def fit(self, X: Features, y: Target) -> "KNN":
        self.X_train = X
        self.y_train = y
        return self

    @jaxtyped(typechecker=beartype)
    def predict(self, X: Features) -> Prediction:
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("Before calling predict, you must fit the model.")
        X_train, y_train = self.X_train, self.y_train
        return np.array([self._predict(x, X_train, y_train) for x in X])

    def _predict(
        self, x: np.ndarray, X_train: Features, y_train: Target
    ) -> int | float:
        distances = np.sqrt(np.sum((X_train - x) ** 2, axis=1))
        k_nearest_indeces = np.argsort(distances)[: self.k]
        k_nearest_labels = y_train[k_nearest_indeces]
        return Counter(k_nearest_labels).most_common(1)[0][0]
