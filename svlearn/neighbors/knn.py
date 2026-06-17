from collections import Counter
from typing import Self
import numpy as np

from ..base import BaseEstimator, Features, Target, Prediction


class KNN(BaseEstimator):
    def __init__(self, k: int = 3) -> None:
        self.k = k
        self.X_train: Features | None = None
        self.y_train: Target | None = None

    def fit(self, X: Features, y: Target) -> Self:
        self.X_train = X 
        self.y_train = y
        return self
    
    def predict(self, X: Features) -> Prediction:
        if self.X_train is None or self.y_train is None:
            raise RuntimeError('Before calling predict, you must fit the model.')

        return np.array([self._predict(x) for x in X])

    def _predict(self, x: np.ndarray) -> int | float:
        distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1)) 

        k_nearest_indeces = np.argsort(distances)[:self.k]

        k_nearest_labels = self.y_train[k_nearest_indeces] 

        counter = Counter(k_nearest_labels)
        return counter.most_common(1)[0][0]

