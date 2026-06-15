from abc import ABC, abstractmethod
from typing import Self

from jaxtyping import Num 
from numpy.typing import NDArray

type Features = Num[NDArray, 'samples features']
type Target = Num[NDArray, 'samples']
type Prediction = Num[NDArray, 'samples']

class BaseEstimator(ABC):
    @abstractmethod
    def fit(self, X: Features, y_true: Target | None = None) -> Self: 
        ...

    @abstractmethod
    def predict(self, X: Features) -> Prediction:
        ...
