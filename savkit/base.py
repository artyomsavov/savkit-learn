from abc import ABC, abstractmethod
from typing import Self

from jaxtyping import Float
from numpy.typing import NDArray

type Features = Float[NDArray, 'samples features']
type Weights = Float[NDArray, 'features 1']
type Target = Float[NDArray, 'samples 1']
type Prediction = Float[NDArray, 'samples 1']

class BaseEstimator(ABC):
    @abstractmethod
    def fit(self, X: Features, y_true: Target | None = None) -> Self: 
        ...

    @abstractmethod
    def predict(self, X: Features) -> Prediction:
        ...
