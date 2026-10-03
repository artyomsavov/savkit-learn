from abc import ABC, abstractmethod
from typing import Any, Self

from jaxtyping import Num
from numpy.typing import NDArray

type Features = Num[NDArray[Any], "samples features"]
type Target = Num[NDArray[Any], "samples"]
type Prediction = Num[NDArray[Any], "samples"]


class BaseEstimator(ABC):
    @abstractmethod
    def fit(self, X: Features, y: Target) -> Self: ...

    @abstractmethod
    def predict(self, X: Features) -> Prediction: ...
