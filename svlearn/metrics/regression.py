import numpy as np
from beartype import beartype
from jaxtyping import Float, jaxtyped

type Target = Float[np.ndarray, 'size 1']
type Prediction = Float[np.ndarray, 'size 1']

@jaxtyped(typechecker=beartype)
def mean_squared_error(
    y_true: Target,
    y_pred: Prediction,
) -> float:
    errors = (y_true - y_pred) ** 2
    return np.mean(errors).item()
