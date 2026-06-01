import numpy as np
from jaxtyping import Float

def mean_squared_error(
    y_true: Float[ndarray, 'size'], 
    y_pred: Float[ndarray, 'size'],
) -> Float[np.ndarray, '']:
    return np.mean((y_true - y_pred) ** 2)
