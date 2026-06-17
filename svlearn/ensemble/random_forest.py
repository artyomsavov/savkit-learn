from collections import Counter
from typing import Self
import numpy as np

from svlearn import DecisionTree
from ..base import BaseEstimator, Features, Target, Prediction


class RandomForest(BaseEstimator):
    def __init__(
            self,
            n_trees: int = 30, 
            max_depth: int = 100, 
            min_samples_split: int = 2, 
            min_impurity_decrease: float = 0.0, 
            n_features: int | None = None
    ) -> None:
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.n_features = n_features
        self.trees: list[DecisionTree] | None = None 

    def fit(self, X: Features, y: Target) -> Self:
        self.trees = [] 

        for _ in range(self.n_trees): 
            tree = DecisionTree(
                max_depth = self.max_depth,
                min_samples_split = self.min_samples_split,
                min_impurity_decrease = self.min_impurity_decrease,
                n_features = self.n_features
            )
            X_sample, y_sample = self._bootstrap_samples(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

        return self
        
    def _bootstrap_samples(self, X: Features, y: Target) -> tuple[Features, Target]:
        n_samples = X.shape[0]
        sample_idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[sample_idxs], y[sample_idxs] 

    def predict(self, X: Features) -> Prediction:
        if self.trees is None:
            raise RuntimeError('Before calling predict, you must fit the model.')

        y_pred_unmcld = np.array([tree.predict(X) for tree in self.trees]).T

        y_pred = np.array(
                [self._most_common_label(predictions) 
                 for predictions in y_pred_unmcld]
        )
        return y_pred

    def _most_common_label(self, predictions: np.ndarray) -> int | float:
        counter = Counter(predictions)
        return counter.most_common(1)[0][0]

